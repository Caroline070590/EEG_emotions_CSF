"""EEGNet-8,2 and matched additive feature branches for 200 Hz EEG.

Architecture provenance
-----------------------
The backbone follows the EEGNet function in the authors' implementation:
https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py
(Lawhern et al., EEGNet, https://arxiv.org/abs/1611.08024).

The official defaults assume 128 Hz: F1=8, D=2, F2=16, temporal kernels
64 and 16, average pools 4 and 8, ELU, ordinary dropout 0.5, spatial
depthwise max-norm 1, and dense-classifier max-norm 0.25. Here the temporal
kernels are explicitly adapted to 100 and 25 samples at 200 Hz (0.5 s and
0.125 s). Pool sizes remain 4 and 8, as specified for this comparison.
Thus this is a documented sampling-rate adaptation, not a claim to reproduce
the official 128 Hz experiment. Intended inputs are 30 channels and either
2,000 samples (10 s) or 6,000 samples (30 s).

Batch normalization uses epsilon=0.001 and PyTorch momentum=0.01, the update
coefficient corresponding to Keras' retained-statistic momentum=0.99.
Convolution/dense initialization is Glorot uniform (including the Keras
depthwise-kernel fan convention); biases start at zero. Flattening is in
time-then-feature order, matching the channels-last Keras backbone. These
choices match the architecture; cross-framework numerical identity is not
assumed. Outputs are logits, without the official final softmax, for use
with PyTorch cross-entropy.

The feature branch is a new comparison component, not part of EEGNet:
Linear(side_dim, 32, bias=False), ELU, dropout 0.5, and
Linear(32, n_classes, bias=False), added to the EEGNet logits. Its last
matrix starts at zero, so every fused model initially has exactly the raw
EEG model's predictions. Its hidden layer receives gradients after the
last matrix first updates. A zero side vector always contributes zero.
The shuffled variant uses the same combined branch; the runner, not this
module, supplies the shuffled features.

The runner must normalize raw EEG and side features using training data,
choose participant-held-out validation/checkpoints, and call
``model.apply_constraints()`` after every optimizer step. Only the spatial
depthwise weights and the original raw-EEG classifier are constrained;
the new additive branch is unconstrained. No training or split logic is
implemented here.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


SAMPLING_RATE_HZ = 200
SIDE_DIMENSIONS = {
    "eegnet": 0,
    "eegnet_connectivity": 4785,
    "eegnet_recurrence": 585,
    "eegnet_combined": 5370,
    "eegnet_shuffled": 5370,
}


class SamePadTemporalConv2d(nn.Conv2d):
    """Stride-one temporal convolution with TensorFlow/Keras SAME padding.

    For an even kernel the extra pad is on the right: a length-100 kernel
    receives 49 samples on the left and 50 on the right.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_length: int, *, groups: int = 1) -> None:
        super().__init__(in_channels, out_channels, (1, kernel_length),
                         stride=1, padding=0, groups=groups, bias=False)
        self.left_padding = (kernel_length - 1) // 2
        self.right_padding = kernel_length - 1 - self.left_padding

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(F.pad(x, (self.left_padding,
                                        self.right_padding, 0, 0)))


def _glorot_depthwise_(weight: Tensor, *, channels: int,
                      multiplier: int, kernel_area: int) -> None:
    """Use fans of Keras' (height, width, input_channels, multiplier) tensor."""
    bound = math.sqrt(6.0 / (kernel_area * (channels + multiplier)))
    nn.init.uniform_(weight, -bound, bound)


class EEGNet(nn.Module):
    """EEGNet-8,2 with an optional zero-initialized additive feature branch.

    Args:
        n_classes: Number of output classes (7 or 9 in this analysis).
        n_samples: Exact temporal input length (2,000 or 6,000 here).
        n_channels: Number of EEG channels, default 30.
        side_dim: Optional feature-vector length; zero means raw EEG only.

    Reset ``torch.manual_seed`` before constructing each variant to obtain
    identical backbone and raw-classifier initialization. Feature modules
    are constructed only after all raw-EEG modules have been initialized.
    Fused models require their side input; omission raises an error instead
    of silently disabling the branch. ``forward_raw`` explicitly evaluates
    the raw path when an ablation is intended.
    """

    def __init__(self, n_classes: int, n_samples: int,
                 n_channels: int = 30, side_dim: int = 0) -> None:
        super().__init__()
        if n_classes < 2 or int(n_classes) != n_classes:
            raise ValueError("n_classes must be an integer of at least 2")
        if n_channels < 1 or int(n_channels) != n_channels:
            raise ValueError("n_channels must be a positive integer")
        if n_samples < 32 or int(n_samples) != n_samples:
            raise ValueError("n_samples must be an integer of at least 32")
        if side_dim < 0 or int(side_dim) != side_dim:
            raise ValueError("side_dim must be a non-negative integer")
        self.n_classes = int(n_classes)
        self.n_samples = int(n_samples)
        self.n_channels = int(n_channels)
        self.side_dim = int(side_dim)
        self.feature_bins = (self.n_samples // 4) // 8
        self.flattened_features = 16 * self.feature_bins

        self.temporal_conv = SamePadTemporalConv2d(1, 8, 100)
        self.temporal_bn = nn.BatchNorm2d(8, eps=0.001, momentum=0.01)
        self.spatial_depthwise = nn.Conv2d(
            8, 16, (self.n_channels, 1), groups=8, bias=False)
        self.spatial_bn = nn.BatchNorm2d(16, eps=0.001, momentum=0.01)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(0.5)

        self.separable_depthwise = SamePadTemporalConv2d(16, 16, 25, groups=16)
        self.separable_pointwise = nn.Conv2d(16, 16, 1, bias=False)
        self.separable_bn = nn.BatchNorm2d(16, eps=0.001, momentum=0.01)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(0.5)
        self.raw_classifier = nn.Linear(self.flattened_features, self.n_classes)
        self._initialize_raw_parameters()

        self.side_hidden: nn.Linear | None = None
        self.side_classifier: nn.Linear | None = None
        self.side_dropout: nn.Dropout | None = None
        if self.side_dim:
            self.side_hidden = nn.Linear(self.side_dim, 32, bias=False)
            self.side_dropout = nn.Dropout(0.5)
            self.side_classifier = nn.Linear(32, self.n_classes, bias=False)
            nn.init.xavier_uniform_(self.side_hidden.weight)
            nn.init.zeros_(self.side_classifier.weight)

    def _initialize_raw_parameters(self) -> None:
        nn.init.xavier_uniform_(self.temporal_conv.weight)
        _glorot_depthwise_(self.spatial_depthwise.weight, channels=8,
                          multiplier=2, kernel_area=self.n_channels)
        _glorot_depthwise_(self.separable_depthwise.weight, channels=16,
                          multiplier=1, kernel_area=25)
        nn.init.xavier_uniform_(self.separable_pointwise.weight)
        nn.init.xavier_uniform_(self.raw_classifier.weight)
        nn.init.zeros_(self.raw_classifier.bias)
        for module in (self.temporal_bn, self.spatial_bn, self.separable_bn):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _validate_raw(self, raw: Tensor) -> None:
        if raw.ndim != 3 or tuple(raw.shape[1:]) != (self.n_channels, self.n_samples):
            raise ValueError(
                f"raw must have shape (N, {self.n_channels}, {self.n_samples}); "
                f"got {tuple(raw.shape)}")
        if not raw.is_floating_point():
            raise TypeError("raw must be a floating-point tensor")

    def forward_features(self, raw: Tensor) -> Tensor:
        """Return the canonical flattened EEGNet representation."""
        self._validate_raw(raw)
        x = self.temporal_bn(self.temporal_conv(raw.unsqueeze(1)))
        x = F.elu(self.spatial_bn(self.spatial_depthwise(x)))
        x = self.dropout1(self.pool1(x))
        x = self.separable_pointwise(self.separable_depthwise(x))
        x = self.dropout2(self.pool2(F.elu(self.separable_bn(x))))
        # N, features, 1, time -> N, 1, time, features, as in Keras Flatten.
        return x.permute(0, 2, 3, 1).reshape(raw.shape[0], self.flattened_features)

    def forward_raw(self, raw: Tensor) -> Tensor:
        """Evaluate the raw path explicitly, including its normal dropout mode."""
        return self.raw_classifier(self.forward_features(raw))

    def forward(self, raw: Tensor, side: Tensor | None = None) -> Tensor:
        self._validate_raw(raw)
        if self.side_dim:
            if side is None:
                raise ValueError(f"this model requires side features of shape (N, {self.side_dim})")
            if side.ndim != 2 or tuple(side.shape) != (raw.shape[0], self.side_dim):
                raise ValueError(
                    f"side must have shape ({raw.shape[0]}, {self.side_dim}); "
                    f"got {tuple(side.shape)}")
            if side.dtype != raw.dtype or side.device != raw.device:
                raise ValueError("side must have the same dtype and device as raw")
        elif side is not None:
            raise ValueError("the raw-only EEGNet model does not accept side features")
        logits = self.forward_raw(raw)
        if self.side_dim:
            assert self.side_hidden is not None
            assert self.side_dropout is not None
            assert self.side_classifier is not None
            logits = logits + self.side_classifier(
                self.side_dropout(F.elu(self.side_hidden(side))))
        return logits

    @torch.no_grad()
    def apply_constraints(self) -> None:
        """Apply canonical max norms after each optimizer step, in place.

        Norms are per spatial-depthwise output filter and per class's raw
        classifier input-weight vector. Biases and the side branch are not
        constrained. Parameters already inside each ball are unchanged.
        """
        for weight, max_norm in ((self.spatial_depthwise.weight, 1.0),
                                 (self.raw_classifier.weight, 0.25)):
            dimensions = tuple(range(1, weight.ndim))
            norms = torch.linalg.vector_norm(weight, dim=dimensions, keepdim=True)
            scale = max_norm / norms.clamp_min(torch.finfo(weight.dtype).tiny)
            weight.mul_(scale.clamp_max(1.0))


def build_model(variant: str, n_classes: int, n_samples: int,
                n_channels: int = 30) -> EEGNet:
    """Build a named matched model; shuffling is performed by the runner."""
    if variant not in SIDE_DIMENSIONS:
        raise ValueError(f"unknown variant {variant!r}; choose from {tuple(SIDE_DIMENSIONS)}")
    model = EEGNet(n_classes, n_samples, n_channels, SIDE_DIMENSIONS[variant])
    model.variant = variant
    return model
