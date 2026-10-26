"""Portable focused checks for the matched EEGNet architecture and fusion.

Run with ``python -m unittest discover -s matched_eegnet_2026-10-24
-p test_models.py -v``. These tests do not import data, other pipelines, or
the training runner and require only PyTorch.
"""

from __future__ import annotations

import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from models import EEGNet, SIDE_DIMENSIONS, SamePadTemporalConv2d, build_model


class EEGNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def setUp(self) -> None:
        torch.manual_seed(20261024)

    def test_requested_window_class_and_variant_shapes(self) -> None:
        for samples in (2000, 6000):
            raw = torch.randn(1, 30, samples)
            for n_classes in (7, 9):
                for variant, side_dim in SIDE_DIMENSIONS.items():
                    with self.subTest(samples=samples, classes=n_classes, variant=variant):
                        model = build_model(variant, n_classes, samples).eval()
                        side = torch.randn(1, side_dim) if side_dim else None
                        with torch.no_grad():
                            logits = model(raw, side)
                        self.assertEqual(tuple(logits.shape), (1, n_classes))
                        self.assertTrue(torch.isfinite(logits).all())
                        self.assertEqual(model.flattened_features,
                                         992 if samples == 2000 else 2992)

    def test_canonical_architecture_and_parameter_count(self) -> None:
        model = build_model("eegnet", 9, 2000)
        self.assertEqual(model.temporal_conv.kernel_size, (1, 100))
        self.assertEqual(model.spatial_depthwise.kernel_size, (30, 1))
        self.assertEqual(model.spatial_depthwise.groups, 8)
        self.assertEqual(model.spatial_depthwise.out_channels, 16)
        self.assertEqual(model.separable_depthwise.groups, 16)
        self.assertEqual(model.separable_depthwise.kernel_size, (1, 25))
        self.assertEqual(model.separable_pointwise.kernel_size, (1, 1))
        self.assertEqual(model.pool1.kernel_size, (1, 4))
        self.assertEqual(model.pool2.kernel_size, (1, 8))
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                self.assertIsNone(module.bias)
            if isinstance(module, nn.BatchNorm2d):
                self.assertEqual(module.eps, 0.001)
                self.assertEqual(module.momentum, 0.01)
            if isinstance(module, nn.Dropout):
                self.assertEqual(module.p, 0.5)
        # 800 temporal + 480 spatial + 400 depthwise + 256 pointwise;
        # BN affine 2*(8+16+16); dense 992*9+9.
        expected = 800 + 480 + 400 + 256 + 80 + 992 * 9 + 9
        self.assertEqual(sum(p.numel() for p in model.parameters()), expected)
        for variant, side_dim in SIDE_DIMENSIONS.items():
            fused = build_model(variant, 9, 2000)
            added = (side_dim * 32 + 32 * 9) if side_dim else 0
            self.assertEqual(sum(p.numel() for p in fused.parameters()), expected + added)
            if side_dim:
                self.assertIsNone(fused.side_hidden.bias)
                self.assertIsNone(fused.side_classifier.bias)

    def test_keras_same_even_and_odd_padding_alignment(self) -> None:
        # An ordered asymmetric kernel detects an off-by-one padding shift.
        signal = torch.arange(137, dtype=torch.float32).reshape(1, 1, 1, -1)
        for length, left, right in ((100, 49, 50), (25, 12, 12)):
            with self.subTest(length=length):
                conv = SamePadTemporalConv2d(1, 1, length)
                with torch.no_grad():
                    conv.weight.copy_(torch.arange(length, dtype=torch.float32).reshape(1, 1, 1, -1))
                expected = F.conv2d(F.pad(signal, (left, right, 0, 0)), conv.weight)
                self.assertEqual(conv.left_padding, left)
                self.assertEqual(conv.right_padding, right)
                self.assertEqual(tuple(conv(signal).shape), tuple(signal.shape))
                torch.testing.assert_close(conv(signal), expected, rtol=0, atol=0)
        # The final EEGNet flatten must preserve time-major Keras ordering.
        model = EEGNet(7, 2000).eval()
        captured = []
        handle = model.dropout2.register_forward_hook(
            lambda _module, _inputs, output: captured.append(output.detach().clone()))
        features = model.forward_features(torch.randn(1, 30, 2000))
        handle.remove()
        expected = captured[0].permute(0, 2, 3, 1).reshape(1, -1)
        torch.testing.assert_close(features, expected, rtol=0, atol=0)

    def test_identical_seed_backbone_and_initial_fused_predictions(self) -> None:
        raw = torch.randn(2, 30, 2000)
        torch.manual_seed(11)
        baseline = build_model("eegnet", 7, 2000)
        base_state = baseline.state_dict()
        for variant, side_dim in SIDE_DIMENSIONS.items():
            with self.subTest(variant=variant):
                torch.manual_seed(11)
                model = build_model(variant, 7, 2000)
                for name, value in base_state.items():
                    torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
                side = torch.randn(2, side_dim) if side_dim else None
                baseline.eval()
                model.eval()
                with torch.no_grad():
                    torch.testing.assert_close(model(raw, side), baseline(raw), rtol=0, atol=0)
                # Raw dropout consumes the same stream before the side branch.
                baseline.train()
                model.train()
                torch.manual_seed(29)
                expected = baseline(raw)
                torch.manual_seed(29)
                actual = model(raw, side)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                # Avoid carrying baseline BN updates into the next subtest.
                torch.manual_seed(11)
                baseline = build_model("eegnet", 7, 2000)
                base_state = baseline.state_dict()

    def test_shuffled_control_has_exact_same_architecture(self) -> None:
        torch.manual_seed(91)
        combined = build_model("eegnet_combined", 9, 2000)
        torch.manual_seed(91)
        shuffled = build_model("eegnet_shuffled", 9, 2000)
        self.assertEqual(tuple(combined.state_dict()), tuple(shuffled.state_dict()))
        for name, value in combined.state_dict().items():
            torch.testing.assert_close(shuffled.state_dict()[name], value, rtol=0, atol=0)

    def test_max_norm_is_explicit_per_filter_and_per_class(self) -> None:
        model = build_model("eegnet_combined", 7, 2000)
        with torch.no_grad():
            model.spatial_depthwise.weight.fill_(2)
            model.raw_classifier.weight.fill_(2)
            model.spatial_depthwise.weight[0].zero_()
            model.raw_classifier.weight[0].fill_(0.0001)
            model.side_classifier.weight.fill_(3)
        untouched = {name: value.detach().clone() for name, value in model.named_parameters()
                     if name not in ("spatial_depthwise.weight", "raw_classifier.weight")}
        small = model.raw_classifier.weight[0].detach().clone()
        spatial_before = model.spatial_depthwise.weight.detach().clone()
        raw = torch.randn(1, 30, 2000)
        model.eval()(raw, torch.ones(1, 5370))
        # A forward pass must not silently mutate parameters.
        torch.testing.assert_close(model.spatial_depthwise.weight, spatial_before, rtol=0, atol=0)
        model.apply_constraints()
        norms = model.spatial_depthwise.weight.detach().flatten(1).norm(dim=1)
        dense_norms = model.raw_classifier.weight.detach().norm(dim=1)
        self.assertTrue(torch.all(norms <= 1 + 1e-6))
        self.assertTrue(torch.all(dense_norms <= 0.25 + 1e-6))
        torch.testing.assert_close(norms[1:], torch.ones(15))
        torch.testing.assert_close(dense_norms[1:], torch.full((6,), 0.25))
        self.assertEqual(float(norms[0]), 0)
        torch.testing.assert_close(model.raw_classifier.weight[0], small, rtol=0, atol=0)
        for name, parameter in model.named_parameters():
            if name in untouched:
                torch.testing.assert_close(parameter, untouched[name], rtol=0, atol=0)
        # The API also enforces the constraints following a genuine optimizer update.
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(raw, torch.randn(1, 5370)), torch.tensor([3]))
        loss.backward()
        optimizer.step()
        model.apply_constraints()
        self.assertTrue(torch.all(model.spatial_depthwise.weight.flatten(1).norm(dim=1) <= 1 + 1e-6))
        self.assertTrue(torch.all(model.raw_classifier.weight.norm(dim=1) <= 0.25 + 1e-6))

    def test_raw_and_feature_gradient_paths_with_zero_initial_head(self) -> None:
        model = build_model("eegnet_recurrence", 7, 2000).eval()
        raw = torch.randn(2, 30, 2000, requires_grad=True)
        side = torch.randn(2, 585, requires_grad=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        target = torch.tensor([1, 4])
        F.cross_entropy(model(raw, side), target).backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertGreater(float(raw.grad.abs().sum()), 0)
        self.assertGreater(float(model.temporal_conv.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.side_classifier.weight.grad.abs().sum()), 0)
        # The zero head deliberately prevents hidden/input side gradients at step 0.
        self.assertEqual(float(side.grad.abs().sum()), 0)
        self.assertEqual(float(model.side_hidden.weight.grad.abs().sum()), 0)
        optimizer.step()
        model.apply_constraints()
        optimizer.zero_grad(set_to_none=True)
        raw.grad = None
        side.grad = None
        F.cross_entropy(model(raw, side), target).backward()
        self.assertGreater(float(side.grad.abs().sum()), 0)
        self.assertGreater(float(model.side_hidden.weight.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(side.grad).all())
        # With no branch biases, even a trained branch cannot change zero-side logits.
        with torch.no_grad():
            torch.testing.assert_close(model(raw, torch.zeros_like(side)),
                                       model.forward_raw(raw), rtol=0, atol=0)

    def test_initialization_fans_and_input_validation(self) -> None:
        model = build_model("eegnet_connectivity", 7, 2000)
        self.assertLessEqual(float(model.spatial_depthwise.weight.detach().abs().max()),
                             math.sqrt(6 / (30 * (8 + 2))))
        self.assertEqual(int(torch.count_nonzero(model.side_classifier.weight)), 0)
        raw = torch.randn(1, 30, 2000)
        with self.assertRaisesRegex(ValueError, "requires side"):
            model(raw)
        with self.assertRaisesRegex(ValueError, "side must have shape"):
            model(raw, torch.randn(1, 585))
        with self.assertRaisesRegex(ValueError, "same dtype"):
            model(raw, torch.randn(1, 4785, dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "raw must have shape"):
            model(torch.randn(1, 29, 2000), torch.randn(1, 4785))
        with self.assertRaisesRegex(TypeError, "floating-point"):
            model(raw.long(), torch.randn(1, 4785))
        with self.assertRaisesRegex(ValueError, "does not accept"):
            build_model("eegnet", 7, 2000)(raw, torch.randn(1, 4785))
        with self.assertRaisesRegex(ValueError, "unknown variant"):
            build_model("unknown", 7, 2000)


if __name__ == "__main__":
    unittest.main()
