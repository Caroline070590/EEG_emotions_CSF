"""Focused dimensional and FACED-equivalence tests for the SEED-VII adaptation.

A byte-identical FACED reference is bundled in vendor/ for portable equivalence
checks without the sibling project. The synthetic arrays exercise the actual 30-channel, ten-node-feature, twelve-edge
input shape, without loading participant data or changing either analysis pipeline.
"""
import hashlib
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "vendor/sdgnn_faced_reference.py"
ORIGINAL_SHA256 = "056cad7777309868000da8ec00da6288e196b1164deac06e5ff317f4d7ce0bde"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapted = load_module(ROOT / "vendor/sdgnn.py", "seed7_sdgnn_test")


def synthetic_data(mapping, seed=19):
    """Two participants, each viewing every mapped video exactly once."""
    rng = np.random.default_rng(seed)
    mapping = np.asarray(mapping, dtype=np.int64)
    videos = np.tile(np.arange(len(mapping)), 2)
    subjects = np.repeat(np.arange(2), len(mapping))
    X = rng.normal(size=(len(videos), 30, 10)).astype(np.float32)
    A = rng.normal(size=(len(videos), 12, 30, 30)).astype(np.float32)
    A = (A + A.swapaxes(-1, -2)) * np.float32(0.5)
    A[:, :, np.arange(30), np.arange(30)] = 0
    data = dict(X=X, A=A, video=videos, subject=subjects, y=mapping[videos])
    return data, np.flatnonzero(subjects == 0), np.flatnonzero(subjects == 1)


class Seed7ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        if hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() != ORIGINAL_SHA256:
            raise AssertionError("The frozen FACED source changed; recheck the adaptation baseline")
        cls.original = load_module(ORIGINAL, "frozen_faced_sdgnn_test")

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_seven_classes_eighty_videos_with_and_without_adversary(self):
        mapping = np.repeat(np.arange(7), [12, 12, 12, 12, 12, 12, 8])
        data, train, test = synthetic_data(mapping)
        for model in ("hybrid", "sfgnn"):
            for lam in (0.0, 1.0):
                with self.subTest(model=model, adversary=lam):
                    pred, net = adapted.train_predict(data, train, test, mapping, model=model,
                                                      lam=lam, epochs=1, batch=40, seed=31,
                                                      return_model=True)
                    self.assertEqual(pred.shape, (80,))
                    self.assertTrue(np.all((pred >= 0) & (pred < 7)))
                    self.assertEqual(net.cls.out_features, 7)
                    self.assertEqual(net.adv[-1].out_features, 80)
                    with torch.no_grad():
                        z, logits = net(torch.from_numpy(data["X"][test]), torch.from_numpy(data["A"][test]))
                        probabilities = torch.softmax(logits, dim=1)
                        onehot = torch.nn.functional.one_hot(torch.from_numpy(mapping), 7).float()
                        adversary = net.adv(torch.cat([z, onehot], dim=1))
                    self.assertEqual(tuple(probabilities.shape), (80, 7))
                    self.assertEqual(tuple(adversary.shape), (80, 80))
                    self.assertTrue(torch.isfinite(probabilities).all().item())
                    torch.testing.assert_close(probabilities.sum(1), torch.ones(80))
                    np.testing.assert_array_equal(pred, logits.argmax(1).numpy())

    def test_other_constructor_paths_receive_dynamic_dimensions(self):
        mapping = np.repeat(np.arange(7), [12, 12, 12, 12, 12, 12, 8])
        data, train, test = synthetic_data(mapping)
        for model in ("sdgnn", "cgnn", "mlp"):
            with self.subTest(model=model):
                _, net = adapted.train_predict(data, train, test, mapping, model=model,
                                                epochs=1, batch=40, seed=31, return_model=True)
                with torch.no_grad():
                    _, logits = net(torch.from_numpy(data["X"][test]), torch.from_numpy(data["A"][test]))
                self.assertEqual(tuple(logits.shape), (80, 7))
                if model != "mlp":
                    self.assertEqual(net.adv[-1].out_features, 80)

    def test_exact_faced_equivalence_after_one_epoch(self):
        mapping = np.repeat(np.arange(9), [3, 3, 3, 3, 4, 3, 3, 3, 3])
        data, train, test = synthetic_data(mapping)
        for model in ("hybrid", "sfgnn"):
            for lam in (0.0, 1.0):
                with self.subTest(model=model, adversary=lam):
                    kwargs = dict(model=model, lam=lam, epochs=1, batch=14, seed=73, return_model=True)
                    old_pred, old_net = self.original.train_predict(data, train, test, mapping, **kwargs)
                    new_pred, new_net = adapted.train_predict(data, train, test, mapping, **kwargs)
                    np.testing.assert_array_equal(old_pred, new_pred)
                    self.assertEqual(old_net.state_dict().keys(), new_net.state_dict().keys())
                    for name, tensor in old_net.state_dict().items():
                        torch.testing.assert_close(tensor, new_net.state_dict()[name], rtol=0, atol=0)
                    with torch.no_grad():
                        inputs = (torch.from_numpy(data["X"][test]), torch.from_numpy(data["A"][test]))
                        old_logits = old_net(*inputs)[1]
                        new_logits = new_net(*inputs)[1]
                    torch.testing.assert_close(old_logits, new_logits, rtol=0, atol=0)

    def test_invalid_condition_maps_and_video_ids_fail_clearly(self):
        mapping = np.repeat(np.arange(7), [12, 12, 12, 12, 12, 12, 8])
        data, train, test = synthetic_data(mapping)
        for invalid in (np.array([], dtype=int), mapping + 1, mapping.astype(float),
                        np.where(mapping == 1, 0, mapping), mapping.reshape(8, 10)):
            with self.subTest(mapping_shape=invalid.shape):
                with self.assertRaisesRegex(ValueError, "video_to_cls"):
                    adapted.train_predict(data, train, test, invalid, epochs=0)
        invalid = dict(data, video=data["video"].copy())
        invalid["video"][0] = 80
        with self.assertRaisesRegex(ValueError, "video ID"):
            adapted.train_predict(invalid, train, test, mapping, epochs=0)

    def test_lovo_iterates_present_video_ids_including_above_27(self):
        mapping = np.arange(80, dtype=np.int64) % 7
        data, _, _ = synthetic_data(mapping)
        chosen = np.isin(data["video"], [2, 43, 79])
        small = {key: value[chosen] for key, value in data.items()}
        held = []
        def fake_train_predict(data, train, test, video_to_cls, seed, **kwargs):
            video = np.unique(data["video"][test]).item()
            held.append(video)
            self.assertEqual(seed, video)
            self.assertFalse(np.any(data["video"][train] == video))
            return video_to_cls[data["video"][test]]
        with patch.object(adapted, "train_predict", side_effect=fake_train_predict):
            prediction = adapted.lovo(small, mapping)
        self.assertEqual(held, [2, 43, 79])
        np.testing.assert_array_equal(prediction, small["y"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
