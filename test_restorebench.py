import unittest
import numpy as np
import torch
from restorebench import RestoreCNN, bicubic, data, degrade, fit, metrics


class RestoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_shape(self):
        self.assertEqual(tuple(RestoreCNN()(torch.zeros(2, 1, 4, 4)).shape), (2, 1, 8, 8))

    def test_degradation(self):
        self.assertEqual(tuple(degrade(torch.ones(2, 1, 8, 8)).shape), (2, 1, 4, 4))

    def test_constant_preservation(self):
        torch.testing.assert_close(bicubic(torch.ones(2, 1, 4, 4)), torch.ones(2, 1, 8, 8))

    def test_initial_matches_bicubic(self):
        low = torch.rand(2, 1, 4, 4)
        torch.testing.assert_close(RestoreCNN()(low), bicubic(low))

    def test_bad_shape(self):
        with self.assertRaises(ValueError):
            RestoreCNN()(torch.zeros(2, 16))

    def test_nonfinite(self):
        with self.assertRaises(ValueError):
            degrade(torch.full((2, 1, 8, 8), float('nan')))

    def test_psnr(self):
        stats, _ = metrics(torch.full((2, 1, 8, 8), 0.1), torch.zeros(2, 1, 8, 8))
        self.assertAlmostEqual(stats['mean_image_psnr_db'], 20, places=4)

    def test_perfect_psnr_cap(self):
        stats, _ = metrics(torch.zeros(2, 1, 8, 8), torch.zeros(2, 1, 8, 8))
        self.assertAlmostEqual(stats['mean_image_psnr_db'], 120, places=3)

    def test_mismatch(self):
        with self.assertRaises(ValueError):
            metrics(torch.zeros(2, 1, 8, 8), torch.zeros(2, 1, 4, 4))

    def test_splits(self):
        _, train, val, test = data()
        self.assertFalse(set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
        self.assertEqual(len(train) + len(val) + len(test), 1797)

    def test_finite_gradients(self):
        model = RestoreCNN()
        model(torch.rand(2, 1, 4, 4)).sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_training_roundtrip(self):
        images, train, val, _ = data()
        model, selected, history = fit(images, train[:32], val[:8], epochs=2)
        self.assertIn(selected, (1, 2))
        self.assertEqual(len(history), 2)
        restored = RestoreCNN().eval()
        restored.load_state_dict(model.state_dict())
        low = degrade(images[val[:2]])
        torch.testing.assert_close(model(low), restored(low))


if __name__ == '__main__':
    unittest.main()
