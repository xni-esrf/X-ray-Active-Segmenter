from __future__ import annotations

import unittest

try:
    import torch
except Exception:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]

from src.learning import DestVolBuffer, EvalBBoxDataset, InferenceDestVolBuffer


@unittest.skipUnless(torch is not None, "PyTorch is not available")
class EvalBBoxDatasetTests(unittest.TestCase):
    def test_len_and_item_shape(self) -> None:
        vol = torch.arange(6 * 6 * 6, dtype=torch.float32).reshape((6, 6, 6))
        dataset = EvalBBoxDataset(vol, minivol_size=4)

        self.assertEqual(len(dataset), 8)
        patch, coords = dataset[0]
        self.assertEqual(tuple(patch.shape), (1, 4, 4, 4))
        self.assertEqual(coords, (0, 0, 0))

    def test_rejects_out_of_range_index(self) -> None:
        vol = torch.ones((6, 6, 6), dtype=torch.float32)
        dataset = EvalBBoxDataset(vol, minivol_size=4)
        with self.assertRaises(IndexError):
            _ = dataset[len(dataset)]


@unittest.skipUnless(torch is not None, "PyTorch is not available")
class DestVolBufferTests(unittest.TestCase):
    def test_accuracy_uses_label_mapping_and_mask(self) -> None:
        ground_truth = torch.tensor(
            [
                [[5, 5], [5, -100]],
                [[5, 5], [5, 5]],
            ],
            dtype=torch.int16,
        )
        label_values = (0, 1, 5, 6)
        buffer = DestVolBuffer(
            ground_truth,
            volume_shape=tuple(int(v) for v in ground_truth.shape),
            label_values=label_values,
            minivol_size=2,
        )

        batch = torch.zeros((1, 4, 2, 2, 2), dtype=torch.float32)
        # Channel index 2 maps to label value 5.
        batch[0, 2, :, :, :] = 2.0
        buffer.add_batch(batch, ([0], [0], [0]))

        accuracy = buffer.get_acc_pred()
        self.assertAlmostEqual(float(accuracy.item()), 1.0, places=6)

    def test_rejects_mask_label_in_label_values(self) -> None:
        ground_truth = torch.zeros((2, 2, 2), dtype=torch.int16)
        with self.assertRaises(ValueError):
            DestVolBuffer(
                ground_truth,
                volume_shape=(2, 2, 2),
                label_values=(0, 1, -100),
                minivol_size=2,
            )


@unittest.skipUnless(torch is not None, "PyTorch is not available")
class InferenceDestVolBufferTests(unittest.TestCase):
    def test_get_pred_labels_uses_label_mapping(self) -> None:
        buffer = InferenceDestVolBuffer(
            volume_shape=(2, 2, 2),
            label_values=(0, 5, 9),
            minivol_size=2,
        )

        batch = torch.zeros((1, 3, 2, 2, 2), dtype=torch.float32)
        # Channel index 1 maps to label value 5.
        batch[0, 1, :, :, :] = 4.0
        buffer.add_batch(batch, ([0], [0], [0]))

        pred = buffer.get_pred_labels()
        self.assertEqual(tuple(pred.shape), (2, 2, 2))
        self.assertTrue(bool(torch.all(pred == 5).item()))

    def test_rejects_mask_label_in_label_values(self) -> None:
        with self.assertRaises(ValueError):
            InferenceDestVolBuffer(
                volume_shape=(2, 2, 2),
                label_values=(0, 1, -100),
                minivol_size=2,
            )


if __name__ == "__main__":
    unittest.main()
