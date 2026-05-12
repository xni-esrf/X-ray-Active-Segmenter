from __future__ import annotations

import unittest

try:
    import torch
except Exception:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]

from src.learning import DestVolBuffer, EvalBBoxDataset, InferenceDestVolBuffer
from src.learning.eval_bbox_dataset import compute_volume_weighted_mean_score


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
    def test_dice_uses_label_mapping_and_mask(self) -> None:
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

        pred_labels = torch.full(tuple(int(v) for v in ground_truth.shape), 5, dtype=torch.int16)
        for channel_index, label in enumerate(label_values):
            buffer.buffer_vol[channel_index] = (pred_labels == label).to(dtype=torch.float32)

        dice = buffer.get_dice_pred()
        self.assertAlmostEqual(float(dice.item()), 1.0, places=6)

    def test_rejects_mask_label_in_label_values(self) -> None:
        ground_truth = torch.zeros((2, 2, 2), dtype=torch.int16)
        with self.assertRaises(ValueError):
            DestVolBuffer(
                ground_truth,
                volume_shape=(2, 2, 2),
                label_values=(0, 1, -100),
                minivol_size=2,
            )

    def test_dice_keeps_boundary_class_with_inclusive_threshold(self) -> None:
        ground_truth_flat = torch.tensor(
            ([0] * 100) + ([2] * 24) + ([1] * 1),
            dtype=torch.int16,
        )
        ground_truth = ground_truth_flat.reshape((5, 5, 5))
        pred_labels = ground_truth.clone()
        pred_labels.reshape(-1)[124] = 0

        buffer = DestVolBuffer(
            ground_truth,
            volume_shape=(5, 5, 5),
            label_values=(0, 1, 2),
            minivol_size=2,
        )
        for channel_index, label in enumerate((0, 1, 2)):
            buffer.buffer_vol[channel_index] = (pred_labels == label).to(dtype=torch.float32)

        dice = float(buffer.get_dice_pred().item())
        expected = ((200.0 / 201.0) + 0.0 + 1.0) / 3.0
        self.assertAlmostEqual(dice, expected, places=6)

    def test_dice_can_filter_background_when_it_is_rare(self) -> None:
        ground_truth_flat = torch.tensor(
            ([1] * 991) + ([0] * 9),
            dtype=torch.int16,
        )
        ground_truth = ground_truth_flat.reshape((10, 10, 10))
        pred_labels = ground_truth.clone()
        pred_labels.reshape(-1)[0] = 0

        buffer = DestVolBuffer(
            ground_truth,
            volume_shape=(10, 10, 10),
            label_values=(0, 1),
            minivol_size=2,
        )
        for channel_index, label in enumerate((0, 1)):
            buffer.buffer_vol[channel_index] = (pred_labels == label).to(dtype=torch.float32)

        dice = float(buffer.get_dice_pred().item())
        expected_class_1_dice = 1980.0 / 1981.0
        self.assertAlmostEqual(dice, expected_class_1_dice, places=6)

    def test_dice_rejects_fully_masked_ground_truth(self) -> None:
        ground_truth = torch.full((2, 2, 2), -100, dtype=torch.int16)
        buffer = DestVolBuffer(
            ground_truth,
            volume_shape=(2, 2, 2),
            label_values=(0, 1),
            minivol_size=2,
        )
        with self.assertRaisesRegex(ValueError, "No valid annotated voxels found"):
            _ = buffer.get_dice_pred()


@unittest.skipUnless(torch is not None, "PyTorch is not available")
class InferenceDestVolBufferTests(unittest.TestCase):
    def test_get_pred_labels_uses_label_mapping(self) -> None:
        buffer = InferenceDestVolBuffer(
            volume_shape=(2, 2, 2),
            label_values=(0, 5, 9),
            minivol_size=2,
        )

        pred_labels = torch.full((2, 2, 2), 5, dtype=torch.int16)
        for channel_index, label in enumerate((0, 5, 9)):
            buffer.buffer_vol[channel_index] = (pred_labels == label).to(dtype=torch.float32)

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


class VolumeWeightedMeanScoreTests(unittest.TestCase):
    def test_computes_standard_volume_weighted_mean(self) -> None:
        weighted = compute_volume_weighted_mean_score(
            score_by_box_id={"bbox_a": 0.5, "bbox_b": 0.8},
            bbox_volume_by_box_id={"bbox_a": 100, "bbox_b": 300},
        )
        self.assertAlmostEqual(weighted, 0.725, places=8)

    def test_rejects_missing_bbox_volume(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing bbox volume"):
            compute_volume_weighted_mean_score(
                score_by_box_id={"bbox_a": 0.5},
                bbox_volume_by_box_id={},
            )


if __name__ == "__main__":
    unittest.main()
