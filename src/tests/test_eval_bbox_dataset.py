from __future__ import annotations

import unittest

try:
    import torch
except Exception:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]

from src.learning import (
    DestVolBuffer,
    EvalBBoxDataset,
    InferenceDestVolBuffer,
    TiledInferenceDestVolBuffer,
)
from src.learning.eval_bbox_dataset import (
    _add_weighted_minivol_to_buffer_region,
    _weighted_minivol_intersection,
    compute_volume_weighted_mean_score,
)


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
class WeightedMinivolIntersectionTests(unittest.TestCase):
    def test_full_buffer_intersection_uses_original_minivol_slices(self) -> None:
        placement = _weighted_minivol_intersection(
            minivol_coordinates=(2, 4, 6),
            minivol_size=4,
            target_origin=(0, 0, 0),
            target_shape=(10, 12, 14),
        )

        self.assertIsNotNone(placement)
        assert placement is not None
        self.assertEqual(placement.target_slices, (slice(2, 6), slice(4, 8), slice(6, 10)))
        self.assertEqual(placement.minivol_slices, (slice(0, 4), slice(0, 4), slice(0, 4)))

    def test_tile_intersection_returns_relative_target_and_minivol_slices(self) -> None:
        placement = _weighted_minivol_intersection(
            minivol_coordinates=(2, 4, 6),
            minivol_size=4,
            target_origin=(4, 5, 6),
            target_shape=(3, 3, 3),
        )

        self.assertIsNotNone(placement)
        assert placement is not None
        self.assertEqual(placement.target_slices, (slice(0, 2), slice(0, 3), slice(0, 3)))
        self.assertEqual(placement.minivol_slices, (slice(2, 4), slice(1, 4), slice(0, 3)))

    def test_disjoint_tile_intersection_returns_none(self) -> None:
        placement = _weighted_minivol_intersection(
            minivol_coordinates=(0, 0, 0),
            minivol_size=4,
            target_origin=(5, 0, 0),
            target_shape=(2, 2, 2),
        )

        self.assertIsNone(placement)

    def test_add_weighted_minivol_to_buffer_region_matches_manual_partial_update(self) -> None:
        minivol = torch.arange(1 * 4 * 4 * 4, dtype=torch.float32).reshape((1, 4, 4, 4))
        hann = torch.ones((1, 4, 4, 4), dtype=torch.float32) * 2.0
        buffer = torch.zeros((1, 3, 3, 3), dtype=torch.float32)

        added = _add_weighted_minivol_to_buffer_region(
            minivol=minivol,
            minivol_coordinates=(2, 4, 6),
            buffer_vol=buffer,
            hann_window=hann,
            minivol_size=4,
            target_origin=(4, 5, 6),
        )

        expected = torch.zeros((1, 3, 3, 3), dtype=torch.float32)
        expected[:, 0:2, 0:3, 0:3] = minivol[:, 2:4, 1:4, 0:3] * 2.0
        self.assertTrue(added)
        self.assertTrue(bool(torch.equal(buffer, expected)))


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

    def test_get_pred_labels_decodes_non_contiguous_training_labels(self) -> None:
        buffer = InferenceDestVolBuffer(
            volume_shape=(2, 2, 2),
            label_values=(0, 1, 2, 4),
            minivol_size=2,
        )

        buffer.buffer_vol[3] = torch.ones((2, 2, 2), dtype=torch.float32)

        pred = buffer.get_pred_labels()
        self.assertEqual(tuple(pred.shape), (2, 2, 2))
        self.assertTrue(bool(torch.all(pred == 4).item()))

    def test_rejects_mask_label_in_label_values(self) -> None:
        with self.assertRaises(ValueError):
            InferenceDestVolBuffer(
                volume_shape=(2, 2, 2),
                label_values=(0, 1, -100),
                minivol_size=2,
            )

    def test_tiled_accumulation_matches_full_buffer_labels_for_aligned_tiles(self) -> None:
        label_values = (0, 5, 9)
        volume_shape = (4, 4, 4)
        minivol_size = 2
        batch = torch.zeros((3, len(label_values), minivol_size, minivol_size, minivol_size))
        batch[0, 1] = 2.0
        batch[1, 2] = 4.0
        batch[2, 0] = 6.0
        batch_coordinates = (
            torch.tensor([0, 1, 2]),
            torch.tensor([0, 1, 2]),
            torch.tensor([0, 1, 2]),
        )

        full_buffer = InferenceDestVolBuffer(
            volume_shape=volume_shape,
            label_values=label_values,
            minivol_size=minivol_size,
        )
        full_buffer.add_batch(batch, batch_coordinates)

        tiled_buffer = TiledInferenceDestVolBuffer(
            volume_shape=volume_shape,
            label_values=label_values,
            minivol_size=minivol_size,
            tile_shape=(2, 2, 2),
        )
        try:
            tiled_buffer.add_batch(batch, batch_coordinates)
            tiled_labels = tiled_buffer.get_pred_labels()
        finally:
            tiled_buffer.close()

        self.assertTrue(bool(torch.equal(tiled_labels, full_buffer.get_pred_labels())))

    def test_tiled_accumulation_matches_full_buffer_labels_for_edge_tiles(self) -> None:
        label_values = (1, 7)
        volume_shape = (5, 4, 3)
        minivol_size = 2
        batch = torch.zeros((4, len(label_values), minivol_size, minivol_size, minivol_size))
        batch[0, 0] = 1.0
        batch[0, 1] = 3.0
        batch[1, 0] = 5.0
        batch[2, 1] = 7.0
        batch[3, 0] = 2.0
        batch[3, 1] = 2.5
        batch_coordinates = (
            torch.tensor([0, 2, 3, 1]),
            torch.tensor([0, 1, 2, 0]),
            torch.tensor([0, 1, 1, 1]),
        )

        full_buffer = InferenceDestVolBuffer(
            volume_shape=volume_shape,
            label_values=label_values,
            minivol_size=minivol_size,
        )
        full_buffer.add_batch(batch, batch_coordinates)

        tiled_buffer = TiledInferenceDestVolBuffer(
            volume_shape=volume_shape,
            label_values=label_values,
            minivol_size=minivol_size,
            tile_shape=(3, 2, 2),
        )
        try:
            tiled_buffer.add_batch(batch, batch_coordinates)
            tiled_labels = tiled_buffer.get_pred_labels()
        finally:
            tiled_buffer.close()

        self.assertTrue(bool(torch.equal(tiled_labels, full_buffer.get_pred_labels())))

    def test_tiled_buffer_creates_memmap_tiles_and_cleans_temporary_directory(self) -> None:
        buffer = TiledInferenceDestVolBuffer(
            volume_shape=(4, 4, 4),
            label_values=(0, 1),
            minivol_size=2,
            tile_shape=(2, 2, 2),
        )
        temp_dir = buffer.temp_dir_path
        batch = torch.ones((1, 2, 2, 2, 2), dtype=torch.float32)
        batch_coordinates = (
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
        )

        buffer.add_batch(batch, batch_coordinates)
        tile_paths = tuple(buffer._tile_paths.values())

        self.assertTrue(temp_dir.exists())
        self.assertTrue(tile_paths)
        self.assertTrue(all(path.exists() for path in tile_paths))

        buffer.close()

        self.assertFalse(temp_dir.exists())

        buffer.close()
        self.assertFalse(temp_dir.exists())

    def test_tiled_buffer_context_manager_cleans_temporary_directory(self) -> None:
        with TiledInferenceDestVolBuffer(
            volume_shape=(4, 4, 4),
            label_values=(0, 1),
            minivol_size=2,
            tile_shape=(2, 2, 2),
        ) as buffer:
            temp_dir = buffer.temp_dir_path
            batch = torch.ones((1, 2, 2, 2, 2), dtype=torch.float32)
            batch_coordinates = (
                torch.tensor([0]),
                torch.tensor([0]),
                torch.tensor([0]),
            )
            buffer.add_batch(batch, batch_coordinates)
            self.assertTrue(temp_dir.exists())

        self.assertFalse(temp_dir.exists())

    def test_tiled_buffer_cleanup_aliases_delete_temporary_directory(self) -> None:
        for method_name in ("shutdown", "stop", "terminate"):
            buffer = TiledInferenceDestVolBuffer(
                volume_shape=(4, 4, 4),
                label_values=(0, 1),
                minivol_size=2,
                tile_shape=(2, 2, 2),
            )
            temp_dir = buffer.temp_dir_path
            batch = torch.ones((1, 2, 2, 2, 2), dtype=torch.float32)
            batch_coordinates = (
                torch.tensor([0]),
                torch.tensor([0]),
                torch.tensor([0]),
            )
            buffer.add_batch(batch, batch_coordinates)

            getattr(buffer, method_name)()

            self.assertFalse(temp_dir.exists(), method_name)


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
