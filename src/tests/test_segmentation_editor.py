from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.annotation import SegmentationEditor
from src.data import build_segmentation_pyramid_lazy, open_volume
from src.io.loader import InMemoryVolumeLoader
from src.io.saver import save_segmentation_volume


class SegmentationEditorTests(unittest.TestCase):
    def test_default_active_label_is_lowest_missing_positive(self) -> None:
        volume = np.array(
            [
                [[0, 1, 2]],
                [[1, 2, 4]],
            ],
            dtype=np.uint16,
        )
        editor = SegmentationEditor(volume, kind="instance")
        self.assertEqual(editor.active_label, 3)

    def test_paint_voxel_updates_data_and_history(self) -> None:
        editor = SegmentationEditor.create_empty((3, 3, 3), kind="semantic", dtype=np.uint8)

        operation = editor.paint_voxel((1, 1, 1), label=7)

        self.assertEqual(operation.name, "paint_voxel")
        self.assertEqual(operation.changed_voxels, 1)
        self.assertTrue(editor.dirty)
        self.assertEqual(int(editor.array_view()[1, 1, 1]), 7)
        self.assertEqual(len(editor.history), 1)
        history_entry = editor.history[0]
        np.testing.assert_array_equal(history_entry.delta.previous_values, np.array([0], dtype=np.uint8))
        np.testing.assert_array_equal(history_entry.delta.new_values, np.array([7], dtype=np.uint8))

    def test_paint_stroke_interpolates_between_points(self) -> None:
        editor = SegmentationEditor.create_empty((1, 1, 5), kind="instance", dtype=np.uint8)

        operation = editor.paint_stroke([(0, 0, 0), (0, 0, 4)], label=2)

        self.assertEqual(operation.changed_voxels, 5)
        np.testing.assert_array_equal(
            editor.array_view()[0, 0, :],
            np.array([2, 2, 2, 2, 2], dtype=np.uint8),
        )

    def test_paint_brush_voxel_respects_radius_footprint(self) -> None:
        editor = SegmentationEditor.create_empty((9, 9, 9), kind="semantic", dtype=np.uint8)

        center = (4, 4, 4)
        r0 = editor.paint_brush_voxel(center, axis=0, brush_radius=0, label=1)
        r1 = editor.paint_brush_voxel(center, axis=0, brush_radius=1, label=2)
        r2 = editor.paint_brush_voxel(center, axis=0, brush_radius=2, label=3)
        r3 = editor.paint_brush_voxel(center, axis=0, brush_radius=3, label=4)
        r4 = editor.paint_brush_voxel(center, axis=0, brush_radius=4, label=5)

        self.assertEqual(r0.changed_voxels, 1)
        self.assertEqual(r1.changed_voxels, 7)
        self.assertEqual(r2.changed_voxels, 33)
        self.assertEqual(r3.changed_voxels, 123)
        self.assertEqual(r4.changed_voxels, 257)

    def test_paint_brush_voxel_uses_spherical_footprint(self) -> None:
        editor = SegmentationEditor.create_empty((5, 5, 5), kind="instance", dtype=np.uint8)

        operation = editor.paint_brush_voxel((2, 2, 2), axis=1, brush_radius=1, label=9)

        self.assertEqual(operation.changed_voxels, 7)
        changed = np.argwhere(editor.array_view() == 9)
        self.assertGreater(changed.shape[0], 0)
        # A sphere spans neighboring slices along every axis.
        self.assertEqual(np.min(changed[:, 0]), 1)
        self.assertEqual(np.max(changed[:, 0]), 3)
        self.assertEqual(np.min(changed[:, 1]), 1)
        self.assertEqual(np.max(changed[:, 1]), 3)
        self.assertEqual(np.min(changed[:, 2]), 1)
        self.assertEqual(np.max(changed[:, 2]), 3)

    def test_paint_brush_voxel_rejects_out_of_range_radius(self) -> None:
        editor = SegmentationEditor.create_empty((5, 5, 5), kind="semantic", dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "Brush radius must be between 0 and 9"):
            editor.paint_brush_voxel((2, 2, 2), axis=0, brush_radius=10, label=1)

    def test_erase_brush_voxel_clears_nonzero_labels(self) -> None:
        editor = SegmentationEditor.create_empty((9, 9, 9), kind="semantic", dtype=np.uint8)
        center = (4, 4, 4)
        editor.paint_brush_voxel(center, axis=0, brush_radius=2, label=12)
        editor.mark_clean()

        operation = editor.erase_brush_voxel(center, axis=0, brush_radius=2)

        self.assertEqual(operation.name, "erase_brush_voxel")
        self.assertEqual(operation.changed_voxels, 33)
        self.assertEqual(int(np.count_nonzero(editor.array_view())), 0)
        self.assertTrue(editor.dirty)

    def test_erase_brush_voxel_can_target_specific_label(self) -> None:
        editor = SegmentationEditor.create_empty((9, 9, 9), kind="instance", dtype=np.uint8)
        center = (4, 4, 4)
        editor.paint_brush_voxel(center, axis=0, brush_radius=1, label=3)
        editor.paint_voxel(center, label=9)
        editor.mark_clean()

        operation = editor.erase_brush_voxel(
            center,
            axis=0,
            brush_radius=1,
            target_label=3,
        )

        self.assertEqual(operation.changed_voxels, 6)
        self.assertEqual(int(editor.array_view()[center]), 9)
        self.assertEqual(int(np.count_nonzero(editor.array_view() == 3)), 0)
        self.assertEqual(int(np.count_nonzero(editor.array_view() == 9)), 1)
        self.assertTrue(editor.dirty)

        no_change = editor.erase_brush_voxel(
            center,
            axis=0,
            brush_radius=1,
            target_label=42,
        )
        self.assertEqual(no_change.changed_voxels, 0)

    def test_next_available_label_reflects_changes(self) -> None:
        editor = SegmentationEditor.create_empty((1, 1, 3), kind="instance", dtype=np.uint8)
        editor.paint_voxel((0, 0, 0), label=1)
        editor.paint_voxel((0, 0, 1), label=2)
        self.assertEqual(editor.next_available_label(), 3)

        editor.paint_voxel((0, 0, 1), label=0)
        self.assertEqual(editor.next_available_label(), 2)

    def test_merge_labels_reassigns_to_target_label(self) -> None:
        array = np.array([[[1, 2, 3, 2]]], dtype=np.uint8)
        editor = SegmentationEditor(array, kind="semantic")

        operation = editor.merge_labels([2, 3], target_label=5)

        self.assertEqual(operation.name, "merge_labels")
        self.assertEqual(operation.changed_voxels, 3)
        np.testing.assert_array_equal(
            editor.array_view(),
            np.array([[[1, 5, 5, 5]]], dtype=np.uint8),
        )

    def test_flood_fill_reassigns_only_connected_component(self) -> None:
        array = np.array(
            [
                [
                    [1, 1, 0, 1, 1],
                    [1, 1, 0, 1, 1],
                ]
            ],
            dtype=np.uint8,
        )
        editor = SegmentationEditor(array, kind="semantic")

        operation = editor.flood_fill_from_seed((0, 0, 0), label=7)

        self.assertEqual(operation.name, "flood_fill")
        self.assertEqual(operation.changed_voxels, 4)
        np.testing.assert_array_equal(
            editor.array_view(),
            np.array(
                [
                    [
                        [7, 7, 0, 1, 1],
                        [7, 7, 0, 1, 1],
                    ]
                ],
                dtype=np.uint8,
            ),
        )
        self.assertTrue(editor.dirty)
        self.assertEqual(len(editor.history), 1)

    def test_flood_fill_noop_when_target_matches_source(self) -> None:
        array = np.array([[[2, 2], [2, 0]]], dtype=np.uint8)
        editor = SegmentationEditor(array, kind="instance")
        editor.mark_clean()

        operation = editor.flood_fill_from_seed((0, 0, 0), label=2)

        self.assertEqual(operation.changed_voxels, 0)
        np.testing.assert_array_equal(editor.array_view(), array)
        self.assertFalse(editor.dirty)

    def test_flood_fill_timeout_cancels_and_restores_partial_changes(self) -> None:
        array = np.ones((1, 128, 128), dtype=np.uint8)
        editor = SegmentationEditor(array, kind="semantic")
        before = np.array(editor.array_view(), copy=True)

        with self.assertRaisesRegex(ValueError, "exceeded time limit"):
            editor.flood_fill_from_seed(
                (0, 0, 0),
                label=2,
                max_duration_seconds=1e-9,
            )

        np.testing.assert_array_equal(editor.array_view(), before)
        self.assertEqual(editor.undo_depth(), 0)
        self.assertFalse(editor.dirty)

    def test_undo_reverts_single_auto_modification(self) -> None:
        editor = SegmentationEditor.create_empty((1, 1, 3), kind="semantic", dtype=np.uint8)
        editor.mark_clean()
        editor.paint_voxel((0, 0, 1), label=9)

        self.assertEqual(editor.undo_depth(), 1)
        self.assertTrue(editor.dirty)

        operation = editor.undo_last_modification()
        self.assertIsNotNone(operation)
        np.testing.assert_array_equal(editor.array_view(), np.zeros((1, 1, 3), dtype=np.uint8))
        self.assertEqual(editor.undo_depth(), 0)
        self.assertFalse(editor.dirty)

    def test_latest_undo_and_redo_operation_ids_track_stack_tops(self) -> None:
        editor = SegmentationEditor.create_empty((1, 1, 3), kind="semantic", dtype=np.uint8)
        editor.paint_voxel((0, 0, 0), label=1)
        op1_undo_id = editor.latest_undo_operation_id()
        self.assertIsNotNone(op1_undo_id)
        editor.paint_voxel((0, 0, 1), label=2)
        op2_undo_id = editor.latest_undo_operation_id()
        self.assertIsNotNone(op2_undo_id)

        self.assertEqual(editor.latest_undo_operation_id(), op2_undo_id)
        self.assertIsNone(editor.latest_redo_operation_id())

        editor.undo_last_modification()
        self.assertEqual(editor.latest_undo_operation_id(), op1_undo_id)
        self.assertEqual(editor.latest_redo_operation_id(), op2_undo_id)

    def test_discard_undo_and_redo_operation_removes_specific_entries(self) -> None:
        editor = SegmentationEditor.create_empty((1, 1, 3), kind="semantic", dtype=np.uint8)
        editor.paint_voxel((0, 0, 0), label=1)
        op1_id = editor.latest_undo_operation_id()
        self.assertIsNotNone(op1_id)
        editor.paint_voxel((0, 0, 1), label=2)
        op2_id = editor.latest_undo_operation_id()
        self.assertIsNotNone(op2_id)
        self.assertEqual(editor.undo_depth(), 2)

        self.assertTrue(editor.discard_undo_operation(int(op1_id)))
        self.assertEqual(editor.undo_depth(), 1)
        self.assertEqual(editor.latest_undo_operation_id(), op2_id)

        editor.undo_last_modification()
        self.assertEqual(editor.redo_depth(), 1)
        self.assertEqual(editor.latest_redo_operation_id(), op2_id)

        self.assertTrue(editor.discard_redo_operation(int(op2_id)))
        self.assertEqual(editor.redo_depth(), 0)
        self.assertIsNone(editor.latest_redo_operation_id())

    def test_undo_groups_changes_from_explicit_modification(self) -> None:
        editor = SegmentationEditor.create_empty((1, 1, 3), kind="instance", dtype=np.uint8)
        editor.begin_modification("annotation_stroke")
        editor.paint_voxel((0, 0, 0), label=3)
        editor.paint_voxel((0, 0, 1), label=3)
        editor.paint_voxel((0, 0, 0), label=7)
        editor.commit_modification()

        self.assertEqual(editor.undo_depth(), 1)
        np.testing.assert_array_equal(
            editor.array_view(),
            np.array([[[7, 3, 0]]], dtype=np.uint8),
        )

        editor.undo_last_modification()
        np.testing.assert_array_equal(
            editor.array_view(),
            np.array([[[0, 0, 0]]], dtype=np.uint8),
        )

    def test_undo_keeps_last_ten_modifications(self) -> None:
        editor = SegmentationEditor.create_empty((1, 1, 11), kind="semantic", dtype=np.uint8)

        for index in range(11):
            editor.paint_voxel((0, 0, index), label=index + 1)

        self.assertEqual(editor.undo_depth(), 10)
        for _ in range(10):
            editor.undo_last_modification()

        np.testing.assert_array_equal(
            editor.array_view(),
            np.array([[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=np.uint8),
        )

    def test_redo_reapplies_last_undo(self) -> None:
        editor = SegmentationEditor.create_empty((1, 1, 3), kind="semantic", dtype=np.uint8)
        editor.mark_clean()
        editor.paint_voxel((0, 0, 1), label=9)

        editor.undo_last_modification()
        self.assertEqual(editor.undo_depth(), 0)
        self.assertEqual(editor.redo_depth(), 1)
        self.assertFalse(editor.dirty)

        operation = editor.redo_last_modification()
        self.assertIsNotNone(operation)
        np.testing.assert_array_equal(
            editor.array_view(),
            np.array([[[0, 9, 0]]], dtype=np.uint8),
        )
        self.assertEqual(editor.undo_depth(), 1)
        self.assertEqual(editor.redo_depth(), 0)
        self.assertTrue(editor.dirty)

    def test_redo_is_cleared_by_new_modification(self) -> None:
        editor = SegmentationEditor.create_empty((1, 1, 4), kind="instance", dtype=np.uint8)
        editor.paint_voxel((0, 0, 0), label=1)
        editor.paint_voxel((0, 0, 1), label=2)

        editor.undo_last_modification()
        self.assertEqual(editor.redo_depth(), 1)

        editor.paint_voxel((0, 0, 2), label=3)
        self.assertEqual(editor.redo_depth(), 0)
        self.assertIsNone(editor.redo_last_modification())
        np.testing.assert_array_equal(
            editor.array_view(),
            np.array([[[1, 0, 3, 0]]], dtype=np.uint8),
        )

    def test_redo_roundtrip_with_ten_modification_limit(self) -> None:
        editor = SegmentationEditor.create_empty((1, 1, 11), kind="semantic", dtype=np.uint8)
        for index in range(11):
            editor.paint_voxel((0, 0, index), label=index + 1)

        for _ in range(10):
            editor.undo_last_modification()
        self.assertEqual(editor.redo_depth(), 10)

        for _ in range(10):
            editor.redo_last_modification()
        self.assertEqual(editor.undo_depth(), 10)
        self.assertEqual(editor.redo_depth(), 0)
        np.testing.assert_array_equal(
            editor.array_view(),
            np.array([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]], dtype=np.uint8),
        )

    def test_save_uses_editor_backed_volume_data(self) -> None:
        editor = SegmentationEditor.create_empty((1, 2, 3), kind="instance", dtype=np.uint16)
        editor.paint_voxel((0, 1, 2), label=17)
        volume = editor.to_volume_data(path="editable-instance")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = str(Path(temp_dir) / "segmentation.npy")
            save_segmentation_volume(volume, output_path, save_format="npy")
            loaded = np.load(output_path)

        self.assertEqual(loaded.shape, (1, 2, 3))
        self.assertEqual(loaded.dtype, np.dtype(np.uint16))
        self.assertEqual(int(loaded[0, 1, 2]), 17)

    def test_save_refuses_overwrite_without_explicit_flag(self) -> None:
        editor = SegmentationEditor.create_empty((1, 1, 1), kind="semantic", dtype=np.uint8)
        volume = editor.to_volume_data(path="editable-semantic")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "segmentation.npy"
            output_path.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                save_segmentation_volume(volume, str(output_path), save_format="npy")
            save_segmentation_volume(
                volume,
                str(output_path),
                save_format="npy",
                overwrite=True,
            )

    def test_segmentation_pyramid_preserves_off_grid_single_voxel(self) -> None:
        array = np.zeros((4, 4, 4), dtype=np.uint16)
        array[1, 1, 1] = 13
        volume = open_volume(
            InMemoryVolumeLoader(
                path="seg",
                array=array,
                voxel_spacing=(1.0, 1.0, 1.0),
                axes="zyx",
            ),
            cache=None,
        )
        levels = build_segmentation_pyramid_lazy(volume, levels=2)
        self.assertEqual(len(levels), 2)
        level1_full = levels[1].get_chunk((slice(None), slice(None), slice(None)))
        self.assertEqual(level1_full.shape, (2, 2, 2))
        self.assertEqual(int(level1_full[0, 0, 0]), 13)


if __name__ == "__main__":
    unittest.main()
