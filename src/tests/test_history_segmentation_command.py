from __future__ import annotations

import unittest

import numpy as np

from src.annotation import SegmentationEditor
from src.history import (
    GlobalHistoryManager,
    SegmentationHistoryCommand,
    estimate_segmentation_history_bytes,
)


class SegmentationHistoryCommandTests(unittest.TestCase):
    def test_command_undo_redo_roundtrip(self) -> None:
        editor = SegmentationEditor.create_empty((1, 1, 3), kind="semantic", dtype=np.uint8)
        editor.paint_voxel((0, 0, 1), label=9)
        operation_id = editor.latest_undo_operation_id()
        self.assertIsNotNone(operation_id)
        history_entry = editor.history[-1]

        command = SegmentationHistoryCommand(
            editor=editor,
            operation_id=int(operation_id),
            bytes_used=estimate_segmentation_history_bytes(editor, history_entry.operation),
        )

        command.undo()
        np.testing.assert_array_equal(editor.array_view(), np.zeros((1, 1, 3), dtype=np.uint8))

        command.redo()
        np.testing.assert_array_equal(
            editor.array_view(),
            np.array([[[0, 9, 0]]], dtype=np.uint8),
        )

    def test_command_rejects_out_of_sync_undo(self) -> None:
        editor = SegmentationEditor.create_empty((1, 1, 3), kind="semantic", dtype=np.uint8)
        editor.paint_voxel((0, 0, 0), label=1)
        first_operation_id = editor.latest_undo_operation_id()
        self.assertIsNotNone(first_operation_id)
        first_operation = editor.history[-1].operation
        editor.paint_voxel((0, 0, 1), label=2)
        command = SegmentationHistoryCommand(
            editor=editor,
            operation_id=int(first_operation_id),
            bytes_used=estimate_segmentation_history_bytes(editor, first_operation),
        )

        with self.assertRaisesRegex(RuntimeError, "out of sync"):
            command.undo()

    def test_estimated_bytes_follow_changed_voxels_and_dtype(self) -> None:
        editor_u8 = SegmentationEditor.create_empty((1, 1, 2), kind="semantic", dtype=np.uint8)
        op_u8 = editor_u8.paint_stroke([(0, 0, 0), (0, 0, 1)], label=7)
        # changed_voxels * (index bytes + value bytes) -> 2 * (8 + 1)
        self.assertEqual(estimate_segmentation_history_bytes(editor_u8, op_u8), 18)

        editor_u16 = SegmentationEditor.create_empty((1, 1, 2), kind="semantic", dtype=np.uint16)
        op_u16 = editor_u16.paint_stroke([(0, 0, 0), (0, 0, 1)], label=7)
        # changed_voxels * (index bytes + value bytes) -> 2 * (8 + 2)
        self.assertEqual(estimate_segmentation_history_bytes(editor_u16, op_u16), 20)

    def test_global_eviction_discards_evicted_segmentation_undo_entries(self) -> None:
        history = GlobalHistoryManager(max_history_entries=1, max_history_bytes=1024 * 1024)
        editor = SegmentationEditor.create_empty((1, 1, 3), kind="semantic", dtype=np.uint8)

        editor.paint_voxel((0, 0, 0), label=1)
        op1_id = editor.latest_undo_operation_id()
        self.assertIsNotNone(op1_id)
        op1 = editor.history[-1].operation
        history.push(
            SegmentationHistoryCommand(
                editor=editor,
                operation_id=int(op1_id),
                bytes_used=estimate_segmentation_history_bytes(editor, op1),
            )
        )
        self.assertEqual(editor.undo_depth(), 1)

        editor.paint_voxel((0, 0, 1), label=2)
        op2_id = editor.latest_undo_operation_id()
        self.assertIsNotNone(op2_id)
        op2 = editor.history[-1].operation
        history.push(
            SegmentationHistoryCommand(
                editor=editor,
                operation_id=int(op2_id),
                bytes_used=estimate_segmentation_history_bytes(editor, op2),
            )
        )

        # The first command is evicted by global cap and removed from editor undo stack.
        self.assertEqual(history.undo_depth(), 1)
        self.assertEqual(editor.undo_depth(), 1)
        self.assertEqual(editor.latest_undo_operation_id(), op2_id)

        history.undo()
        # Only op2 was undoable; op1 is no longer retained.
        self.assertEqual(int(editor.array_view()[0, 0, 0]), 1)
        self.assertEqual(int(editor.array_view()[0, 0, 1]), 0)

    def test_global_redo_clear_discards_segmentation_redo_entries(self) -> None:
        history = GlobalHistoryManager(max_history_entries=10, max_history_bytes=1024 * 1024)
        editor = SegmentationEditor.create_empty((1, 1, 3), kind="semantic", dtype=np.uint8)

        editor.paint_voxel((0, 0, 0), label=1)
        op1_id = editor.latest_undo_operation_id()
        self.assertIsNotNone(op1_id)
        op1 = editor.history[-1].operation
        history.push(
            SegmentationHistoryCommand(
                editor=editor,
                operation_id=int(op1_id),
                bytes_used=estimate_segmentation_history_bytes(editor, op1),
            )
        )

        history.undo()
        self.assertEqual(history.redo_depth(), 1)
        self.assertEqual(editor.redo_depth(), 1)

        editor.paint_voxel((0, 0, 1), label=2)
        op2_id = editor.latest_undo_operation_id()
        self.assertIsNotNone(op2_id)
        op2 = editor.history[-1].operation
        history.push(
            SegmentationHistoryCommand(
                editor=editor,
                operation_id=int(op2_id),
                bytes_used=estimate_segmentation_history_bytes(editor, op2),
            )
        )

        # Pushing new op clears redo globally and in editor.
        self.assertEqual(history.redo_depth(), 0)
        self.assertEqual(editor.redo_depth(), 0)


if __name__ == "__main__":
    unittest.main()
