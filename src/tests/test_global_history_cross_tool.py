from __future__ import annotations

import unittest

import numpy as np

from src.annotation import SegmentationEditor
from src.history import (
    BoundingBoxAddCommand,
    BoundingBoxUpdateCommand,
    GlobalHistoryManager,
    SegmentationHistoryCommand,
    estimate_bounding_box_history_bytes,
    estimate_segmentation_history_bytes,
)
from src.bbox import BoundingBoxManager


class GlobalHistoryCrossToolTests(unittest.TestCase):
    def _segmentation_command_for_latest_operation(
        self,
        editor: SegmentationEditor,
    ) -> SegmentationHistoryCommand:
        operation_id = editor.latest_undo_operation_id()
        self.assertIsNotNone(operation_id)
        operation = editor.history[-1].operation
        return SegmentationHistoryCommand(
            editor=editor,
            operation_id=int(operation_id),
            bytes_used=estimate_segmentation_history_bytes(editor, operation),
        )

    def test_chronological_undo_redo_across_segmentation_and_bbox(self) -> None:
        history = GlobalHistoryManager(max_history_entries=10, max_history_bytes=1024 * 1024)
        editor = SegmentationEditor.create_empty((1, 1, 4), kind="semantic", dtype=np.uint8)
        manager = BoundingBoxManager((10, 10, 10))

        editor.paint_voxel((0, 0, 0), label=7)
        history.push(self._segmentation_command_for_latest_operation(editor))

        before_selected = manager.selected_id
        box = manager.add_from_corners((1, 1, 1), (3, 3, 3), select=True)
        history.push(
            BoundingBoxAddCommand(
                manager=manager,
                box=box,
                before_selected_id=before_selected,
                after_selected_id=manager.selected_id,
                bytes_used=estimate_bounding_box_history_bytes(after_box=box),
            )
        )

        editor.paint_voxel((0, 0, 1), label=9)
        history.push(self._segmentation_command_for_latest_operation(editor))
        self.assertEqual(history.undo_depth(), 3)

        history.undo()
        self.assertEqual(int(editor.array_view()[0, 0, 1]), 0)
        self.assertIsNotNone(manager.get(box.id))

        history.undo()
        self.assertIsNone(manager.get(box.id))

        history.undo()
        self.assertEqual(int(editor.array_view()[0, 0, 0]), 0)

        history.redo()
        self.assertEqual(int(editor.array_view()[0, 0, 0]), 7)

        history.redo()
        self.assertIsNotNone(manager.get(box.id))

        history.redo()
        self.assertEqual(int(editor.array_view()[0, 0, 1]), 9)

    def test_new_operation_clears_redo_across_tool_boundaries(self) -> None:
        history = GlobalHistoryManager(max_history_entries=10, max_history_bytes=1024 * 1024)
        editor = SegmentationEditor.create_empty((1, 1, 4), kind="semantic", dtype=np.uint8)
        manager = BoundingBoxManager((10, 10, 10))

        editor.paint_voxel((0, 0, 0), label=1)
        history.push(self._segmentation_command_for_latest_operation(editor))

        before_selected = manager.selected_id
        box = manager.add_from_corners((1, 1, 1), (2, 2, 2), select=True)
        history.push(
            BoundingBoxAddCommand(
                manager=manager,
                box=box,
                before_selected_id=before_selected,
                after_selected_id=manager.selected_id,
                bytes_used=estimate_bounding_box_history_bytes(after_box=box),
            )
        )

        history.undo()
        self.assertEqual(history.redo_depth(), 1)
        self.assertIsNone(manager.get(box.id))

        editor.paint_voxel((0, 0, 1), label=2)
        history.push(self._segmentation_command_for_latest_operation(editor))
        self.assertEqual(history.redo_depth(), 0)
        self.assertIsNone(history.redo())

    def test_bbox_drag_transaction_is_single_undo_step(self) -> None:
        history = GlobalHistoryManager(max_history_entries=10, max_history_bytes=1024 * 1024)
        manager = BoundingBoxManager((20, 20, 20))
        box = manager.add_from_corners((1, 1, 1), (4, 4, 4), box_id="bbox_0001", select=True)

        history.begin_transaction("bbox_drag")
        before_first = manager.get(box.id)
        self.assertIsNotNone(before_first)
        if before_first is None:
            raise AssertionError("Expected existing bounding box before first drag update")
        first = manager.move_face(box.id, "x_max", 10)
        history.push(
            BoundingBoxUpdateCommand(
                manager=manager,
                before_box=before_first,
                after_box=first,
                before_selected_id=manager.selected_id,
                after_selected_id=manager.selected_id,
                bytes_used=estimate_bounding_box_history_bytes(
                    before_box=before_first,
                    after_box=first,
                ),
            )
        )
        before_second = manager.get(box.id)
        self.assertIsNotNone(before_second)
        if before_second is None:
            raise AssertionError("Expected existing bounding box before second drag update")
        second = manager.move_face(box.id, "y_max", 12)
        history.push(
            BoundingBoxUpdateCommand(
                manager=manager,
                before_box=before_second,
                after_box=second,
                before_selected_id=manager.selected_id,
                after_selected_id=manager.selected_id,
                bytes_used=estimate_bounding_box_history_bytes(
                    before_box=before_second,
                    after_box=second,
                ),
            )
        )

        stored = history.commit_transaction()
        self.assertTrue(stored)
        self.assertEqual(history.undo_depth(), 1)

        history.undo()
        self.assertEqual(manager.get(box.id), box)

        history.redo()
        self.assertEqual(manager.get(box.id), second)

    def test_committing_open_transaction_before_undo_avoids_undo_runtime_error(self) -> None:
        history = GlobalHistoryManager(max_history_entries=10, max_history_bytes=1024 * 1024)
        manager = BoundingBoxManager((20, 20, 20))
        box = manager.add_from_corners((1, 1, 1), (4, 4, 4), box_id="bbox_0001", select=True)

        history.begin_transaction("bbox_drag")
        before = manager.get(box.id)
        self.assertIsNotNone(before)
        if before is None:
            raise AssertionError("Expected existing bounding box before drag update")
        after = manager.move_face(box.id, "x_max", 10)
        history.push(
            BoundingBoxUpdateCommand(
                manager=manager,
                before_box=before,
                after_box=after,
                before_selected_id=manager.selected_id,
                after_selected_id=manager.selected_id,
                bytes_used=estimate_bounding_box_history_bytes(before_box=before, after_box=after),
            )
        )

        with self.assertRaisesRegex(RuntimeError, "transaction is active"):
            history.undo()

        # Step-6 behavior in MainWindow finalizes active drag transactions before undo/redo.
        history.commit_transaction()
        undone = history.undo()
        self.assertIsNotNone(undone)
        self.assertEqual(manager.get(box.id), box)


if __name__ == "__main__":
    unittest.main()
