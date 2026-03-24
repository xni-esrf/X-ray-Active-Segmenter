from __future__ import annotations

import unittest

from src.bbox import BoundingBox, BoundingBoxManager
from src.history import (
    BoundingBoxAddCommand,
    BoundingBoxDeleteCommand,
    BoundingBoxUpdateCommand,
    estimate_bounding_box_history_bytes,
)


class BoundingBoxHistoryCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = BoundingBoxManager((20, 30, 40))

    def test_add_command_undo_redo_roundtrip(self) -> None:
        box = BoundingBox.from_bounds(
            box_id="bbox_0001",
            z0=1,
            z1=4,
            y0=2,
            y1=6,
            x0=3,
            x1=8,
            volume_shape=(20, 30, 40),
        )
        command = BoundingBoxAddCommand(
            manager=self.manager,
            box=box,
            before_selected_id=None,
            after_selected_id=box.id,
            bytes_used=estimate_bounding_box_history_bytes(after_box=box),
        )

        command.redo()
        self.assertEqual(self.manager.box_ids(), (box.id,))
        self.assertEqual(self.manager.selected_id, box.id)

        command.undo()
        self.assertEqual(self.manager.box_ids(), tuple())
        self.assertIsNone(self.manager.selected_id)

    def test_delete_command_undo_redo_roundtrip(self) -> None:
        box = self.manager.add_from_corners((1, 2, 3), (4, 5, 6), box_id="bbox_0002", select=True)
        command = BoundingBoxDeleteCommand(
            manager=self.manager,
            box=box,
            before_selected_id=box.id,
            after_selected_id=None,
            bytes_used=estimate_bounding_box_history_bytes(before_box=box),
        )

        command.redo()
        self.assertEqual(self.manager.box_ids(), tuple())
        self.assertIsNone(self.manager.selected_id)

        command.undo()
        self.assertEqual(self.manager.box_ids(), (box.id,))
        self.assertEqual(self.manager.selected_id, box.id)

    def test_update_command_restores_geometry_and_selection(self) -> None:
        first = self.manager.add_from_corners((1, 2, 3), (4, 5, 6), box_id="bbox_0003", select=False)
        second = self.manager.add_from_corners((6, 7, 8), (8, 9, 10), box_id="bbox_0004", select=True)
        self.assertEqual(self.manager.selected_id, second.id)
        updated = first.move_face("x_max", 15, volume_shape=self.manager.volume_shape)

        command = BoundingBoxUpdateCommand(
            manager=self.manager,
            before_box=first,
            after_box=updated,
            before_selected_id=second.id,
            after_selected_id=second.id,
            bytes_used=estimate_bounding_box_history_bytes(before_box=first, after_box=updated),
        )

        command.redo()
        self.assertEqual(self.manager.get(first.id), updated)
        self.assertEqual(self.manager.selected_id, second.id)

        command.undo()
        self.assertEqual(self.manager.get(first.id), first)
        self.assertEqual(self.manager.selected_id, second.id)

    def test_add_command_undo_fails_when_box_missing(self) -> None:
        box = BoundingBox.from_bounds(
            box_id="bbox_0005",
            z0=1,
            z1=2,
            y0=1,
            y1=2,
            x0=1,
            x1=2,
            volume_shape=(20, 30, 40),
        )
        command = BoundingBoxAddCommand(
            manager=self.manager,
            box=box,
            before_selected_id=None,
            after_selected_id=box.id,
            bytes_used=estimate_bounding_box_history_bytes(after_box=box),
        )

        with self.assertRaisesRegex(RuntimeError, "missing"):
            command.undo()


if __name__ == "__main__":
    unittest.main()
