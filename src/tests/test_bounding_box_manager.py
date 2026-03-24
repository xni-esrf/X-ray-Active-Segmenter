from __future__ import annotations

import unittest

from src.bbox import BoundingBox, BoundingBoxChange, BoundingBoxManager


class BoundingBoxManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = BoundingBoxManager((12, 20, 30))

    def test_add_from_corners_assigns_auto_id_and_marks_dirty(self) -> None:
        box = self.manager.add_from_corners((1, 2, 3), (4, 5, 6))

        self.assertEqual(box.id, "bbox_0001")
        self.assertEqual(box.as_tuple(), (1, 5, 2, 6, 3, 7))
        self.assertEqual(self.manager.revision, 1)
        self.assertTrue(self.manager.dirty)
        self.assertEqual(self.manager.box_ids(), ("bbox_0001",))

    def test_add_rejects_duplicate_id(self) -> None:
        box = BoundingBox.from_bounds(
            box_id="custom_1",
            z0=1,
            z1=3,
            y0=2,
            y1=5,
            x0=4,
            x1=8,
            volume_shape=(12, 20, 30),
        )
        self.manager.add(box)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.manager.add(box)

    def test_move_face_updates_geometry_and_emits_event(self) -> None:
        events = []
        self.manager.on_changed(events.append)
        box = self.manager.add_from_corners((2, 2, 2), (5, 6, 7), box_id="box-a")
        updated = self.manager.move_face("box-a", "x_max", 20)

        self.assertEqual(updated.as_tuple(), (2, 6, 2, 7, 2, 20))
        self.assertEqual(len(events), 2)
        self.assertIsInstance(events[0], BoundingBoxChange)
        self.assertEqual(events[0].kind, "added")
        self.assertEqual(events[1].kind, "updated")
        self.assertEqual(events[1].box_id, "box-a")
        self.assertEqual(events[1].revision, 2)
        self.assertEqual(self.manager.get(box.id), updated)

    def test_select_does_not_change_revision_or_dirty(self) -> None:
        self.manager.add_from_corners((1, 1, 1), (2, 2, 2), box_id="box-a")
        self.manager.mark_clean()
        revision_before = self.manager.revision

        selected = self.manager.select("box-a")

        self.assertIsNotNone(selected)
        self.assertEqual(selected.id, "box-a")
        self.assertEqual(self.manager.selected_id, "box-a")
        self.assertEqual(self.manager.revision, revision_before)
        self.assertFalse(self.manager.dirty)

    def test_delete_and_clear_update_state(self) -> None:
        self.manager.add_from_corners((0, 0, 0), (1, 1, 1), box_id="a")
        self.manager.add_from_corners((2, 2, 2), (3, 3, 3), box_id="b")
        self.manager.select("b")

        deleted = self.manager.delete("b")
        self.assertTrue(deleted)
        self.assertIsNone(self.manager.selected_id)
        self.assertEqual(self.manager.box_ids(), ("a",))

        self.manager.clear()
        self.assertEqual(self.manager.boxes(), tuple())
        self.assertIsNone(self.manager.selected_id)

    def test_add_then_delete_restores_clean_state(self) -> None:
        self.assertFalse(self.manager.dirty)
        box = self.manager.add_from_corners((1, 1, 1), (2, 2, 2), box_id="a")
        self.assertTrue(self.manager.dirty)

        self.manager.delete(box.id)
        self.assertFalse(self.manager.dirty)

    def test_geometry_roundtrip_back_to_clean_state(self) -> None:
        box = self.manager.add_from_corners((1, 1, 1), (4, 4, 4), box_id="a")
        self.manager.mark_clean()
        self.assertFalse(self.manager.dirty)

        moved = self.manager.move_face(box.id, "x_max", 10)
        self.assertTrue(self.manager.dirty)
        restored = moved.move_face("x_max", box.x1, volume_shape=self.manager.volume_shape)
        self.manager.replace(box.id, restored)
        self.assertFalse(self.manager.dirty)

    def test_replace_all_validates_duplicates_and_can_mark_clean(self) -> None:
        a = BoundingBox.from_bounds(
            box_id="a",
            z0=1,
            z1=3,
            y0=1,
            y1=3,
            x0=1,
            x1=3,
            volume_shape=(12, 20, 30),
        )
        b = BoundingBox.from_bounds(
            box_id="b",
            z0=4,
            z1=8,
            y0=2,
            y1=6,
            x0=5,
            x1=9,
            volume_shape=(12, 20, 30),
        )
        boxes = self.manager.replace_all((a, b), selected_id="b", mark_clean=True)

        self.assertEqual(tuple(box.id for box in boxes), ("a", "b"))
        self.assertEqual(self.manager.selected_id, "b")
        self.assertFalse(self.manager.dirty)

        with self.assertRaisesRegex(ValueError, "Duplicate bounding box id"):
            self.manager.replace_all((a, a))

    def test_mark_clean_emits_cleaned_event(self) -> None:
        events = []
        self.manager.on_changed(events.append)
        self.manager.add_from_corners((1, 1, 1), (2, 2, 2), box_id="a")

        self.manager.mark_clean()

        self.assertEqual(len(events), 2)
        self.assertEqual(events[-1].kind, "cleaned")
        self.assertFalse(events[-1].dirty)
        self.assertFalse(self.manager.dirty)


if __name__ == "__main__":
    unittest.main()
