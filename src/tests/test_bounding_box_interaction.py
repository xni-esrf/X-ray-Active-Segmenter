from __future__ import annotations

import unittest
from typing import cast

from src.bbox import (
    BoundingBoxHandleHit,
    ProjectedBoundingBox2D,
    face_updates_for_handle_drag,
    hit_test_projected_box_handles,
    translation_delta_for_edge_drag,
)


class BoundingBoxInteractionTests(unittest.TestCase):
    def test_hit_test_prefers_corner_over_edge(self) -> None:
        box = ProjectedBoundingBox2D(
            box_id="bbox_0001",
            row0=10,
            row1=20,
            col0=30,
            col1=40,
        )

        hit = hit_test_projected_box_handles((box,), row=10.0, col=30.0, tolerance=1.0)

        self.assertIsNotNone(hit)
        hit = cast(BoundingBoxHandleHit, hit)
        self.assertEqual(hit.kind, "corner")
        self.assertEqual(hit.handle, "top_left")
        self.assertEqual(hit.box_id, "bbox_0001")

    def test_hit_test_detects_edge_when_not_close_to_corner(self) -> None:
        box = ProjectedBoundingBox2D(
            box_id="bbox_0001",
            row0=10,
            row1=20,
            col0=30,
            col1=40,
        )

        hit = hit_test_projected_box_handles((box,), row=10.2, col=35.0, tolerance=0.5)

        self.assertIsNotNone(hit)
        hit = cast(BoundingBoxHandleHit, hit)
        self.assertEqual(hit.kind, "edge")
        self.assertEqual(hit.handle, "top")

    def test_hit_test_returns_none_when_far_from_handles(self) -> None:
        box = ProjectedBoundingBox2D(
            box_id="bbox_0001",
            row0=10,
            row1=20,
            col0=30,
            col1=40,
        )

        hit = hit_test_projected_box_handles((box,), row=15.0, col=35.0, tolerance=1.0)

        self.assertIsNone(hit)

    def test_hit_test_uses_selected_id_as_tie_breaker(self) -> None:
        first = ProjectedBoundingBox2D(
            box_id="first",
            row0=10,
            row1=20,
            col0=30,
            col1=40,
        )
        second = ProjectedBoundingBox2D(
            box_id="second",
            row0=10,
            row1=20,
            col0=30,
            col1=40,
        )

        hit = hit_test_projected_box_handles(
            (first, second),
            row=10.0,
            col=30.0,
            tolerance=1.0,
            selected_id="first",
        )

        self.assertIsNotNone(hit)
        hit = cast(BoundingBoxHandleHit, hit)
        self.assertEqual(hit.box_id, "first")

    def test_edge_translation_delta_maps_for_each_axis(self) -> None:
        edge_hit = BoundingBoxHandleHit(
            box_id="bbox_0001",
            kind="edge",
            handle="right",
        )
        self.assertEqual(
            translation_delta_for_edge_drag(
                edge_hit,
                axis=0,
                row_delta=5,
                col_delta=11,
            ),
            (0, 5, 11),
        )
        self.assertEqual(
            translation_delta_for_edge_drag(
                edge_hit,
                axis=1,
                row_delta=5,
                col_delta=11,
            ),
            (5, 0, 11),
        )
        self.assertEqual(
            translation_delta_for_edge_drag(
                edge_hit,
                axis=2,
                row_delta=5,
                col_delta=11,
            ),
            (5, 11, 0),
        )

    def test_face_updates_map_corner_to_row_and_col_faces(self) -> None:
        corner_hit = BoundingBoxHandleHit(
            box_id="bbox_0001",
            kind="corner",
            handle="bottom_left",
        )

        updates = face_updates_for_handle_drag(
            corner_hit,
            axis=2,
            row_boundary=17,
            col_boundary=9,
        )

        self.assertEqual(updates, (("z_max", 17), ("y_min", 9)))

    def test_face_updates_reject_edge_handles(self) -> None:
        with self.assertRaisesRegex(ValueError, "Edge handles do not produce face updates"):
            face_updates_for_handle_drag(
                BoundingBoxHandleHit(
                    box_id="bbox_0001",
                    kind="edge",
                    handle="right",
                ),
                axis=0,
                row_boundary=0,
                col_boundary=0,
            )

    def test_edge_translation_rejects_corner_handles(self) -> None:
        with self.assertRaisesRegex(ValueError, "Only edge handles can translate"):
            translation_delta_for_edge_drag(
                BoundingBoxHandleHit(
                    box_id="bbox_0001",
                    kind="corner",
                    handle="top_left",
                ),
                axis=0,
                row_delta=1,
                col_delta=1,
            )


if __name__ == "__main__":
    unittest.main()
