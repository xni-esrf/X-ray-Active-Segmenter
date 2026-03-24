from __future__ import annotations

import unittest

from src.bbox import BoundingBox, project_box_to_slice, project_boxes_to_slice


class BoundingBoxProjectionTests(unittest.TestCase):
    def test_projection_for_each_axis(self) -> None:
        box = BoundingBox.from_bounds(
            box_id="b1",
            z0=2,
            z1=6,
            y0=4,
            y1=9,
            x0=10,
            x1=15,
            volume_shape=(20, 20, 20),
        )

        axial = project_box_to_slice(
            box,
            axis=0,
            slice_index=3,
            level_scale=1,
            image_shape=(20, 20),
        )
        coronal = project_box_to_slice(
            box,
            axis=1,
            slice_index=7,
            level_scale=1,
            image_shape=(20, 20),
        )
        sagittal = project_box_to_slice(
            box,
            axis=2,
            slice_index=12,
            level_scale=1,
            image_shape=(20, 20),
        )

        self.assertIsNotNone(axial)
        self.assertEqual((axial.row0, axial.row1, axial.col0, axial.col1), (4, 9, 10, 15))
        self.assertIsNotNone(coronal)
        self.assertEqual((coronal.row0, coronal.row1, coronal.col0, coronal.col1), (2, 6, 10, 15))
        self.assertIsNotNone(sagittal)
        self.assertEqual((sagittal.row0, sagittal.row1, sagittal.col0, sagittal.col1), (2, 6, 4, 9))

    def test_projection_preserves_box_label(self) -> None:
        box = BoundingBox.from_bounds(
            box_id="labeled",
            z0=1,
            z1=3,
            y0=2,
            y1=5,
            x0=4,
            x1=8,
            label="validation",
            volume_shape=(10, 10, 10),
        )

        projected = project_box_to_slice(
            box,
            axis=0,
            slice_index=1,
            level_scale=1,
            image_shape=(10, 10),
        )

        self.assertIsNotNone(projected)
        self.assertEqual(projected.label, "validation")

    def test_projection_rejects_non_intersecting_slice(self) -> None:
        box = BoundingBox.from_bounds(
            box_id="b1",
            z0=2,
            z1=6,
            y0=4,
            y1=9,
            x0=10,
            x1=15,
            volume_shape=(20, 20, 20),
        )
        self.assertIsNone(project_box_to_slice(box, axis=0, slice_index=6, level_scale=1))
        self.assertIsNone(project_box_to_slice(box, axis=1, slice_index=3, level_scale=1))
        self.assertIsNone(project_box_to_slice(box, axis=2, slice_index=15, level_scale=1))

    def test_projection_is_level_scale_aware(self) -> None:
        box = BoundingBox.from_bounds(
            box_id="b1",
            z0=1,
            z1=5,
            y0=2,
            y1=7,
            x0=3,
            x1=8,
            volume_shape=(20, 20, 20),
        )
        projected = project_box_to_slice(
            box,
            axis=0,
            slice_index=4,
            level_scale=2,
            image_shape=(10, 10),
        )
        self.assertIsNotNone(projected)
        # y:[2,7)->[1,4), x:[3,8)->[1,4) at level scale 2
        self.assertEqual((projected.row0, projected.row1, projected.col0, projected.col1), (1, 4, 1, 4))

    def test_projection_clips_to_image_shape(self) -> None:
        box = BoundingBox.from_bounds(
            box_id="b1",
            z0=0,
            z1=8,
            y0=0,
            y1=8,
            x0=0,
            x1=8,
            volume_shape=(20, 20, 20),
        )
        projected = project_box_to_slice(
            box,
            axis=0,
            slice_index=2,
            level_scale=1,
            image_shape=(5, 4),
        )
        self.assertIsNotNone(projected)
        self.assertEqual((projected.row0, projected.row1, projected.col0, projected.col1), (0, 5, 0, 4))

    def test_projection_returns_none_when_clipping_collapses_box(self) -> None:
        box = BoundingBox.from_bounds(
            box_id="b1",
            z0=0,
            z1=8,
            y0=7,
            y1=9,
            x0=9,
            x1=10,
            volume_shape=(20, 20, 20),
        )
        projected = project_box_to_slice(
            box,
            axis=0,
            slice_index=2,
            level_scale=1,
            image_shape=(5, 5),
        )
        self.assertIsNone(projected)

    def test_project_boxes_to_slice_preserves_order_for_visible_boxes(self) -> None:
        boxes = (
            BoundingBox.from_bounds(
                box_id="first",
                z0=1,
                z1=4,
                y0=1,
                y1=3,
                x0=1,
                x1=3,
                volume_shape=(10, 10, 10),
            ),
            BoundingBox.from_bounds(
                box_id="second",
                z0=5,
                z1=7,
                y0=1,
                y1=3,
                x0=1,
                x1=3,
                volume_shape=(10, 10, 10),
            ),
            BoundingBox.from_bounds(
                box_id="third",
                z0=2,
                z1=6,
                y0=2,
                y1=6,
                x0=2,
                x1=6,
                volume_shape=(10, 10, 10),
            ),
        )

        projected = project_boxes_to_slice(
            boxes,
            axis=0,
            slice_index=2,
            level_scale=1,
            image_shape=(10, 10),
        )
        self.assertEqual(tuple(item.box_id for item in projected), ("first", "third"))


if __name__ == "__main__":
    unittest.main()
