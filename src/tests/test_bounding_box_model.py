from __future__ import annotations

import unittest

from src.bbox import BoundingBox


class BoundingBoxModelTests(unittest.TestCase):
    def test_default_label_is_train(self) -> None:
        box = BoundingBox.from_bounds(
            box_id="labeled",
            z0=1,
            z1=2,
            y0=3,
            y1=4,
            x0=5,
            x1=6,
        )
        self.assertEqual(box.label, "train")

    def test_invalid_label_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "label must be one of"):
            BoundingBox.from_bounds(
                box_id="invalid-label",
                z0=0,
                z1=1,
                y0=0,
                y1=1,
                x0=0,
                x1=1,
                label="dev",  # type: ignore[arg-type]
            )

    def test_from_voxel_corners_is_order_independent(self) -> None:
        box_a = BoundingBox.from_voxel_corners(
            box_id="box-a",
            p0=(5, 2, 9),
            p1=(1, 7, 3),
        )
        box_b = BoundingBox.from_voxel_corners(
            box_id="box-a",
            p0=(1, 7, 3),
            p1=(5, 2, 9),
        )
        self.assertEqual(box_a, box_b)
        self.assertEqual(box_a.as_tuple(), (1, 6, 2, 8, 3, 10))

    def test_from_voxel_corners_same_voxel_creates_one_voxel_box(self) -> None:
        box = BoundingBox.from_voxel_corners(
            box_id="single",
            p0=(4, 5, 6),
            p1=(4, 5, 6),
        )
        self.assertEqual(box.size_voxels, (1, 1, 1))
        self.assertTrue(box.contains_voxel((4, 5, 6)))
        self.assertFalse(box.contains_voxel((5, 5, 6)))

    def test_validation_rejects_invalid_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "z0 < z1"):
            BoundingBox.from_bounds(
                box_id="invalid-z",
                z0=3,
                z1=3,
                y0=0,
                y1=1,
                x0=0,
                x1=1,
            )
        with self.assertRaisesRegex(ValueError, "lower bounds must be >= 0"):
            BoundingBox.from_bounds(
                box_id="invalid-negative",
                z0=-1,
                z1=1,
                y0=0,
                y1=1,
                x0=0,
                x1=1,
            )
        with self.assertRaisesRegex(ValueError, "exceeds volume shape"):
            BoundingBox.from_bounds(
                box_id="invalid-shape",
                z0=0,
                z1=5,
                y0=0,
                y1=4,
                x0=0,
                x1=4,
                volume_shape=(4, 8, 8),
            )

    def test_size_and_center_are_derived_in_index_space(self) -> None:
        box = BoundingBox.from_bounds(
            box_id="derived",
            z0=2,
            z1=6,
            y0=4,
            y1=10,
            x0=1,
            x1=5,
        )
        self.assertEqual(box.size_voxels, (4, 6, 4))
        self.assertEqual(box.center_index_space, (3.5, 6.5, 2.5))

    def test_slice_intersections_and_projection(self) -> None:
        box = BoundingBox.from_bounds(
            box_id="proj",
            z0=2,
            z1=5,
            y0=4,
            y1=8,
            x0=10,
            x1=13,
        )
        self.assertTrue(box.intersects_slice(0, 2))
        self.assertFalse(box.intersects_slice(0, 5))
        self.assertEqual(box.slice_projection(0, 3), ((4, 8), (10, 13)))
        self.assertEqual(box.slice_projection(1, 7), ((2, 5), (10, 13)))
        self.assertEqual(box.slice_projection(2, 11), ((2, 5), (4, 8)))
        self.assertIsNone(box.slice_projection(2, 13))

    def test_move_preserves_size_and_clamps_to_volume(self) -> None:
        box = BoundingBox.from_bounds(
            box_id="move",
            z0=2,
            z1=5,
            y0=1,
            y1=4,
            x0=3,
            x1=7,
            volume_shape=(10, 10, 12),
        )
        moved = box.move(dz=20, dy=-20, dx=5, volume_shape=(10, 10, 12))
        self.assertEqual(moved.size_voxels, box.size_voxels)
        self.assertEqual(moved.as_tuple(), (7, 10, 0, 3, 8, 12))

    def test_move_face_clamps_and_keeps_positive_thickness(self) -> None:
        box = BoundingBox.from_bounds(
            box_id="face",
            z0=2,
            z1=6,
            y0=3,
            y1=8,
            x0=4,
            x1=9,
            volume_shape=(10, 12, 16),
        )
        moved_min = box.move_face("z_min", 100, volume_shape=(10, 12, 16))
        self.assertEqual((moved_min.z0, moved_min.z1), (5, 6))

        moved_max = box.move_face("x_max", 100, volume_shape=(10, 12, 16))
        self.assertEqual((moved_max.x0, moved_max.x1), (4, 16))

        moved_collapse = box.move_face("y_max", 0, volume_shape=(10, 12, 16))
        self.assertEqual((moved_collapse.y0, moved_collapse.y1), (3, 4))

    def test_corner_coordinate_and_corner_move(self) -> None:
        box = BoundingBox.from_bounds(
            box_id="corner",
            z0=2,
            z1=6,
            y0=3,
            y1=7,
            x0=4,
            x1=9,
            volume_shape=(10, 12, 16),
        )
        self.assertEqual(box.corner_coordinate("z_min_y_max_x_min"), (2, 7, 4))

        moved = box.move_corner(
            "z_min_y_max_x_max",
            (1, 11, 15),
            volume_shape=(10, 12, 16),
        )
        self.assertEqual(moved.as_tuple(), (1, 6, 3, 11, 4, 15))

        clamped = box.move_corner(
            "z_max_y_min_x_min",
            (0, -5, -7),
            volume_shape=(10, 12, 16),
        )
        self.assertEqual(clamped.as_tuple(), (2, 3, 0, 7, 0, 9))


if __name__ == "__main__":
    unittest.main()
