from __future__ import annotations

import unittest

import numpy as np

from src.annotation import bbox_segmentation_ops as ops
from src.bbox import BoundingBox


class BBoxSegmentationOpsTests(unittest.TestCase):
    def _box(
        self,
        *,
        box_id: str = "bbox_0001",
        z0: int = 1,
        z1: int = 4,
        y0: int = 2,
        y1: int = 6,
        x0: int = 3,
        x1: int = 8,
        label: str = "train",
    ) -> BoundingBox:
        return BoundingBox.from_bounds(
            box_id=box_id,
            z0=z0,
            z1=z1,
            y0=y0,
            y1=y1,
            x0=x0,
            x1=x1,
            label=label,
            volume_shape=(20, 30, 40),
        )

    def test_build_selected_bbox_union_domain_returns_dataclass_and_union_mask(
        self,
    ) -> None:
        box1 = self._box(box_id="bbox_0001", z0=1, z1=3, y0=2, y1=4, x0=3, x1=5)
        box2 = self._box(box_id="bbox_0002", z0=4, z1=6, y0=5, y1=7, x0=6, x1=8)

        domain = ops.build_selected_bbox_union_domain((box1, box2))
        z_bounds, y_bounds, x_bounds, union_mask = domain

        self.assertIsInstance(domain, ops.BBoxUnionDomain)
        self.assertEqual(z_bounds, (1, 6))
        self.assertEqual(y_bounds, (2, 7))
        self.assertEqual(x_bounds, (3, 8))
        self.assertEqual(union_mask.shape, (5, 5, 5))

        expected = np.zeros((5, 5, 5), dtype=bool)
        expected[0:2, 0:2, 0:2] = True
        expected[3:5, 3:5, 3:5] = True
        self.assertTrue(np.array_equal(union_mask, expected))

    def test_build_selected_bbox_union_domain_merges_overlaps_once(self) -> None:
        box1 = self._box(box_id="bbox_0001", z0=1, z1=4, y0=1, y1=4, x0=1, x1=4)
        box2 = self._box(box_id="bbox_0002", z0=2, z1=5, y0=2, y1=5, x0=2, x1=5)

        domain = ops.build_selected_bbox_union_domain((box1, box2))

        self.assertEqual(domain.union_mask.shape, (4, 4, 4))
        self.assertEqual(int(np.count_nonzero(domain.union_mask)), 46)

    def test_build_selected_bbox_processing_regions_returns_dataclass_bounds(
        self,
    ) -> None:
        box1 = self._box(box_id="bbox_0001", z0=1, z1=3, y0=2, y1=4, x0=3, x1=5)
        box2 = self._box(box_id="bbox_0002", z0=4, z1=6, y0=5, y1=7, x0=6, x1=8)

        regions = ops.build_selected_bbox_processing_regions(
            (box1, box2),
            volume_shape=(20, 30, 40),
            halo_size=1,
        )

        self.assertIsInstance(regions, ops.BBoxProcessingRegions)
        self.assertEqual(regions.core_z_bounds, (1, 6))
        self.assertEqual(regions.core_y_bounds, (2, 7))
        self.assertEqual(regions.core_x_bounds, (3, 8))
        self.assertEqual(regions.union_mask.shape, (5, 5, 5))
        self.assertEqual(regions.extended_z_bounds, (0, 7))
        self.assertEqual(regions.extended_y_bounds, (1, 8))
        self.assertEqual(regions.extended_x_bounds, (2, 9))

    def test_build_selected_bbox_processing_regions_clamps_extended_bounds_to_volume_edges(
        self,
    ) -> None:
        box = self._box(
            box_id="bbox_0001",
            z0=0,
            z1=2,
            y0=28,
            y1=30,
            x0=39,
            x1=40,
        )

        regions = ops.build_selected_bbox_processing_regions(
            (box,),
            volume_shape=(20, 30, 40),
            halo_size=1,
        )

        self.assertEqual(regions.core_z_bounds, (0, 2))
        self.assertEqual(regions.core_y_bounds, (28, 30))
        self.assertEqual(regions.core_x_bounds, (39, 40))
        self.assertEqual(regions.extended_z_bounds, (0, 3))
        self.assertEqual(regions.extended_y_bounds, (27, 30))
        self.assertEqual(regions.extended_x_bounds, (38, 40))

    def test_build_extended_foreground_with_halo_padding_uses_nearest_outside_core(
        self,
    ) -> None:
        volume = np.zeros((8, 8, 8), dtype=np.uint16)
        volume[2, 3, 3] = 1
        volume[4, 3, 3] = 1
        volume[3, 2, 3] = 1
        volume[3, 3, 2] = 1

        expanded = ops.build_extended_foreground_with_halo_padding(
            segmentation_volume=volume,
            core_z_bounds=(2, 5),
            core_y_bounds=(2, 5),
            core_x_bounds=(2, 5),
            halo_size=1,
        )

        expected_core = np.asarray(volume[2:5, 2:5, 2:5] != 0, dtype=bool)
        self.assertEqual(expanded.shape, (5, 5, 5))
        self.assertTrue(np.array_equal(expanded[1:4, 1:4, 1:4], expected_core))
        self.assertTrue(bool(expanded[0, 2, 2]))
        self.assertTrue(bool(expanded[4, 2, 2]))
        self.assertTrue(bool(expanded[2, 0, 2]))
        self.assertTrue(bool(expanded[2, 2, 0]))

    def test_build_extended_foreground_with_halo_padding_reflects_at_global_volume_edges(
        self,
    ) -> None:
        volume = np.zeros((5, 5, 5), dtype=np.uint16)
        volume[1, 2, 3] = 1

        expanded = ops.build_extended_foreground_with_halo_padding(
            segmentation_volume=volume,
            core_z_bounds=(0, 3),
            core_y_bounds=(1, 4),
            core_x_bounds=(2, 5),
            halo_size=1,
        )

        self.assertTrue(bool(expanded[0, 2, 2]))
        self.assertTrue(bool(expanded[2, 2, 4]))

    def test_count_true_neighbors_3x3x3_counts_center_and_corner(self) -> None:
        mask = np.zeros((3, 3, 3), dtype=bool)
        mask[:, :, :] = True

        counts = ops.count_true_neighbors_3x3x3(mask)

        self.assertEqual(counts.shape, (3, 3, 3))
        self.assertEqual(int(counts[1, 1, 1]), 27)
        self.assertEqual(int(counts[0, 0, 0]), 8)

    def test_compute_selected_bbox_binary_operation_dilation_uses_full_cube(
        self,
    ) -> None:
        foreground = np.zeros((5, 5, 5), dtype=bool)
        foreground[2, 2, 2] = True
        union = np.ones((5, 5, 5), dtype=bool)

        dilated = ops.compute_selected_bbox_binary_operation(
            operation="dilation",
            foreground_mask=foreground,
            union_mask=union,
        )

        expected = np.zeros((5, 5, 5), dtype=bool)
        expected[1:4, 1:4, 1:4] = True
        self.assertTrue(np.array_equal(dilated, expected))

    def test_compute_selected_bbox_binary_operation_erosion_shrinks_by_one_voxel(
        self,
    ) -> None:
        foreground = np.ones((5, 5, 5), dtype=bool)
        union = np.ones((5, 5, 5), dtype=bool)

        eroded = ops.compute_selected_bbox_binary_operation(
            operation="erosion",
            foreground_mask=foreground,
            union_mask=union,
        )

        expected = np.zeros((5, 5, 5), dtype=bool)
        expected[1:4, 1:4, 1:4] = True
        self.assertTrue(np.array_equal(eroded, expected))

    def test_compute_selected_bbox_binary_operation_median_uses_14_of_27_threshold(
        self,
    ) -> None:
        foreground = np.zeros((3, 3, 3), dtype=bool)
        foreground.reshape(-1)[:13] = True
        union = np.ones((3, 3, 3), dtype=bool)

        median13 = ops.compute_selected_bbox_binary_operation(
            operation="median_filter",
            foreground_mask=foreground,
            union_mask=union,
        )

        foreground.reshape(-1)[13] = True
        median14 = ops.compute_selected_bbox_binary_operation(
            operation="median_filter",
            foreground_mask=foreground,
            union_mask=union,
        )

        self.assertFalse(bool(median13[1, 1, 1]))
        self.assertTrue(bool(median14[1, 1, 1]))

    def test_compute_selected_bbox_binary_operation_is_constrained_to_union_mask(
        self,
    ) -> None:
        foreground = np.zeros((5, 5, 5), dtype=bool)
        foreground[2, 2, 2] = True
        union = np.zeros((5, 5, 5), dtype=bool)
        union[2, 2, 2] = True

        dilated = ops.compute_selected_bbox_binary_operation(
            operation="dilation",
            foreground_mask=foreground,
            union_mask=union,
        )

        expected = np.zeros((5, 5, 5), dtype=bool)
        expected[2, 2, 2] = True
        self.assertTrue(np.array_equal(dilated, expected))

    def test_compute_selected_bbox_binary_operation_with_halo_context_matches_extended_crop(
        self,
    ) -> None:
        volume = np.zeros((8, 8, 8), dtype=np.uint16)
        volume[3, 3, 3] = 1
        volume[3, 3, 4] = 1

        result = ops.compute_selected_bbox_binary_operation_with_halo_context(
            operation="dilation",
            segmentation_volume=volume,
            core_z_bounds=(2, 5),
            core_y_bounds=(2, 5),
            core_x_bounds=(2, 5),
            halo_size=1,
        )

        extended = ops.build_extended_foreground_with_halo_padding(
            segmentation_volume=volume,
            core_z_bounds=(2, 5),
            core_y_bounds=(2, 5),
            core_x_bounds=(2, 5),
            halo_size=1,
        )
        transformed_extended = ops.compute_selected_bbox_binary_operation(
            operation="dilation",
            foreground_mask=extended,
            union_mask=np.ones_like(extended, dtype=bool),
        )
        expected = np.asarray(transformed_extended[1:4, 1:4, 1:4], dtype=bool)
        self.assertTrue(np.array_equal(result, expected))

    def test_compute_selected_bbox_binary_operation_with_halo_context_erosion_keeps_uniform_core(
        self,
    ) -> None:
        volume = np.zeros((12, 12, 12), dtype=np.uint16)
        volume[4:9, 4:9, 4:9] = 1

        eroded = ops.compute_selected_bbox_binary_operation_with_halo_context(
            operation="erosion",
            segmentation_volume=volume,
            core_z_bounds=(4, 9),
            core_y_bounds=(4, 9),
            core_x_bounds=(4, 9),
            halo_size=1,
        )

        self.assertTrue(np.array_equal(eroded, np.ones((5, 5, 5), dtype=bool)))

    def test_compute_selected_bbox_binary_operation_with_halo_context_median_treats_border_like_interior(
        self,
    ) -> None:
        volume = np.zeros((12, 12, 12), dtype=np.uint16)
        volume[4:9, 4:9, 4:9] = 1
        volume[6, 6, 4] = 0
        volume[6, 6, 6] = 0

        filtered = ops.compute_selected_bbox_binary_operation_with_halo_context(
            operation="median_filter",
            segmentation_volume=volume,
            core_z_bounds=(4, 9),
            core_y_bounds=(4, 9),
            core_x_bounds=(4, 9),
            halo_size=1,
        )

        self.assertTrue(bool(filtered[2, 2, 0]))
        self.assertTrue(bool(filtered[2, 2, 2]))

    def test_compute_selected_bbox_binary_operation_with_halo_context_accepts_core_shaped_result(
        self,
    ) -> None:
        volume = np.zeros((8, 8, 8), dtype=np.uint16)
        mocked_core = np.zeros((3, 3, 3), dtype=bool)
        mocked_core[1, 1, 1] = True

        result = ops.compute_selected_bbox_binary_operation_with_halo_context(
            operation="median_filter",
            segmentation_volume=volume,
            core_z_bounds=(2, 5),
            core_y_bounds=(2, 5),
            core_x_bounds=(2, 5),
            halo_size=1,
            binary_operation_func=lambda **_kwargs: mocked_core,
        )

        self.assertTrue(np.array_equal(result, mocked_core))

    def test_compute_selected_bbox_binary_operation_ignores_foreground_outside_union(
        self,
    ) -> None:
        foreground = np.zeros((5, 5, 5), dtype=bool)
        foreground[2, 2, 1] = True
        union = np.zeros((5, 5, 5), dtype=bool)
        union[2, 2, 2] = True

        dilated = ops.compute_selected_bbox_binary_operation(
            operation="dilation",
            foreground_mask=foreground,
            union_mask=union,
        )

        self.assertFalse(bool(dilated[2, 2, 2]))

    def test_compute_selected_bbox_binary_operation_rejects_shape_mismatch(
        self,
    ) -> None:
        foreground = np.zeros((3, 3, 3), dtype=bool)
        union = np.zeros((2, 2, 2), dtype=bool)

        with self.assertRaises(ValueError):
            ops.compute_selected_bbox_binary_operation(
                operation="dilation",
                foreground_mask=foreground,
                union_mask=union,
            )

    def test_count_true_neighbors_3x3x3_rejects_non_3d_input(self) -> None:
        with self.assertRaises(ValueError):
            ops.count_true_neighbors_3x3x3(np.zeros((3, 3), dtype=bool))

    def test_mask_to_absolute_coordinates_applies_origin_offset(self) -> None:
        mask = np.zeros((3, 3, 3), dtype=bool)
        mask[0, 1, 2] = True
        mask[2, 0, 0] = True

        coordinates = ops.mask_to_absolute_coordinates(mask, origin=(10, 20, 30))

        expected = np.asarray([[10, 21, 32], [12, 20, 30]], dtype=np.int64)
        np.testing.assert_array_equal(coordinates, expected)

    def test_mask_to_absolute_coordinates_returns_empty_2d_array_for_empty_mask(
        self,
    ) -> None:
        mask = np.zeros((2, 2, 2), dtype=bool)

        coordinates = ops.mask_to_absolute_coordinates(mask, origin=(5, 6, 7))

        self.assertEqual(coordinates.shape, (0, 3))
        self.assertEqual(coordinates.dtype, np.int64)

    def test_bbox_segmentation_operation_display_name(self) -> None:
        self.assertEqual(
            ops.bbox_segmentation_operation_display_name("median_filter"),
            "Median Filter Selected",
        )
        self.assertEqual(
            ops.bbox_segmentation_operation_display_name("erosion"),
            "Erosion Selected",
        )
        self.assertEqual(
            ops.bbox_segmentation_operation_display_name("dilation"),
            "Dilation Selected",
        )

    def test_bbox_segmentation_operation_display_name_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            ops.bbox_segmentation_operation_display_name("unknown")  # type: ignore[arg-type]

    def test_compute_set_mask_labels_uses_majority_and_breaks_ties_with_smallest_label(
        self,
    ) -> None:
        segmentation_roi = np.zeros((3, 3, 3), dtype=np.int32)
        segmentation_roi[1, 1, 0] = 4
        segmentation_roi[1, 0, 1] = 4
        segmentation_roi[0, 1, 1] = 2
        segmentation_roi[2, 1, 1] = 2
        set_mask = np.zeros((3, 3, 3), dtype=bool)
        set_mask[1, 1, 1] = True
        union_mask = np.ones((3, 3, 3), dtype=bool)

        labels = ops.compute_set_mask_labels(
            segmentation_roi=segmentation_roi,
            set_mask=set_mask,
            union_mask=union_mask,
            fallback_label=9,
        )

        np.testing.assert_array_equal(labels, np.asarray([2], dtype=np.int64))

    def test_compute_set_mask_labels_falls_back_when_no_neighbor_label_exists(
        self,
    ) -> None:
        segmentation_roi = np.zeros((3, 3, 3), dtype=np.int32)
        set_mask = np.zeros((3, 3, 3), dtype=bool)
        set_mask[1, 1, 1] = True
        union_mask = np.ones((3, 3, 3), dtype=bool)

        labels = ops.compute_set_mask_labels(
            segmentation_roi=segmentation_roi,
            set_mask=set_mask,
            union_mask=union_mask,
            fallback_label=7,
        )

        np.testing.assert_array_equal(labels, np.asarray([7], dtype=np.int64))

    def test_compute_set_mask_labels_ignores_nonzero_labels_outside_union_domain(
        self,
    ) -> None:
        segmentation_roi = np.zeros((3, 3, 3), dtype=np.int32)
        segmentation_roi[1, 1, 2] = 5
        set_mask = np.zeros((3, 3, 3), dtype=bool)
        set_mask[1, 1, 1] = True
        union_mask = np.ones((3, 3, 3), dtype=bool)
        union_mask[1, 1, 2] = False

        labels = ops.compute_set_mask_labels(
            segmentation_roi=segmentation_roi,
            set_mask=set_mask,
            union_mask=union_mask,
            fallback_label=7,
        )

        np.testing.assert_array_equal(labels, np.asarray([7], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
