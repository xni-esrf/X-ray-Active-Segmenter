from __future__ import annotations

import unittest

import numpy as np

from src.learning.large_crop_inference_plan import (
    LargeCropInferencePlan,
    build_large_crop_inference_plan,
)


def _slice_tuple_to_bounds(axis_slice: slice) -> tuple[int, int]:
    return (int(axis_slice.start), int(axis_slice.stop))


def _axis_valid_bounds(plan: LargeCropInferencePlan, axis: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        _slice_tuple_to_bounds(window.valid_slices[axis])
        for window in plan.windows
        if all(
            int(window.grid_index[other_axis]) == 0
            for other_axis in range(3)
            if other_axis != axis
        )
    )


def _coverage_for_slices(
    shape: tuple[int, int, int],
    slices_by_window: tuple[tuple[slice, slice, slice], ...],
) -> np.ndarray:
    coverage = np.zeros(shape, dtype=np.uint8)
    for z_slice, y_slice, x_slice in slices_by_window:
        coverage[z_slice, y_slice, x_slice] += 1
    return coverage


class LargeCropInferencePlanTests(unittest.TestCase):
    def test_long_thin_bbox_below_voxel_budget_is_single_crop(self) -> None:
        plan = build_large_crop_inference_plan(
            requested_bounds=((0, 5000), (0, 1000), (0, 1000)),
            raw_volume_shape=(5000, 1000, 1000),
            context_margin=0,
        )

        self.assertFalse(plan.requires_cropping)
        self.assertEqual(plan.normalized_shape, (5000, 1000, 1000))
        self.assertEqual(plan.crop_grid_shape, (1, 1, 1))
        self.assertEqual(plan.total_crop_count, 1)
        window = plan.windows[0]
        self.assertEqual(window.crop_shape, (5000, 1000, 1000))
        self.assertEqual(window.valid_shape, (5000, 1000, 1000))
        self.assertEqual(
            tuple(_slice_tuple_to_bounds(s) for s in window.requested_output_slices),
            ((0, 5000), (0, 1000), (0, 1000)),
        )

    def test_large_bbox_splits_without_cutting_small_axes(self) -> None:
        plan = build_large_crop_inference_plan(
            requested_bounds=((0, 8000), (0, 1900), (0, 1000)),
            raw_volume_shape=(8000, 1900, 1000),
            context_margin=0,
        )

        self.assertTrue(plan.requires_cropping)
        self.assertEqual(plan.crop_grid_shape[1:], (1, 1))
        self.assertEqual(plan.valid_step_shape[1:], (1900, 1000))
        self.assertEqual(plan.total_crop_count, plan.crop_grid_shape[0])
        self.assertLessEqual(
            max(np.prod(window.crop_shape) for window in plan.windows),
            2200**3,
        )

    def test_internal_crop_boundaries_overlap_by_two_discard_margins(self) -> None:
        plan = build_large_crop_inference_plan(
            requested_bounds=((0, 4000), (0, 400), (0, 400)),
            raw_volume_shape=(4000, 400, 400),
            context_margin=0,
            voxel_budget=2200 * 400 * 400,
        )

        self.assertEqual(plan.internal_discard_margin, 100)
        self.assertEqual(plan.crop_extent_overlap, 200)
        self.assertEqual(plan.crop_grid_shape, (2, 1, 1))
        first, second = plan.windows
        self.assertEqual(_slice_tuple_to_bounds(first.valid_slices[0]), (0, 2000))
        self.assertEqual(_slice_tuple_to_bounds(second.valid_slices[0]), (2000, 4000))
        self.assertEqual(_slice_tuple_to_bounds(first.crop_slices[0]), (0, 2100))
        self.assertEqual(_slice_tuple_to_bounds(second.crop_slices[0]), (1900, 4000))
        self.assertEqual(
            _slice_tuple_to_bounds(first.valid_slices_in_crop[0]),
            (0, 2000),
        )
        self.assertEqual(
            _slice_tuple_to_bounds(second.valid_slices_in_crop[0]),
            (100, 2100),
        )
        self.assertEqual(first.crop_shape[0], 2100)
        self.assertEqual(second.crop_shape[0], 2100)

    def test_normalized_dimensions_are_divisible_by_valid_steps(self) -> None:
        plan = build_large_crop_inference_plan(
            requested_bounds=((0, 570), (0, 400), (0, 370)),
            raw_volume_shape=(570, 400, 370),
            context_margin=0,
            voxel_budget=400 * 400 * 400,
        )

        self.assertGreater(plan.total_crop_count, 1)
        for axis in range(3):
            self.assertEqual(
                plan.normalized_shape[axis] % plan.valid_step_shape[axis],
                0,
            )

    def test_normalized_region_coverage_has_no_gaps_or_overlaps(self) -> None:
        plan = build_large_crop_inference_plan(
            requested_bounds=((0, 570), (0, 400), (0, 370)),
            raw_volume_shape=(570, 400, 370),
            context_margin=0,
            voxel_budget=400 * 400 * 400,
        )

        coverage = _coverage_for_slices(
            plan.normalized_shape,
            tuple(window.valid_slices for window in plan.windows),
        )

        self.assertTrue(bool(np.all(coverage == 1)))

    def test_full_region_borders_do_not_discard_internal_margin(self) -> None:
        plan = build_large_crop_inference_plan(
            requested_bounds=((0, 4000), (0, 400), (0, 400)),
            raw_volume_shape=(4000, 400, 400),
            context_margin=0,
            voxel_budget=2200 * 400 * 400,
        )

        first, second = plan.windows
        self.assertEqual(first.crop_slices[0].start, 0)
        self.assertEqual(first.valid_slices_in_crop[0].start, 0)
        self.assertEqual(second.crop_slices[0].stop, plan.normalized_shape[0])
        self.assertEqual(second.valid_slices_in_crop[0].stop, second.crop_shape[0])

    def test_requested_dimensions_are_padded_to_stride_grid(self) -> None:
        plan = build_large_crop_inference_plan(
            requested_bounds=((0, 2050), (0, 310), (0, 120)),
            raw_volume_shape=(3000, 1000, 1000),
            context_margin=0,
        )

        self.assertEqual(plan.normalized_shape, (2100, 400, 200))
        self.assertEqual(
            tuple(
                _slice_tuple_to_bounds(axis_slice)
                for axis_slice in plan.requested_slices_in_normalized
            ),
            ((0, 2050), (0, 310), (0, 120)),
        )

    def test_context_margin_uses_raw_context_and_records_reflect_padding(self) -> None:
        plan = build_large_crop_inference_plan(
            requested_bounds=((0, 500), (10, 510), (20, 520)),
            raw_volume_shape=(1000, 1000, 1000),
            context_margin=100,
        )

        self.assertEqual(plan.normalized_origin_in_raw, (-100, -90, -80))
        self.assertEqual(plan.requested_shape, (500, 500, 500))
        self.assertEqual(plan.normalized_shape, (700, 700, 700))
        self.assertEqual(
            tuple(
                _slice_tuple_to_bounds(axis_slice)
                for axis_slice in plan.requested_slices_in_normalized
            ),
            ((100, 600), (100, 600), (100, 600)),
        )
        extraction = plan.windows[0].extraction
        self.assertEqual(extraction.pad_before, (100, 90, 80))
        self.assertEqual(extraction.pad_after, (0, 0, 0))
        self.assertEqual(
            tuple(_slice_tuple_to_bounds(axis_slice) for axis_slice in extraction.raw_slices),
            ((0, 600), (0, 610), (0, 620)),
        )

    def test_high_side_context_records_reflect_padding(self) -> None:
        plan = build_large_crop_inference_plan(
            requested_bounds=((500, 1000), (490, 990), (480, 980)),
            raw_volume_shape=(1000, 1000, 1000),
            context_margin=100,
        )

        self.assertEqual(plan.normalized_origin_in_raw, (400, 390, 380))
        self.assertEqual(plan.normalized_shape, (700, 700, 700))
        extraction = plan.windows[0].extraction
        self.assertEqual(extraction.pad_before, (0, 0, 0))
        self.assertEqual(extraction.pad_after, (100, 90, 80))
        self.assertEqual(
            tuple(_slice_tuple_to_bounds(axis_slice) for axis_slice in extraction.raw_slices),
            ((400, 1000), (390, 1000), (380, 1000)),
        )

    def test_requested_output_coverage_has_no_gaps_or_overlaps(self) -> None:
        plan = build_large_crop_inference_plan(
            requested_bounds=((0, 570), (0, 400), (0, 370)),
            raw_volume_shape=(570, 400, 370),
            context_margin=0,
            voxel_budget=400 * 400 * 400,
        )

        coverage = np.zeros(plan.requested_shape, dtype=np.uint8)
        for window in plan.windows:
            z_slice, y_slice, x_slice = window.requested_output_slices
            coverage[z_slice, y_slice, x_slice] += 1

        self.assertGreater(plan.total_crop_count, 1)
        self.assertTrue(bool(np.all(coverage == 1)))
        for axis in range(3):
            valid_bounds = _axis_valid_bounds(plan, axis)
            if len(valid_bounds) <= 1:
                continue
            for previous, current in zip(valid_bounds[:-1], valid_bounds[1:]):
                self.assertEqual(previous[1], current[0])

    def test_rejects_invalid_geometry_inputs(self) -> None:
        invalid_cases = (
            dict(
                requested_bounds=((0, 10), (0, 10), (0, 10)),
                raw_volume_shape=(10, 10),
            ),
            dict(
                requested_bounds=((0, 10), (0, 10), (0, 11)),
                raw_volume_shape=(10, 10, 10),
            ),
            dict(
                requested_bounds=((0, 0), (0, 10), (0, 10)),
                raw_volume_shape=(10, 10, 10),
            ),
            dict(
                requested_bounds=((0, 10), (0, 10), (0, 10)),
                raw_volume_shape=(10, 10, 10),
                context_margin=-1,
            ),
            dict(
                requested_bounds=((0, 10), (0, 10), (0, 10)),
                raw_volume_shape=(10, 10, 10),
                voxel_budget=0,
            ),
        )

        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    build_large_crop_inference_plan(**kwargs)


if __name__ == "__main__":
    unittest.main()
