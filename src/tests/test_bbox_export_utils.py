from __future__ import annotations

import unittest

import numpy as np

from src.io.bbox_export_utils import (
    BBoxContextPlan,
    extract_bbox_context_from_array,
    extract_planned_bbox_context_from_array,
    plan_axis_context,
    plan_bbox_context,
    split_centered_extension,
    target_size_for_bbox_dimension,
)


class BBoxExportUtilsTests(unittest.TestCase):
    def test_target_size_rule_thresholds(self) -> None:
        self.assertEqual(target_size_for_bbox_dimension(1), 300)
        self.assertEqual(target_size_for_bbox_dimension(250), 300)
        self.assertEqual(target_size_for_bbox_dimension(251), 400)
        self.assertEqual(target_size_for_bbox_dimension(350), 400)
        self.assertEqual(target_size_for_bbox_dimension(351), 500)
        self.assertEqual(target_size_for_bbox_dimension(450), 500)
        self.assertEqual(target_size_for_bbox_dimension(451), 600)
        self.assertEqual(target_size_for_bbox_dimension(550), 600)
        self.assertEqual(target_size_for_bbox_dimension(551), 700)

    def test_target_size_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            target_size_for_bbox_dimension(0)
        with self.assertRaises(ValueError):
            target_size_for_bbox_dimension(-1)
        with self.assertRaises(TypeError):
            target_size_for_bbox_dimension(True)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            target_size_for_bbox_dimension(12.5)  # type: ignore[arg-type]

    def test_split_centered_extension_even_and_odd(self) -> None:
        self.assertEqual(split_centered_extension(0), (0, 0))
        self.assertEqual(split_centered_extension(1), (1, 0))
        self.assertEqual(split_centered_extension(2), (1, 1))
        self.assertEqual(split_centered_extension(3), (2, 1))
        self.assertEqual(split_centered_extension(4), (2, 2))
        self.assertEqual(split_centered_extension(5), (3, 2))

    def test_split_centered_extension_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            split_centered_extension(-1)
        with self.assertRaises(TypeError):
            split_centered_extension(False)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            split_centered_extension(2.2)  # type: ignore[arg-type]

    def test_threshold_cases_apply_odd_extension_before(self) -> None:
        size = 251
        target = target_size_for_bbox_dimension(size)
        before, after = split_centered_extension(target - size)
        self.assertEqual(target, 400)
        self.assertEqual((before, after), (75, 74))

        size = 351
        target = target_size_for_bbox_dimension(size)
        before, after = split_centered_extension(target - size)
        self.assertEqual(target, 500)
        self.assertEqual((before, after), (75, 74))

    def test_plan_axis_context_centered_without_padding(self) -> None:
        plan = plan_axis_context(start=100, stop=200, volume_size=600)
        self.assertEqual(plan.original_size, 100)
        self.assertEqual(plan.target_size, 300)
        self.assertEqual((plan.extend_before, plan.extend_after), (100, 100))
        self.assertEqual((plan.planned_start, plan.planned_stop), (0, 300))
        self.assertEqual((plan.clipped_start, plan.clipped_stop), (0, 300))
        self.assertEqual((plan.pad_before, plan.pad_after), (0, 0))
        self.assertEqual(plan.final_size, 300)

    def test_plan_axis_context_odd_extension_adds_before(self) -> None:
        plan = plan_axis_context(start=100, stop=251, volume_size=900)
        self.assertEqual(plan.original_size, 151)
        self.assertEqual(plan.target_size, 300)
        self.assertEqual((plan.extend_before, plan.extend_after), (75, 74))
        self.assertEqual((plan.planned_start, plan.planned_stop), (25, 325))
        self.assertEqual((plan.pad_before, plan.pad_after), (0, 0))
        self.assertEqual(plan.final_size, 300)

    def test_plan_axis_context_preserves_centering_then_pads_at_low_border(self) -> None:
        plan = plan_axis_context(start=10, stop=110, volume_size=1000)
        self.assertEqual(plan.target_size, 300)
        self.assertEqual((plan.extend_before, plan.extend_after), (100, 100))
        self.assertEqual((plan.planned_start, plan.planned_stop), (-90, 210))
        self.assertEqual((plan.clipped_start, plan.clipped_stop), (0, 210))
        self.assertEqual((plan.pad_before, plan.pad_after), (90, 0))
        self.assertEqual(plan.final_size, 300)

    def test_plan_axis_context_preserves_centering_then_pads_at_high_border(self) -> None:
        plan = plan_axis_context(start=900, stop=990, volume_size=1000)
        self.assertEqual(plan.target_size, 300)
        self.assertEqual((plan.extend_before, plan.extend_after), (105, 105))
        self.assertEqual((plan.planned_start, plan.planned_stop), (795, 1095))
        self.assertEqual((plan.clipped_start, plan.clipped_stop), (795, 1000))
        self.assertEqual((plan.pad_before, plan.pad_after), (0, 95))
        self.assertEqual(plan.final_size, 300)

    def test_plan_axis_context_pads_both_sides_when_target_exceeds_volume(self) -> None:
        plan = plan_axis_context(start=10, stop=40, volume_size=120)
        self.assertEqual(plan.original_size, 30)
        self.assertEqual(plan.target_size, 300)
        self.assertEqual((plan.planned_start, plan.planned_stop), (-125, 175))
        self.assertEqual((plan.clipped_start, plan.clipped_stop), (0, 120))
        self.assertEqual((plan.pad_before, plan.pad_after), (125, 55))
        self.assertEqual(plan.final_size, 300)

    def test_plan_axis_context_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            plan_axis_context(start=4, stop=4, volume_size=10)
        with self.assertRaises(ValueError):
            plan_axis_context(start=5, stop=4, volume_size=10)
        with self.assertRaises(ValueError):
            plan_axis_context(start=-1, stop=4, volume_size=10)
        with self.assertRaises(ValueError):
            plan_axis_context(start=0, stop=11, volume_size=10)
        with self.assertRaises(ValueError):
            plan_axis_context(start=0, stop=4, volume_size=10, target_size=3)

    def test_plan_axis_context_large_thresholds(self) -> None:
        plan_550 = plan_axis_context(start=100, stop=650, volume_size=2000)
        self.assertEqual(plan_550.original_size, 550)
        self.assertEqual(plan_550.target_size, 600)
        self.assertEqual((plan_550.extend_before, plan_550.extend_after), (25, 25))
        self.assertEqual(plan_550.final_size, 600)

        plan_551 = plan_axis_context(start=100, stop=651, volume_size=3000)
        self.assertEqual(plan_551.original_size, 551)
        self.assertEqual(plan_551.target_size, 700)
        self.assertEqual((plan_551.extend_before, plan_551.extend_after), (75, 74))
        self.assertEqual(plan_551.final_size, 700)

    def test_plan_bbox_context_returns_axis_plans_and_pad_width(self) -> None:
        plan = plan_bbox_context(
            z_bounds=(10, 110),
            y_bounds=(900, 990),
            x_bounds=(10, 40),
            volume_shape=(1000, 1000, 120),
        )
        self.assertEqual(plan.z.target_size, 300)
        self.assertEqual(plan.y.target_size, 300)
        self.assertEqual(plan.x.target_size, 300)
        self.assertEqual(plan.clipped_bounds, ((0, 210), (795, 1000), (0, 120)))
        self.assertEqual(plan.pad_width, ((90, 0), (0, 95), (125, 55)))

    def test_plan_bbox_context_rejects_invalid_shapes(self) -> None:
        with self.assertRaises(ValueError):
            plan_bbox_context(
                z_bounds=(0, 1),
                y_bounds=(0, 1),
                x_bounds=(0, 1),
                volume_shape=(10, 10),  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            plan_bbox_context(
                z_bounds=(0,),  # type: ignore[arg-type]
                y_bounds=(0, 1),
                x_bounds=(0, 1),
                volume_shape=(10, 10, 10),
            )

    def test_extract_planned_bbox_context_without_padding(self) -> None:
        array = np.arange(6 * 7 * 8, dtype=np.uint16).reshape((6, 7, 8))
        # Keep this test small by fixing targets to the original size.
        z_plan = plan_axis_context(start=1, stop=5, volume_size=6, target_size=4)
        y_plan = plan_axis_context(start=2, stop=6, volume_size=7, target_size=4)
        x_plan = plan_axis_context(start=3, stop=7, volume_size=8, target_size=4)
        small_plan = BBoxContextPlan(z=z_plan, y=y_plan, x=x_plan)

        cropped = extract_planned_bbox_context_from_array(array, plan=small_plan)
        expected = array[1:5, 2:6, 3:7]
        np.testing.assert_array_equal(cropped, expected)
        self.assertEqual(cropped.dtype, array.dtype)

    def test_extract_planned_bbox_context_with_reflect_padding(self) -> None:
        array = np.arange(4 * 5 * 6, dtype=np.int16).reshape((4, 5, 6))
        z_plan = plan_axis_context(start=0, stop=2, volume_size=4, target_size=4)
        y_plan = plan_axis_context(start=1, stop=3, volume_size=5, target_size=4)
        x_plan = plan_axis_context(start=2, stop=4, volume_size=6, target_size=4)
        small_plan = BBoxContextPlan(z=z_plan, y=y_plan, x=x_plan)

        cropped = extract_planned_bbox_context_from_array(array, plan=small_plan)
        expected = np.pad(
            array[0:3, 0:4, 1:5],
            ((1, 0), (0, 0), (0, 0)),
            mode="reflect",
        )
        np.testing.assert_array_equal(cropped, expected)
        self.assertEqual(tuple(cropped.shape), (4, 4, 4))
        self.assertEqual(cropped.dtype, array.dtype)

    def test_extract_planned_bbox_context_rejects_reflect_with_singleton_axis(self) -> None:
        array = np.arange(1 * 4 * 4, dtype=np.uint8).reshape((1, 4, 4))
        z_plan = plan_axis_context(start=0, stop=1, volume_size=1, target_size=4)
        y_plan = plan_axis_context(start=0, stop=4, volume_size=4, target_size=4)
        x_plan = plan_axis_context(start=0, stop=4, volume_size=4, target_size=4)
        small_plan = BBoxContextPlan(z=z_plan, y=y_plan, x=x_plan)

        with self.assertRaisesRegex(ValueError, "Cannot apply reflect padding on z axis"):
            extract_planned_bbox_context_from_array(array, plan=small_plan)

    def test_extract_bbox_context_from_array_uses_default_target_rule(self) -> None:
        array = np.zeros((260, 260, 260), dtype=np.uint8)
        cropped = extract_bbox_context_from_array(
            array,
            z_bounds=(5, 255),
            y_bounds=(5, 255),
            x_bounds=(5, 255),
        )
        self.assertEqual(tuple(cropped.shape), (300, 300, 300))
        self.assertEqual(cropped.dtype, array.dtype)


if __name__ == "__main__":
    unittest.main()
