from __future__ import annotations

import unittest

from src.learning.inference_geometry import (
    DEFAULT_INFERENCE_MINIVOL_SIZE,
    DEFAULT_INFERENCE_STRIDE,
    DEFAULT_LARGE_CROP_EDGE_VOXELS,
    DEFAULT_LARGE_CROP_VOXEL_BUDGET,
    INFERENCE_CROP_EXTENT_OVERLAP,
    INFERENCE_INTERNAL_CROP_DISCARD_MARGIN,
    coerce_inference_minivol_size,
    inference_crop_extent_overlap_for_minivol_size,
    inference_internal_crop_discard_margin_for_minivol_size,
    inference_stride_for_minivol_size,
    inference_valid_step_for_crop_size,
)


class InferenceGeometryTests(unittest.TestCase):
    def test_default_inference_geometry_contract(self) -> None:
        self.assertEqual(DEFAULT_INFERENCE_MINIVOL_SIZE, 200)
        self.assertEqual(DEFAULT_INFERENCE_STRIDE, 100)
        self.assertEqual(INFERENCE_INTERNAL_CROP_DISCARD_MARGIN, 100)
        self.assertEqual(INFERENCE_CROP_EXTENT_OVERLAP, 200)
        self.assertEqual(DEFAULT_LARGE_CROP_EDGE_VOXELS, 2200)
        self.assertEqual(DEFAULT_LARGE_CROP_VOXEL_BUDGET, 2200**3)

    def test_geometry_is_derived_from_minivol_size(self) -> None:
        self.assertEqual(inference_stride_for_minivol_size(200), 100)
        self.assertEqual(
            inference_internal_crop_discard_margin_for_minivol_size(200),
            100,
        )
        self.assertEqual(inference_crop_extent_overlap_for_minivol_size(200), 200)
        self.assertEqual(inference_valid_step_for_crop_size(2200), 2000)

    def test_custom_minivol_size_uses_half_stride_contract(self) -> None:
        self.assertEqual(inference_stride_for_minivol_size(128), 64)
        self.assertEqual(inference_crop_extent_overlap_for_minivol_size(128), 128)
        self.assertEqual(
            inference_valid_step_for_crop_size(512, minivol_size=128),
            384,
        )

    def test_rejects_invalid_minivol_size(self) -> None:
        invalid_values = (True, 1.5, "200", 1, 0)
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    coerce_inference_minivol_size(value)

    def test_rejects_crop_size_that_cannot_hold_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "crop_size"):
            inference_valid_step_for_crop_size(200)
        with self.assertRaisesRegex(TypeError, "crop_size"):
            inference_valid_step_for_crop_size(True)


if __name__ == "__main__":
    unittest.main()

