from __future__ import annotations

import unittest

from src.learning.inference_geometry import (
    DEFAULT_INFERENCE_MINIVOL_SIZE,
    coerce_inference_minivol_size,
    inference_stride_for_minivol_size,
)


class InferenceGeometryTests(unittest.TestCase):
    def test_default_minivol_size_contract(self) -> None:
        self.assertEqual(DEFAULT_INFERENCE_MINIVOL_SIZE, 200)

    def test_stride_is_half_the_minivol_size(self) -> None:
        self.assertEqual(inference_stride_for_minivol_size(200), 100)
        self.assertEqual(inference_stride_for_minivol_size(128), 64)

    def test_rejects_invalid_minivol_size(self) -> None:
        invalid_values = (True, 1.5, "200", 1, 0)
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    coerce_inference_minivol_size(value)


if __name__ == "__main__":
    unittest.main()
