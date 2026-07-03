from __future__ import annotations

import unittest

import numpy as np

from src.learning.large_crop_extraction import (
    extract_large_crop_from_array,
    extract_large_crop_from_volume,
)
from src.learning.large_crop_inference_plan import build_large_crop_inference_plan


class _FakeVolume:
    def __init__(self, array: np.ndarray) -> None:
        self.array = np.asarray(array)
        self.calls: list[tuple[slice, slice, slice]] = []

    def get_chunk(self, zyx_slices: tuple[slice, slice, slice]) -> np.ndarray:
        self.calls.append(zyx_slices)
        return self.array[zyx_slices]


def _first_window_for_plan(**kwargs):
    return build_large_crop_inference_plan(**kwargs).windows[0]


class LargeCropExtractionTests(unittest.TestCase):
    def test_extract_large_crop_from_array_without_padding(self) -> None:
        array = np.arange(20 * 30 * 40, dtype=np.uint16).reshape((20, 30, 40))
        window = _first_window_for_plan(
            requested_bounds=((2, 12), (3, 15), (4, 18)),
            raw_volume_shape=array.shape,
            context_margin=0,
            minivol_size=4,
        )

        cropped = extract_large_crop_from_array(array, window=window)

        expected = array[2:12, 3:15, 4:18]
        np.testing.assert_array_equal(cropped, expected)
        self.assertEqual(cropped.dtype, array.dtype)

    def test_extract_large_crop_from_array_with_low_side_reflect_padding(self) -> None:
        array = np.arange(8 * 9 * 10, dtype=np.int16).reshape((8, 9, 10))
        window = _first_window_for_plan(
            requested_bounds=((0, 4), (1, 5), (2, 6)),
            raw_volume_shape=array.shape,
            context_margin=2,
            minivol_size=4,
        )

        cropped = extract_large_crop_from_array(array, window=window)

        expected = np.pad(
            array[0:6, 0:7, 0:8],
            ((2, 0), (1, 0), (0, 0)),
            mode="reflect",
        )
        np.testing.assert_array_equal(cropped, expected)
        self.assertEqual(tuple(cropped.shape), window.crop_shape)

    def test_extract_large_crop_from_array_with_high_side_reflect_padding(self) -> None:
        array = np.arange(8 * 9 * 10, dtype=np.int16).reshape((8, 9, 10))
        window = _first_window_for_plan(
            requested_bounds=((4, 8), (4, 9), (5, 10)),
            raw_volume_shape=array.shape,
            context_margin=2,
            minivol_size=4,
        )

        cropped = extract_large_crop_from_array(array, window=window)

        expected = np.pad(
            array[2:8, 2:9, 3:10],
            ((0, 2), (0, 3), (0, 3)),
            mode="reflect",
        )
        np.testing.assert_array_equal(cropped, expected)
        self.assertEqual(tuple(cropped.shape), window.crop_shape)

    def test_extract_large_crop_from_volume_uses_planned_raw_slices(self) -> None:
        array = np.arange(8 * 9 * 10, dtype=np.float32).reshape((8, 9, 10))
        volume = _FakeVolume(array)
        window = _first_window_for_plan(
            requested_bounds=((0, 4), (1, 5), (2, 6)),
            raw_volume_shape=array.shape,
            context_margin=2,
            minivol_size=4,
        )

        cropped = extract_large_crop_from_volume(volume, window=window)

        self.assertEqual(volume.calls, [window.extraction.raw_slices])
        np.testing.assert_array_equal(
            cropped,
            extract_large_crop_from_array(array, window=window),
        )

    def test_extract_large_crop_rejects_reflect_padding_on_singleton_axis(self) -> None:
        array = np.arange(1 * 4 * 4, dtype=np.uint8).reshape((1, 4, 4))
        window = _first_window_for_plan(
            requested_bounds=((0, 1), (0, 4), (0, 4)),
            raw_volume_shape=array.shape,
            context_margin=2,
            minivol_size=4,
        )

        with self.assertRaisesRegex(ValueError, "Cannot apply reflect padding on z axis"):
            extract_large_crop_from_array(array, window=window)

    def test_extract_large_crop_rejects_invalid_sources(self) -> None:
        window = _first_window_for_plan(
            requested_bounds=((0, 4), (0, 4), (0, 4)),
            raw_volume_shape=(4, 4, 4),
            context_margin=0,
            minivol_size=4,
        )

        with self.assertRaisesRegex(ValueError, "3D"):
            extract_large_crop_from_array(np.zeros((4, 4), dtype=np.uint8), window=window)
        with self.assertRaisesRegex(TypeError, "get_chunk"):
            extract_large_crop_from_volume(object(), window=window)


if __name__ == "__main__":
    unittest.main()
