from __future__ import annotations

import unittest

import numpy as np

from src.learning.zero_occupancy import (
    build_zero_occupancy_grid,
    region_is_definitely_empty,
)


class _FakeVolume:
    def __init__(self, array: np.ndarray, *, chunk_shape: tuple[int, int, int] | None = None) -> None:
        self.array = np.asarray(array)
        self.shape = tuple(int(axis) for axis in self.array.shape)
        self.chunk_shape = chunk_shape

    def get_chunk(self, zyx_slices: tuple[slice, slice, slice]) -> np.ndarray:
        return self.array[zyx_slices]


class ZeroOccupancyGridTests(unittest.TestCase):
    def test_all_zero_region_is_fully_empty(self) -> None:
        volume = _FakeVolume(np.zeros((40, 40, 40), dtype=np.float32))
        grid = build_zero_occupancy_grid(
            volume,
            bounds=((0, 40), (0, 40), (0, 40)),
            block_size=10,
        )
        self.assertEqual(grid.occupied.shape, (4, 4, 4))
        self.assertFalse(bool(grid.occupied.any()))
        self.assertTrue(
            region_is_definitely_empty(grid, raw_slices=(slice(0, 40), slice(0, 40), slice(0, 40)))
        )

    def test_localized_nonzero_block_is_detected(self) -> None:
        array = np.zeros((40, 40, 40), dtype=np.float32)
        array[25:27, 5:7, 5:7] = 1.0
        volume = _FakeVolume(array)
        grid = build_zero_occupancy_grid(
            volume,
            bounds=((0, 40), (0, 40), (0, 40)),
            block_size=10,
        )
        # Region far from the nonzero block is still reported empty.
        self.assertTrue(
            region_is_definitely_empty(grid, raw_slices=(slice(0, 10), slice(0, 10), slice(0, 10)))
        )
        # Region overlapping the nonzero block's cell is not reported empty.
        self.assertFalse(
            region_is_definitely_empty(grid, raw_slices=(slice(20, 30), slice(0, 10), slice(0, 10)))
        )

    def test_negative_values_are_treated_as_nonzero(self) -> None:
        array = np.zeros((20, 20, 20), dtype=np.float32)
        array[:, :, :] = -3.0
        volume = _FakeVolume(array)
        grid = build_zero_occupancy_grid(
            volume,
            bounds=((0, 20), (0, 20), (0, 20)),
            block_size=5,
        )
        # A max-of-raw-value reduction would incorrectly see this as all-zero.
        self.assertTrue(bool(grid.occupied.all()))
        self.assertFalse(
            region_is_definitely_empty(grid, raw_slices=(slice(0, 20), slice(0, 20), slice(0, 20)))
        )

    def test_bounds_not_multiple_of_block_size(self) -> None:
        array = np.zeros((23, 17, 9), dtype=np.float32)
        array[-1, -1, -1] = 5.0
        volume = _FakeVolume(array)
        grid = build_zero_occupancy_grid(
            volume,
            bounds=((0, 23), (0, 17), (0, 9)),
            block_size=5,
        )
        self.assertEqual(grid.occupied.shape, (5, 4, 2))
        self.assertTrue(
            region_is_definitely_empty(grid, raw_slices=(slice(0, 20), slice(0, 15), slice(0, 5)))
        )
        self.assertFalse(
            region_is_definitely_empty(grid, raw_slices=(slice(20, 23), slice(15, 17), slice(5, 9)))
        )

    def test_bounds_with_nonzero_origin(self) -> None:
        array = np.zeros((60, 60, 60), dtype=np.float32)
        array[55, 55, 55] = 2.0
        volume = _FakeVolume(array)
        grid = build_zero_occupancy_grid(
            volume,
            bounds=((50, 60), (50, 60), (50, 60)),
            block_size=5,
        )
        self.assertEqual(grid.origin, (50, 50, 50))
        self.assertTrue(
            region_is_definitely_empty(grid, raw_slices=(slice(50, 55), slice(50, 55), slice(50, 55)))
        )
        self.assertFalse(
            region_is_definitely_empty(grid, raw_slices=(slice(55, 60), slice(55, 60), slice(55, 60)))
        )

    def test_native_chunk_shape_is_respected_and_snapped_to_block_size(self) -> None:
        array = np.zeros((30, 30, 30), dtype=np.float32)
        array[29, 29, 29] = 1.0
        volume = _FakeVolume(array, chunk_shape=(7, 11, 4))
        grid = build_zero_occupancy_grid(
            volume,
            bounds=((0, 30), (0, 30), (0, 30)),
            block_size=5,
        )
        self.assertEqual(grid.occupied.shape, (6, 6, 6))
        self.assertFalse(
            region_is_definitely_empty(grid, raw_slices=(slice(25, 30), slice(25, 30), slice(25, 30)))
        )

    def test_region_outside_scanned_bounds_is_not_reported_empty(self) -> None:
        volume = _FakeVolume(np.zeros((40, 40, 40), dtype=np.float32))
        grid = build_zero_occupancy_grid(
            volume,
            bounds=((10, 30), (10, 30), (10, 30)),
            block_size=5,
        )
        self.assertFalse(
            region_is_definitely_empty(grid, raw_slices=(slice(0, 15), slice(10, 15), slice(10, 15)))
        )


if __name__ == "__main__":
    unittest.main()
