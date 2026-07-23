from __future__ import annotations

import unittest

import numpy as np

from src.io.loader import ZeroVolumeLoader


class ZeroVolumeLoaderTests(unittest.TestCase):
    def test_reports_requested_shape_dtype_and_metadata(self) -> None:
        loader = ZeroVolumeLoader(
            path="<zeros>",
            shape=(4, 5, 6),
            dtype="uint16",
            voxel_spacing=(2.0, 3.0, 4.0),
            axes="zyx",
        )
        info = loader.info
        self.assertEqual(info.shape, (4, 5, 6))
        self.assertEqual(info.dtype, str(np.dtype(np.uint16)))
        self.assertEqual(info.voxel_spacing, (2.0, 3.0, 4.0))
        self.assertEqual(info.axes, "zyx")

    def test_get_chunk_returns_region_sized_zeros(self) -> None:
        loader = ZeroVolumeLoader(path="<zeros>", shape=(10, 10, 10), dtype="uint8")
        chunk = loader.get_chunk((slice(2, 5), slice(0, 4), slice(6, 10)))
        self.assertEqual(chunk.shape, (3, 4, 4))
        self.assertEqual(np.dtype(chunk.dtype), np.dtype(np.uint8))
        self.assertFalse(np.any(chunk))

    def test_get_slice_returns_zeros_plane(self) -> None:
        loader = ZeroVolumeLoader(path="<zeros>", shape=(3, 4, 5), dtype="uint8")
        plane = loader.get_slice(0, 1)
        self.assertEqual(plane.shape, (4, 5))
        self.assertFalse(np.any(plane))

    def test_huge_shape_does_not_allocate_full_volume(self) -> None:
        # A dense array of this shape would be ~1 TiB; the lazy loader must only
        # allocate the small requested region.
        loader = ZeroVolumeLoader(
            path="<zeros>",
            shape=(11602, 11488, 7771),
            dtype="uint8",
        )
        chunk = loader.get_chunk((slice(0, 2), slice(0, 2), slice(0, 2)))
        self.assertEqual(chunk.shape, (2, 2, 2))


if __name__ == "__main__":
    unittest.main()
