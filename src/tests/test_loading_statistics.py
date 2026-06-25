from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from src import loading
from src.io.loader import VolumeInfo, VolumeLoader
from src.io.tiff_loader import TiffLoader


class _ArrayLoader(VolumeLoader):
    def __init__(
        self,
        array: np.ndarray,
        *,
        path: str = "<array-loader>",
        chunk_shape: tuple[int, int, int] | None = (1, 2, 2),
    ) -> None:
        super().__init__(path)
        self._array = np.asarray(array)
        self.calls = 0
        self._info = VolumeInfo(
            shape=tuple(int(dim) for dim in self._array.shape),
            dtype=str(self._array.dtype),
            voxel_spacing=(1.0, 1.0, 1.0),
            chunk_shape=chunk_shape,
            axes="zyx",
        )

    @property
    def info(self) -> VolumeInfo:
        return self._info

    def get_chunk(self, zyx_slices: tuple[slice, slice, slice]) -> np.ndarray:
        self.calls += 1
        return np.asarray(self._array[zyx_slices])


class LoadingStatisticsTests(unittest.TestCase):
    def test_tiff_loader_does_not_compute_diagnostic_statistics(self) -> None:
        array = np.array([[[1, 2], [3, 4]]], dtype=np.uint16)

        with patch.object(TiffLoader, "_open_tiff", return_value=array), patch.object(
            np, "min", wraps=np.min
        ) as min_mock, patch.object(
            np, "max", wraps=np.max
        ) as max_mock, patch.object(
            np, "mean", wraps=np.mean
        ) as mean_mock:
            loader = TiffLoader("/tmp/segmentation.tif")

        self.assertEqual(loader.info.shape, (1, 2, 2))
        min_mock.assert_not_called()
        max_mock.assert_not_called()
        mean_mock.assert_not_called()

    def test_raw_load_precomputes_data_range_without_mean(self) -> None:
        array = np.arange(8, dtype=np.uint16).reshape((2, 2, 2)) + 10
        loader = _ArrayLoader(array)

        with patch.object(loading, "create_loader", return_value=loader), patch.object(
            np, "mean", wraps=np.mean
        ) as mean_mock:
            prepared = loading.load_prepared_volume(
                "/tmp/raw.npy",
                kind="raw",
                load_mode="lazy",
                cache_max_bytes=1024,
                pyramid_levels=1,
            )

        self.assertEqual(prepared.volume.data_range, (10.0, 17.0))
        self.assertGreater(loader.calls, 0)
        mean_mock.assert_not_called()

    def test_semantic_lazy_load_does_not_compute_data_range_or_mean(self) -> None:
        array = np.arange(8, dtype=np.uint16).reshape((2, 2, 2)) + 10
        loader = _ArrayLoader(array)

        with patch.object(loading, "create_loader", return_value=loader), patch.object(
            np, "min", wraps=np.min
        ) as min_mock, patch.object(
            np, "max", wraps=np.max
        ) as max_mock, patch.object(
            np, "mean", wraps=np.mean
        ) as mean_mock:
            prepared = loading.load_prepared_volume(
                "/tmp/semantic.npy",
                kind="semantic",
                load_mode="lazy",
                cache_max_bytes=1024,
                pyramid_levels=1,
            )

        self.assertIsNone(prepared.volume.data_range)
        min_mock.assert_not_called()
        max_mock.assert_not_called()
        mean_mock.assert_not_called()

    def test_raw_load_rejects_non_finite_values_during_opening(self) -> None:
        array = np.asarray(
            [[[0.0, 1.0], [2.0, np.nan]], [[4.0, 5.0], [6.0, 7.0]]],
            dtype=np.float32,
        )
        loader = _ArrayLoader(array)

        with patch.object(loading, "create_loader", return_value=loader):
            with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                loading.load_prepared_volume(
                    "/tmp/raw.npy",
                    kind="raw",
                    load_mode="lazy",
                    cache_max_bytes=1024,
                    pyramid_levels=1,
                )


if __name__ == "__main__":
    unittest.main()
