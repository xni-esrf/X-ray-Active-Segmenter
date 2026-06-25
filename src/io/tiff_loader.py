from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .loader import VolumeInfo, VolumeLoader


class TiffLoader(VolumeLoader):
    def __init__(
        self,
        path: str,
        *,
        voxel_spacing: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        super().__init__(path)
        self._array = self._open_tiff(path)
        self._info = VolumeInfo(
            shape=tuple(self._array.shape),
            dtype=str(self._array.dtype),
            voxel_spacing=voxel_spacing or (1.0, 1.0, 1.0),
            chunk_shape=None,
            axes="zyx",
        )

    @property
    def info(self) -> VolumeInfo:
        return self._info

    def get_chunk(self, zyx_slices: Tuple[slice, slice, slice]) -> np.ndarray:
        return np.asarray(self._array[zyx_slices])

    @staticmethod
    def _open_tiff(path: str) -> np.ndarray:
        try:
            import tifffile
        except ImportError as exc:
            raise ImportError("tifffile is required to load TIFF volumes") from exc

        try:
            return tifffile.memmap(path)
        except Exception:
            return tifffile.imread(path)
