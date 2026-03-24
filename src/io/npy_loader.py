from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .loader import VolumeInfo, VolumeLoader, _split_qualified_path


class NpyLoader(VolumeLoader):
    def __init__(self, path: str, *, voxel_spacing: Optional[Tuple[float, float, float]] = None) -> None:
        super().__init__(path)
        base, qualifier = _split_qualified_path(path)
        self._path = base
        self._qualifier = qualifier
        self._array = self._load_array()
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

    def _load_array(self) -> np.ndarray:
        path = Path(self._path)
        suffix = path.suffix.lower()
        if suffix == ".npy":
            return np.load(path, mmap_mode="r")
        if suffix == ".npz":
            npz = np.load(path, mmap_mode="r")
            try:
                if self._qualifier and self._qualifier in npz:
                    return npz[self._qualifier]
                first_key = next(iter(npz.files))
                return npz[first_key]
            finally:
                npz.close()
        raise ValueError(f"Unsupported npy/npz file: {path}")
