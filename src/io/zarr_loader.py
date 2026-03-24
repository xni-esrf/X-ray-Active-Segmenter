from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .loader import VolumeInfo, VolumeLoader, _split_qualified_path


class ZarrLoader(VolumeLoader):
    def __init__(self, path: str, *, voxel_spacing: Optional[Tuple[float, float, float]] = None) -> None:
        super().__init__(path)
        base, qualifier = _split_qualified_path(path)
        self._path = base
        self._qualifier = qualifier
        self._array = self._open_zarr()
        chunk_shape = getattr(self._array, "chunks", None)
        self._info = VolumeInfo(
            shape=tuple(self._array.shape),
            dtype=str(self._array.dtype),
            voxel_spacing=voxel_spacing or (1.0, 1.0, 1.0),
            chunk_shape=tuple(chunk_shape) if chunk_shape else None,
            axes="zyx",
        )

    @property
    def info(self) -> VolumeInfo:
        return self._info

    def get_chunk(self, zyx_slices: Tuple[slice, slice, slice]) -> np.ndarray:
        return np.asarray(self._array[zyx_slices])

    def _open_zarr(self):
        try:
            import zarr
        except ImportError as exc:
            raise ImportError("zarr is required to load Zarr volumes") from exc

        root = zarr.open(self._path, mode="r")
        if hasattr(root, "shape"):
            return root
        if self._qualifier and self._qualifier in root:
            return root[self._qualifier]
        for name in root.array_keys():
            return root[name]
        raise ValueError("No array found in Zarr group")
