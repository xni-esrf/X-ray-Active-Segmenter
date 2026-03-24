from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .loader import VolumeInfo, VolumeLoader, _split_qualified_path


class Hdf5Loader(VolumeLoader):
    _RDCC_NBYTES = 256 * 1024 * 1024
    _RDCC_NSLOTS = 1_000_003

    def __init__(self, path: str, *, voxel_spacing: Optional[Tuple[float, float, float]] = None) -> None:
        super().__init__(path)
        base, qualifier = _split_qualified_path(path)
        self._path = base
        self._qualifier = qualifier
        self._file = self._open_file()
        self._dataset = self._select_dataset()
        chunk_shape = getattr(self._dataset, "chunks", None)
        self._info = VolumeInfo(
            shape=tuple(self._dataset.shape),
            dtype=str(self._dataset.dtype),
            voxel_spacing=voxel_spacing or (1.0, 1.0, 1.0),
            chunk_shape=tuple(chunk_shape) if chunk_shape else None,
            axes="zyx",
        )

    @property
    def info(self) -> VolumeInfo:
        return self._info

    def get_chunk(self, zyx_slices: Tuple[slice, slice, slice]) -> np.ndarray:
        return np.asarray(self._dataset[zyx_slices])

    def close(self) -> None:
        try:
            self._file.close()
        finally:
            return None

    def _open_file(self):
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py is required to load HDF5 volumes") from exc
        try:
            return h5py.File(
                self._path,
                "r",
                rdcc_nbytes=self._RDCC_NBYTES,
                rdcc_nslots=self._RDCC_NSLOTS,
            )
        except TypeError:
            return h5py.File(self._path, "r")

    def _select_dataset(self):
        if self._qualifier and self._qualifier in self._file:
            return self._file[self._qualifier]
        for name, obj in self._file.items():
            if hasattr(obj, "shape"):
                return obj
        raise ValueError("No dataset found in HDF5 file")
