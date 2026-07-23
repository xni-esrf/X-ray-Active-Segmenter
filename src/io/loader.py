from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import logging
import numpy as np


VoxelSpacing = Tuple[float, float, float]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VolumeInfo:
    shape: Tuple[int, int, int]
    dtype: str
    voxel_spacing: VoxelSpacing = (1.0, 1.0, 1.0)
    chunk_shape: Optional[Tuple[int, int, int]] = None
    axes: str = "zyx"


class VolumeLoader(ABC):
    def __init__(self, path: str) -> None:
        self.path = path

    @property
    @abstractmethod
    def info(self) -> VolumeInfo:
        raise NotImplementedError

    @abstractmethod
    def get_chunk(self, zyx_slices: Tuple[slice, slice, slice]) -> np.ndarray:
        raise NotImplementedError

    def get_slice(self, axis: int, index: int) -> np.ndarray:
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2")
        slices = [slice(None), slice(None), slice(None)]
        slices[axis] = slice(index, index + 1)
        return self.get_chunk(tuple(slices)).squeeze(axis=axis)

    def close(self) -> None:
        return None


class Float16Loader(VolumeLoader):
    def __init__(self, loader: VolumeLoader) -> None:
        super().__init__(loader.path)
        self._loader = loader
        self._conversion_logged = False
        src_info = loader.info
        self._info = VolumeInfo(
            shape=src_info.shape,
            dtype=str(np.dtype(np.float16)),
            voxel_spacing=src_info.voxel_spacing,
            chunk_shape=src_info.chunk_shape,
            axes=src_info.axes,
        )

    @property
    def info(self) -> VolumeInfo:
        return self._info

    def get_chunk(self, zyx_slices: Tuple[slice, slice, slice]) -> np.ndarray:
        chunk = self._loader.get_chunk(zyx_slices)
        if np.dtype(chunk.dtype) == np.dtype(np.float32):
            self._log_conversion()
            return chunk.astype(np.float16, copy=False)
        return chunk

    def get_slice(self, axis: int, index: int) -> np.ndarray:
        chunk = self._loader.get_slice(axis, index)
        if np.dtype(chunk.dtype) == np.dtype(np.float32):
            self._log_conversion()
            return chunk.astype(np.float16, copy=False)
        return chunk

    def close(self) -> None:
        return self._loader.close()

    def _log_conversion(self) -> None:
        if self._conversion_logged:
            return
        self._conversion_logged = True
        logger.info(
            "Applied float32->float16 conversion while reading volume: %s",
            self.path,
        )


class InMemoryVolumeLoader(VolumeLoader):
    def __init__(
        self,
        *,
        path: str,
        array: np.ndarray,
        voxel_spacing: VoxelSpacing = (1.0, 1.0, 1.0),
        axes: str = "zyx",
    ) -> None:
        super().__init__(path)
        self._array = np.asarray(array)
        self._info = VolumeInfo(
            shape=tuple(self._array.shape),
            dtype=str(self._array.dtype),
            voxel_spacing=voxel_spacing,
            chunk_shape=None,
            axes=axes,
        )

    @property
    def info(self) -> VolumeInfo:
        return self._info

    def get_chunk(self, zyx_slices: Tuple[slice, slice, slice]) -> np.ndarray:
        return np.asarray(self._array[zyx_slices])


class ZeroVolumeLoader(VolumeLoader):
    """Virtual all-zeros volume that never materializes the full array.

    Reports the requested shape/dtype but allocates only the region asked for by
    each ``get_chunk`` call, so it can stand in for a full-volume placeholder
    (e.g. an empty segmentation during inference) without the memory cost of a
    dense array.
    """

    def __init__(
        self,
        *,
        path: str,
        shape: Tuple[int, int, int],
        dtype: str = "uint8",
        voxel_spacing: VoxelSpacing = (1.0, 1.0, 1.0),
        axes: str = "zyx",
    ) -> None:
        super().__init__(path)
        self._info = VolumeInfo(
            shape=tuple(int(size) for size in shape),
            dtype=str(np.dtype(dtype)),
            voxel_spacing=voxel_spacing,
            chunk_shape=None,
            axes=axes,
        )

    @property
    def info(self) -> VolumeInfo:
        return self._info

    def get_chunk(self, zyx_slices: Tuple[slice, slice, slice]) -> np.ndarray:
        region_shape = tuple(
            len(range(*zyx_slices[axis].indices(self._info.shape[axis])))
            for axis in range(3)
        )
        return np.zeros(region_shape, dtype=np.dtype(self._info.dtype))


def cast_float32_to_float16(loader: VolumeLoader) -> VolumeLoader:
    if np.dtype(loader.info.dtype) == np.dtype(np.float32):
        return Float16Loader(loader)
    return loader


def _normalize_path(path: str) -> str:
    return str(Path(path).expanduser())


def _split_qualified_path(path: str) -> Tuple[str, Optional[str]]:
    if "::" in path:
        base, qualifier = path.split("::", 1)
        return base, qualifier or None
    return path, None


def create_loader(path: str) -> VolumeLoader:
    path = _normalize_path(path)
    base, qualifier = _split_qualified_path(path)
    suffix = Path(base).suffix.lower()

    if suffix in {".npy", ".npz"}:
        from .npy_loader import NpyLoader

        return NpyLoader(path)
    if suffix in {".tif", ".tiff"}:
        from .tiff_loader import TiffLoader

        return TiffLoader(path)
    if suffix in {".zarr"} or Path(base).is_dir():
        from .zarr_loader import ZarrLoader

        return ZarrLoader(path)
    if suffix in {".h5", ".hdf5", ".hdf"}:
        from .hdf5_loader import Hdf5Loader

        return Hdf5Loader(path)

    raise ValueError(f"Unsupported volume format: {suffix or base}")
