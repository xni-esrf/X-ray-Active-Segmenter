from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

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


class NonNegativeLoader(VolumeLoader):
    def __init__(self, loader: VolumeLoader) -> None:
        super().__init__(loader.path)
        self._loader = loader

    @property
    def info(self) -> VolumeInfo:
        return self._loader.info

    def get_chunk(self, zyx_slices: Tuple[slice, slice, slice]) -> np.ndarray:
        chunk = self._loader.get_chunk(zyx_slices)
        if np.issubdtype(chunk.dtype, np.signedinteger):
            return np.maximum(chunk, 0)
        return chunk

    def get_slice(self, axis: int, index: int) -> np.ndarray:
        chunk = self._loader.get_slice(axis, index)
        if np.issubdtype(chunk.dtype, np.signedinteger):
            return np.maximum(chunk, 0)
        return chunk

    def close(self) -> None:
        return self._loader.close()


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


class IntegerRangeCastLoader(VolumeLoader):
    def __init__(
        self,
        loader: VolumeLoader,
        *,
        target_dtype: np.dtype,
        value_range: Tuple[int, int],
    ) -> None:
        super().__init__(loader.path)
        self._loader = loader
        self._target_dtype = np.dtype(target_dtype)
        self._value_range = value_range
        src_info = loader.info
        self._info = VolumeInfo(
            shape=src_info.shape,
            dtype=str(self._target_dtype),
            voxel_spacing=src_info.voxel_spacing,
            chunk_shape=src_info.chunk_shape,
            axes=src_info.axes,
        )
        logger.info(
            "Applied integer range cast while reading volume: %s (%s -> %s, value range [%d, %d])",
            self.path,
            src_info.dtype,
            self._info.dtype,
            value_range[0],
            value_range[1],
        )

    @property
    def info(self) -> VolumeInfo:
        return self._info

    def get_chunk(self, zyx_slices: Tuple[slice, slice, slice]) -> np.ndarray:
        chunk = self._loader.get_chunk(zyx_slices)
        return chunk.astype(self._target_dtype, copy=False)

    def get_slice(self, axis: int, index: int) -> np.ndarray:
        chunk = self._loader.get_slice(axis, index)
        return chunk.astype(self._target_dtype, copy=False)

    def close(self) -> None:
        return self._loader.close()


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


def cast_float32_to_float16(loader: VolumeLoader) -> VolumeLoader:
    if np.dtype(loader.info.dtype) == np.dtype(np.float32):
        return Float16Loader(loader)
    return loader


def cast_float32_to_float16_eager(loader: VolumeLoader) -> VolumeLoader:
    source_dtype = np.dtype(loader.info.dtype)
    if source_dtype != np.dtype(np.float32):
        logger.info(
            "Skipping eager float32->float16 conversion for volume %s: dtype is already %s",
            loader.path,
            str(source_dtype),
        )
        return loader

    full_slices = (slice(None), slice(None), slice(None))
    source_info = loader.info
    try:
        full_array = np.asarray(loader.get_chunk(full_slices))
    finally:
        loader.close()
    cast_array = full_array.astype(np.float16, copy=False)
    logger.info(
        "Applied eager float32->float16 conversion while loading volume: %s",
        loader.path,
    )
    return InMemoryVolumeLoader(
        path=loader.path,
        array=cast_array,
        voxel_spacing=source_info.voxel_spacing,
        axes=source_info.axes,
    )


def cast_integer_to_smallest_dtype(loader: VolumeLoader) -> VolumeLoader:
    source_dtype = np.dtype(loader.info.dtype)
    if not np.issubdtype(source_dtype, np.integer):
        return loader

    value_range = _compute_integer_value_range(loader)
    target_dtype = _smallest_integer_dtype_for_range(*value_range)
    if target_dtype == source_dtype:
        return loader
    return IntegerRangeCastLoader(
        loader,
        target_dtype=target_dtype,
        value_range=value_range,
    )


def cast_integer_to_smallest_dtype_eager(loader: VolumeLoader) -> VolumeLoader:
    source_dtype = np.dtype(loader.info.dtype)
    if not np.issubdtype(source_dtype, np.integer):
        logger.info(
            "Skipping eager integer range cast for volume %s: dtype %s is not integer",
            loader.path,
            str(source_dtype),
        )
        return loader

    full_slices = (slice(None), slice(None), slice(None))
    source_info = loader.info
    full_array = np.asarray(loader.get_chunk(full_slices))

    if full_array.size == 0:
        min_value, max_value = 0, 0
    else:
        min_value = int(np.min(full_array))
        max_value = int(np.max(full_array))
    target_dtype = _smallest_integer_dtype_for_range(min_value, max_value)
    if target_dtype == source_dtype:
        logger.info(
            "Skipping eager integer range cast for volume %s: dtype %s already matches range [%d, %d]",
            loader.path,
            str(source_dtype),
            min_value,
            max_value,
        )
        return loader

    loader.close()
    cast_array = full_array.astype(target_dtype, copy=False)
    logger.info(
        "Applied eager integer range cast while loading volume: %s (%s -> %s, value range [%d, %d])",
        loader.path,
        source_info.dtype,
        str(np.dtype(target_dtype)),
        min_value,
        max_value,
    )
    return InMemoryVolumeLoader(
        path=loader.path,
        array=cast_array,
        voxel_spacing=source_info.voxel_spacing,
        axes=source_info.axes,
    )


def _compute_integer_value_range(loader: VolumeLoader) -> Tuple[int, int]:
    shape = loader.info.shape
    if any(dim <= 0 for dim in shape):
        return (0, 0)
    chunk_shape = _scan_chunk_shape(shape, loader.info.chunk_shape)
    min_value: Optional[int] = None
    max_value: Optional[int] = None
    for zyx_slices in _iter_chunk_slices(shape, chunk_shape):
        chunk = np.asarray(loader.get_chunk(zyx_slices))
        if chunk.size == 0:
            continue
        local_min = int(np.min(chunk))
        local_max = int(np.max(chunk))
        min_value = local_min if min_value is None else min(min_value, local_min)
        max_value = local_max if max_value is None else max(max_value, local_max)
    if min_value is None or max_value is None:
        return (0, 0)
    return (min_value, max_value)


def _smallest_integer_dtype_for_range(min_value: int, max_value: int) -> np.dtype:
    if min_value < 0:
        for candidate in (np.int8, np.int16, np.int32, np.int64):
            info = np.iinfo(candidate)
            if min_value >= info.min and max_value <= info.max:
                return np.dtype(candidate)
        return np.dtype(np.int64)
    for candidate in (np.uint8, np.uint16, np.uint32, np.uint64):
        info = np.iinfo(candidate)
        if min_value >= info.min and max_value <= info.max:
            return np.dtype(candidate)
    return np.dtype(np.uint64)


def _scan_chunk_shape(
    shape: Tuple[int, int, int],
    chunk_shape: Optional[Tuple[int, int, int]],
) -> Tuple[int, int, int]:
    if (
        chunk_shape is not None
        and len(chunk_shape) == 3
        and all(int(dim) > 0 for dim in chunk_shape)
    ):
        return (int(chunk_shape[0]), int(chunk_shape[1]), int(chunk_shape[2]))

    max_elements = 4_000_000
    base = max(1, int(round(max_elements ** (1.0 / 3.0))))
    candidate = [max(1, min(int(dim), base)) for dim in shape]
    while candidate[0] * candidate[1] * candidate[2] > max_elements:
        largest_axis = max(range(3), key=lambda axis: candidate[axis])
        if candidate[largest_axis] <= 1:
            break
        candidate[largest_axis] = max(1, candidate[largest_axis] // 2)
    return (candidate[0], candidate[1], candidate[2])


def _iter_chunk_slices(
    shape: Tuple[int, int, int],
    chunk_shape: Tuple[int, int, int],
) -> Iterable[Tuple[slice, slice, slice]]:
    for z_start in range(0, shape[0], chunk_shape[0]):
        z_stop = min(z_start + chunk_shape[0], shape[0])
        for y_start in range(0, shape[1], chunk_shape[1]):
            y_stop = min(y_start + chunk_shape[1], shape[1])
            for x_start in range(0, shape[2], chunk_shape[2]):
                x_stop = min(x_start + chunk_shape[2], shape[2])
                yield (slice(z_start, z_stop), slice(y_start, y_stop), slice(x_start, x_stop))


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


def slice_from_bounds(start: int, stop: int) -> slice:
    return slice(start, stop)


def to_zyx_slices(bounds: Iterable[Tuple[int, int]]) -> Tuple[slice, slice, slice]:
    bounds = list(bounds)
    if len(bounds) != 3:
        raise ValueError("bounds must have 3 (start, stop) pairs")
    return tuple(slice_from_bounds(start, stop) for start, stop in bounds)  # type: ignore[return-value]
