from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Tuple

import logging
import numpy as np

from .data import ChunkCache, VolumeData, build_pyramid, build_pyramid_lazy, open_volume
from .io import cast_float32_to_float16, create_loader
from .io.loader import InMemoryVolumeLoader, VolumeLoader


logger = logging.getLogger(__name__)

LoadMode = Literal["ram", "lazy"]
VolumeKind = Literal["raw", "semantic", "instance"]


@dataclass(frozen=True)
class PreparedVolume:
    volume: VolumeData
    levels: Tuple[VolumeData, ...]
    cache: ChunkCache


@dataclass(frozen=True)
class _PreparedLoader:
    loader: VolumeLoader
    data_range: Optional[Tuple[float, float]] = None


def load_prepared_volume(
    path: str,
    *,
    kind: VolumeKind,
    load_mode: str,
    cache_max_bytes: int,
    pyramid_levels: int = 4,
) -> PreparedVolume:
    loader = create_loader(path)
    normalized_mode = str(load_mode).strip().lower()
    data_range: Optional[Tuple[float, float]] = None
    if normalized_mode == "ram":
        prepared_loader = _prepare_loader_ram(loader, kind=kind)
        loader = prepared_loader.loader
        data_range = prepared_loader.data_range
        levels_builder = build_pyramid
    elif normalized_mode == "lazy":
        loader = _prepare_loader_lazy(loader, kind=kind)
        levels_builder = build_pyramid_lazy
    else:
        raise ValueError("load_mode must be 'ram' or 'lazy'")

    if kind == "raw" and data_range is None:
        data_range = _compute_raw_data_range(loader)
    cache = ChunkCache(max_bytes=cache_max_bytes)
    volume = open_volume(loader, cache=cache, data_range=data_range)
    levels = levels_builder(volume, levels=pyramid_levels)
    return PreparedVolume(volume=volume, levels=levels, cache=cache)


def _prepare_loader_lazy(loader: VolumeLoader, *, kind: VolumeKind) -> VolumeLoader:
    if kind == "raw":
        return cast_float32_to_float16(loader)
    return loader


def _prepare_loader_ram(loader: VolumeLoader, *, kind: VolumeKind) -> _PreparedLoader:
    source_info = loader.info
    try:
        # RAM mode should detach from mmap/lazy backing storage.  A plain
        # np.asarray(...) can keep a memmap-backed view for TIFFs, causing later
        # startup work to fault pages from storage while the UI is still hidden.
        array = np.array(
            loader.get_chunk((slice(None), slice(None), slice(None))),
            copy=True,
        )
    finally:
        loader.close()

    data_range: Optional[Tuple[float, float]] = None
    if kind == "raw":
        if np.dtype(array.dtype) == np.dtype(np.float32):
            array = array.astype(np.float16, copy=False)
            logger.info(
                "RAM mode: cast raw volume to float16 while materializing %s",
                loader.path,
            )
        data_range = _compute_raw_array_data_range(array)
    elif kind in ("semantic", "instance"):
        if np.issubdtype(array.dtype, np.integer):
            min_value, max_value = _value_range(array)
            target_dtype = _smallest_integer_dtype_for_range(min_value, max_value)
            if np.dtype(array.dtype) != target_dtype:
                array = array.astype(target_dtype, copy=False)
                logger.info(
                    "RAM mode: cast %s map %s from %s to %s for value range [%d, %d]",
                    kind,
                    loader.path,
                    source_info.dtype,
                    str(target_dtype),
                    min_value,
                    max_value,
                )

    return _PreparedLoader(
        loader=InMemoryVolumeLoader(
            path=loader.path,
            array=array,
            voxel_spacing=source_info.voxel_spacing,
            axes=source_info.axes,
        ),
        data_range=data_range,
    )


def _value_range(array: np.ndarray) -> Tuple[int, int]:
    if array.size == 0:
        return (0, 0)
    return (int(np.min(array)), int(np.max(array)))


def _compute_raw_data_range(loader: VolumeLoader) -> Tuple[float, float]:
    shape = tuple(int(dim) for dim in loader.info.shape)
    if len(shape) != 3 or any(dim <= 0 for dim in shape):
        raise ValueError(
            "Raw volume must have a strictly positive 3D shape (z, y, x), "
            f"got {shape}."
        )

    declared_dtype = np.dtype(loader.info.dtype)
    if np.issubdtype(declared_dtype, np.complexfloating):
        raise ValueError("Complex-valued raw volumes are not supported for rendering.")

    chunk_shape = _scan_chunk_shape(shape, loader.info.chunk_shape)
    global_min: Optional[float] = None
    global_max: Optional[float] = None
    for zyx_slices in _iter_chunk_slices(shape, chunk_shape):
        chunk = np.asarray(loader.get_chunk(zyx_slices))
        if chunk.size == 0:
            continue
        chunk_dtype = np.dtype(chunk.dtype)
        if np.issubdtype(chunk_dtype, np.complexfloating):
            raise ValueError("Complex-valued raw volumes are not supported for rendering.")
        if np.issubdtype(chunk_dtype, np.floating) and not np.all(np.isfinite(chunk)):
            raise ValueError("Raw volume contains NaN or Inf values and cannot be displayed.")
        local_min = float(np.min(chunk))
        local_max = float(np.max(chunk))
        if global_min is None or local_min < global_min:
            global_min = local_min
        if global_max is None or local_max > global_max:
            global_max = local_max

    if global_min is None or global_max is None:
        raise ValueError("Raw volume has no voxels to render.")
    return (global_min, global_max)


def _compute_raw_array_data_range(array: np.ndarray) -> Tuple[float, float]:
    shape = tuple(int(dim) for dim in array.shape)
    if len(shape) != 3 or any(dim <= 0 for dim in shape):
        raise ValueError(
            "Raw volume must have a strictly positive 3D shape (z, y, x), "
            f"got {shape}."
        )

    dtype = np.dtype(array.dtype)
    if np.issubdtype(dtype, np.complexfloating):
        raise ValueError("Complex-valued raw volumes are not supported for rendering.")
    if np.issubdtype(dtype, np.floating) and not np.all(np.isfinite(array)):
        raise ValueError("Raw volume contains NaN or Inf values and cannot be displayed.")
    if array.size == 0:
        raise ValueError("Raw volume has no voxels to render.")
    return (float(np.min(array)), float(np.max(array)))


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
