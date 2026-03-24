from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

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
    if normalized_mode == "ram":
        loader = _prepare_loader_ram(loader, kind=kind)
        levels_builder = build_pyramid
    elif normalized_mode == "lazy":
        loader = _prepare_loader_lazy(loader, kind=kind)
        levels_builder = build_pyramid_lazy
    else:
        raise ValueError("load_mode must be 'ram' or 'lazy'")

    cache = ChunkCache(max_bytes=cache_max_bytes)
    volume = open_volume(loader, cache=cache)
    levels = levels_builder(volume, levels=pyramid_levels)
    return PreparedVolume(volume=volume, levels=levels, cache=cache)


def _prepare_loader_lazy(loader: VolumeLoader, *, kind: VolumeKind) -> VolumeLoader:
    if kind == "raw":
        return cast_float32_to_float16(loader)
    return loader


def _prepare_loader_ram(loader: VolumeLoader, *, kind: VolumeKind) -> VolumeLoader:
    source_info = loader.info
    try:
        array = np.asarray(loader.get_chunk((slice(None), slice(None), slice(None))))
    finally:
        loader.close()

    if kind == "raw":
        if np.dtype(array.dtype) == np.dtype(np.float32):
            array = array.astype(np.float16, copy=False)
            logger.info(
                "RAM mode: cast raw volume to float16 while materializing %s",
                loader.path,
            )
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

    return InMemoryVolumeLoader(
        path=loader.path,
        array=array,
        voxel_spacing=source_info.voxel_spacing,
        axes=source_info.axes,
    )


def _value_range(array: np.ndarray) -> Tuple[int, int]:
    if array.size == 0:
        return (0, 0)
    return (int(np.min(array)), int(np.max(array)))


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
