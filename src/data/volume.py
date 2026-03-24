from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..io.loader import VolumeInfo, VolumeLoader
from .chunk_cache import ChunkCache


@dataclass(frozen=True)
class VolumeData:
    loader: VolumeLoader
    info: VolumeInfo
    cache: Optional[ChunkCache] = None

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.info.shape

    @property
    def dtype(self) -> str:
        return self.info.dtype

    @property
    def voxel_spacing(self) -> Tuple[float, float, float]:
        return self.info.voxel_spacing

    @property
    def chunk_shape(self) -> Optional[Tuple[int, int, int]]:
        return self.info.chunk_shape

    def cache_key(
        self, zyx_slices: Tuple[slice, slice, slice]
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]:
        zyx_shape = self.shape
        z_slice, y_slice, x_slice = zyx_slices
        z_key = z_slice.indices(zyx_shape[0])
        y_key = y_slice.indices(zyx_shape[1])
        x_key = x_slice.indices(zyx_shape[2])
        return (z_key, y_key, x_key)

    def get_chunk(self, zyx_slices: Tuple[slice, slice, slice]) -> np.ndarray:
        key = self.cache_key(zyx_slices)
        if self.cache is None:
            return self.loader.get_chunk(self._slices_from_key(key))

        if self._can_use_chunk_cache(key):
            return self._get_chunk_chunkaware(key)

        cached = self.cache.get(key)
        if cached is not None:
            return cached
        chunk = self.loader.get_chunk(self._slices_from_key(key))
        self.cache.set(key, chunk)
        return chunk

    def get_slice(self, axis: int, index: int) -> np.ndarray:
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2")
        slices = [slice(None), slice(None), slice(None)]
        slices[axis] = slice(index, index + 1)
        return self.get_chunk(tuple(slices)).squeeze(axis=axis)

    def close(self) -> None:
        self.loader.close()

    def _slices_from_key(
        self, key: Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]
    ) -> Tuple[slice, slice, slice]:
        return (
            slice(key[0][0], key[0][1], key[0][2]),
            slice(key[1][0], key[1][1], key[1][2]),
            slice(key[2][0], key[2][1], key[2][2]),
        )

    def _can_use_chunk_cache(
        self, key: Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]
    ) -> bool:
        chunk_shape = self.chunk_shape
        if chunk_shape is None:
            return False
        if len(chunk_shape) != 3:
            return False
        if any(size <= 0 for size in chunk_shape):
            return False
        return all(step == 1 for _start, _stop, step in key)

    def _chunk_cache_key(self, coord: Tuple[int, int, int]) -> Tuple[str, int, int, int]:
        return ("chunk", coord[0], coord[1], coord[2])

    def _chunk_slices(
        self, coord: Tuple[int, int, int], chunk_shape: Tuple[int, int, int]
    ) -> Tuple[slice, slice, slice]:
        z0 = coord[0] * chunk_shape[0]
        y0 = coord[1] * chunk_shape[1]
        x0 = coord[2] * chunk_shape[2]
        z1 = min(z0 + chunk_shape[0], self.shape[0])
        y1 = min(y0 + chunk_shape[1], self.shape[1])
        x1 = min(x0 + chunk_shape[2], self.shape[2])
        return (slice(z0, z1), slice(y0, y1), slice(x0, x1))

    def _load_cached_chunk(
        self, coord: Tuple[int, int, int], chunk_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        if self.cache is None:
            raise RuntimeError("Chunk cache requested but cache is not configured")
        cache_key = self._chunk_cache_key(coord)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        chunk = self.loader.get_chunk(self._chunk_slices(coord, chunk_shape))
        self.cache.set(cache_key, chunk)
        return chunk

    def _get_chunk_chunkaware(
        self, key: Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]
    ) -> np.ndarray:
        chunk_shape = self.chunk_shape
        if chunk_shape is None:
            raise RuntimeError("Chunk-aware cache requested without chunk shape")

        z_start, z_stop, _ = key[0]
        y_start, y_stop, _ = key[1]
        x_start, x_stop, _ = key[2]
        out_shape = (
            max(0, z_stop - z_start),
            max(0, y_stop - y_start),
            max(0, x_stop - x_start),
        )
        if 0 in out_shape:
            return np.empty(out_shape, dtype=np.dtype(self.dtype))

        result = np.empty(out_shape, dtype=np.dtype(self.dtype))
        z_coords = range(z_start // chunk_shape[0], (z_stop - 1) // chunk_shape[0] + 1)
        y_coords = range(y_start // chunk_shape[1], (y_stop - 1) // chunk_shape[1] + 1)
        x_coords = range(x_start // chunk_shape[2], (x_stop - 1) // chunk_shape[2] + 1)

        for zc in z_coords:
            for yc in y_coords:
                for xc in x_coords:
                    coord = (zc, yc, xc)
                    chunk = self._load_cached_chunk(coord, chunk_shape)
                    chunk_z0 = zc * chunk_shape[0]
                    chunk_y0 = yc * chunk_shape[1]
                    chunk_x0 = xc * chunk_shape[2]
                    chunk_z1 = min(chunk_z0 + chunk_shape[0], self.shape[0])
                    chunk_y1 = min(chunk_y0 + chunk_shape[1], self.shape[1])
                    chunk_x1 = min(chunk_x0 + chunk_shape[2], self.shape[2])

                    inter_z0 = max(z_start, chunk_z0)
                    inter_y0 = max(y_start, chunk_y0)
                    inter_x0 = max(x_start, chunk_x0)
                    inter_z1 = min(z_stop, chunk_z1)
                    inter_y1 = min(y_stop, chunk_y1)
                    inter_x1 = min(x_stop, chunk_x1)
                    if inter_z0 >= inter_z1 or inter_y0 >= inter_y1 or inter_x0 >= inter_x1:
                        continue

                    src = chunk[
                        inter_z0 - chunk_z0 : inter_z1 - chunk_z0,
                        inter_y0 - chunk_y0 : inter_y1 - chunk_y0,
                        inter_x0 - chunk_x0 : inter_x1 - chunk_x0,
                    ]
                    result[
                        inter_z0 - z_start : inter_z1 - z_start,
                        inter_y0 - y_start : inter_y1 - y_start,
                        inter_x0 - x_start : inter_x1 - x_start,
                    ] = src
        return result


def open_volume(loader: VolumeLoader, *, cache: Optional[ChunkCache] = None) -> VolumeData:
    return VolumeData(loader=loader, info=loader.info, cache=cache)
