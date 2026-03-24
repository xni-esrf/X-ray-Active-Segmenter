from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..io.loader import VolumeInfo, VolumeLoader
from .volume import VolumeData, open_volume


class InMemoryLoader(VolumeLoader):
    def __init__(
        self,
        array: np.ndarray,
        *,
        path: str,
        voxel_spacing: Tuple[float, float, float],
    ) -> None:
        super().__init__(path)
        self._array = np.asarray(array)
        if self._array.ndim != 3:
            raise ValueError("In-memory volume must be 3D (z, y, x)")
        self._info = VolumeInfo(
            shape=tuple(self._array.shape),
            dtype=str(self._array.dtype),
            voxel_spacing=voxel_spacing,
            chunk_shape=None,
            axes="zyx",
        )

    @property
    def info(self) -> VolumeInfo:
        return self._info

    def get_chunk(self, zyx_slices: Tuple[slice, slice, slice]) -> np.ndarray:
        return np.asarray(self._array[zyx_slices])


class DownsampledLoader(VolumeLoader):
    def __init__(self, source: VolumeData, *, level: int) -> None:
        if level < 1:
            raise ValueError("DownsampledLoader level must be >= 1")
        super().__init__(f"{source.loader.path}::lazy_pyr/{level}")
        self._source = source
        self._level = level
        self._factor = 1 << level
        source_shape = source.info.shape
        source_spacing = source.info.voxel_spacing
        self._info = VolumeInfo(
            shape=tuple((dim + self._factor - 1) // self._factor for dim in source_shape),
            dtype=source.info.dtype,
            voxel_spacing=tuple(spacing * self._factor for spacing in source_spacing),
            chunk_shape=self._downsample_chunk_shape(source.info.chunk_shape),
            axes=source.info.axes,
        )

    @property
    def info(self) -> VolumeInfo:
        return self._info

    def get_chunk(self, zyx_slices: Tuple[slice, slice, slice]) -> np.ndarray:
        down_shape = self._info.shape
        mapped = tuple(
            self._map_slice(slc, down_dim)
            for slc, down_dim in zip(zyx_slices, down_shape)
        )
        return np.asarray(self._source.get_chunk(mapped))

    def _map_slice(self, slc: slice, down_dim: int) -> slice:
        start, stop, step = slc.indices(down_dim)
        return slice(start * self._factor, stop * self._factor, step * self._factor)

    def _downsample_chunk_shape(
        self,
        chunk_shape: Optional[Tuple[int, int, int]],
    ) -> Optional[Tuple[int, int, int]]:
        if chunk_shape is None or len(chunk_shape) != 3:
            return None
        reduced = tuple(max(1, int(np.ceil(dim / self._factor))) for dim in chunk_shape)
        return reduced


class SegmentationDownsampledLoader(VolumeLoader):
    def __init__(self, source: VolumeData, *, level: int) -> None:
        if level < 1:
            raise ValueError("SegmentationDownsampledLoader level must be >= 1")
        super().__init__(f"{source.loader.path}::lazy_seg_pyr/{level}")
        self._source = source
        self._level = level
        self._factor = 1 << level
        source_shape = source.info.shape
        source_spacing = source.info.voxel_spacing
        self._info = VolumeInfo(
            shape=tuple((dim + self._factor - 1) // self._factor for dim in source_shape),
            dtype=source.info.dtype,
            voxel_spacing=tuple(spacing * self._factor for spacing in source_spacing),
            chunk_shape=self._downsample_chunk_shape(source.info.chunk_shape),
            axes=source.info.axes,
        )

    @property
    def info(self) -> VolumeInfo:
        return self._info

    def get_chunk(self, zyx_slices: Tuple[slice, slice, slice]) -> np.ndarray:
        down_shape = self._info.shape
        normalized = [slc.indices(dim) for slc, dim in zip(zyx_slices, down_shape)]
        unit_slices = tuple(slice(start, stop, 1) for start, stop, _step in normalized)
        unit_chunk = self._get_chunk_unit_step(unit_slices)
        steps = tuple(int(step) for _start, _stop, step in normalized)
        if steps == (1, 1, 1):
            return unit_chunk
        return np.asarray(unit_chunk[:: steps[0], :: steps[1], :: steps[2]])

    def _get_chunk_unit_step(self, zyx_slices: Tuple[slice, slice, slice]) -> np.ndarray:
        z_start, z_stop, _ = zyx_slices[0].indices(self._info.shape[0])
        y_start, y_stop, _ = zyx_slices[1].indices(self._info.shape[1])
        x_start, x_stop, _ = zyx_slices[2].indices(self._info.shape[2])
        out_shape = (
            max(0, z_stop - z_start),
            max(0, y_stop - y_start),
            max(0, x_stop - x_start),
        )
        if 0 in out_shape:
            return np.zeros(out_shape, dtype=np.dtype(self._info.dtype))

        source_shape = self._source.shape
        source_slices = (
            slice(z_start * self._factor, min(z_stop * self._factor, source_shape[0])),
            slice(y_start * self._factor, min(y_stop * self._factor, source_shape[1])),
            slice(x_start * self._factor, min(x_stop * self._factor, source_shape[2])),
        )
        chunk = np.asarray(self._source.get_chunk(source_slices))

        reduced = chunk
        for _ in range(self._level):
            reduced = _downsample_labels_by_2(reduced)

        if reduced.shape == out_shape:
            return reduced

        result = np.zeros(out_shape, dtype=np.dtype(self._info.dtype))
        copy_shape = (
            min(out_shape[0], reduced.shape[0]),
            min(out_shape[1], reduced.shape[1]),
            min(out_shape[2], reduced.shape[2]),
        )
        result[: copy_shape[0], : copy_shape[1], : copy_shape[2]] = reduced[
            : copy_shape[0],
            : copy_shape[1],
            : copy_shape[2],
        ]
        return result

    def _downsample_chunk_shape(
        self,
        chunk_shape: Optional[Tuple[int, int, int]],
    ) -> Optional[Tuple[int, int, int]]:
        if chunk_shape is None or len(chunk_shape) != 3:
            return None
        reduced = tuple(max(1, int(np.ceil(dim / self._factor))) for dim in chunk_shape)
        return reduced


def _downsample_labels_by_2(array: np.ndarray) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError("Array must be 3D (z, y, x)")
    out_shape = tuple((dim + 1) // 2 for dim in array.shape)
    out = np.zeros(out_shape, dtype=array.dtype)
    for z_offset in (0, 1):
        z_view = array[z_offset::2, :, :]
        for y_offset in (0, 1):
            zy_view = z_view[:, y_offset::2, :]
            for x_offset in (0, 1):
                block_view = zy_view[:, :, x_offset::2]
                if block_view.size == 0:
                    continue
                z_size = min(out_shape[0], block_view.shape[0])
                y_size = min(out_shape[1], block_view.shape[1])
                x_size = min(out_shape[2], block_view.shape[2])
                np.maximum(
                    out[:z_size, :y_size, :x_size],
                    block_view[:z_size, :y_size, :x_size],
                    out=out[:z_size, :y_size, :x_size],
                )
    return out


def _downsample_by_2(array: np.ndarray) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError("Array must be 3D (z, y, x)")
    return np.ascontiguousarray(array[::2, ::2, ::2])


def build_pyramid(volume: VolumeData, levels: int = 4) -> Tuple[VolumeData, ...]:
    if levels < 1:
        raise ValueError("levels must be >= 1")

    pyramid = [volume]
    if levels == 1:
        return tuple(pyramid)

    current = np.asarray(volume.loader.get_chunk((slice(None), slice(None), slice(None))))
    current_spacing = volume.info.voxel_spacing
    for level in range(1, levels):
        if min(current.shape) <= 1:
            break
        current = _downsample_by_2(current)
        current_spacing = (
            current_spacing[0] * 2.0,
            current_spacing[1] * 2.0,
            current_spacing[2] * 2.0,
        )
        loader = InMemoryLoader(
            current,
            path=f"{volume.loader.path}::pyr/{level}",
            voxel_spacing=current_spacing,
        )
        pyramid.append(open_volume(loader, cache=None))
    return tuple(pyramid)


def build_pyramid_lazy(volume: VolumeData, levels: int = 4) -> Tuple[VolumeData, ...]:
    if levels < 1:
        raise ValueError("levels must be >= 1")

    pyramid = [volume]
    base_shape = volume.info.shape
    for level in range(1, levels):
        factor = 1 << level
        level_shape = tuple((dim + factor - 1) // factor for dim in base_shape)
        if min(level_shape) <= 0:
            break
        if tuple(level_shape) == tuple(pyramid[-1].shape):
            break
        loader = DownsampledLoader(volume, level=level)
        pyramid.append(open_volume(loader, cache=None))
        if min(level_shape) <= 1:
            break
    return tuple(pyramid)


def build_segmentation_pyramid_lazy(volume: VolumeData, levels: int = 4) -> Tuple[VolumeData, ...]:
    if levels < 1:
        raise ValueError("levels must be >= 1")

    pyramid = [volume]
    base_shape = volume.info.shape
    for level in range(1, levels):
        factor = 1 << level
        level_shape = tuple((dim + factor - 1) // factor for dim in base_shape)
        if min(level_shape) <= 0:
            break
        if tuple(level_shape) == tuple(pyramid[-1].shape):
            break
        loader = SegmentationDownsampledLoader(volume, level=level)
        pyramid.append(open_volume(loader, cache=None))
        if min(level_shape) <= 1:
            break
    return tuple(pyramid)
