from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, Optional, Tuple

import numpy as np


AxisBounds = Tuple[int, int]
BoundsZYX = Tuple[AxisBounds, AxisBounds, AxisBounds]
ShapeZYX = Tuple[int, int, int]
SliceZYX = Tuple[slice, slice, slice]


@dataclass(frozen=True)
class ZeroOccupancyGrid:
    origin: ShapeZYX
    block_size: int
    occupied: np.ndarray


def build_zero_occupancy_grid(
    raw_volume: object,
    *,
    bounds: BoundsZYX,
    block_size: int,
) -> ZeroOccupancyGrid:
    normalized_bounds = _coerce_bounds(bounds)
    normalized_block_size = _coerce_positive_int(block_size, name="block_size")
    get_chunk = getattr(raw_volume, "get_chunk", None)
    if not callable(get_chunk):
        raise TypeError("raw_volume must define get_chunk(zyx_slices)")

    origin = tuple(int(axis_bounds[0]) for axis_bounds in normalized_bounds)
    region_shape = tuple(
        int(axis_bounds[1]) - int(axis_bounds[0]) for axis_bounds in normalized_bounds
    )
    grid_shape = tuple(_ceil_div(dim, normalized_block_size) for dim in region_shape)
    occupied = np.zeros(grid_shape, dtype=bool)

    scan_chunk_shape = _scan_chunk_shape_for_block_size(
        region_shape,
        chunk_shape=getattr(raw_volume, "chunk_shape", None),
        block_size=normalized_block_size,
    )

    for local_slices in _iter_local_chunk_slices(region_shape, scan_chunk_shape):
        raw_slices = tuple(
            slice(
                int(origin[axis]) + int(local_slices[axis].start),
                int(origin[axis]) + int(local_slices[axis].stop),
            )
            for axis in range(3)
        )
        chunk = np.asarray(get_chunk(raw_slices))
        if chunk.size == 0:
            continue
        reduced = _nonzero_block_any(chunk, normalized_block_size)
        cell_start = tuple(
            int(local_slices[axis].start) // normalized_block_size for axis in range(3)
        )
        cell_slices = tuple(
            slice(cell_start[axis], cell_start[axis] + reduced.shape[axis])
            for axis in range(3)
        )
        occupied[cell_slices] |= reduced

    return ZeroOccupancyGrid(
        origin=(int(origin[0]), int(origin[1]), int(origin[2])),
        block_size=int(normalized_block_size),
        occupied=occupied,
    )


def region_is_definitely_empty(
    grid: ZeroOccupancyGrid,
    *,
    raw_slices: SliceZYX,
) -> bool:
    cell_slices = []
    for axis in range(3):
        axis_start = int(raw_slices[axis].start)
        axis_stop = int(raw_slices[axis].stop)
        grid_origin = int(grid.origin[axis])
        grid_extent = grid_origin + int(grid.occupied.shape[axis]) * int(grid.block_size)
        if axis_stop > axis_start and (axis_start < grid_origin or axis_stop > grid_extent):
            return False
        start_cell = max(0, (axis_start - grid_origin) // int(grid.block_size))
        stop_cell = max(start_cell, _ceil_div(axis_stop - grid_origin, int(grid.block_size)))
        cell_slices.append(slice(start_cell, stop_cell))
    region = grid.occupied[tuple(cell_slices)]
    if region.size == 0:
        return True
    return not bool(region.any())


def _nonzero_block_any(chunk: np.ndarray, block_size: int) -> np.ndarray:
    if chunk.ndim != 3:
        raise ValueError(f"chunk must be 3D (z, y, x), got ndim={chunk.ndim}")
    nonzero = chunk != 0
    shape = nonzero.shape
    out_shape = tuple(_ceil_div(dim, block_size) for dim in shape)
    pad_width = tuple(
        (0, int(out_shape[axis]) * block_size - int(shape[axis])) for axis in range(3)
    )
    padded = nonzero if not any(after for _before, after in pad_width) else np.pad(
        nonzero, pad_width, mode="constant", constant_values=False
    )
    reshaped = padded.reshape(
        out_shape[0], block_size,
        out_shape[1], block_size,
        out_shape[2], block_size,
    )
    return reshaped.any(axis=(1, 3, 5))


def _default_scan_chunk_shape(shape: ShapeZYX) -> ShapeZYX:
    max_elements = 4_000_000
    base_edge = max(1, int(round(max_elements ** (1.0 / 3.0))))
    candidate = [max(1, min(int(dim), base_edge)) for dim in shape]
    while candidate[0] * candidate[1] * candidate[2] > max_elements:
        largest_axis = max(range(3), key=lambda axis: candidate[axis])
        if candidate[largest_axis] <= 1:
            break
        candidate[largest_axis] = max(1, candidate[largest_axis] // 2)
    return (candidate[0], candidate[1], candidate[2])


def _scan_chunk_shape_for_block_size(
    shape: ShapeZYX,
    *,
    chunk_shape: Optional[ShapeZYX],
    block_size: int,
) -> ShapeZYX:
    if (
        chunk_shape is not None
        and len(chunk_shape) == 3
        and all(int(dim) > 0 for dim in chunk_shape)
    ):
        base = tuple(int(dim) for dim in chunk_shape)
    else:
        base = _default_scan_chunk_shape(shape)
    return tuple(
        _ceil_to_multiple(max(int(dim), block_size), block_size) for dim in base
    )


def _iter_local_chunk_slices(
    shape: ShapeZYX,
    chunk_shape: ShapeZYX,
) -> Iterable[SliceZYX]:
    for z_start in range(0, shape[0], chunk_shape[0]):
        z_stop = min(z_start + chunk_shape[0], shape[0])
        for y_start in range(0, shape[1], chunk_shape[1]):
            y_stop = min(y_start + chunk_shape[1], shape[1])
            for x_start in range(0, shape[2], chunk_shape[2]):
                x_stop = min(x_start + chunk_shape[2], shape[2])
                yield (slice(z_start, z_stop), slice(y_start, y_stop), slice(x_start, x_stop))


def _coerce_bounds(bounds: BoundsZYX) -> BoundsZYX:
    if len(bounds) != 3:
        raise ValueError("bounds must contain exactly 3 axes (z, y, x)")
    normalized = []
    for axis, axis_bounds in enumerate(bounds):
        if len(axis_bounds) != 2:
            raise ValueError(f"bounds[{axis}] must contain exactly 2 values")
        start = int(axis_bounds[0])
        stop = int(axis_bounds[1])
        if stop <= start:
            raise ValueError(f"bounds[{axis}] must satisfy start < stop")
        normalized.append((start, stop))
    return (normalized[0], normalized[1], normalized[2])


def _coerce_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    normalized = int(value)
    if normalized < 1:
        raise ValueError(f"{name} must be >= 1")
    return normalized


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return _ceil_div(value, multiple) * multiple
