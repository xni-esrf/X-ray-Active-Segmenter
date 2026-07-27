from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .inference_geometry import (
    DEFAULT_INFERENCE_MINIVOL_SIZE,
    inference_stride_for_minivol_size,
)


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


# ---------------------------------------------------------------------------
# Streaming inference pre-pass: shared geometry, occupancy + intensity scan,
# and global normalization statistics.
#
# The streaming executor normalizes every minivolume with a single global
# ``(mean, std)`` computed once over "all voxels inside non-skipped
# minivolumes".  That domain is only known after the occupancy grid is built,
# so the scan accumulates per-block partial sums that are later restricted to
# the run domain.  The run rule (``minivol_is_run``) and grid-index math
# (``streaming_block_grid_bounds``) are shared with the geometry planner so the
# two phases are guaranteed to agree on which minivolumes run.
# ---------------------------------------------------------------------------


def minivol_is_run(
    occupancy: Optional[ZeroOccupancyGrid],
    *,
    start: ShapeZYX,
    minivol_size: int,
    stride: int,
    raw_shape: ShapeZYX,
) -> bool:
    """Whether the minivolume at ``start`` must be run under the skip rule.

    A minivolume is *skipped* only if it and its full 3x3x3 cell neighbourhood
    are empty, i.e. the dilated region ``[start - stride, start + minivol_size +
    stride)`` is definitely empty.  When ``occupancy`` is ``None`` every
    minivolume runs.
    """
    if occupancy is None:
        return True
    dilated = tuple(
        slice(
            max(0, int(start[axis]) - int(stride)),
            min(
                int(raw_shape[axis]),
                int(start[axis]) + int(minivol_size) + int(stride),
            ),
        )
        for axis in range(3)
    )
    return not region_is_definitely_empty(occupancy, raw_slices=dilated)


def streaming_block_grid_bounds(
    requested_bounds: BoundsZYX,
    *,
    stride: int,
) -> Tuple[ShapeZYX, ShapeZYX, ShapeZYX, ShapeZYX]:
    """Return ``(write_block_lo, write_block_hi, cell_lo, cell_hi)`` (inclusive).

    Write blocks are the stride blocks intersecting the bounding box.  The cell
    grid extends one block below (context/COLA on the low side; the top cell's
    extent already reaches one stride past the top block).  Indices are global
    (``cell_lo`` may be ``-1`` when the bbox touches voxel 0).
    """
    s = int(stride)
    write_lo = tuple(int(requested_bounds[axis][0]) // s for axis in range(3))
    write_hi = tuple((int(requested_bounds[axis][1]) - 1) // s for axis in range(3))
    cell_lo = tuple(int(write_lo[axis]) - 1 for axis in range(3))
    cell_hi = tuple(int(write_hi[axis]) for axis in range(3))
    return (
        (int(write_lo[0]), int(write_lo[1]), int(write_lo[2])),
        (int(write_hi[0]), int(write_hi[1]), int(write_hi[2])),
        (int(cell_lo[0]), int(cell_lo[1]), int(cell_lo[2])),
        (int(cell_hi[0]), int(cell_hi[1]), int(cell_hi[2])),
    )


def streaming_occupancy_scan_bounds(
    requested_bounds: BoundsZYX,
    *,
    raw_volume_shape: Sequence[object],
    stride: int,
    margin_blocks: int = 2,
) -> BoundsZYX:
    """Globally stride-aligned region to scan for occupancy + statistics.

    The bounding box is expanded by ``margin_blocks`` stride blocks on each side
    (default 2), which is exactly enough for every planner skip query -- which
    dilates one stride beyond a cell grid that itself extends one stride past the
    bbox -- to fall inside the scanned grid, so no skip decision is forced to be
    conservative.  The region is aligned to the global stride grid and clamped to
    the volume.
    """
    s = int(stride)
    bounds = _coerce_bounds(requested_bounds)
    raw_shape = _as_shape(raw_volume_shape)
    write_lo, write_hi, _, _ = streaming_block_grid_bounds(bounds, stride=s)
    n_blocks = tuple(_ceil_div(int(raw_shape[axis]), s) for axis in range(3))
    scan: List[AxisBounds] = []
    for axis in range(3):
        lo_block = max(0, int(write_lo[axis]) - int(margin_blocks))
        hi_block = min(int(n_blocks[axis]) - 1, int(write_hi[axis]) + int(margin_blocks))
        lo = lo_block * s
        hi = min(int(raw_shape[axis]), (hi_block + 1) * s)
        scan.append((int(lo), int(hi)))
    return (scan[0], scan[1], scan[2])


@dataclass(frozen=True)
class OccupancyStatsScan:
    """Result of a single occupancy + intensity read over the scan region."""

    grid: ZeroOccupancyGrid
    block_sum: np.ndarray  # float64, shape == grid.occupied.shape
    block_sq: np.ndarray  # float64, shape == grid.occupied.shape


def scan_occupancy_and_intensity(
    raw_volume: object,
    *,
    bounds: BoundsZYX,
    block_size: int,
) -> OccupancyStatsScan:
    """Single read building the occupancy grid and per-block intensity partials.

    ``block_sum[c]`` / ``block_sq[c]`` hold ``sum(x)`` / ``sum(x**2)`` (float64)
    over the voxels of block ``c``; empty blocks contribute zero.  Real voxel
    counts are derived analytically later so padding never distorts them.
    """
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
    block_sum = np.zeros(grid_shape, dtype=np.float64)
    block_sq = np.zeros(grid_shape, dtype=np.float64)

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
        chunk64 = chunk.astype(np.float64, copy=False)
        summed = _block_reduce_sum(chunk64, normalized_block_size)
        squared = _block_reduce_sum(chunk64 * chunk64, normalized_block_size)
        cell_start = tuple(
            int(local_slices[axis].start) // normalized_block_size for axis in range(3)
        )
        cell_slices = tuple(
            slice(cell_start[axis], cell_start[axis] + reduced.shape[axis])
            for axis in range(3)
        )
        occupied[cell_slices] |= reduced
        block_sum[cell_slices] += summed
        block_sq[cell_slices] += squared

    grid = ZeroOccupancyGrid(
        origin=(int(origin[0]), int(origin[1]), int(origin[2])),
        block_size=int(normalized_block_size),
        occupied=occupied,
    )
    return OccupancyStatsScan(grid=grid, block_sum=block_sum, block_sq=block_sq)


@dataclass(frozen=True)
class StreamingNormalizationStats:
    """Global normalization applied to every minivolume: ``(x - mean) / std``."""

    mean: float
    std: float
    voxel_count: int


def compute_streaming_normalization_from_sums(
    *,
    sum_x: float,
    sum_x2: float,
    voxel_count: int,
) -> StreamingNormalizationStats:
    n = int(voxel_count)
    if n <= 0:
        return StreamingNormalizationStats(mean=0.0, std=1.0, voxel_count=0)
    mean = float(sum_x) / float(n)
    variance = float(sum_x2) / float(n) - mean * mean
    if variance < 0.0:
        variance = 0.0
    std = float(np.sqrt(variance))
    if std == 0.0:
        std = 1.0
    return StreamingNormalizationStats(mean=mean, std=std, voxel_count=n)


def compute_streaming_normalization_stats(
    scan: OccupancyStatsScan,
    *,
    requested_bounds: BoundsZYX,
    raw_volume_shape: Sequence[object],
    minivol_size: int,
    stride: int,
) -> StreamingNormalizationStats:
    """Global ``(mean, std)`` over all real voxels inside non-skipped minivolumes.

    The run domain is the union of blocks belonging to any run minivolume
    (derived with the same rule as the planner).  ``sum(x)`` and ``sum(x**2)``
    are restricted to that domain; the voxel count includes the background zeros
    of run minivolumes (Choice 1) but only real, in-volume voxels (reflect-pad
    is excluded).
    """
    bounds = _coerce_bounds(requested_bounds)
    raw_shape = _as_shape(raw_volume_shape)
    normalized_minivol_size = _coerce_positive_int(minivol_size, name="minivol_size")
    normalized_stride = _coerce_positive_int(stride, name="stride")

    run_mask = _run_domain_block_mask(
        scan.grid,
        requested_bounds=bounds,
        raw_shape=raw_shape,
        minivol_size=normalized_minivol_size,
        stride=normalized_stride,
    )
    if not bool(run_mask.any()):
        return compute_streaming_normalization_from_sums(
            sum_x=0.0, sum_x2=0.0, voxel_count=0
        )
    sum_x = float(np.asarray(scan.block_sum)[run_mask].sum())
    sum_x2 = float(np.asarray(scan.block_sq)[run_mask].sum())
    counts = _block_voxel_counts(scan.grid, raw_shape=raw_shape)
    voxel_count = int(counts[run_mask].sum())
    return compute_streaming_normalization_from_sums(
        sum_x=sum_x, sum_x2=sum_x2, voxel_count=voxel_count
    )


@dataclass(frozen=True)
class StreamingPrePassResult:
    grid: ZeroOccupancyGrid
    normalization: StreamingNormalizationStats


def prepare_streaming_occupancy_and_stats(
    raw_volume: object,
    *,
    requested_bounds: BoundsZYX,
    raw_volume_shape: Sequence[object],
    minivol_size: int = DEFAULT_INFERENCE_MINIVOL_SIZE,
    stride: Optional[int] = None,
    margin_blocks: int = 2,
    skip_empty_regions: bool = True,
) -> StreamingPrePassResult:
    """Run the whole pre-pass: one read -> occupancy grid + global normalization.

    The returned ``grid`` feeds the geometry planner; ``normalization`` feeds the
    streaming reader.

    With ``skip_empty_regions`` (the default) background minivolumes are skipped
    and normalization spans only the occupied run domain.  When it is ``False``
    the scanned grid is marked fully occupied, so the planner runs every
    minivolume and normalization spans all voxels in the bbox.
    """
    normalized_minivol_size = _coerce_positive_int(minivol_size, name="minivol_size")
    normalized_stride = (
        inference_stride_for_minivol_size(normalized_minivol_size)
        if stride is None
        else _coerce_positive_int(stride, name="stride")
    )
    scan_bounds = streaming_occupancy_scan_bounds(
        requested_bounds,
        raw_volume_shape=raw_volume_shape,
        stride=normalized_stride,
        margin_blocks=margin_blocks,
    )
    scan = scan_occupancy_and_intensity(
        raw_volume,
        bounds=scan_bounds,
        block_size=normalized_stride,
    )
    if not skip_empty_regions:
        # No skipping: treat the whole scanned domain as occupied so the planner
        # runs every minivolume and the normalization spans all bbox voxels (not
        # just the occupied ones).  The intensity sums are unaffected.
        grid = replace(scan.grid, occupied=np.ones_like(scan.grid.occupied))
        scan = replace(scan, grid=grid)
    normalization = compute_streaming_normalization_stats(
        scan,
        requested_bounds=requested_bounds,
        raw_volume_shape=raw_volume_shape,
        minivol_size=normalized_minivol_size,
        stride=normalized_stride,
    )
    return StreamingPrePassResult(grid=scan.grid, normalization=normalization)


def _run_domain_block_mask(
    grid: ZeroOccupancyGrid,
    *,
    requested_bounds: BoundsZYX,
    raw_shape: ShapeZYX,
    minivol_size: int,
    stride: int,
) -> np.ndarray:
    s = int(stride)
    base_block = tuple(int(grid.origin[axis]) // s for axis in range(3))
    grid_shape = tuple(int(dim) for dim in grid.occupied.shape)
    _, _, cell_lo, cell_hi = streaming_block_grid_bounds(requested_bounds, stride=s)
    mask = np.zeros(grid_shape, dtype=bool)
    for cell_z in range(int(cell_lo[0]), int(cell_hi[0]) + 1):
        for cell_y in range(int(cell_lo[1]), int(cell_hi[1]) + 1):
            for cell_x in range(int(cell_lo[2]), int(cell_hi[2]) + 1):
                start = (cell_z * s, cell_y * s, cell_x * s)
                if not minivol_is_run(
                    grid,
                    start=start,
                    minivol_size=int(minivol_size),
                    stride=s,
                    raw_shape=raw_shape,
                ):
                    continue
                # The cell covers global blocks {cz, cz+1} x {cy, cy+1} x {cx, cx+1}.
                for block_z in (cell_z, cell_z + 1):
                    local_z = block_z - int(base_block[0])
                    if local_z < 0 or local_z >= grid_shape[0]:
                        continue
                    for block_y in (cell_y, cell_y + 1):
                        local_y = block_y - int(base_block[1])
                        if local_y < 0 or local_y >= grid_shape[1]:
                            continue
                        for block_x in (cell_x, cell_x + 1):
                            local_x = block_x - int(base_block[2])
                            if local_x < 0 or local_x >= grid_shape[2]:
                                continue
                            mask[local_z, local_y, local_x] = True
    return mask


def _block_voxel_counts(grid: ZeroOccupancyGrid, *, raw_shape: ShapeZYX) -> np.ndarray:
    s = int(grid.block_size)
    base_block = tuple(int(grid.origin[axis]) // s for axis in range(3))
    grid_shape = tuple(int(dim) for dim in grid.occupied.shape)
    per_axis: List[np.ndarray] = []
    for axis in range(3):
        counts = np.zeros(int(grid_shape[axis]), dtype=np.int64)
        for cell in range(int(grid_shape[axis])):
            global_block = int(base_block[axis]) + cell
            lo = max(0, global_block * s)
            hi = min(int(raw_shape[axis]), (global_block + 1) * s)
            counts[cell] = max(0, hi - lo)
        per_axis.append(counts)
    return (
        per_axis[0][:, None, None]
        * per_axis[1][None, :, None]
        * per_axis[2][None, None, :]
    )


def _block_reduce_sum(values: np.ndarray, block_size: int) -> np.ndarray:
    if values.ndim != 3:
        raise ValueError(f"values must be 3D (z, y, x), got ndim={values.ndim}")
    shape = values.shape
    out_shape = tuple(_ceil_div(dim, block_size) for dim in shape)
    pad_width = tuple(
        (0, int(out_shape[axis]) * block_size - int(shape[axis])) for axis in range(3)
    )
    padded = (
        values
        if not any(after for _before, after in pad_width)
        else np.pad(values, pad_width, mode="constant", constant_values=0.0)
    )
    reshaped = padded.reshape(
        out_shape[0], block_size,
        out_shape[1], block_size,
        out_shape[2], block_size,
    )
    return reshaped.sum(axis=(1, 3, 5))


def _as_shape(values: Sequence[object]) -> ShapeZYX:
    if len(values) != 3:
        raise ValueError("shape must contain exactly 3 values (z, y, x)")
    return (
        _coerce_positive_int(values[0], name="shape[0]"),
        _coerce_positive_int(values[1], name="shape[1]"),
        _coerce_positive_int(values[2], name="shape[2]"),
    )
