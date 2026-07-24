"""Pure geometry planner for streaming (minivolume-by-minivolume) inference.

This module turns a single inference bounding box into a fully specified,
raster-ordered schedule that the streaming executor (Phase 3) can replay
without re-deriving any geometry.  It performs no I/O and holds no model or
volume data -- it only manipulates integer coordinates and an optional
occupancy grid.

Coordinate model
----------------
* Everything is expressed in *global* voxel coordinates (origin at the raw
  volume origin, not the bounding box).
* ``stride`` (``s``) is the minivolume hop.  We require
  ``minivol_size == 2 * stride`` (50% overlap), which is the condition under
  which the Hann window used by the executor forms a constant overlap-add
  (COLA) reconstruction: every interior output voxel is covered by exactly two
  minivolumes per axis.
* A *block* is a ``stride``-sized cube aligned to the global stride grid.
  ``block (i, j, k)`` spans ``[s*i, s*i + s)`` per axis.  Blocks tile space and
  never overlap.
* A *minivolume cell* ``(i, j, k)`` starts at ``(s*i, s*j, s*k)`` and spans
  ``[s*i, s*i + 2s)`` per axis, i.e. it is exactly the ``2 x 2 x 2`` block
  neighbourhood ``{i, i+1} x {j, j+1} x {k, k+1}``.
* A *chunk* is the Zarr output chunk, ``chunk_stride_multiple`` (``k``) blocks
  per axis, i.e. ``k * stride`` voxels, aligned to the global chunk grid.  With
  the default ``k = 2`` each chunk is ``minivol_size`` voxels per axis and holds
  up to ``2 x 2 x 2`` blocks.

Finalization rule
-----------------
Output block ``B(i, j, k)`` is contributed to by the (up to) eight cells in the
neighbourhood ``{i-1, i} x {j-1, j} x {k-1, k}``.  Processed in raster order
(scan axis outermost), the last of those cells is ``(i, j, k)`` itself, so
processing cell ``(i, j, k)`` *finalizes* exactly the block ``B(i, j, k)`` --
its lower-corner block.  This gives a deterministic, counter-free flush
schedule: after each cell, argmax and emit one block; once every block of a
chunk is finalized, that chunk is complete and is written exactly once.

Skip rule
---------
When an occupancy grid is supplied, a cell is *skipped* only if the cell and its
full ``3 x 3 x 3`` cell neighbourhood are empty (equivalently: the region
``[start - stride, start + minivol_size + stride)`` is definitely empty).  This
guarantees no output voxel ever mixes a live-foreground contribution with a
skipped one -- if any cell covering a block is non-empty, none of that block's
covering cells are skipped -- so skipping never biases a boundary voxel.  Empty
regions still surrounded by foreground are *run* (the model sees all-zeros and
predicts background); only genuinely isolated empty regions are skipped.

Edge handling
-------------
The cell grid is extended one stride past the bounding box on each side so that
every written voxel sits in the COLA-valid interior.  Cells that reach outside
the real volume carry a reflect-pad plan (read real neighbouring data where it
exists, reflect-pad only at true volume edges).
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Dict, List, Optional, Sequence, Tuple

from .inference_geometry import (
    DEFAULT_INFERENCE_MINIVOL_SIZE,
    inference_stride_for_minivol_size,
)
from .zero_occupancy import (
    ZeroOccupancyGrid,
    minivol_is_run,
    streaming_block_grid_bounds,
)


AxisBounds = Tuple[int, int]
BoundsZYX = Tuple[AxisBounds, AxisBounds, AxisBounds]
ShapeZYX = Tuple[int, int, int]
SliceZYX = Tuple[slice, slice, slice]
GridIndex = Tuple[int, int, int]

DEFAULT_CHUNK_STRIDE_MULTIPLE = 2


@dataclass(frozen=True)
class MinivolExtractionPlan:
    """How to read one minivolume from the raw volume.

    Read ``raw_volume[raw_slices]`` then ``np.pad`` with ``(pad_before,
    pad_after)`` in reflect mode to obtain a ``minivol_size`` cube.  Non-zero
    padding only ever occurs at true volume edges.
    """

    raw_slices: SliceZYX
    pad_before: ShapeZYX
    pad_after: ShapeZYX

    @property
    def has_padding(self) -> bool:
        return any(
            int(before) > 0 or int(after) > 0
            for before, after in zip(self.pad_before, self.pad_after)
        )


@dataclass(frozen=True)
class MinivolCell:
    """One minivolume in the raster-ordered processing schedule."""

    index: int
    grid_index: GridIndex
    start: GridIndex
    run: bool
    extraction: MinivolExtractionPlan
    # Block finalized right after this cell is processed (``None`` for context
    # cells whose lower-corner block lies outside the bounding box).
    finalizes_block_id: Optional[int]
    # Chunks that become complete right after this cell is processed.
    completes_chunk_ids: Tuple[int, ...]


@dataclass(frozen=True)
class OutputBlock:
    """A ``stride``-sized output block that intersects the bounding box."""

    id: int
    grid_index: GridIndex
    # Voxels actually written = block extent intersected with the bounding box,
    # in global coordinates.
    write_slices: SliceZYX
    chunk_id: int
    # ``write_slices`` expressed relative to the (volume-clamped) chunk origin.
    slices_in_chunk: SliceZYX
    finalizing_cell_index: int
    # False => every covering cell was skipped, so the block is pure background
    # (writer fill value); the executor need not accumulate or write it.
    any_covering_run: bool


@dataclass(frozen=True)
class OutputChunk:
    """A Zarr output chunk, written exactly once when complete."""

    id: int
    chunk_index: GridIndex
    # Global voxel extent, clamped to the volume (edge chunks are partial).
    chunk_slices: SliceZYX
    block_ids: Tuple[int, ...]
    # After this (raster-ordered) cell is processed, all of the chunk's blocks
    # are finalized and the chunk can be written.
    completion_cell_index: int
    # False => none of the chunk's blocks receive data; it stays fill and need
    # not be written at all.
    has_data: bool


@dataclass(frozen=True)
class StreamingInferencePlan:
    requested_bounds: BoundsZYX
    raw_volume_shape: ShapeZYX
    minivol_size: int
    stride: int
    chunk_size: int
    scan_axis: int
    cells: Tuple[MinivolCell, ...]
    blocks: Tuple[OutputBlock, ...]
    chunks: Tuple[OutputChunk, ...]

    @property
    def total_cell_count(self) -> int:
        return len(self.cells)

    @property
    def run_cell_count(self) -> int:
        return sum(1 for cell in self.cells if cell.run)

    @property
    def skipped_cell_count(self) -> int:
        return sum(1 for cell in self.cells if not cell.run)

    @property
    def write_block_count(self) -> int:
        return len(self.blocks)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def written_chunk_count(self) -> int:
        return sum(1 for chunk in self.chunks if chunk.has_data)


def build_streaming_inference_plan(
    *,
    requested_bounds: BoundsZYX,
    raw_volume_shape: Sequence[object],
    minivol_size: int = DEFAULT_INFERENCE_MINIVOL_SIZE,
    stride: Optional[int] = None,
    occupancy: Optional[ZeroOccupancyGrid] = None,
    chunk_stride_multiple: int = DEFAULT_CHUNK_STRIDE_MULTIPLE,
) -> StreamingInferencePlan:
    """Build the raster-ordered streaming inference schedule for one bbox.

    ``occupancy`` (if given) drives the skip decisions; when ``None`` every cell
    runs.  The returned plan is self-contained: iterate ``plan.cells`` in order,
    run the model on ``cell.run`` cells, and after each cell finalize
    ``cell.finalizes_block_id`` and flush ``cell.completes_chunk_ids``.
    """
    raw_shape = _coerce_shape(raw_volume_shape, name="raw_volume_shape")
    bounds = _coerce_bounds(requested_bounds, raw_volume_shape=raw_shape)
    normalized_minivol_size = _coerce_positive_int(minivol_size, name="minivol_size")
    normalized_stride = (
        inference_stride_for_minivol_size(normalized_minivol_size)
        if stride is None
        else _coerce_positive_int(stride, name="stride")
    )
    if normalized_minivol_size != 2 * normalized_stride:
        raise ValueError(
            "streaming inference requires minivol_size == 2 * stride (50% overlap "
            f"for Hann COLA); got minivol_size={normalized_minivol_size}, "
            f"stride={normalized_stride}"
        )
    normalized_chunk_multiple = _coerce_positive_int(
        chunk_stride_multiple, name="chunk_stride_multiple"
    )
    chunk_size = int(normalized_chunk_multiple) * int(normalized_stride)

    stride_value = int(normalized_stride)
    minivol_value = int(normalized_minivol_size)

    # Per-axis block index range covered by the bounding box, and the cell grid
    # extended one block below (context/COLA on the low side; the top cell's
    # extent already reaches one stride past the top block).  Shared with the
    # pre-pass so both phases agree on the grid.
    write_block_lo, write_block_hi, cell_lo, cell_hi = streaming_block_grid_bounds(
        bounds, stride=stride_value
    )

    scan_axis = _select_scan_axis(bounds)

    # --- cells (raster ordered) -------------------------------------------
    ordered_grid_indices = tuple(
        _raster_ordered_grid_indices(cell_lo, cell_hi, scan_axis=scan_axis)
    )
    cell_index_by_grid: Dict[GridIndex, int] = {
        grid_index: index for index, grid_index in enumerate(ordered_grid_indices)
    }
    cell_run: List[bool] = []
    cell_extraction: List[MinivolExtractionPlan] = []
    for grid_index in ordered_grid_indices:
        start = tuple(int(grid_index[axis]) * stride_value for axis in range(3))
        cell_extraction.append(
            _extraction_plan(
                start=start,
                minivol_size=minivol_value,
                raw_shape=raw_shape,
            )
        )
        cell_run.append(
            minivol_is_run(
                occupancy,
                start=start,
                minivol_size=minivol_value,
                stride=stride_value,
                raw_shape=raw_shape,
            )
        )

    # --- output blocks (raster ordered) -----------------------------------
    ordered_block_indices = tuple(
        _raster_ordered_grid_indices(write_block_lo, write_block_hi, scan_axis=scan_axis)
    )
    blocks: List[OutputBlock] = []
    block_ids_by_chunk_index: Dict[GridIndex, List[int]] = {}
    finalizing_cell_to_block: Dict[int, int] = {}
    for block_id, block_index in enumerate(ordered_block_indices):
        block_start = tuple(int(block_index[axis]) * stride_value for axis in range(3))
        block_stop = tuple(int(block_start[axis]) + stride_value for axis in range(3))
        write_slices = tuple(
            slice(
                max(int(block_start[axis]), int(bounds[axis][0])),
                min(int(block_stop[axis]), int(bounds[axis][1])),
            )
            for axis in range(3)
        )
        chunk_index = tuple(
            int(block_index[axis]) // int(normalized_chunk_multiple) for axis in range(3)
        )
        chunk_origin = tuple(int(chunk_index[axis]) * chunk_size for axis in range(3))
        slices_in_chunk = tuple(
            slice(
                int(write_slices[axis].start) - int(chunk_origin[axis]),
                int(write_slices[axis].stop) - int(chunk_origin[axis]),
            )
            for axis in range(3)
        )
        finalizing_cell_index = cell_index_by_grid[tuple(int(v) for v in block_index)]
        any_covering_run = _any_covering_cell_runs(
            block_index=block_index,
            cell_index_by_grid=cell_index_by_grid,
            cell_run=cell_run,
        )
        blocks.append(
            OutputBlock(
                id=block_id,
                grid_index=tuple(int(v) for v in block_index),
                write_slices=write_slices,
                chunk_id=-1,  # filled once chunks are numbered below
                slices_in_chunk=slices_in_chunk,
                finalizing_cell_index=int(finalizing_cell_index),
                any_covering_run=bool(any_covering_run),
            )
        )
        block_ids_by_chunk_index.setdefault(
            tuple(int(v) for v in chunk_index), []
        ).append(block_id)
        finalizing_cell_to_block[int(finalizing_cell_index)] = block_id

    # --- output chunks (raster ordered by chunk index) --------------------
    chunk_multiple = int(normalized_chunk_multiple)
    write_chunk_lo = tuple(int(write_block_lo[axis]) // chunk_multiple for axis in range(3))
    write_chunk_hi = tuple(int(write_block_hi[axis]) // chunk_multiple for axis in range(3))
    ordered_chunk_indices = tuple(
        _raster_ordered_grid_indices(write_chunk_lo, write_chunk_hi, scan_axis=scan_axis)
    )
    chunks: List[OutputChunk] = []
    chunk_id_by_index: Dict[GridIndex, int] = {}
    completion_cell_to_chunks: Dict[int, List[int]] = {}
    for chunk_id, chunk_index in enumerate(ordered_chunk_indices):
        member_block_ids = tuple(
            sorted(block_ids_by_chunk_index.get(tuple(int(v) for v in chunk_index), []))
        )
        if not member_block_ids:
            # A chunk index in the raster span with no member blocks cannot
            # occur (every chunk in range contains at least one write block),
            # but guard defensively rather than emit an empty chunk.
            continue
        chunk_origin = tuple(int(chunk_index[axis]) * chunk_size for axis in range(3))
        chunk_slices = tuple(
            slice(
                int(chunk_origin[axis]),
                min(int(chunk_origin[axis]) + chunk_size, int(raw_shape[axis])),
            )
            for axis in range(3)
        )
        completion_cell_index = max(
            int(blocks[block_id].finalizing_cell_index) for block_id in member_block_ids
        )
        has_data = any(
            bool(blocks[block_id].any_covering_run) for block_id in member_block_ids
        )
        chunk_id_by_index[tuple(int(v) for v in chunk_index)] = chunk_id
        completion_cell_to_chunks.setdefault(int(completion_cell_index), []).append(chunk_id)
        chunks.append(
            OutputChunk(
                id=chunk_id,
                chunk_index=tuple(int(v) for v in chunk_index),
                chunk_slices=chunk_slices,
                block_ids=member_block_ids,
                completion_cell_index=int(completion_cell_index),
                has_data=bool(has_data),
            )
        )

    # Backfill each block's chunk_id now that chunks are numbered.
    blocks = [
        _with_chunk_id(
            block,
            chunk_id_by_index[
                tuple(
                    int(block.grid_index[axis]) // chunk_multiple for axis in range(3)
                )
            ],
        )
        for block in blocks
    ]

    # --- attach per-cell finalize/flush events ----------------------------
    cells: List[MinivolCell] = []
    for index, grid_index in enumerate(ordered_grid_indices):
        cells.append(
            MinivolCell(
                index=int(index),
                grid_index=tuple(int(v) for v in grid_index),
                start=tuple(int(grid_index[axis]) * stride_value for axis in range(3)),
                run=bool(cell_run[index]),
                extraction=cell_extraction[index],
                finalizes_block_id=finalizing_cell_to_block.get(int(index)),
                completes_chunk_ids=tuple(
                    sorted(completion_cell_to_chunks.get(int(index), []))
                ),
            )
        )

    return StreamingInferencePlan(
        requested_bounds=bounds,
        raw_volume_shape=raw_shape,
        minivol_size=minivol_value,
        stride=stride_value,
        chunk_size=int(chunk_size),
        scan_axis=int(scan_axis),
        cells=tuple(cells),
        blocks=tuple(blocks),
        chunks=tuple(chunks),
    )


def _with_chunk_id(block: OutputBlock, chunk_id: int) -> OutputBlock:
    return OutputBlock(
        id=block.id,
        grid_index=block.grid_index,
        write_slices=block.write_slices,
        chunk_id=int(chunk_id),
        slices_in_chunk=block.slices_in_chunk,
        finalizing_cell_index=block.finalizing_cell_index,
        any_covering_run=block.any_covering_run,
    )


def _select_scan_axis(bounds: BoundsZYX) -> int:
    sizes = tuple(int(bounds[axis][1]) - int(bounds[axis][0]) for axis in range(3))
    best_axis = 0
    for axis in range(1, 3):
        if sizes[axis] > sizes[best_axis]:
            best_axis = axis
    return int(best_axis)


def _raster_ordered_grid_indices(
    lo: GridIndex,
    hi: GridIndex,
    *,
    scan_axis: int,
):
    axis_order = [int(scan_axis)] + [axis for axis in range(3) if axis != int(scan_axis)]
    for outer in range(int(lo[axis_order[0]]), int(hi[axis_order[0]]) + 1):
        for middle in range(int(lo[axis_order[1]]), int(hi[axis_order[1]]) + 1):
            for inner in range(int(lo[axis_order[2]]), int(hi[axis_order[2]]) + 1):
                index = [0, 0, 0]
                index[axis_order[0]] = outer
                index[axis_order[1]] = middle
                index[axis_order[2]] = inner
                yield (int(index[0]), int(index[1]), int(index[2]))


def _extraction_plan(
    *,
    start: GridIndex,
    minivol_size: int,
    raw_shape: ShapeZYX,
) -> MinivolExtractionPlan:
    raw_slices: List[slice] = []
    pad_before: List[int] = []
    pad_after: List[int] = []
    for axis in range(3):
        raw_start = int(start[axis])
        raw_stop = int(start[axis]) + int(minivol_size)
        clipped_start = max(0, raw_start)
        clipped_stop = min(int(raw_shape[axis]), raw_stop)
        raw_slices.append(slice(clipped_start, clipped_stop))
        pad_before.append(max(0, -raw_start))
        pad_after.append(max(0, raw_stop - int(raw_shape[axis])))
    return MinivolExtractionPlan(
        raw_slices=(raw_slices[0], raw_slices[1], raw_slices[2]),
        pad_before=(int(pad_before[0]), int(pad_before[1]), int(pad_before[2])),
        pad_after=(int(pad_after[0]), int(pad_after[1]), int(pad_after[2])),
    )


def _any_covering_cell_runs(
    *,
    block_index: GridIndex,
    cell_index_by_grid: Dict[GridIndex, int],
    cell_run: Sequence[bool],
) -> bool:
    for delta_z in (-1, 0):
        for delta_y in (-1, 0):
            for delta_x in (-1, 0):
                covering = (
                    int(block_index[0]) + delta_z,
                    int(block_index[1]) + delta_y,
                    int(block_index[2]) + delta_x,
                )
                cell_index = cell_index_by_grid.get(covering)
                if cell_index is not None and bool(cell_run[cell_index]):
                    return True
    return False


def _coerce_bounds(
    bounds: BoundsZYX,
    *,
    raw_volume_shape: ShapeZYX,
) -> BoundsZYX:
    if len(bounds) != 3:
        raise ValueError("requested_bounds must contain exactly 3 axes (z, y, x)")
    normalized: List[AxisBounds] = []
    for axis, axis_bounds in enumerate(bounds):
        if len(axis_bounds) != 2:
            raise ValueError(f"requested_bounds[{axis}] must contain exactly 2 values")
        start = _coerce_non_negative_int(axis_bounds[0], name=f"requested_bounds[{axis}][0]")
        stop = _coerce_positive_int(axis_bounds[1], name=f"requested_bounds[{axis}][1]")
        if int(stop) <= int(start):
            raise ValueError(f"requested_bounds[{axis}] must satisfy start < stop")
        if int(stop) > int(raw_volume_shape[axis]):
            raise ValueError(f"requested_bounds[{axis}] stop exceeds raw volume shape")
        normalized.append((int(start), int(stop)))
    return (normalized[0], normalized[1], normalized[2])


def _coerce_shape(values: Sequence[object], *, name: str) -> ShapeZYX:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 values (z, y, x)")
    return (
        _coerce_positive_int(values[0], name=f"{name}[0]"),
        _coerce_positive_int(values[1], name=f"{name}[1]"),
        _coerce_positive_int(values[2], name=f"{name}[2]"),
    )


def _coerce_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    normalized = int(value)
    if normalized < 1:
        raise ValueError(f"{name} must be >= 1")
    return normalized


def _coerce_non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be >= 0")
    return normalized
