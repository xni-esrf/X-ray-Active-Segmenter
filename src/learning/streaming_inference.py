"""Streaming (minivolume-by-minivolume) inference executor.

Given a :class:`StreamingInferencePlan` (Phase 1), a global normalization
(:class:`StreamingNormalizationStats`, Phase 2), a model forward callable and a
chunk writer (Phase 4), this replays the plan's raster-ordered schedule:

1. Read each *run* minivolume from the raw volume (reflect-padded at true edges),
   normalize it with the single global ``(mean, std)``.
2. Forward a batch of minivolumes through the model to obtain per-voxel class
   scores, Hann-weight them and add them into the buffers of the (up to eight)
   output blocks they cover.
3. After each batch, finalize every block whose covering cells are all
   accumulated (argmax -> label), placing its labels into its chunk buffer; once
   all of a chunk's blocks are finalized, write that chunk exactly once and free
   its memory.

The core (:func:`run_streaming_inference`) is deliberately free of any torch or
zarr dependency: the model forward is an injected callable and the destination
is an injected writer, so the geometry/accumulation logic is fully CPU-testable.
:func:`build_torch_minivol_forward` provides the real GPU forward for the runner.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable, Optional, Protocol, Sequence, Tuple

import numpy as np

from .streaming_inference_plan import MinivolExtractionPlan, StreamingInferencePlan
from .zero_occupancy import StreamingNormalizationStats


LOGGER = logging.getLogger(__name__)

SliceZYX = Tuple[slice, slice, slice]

# forward(batch[B, m, m, m] float32) -> logits[B, C, m, m, m]
ForwardMinivolBatch = Callable[[np.ndarray], np.ndarray]


class StreamingInferenceStopRequested(Exception):
    """Raised to abort a streaming inference run cleanly on request."""


class StreamingChunkWriter(Protocol):
    def write_chunk(self, array: np.ndarray, *, chunk_slices: SliceZYX) -> None:
        ...


@dataclass(frozen=True)
class StreamingInferenceResult:
    plan: StreamingInferencePlan
    processed_minivol_count: int
    written_chunk_count: int


def run_streaming_inference(
    *,
    plan: StreamingInferencePlan,
    raw_volume: object,
    normalization: StreamingNormalizationStats,
    label_values: Sequence[int],
    output_dtype: object,
    forward_minivol_batch: ForwardMinivolBatch,
    writer: StreamingChunkWriter,
    batch_size: int = 16,
    should_stop: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> StreamingInferenceResult:
    """Execute the streaming plan, writing the segmentation crop-by-chunk.

    ``forward_minivol_batch`` maps a ``[B, m, m, m]`` float32 batch to a
    ``[B, C, m, m, m]`` float32 array of per-voxel class scores.  ``writer`` is
    any object with ``write_chunk(array, *, chunk_slices)``.
    """
    output_dtype_np = np.dtype(output_dtype)
    label_lookup = np.asarray([int(v) for v in label_values], dtype=output_dtype_np)
    num_classes = int(label_lookup.shape[0])
    if num_classes == 0:
        raise ValueError("label_values must be non-empty")
    fill_value = label_lookup[0]

    minivol_size = int(plan.minivol_size)
    hann = _hann_window_3d(minivol_size)
    mean = float(normalization.mean)
    std = float(normalization.std)
    if std == 0.0:
        std = 1.0

    total_data_chunks = int(plan.written_chunk_count)
    LOGGER.info(
        "Streaming inference: cells=%d run=%d write_blocks=%d chunks=%d "
        "data_chunks=%d scan_axis=%d batch_size=%d",
        plan.total_cell_count,
        plan.run_cell_count,
        plan.write_block_count,
        plan.chunk_count,
        total_data_chunks,
        plan.scan_axis,
        int(batch_size),
    )

    block_id_by_grid = {block.grid_index: block.id for block in plan.blocks}
    open_blocks: dict = {}
    open_chunks: dict = {}
    state = {"written": 0, "processed": 0, "events_upto": -1}
    batch_capacity = max(1, int(batch_size))

    def accumulate(cell, logits: np.ndarray) -> None:
        weighted = logits * hann[np.newaxis, :, :, :]
        grid = cell.grid_index
        start = cell.start
        for block_z in (grid[0], grid[0] + 1):
            for block_y in (grid[1], grid[1] + 1):
                for block_x in (grid[2], grid[2] + 1):
                    block_id = block_id_by_grid.get((block_z, block_y, block_x))
                    if block_id is None:
                        continue
                    block = plan.blocks[block_id]
                    write_slices = block.write_slices
                    minivol_local = []
                    block_local = []
                    ok = True
                    for axis in range(3):
                        lo = max(int(start[axis]), int(write_slices[axis].start))
                        hi = min(int(start[axis]) + minivol_size, int(write_slices[axis].stop))
                        if hi <= lo:
                            ok = False
                            break
                        minivol_local.append(slice(lo - int(start[axis]), hi - int(start[axis])))
                        block_local.append(
                            slice(lo - int(write_slices[axis].start), hi - int(write_slices[axis].start))
                        )
                    if not ok:
                        continue
                    buffer = open_blocks.get(block_id)
                    if buffer is None:
                        write_shape = tuple(
                            int(write_slices[axis].stop) - int(write_slices[axis].start)
                            for axis in range(3)
                        )
                        buffer = np.zeros((num_classes,) + write_shape, dtype=np.float32)
                        open_blocks[block_id] = buffer
                    buffer[:, block_local[0], block_local[1], block_local[2]] += weighted[
                        :, minivol_local[0], minivol_local[1], minivol_local[2]
                    ]

    def finalize_block(block_id: int) -> None:
        block = plan.blocks[block_id]
        buffer = open_blocks.pop(block_id, None)
        if buffer is None:
            # No covering cell ran: the block is pure background and stays at the
            # writer's fill value (label_values[0]).
            return
        channel = np.argmax(buffer, axis=0)
        labels = label_lookup[channel]
        chunk = plan.chunks[block.chunk_id]
        chunk_buffer = open_chunks.get(block.chunk_id)
        if chunk_buffer is None:
            chunk_buffer = np.full(
                _slices_shape(chunk.chunk_slices), fill_value, dtype=output_dtype_np
            )
            open_chunks[block.chunk_id] = chunk_buffer
        sic = block.slices_in_chunk
        chunk_buffer[sic[0], sic[1], sic[2]] = labels

    def flush_chunk(chunk_id: int) -> None:
        chunk = plan.chunks[chunk_id]
        chunk_buffer = open_chunks.pop(chunk_id, None)
        if not chunk.has_data:
            # Entirely background: leave the region at the store's fill value.
            return
        if chunk_buffer is None:
            chunk_buffer = np.full(
                _slices_shape(chunk.chunk_slices), fill_value, dtype=output_dtype_np
            )
        writer.write_chunk(chunk_buffer, chunk_slices=chunk.chunk_slices)
        state["written"] += 1
        if progress_callback is not None:
            progress_callback(state["written"], total_data_chunks)

    def drain_events(upto_index: int) -> None:
        for index in range(state["events_upto"] + 1, upto_index + 1):
            cell = plan.cells[index]
            if cell.finalizes_block_id is not None:
                finalize_block(cell.finalizes_block_id)
            for chunk_id in cell.completes_chunk_ids:
                flush_chunk(chunk_id)
        state["events_upto"] = upto_index

    pending: list = []

    def forward_pending() -> None:
        if not pending:
            return
        if should_stop is not None and should_stop():
            raise StreamingInferenceStopRequested()
        batch = np.stack(
            [
                _read_and_normalize(raw_volume, cell.extraction, minivol_size, mean, std)
                for cell in pending
            ],
            axis=0,
        )
        logits = np.asarray(forward_minivol_batch(batch))
        expected = (len(pending), num_classes, minivol_size, minivol_size, minivol_size)
        if logits.shape != expected:
            raise ValueError(
                f"forward_minivol_batch must return {expected}; got {logits.shape}"
            )
        for index_in_batch, cell in enumerate(pending):
            accumulate(cell, logits[index_in_batch].astype(np.float32, copy=False))
        state["processed"] += len(pending)
        last_index = pending[-1].index
        pending.clear()
        drain_events(last_index)

    for cell in plan.cells:
        if cell.run:
            pending.append(cell)
            if len(pending) >= batch_capacity:
                forward_pending()
    forward_pending()
    drain_events(plan.total_cell_count - 1)

    if open_blocks or open_chunks:
        LOGGER.warning(
            "Streaming inference ended with %d open blocks and %d open chunks",
            len(open_blocks),
            len(open_chunks),
        )
    LOGGER.info(
        "Streaming inference completed: processed_minivols=%d written_chunks=%d/%d",
        state["processed"],
        state["written"],
        total_data_chunks,
    )
    return StreamingInferenceResult(
        plan=plan,
        processed_minivol_count=int(state["processed"]),
        written_chunk_count=int(state["written"]),
    )


def build_torch_minivol_forward(
    model_runtime: object,
    *,
    device: object = None,
    autocast: bool = True,
) -> ForwardMinivolBatch:
    """Wrap a torch model runtime as a ``forward_minivol_batch`` callable.

    Adds the channel dim, moves the batch to the model device (float16 on CUDA),
    runs the forward under ``no_grad`` (and bfloat16 autocast on CUDA), and
    returns float32 CPU class scores.  Torch is imported lazily.
    """
    import torch

    model = getattr(model_runtime, "model", model_runtime)
    resolved = _resolve_torch_device(model_runtime, model, device, torch)
    input_dtype = torch.float16 if resolved.type == "cuda" else torch.float32

    def forward(batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32))
            tensor = tensor.to(device=resolved, dtype=input_dtype).unsqueeze(1)
            with torch.autocast(
                device_type=resolved.type,
                enabled=bool(autocast) and resolved.type == "cuda",
                dtype=torch.bfloat16,
            ):
                output = model(tensor)
            return output.detach().to(dtype=torch.float32).cpu().numpy()

    return forward


def _resolve_torch_device(model_runtime, model, device, torch):
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        device_ids = getattr(model_runtime, "device_ids", None)
        if device_ids:
            return torch.device(f"cuda:{int(device_ids[0])}")
        return torch.device("cuda:0")
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _read_and_normalize(
    raw_volume: object,
    extraction: MinivolExtractionPlan,
    minivol_size: int,
    mean: float,
    std: float,
) -> np.ndarray:
    extracted = _extract_minivol(raw_volume, extraction, minivol_size)
    normalized = (extracted.astype(np.float32, copy=False) - np.float32(mean)) / np.float32(std)
    return normalized


def _extract_minivol(
    raw_volume: object,
    extraction: MinivolExtractionPlan,
    minivol_size: int,
) -> np.ndarray:
    get_chunk = getattr(raw_volume, "get_chunk", None)
    if not callable(get_chunk):
        raise TypeError("raw_volume must define get_chunk(zyx_slices)")
    clipped = np.asarray(get_chunk(extraction.raw_slices))
    if clipped.ndim != 3:
        raise ValueError(f"extracted minivolume must be 3D (z, y, x), got ndim={clipped.ndim}")
    pad_width = (
        (int(extraction.pad_before[0]), int(extraction.pad_after[0])),
        (int(extraction.pad_before[1]), int(extraction.pad_after[1])),
        (int(extraction.pad_before[2]), int(extraction.pad_after[2])),
    )
    if any(before > 0 or after > 0 for before, after in pad_width):
        for axis, (before, after) in enumerate(pad_width):
            if (before > 0 or after > 0) and int(clipped.shape[axis]) <= 1:
                raise ValueError(
                    f"cannot reflect-pad axis {axis} of length <= 1 during extraction"
                )
        clipped = np.pad(clipped, pad_width, mode="reflect")
    expected = (minivol_size, minivol_size, minivol_size)
    if tuple(int(axis) for axis in clipped.shape) != expected:
        raise RuntimeError(
            f"unexpected minivolume shape: got={tuple(clipped.shape)} expected={expected}"
        )
    return clipped


def _hann_window_3d(minivol_size: int) -> np.ndarray:
    n = int(minivol_size)
    if n < 2:
        raise ValueError("minivol_size must be >= 2 for a Hann window")
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / float(n - 1)))
    window = window.astype(np.float32)
    return (
        window[:, np.newaxis, np.newaxis]
        * window[np.newaxis, :, np.newaxis]
        * window[np.newaxis, np.newaxis, :]
    ).astype(np.float32)


def _slices_shape(slices: SliceZYX) -> Tuple[int, int, int]:
    return (
        int(slices[0].stop) - int(slices[0].start),
        int(slices[1].stop) - int(slices[1].start),
        int(slices[2].stop) - int(slices[2].start),
    )
