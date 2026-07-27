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
from queue import Queue
from threading import Lock, Thread
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


class _StreamingAccumulator:
    """Owns all mutable accumulation state for a streaming inference run.

    Batches of forwarded logits are fed in raster order via
    :meth:`process_batch`; after the last batch, :meth:`finish` drains the
    remaining finalize/flush events.  The class carries no threading of its own,
    so it can be driven either inline (synchronous path) or from a single worker
    thread (async path) with bit-identical results: a single owner processing
    batches in order preserves the per-block accumulation sequence.
    """

    def __init__(
        self,
        plan: StreamingInferencePlan,
        *,
        label_values: Sequence[int],
        output_dtype: object,
        writer: StreamingChunkWriter,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        self._plan = plan
        self._writer = writer
        self._progress_callback = progress_callback

        self._output_dtype_np = np.dtype(output_dtype)
        self._label_lookup = np.asarray(
            [int(v) for v in label_values], dtype=self._output_dtype_np
        )
        self.num_classes = int(self._label_lookup.shape[0])
        if self.num_classes == 0:
            raise ValueError("label_values must be non-empty")
        self._fill_value = self._label_lookup[0]

        self.minivol_size = int(plan.minivol_size)
        self._hann = _hann_window_3d(self.minivol_size)
        self._total_data_chunks = int(plan.written_chunk_count)

        self._block_id_by_grid = {block.grid_index: block.id for block in plan.blocks}
        self._open_blocks: dict = {}
        self._open_chunks: dict = {}
        self._written = 0
        self._processed = 0
        self._events_upto = -1

    def process_batch(
        self, cells: Sequence, logits: np.ndarray, last_index: int
    ) -> None:
        """Accumulate one forwarded batch and drain events up to ``last_index``.

        ``cells`` are the run cells of the batch (in raster order); ``logits`` is
        the ``[B, C, m, m, m]`` array returned by the model forward.
        """
        for index_in_batch, cell in enumerate(cells):
            self._accumulate(
                cell, logits[index_in_batch].astype(np.float32, copy=False)
            )
        self._processed += len(cells)
        self._drain_events(int(last_index))

    def finish(self) -> Tuple[int, int]:
        """Drain all remaining events; return ``(processed, written)``."""
        self._drain_events(self._plan.total_cell_count - 1)
        if self._open_blocks or self._open_chunks:
            LOGGER.warning(
                "Streaming inference ended with %d open blocks and %d open chunks",
                len(self._open_blocks),
                len(self._open_chunks),
            )
        return int(self._processed), int(self._written)

    def _accumulate(self, cell, logits: np.ndarray) -> None:
        weighted = logits * self._hann[np.newaxis, :, :, :]
        grid = cell.grid_index
        start = cell.start
        for block_z in (grid[0], grid[0] + 1):
            for block_y in (grid[1], grid[1] + 1):
                for block_x in (grid[2], grid[2] + 1):
                    block_id = self._block_id_by_grid.get((block_z, block_y, block_x))
                    if block_id is None:
                        continue
                    block = self._plan.blocks[block_id]
                    write_slices = block.write_slices
                    minivol_local = []
                    block_local = []
                    ok = True
                    for axis in range(3):
                        lo = max(int(start[axis]), int(write_slices[axis].start))
                        hi = min(
                            int(start[axis]) + self.minivol_size,
                            int(write_slices[axis].stop),
                        )
                        if hi <= lo:
                            ok = False
                            break
                        minivol_local.append(
                            slice(lo - int(start[axis]), hi - int(start[axis]))
                        )
                        block_local.append(
                            slice(
                                lo - int(write_slices[axis].start),
                                hi - int(write_slices[axis].start),
                            )
                        )
                    if not ok:
                        continue
                    buffer = self._open_blocks.get(block_id)
                    if buffer is None:
                        write_shape = tuple(
                            int(write_slices[axis].stop) - int(write_slices[axis].start)
                            for axis in range(3)
                        )
                        buffer = np.zeros(
                            (self.num_classes,) + write_shape, dtype=np.float32
                        )
                        self._open_blocks[block_id] = buffer
                    buffer[:, block_local[0], block_local[1], block_local[2]] += weighted[
                        :, minivol_local[0], minivol_local[1], minivol_local[2]
                    ]

    def _finalize_block(self, block_id: int) -> None:
        block = self._plan.blocks[block_id]
        buffer = self._open_blocks.pop(block_id, None)
        if buffer is None:
            # No covering cell ran: the block is pure background and stays at the
            # writer's fill value (label_values[0]).
            return
        channel = np.argmax(buffer, axis=0)
        labels = self._label_lookup[channel]
        chunk = self._plan.chunks[block.chunk_id]
        chunk_buffer = self._open_chunks.get(block.chunk_id)
        if chunk_buffer is None:
            chunk_buffer = np.full(
                _slices_shape(chunk.chunk_slices),
                self._fill_value,
                dtype=self._output_dtype_np,
            )
            self._open_chunks[block.chunk_id] = chunk_buffer
        sic = block.slices_in_chunk
        chunk_buffer[sic[0], sic[1], sic[2]] = labels

    def _flush_chunk(self, chunk_id: int) -> None:
        chunk = self._plan.chunks[chunk_id]
        chunk_buffer = self._open_chunks.pop(chunk_id, None)
        if not chunk.has_data:
            # Entirely background: leave the region at the store's fill value.
            return
        if chunk_buffer is None:
            chunk_buffer = np.full(
                _slices_shape(chunk.chunk_slices),
                self._fill_value,
                dtype=self._output_dtype_np,
            )
        self._writer.write_chunk(chunk_buffer, chunk_slices=chunk.chunk_slices)
        self._written += 1
        if self._progress_callback is not None:
            self._progress_callback(self._written, self._total_data_chunks)

    def _drain_events(self, upto_index: int) -> None:
        for index in range(self._events_upto + 1, upto_index + 1):
            cell = self._plan.cells[index]
            if cell.finalizes_block_id is not None:
                self._finalize_block(cell.finalizes_block_id)
            for chunk_id in cell.completes_chunk_ids:
                self._flush_chunk(chunk_id)
        self._events_upto = upto_index


_STREAMING_ACCUMULATOR_FINISH = object()
_STREAMING_ACCUMULATOR_ABORT = object()


class _AsyncStreamingAccumulator:
    """Runs a :class:`_StreamingAccumulator` on a single background thread.

    The producer thread submits forwarded batches via :meth:`submit`; a bounded
    queue provides back-pressure, capping in-flight logits at ``max_queue_size``
    batches.  The worker owns the accumulator exclusively and consumes batches
    in submission (raster) order, so the output is bit-identical to driving the
    accumulator inline.  Exceptions raised inside the worker are captured and
    re-raised on the producer thread at the next :meth:`submit`/:meth:`finish`.

    :meth:`finish` drains all queued batches, finalizes the tail and returns the
    ``(processed, written)`` counts; :meth:`close` aborts without finalizing (for
    stop requests and error unwinding) and never raises.  Note that any
    ``progress_callback`` held by the inner accumulator fires on the worker
    thread in this mode.
    """

    def __init__(
        self,
        accumulator: "_StreamingAccumulator",
        *,
        max_queue_size: int = 2,
    ) -> None:
        self._inner = accumulator
        self._queue: "Queue[object]" = Queue(maxsize=max(1, int(max_queue_size)))
        self._error: Optional[BaseException] = None
        self._result: Optional[Tuple[int, int]] = None
        self._lock = Lock()
        self._closed = False
        self._worker = Thread(
            target=self._run,
            name="xray-streaming-accumulator",
            daemon=True,
        )
        self._worker.start()

    def submit(self, cells: Sequence, logits: np.ndarray, last_index: int) -> None:
        """Hand one forwarded batch to the worker (blocks if the queue is full)."""
        self._raise_if_failed()
        if self._closed:
            raise RuntimeError("Async streaming accumulator is closed")
        self._queue.put((cells, logits, int(last_index)))
        self._raise_if_failed()

    def finish(self) -> Tuple[int, int]:
        """Drain queued batches, finalize the tail, return ``(processed, written)``."""
        if not self._closed:
            self._closed = True
            self._queue.put(_STREAMING_ACCUMULATOR_FINISH)
        self._worker.join()
        self._raise_if_failed()
        if self._result is None:
            raise RuntimeError("Async streaming accumulator produced no result")
        return self._result

    def close(self) -> None:
        """Abort without finalizing and join the worker.  Never raises."""
        if not self._closed:
            self._closed = True
            try:
                self._queue.put(_STREAMING_ACCUMULATOR_ABORT)
            except Exception:
                return
        try:
            self._worker.join()
        except RuntimeError:
            return

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is _STREAMING_ACCUMULATOR_ABORT:
                    return
                if job is _STREAMING_ACCUMULATOR_FINISH:
                    if self._error is None:
                        try:
                            self._result = self._inner.finish()
                        except BaseException as exc:  # noqa: BLE001
                            with self._lock:
                                if self._error is None:
                                    self._error = exc
                    return
                if self._error is not None:
                    # A previous batch failed: drain the backlog without work so
                    # the producer's next submit/finish can surface the error.
                    continue
                cells, logits, last_index = job
                self._inner.process_batch(cells, logits, last_index)
            except BaseException as exc:  # noqa: BLE001 - surfaced to producer
                with self._lock:
                    if self._error is None:
                        self._error = exc
            finally:
                self._queue.task_done()

    def _raise_if_failed(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError(
                f"Async streaming accumulation failed: {error}"
            ) from error


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
    accumulation_queue_size: int = 2,
) -> StreamingInferenceResult:
    """Execute the streaming plan, writing the segmentation crop-by-chunk.

    ``forward_minivol_batch`` maps a ``[B, m, m, m]`` float32 batch to a
    ``[B, C, m, m, m]`` float32 array of per-voxel class scores.  ``writer`` is
    any object with ``write_chunk(array, *, chunk_slices)``.

    The Hann-weight/accumulate/finalize/write work runs on a single background
    worker so it overlaps the next batch's read+forward+transfer;
    ``accumulation_queue_size`` bounds the in-flight logits (and thus host RAM).
    The worker consumes batches in raster order, so the output is deterministic.
    ``progress_callback`` fires on the worker thread.
    """
    mean = float(normalization.mean)
    std = float(normalization.std)
    if std == 0.0:
        std = 1.0

    accumulator = _StreamingAccumulator(
        plan,
        label_values=label_values,
        output_dtype=output_dtype,
        writer=writer,
        progress_callback=progress_callback,
    )
    minivol_size = accumulator.minivol_size
    num_classes = accumulator.num_classes
    total_data_chunks = int(plan.written_chunk_count)

    sink = _AsyncStreamingAccumulator(
        accumulator, max_queue_size=accumulation_queue_size
    )

    LOGGER.info(
        "Streaming inference: cells=%d run=%d write_blocks=%d chunks=%d "
        "data_chunks=%d scan_axis=%d batch_size=%d queue=%d",
        plan.total_cell_count,
        plan.run_cell_count,
        plan.write_block_count,
        plan.chunk_count,
        total_data_chunks,
        plan.scan_axis,
        int(batch_size),
        int(accumulation_queue_size),
    )

    batch_capacity = max(1, int(batch_size))
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
        last_index = pending[-1].index
        sink.submit(list(pending), logits, last_index)
        pending.clear()

    try:
        for cell in plan.cells:
            if cell.run:
                pending.append(cell)
                if len(pending) >= batch_capacity:
                    forward_pending()
        forward_pending()
        processed, written = sink.finish()
    finally:
        # On a stop request or any producer/worker error, abort the worker
        # without finalizing; on the normal path this is a no-op join.
        sink.close()

    LOGGER.info(
        "Streaming inference completed: processed_minivols=%d written_chunks=%d/%d",
        processed,
        written,
        total_data_chunks,
    )
    return StreamingInferenceResult(
        plan=plan,
        processed_minivol_count=processed,
        written_chunk_count=written,
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
