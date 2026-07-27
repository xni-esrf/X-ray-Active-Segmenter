from __future__ import annotations

import threading
import unittest

import numpy as np

from src.learning.streaming_inference import (
    StreamingInferenceStopRequested,
    build_torch_minivol_forward,
    run_streaming_inference,
    _hann_window_3d,
)
from src.learning.streaming_inference_plan import build_streaming_inference_plan
from src.learning.zero_occupancy import (
    StreamingNormalizationStats,
    prepare_streaming_occupancy_and_stats,
)


MINIVOL = 20
STRIDE = 10
LABEL_VALUES = (0, 1, 2, 3)
NUM_CLASSES = len(LABEL_VALUES)


class _FakeVolume:
    def __init__(self, array: np.ndarray) -> None:
        self.array = np.asarray(array)
        self.chunk_shape = None

    def get_chunk(self, zyx_slices):
        return self.array[zyx_slices]


class _ArrayWriter:
    def __init__(self, shape, dtype, fill) -> None:
        self.array = np.full(shape, fill, dtype=dtype)
        self.writes = []

    def write_chunk(self, array, *, chunk_slices):
        self.writes.append(tuple((s.start, s.stop) for s in chunk_slices))
        self.array[chunk_slices] = array


class _RaisingWriter:
    """Writer whose first ``write_chunk`` raises, to test error propagation."""

    def __init__(self, message="boom write") -> None:
        self.message = message
        self.calls = 0

    def write_chunk(self, array, *, chunk_slices):
        self.calls += 1
        raise RuntimeError(self.message)


def _call_with_timeout(fn, timeout=15.0):
    """Run ``fn`` on a worker thread; fail on hang, else return/re-raise.

    Guards against a deadlock in the accumulation worker silently turning into a
    hung process: a stuck worker surfaces as an explicit AssertionError.
    """
    box: dict = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on caller thread
            box["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise AssertionError(
            f"run_streaming_inference did not return within {timeout}s (deadlock)"
        )
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _mock_forward(batch: np.ndarray) -> np.ndarray:
    # Deterministic, per-sample-independent, well varied across channels so the
    # argmax genuinely depends on position and value.
    b = batch.shape[0]
    out = np.empty((b, NUM_CLASSES) + batch.shape[1:], dtype=np.float32)
    for c in range(NUM_CLASSES):
        out[:, c] = np.cos(batch * (0.5 + 0.37 * c) + 0.3 * c).astype(np.float32)
    return out


def _extract_minivol_ref(array, extraction, minivol_size):
    clipped = np.asarray(array[extraction.raw_slices])
    pad = (
        (extraction.pad_before[0], extraction.pad_after[0]),
        (extraction.pad_before[1], extraction.pad_after[1]),
        (extraction.pad_before[2], extraction.pad_after[2]),
    )
    if any(a or b for a, b in pad):
        clipped = np.pad(clipped, pad, mode="reflect")
    assert clipped.shape == (minivol_size,) * 3
    return clipped


def _dense_reference(plan, array, *, mean, std, forward, minivol_size, label_values):
    """Straightforward dense overlap-add over the bbox, respecting run flags.

    Accumulates run minivolumes in raster (cell) order so the per-voxel float32
    sum order matches the streaming executor exactly.  Independent of the
    executor's implementation, so it validates the accumulation worker's output.
    """
    bounds = plan.requested_bounds
    bshape = tuple(bounds[a][1] - bounds[a][0] for a in range(3))
    buffer = np.zeros((len(label_values),) + bshape, dtype=np.float32)
    hann = _hann_window_3d(minivol_size)
    for cell in plan.cells:
        if not cell.run:
            continue
        minivol = _extract_minivol_ref(array, cell.extraction, minivol_size)
        normalized = (minivol.astype(np.float32) - np.float32(mean)) / np.float32(std)
        logits = forward(normalized[np.newaxis, ...])[0].astype(np.float32)
        weighted = logits * hann[np.newaxis, ...]
        start = cell.start
        m_local = []
        b_local = []
        ok = True
        for a in range(3):
            lo = max(start[a], bounds[a][0])
            hi = min(start[a] + minivol_size, bounds[a][1])
            if hi <= lo:
                ok = False
                break
            m_local.append(slice(lo - start[a], hi - start[a]))
            b_local.append(slice(lo - bounds[a][0], hi - bounds[a][0]))
        if not ok:
            continue
        buffer[:, b_local[0], b_local[1], b_local[2]] += weighted[
            :, m_local[0], m_local[1], m_local[2]
        ]
    channel = np.argmax(buffer, axis=0)
    lookup = np.asarray(label_values, dtype=np.uint8)
    return lookup[channel]


def _random_volume(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-2.0, 30.0, size=shape).astype(np.float32)


def _blobby_volume(shape, seed=7):
    rng = np.random.default_rng(seed)
    arr = np.zeros(shape, dtype=np.float32)

    def put(lo_frac, hi_frac):
        sl = tuple(
            slice(int(shape[a] * lo_frac[a]), int(shape[a] * hi_frac[a])) for a in range(3)
        )
        size = tuple(sl[a].stop - sl[a].start for a in range(3))
        if all(s > 0 for s in size):
            arr[sl] = rng.uniform(3.0, 30.0, size=size).astype(np.float32)

    put((0.35, 0.30, 0.40), (0.62, 0.52, 0.60))
    put((0.10, 0.72, 0.12), (0.28, 0.90, 0.30))
    return arr


def _single_blob_volume(shape, seed=11):
    # One small central blob so most of the volume is background and many output
    # chunks are entirely empty (never written).
    rng = np.random.default_rng(seed)
    arr = np.zeros(shape, dtype=np.float32)
    sl = tuple(slice(int(shape[a] * 0.42), int(shape[a] * 0.55)) for a in range(3))
    size = tuple(sl[a].stop - sl[a].start for a in range(3))
    arr[sl] = rng.uniform(4.0, 25.0, size=size).astype(np.float32)
    return arr


class StreamingInferenceDenseEquivalenceTest(unittest.TestCase):
    CONFIGS = [
        ((80, 80, 80), ((10, 70), (5, 75), (20, 60))),
        ((60, 80, 72), ((0, 60), (13, 77), (7, 65))),  # touches z edge (reflect)
        ((80, 80, 80), ((0, 40), (0, 40), (0, 40))),  # corner: reflect on all axes
    ]

    def test_streaming_matches_dense_without_skipping(self):
        for raw_shape, bounds in self.CONFIGS:
            with self.subTest(bounds=bounds):
                arr = _random_volume(raw_shape, seed=hash(bounds) % 1000)
                plan = build_streaming_inference_plan(
                    requested_bounds=bounds,
                    raw_volume_shape=raw_shape,
                    minivol_size=MINIVOL,
                    occupancy=None,
                )
                writer = _ArrayWriter(raw_shape, np.uint8, fill=LABEL_VALUES[0])
                result = run_streaming_inference(
                    plan=plan,
                    raw_volume=_FakeVolume(arr),
                    normalization=StreamingNormalizationStats(
                        mean=6.0, std=3.0, voxel_count=1
                    ),
                    label_values=LABEL_VALUES,
                    output_dtype=np.uint8,
                    forward_minivol_batch=_mock_forward,
                    writer=writer,
                    batch_size=7,
                )
                dense = _dense_reference(
                    plan, arr, mean=6.0, std=3.0, forward=_mock_forward,
                    minivol_size=MINIVOL, label_values=LABEL_VALUES,
                )
                out = writer.array[
                    bounds[0][0] : bounds[0][1],
                    bounds[1][0] : bounds[1][1],
                    bounds[2][0] : bounds[2][1],
                ]
                np.testing.assert_array_equal(out, dense)
                self.assertEqual(result.processed_minivol_count, plan.run_cell_count)

    def test_streaming_matches_dense_with_skipping(self):
        raw_shape = (80, 80, 80)
        bounds = ((5, 75), (5, 75), (5, 75))
        arr = _blobby_volume(raw_shape)
        prepass = prepare_streaming_occupancy_and_stats(
            _FakeVolume(arr), requested_bounds=bounds, raw_volume_shape=raw_shape,
            minivol_size=MINIVOL,
        )
        plan = build_streaming_inference_plan(
            requested_bounds=bounds,
            raw_volume_shape=raw_shape,
            minivol_size=MINIVOL,
            occupancy=prepass.grid,
        )
        # genuinely a mix of run and skipped
        self.assertGreater(plan.run_cell_count, 0)
        self.assertLess(plan.run_cell_count, plan.total_cell_count)

        norm = prepass.normalization
        writer = _ArrayWriter(raw_shape, np.uint8, fill=LABEL_VALUES[0])
        run_streaming_inference(
            plan=plan,
            raw_volume=_FakeVolume(arr),
            normalization=norm,
            label_values=LABEL_VALUES,
            output_dtype=np.uint8,
            forward_minivol_batch=_mock_forward,
            writer=writer,
            batch_size=5,
        )
        dense = _dense_reference(
            plan, arr, mean=norm.mean, std=norm.std, forward=_mock_forward,
            minivol_size=MINIVOL, label_values=LABEL_VALUES,
        )
        out = writer.array[
            bounds[0][0] : bounds[0][1],
            bounds[1][0] : bounds[1][1],
            bounds[2][0] : bounds[2][1],
        ]
        np.testing.assert_array_equal(out, dense)

    def test_result_is_batch_size_invariant(self):
        raw_shape = (80, 80, 80)
        bounds = ((7, 73), (11, 69), (5, 75))
        arr = _random_volume(raw_shape, seed=3)
        outputs = []
        for batch_size in (1, 4, 16, 64):
            plan = build_streaming_inference_plan(
                requested_bounds=bounds, raw_volume_shape=raw_shape, minivol_size=MINIVOL
            )
            writer = _ArrayWriter(raw_shape, np.uint8, fill=LABEL_VALUES[0])
            run_streaming_inference(
                plan=plan,
                raw_volume=_FakeVolume(arr),
                normalization=StreamingNormalizationStats(4.0, 2.0, 1),
                label_values=LABEL_VALUES,
                output_dtype=np.uint8,
                forward_minivol_batch=_mock_forward,
                writer=writer,
                batch_size=batch_size,
            )
            outputs.append(writer.array.copy())
        for other in outputs[1:]:
            np.testing.assert_array_equal(outputs[0], other)


class StreamingInferenceWriteBehaviourTest(unittest.TestCase):
    def test_each_chunk_written_exactly_once(self):
        raw_shape = (120, 120, 120)  # 6x6x6 chunk grid; central blob leaves empties
        bounds = ((0, 120), (0, 120), (0, 120))
        arr = _single_blob_volume(raw_shape)
        prepass = prepare_streaming_occupancy_and_stats(
            _FakeVolume(arr), requested_bounds=bounds, raw_volume_shape=raw_shape,
            minivol_size=MINIVOL,
        )
        plan = build_streaming_inference_plan(
            requested_bounds=bounds, raw_volume_shape=raw_shape, minivol_size=MINIVOL,
            occupancy=prepass.grid,
        )
        writer = _ArrayWriter(raw_shape, np.uint8, fill=LABEL_VALUES[0])
        result = run_streaming_inference(
            plan=plan,
            raw_volume=_FakeVolume(arr),
            normalization=prepass.normalization,
            label_values=LABEL_VALUES,
            output_dtype=np.uint8,
            forward_minivol_batch=_mock_forward,
            writer=writer,
        )
        # no chunk written more than once
        self.assertEqual(len(writer.writes), len(set(writer.writes)))
        # only the data chunks are written
        self.assertEqual(len(writer.writes), plan.written_chunk_count)
        self.assertEqual(result.written_chunk_count, plan.written_chunk_count)
        self.assertLess(plan.written_chunk_count, plan.chunk_count)

    def test_skipped_regions_stay_at_fill_value(self):
        raw_shape = (80, 80, 80)
        bounds = ((0, 80), (0, 80), (0, 80))
        arr = _blobby_volume(raw_shape)
        prepass = prepare_streaming_occupancy_and_stats(
            _FakeVolume(arr), requested_bounds=bounds, raw_volume_shape=raw_shape,
            minivol_size=MINIVOL,
        )
        plan = build_streaming_inference_plan(
            requested_bounds=bounds, raw_volume_shape=raw_shape, minivol_size=MINIVOL,
            occupancy=prepass.grid,
        )
        writer = _ArrayWriter(raw_shape, np.uint8, fill=LABEL_VALUES[0])
        run_streaming_inference(
            plan=plan,
            raw_volume=_FakeVolume(arr),
            normalization=prepass.normalization,
            label_values=LABEL_VALUES,
            output_dtype=np.uint8,
            forward_minivol_batch=_mock_forward,
            writer=writer,
        )
        # a far-background corner must be untouched (fill == background label)
        self.assertEqual(int(writer.array[0, -1, -1]), LABEL_VALUES[0])
        # but the volume is not entirely background
        self.assertGreater(int((writer.array != LABEL_VALUES[0]).sum()), 0)

    def test_output_dtype_is_respected(self):
        raw_shape = (60, 60, 60)
        bounds = ((5, 55), (5, 55), (5, 55))
        arr = _random_volume(raw_shape, seed=9)
        plan = build_streaming_inference_plan(
            requested_bounds=bounds, raw_volume_shape=raw_shape, minivol_size=MINIVOL
        )
        writer = _ArrayWriter(raw_shape, np.uint16, fill=LABEL_VALUES[0])
        run_streaming_inference(
            plan=plan,
            raw_volume=_FakeVolume(arr),
            normalization=StreamingNormalizationStats(0.0, 1.0, 1),
            label_values=LABEL_VALUES,
            output_dtype=np.uint16,
            forward_minivol_batch=_mock_forward,
            writer=writer,
        )
        self.assertEqual(writer.array.dtype, np.uint16)

    def test_zero_run_cell_box_writes_nothing(self):
        # An all-background box skips every minivolume: no forwards, no writes,
        # result counts are zero, and the store stays at the fill value.
        raw_shape = (80, 80, 80)
        bounds = ((5, 75), (5, 75), (5, 75))
        arr = np.zeros(raw_shape, dtype=np.float32)
        prepass = prepare_streaming_occupancy_and_stats(
            _FakeVolume(arr), requested_bounds=bounds, raw_volume_shape=raw_shape,
            minivol_size=MINIVOL,
        )
        plan = build_streaming_inference_plan(
            requested_bounds=bounds, raw_volume_shape=raw_shape, minivol_size=MINIVOL,
            occupancy=prepass.grid,
        )
        self.assertEqual(plan.run_cell_count, 0)
        writer = _ArrayWriter(raw_shape, np.uint8, fill=LABEL_VALUES[0])
        result = _call_with_timeout(
            lambda: run_streaming_inference(
                plan=plan,
                raw_volume=_FakeVolume(arr),
                normalization=prepass.normalization,
                label_values=LABEL_VALUES,
                output_dtype=np.uint8,
                forward_minivol_batch=_mock_forward,
                writer=writer,
            )
        )
        self.assertEqual(result.processed_minivol_count, 0)
        self.assertEqual(result.written_chunk_count, 0)
        self.assertEqual(len(writer.writes), 0)
        self.assertTrue(bool(np.all(writer.array == LABEL_VALUES[0])))


class StreamingInferenceStopTest(unittest.TestCase):
    def test_stop_request_aborts(self):
        raw_shape = (80, 80, 80)
        bounds = ((5, 75), (5, 75), (5, 75))
        arr = _random_volume(raw_shape, seed=1)
        plan = build_streaming_inference_plan(
            requested_bounds=bounds, raw_volume_shape=raw_shape, minivol_size=MINIVOL
        )
        writer = _ArrayWriter(raw_shape, np.uint8, fill=LABEL_VALUES[0])
        # Timeout-guarded: a stop that fails to join the worker would otherwise
        # hang the test process rather than fail it.
        with self.assertRaises(StreamingInferenceStopRequested):
            _call_with_timeout(
                lambda: run_streaming_inference(
                    plan=plan,
                    raw_volume=_FakeVolume(arr),
                    normalization=StreamingNormalizationStats(0.0, 1.0, 1),
                    label_values=LABEL_VALUES,
                    output_dtype=np.uint8,
                    forward_minivol_batch=_mock_forward,
                    writer=writer,
                    should_stop=lambda: True,
                )
            )


class StreamingInferenceWorkerErrorTest(unittest.TestCase):
    def test_worker_write_exception_propagates_without_hang(self):
        # A writer failure happens on the accumulation worker thread; it must be
        # re-raised on the caller thread (as a RuntimeError) and never deadlock.
        raw_shape = (60, 60, 60)
        bounds = ((0, 60), (0, 60), (0, 60))
        arr = _random_volume(raw_shape, seed=5)
        plan = build_streaming_inference_plan(
            requested_bounds=bounds, raw_volume_shape=raw_shape, minivol_size=MINIVOL
        )
        writer = _RaisingWriter(message="boom write")
        with self.assertRaises(RuntimeError) as ctx:
            _call_with_timeout(
                lambda: run_streaming_inference(
                    plan=plan,
                    raw_volume=_FakeVolume(arr),
                    normalization=StreamingNormalizationStats(3.0, 2.0, 1),
                    label_values=LABEL_VALUES,
                    output_dtype=np.uint8,
                    forward_minivol_batch=_mock_forward,
                    writer=writer,
                )
            )
        self.assertIn("boom write", str(ctx.exception))
        self.assertGreaterEqual(writer.calls, 1)


class TorchMinivolForwardTest(unittest.TestCase):
    def test_torch_forward_integrates_on_cpu(self):
        import torch

        class _TinyNet(torch.nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.conv = torch.nn.Conv3d(1, num_classes, kernel_size=3, padding=1)

            def forward(self, x):
                return self.conv(x)

        torch.manual_seed(0)
        net = _TinyNet(NUM_CLASSES).eval()
        runtime = type("RT", (), {"model": net})()
        forward = build_torch_minivol_forward(runtime, device="cpu", autocast=False)

        raw_shape = (40, 40, 40)
        bounds = ((5, 35), (5, 35), (5, 35))
        arr = _random_volume(raw_shape, seed=2)
        plan = build_streaming_inference_plan(
            requested_bounds=bounds, raw_volume_shape=raw_shape, minivol_size=MINIVOL
        )
        writer = _ArrayWriter(raw_shape, np.uint8, fill=LABEL_VALUES[0])
        run_streaming_inference(
            plan=plan,
            raw_volume=_FakeVolume(arr),
            normalization=StreamingNormalizationStats(5.0, 3.0, 1),
            label_values=LABEL_VALUES,
            output_dtype=np.uint8,
            forward_minivol_batch=forward,
            writer=writer,
            batch_size=6,
        )
        dense = _dense_reference(
            plan, arr, mean=5.0, std=3.0, forward=forward,
            minivol_size=MINIVOL, label_values=LABEL_VALUES,
        )
        out = writer.array[
            bounds[0][0] : bounds[0][1],
            bounds[1][0] : bounds[1][1],
            bounds[2][0] : bounds[2][1],
        ]
        np.testing.assert_array_equal(out, dense)


if __name__ == "__main__":
    unittest.main()
