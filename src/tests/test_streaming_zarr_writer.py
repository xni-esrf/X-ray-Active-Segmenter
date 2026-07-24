from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from src.learning.streaming_inference import run_streaming_inference
from src.learning.streaming_inference_plan import build_streaming_inference_plan
from src.learning.streaming_zarr_writer import (
    StreamingZarrWriter,
    create_streaming_zarr_writer,
)
from src.learning.zero_occupancy import prepare_streaming_occupancy_and_stats


MINIVOL = 20
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

    def write_chunk(self, array, *, chunk_slices):
        self.array[chunk_slices] = array


def _mock_forward(batch: np.ndarray) -> np.ndarray:
    b = batch.shape[0]
    out = np.empty((b, NUM_CLASSES) + batch.shape[1:], dtype=np.float32)
    for c in range(NUM_CLASSES):
        out[:, c] = np.cos(batch * (0.5 + 0.37 * c) + 0.3 * c).astype(np.float32)
    return out


def _single_blob_volume(shape, seed=11):
    rng = np.random.default_rng(seed)
    arr = np.zeros(shape, dtype=np.float32)
    sl = tuple(slice(int(shape[a] * 0.42), int(shape[a] * 0.55)) for a in range(3))
    size = tuple(sl[a].stop - sl[a].start for a in range(3))
    arr[sl] = rng.uniform(4.0, 25.0, size=size).astype(np.float32)
    return arr


def _read_zarr(path):
    import zarr

    return np.asarray(zarr.open(path, mode="r")[:])


class StreamingZarrWriterUnitTest(unittest.TestCase):
    def test_creates_array_with_chunks_fill_and_dtype(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.zarr")
            writer = create_streaming_zarr_writer(
                path,
                shape=(50, 40, 30),
                dtype=np.uint16,
                chunks=(20, 20, 20),
                fill_value=0,
                overwrite=False,
            )
            self.assertEqual(tuple(writer.array.shape), (50, 40, 30))
            self.assertEqual(tuple(writer.array.chunks), (20, 20, 20))
            self.assertEqual(np.dtype(writer.array.dtype), np.uint16)
            block = np.ones((20, 20, 20), np.uint16) * 5
            writer.write_chunk(
                block, chunk_slices=(slice(0, 20), slice(0, 20), slice(0, 20))
            )
            data = _read_zarr(path)
            self.assertEqual(int(data[0, 0, 0]), 5)
            # unwritten region reads back as fill
            self.assertEqual(int(data[49, 39, 29]), 0)

    def test_fill_value_is_background_label(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.zarr")
            create_streaming_zarr_writer(
                path,
                shape=(40, 40, 40),
                dtype=np.uint8,
                chunks=(20, 20, 20),
                fill_value=7,
                overwrite=False,
            )
            data = _read_zarr(path)
            self.assertTrue(np.all(data == 7))

    def test_refuses_overwrite_without_flag(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.zarr")
            create_streaming_zarr_writer(
                path, shape=(20, 20, 20), dtype=np.uint8, chunks=(10, 10, 10),
                fill_value=0,
            )
            with self.assertRaises(FileExistsError):
                create_streaming_zarr_writer(
                    path, shape=(20, 20, 20), dtype=np.uint8, chunks=(10, 10, 10),
                    fill_value=0,
                )
            # overwrite clears stale content and recreates with the new shape
            writer = create_streaming_zarr_writer(
                path, shape=(10, 10, 10), dtype=np.uint8, chunks=(5, 5, 5),
                fill_value=0, overwrite=True,
            )
            self.assertEqual(tuple(writer.array.shape), (10, 10, 10))

    def test_chunks_are_clamped_to_shape(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.zarr")
            writer = create_streaming_zarr_writer(
                path, shape=(30, 15, 40), dtype=np.uint8, chunks=(20, 20, 20),
                fill_value=0,
            )
            # y axis chunk clamped from 20 to 15
            self.assertEqual(tuple(writer.array.chunks), (20, 15, 20))

    def test_shape_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.zarr")
            writer = create_streaming_zarr_writer(
                path, shape=(40, 40, 40), dtype=np.uint8, chunks=(20, 20, 20),
                fill_value=0,
            )
            with self.assertRaises(ValueError):
                writer.write_chunk(
                    np.zeros((10, 20, 20), np.uint8),
                    chunk_slices=(slice(0, 20), slice(0, 20), slice(0, 20)),
                )

    def test_rejects_non_integer_dtype(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.zarr")
            with self.assertRaises(TypeError):
                create_streaming_zarr_writer(
                    path, shape=(20, 20, 20), dtype=np.float32, chunks=(10, 10, 10),
                    fill_value=0,
                )


class StreamingZarrWriterIntegrationTest(unittest.TestCase):
    def test_zarr_output_matches_reference_writer(self):
        raw_shape = (120, 120, 120)
        bounds = ((0, 120), (0, 120), (0, 120))
        arr = _single_blob_volume(raw_shape)
        prepass = prepare_streaming_occupancy_and_stats(
            _FakeVolume(arr),
            requested_bounds=bounds,
            raw_volume_shape=raw_shape,
            minivol_size=MINIVOL,
        )
        plan = build_streaming_inference_plan(
            requested_bounds=bounds,
            raw_volume_shape=raw_shape,
            minivol_size=MINIVOL,
            occupancy=prepass.grid,
        )

        # reference: the already-validated in-memory array writer
        ref_writer = _ArrayWriter(raw_shape, np.uint8, fill=LABEL_VALUES[0])
        run_streaming_inference(
            plan=plan,
            raw_volume=_FakeVolume(arr),
            normalization=prepass.normalization,
            label_values=LABEL_VALUES,
            output_dtype=np.uint8,
            forward_minivol_batch=_mock_forward,
            writer=ref_writer,
            batch_size=5,
        )

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "seg.zarr")
            zarr_writer = StreamingZarrWriter(
                path,
                shape=raw_shape,
                dtype=np.uint8,
                chunks=(plan.chunk_size,) * 3,
                fill_value=LABEL_VALUES[0],
                overwrite=True,
            )
            run_streaming_inference(
                plan=plan,
                raw_volume=_FakeVolume(arr),
                normalization=prepass.normalization,
                label_values=LABEL_VALUES,
                output_dtype=np.uint8,
                forward_minivol_batch=_mock_forward,
                writer=zarr_writer,
                batch_size=5,
            )
            # chunk grid matches the plan's output chunk size
            self.assertEqual(tuple(zarr_writer.array.chunks), (plan.chunk_size,) * 3)
            data = _read_zarr(path)

        np.testing.assert_array_equal(data, ref_writer.array)
        # background is preserved as the fill value in genuinely empty regions
        self.assertEqual(int(data[0, -1, -1]), LABEL_VALUES[0])
        self.assertGreater(int((data != LABEL_VALUES[0]).sum()), 0)


if __name__ == "__main__":
    unittest.main()
