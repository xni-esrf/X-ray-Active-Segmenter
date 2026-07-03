from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from src.learning.large_crop_inference_plan import build_large_crop_inference_plan
from src.learning.large_crop_zarr_writer import LargeCropZarrOutputWriter


class _MemoryZarrArray:
    def __init__(
        self,
        _path: str,
        *,
        shape: tuple[int, int, int],
        dtype: np.dtype,
        chunks: tuple[int, int, int] | None = None,
    ) -> None:
        self.shape = tuple(int(axis) for axis in shape)
        self.dtype = np.dtype(dtype)
        self.chunks = None if chunks is None else tuple(int(axis) for axis in chunks)
        self.data = np.zeros(self.shape, dtype=self.dtype)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value) -> None:
        self.data[key] = value


class _MemoryZarrFactory:
    def __init__(self) -> None:
        self.arrays: dict[str, _MemoryZarrArray] = {}

    def __call__(
        self,
        path: str,
        *,
        shape: tuple[int, int, int],
        dtype: np.dtype,
        chunks: tuple[int, int, int] | None = None,
    ) -> _MemoryZarrArray:
        array = _MemoryZarrArray(path, shape=shape, dtype=dtype, chunks=chunks)
        self.arrays[path] = array
        return array


class LargeCropZarrOutputWriterTests(unittest.TestCase):
    def test_write_slices_creates_zarr_and_writes_region(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "output.zarr")
            factory = _MemoryZarrFactory()
            writer = LargeCropZarrOutputWriter(
                path,
                shape=(6, 7, 8),
                dtype=np.uint8,
                chunks=(3, 4, 4),
                array_factory=factory,
            )

            data = np.arange(2 * 3 * 4, dtype=np.uint8).reshape((2, 3, 4))
            writer.write_slices(
                data,
                destination_slices=(slice(1, 3), slice(2, 5), slice(3, 7)),
            )

            arr = factory.arrays[path]
            self.assertEqual(tuple(arr.shape), (6, 7, 8))
            self.assertEqual(np.dtype(arr.dtype), np.dtype(np.uint8))
            self.assertEqual(tuple(arr.chunks), (3, 4, 4))
            expected = np.zeros((6, 7, 8), dtype=np.uint8)
            expected[1:3, 2:5, 3:7] = data
            np.testing.assert_array_equal(arr.data, expected)

    def test_default_writer_creates_readable_uncompressed_zarr_v2_directory(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output.zarr"
            writer = LargeCropZarrOutputWriter(
                str(path),
                shape=(4, 5, 6),
                dtype=np.uint16,
                chunks=(2, 3, 4),
            )

            data = np.arange(3 * 3 * 3, dtype=np.uint16).reshape((3, 3, 3))
            writer.write_slices(
                data,
                destination_slices=(slice(1, 4), slice(2, 5), slice(3, 6)),
            )

            metadata = json.loads((path / ".zarray").read_text(encoding="utf-8"))
            self.assertEqual(metadata["zarr_format"], 2)
            self.assertEqual(metadata["shape"], [4, 5, 6])
            self.assertEqual(metadata["chunks"], [2, 3, 4])
            self.assertIsNone(metadata["compressor"])
            self.assertTrue((path / "0.0.0").exists())
            self.assertTrue((path / "1.1.1").exists())

            expected = np.zeros((4, 5, 6), dtype=np.uint16)
            expected[1:4, 2:5, 3:6] = data
            np.testing.assert_array_equal(
                writer.array[(slice(None), slice(None), slice(None))],
                expected,
            )

    def test_write_slices_casts_to_output_dtype(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "output.zarr")
            factory = _MemoryZarrFactory()
            writer = LargeCropZarrOutputWriter(
                path,
                shape=(2, 2, 2),
                dtype=np.uint16,
                array_factory=factory,
            )

            writer.write_slices(
                np.ones((2, 2, 2), dtype=np.uint8),
                destination_slices=(slice(0, 2), slice(0, 2), slice(0, 2)),
            )

            arr = factory.arrays[path]
            self.assertEqual(np.dtype(arr.dtype), np.dtype(np.uint16))
            np.testing.assert_array_equal(
                arr.data,
                np.ones((2, 2, 2), dtype=np.uint16),
            )

    def test_write_window_prediction_stitches_requested_output_intersection(self) -> None:
        plan = build_large_crop_inference_plan(
            requested_bounds=((0, 16), (0, 4), (0, 4)),
            raw_volume_shape=(16, 4, 4),
            context_margin=0,
            minivol_size=4,
            voxel_budget=10 * 4 * 4,
        )

        with TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "output.zarr")
            factory = _MemoryZarrFactory()
            writer = LargeCropZarrOutputWriter(
                path,
                shape=plan.requested_shape,
                dtype=np.uint8,
                array_factory=factory,
            )

            expected = np.zeros(plan.requested_shape, dtype=np.uint8)
            for window in plan.windows:
                value = int(window.index) + 1
                prediction = np.full(window.crop_shape, value, dtype=np.uint8)
                self.assertTrue(
                    writer.write_window_prediction(
                        prediction,
                        window=window,
                        crop_number=int(window.index) + 1,
                        total_crops=plan.total_crop_count,
                    )
                )
                expected[window.requested_output_slices] = value

            arr = factory.arrays[path].data
            np.testing.assert_array_equal(arr, expected)

    def test_write_window_prediction_logs_stitching_progress(self) -> None:
        plan = build_large_crop_inference_plan(
            requested_bounds=((0, 4), (0, 4), (0, 4)),
            raw_volume_shape=(4, 4, 4),
            context_margin=0,
            minivol_size=4,
        )

        with TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "output.zarr")
            factory = _MemoryZarrFactory()
            writer = LargeCropZarrOutputWriter(
                path,
                shape=plan.requested_shape,
                dtype=np.uint8,
                array_factory=factory,
            )
            prediction = np.ones(plan.windows[0].crop_shape, dtype=np.uint8)

            with self.assertLogs("src.learning.large_crop_zarr_writer", level="INFO") as logs:
                writer.write_window_prediction(
                    prediction,
                    window=plan.windows[0],
                    crop_number=1,
                    total_crops=1,
                )

        joined = "\n".join(logs.output)
        self.assertIn("Writing/stitching large crop 1/1", joined)
        self.assertIn("Finished writing/stitching large crop 1/1", joined)

    def test_rejects_existing_output_without_overwrite(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output.zarr"
            path.mkdir()

            with self.assertRaises(FileExistsError):
                LargeCropZarrOutputWriter(
                    str(path),
                    shape=(2, 2, 2),
                    dtype=np.uint8,
                )

            writer = LargeCropZarrOutputWriter(
                str(path),
                shape=(2, 2, 2),
                dtype=np.uint8,
                overwrite=True,
                array_factory=_MemoryZarrFactory(),
            )
            self.assertEqual(tuple(writer.array.shape), (2, 2, 2))

    def test_default_writer_overwrite_removes_stale_chunks(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output.zarr"
            path.mkdir()
            stale_chunk = path / "99.99.99"
            stale_chunk.write_bytes(b"stale")

            writer = LargeCropZarrOutputWriter(
                str(path),
                shape=(2, 2, 2),
                dtype=np.uint8,
                overwrite=True,
            )

            self.assertFalse(stale_chunk.exists())
            writer.write_slices(
                np.ones((2, 2, 2), dtype=np.uint8),
                destination_slices=(slice(0, 2), slice(0, 2), slice(0, 2)),
            )
            self.assertTrue((path / "0.0.0").exists())

    def test_rejects_invalid_write_shapes_and_sources(self) -> None:
        plan = build_large_crop_inference_plan(
            requested_bounds=((0, 4), (0, 4), (0, 4)),
            raw_volume_shape=(4, 4, 4),
            context_margin=0,
            minivol_size=4,
        )

        with TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "output.zarr")
            factory = _MemoryZarrFactory()
            writer = LargeCropZarrOutputWriter(
                path,
                shape=(4, 4, 4),
                dtype=np.uint8,
                array_factory=factory,
            )

            with self.assertRaisesRegex(ValueError, "data shape"):
                writer.write_slices(
                    np.zeros((1, 1, 1), dtype=np.uint8),
                    destination_slices=(slice(0, 2), slice(0, 2), slice(0, 2)),
                )
            with self.assertRaisesRegex(ValueError, "prediction_crop must be 3D"):
                writer.write_window_prediction(
                    np.zeros((4, 4), dtype=np.uint8),
                    window=plan.windows[0],
                )
            with self.assertRaisesRegex(ValueError, "planned crop shape"):
                writer.write_window_prediction(
                    np.zeros((2, 2, 2), dtype=np.uint8),
                    window=plan.windows[0],
                )


if __name__ == "__main__":
    unittest.main()
