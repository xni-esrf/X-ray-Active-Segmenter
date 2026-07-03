from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from src.learning.inference import LearningInferencePrediction
from src.learning.large_crop_inference import (
    run_large_crop_inference_to_zarr,
    run_one_crop_large_crop_inference_to_zarr,
)


class _FakeVolume:
    def __init__(self, array: np.ndarray) -> None:
        self.array = np.asarray(array)
        self.shape = tuple(int(axis) for axis in self.array.shape)

    def get_chunk(self, zyx_slices: tuple[slice, slice, slice]) -> np.ndarray:
        return self.array[zyx_slices]


class _FakeWriter:
    def __init__(
        self,
        path: str,
        *,
        shape: tuple[int, int, int],
        dtype: np.dtype,
        chunks: tuple[int, int, int] | None = None,
        overwrite: bool = False,
    ) -> None:
        self.path = str(path)
        self.shape = tuple(int(axis) for axis in shape)
        self.dtype = np.dtype(dtype)
        self.chunks = chunks
        self.overwrite = bool(overwrite)
        self.output = np.zeros(self.shape, dtype=self.dtype)
        self.write_calls: list[tuple[object, int | None, int | None]] = []

    def write_window_prediction(
        self,
        prediction_crop: np.ndarray,
        *,
        window: object,
        crop_number: int | None = None,
        total_crops: int | None = None,
    ) -> bool:
        self.write_calls.append((window, crop_number, total_crops))
        crop = np.asarray(prediction_crop)
        self.output[window.requested_output_slices] = crop[
            window.requested_output_slices_in_crop
        ]
        return True


class LargeCropInferenceTests(unittest.TestCase):
    def test_one_crop_output_matches_full_dense_baseline_with_context(self) -> None:
        raw = np.arange(8 * 9 * 10, dtype=np.uint16).reshape((8, 9, 10))
        writer_holder: dict[str, _FakeWriter] = {}

        def fake_writer_factory(path: str, **kwargs: object) -> _FakeWriter:
            writer = _FakeWriter(path, **kwargs)
            writer_holder["writer"] = writer
            return writer

        def fake_run_inference(**kwargs: object) -> object:
            raw_array = np.asarray(kwargs["raw_array"])
            box = tuple(kwargs["inference_boxes"])[0]
            return SimpleNamespace(
                total_count=1,
                predictions=(
                    LearningInferencePrediction(
                        box=box,
                        predicted_bbox=(raw_array.astype(np.uint16, copy=False) + 17),
                    ),
                ),
                failure_by_box_id={},
                cleanup_errors_by_box_id={},
            )

        result = run_one_crop_large_crop_inference_to_zarr(
            model_runtime=object(),
            raw_volume=_FakeVolume(raw),
            requested_bounds=((1, 5), (2, 6), (3, 7)),
            label_values=(0, 1),
            output_path="/tmp/out.zarr",
            output_dtype=np.uint16,
            context_margin=2,
            minivol_size=4,
            writer_factory=fake_writer_factory,
            run_learning_inference_func=fake_run_inference,
        )

        self.assertFalse(result.plan.requires_cropping)
        old_full_dense_baseline = raw[1:5, 2:6, 3:7].astype(np.uint16) + 17
        np.testing.assert_array_equal(
            writer_holder["writer"].output,
            old_full_dense_baseline,
        )

    def test_multi_crop_output_matches_full_dense_baseline(self) -> None:
        raw = np.arange(16 * 4 * 4, dtype=np.uint16).reshape((16, 4, 4))
        writer_holder: dict[str, _FakeWriter] = {}

        def fake_writer_factory(path: str, **kwargs: object) -> _FakeWriter:
            writer = _FakeWriter(path, **kwargs)
            writer_holder["writer"] = writer
            return writer

        def fake_run_inference(**kwargs: object) -> object:
            raw_array = np.asarray(kwargs["raw_array"])
            box = tuple(kwargs["inference_boxes"])[0]
            return SimpleNamespace(
                total_count=1,
                predictions=(
                    LearningInferencePrediction(
                        box=box,
                        predicted_bbox=(raw_array.astype(np.uint16, copy=False) + 23),
                    ),
                ),
                failure_by_box_id={},
                cleanup_errors_by_box_id={},
            )

        result = run_large_crop_inference_to_zarr(
            model_runtime=object(),
            raw_volume=_FakeVolume(raw),
            requested_bounds=((0, 16), (0, 4), (0, 4)),
            label_values=(0, 1),
            output_path="/tmp/out.zarr",
            output_dtype=np.uint16,
            context_margin=0,
            minivol_size=4,
            voxel_budget=10 * 4 * 4,
            writer_factory=fake_writer_factory,
            run_learning_inference_func=fake_run_inference,
        )

        self.assertTrue(result.plan.requires_cropping)
        self.assertGreater(result.plan.total_crop_count, 1)
        old_full_dense_baseline = raw.astype(np.uint16) + 23
        np.testing.assert_array_equal(
            writer_holder["writer"].output,
            old_full_dense_baseline,
        )

    def test_multi_crop_path_runs_each_crop_and_stitches_output(self) -> None:
        raw = np.arange(16 * 4 * 4, dtype=np.float32).reshape((16, 4, 4))
        volume = _FakeVolume(raw)
        run_calls: list[dict[str, object]] = []
        writer_holder: dict[str, _FakeWriter] = {}

        def fake_writer_factory(path: str, **kwargs: object) -> _FakeWriter:
            writer = _FakeWriter(path, **kwargs)
            writer_holder["writer"] = writer
            return writer

        def fake_run_inference(**kwargs: object) -> object:
            run_calls.append(dict(kwargs))
            raw_array = np.asarray(kwargs["raw_array"])
            box = tuple(kwargs["inference_boxes"])[0]
            value = len(run_calls)
            prediction = np.full(raw_array.shape, value, dtype=np.uint8)
            return SimpleNamespace(
                total_count=1,
                predictions=(
                    LearningInferencePrediction(
                        box=box,
                        predicted_bbox=prediction,
                    ),
                ),
                failure_by_box_id={},
                cleanup_errors_by_box_id={},
            )

        result = run_large_crop_inference_to_zarr(
            model_runtime=object(),
            raw_volume=volume,
            requested_bounds=((0, 16), (0, 4), (0, 4)),
            label_values=(0, 1),
            output_path="/tmp/out.zarr",
            context_margin=0,
            minivol_size=4,
            voxel_budget=10 * 4 * 4,
            writer_factory=fake_writer_factory,
            run_learning_inference_func=fake_run_inference,
        )

        self.assertTrue(result.plan.requires_cropping)
        self.assertEqual(result.plan.total_crop_count, 3)
        self.assertEqual(result.written_crop_count, 3)
        self.assertEqual(len(run_calls), 3)
        writer = writer_holder["writer"]
        expected = np.zeros(result.plan.requested_shape, dtype=np.uint8)
        for window in result.plan.windows:
            expected[window.requested_output_slices] = int(window.index) + 1
        np.testing.assert_array_equal(writer.output, expected)
        self.assertEqual(
            [call[1] for call in writer.write_calls],
            [1, 2, 3],
        )
        self.assertEqual(
            [call[2] for call in writer.write_calls],
            [3, 3, 3],
        )

    def test_multi_crop_path_logs_crop_progress(self) -> None:
        raw = np.zeros((16, 4, 4), dtype=np.float32)

        def fake_run_inference(**kwargs: object) -> object:
            raw_array = np.asarray(kwargs["raw_array"])
            box = tuple(kwargs["inference_boxes"])[0]
            progress_callback = kwargs["progress_callback"]
            progress_callback(
                SimpleNamespace(
                    completed_count=1,
                    total_count=1,
                    box_id=str(box.id),
                    succeeded=True,
                    failed_count=0,
                )
            )
            return SimpleNamespace(
                total_count=1,
                predictions=(
                    LearningInferencePrediction(
                        box=box,
                        predicted_bbox=np.zeros(raw_array.shape, dtype=np.uint8),
                    ),
                ),
                failure_by_box_id={},
                cleanup_errors_by_box_id={},
            )

        with self.assertLogs("src.learning.large_crop_inference", level="INFO") as logs:
            run_large_crop_inference_to_zarr(
                model_runtime=object(),
                raw_volume=_FakeVolume(raw),
                requested_bounds=((0, 16), (0, 4), (0, 4)),
                label_values=(0, 1),
                output_path="/tmp/out.zarr",
                context_margin=0,
                minivol_size=4,
                voxel_budget=10 * 4 * 4,
                writer_factory=lambda path, **kwargs: _FakeWriter(path, **kwargs),
                run_learning_inference_func=fake_run_inference,
            )

        joined = "\n".join(logs.output)
        self.assertIn("Large-crop inference started: total_crops=3", joined)
        self.assertIn("Large-crop inference plan: raw_shape=(16, 4, 4)", joined)
        self.assertIn("Large-crop inference parameters: context_margin=0", joined)
        self.assertIn("Processing large crop 1/3", joined)
        self.assertIn("Extracting large crop 1/3", joined)
        self.assertIn("Extracted large crop 1/3", joined)
        self.assertIn("Running dense inference for large crop 1/3", joined)
        self.assertIn("Dense inference progress for large crop 1/3", joined)
        self.assertIn("Dense inference returned for large crop 1/3", joined)
        self.assertIn("Dense inference prediction ready for large crop 1/3", joined)
        self.assertIn("Finished large crop 3/3", joined)
        self.assertIn("Large-crop inference completed: total_crops=3", joined)

    def test_one_crop_path_runs_dense_inference_and_writes_requested_output(self) -> None:
        raw = np.arange(8 * 9 * 10, dtype=np.float32).reshape((8, 9, 10))
        volume = _FakeVolume(raw)
        run_calls: list[dict[str, object]] = []
        writer_holder: dict[str, _FakeWriter] = {}

        def fake_writer_factory(path: str, **kwargs: object) -> _FakeWriter:
            writer = _FakeWriter(path, **kwargs)
            writer_holder["writer"] = writer
            return writer

        def fake_run_inference(**kwargs: object) -> object:
            run_calls.append(dict(kwargs))
            raw_array = np.asarray(kwargs["raw_array"])
            box = tuple(kwargs["inference_boxes"])[0]
            prediction = np.arange(np.prod(raw_array.shape), dtype=np.uint16).reshape(
                raw_array.shape
            )
            return SimpleNamespace(
                total_count=1,
                predictions=(
                    LearningInferencePrediction(
                        box=box,
                        predicted_bbox=prediction,
                    ),
                ),
                failure_by_box_id={},
                cleanup_errors_by_box_id={},
            )

        runtime = object()
        result = run_one_crop_large_crop_inference_to_zarr(
            model_runtime=runtime,
            raw_volume=volume,
            requested_bounds=((0, 4), (1, 5), (2, 6)),
            label_values=(0, 1),
            output_path="/tmp/out.zarr",
            output_dtype=np.uint16,
            overwrite=True,
            batch_size=7,
            context_margin=2,
            minivol_size=4,
            writer_factory=fake_writer_factory,
            run_learning_inference_func=fake_run_inference,
        )

        self.assertEqual(result.written_crop_count, 1)
        self.assertFalse(result.plan.requires_cropping)
        self.assertEqual(len(run_calls), 1)
        call = run_calls[0]
        self.assertIs(call["model_runtime"], runtime)
        self.assertEqual(call["label_values"], (0, 1))
        self.assertEqual(call["volume_shape"], result.plan.windows[0].crop_shape)
        self.assertEqual(call["batch_size"], 7)
        self.assertIs(call["use_tiled_score_buffer"], False)
        self.assertIs(call["async_accumulation"], False)
        self.assertTrue(callable(call["extract_bbox_context_from_array_func"]))
        self.assertTrue(callable(call["plan_bbox_context_func"]))

        writer = writer_holder["writer"]
        self.assertEqual(writer.shape, result.plan.requested_shape)
        self.assertEqual(writer.dtype, np.dtype(np.uint16))
        self.assertTrue(writer.overwrite)
        self.assertEqual(len(writer.write_calls), 1)
        prediction_crop = np.arange(
            np.prod(result.plan.windows[0].crop_shape),
            dtype=np.uint16,
        ).reshape(result.plan.windows[0].crop_shape)
        np.testing.assert_array_equal(
            writer.output,
            prediction_crop[result.plan.windows[0].requested_output_slices_in_crop],
        )

    def test_one_crop_path_rejects_multi_crop_plan(self) -> None:
        raw = np.zeros((16, 4, 4), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "one-crop plan"):
            run_one_crop_large_crop_inference_to_zarr(
                model_runtime=object(),
                raw_volume=_FakeVolume(raw),
                requested_bounds=((0, 16), (0, 4), (0, 4)),
                label_values=(0, 1),
                output_path="/tmp/out.zarr",
                context_margin=0,
                minivol_size=4,
                voxel_budget=10 * 4 * 4,
                writer_factory=lambda *args, **kwargs: None,
                run_learning_inference_func=lambda **_kwargs: None,
            )

    def test_one_crop_path_reports_inference_failure_before_writing(self) -> None:
        raw = np.zeros((4, 4, 4), dtype=np.float32)
        writer_holder: dict[str, _FakeWriter] = {}

        def fake_writer_factory(path: str, **kwargs: object) -> _FakeWriter:
            writer = _FakeWriter(path, **kwargs)
            writer_holder["writer"] = writer
            return writer

        def fake_run_inference(**_kwargs: object) -> object:
            return SimpleNamespace(
                total_count=1,
                predictions=tuple(),
                failure_by_box_id={"large-crop-0": "bad inference"},
                cleanup_errors_by_box_id={},
            )

        with self.assertRaisesRegex(RuntimeError, "bad inference"):
            run_one_crop_large_crop_inference_to_zarr(
                model_runtime=object(),
                raw_volume=_FakeVolume(raw),
                requested_bounds=((0, 4), (0, 4), (0, 4)),
                label_values=(0, 1),
                output_path="/tmp/out.zarr",
                minivol_size=4,
                writer_factory=fake_writer_factory,
                run_learning_inference_func=fake_run_inference,
            )

        self.assertEqual(writer_holder["writer"].write_calls, [])


if __name__ == "__main__":
    unittest.main()
