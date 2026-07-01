from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.bbox import BoundingBox
from src.learning.inference import (
    LearningInferencePrediction,
    _AsyncInferenceAccumulator,
    _copy_prediction_batch_to_cpu,
    apply_inference_predictions_to_array,
    run_learning_inference,
)


class LearningInferenceApplicationTests(unittest.TestCase):
    def test_async_inference_accumulator_applies_jobs_before_finish_returns(self) -> None:
        calls: list[tuple[object, object]] = []
        buffer = SimpleNamespace(
            add_batch=lambda batch, coordinates: calls.append((batch, coordinates))
        )

        accumulator = _AsyncInferenceAccumulator(buffer, max_queue_size=1)
        try:
            accumulator.submit("batch-a", "coords-a")
            accumulator.submit("batch-b", "coords-b")
            accumulator.finish()
        finally:
            accumulator.close()

        self.assertEqual(
            calls,
            [
                ("batch-a", "coords-a"),
                ("batch-b", "coords-b"),
            ],
        )

    def test_async_inference_accumulator_reraises_worker_failure(self) -> None:
        def _raise(_batch: object, _coordinates: object) -> None:
            raise ValueError("bad accumulation")

        accumulator = _AsyncInferenceAccumulator(
            SimpleNamespace(add_batch=_raise),
            max_queue_size=1,
        )
        try:
            accumulator.submit("batch", "coords")
            with self.assertRaisesRegex(RuntimeError, "bad accumulation"):
                accumulator.finish()
        finally:
            accumulator.close()

    def test_async_inference_accumulator_rejects_non_positive_queue_size(self) -> None:
        buffer = SimpleNamespace(add_batch=lambda _batch, _coordinates: None)

        with self.assertRaisesRegex(ValueError, "max_queue_size must be >= 1"):
            _AsyncInferenceAccumulator(buffer, max_queue_size=0)

    def test_async_inference_accumulator_close_stops_worker_and_rejects_submit(self) -> None:
        accumulator = _AsyncInferenceAccumulator(
            SimpleNamespace(add_batch=lambda _batch, _coordinates: None),
            max_queue_size=1,
        )

        accumulator.close()

        self.assertFalse(accumulator._worker.is_alive())
        with self.assertRaisesRegex(RuntimeError, "closed"):
            accumulator.submit("batch", "coords")

    def test_async_inference_accumulator_bounded_queue_does_not_drop_jobs(self) -> None:
        calls: list[tuple[int, int]] = []
        accumulator = _AsyncInferenceAccumulator(
            SimpleNamespace(
                add_batch=lambda batch, coordinates: calls.append((batch, coordinates))
            ),
            max_queue_size=1,
        )
        try:
            for index in range(8):
                accumulator.submit(index, index + 100)
            accumulator.finish()
        finally:
            accumulator.close()

        self.assertEqual(calls, [(index, index + 100) for index in range(8)])

    def test_async_inference_accumulator_emits_debug_timing_logs(self) -> None:
        calls: list[tuple[object, object]] = []
        accumulator = _AsyncInferenceAccumulator(
            SimpleNamespace(
                add_batch=lambda batch, coordinates: calls.append((batch, coordinates))
            ),
            max_queue_size=1,
        )
        try:
            with self.assertLogs("src.learning.inference", level="DEBUG") as logs:
                accumulator.submit("batch", "coords")
                accumulator.finish()
        finally:
            accumulator.close()

        joined_logs = "\n".join(logs.output)
        self.assertIn("Inference async queue submit wait", joined_logs)
        self.assertIn("Inference async buffer accumulation", joined_logs)

    def test_copy_prediction_batch_to_cpu_detaches_before_cpu_copy(self) -> None:
        calls: list[str] = []

        class _FakePrediction:
            def detach(self) -> "_FakePrediction":
                calls.append("detach")
                return self

            def cpu(self) -> str:
                calls.append("cpu")
                return "cpu-copy"

        result = _copy_prediction_batch_to_cpu(_FakePrediction())

        self.assertEqual(result, "cpu-copy")
        self.assertEqual(calls, ["detach", "cpu"])

    def test_apply_inference_predictions_writes_bbox_and_reports_changed_voxels(self) -> None:
        segmentation = np.zeros((3, 3, 3), dtype=np.uint8)
        box = BoundingBox.from_bounds(
            box_id="infer-box",
            z0=1,
            z1=3,
            y0=0,
            y1=2,
            x0=1,
            x1=3,
            label="inference",
            volume_shape=segmentation.shape,
        )
        prediction = LearningInferencePrediction(
            box=box,
            predicted_bbox=np.ones((2, 2, 2), dtype=np.uint8),
        )

        output, changed_count, succeeded_ids, failures = apply_inference_predictions_to_array(
            segmentation,
            (prediction,),
        )

        expected = np.zeros((3, 3, 3), dtype=np.uint8)
        expected[1:3, 0:2, 1:3] = 1
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(segmentation, np.zeros((3, 3, 3), dtype=np.uint8))
        self.assertEqual(changed_count, 8)
        self.assertEqual(succeeded_ids, ("infer-box",))
        self.assertEqual(failures, {})

    def test_apply_inference_predictions_reports_shape_mismatch_per_box(self) -> None:
        segmentation = np.zeros((3, 3, 3), dtype=np.uint8)
        box = BoundingBox.from_bounds(
            box_id="infer-box",
            z0=0,
            z1=2,
            y0=0,
            y1=2,
            x0=0,
            x1=2,
            label="inference",
            volume_shape=segmentation.shape,
        )
        prediction = LearningInferencePrediction(
            box=box,
            predicted_bbox=np.ones((1, 2, 2), dtype=np.uint8),
        )

        output, changed_count, succeeded_ids, failures = apply_inference_predictions_to_array(
            segmentation,
            (prediction,),
        )

        np.testing.assert_array_equal(output, segmentation)
        self.assertEqual(changed_count, 0)
        self.assertEqual(succeeded_ids, ())
        self.assertIn("infer-box", failures)
        self.assertIn("Predicted bbox shape does not match bbox size", failures["infer-box"])

    def test_run_learning_inference_forwards_tiled_score_buffer_flag(self) -> None:
        box = BoundingBox.from_bounds(
            box_id="infer-box",
            z0=0,
            z1=1,
            y0=0,
            y1=1,
            x0=0,
            x1=1,
            label="inference",
            volume_shape=(1, 1, 1),
        )
        build_calls: list[dict[str, object]] = []

        class _FakeTensor:
            def __init__(self, array: object) -> None:
                self.array = np.asarray(array)
                self.device = SimpleNamespace(type="cpu")

            def to(self, *args: object, **kwargs: object) -> "_FakeTensor":
                return self

        class _FakeTorch:
            Tensor = _FakeTensor
            float16 = object()
            bfloat16 = object()
            int16 = object()
            cuda = SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)

            @staticmethod
            def from_numpy(array: np.ndarray) -> _FakeTensor:
                return _FakeTensor(array)

            @staticmethod
            def zeros(*args: object, **kwargs: object) -> _FakeTensor:
                del kwargs
                return _FakeTensor(np.zeros(args[0], dtype=np.int16))

            @staticmethod
            def device(name: str) -> object:
                return SimpleNamespace(type=str(name).split(":", 1)[0])

            @staticmethod
            def no_grad() -> object:
                return _null_context()

            @staticmethod
            def autocast(*args: object, **kwargs: object) -> object:
                del args, kwargs
                return _null_context()

        def build_runtime(entry: object, **kwargs: object) -> object:
            del entry
            build_calls.append(dict(kwargs))
            return SimpleNamespace(
                dataloader=tuple(),
                buffer=SimpleNamespace(
                    add_batch=lambda _batch, _coordinates: None,
                    get_pred_labels=lambda: np.zeros((1, 1, 1), dtype=np.int64),
                ),
            )

        with patch.dict(sys.modules, {"torch": _FakeTorch}):
            result = run_learning_inference(
                model_runtime=SimpleNamespace(
                    model=SimpleNamespace(
                        training=False,
                        eval=lambda: None,
                        parameters=lambda: iter(()),
                    )
                ),
                inference_boxes=(box,),
                raw_array=np.zeros((1, 1, 1), dtype=np.float32),
                label_values=(0, 1),
                volume_shape=(1, 1, 1),
                extract_bbox_context_from_array_func=(
                    lambda array, **_kwargs: np.asarray(array)
                ),
                plan_bbox_context_func=lambda **_kwargs: SimpleNamespace(
                    z=SimpleNamespace(extend_before=0, original_size=1),
                    y=SimpleNamespace(extend_before=0, original_size=1),
                    x=SimpleNamespace(extend_before=0, original_size=1),
                ),
                build_inference_dataloader_runtime_from_entry_func=build_runtime,
                dispose_inference_runtime_func=lambda _runtime: tuple(),
                use_tiled_score_buffer=True,
                tiled_temp_dir="/scratch/tiles",
            )

        self.assertEqual(len(build_calls), 1)
        self.assertIs(build_calls[0]["use_tiled_score_buffer"], True)
        self.assertEqual(build_calls[0]["tiled_temp_dir"], "/scratch/tiles")
        self.assertEqual(result.total_count, 1)
        self.assertEqual(
            tuple(prediction.box.id for prediction in result.predictions),
            ("infer-box",),
        )

    def test_run_learning_inference_async_accumulation_produces_prediction(self) -> None:
        box = BoundingBox.from_bounds(
            box_id="infer-box",
            z0=0,
            z1=1,
            y0=0,
            y1=1,
            x0=0,
            x1=1,
            label="inference",
            volume_shape=(1, 1, 1),
        )
        add_calls: list[tuple[object, object]] = []

        class _FakeTensor:
            def __init__(self, array: object) -> None:
                self.array = np.asarray(array)
                self.device = SimpleNamespace(type="cpu")

            def to(self, *args: object, **kwargs: object) -> "_FakeTensor":
                return self

            def detach(self) -> "_FakeTensor":
                return self

            def cpu(self) -> "_FakeTensor":
                return self

        class _FakeTorch:
            Tensor = _FakeTensor
            float16 = object()
            bfloat16 = object()
            int16 = object()
            cuda = SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)

            @staticmethod
            def from_numpy(array: np.ndarray) -> _FakeTensor:
                return _FakeTensor(array)

            @staticmethod
            def zeros(*args: object, **kwargs: object) -> _FakeTensor:
                del kwargs
                return _FakeTensor(np.zeros(args[0], dtype=np.int16))

            @staticmethod
            def device(name: str) -> object:
                return SimpleNamespace(type=str(name).split(":", 1)[0])

            @staticmethod
            def no_grad() -> object:
                return _null_context()

            @staticmethod
            def autocast(*args: object, **kwargs: object) -> object:
                del args, kwargs
                return _null_context()

        class _FakeModel:
            training = False

            @staticmethod
            def eval() -> None:
                return None

            @staticmethod
            def parameters() -> object:
                return iter(())

            def __call__(self, minivols: object) -> object:
                return minivols

        def build_runtime(entry: object, **kwargs: object) -> object:
            del entry, kwargs

            def _add_batch(batch: object, coordinates: object) -> None:
                add_calls.append((batch, coordinates))

            return SimpleNamespace(
                dataloader=((_FakeTensor(np.ones((1, 1, 1, 1, 1))), ("coords",)),),
                buffer=SimpleNamespace(
                    add_batch=_add_batch,
                    get_pred_labels=lambda: np.ones((1, 1, 1), dtype=np.int64),
                ),
            )

        with patch.dict(sys.modules, {"torch": _FakeTorch}):
            result = run_learning_inference(
                model_runtime=SimpleNamespace(
                    model=_FakeModel()
                ),
                inference_boxes=(box,),
                raw_array=np.zeros((1, 1, 1), dtype=np.float32),
                label_values=(0, 1),
                volume_shape=(1, 1, 1),
                extract_bbox_context_from_array_func=(
                    lambda array, **_kwargs: np.asarray(array)
                ),
                plan_bbox_context_func=lambda **_kwargs: SimpleNamespace(
                    z=SimpleNamespace(extend_before=0, original_size=1),
                    y=SimpleNamespace(extend_before=0, original_size=1),
                    x=SimpleNamespace(extend_before=0, original_size=1),
                ),
                build_inference_dataloader_runtime_from_entry_func=build_runtime,
                dispose_inference_runtime_func=lambda _runtime: tuple(),
                async_accumulation=True,
                async_accumulation_queue_size=1,
            )

        self.assertEqual(len(add_calls), 1)
        self.assertEqual(result.failure_by_box_id, {})
        self.assertEqual(len(result.predictions), 1)
        np.testing.assert_array_equal(
            result.predictions[0].predicted_bbox,
            np.ones((1, 1, 1), dtype=np.int64),
        )

    def test_run_learning_inference_async_accumulation_reports_buffer_failure(self) -> None:
        box = BoundingBox.from_bounds(
            box_id="infer-box",
            z0=0,
            z1=1,
            y0=0,
            y1=1,
            x0=0,
            x1=1,
            label="inference",
            volume_shape=(1, 1, 1),
        )

        class _FakeTensor:
            def __init__(self, array: object) -> None:
                self.array = np.asarray(array)
                self.device = SimpleNamespace(type="cpu")

            def to(self, *args: object, **kwargs: object) -> "_FakeTensor":
                return self

            def detach(self) -> "_FakeTensor":
                return self

            def cpu(self) -> "_FakeTensor":
                return self

        class _FakeTorch:
            Tensor = _FakeTensor
            float16 = object()
            bfloat16 = object()
            int16 = object()
            cuda = SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)

            @staticmethod
            def from_numpy(array: np.ndarray) -> _FakeTensor:
                return _FakeTensor(array)

            @staticmethod
            def zeros(*args: object, **kwargs: object) -> _FakeTensor:
                del kwargs
                return _FakeTensor(np.zeros(args[0], dtype=np.int16))

            @staticmethod
            def device(name: str) -> object:
                return SimpleNamespace(type=str(name).split(":", 1)[0])

            @staticmethod
            def no_grad() -> object:
                return _null_context()

            @staticmethod
            def autocast(*args: object, **kwargs: object) -> object:
                del args, kwargs
                return _null_context()

        class _FakeModel:
            training = False

            @staticmethod
            def eval() -> None:
                return None

            @staticmethod
            def parameters() -> object:
                return iter(())

            def __call__(self, minivols: object) -> object:
                return minivols

        def build_runtime(entry: object, **kwargs: object) -> object:
            del entry, kwargs

            def _add_batch(_batch: object, _coordinates: object) -> None:
                raise ValueError("buffer exploded")

            return SimpleNamespace(
                dataloader=((_FakeTensor(np.ones((1, 1, 1, 1, 1))), ("coords",)),),
                buffer=SimpleNamespace(
                    add_batch=_add_batch,
                    get_pred_labels=lambda: np.ones((1, 1, 1), dtype=np.int64),
                ),
            )

        with patch.dict(sys.modules, {"torch": _FakeTorch}):
            result = run_learning_inference(
                model_runtime=SimpleNamespace(
                    model=_FakeModel()
                ),
                inference_boxes=(box,),
                raw_array=np.zeros((1, 1, 1), dtype=np.float32),
                label_values=(0, 1),
                volume_shape=(1, 1, 1),
                extract_bbox_context_from_array_func=(
                    lambda array, **_kwargs: np.asarray(array)
                ),
                plan_bbox_context_func=lambda **_kwargs: SimpleNamespace(
                    z=SimpleNamespace(extend_before=0, original_size=1),
                    y=SimpleNamespace(extend_before=0, original_size=1),
                    x=SimpleNamespace(extend_before=0, original_size=1),
                ),
                build_inference_dataloader_runtime_from_entry_func=build_runtime,
                dispose_inference_runtime_func=lambda _runtime: tuple(),
                async_accumulation=True,
                async_accumulation_queue_size=1,
            )

        self.assertEqual(result.predictions, ())
        self.assertIn("infer-box", result.failure_by_box_id)
        self.assertIn("buffer exploded", result.failure_by_box_id["infer-box"])

    def test_run_learning_inference_async_accumulation_drains_before_stop_after_submit(self) -> None:
        box = BoundingBox.from_bounds(
            box_id="infer-box",
            z0=0,
            z1=1,
            y0=0,
            y1=1,
            x0=0,
            x1=1,
            label="inference",
            volume_shape=(1, 1, 1),
        )
        add_calls: list[tuple[object, object]] = []
        stop_call_count = 0

        class _FakeTensor:
            def __init__(self, array: object) -> None:
                self.array = np.asarray(array)
                self.device = SimpleNamespace(type="cpu")

            def to(self, *args: object, **kwargs: object) -> "_FakeTensor":
                return self

            def detach(self) -> "_FakeTensor":
                return self

            def cpu(self) -> "_FakeTensor":
                return self

        class _FakeTorch:
            Tensor = _FakeTensor
            float16 = object()
            bfloat16 = object()
            int16 = object()
            cuda = SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)

            @staticmethod
            def from_numpy(array: np.ndarray) -> _FakeTensor:
                return _FakeTensor(array)

            @staticmethod
            def zeros(*args: object, **kwargs: object) -> _FakeTensor:
                del kwargs
                return _FakeTensor(np.zeros(args[0], dtype=np.int16))

            @staticmethod
            def device(name: str) -> object:
                return SimpleNamespace(type=str(name).split(":", 1)[0])

            @staticmethod
            def no_grad() -> object:
                return _null_context()

            @staticmethod
            def autocast(*args: object, **kwargs: object) -> object:
                del args, kwargs
                return _null_context()

        class _FakeModel:
            training = False

            @staticmethod
            def eval() -> None:
                return None

            @staticmethod
            def parameters() -> object:
                return iter(())

            def __call__(self, minivols: object) -> object:
                return minivols

        def build_runtime(entry: object, **kwargs: object) -> object:
            del entry, kwargs

            def _add_batch(batch: object, coordinates: object) -> None:
                add_calls.append((batch, coordinates))

            return SimpleNamespace(
                dataloader=((_FakeTensor(np.ones((1, 1, 1, 1, 1))), ("coords",)),),
                buffer=SimpleNamespace(
                    add_batch=_add_batch,
                    get_pred_labels=lambda: np.ones((1, 1, 1), dtype=np.int64),
                ),
            )

        def stop_requested() -> bool:
            nonlocal stop_call_count
            stop_call_count += 1
            return stop_call_count >= 4

        with patch.dict(sys.modules, {"torch": _FakeTorch}):
            with self.assertRaisesRegex(
                Exception,
                "Inference stop requested by user",
            ):
                run_learning_inference(
                    model_runtime=SimpleNamespace(model=_FakeModel()),
                    inference_boxes=(box,),
                    raw_array=np.zeros((1, 1, 1), dtype=np.float32),
                    label_values=(0, 1),
                    volume_shape=(1, 1, 1),
                    stop_requested=stop_requested,
                    extract_bbox_context_from_array_func=(
                        lambda array, **_kwargs: np.asarray(array)
                    ),
                    plan_bbox_context_func=lambda **_kwargs: SimpleNamespace(
                        z=SimpleNamespace(extend_before=0, original_size=1),
                        y=SimpleNamespace(extend_before=0, original_size=1),
                        x=SimpleNamespace(extend_before=0, original_size=1),
                    ),
                    build_inference_dataloader_runtime_from_entry_func=build_runtime,
                    dispose_inference_runtime_func=lambda _runtime: tuple(),
                    async_accumulation=True,
                    async_accumulation_queue_size=1,
                )

        self.assertEqual(len(add_calls), 1)

    def test_run_learning_inference_rejects_invalid_async_accumulation_options(self) -> None:
        box = BoundingBox.from_bounds(
            box_id="infer-box",
            z0=0,
            z1=1,
            y0=0,
            y1=1,
            x0=0,
            x1=1,
            label="inference",
            volume_shape=(1, 1, 1),
        )

        class _FakeTorch:
            Tensor = object

        common_kwargs = dict(
            model_runtime=SimpleNamespace(
                model=SimpleNamespace(
                    training=False,
                    eval=lambda: None,
                    parameters=lambda: iter(()),
                )
            ),
            inference_boxes=(box,),
            raw_array=np.zeros((1, 1, 1), dtype=np.float32),
            label_values=(0, 1),
            volume_shape=(1, 1, 1),
        )

        with patch.dict(sys.modules, {"torch": _FakeTorch}):
            with self.assertRaisesRegex(TypeError, "async_accumulation must be a bool"):
                run_learning_inference(
                    **common_kwargs,
                    async_accumulation="yes",
                )

            with self.assertRaisesRegex(
                ValueError,
                "async_accumulation_queue_size must be >= 1",
            ):
                run_learning_inference(
                    **common_kwargs,
                    async_accumulation_queue_size=0,
                )


def _null_context() -> object:
    class _Context:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
            del exc_type, exc, traceback
            return False

    return _Context()


if __name__ == "__main__":
    unittest.main()
