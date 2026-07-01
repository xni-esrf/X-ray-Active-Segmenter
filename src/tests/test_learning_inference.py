from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.bbox import BoundingBox
from src.learning.inference import (
    LearningInferencePrediction,
    apply_inference_predictions_to_array,
    run_learning_inference,
)


class LearningInferenceApplicationTests(unittest.TestCase):
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
