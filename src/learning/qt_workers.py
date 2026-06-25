"""Qt workers for long-running learning tasks.

This module owns background training/inference execution only. UI decisions,
thread wiring, volume loading, and applying predictions remain in MainWindow.
"""

from __future__ import annotations

from numbers import Integral
from threading import Event, Lock
from typing import Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from ..bbox import BoundingBox
from .session_store import (
    get_current_learning_model_runtime,
)
from .inference import (
    LearningInferenceBackgroundResult,
    LearningInferencePrediction,
    LearningInferenceStopRequested,
    run_learning_inference,
)


def _save_foundation_model_checkpoint(*, runtime: object, checkpoint_path: str) -> None:
    from .model_instantiation import save_foundation_model_checkpoint

    save_foundation_model_checkpoint(
        runtime=runtime,
        checkpoint_path=checkpoint_path,
    )


def _train_learning_model_with_validation_loop(
    *,
    preconditions: object,
    mixed_precision: bool,
    early_stop_patience: int,
    stop_event: Event,
) -> object:
    from .model_training import train_learning_model_with_validation_loop

    return train_learning_model_with_validation_loop(
        preconditions=preconditions,
        mixed_precision=mixed_precision,
        early_stop_patience=early_stop_patience,
        stop_event=stop_event,
    )


def _is_completion_checkpoint_result(result: object) -> bool:
    from .model_training import LearningTrainingLoopResult

    return isinstance(result, LearningTrainingLoopResult)


def _extract_bbox_context_from_array(*args, **kwargs) -> np.ndarray:
    from ..io.bbox_export_utils import extract_bbox_context_from_array

    return extract_bbox_context_from_array(*args, **kwargs)


def _plan_bbox_context(*args, **kwargs) -> object:
    from ..io.bbox_export_utils import plan_bbox_context

    return plan_bbox_context(*args, **kwargs)


def _build_inference_dataloader_runtime_from_entry(*args, **kwargs) -> object:
    from .eval_dataloader_builder import build_inference_dataloader_runtime_from_entry

    return build_inference_dataloader_runtime_from_entry(*args, **kwargs)


def _dispose_inference_runtime(runtime: object) -> Sequence[str]:
    from .eval_dataloader_builder import dispose_inference_runtime

    return dispose_inference_runtime(runtime)


class LearningTrainingWorker(QObject):
    completed = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, *, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._preconditions = None
        self._stop_event = Event()
        self._early_stop_patience = 2
        self._completion_checkpoint_path: Optional[str] = None
        self._completion_checkpoint_path_lock = Lock()

    def configure(
        self,
        *,
        preconditions: object,
        early_stop_patience: int = 2,
        completion_checkpoint_path: Optional[str] = None,
    ) -> None:
        self._preconditions = preconditions
        self._early_stop_patience = self._coerce_early_stop_patience(
            early_stop_patience
        )
        if completion_checkpoint_path is None:
            self.clear_completion_checkpoint_save_request()
        else:
            self.request_completion_checkpoint_save(completion_checkpoint_path)

    @staticmethod
    def _coerce_early_stop_patience(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(
                "early_stop_patience must be an integer, "
                f"got {type(value).__name__}"
            )
        normalized = int(value)
        if normalized < 1:
            raise ValueError("early_stop_patience must be >= 1")
        return normalized

    def request_stop(self) -> None:
        self._stop_event.set()

    def stop_requested(self) -> bool:
        return bool(self._stop_event.is_set())

    def request_completion_checkpoint_save(self, checkpoint_path: str) -> None:
        normalized_path = str(checkpoint_path).strip()
        if not normalized_path:
            raise ValueError("checkpoint_path must be a non-empty string")
        with self._completion_checkpoint_path_lock:
            self._completion_checkpoint_path = normalized_path

    def clear_completion_checkpoint_save_request(self) -> None:
        with self._completion_checkpoint_path_lock:
            self._completion_checkpoint_path = None

    def _completion_checkpoint_save_path(self) -> Optional[str]:
        with self._completion_checkpoint_path_lock:
            return self._completion_checkpoint_path

    def _completion_checkpoint_model_runtime(self) -> Optional[object]:
        runtime = getattr(self._preconditions, "model_runtime", None)
        if runtime is not None:
            return runtime
        return get_current_learning_model_runtime()

    def _maybe_save_completion_checkpoint(self, *, result: object) -> None:
        checkpoint_path = self._completion_checkpoint_save_path()
        if checkpoint_path is None:
            return
        if not _is_completion_checkpoint_result(result):
            return

        normalized_reason = str(result.stop_reason).strip().lower()
        if normalized_reason not in {"early_stop", "max_epoch"}:
            return

        runtime = self._completion_checkpoint_model_runtime()
        if runtime is None:
            raise RuntimeError(
                "No learning model runtime is available to save the completion checkpoint."
            )
        try:
            _save_foundation_model_checkpoint(
                runtime=runtime,
                checkpoint_path=checkpoint_path,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to save training completion checkpoint to "
                f"{checkpoint_path}: {exc}"
            ) from exc

    @Slot()
    def run(self) -> None:
        try:
            result = _train_learning_model_with_validation_loop(
                preconditions=self._preconditions,
                mixed_precision=True,
                early_stop_patience=int(self._early_stop_patience),
                stop_event=self._stop_event,
            )
            self._maybe_save_completion_checkpoint(result=result)
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class LearningInferenceWorker(QObject):
    completed = Signal(object)
    canceled = Signal(str)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, *, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._stop_event = Event()
        self._model_runtime: Optional[object] = None
        self._inference_boxes: Tuple[BoundingBox, ...] = tuple()
        self._raw_array: Optional[np.ndarray] = None
        self._label_values: Tuple[int, ...] = tuple()
        self._volume_shape: Tuple[int, int, int] = (1, 1, 1)

    def configure(
        self,
        *,
        model_runtime: object,
        inference_boxes: Sequence[BoundingBox],
        raw_array: np.ndarray,
        label_values: Sequence[int],
        volume_shape: Sequence[int],
    ) -> None:
        normalized_boxes: list[BoundingBox] = []
        for raw_box in tuple(inference_boxes):
            if not isinstance(raw_box, BoundingBox):
                raise TypeError(
                    "inference_boxes must contain BoundingBox instances only, "
                    f"got {type(raw_box).__name__}"
                )
            normalized_boxes.append(raw_box)
        if not normalized_boxes:
            raise ValueError("inference_boxes must contain at least one bounding box")

        normalized_labels: list[int] = []
        for raw_label in tuple(label_values):
            normalized_labels.append(int(raw_label))
        if not normalized_labels:
            raise ValueError("label_values must contain at least one class label")

        shape = tuple(int(v) for v in tuple(volume_shape))
        if len(shape) != 3:
            raise ValueError(f"volume_shape must be length 3, got {shape}")
        if any(axis <= 0 for axis in shape):
            raise ValueError(f"volume_shape axes must be positive, got {shape}")

        self._model_runtime = model_runtime
        self._inference_boxes = tuple(normalized_boxes)
        self._raw_array = np.asarray(raw_array)
        self._label_values = tuple(normalized_labels)
        self._volume_shape = (
            int(shape[0]),
            int(shape[1]),
            int(shape[2]),
        )

    def request_stop(self) -> None:
        self._stop_event.set()

    def stop_requested(self) -> bool:
        return bool(self._stop_event.is_set())

    def _raise_if_stop_requested(self) -> None:
        if self.stop_requested():
            raise LearningInferenceStopRequested("Inference stop requested by user.")

    def _configured_model_runtime(self) -> object:
        if self._model_runtime is None:
            raise RuntimeError(
                "Inference worker is not configured with a model runtime."
            )
        return self._model_runtime

    def _configured_raw_array(self) -> np.ndarray:
        if self._raw_array is None:
            raise RuntimeError(
                "Inference worker is not configured with raw input array."
            )
        return self._raw_array

    def _run_inference(self) -> LearningInferenceBackgroundResult:
        return run_learning_inference(
            model_runtime=self._configured_model_runtime(),
            inference_boxes=tuple(self._inference_boxes),
            raw_array=self._configured_raw_array(),
            label_values=tuple(self._label_values),
            volume_shape=tuple(self._volume_shape),
            stop_requested=self.stop_requested,
            extract_bbox_context_from_array_func=_extract_bbox_context_from_array,
            plan_bbox_context_func=_plan_bbox_context,
            build_inference_dataloader_runtime_from_entry_func=(
                _build_inference_dataloader_runtime_from_entry
            ),
            dispose_inference_runtime_func=_dispose_inference_runtime,
        )

    @Slot()
    def run(self) -> None:
        try:
            result = self._run_inference()
            self.completed.emit(result)
        except LearningInferenceStopRequested as exc:
            self.canceled.emit(str(exc))
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


__all__ = [
    "LearningInferenceBackgroundResult",
    "LearningInferencePrediction",
    "LearningInferenceStopRequested",
    "LearningInferenceWorker",
    "LearningTrainingWorker",
]
