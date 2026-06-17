"""Qt workers for long-running learning tasks.

This module owns background training/inference execution only. UI decisions,
thread wiring, volume loading, and applying predictions remain in MainWindow.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Lock
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from ..bbox import BoundingBox
from ..utils import exception_message, torch_from_numpy_safe
from .session_store import (
    LearningBBoxTensorEntry,
    get_current_learning_model_runtime,
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


def _build_inference_dataloader_runtime_from_entry(
    *,
    entry: LearningBBoxTensorEntry,
    label_values: Sequence[int],
    minivol_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
):
    from .eval_dataloader_builder import build_inference_dataloader_runtime_from_entry

    return build_inference_dataloader_runtime_from_entry(
        entry,
        label_values=label_values,
        minivol_size=minivol_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def _dispose_inference_runtime(runtime: object) -> Tuple[str, ...]:
    from .eval_dataloader_builder import dispose_inference_runtime

    return tuple(dispose_inference_runtime(runtime))


def _extract_bbox_context_from_array(
    raw_array: np.ndarray,
    *,
    z_bounds: Tuple[int, int],
    y_bounds: Tuple[int, int],
    x_bounds: Tuple[int, int],
) -> np.ndarray:
    from ..io.bbox_export_utils import extract_bbox_context_from_array

    return extract_bbox_context_from_array(
        raw_array,
        z_bounds=z_bounds,
        y_bounds=y_bounds,
        x_bounds=x_bounds,
    )


def _plan_bbox_context(
    *,
    z_bounds: Tuple[int, int],
    y_bounds: Tuple[int, int],
    x_bounds: Tuple[int, int],
    volume_shape: Tuple[int, int, int],
):
    from ..io.bbox_export_utils import plan_bbox_context

    return plan_bbox_context(
        z_bounds=z_bounds,
        y_bounds=y_bounds,
        x_bounds=x_bounds,
        volume_shape=volume_shape,
    )


class LearningTrainingWorker(QObject):
    completed = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, *, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._preconditions = None
        self._stop_event = Event()
        self._completion_checkpoint_path: Optional[str] = None
        self._completion_checkpoint_path_lock = Lock()

    def configure(
        self,
        *,
        preconditions: object,
        completion_checkpoint_path: Optional[str] = None,
    ) -> None:
        self._preconditions = preconditions
        if completion_checkpoint_path is None:
            self.clear_completion_checkpoint_save_request()
        else:
            self.request_completion_checkpoint_save(completion_checkpoint_path)

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
                early_stop_patience=2,
                stop_event=self._stop_event,
            )
            self._maybe_save_completion_checkpoint(result=result)
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


@dataclass(frozen=True)
class LearningInferencePrediction:
    box: BoundingBox
    predicted_bbox: np.ndarray


@dataclass(frozen=True)
class LearningInferenceBackgroundResult:
    total_count: int
    predictions: Tuple[LearningInferencePrediction, ...]
    failure_by_box_id: Dict[str, str]
    cleanup_errors_by_box_id: Dict[str, Tuple[str, ...]]


class LearningInferenceStopRequested(RuntimeError):
    """Raised internally to stop inference at the next batch boundary."""


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
        try:
            import torch
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                f"PyTorch is required to run Segment Inference BBox: {exc}"
            ) from exc

        model_runtime = self._configured_model_runtime()
        inference_boxes = tuple(self._inference_boxes)
        raw_array = self._configured_raw_array()
        label_values = tuple(self._label_values)
        volume_shape = tuple(self._volume_shape)
        total_count = int(len(inference_boxes))

        model = getattr(model_runtime, "model", None)
        if model is None:
            raise RuntimeError("Model runtime does not expose a model for inference.")

        was_training = bool(getattr(model, "training", False))
        eval_method = getattr(model, "eval", None)
        if callable(eval_method):
            eval_method()

        failure_by_box_id: Dict[str, str] = {}
        cleanup_errors_by_box_id: Dict[str, Tuple[str, ...]] = {}
        predictions: list[LearningInferencePrediction] = []

        try:
            device_ids_obj = getattr(model_runtime, "device_ids", ())
            resolved_device = None
            if (
                isinstance(device_ids_obj, (list, tuple))
                and device_ids_obj
                and bool(torch.cuda.is_available())
            ):
                try:
                    preferred_id = int(device_ids_obj[0])
                except Exception:
                    preferred_id = 0
                if 0 <= preferred_id < int(torch.cuda.device_count()):
                    resolved_device = torch.device(f"cuda:{preferred_id}")
            if resolved_device is None:
                first_param = next(model.parameters(), None)
                if first_param is not None:
                    resolved_device = first_param.device
                else:
                    resolved_device = torch.device(
                        "cuda:0" if bool(torch.cuda.is_available()) else "cpu"
                    )
            resolved_device_type = str(getattr(resolved_device, "type", "cpu"))
            autocast_enabled = bool(resolved_device_type == "cuda")

            with torch.no_grad():
                for order_index, box in enumerate(inference_boxes, start=1):
                    self._raise_if_stop_requested()
                    runtime = None
                    try:
                        z_bounds = (int(box.z0), int(box.z1))
                        y_bounds = (int(box.y0), int(box.y1))
                        x_bounds = (int(box.x0), int(box.x1))
                        raw_context = _extract_bbox_context_from_array(
                            raw_array,
                            z_bounds=z_bounds,
                            y_bounds=y_bounds,
                            x_bounds=x_bounds,
                        )
                        raw_tensor = torch_from_numpy_safe(
                            raw_context,
                            torch_module=torch,
                        ).to(dtype=torch.float16)
                        placeholder_segmentation = torch.zeros(
                            (1, 1, 1),
                            dtype=torch.int16,
                        )
                        entry = LearningBBoxTensorEntry(
                            box_id=str(box.id),
                            index=int(order_index),
                            label="inference",
                            raw_tensor=raw_tensor,
                            segmentation_tensor=placeholder_segmentation,
                        )
                        runtime = _build_inference_dataloader_runtime_from_entry(
                            entry,
                            label_values=label_values,
                            minivol_size=200,
                            batch_size=4,
                            num_workers=8,
                            pin_memory=True,
                            drop_last=False,
                        )

                        add_batch = getattr(runtime.buffer, "add_batch", None)
                        if not callable(add_batch):
                            raise TypeError(
                                f"Inference buffer for box_id={box.id!r} must define add_batch(batch, coordinates)."
                            )
                        get_pred_labels = getattr(
                            runtime.buffer, "get_pred_labels", None
                        )
                        if not callable(get_pred_labels):
                            raise TypeError(
                                f"Inference buffer for box_id={box.id!r} must define get_pred_labels()."
                            )

                        for minivols, coordinates in runtime.dataloader:
                            self._raise_if_stop_requested()
                            minivols = minivols.to(resolved_device)
                            with torch.autocast(
                                device_type=resolved_device_type,
                                enabled=autocast_enabled,
                                dtype=getattr(torch, "bfloat16"),
                            ):
                                pred_minivols = model(minivols)
                            add_batch(pred_minivols.detach().cpu(), coordinates)

                        self._raise_if_stop_requested()
                        predicted_context = get_pred_labels()
                        if isinstance(predicted_context, torch.Tensor):
                            predicted_context_array = np.asarray(
                                predicted_context.detach().cpu()
                            )
                        else:
                            predicted_context_array = np.asarray(predicted_context)

                        context_plan = _plan_bbox_context(
                            z_bounds=z_bounds,
                            y_bounds=y_bounds,
                            x_bounds=x_bounds,
                            volume_shape=volume_shape,
                        )
                        z_start = int(context_plan.z.extend_before)
                        y_start = int(context_plan.y.extend_before)
                        x_start = int(context_plan.x.extend_before)
                        z_size = int(context_plan.z.original_size)
                        y_size = int(context_plan.y.original_size)
                        x_size = int(context_plan.x.original_size)
                        predicted_bbox = np.asarray(
                            predicted_context_array[
                                z_start : z_start + z_size,
                                y_start : y_start + y_size,
                                x_start : x_start + x_size,
                            ]
                        ).copy()
                        predictions.append(
                            LearningInferencePrediction(
                                box=box,
                                predicted_bbox=predicted_bbox,
                            )
                        )
                    except LearningInferenceStopRequested:
                        raise
                    except Exception as exc:
                        failure_by_box_id[str(box.id)] = exception_message(exc)
                    finally:
                        if runtime is not None:
                            dispose_errors = _dispose_inference_runtime(runtime)
                            if dispose_errors:
                                cleanup_errors_by_box_id[str(box.id)] = tuple(
                                    dispose_errors
                                )
        finally:
            if was_training:
                train_method = getattr(model, "train", None)
                if callable(train_method):
                    train_method()

        return LearningInferenceBackgroundResult(
            total_count=total_count,
            predictions=tuple(predictions),
            failure_by_box_id=dict(failure_by_box_id),
            cleanup_errors_by_box_id=dict(cleanup_errors_by_box_id),
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
