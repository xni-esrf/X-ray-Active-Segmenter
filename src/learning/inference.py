from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from ..bbox import BoundingBox
from ..utils import exception_message, torch_from_numpy_safe
from .session_store import LearningBBoxTensorEntry


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


@dataclass(frozen=True)
class LearningInferenceMemoryEstimate:
    box_id: str
    context_shape: Tuple[int, int, int]
    num_classes: int
    score_buffer_bytes: int
    rough_peak_bytes: int


@dataclass(frozen=True)
class LearningInferenceProgress:
    completed_count: int
    total_count: int
    box_id: str
    succeeded: bool
    failed_count: int

    @property
    def percent_complete(self) -> float:
        if self.total_count <= 0:
            return 100.0
        return 100.0 * float(self.completed_count) / float(self.total_count)


class LearningInferenceStopRequested(RuntimeError):
    """Raised internally to stop inference at the next batch boundary."""


def estimate_inference_bbox_memory(
    *,
    box: BoundingBox,
    label_values: Sequence[int],
    volume_shape: Sequence[int],
) -> LearningInferenceMemoryEstimate:
    if not isinstance(box, BoundingBox):
        raise TypeError(f"box must be a BoundingBox, got {type(box).__name__}")
    normalized_labels = _normalize_label_values(label_values)
    normalized_shape = _normalize_volume_shape(volume_shape)
    context_plan = _plan_bbox_context(
        z_bounds=(int(box.z0), int(box.z1)),
        y_bounds=(int(box.y0), int(box.y1)),
        x_bounds=(int(box.x0), int(box.x1)),
        volume_shape=normalized_shape,
    )
    context_shape = (
        int(context_plan.z.target_size),
        int(context_plan.y.target_size),
        int(context_plan.x.target_size),
    )
    voxel_count = int(context_shape[0]) * int(context_shape[1]) * int(context_shape[2])
    num_classes = int(len(normalized_labels))
    score_buffer_bytes = int(num_classes * voxel_count * 4)
    # The dense score buffer is the largest single allocation, but inference also
    # holds the raw crop, normalized tensor, decoded labels, and apply-time masks.
    rough_peak_bytes = int(score_buffer_bytes * 2)
    return LearningInferenceMemoryEstimate(
        box_id=str(box.id),
        context_shape=context_shape,
        num_classes=num_classes,
        score_buffer_bytes=score_buffer_bytes,
        rough_peak_bytes=rough_peak_bytes,
    )


def run_learning_inference(
    *,
    model_runtime: object,
    inference_boxes: Sequence[BoundingBox],
    raw_array: np.ndarray,
    label_values: Sequence[int],
    volume_shape: Sequence[int],
    stop_requested: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[LearningInferenceProgress], None]] = None,
    extract_bbox_context_from_array_func: Optional[Callable[..., np.ndarray]] = None,
    plan_bbox_context_func: Optional[Callable[..., object]] = None,
    build_inference_dataloader_runtime_from_entry_func: Optional[Callable[..., object]] = None,
    dispose_inference_runtime_func: Optional[Callable[[object], Sequence[str]]] = None,
) -> LearningInferenceBackgroundResult:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            f"PyTorch is required to run Segment Inference BBox: {exc}"
        ) from exc

    normalized_boxes = _normalize_inference_boxes(inference_boxes)
    normalized_labels = _normalize_label_values(label_values)
    normalized_shape = _normalize_volume_shape(volume_shape)
    normalized_raw_array = np.asarray(raw_array)
    total_count = int(len(normalized_boxes))
    extract_bbox_context = (
        _extract_bbox_context_from_array
        if extract_bbox_context_from_array_func is None
        else extract_bbox_context_from_array_func
    )
    plan_context = _plan_bbox_context if plan_bbox_context_func is None else plan_bbox_context_func
    build_runtime = (
        _build_inference_dataloader_runtime_from_entry
        if build_inference_dataloader_runtime_from_entry_func is None
        else build_inference_dataloader_runtime_from_entry_func
    )
    dispose_runtime = (
        _dispose_inference_runtime
        if dispose_inference_runtime_func is None
        else dispose_inference_runtime_func
    )

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
        resolved_device = _resolve_inference_device(torch, model_runtime, model)
        resolved_device_type = str(getattr(resolved_device, "type", "cpu"))
        autocast_enabled = bool(resolved_device_type == "cuda")

        with torch.no_grad():
            for order_index, box in enumerate(normalized_boxes, start=1):
                _raise_if_stop_requested(stop_requested)
                runtime = None
                succeeded = False
                try:
                    z_bounds = (int(box.z0), int(box.z1))
                    y_bounds = (int(box.y0), int(box.y1))
                    x_bounds = (int(box.x0), int(box.x1))
                    raw_context = extract_bbox_context(
                        normalized_raw_array,
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
                    runtime = build_runtime(
                        entry,
                        label_values=normalized_labels,
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
                    get_pred_labels = getattr(runtime.buffer, "get_pred_labels", None)
                    if not callable(get_pred_labels):
                        raise TypeError(
                            f"Inference buffer for box_id={box.id!r} must define get_pred_labels()."
                        )

                    for minivols, coordinates in runtime.dataloader:
                        _raise_if_stop_requested(stop_requested)
                        minivols = minivols.to(resolved_device)
                        with torch.autocast(
                            device_type=resolved_device_type,
                            enabled=autocast_enabled,
                            dtype=getattr(torch, "bfloat16"),
                        ):
                            pred_minivols = model(minivols)
                        add_batch(pred_minivols.detach().cpu(), coordinates)

                    _raise_if_stop_requested(stop_requested)
                    predicted_context = get_pred_labels()
                    if isinstance(predicted_context, torch.Tensor):
                        predicted_context_array = np.asarray(
                            predicted_context.detach().cpu()
                        )
                    else:
                        predicted_context_array = np.asarray(predicted_context)

                    context_plan = plan_context(
                        z_bounds=z_bounds,
                        y_bounds=y_bounds,
                        x_bounds=x_bounds,
                        volume_shape=normalized_shape,
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
                    succeeded = True
                except LearningInferenceStopRequested:
                    raise
                except Exception as exc:
                    failure_by_box_id[str(box.id)] = exception_message(exc)
                finally:
                    if runtime is not None:
                        dispose_errors = tuple(dispose_runtime(runtime))
                        if dispose_errors:
                            cleanup_errors_by_box_id[str(box.id)] = tuple(dispose_errors)
                    if progress_callback is not None:
                        progress_callback(
                            LearningInferenceProgress(
                                completed_count=int(order_index),
                                total_count=int(total_count),
                                box_id=str(box.id),
                                succeeded=bool(succeeded),
                                failed_count=int(len(failure_by_box_id)),
                            )
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


def apply_inference_predictions_to_array(
    segmentation_array: np.ndarray,
    predictions: Sequence[LearningInferencePrediction],
) -> Tuple[np.ndarray, int, Tuple[str, ...], Dict[str, str]]:
    output = np.array(segmentation_array, copy=True)
    succeeded_box_ids: list[str] = []
    failure_by_box_id: Dict[str, str] = {}
    changed_voxel_count_total = 0
    for prediction in tuple(predictions):
        box = prediction.box
        try:
            changed_count = _apply_predicted_bbox_to_array(
                output,
                box=box,
                predicted_bbox=prediction.predicted_bbox,
            )
            changed_voxel_count_total += int(changed_count)
            succeeded_box_ids.append(str(box.id))
        except Exception as exc:
            failure_by_box_id[str(box.id)] = exception_message(exc)
    return (
        output,
        int(changed_voxel_count_total),
        tuple(succeeded_box_ids),
        dict(failure_by_box_id),
    )


def _apply_predicted_bbox_to_array(
    segmentation_array: np.ndarray,
    *,
    box: BoundingBox,
    predicted_bbox: np.ndarray,
) -> int:
    target = np.asarray(segmentation_array)
    if target.ndim != 3:
        raise ValueError(f"segmentation_array must be 3D, got ndim={target.ndim}")
    z0 = int(box.z0)
    z1 = int(box.z1)
    y0 = int(box.y0)
    y1 = int(box.y1)
    x0 = int(box.x0)
    x1 = int(box.x1)
    expected_shape = (z1 - z0, y1 - y0, x1 - x0)
    predicted = np.asarray(predicted_bbox)
    if predicted.ndim != 3 or tuple(int(v) for v in predicted.shape) != expected_shape:
        raise ValueError(
            "Predicted bbox shape does not match bbox size: "
            f"pred={tuple(predicted.shape)} expected={expected_shape} box_id={box.id!r}"
        )
    current_bbox = np.asarray(target[z0:z1, y0:y1, x0:x1])
    changed_mask = predicted != current_bbox
    if not np.any(changed_mask):
        return 0

    predicted_changed = np.asarray(predicted[changed_mask], dtype=np.int64)
    if predicted_changed.size == 0:
        return 0
    if not np.issubdtype(target.dtype, np.integer):
        raise ValueError(f"segmentation_array dtype must be integer, got {target.dtype}")
    dtype_info = np.iinfo(target.dtype)
    min_label = int(np.min(predicted_changed))
    max_label = int(np.max(predicted_changed))
    if min_label < int(dtype_info.min) or max_label > int(dtype_info.max):
        raise ValueError(
            "Predicted labels cannot be represented in the segmentation dtype "
            f"{target.dtype}: range=[{min_label}, {max_label}] "
            f"allowed=[{int(dtype_info.min)}, {int(dtype_info.max)}]."
        )
    target[z0:z1, y0:y1, x0:x1] = predicted.astype(target.dtype, copy=False)
    return int(np.count_nonzero(changed_mask))


def _normalize_inference_boxes(values: Sequence[BoundingBox]) -> Tuple[BoundingBox, ...]:
    normalized: list[BoundingBox] = []
    for raw_box in tuple(values):
        if not isinstance(raw_box, BoundingBox):
            raise TypeError(
                "inference_boxes must contain BoundingBox instances only, "
                f"got {type(raw_box).__name__}"
            )
        normalized.append(raw_box)
    if not normalized:
        raise ValueError("inference_boxes must contain at least one bounding box")
    return tuple(normalized)


def _normalize_label_values(values: Sequence[int]) -> Tuple[int, ...]:
    normalized = tuple(int(value) for value in tuple(values))
    if not normalized:
        raise ValueError("label_values must contain at least one class label")
    return normalized


def _normalize_volume_shape(values: Sequence[int]) -> Tuple[int, int, int]:
    shape = tuple(int(v) for v in tuple(values))
    if len(shape) != 3:
        raise ValueError(f"volume_shape must be length 3, got {shape}")
    if any(axis <= 0 for axis in shape):
        raise ValueError(f"volume_shape axes must be positive, got {shape}")
    return (int(shape[0]), int(shape[1]), int(shape[2]))


def _raise_if_stop_requested(stop_requested: Optional[Callable[[], bool]]) -> None:
    if stop_requested is not None and bool(stop_requested()):
        raise LearningInferenceStopRequested("Inference stop requested by user.")


def _resolve_inference_device(torch_mod, model_runtime: object, model: object):
    device_ids_obj = getattr(model_runtime, "device_ids", ())
    if (
        isinstance(device_ids_obj, (list, tuple))
        and device_ids_obj
        and bool(torch_mod.cuda.is_available())
    ):
        try:
            preferred_id = int(device_ids_obj[0])
        except Exception:
            preferred_id = 0
        if 0 <= preferred_id < int(torch_mod.cuda.device_count()):
            return torch_mod.device(f"cuda:{preferred_id}")
    try:
        first_param = next(model.parameters(), None)
    except TypeError:
        first_param = None
    if first_param is not None:
        return first_param.device
    return torch_mod.device("cuda:0" if bool(torch_mod.cuda.is_available()) else "cpu")


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


__all__ = [
    "LearningInferenceBackgroundResult",
    "LearningInferenceMemoryEstimate",
    "LearningInferencePrediction",
    "LearningInferenceProgress",
    "LearningInferenceStopRequested",
    "apply_inference_predictions_to_array",
    "estimate_inference_bbox_memory",
    "run_learning_inference",
]
