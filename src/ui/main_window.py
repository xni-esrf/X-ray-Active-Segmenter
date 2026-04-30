from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from threading import Event, Lock
from typing import Dict, Literal, Mapping, Optional, Sequence, Set, Tuple, cast

import numpy as np

from PySide6.QtCore import QObject, QEvent, QThread, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QCloseEvent, QKeySequence, QResizeEvent, QShortcut
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QGridLayout,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QScrollArea,
    QSplitter,
    QTextEdit,
    QWidget,
)

from ..annotation import BrushRadius, EditOperation, SegmentationEditor, SegmentationKind
from ..bbox import (
    BoundingBox,
    BoundingBoxChange,
    BoundingBoxLabel,
    BoundingBoxManager,
    FaceId,
    load_bounding_boxes,
    save_bounding_boxes,
)
from ..data import VolumeData, build_segmentation_pyramid_lazy
from ..events import InputHandlers, SyncManager
from ..history import (
    BoundingBoxAddCommand,
    BoundingBoxDeleteCommand,
    BoundingBoxUpdateCommand,
    GlobalHistoryManager,
    HistoryCommand,
    SegmentationHistoryCommand,
    estimate_bounding_box_history_bytes,
    estimate_segmentation_history_bytes,
)
from ..io import extract_learning_bboxes_in_memory
from ..io.bbox_export_utils import extract_bbox_context_from_array, plan_bbox_context
from ..io.saver import save_segmentation_volume
from ..loading import load_prepared_volume
from ..learning import (
    DEFAULT_FOUNDATION_CHECKPOINT_PATH,
    LearningTrainingLoopResult,
    LearningBBoxTensorEntry,
    build_inference_dataloader_runtime_from_entry,
    clear_current_learning_bbox_batch,
    compute_and_store_current_learning_class_weights,
    dispose_inference_runtime,
    get_current_learning_eval_runtimes_by_box_id,
    get_current_learning_model_runtime,
    get_current_learning_bbox_batch,
    instantiate_foundation_model_runtime,
    save_foundation_model_checkpoint,
    train_learning_model_with_validation_loop,
    validate_foundation_model_instantiation_preconditions,
    validate_learning_model_training_preconditions,
)
from ..render import Renderer, ViewId
from ..utils import get_logger, torch_from_numpy_safe
from .bottom_panel import BottomPanel
from .dialogs import (
    TrainingCloseDecision,
    UnsavedChangesDecision,
    ask_unsaved_changes,
    ask_training_running_close_decision,
    confirm_reinitialize_model,
    confirm_replace_bounding_boxes,
    confirm_replace_inference_bboxes,
    confirm_replace_training_model_with_default_checkpoint,
    confirm_overwrite,
    open_file_dialog,
    open_model_checkpoint_dialog,
    open_save_model_checkpoint_dialog,
    open_bounding_boxes_dialog,
    open_save_bounding_boxes_dialog,
    open_save_segmentation_dialog,
    show_info,
    show_warning,
)
from .orthogonal_view import AnnotationPaintOutcome, OrthogonalView


AnnotationTool = Literal["brush", "eraser", "flood_filler"]
BBoxSegmentationOperation = Literal["median_filter", "erosion", "dilation"]
DeferredTrainingCloseMode = Literal[
    "none",
    "stop_and_close",
    "continue_in_background",
]

# Keep annotation tool key handling explicitly scoped so existing Ctrl+Z/Y/S
# shortcuts continue to flow through their dedicated handlers.
_ANNOTATION_TOOL_SHORTCUT_BY_KEY: Mapping[int, AnnotationTool] = {
    int(Qt.Key_B): "brush",
    int(Qt.Key_E): "eraser",
    int(Qt.Key_F): "flood_filler",
}
_DEFAULT_TRAINING_FOUNDATION_CHECKPOINT_PATH = DEFAULT_FOUNDATION_CHECKPOINT_PATH
_LOGGER = get_logger(__name__)


def _format_class_weights_for_summary(class_weights: object) -> Optional[str]:
    if class_weights is None:
        return None

    values_obj = class_weights
    detach = getattr(values_obj, "detach", None)
    if callable(detach):
        try:
            values_obj = detach()
        except Exception:
            return None
    cpu = getattr(values_obj, "cpu", None)
    if callable(cpu):
        try:
            values_obj = cpu()
        except Exception:
            return None
    tolist = getattr(values_obj, "tolist", None)
    if callable(tolist):
        try:
            values_obj = tolist()
        except Exception:
            return None

    if isinstance(values_obj, np.ndarray):
        values_obj = values_obj.tolist()
    if not isinstance(values_obj, (list, tuple)):
        return None

    normalized_values = []
    for raw_value in values_obj:
        if isinstance(raw_value, bool):
            return None
        try:
            normalized_values.append(float(raw_value))
        except (TypeError, ValueError):
            return None
    return "[" + ", ".join(f"{value:.6g}" for value in normalized_values) + "]"


def _coerce_eval_label_values(values: object) -> Tuple[int, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(
            "Evaluation buffer label_values must be a sequence of class ids, "
            f"got {type(values).__name__}"
        )
    normalized = []
    for raw_value in tuple(values):
        if isinstance(raw_value, bool) or not isinstance(raw_value, Integral):
            raise TypeError(
                "Evaluation buffer label_values must contain integers only, "
                f"got {type(raw_value).__name__}"
            )
        value = int(raw_value)
        if value == -100:
            raise ValueError("Evaluation buffer label_values must not include -100.")
        if value in normalized:
            raise ValueError(f"Evaluation buffer label_values must not contain duplicates: {value}")
        normalized.append(value)
    if not normalized:
        raise ValueError("Evaluation buffer label_values must contain at least one class id.")
    return tuple(normalized)


def _resolve_shared_eval_label_values(eval_runtimes_by_box_id: Mapping[str, object]) -> Tuple[int, ...]:
    if not isinstance(eval_runtimes_by_box_id, Mapping):
        raise TypeError(
            "eval_runtimes_by_box_id must be a mapping of box_id -> runtime, "
            f"got {type(eval_runtimes_by_box_id).__name__}"
        )
    if not eval_runtimes_by_box_id:
        raise ValueError("No evaluation runtimes/buffers are available in session storage.")

    resolved_label_values: Optional[Tuple[int, ...]] = None
    for box_id, runtime in tuple(eval_runtimes_by_box_id.items()):
        buffer_obj = getattr(runtime, "buffer", None)
        if buffer_obj is None:
            raise ValueError(
                f"Evaluation runtime for box_id={box_id!r} does not expose a buffer."
            )
        if not hasattr(buffer_obj, "label_values"):
            raise ValueError(
                f"Evaluation buffer for box_id={box_id!r} does not expose label_values."
            )
        label_values = _coerce_eval_label_values(getattr(buffer_obj, "label_values"))
        if resolved_label_values is None:
            resolved_label_values = label_values
            continue
        if label_values != resolved_label_values:
            raise ValueError(
                "All evaluation buffers must share the same label_values ordering; "
                f"expected {resolved_label_values}, got {label_values} for box_id={box_id!r}."
            )

    if resolved_label_values is None:
        raise ValueError("No evaluation buffer label_values could be resolved.")
    return resolved_label_values


def _boxes_overlap(first: BoundingBox, second: BoundingBox) -> bool:
    return (
        int(first.z0) < int(second.z1)
        and int(second.z0) < int(first.z1)
        and int(first.y0) < int(second.y1)
        and int(second.y0) < int(first.y1)
        and int(first.x0) < int(second.x1)
        and int(second.x0) < int(first.x1)
    )


def _find_overlapping_box_id_pairs(boxes: Sequence[BoundingBox]) -> Tuple[Tuple[str, str], ...]:
    normalized_boxes = tuple(boxes)
    overlaps = []
    for first_index in range(len(normalized_boxes)):
        first = normalized_boxes[first_index]
        for second_index in range(first_index + 1, len(normalized_boxes)):
            second = normalized_boxes[second_index]
            if _boxes_overlap(first, second):
                overlaps.append((str(first.id), str(second.id)))
    return tuple(overlaps)


def _ordered_inference_boxes(
    *,
    ordered_box_ids: Sequence[str],
    boxes_by_id: Mapping[str, BoundingBox],
) -> Tuple[BoundingBox, ...]:
    selected = []
    seen_ids = set()
    for raw_box_id in tuple(ordered_box_ids):
        box_id = str(raw_box_id).strip()
        if not box_id:
            continue
        box = boxes_by_id.get(box_id)
        if box is None or str(box.label) != "inference":
            continue
        if box.id in seen_ids:
            continue
        selected.append(box)
        seen_ids.add(box.id)
    if selected:
        return tuple(selected)
    return tuple(box for box in tuple(boxes_by_id.values()) if str(box.label) == "inference")


def _exception_message(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return message
    return f"{type(exc).__name__}"


def _normalize_checkpoint_identity(path: object) -> Optional[str]:
    if not isinstance(path, str):
        return None
    normalized = path.strip()
    if not normalized:
        return None
    expanded = Path(normalized).expanduser()
    try:
        return str(expanded.resolve())
    except Exception:
        return str(expanded)


def _apply_predicted_bbox_to_editor(
    *,
    editor: SegmentationEditor,
    box: BoundingBox,
    predicted_bbox: np.ndarray,
) -> int:
    z0 = int(box.z0)
    z1 = int(box.z1)
    y0 = int(box.y0)
    y1 = int(box.y1)
    x0 = int(box.x0)
    x1 = int(box.x1)
    expected_shape = (z1 - z0, y1 - y0, x1 - x0)

    predicted = np.asarray(predicted_bbox)
    if tuple(int(v) for v in predicted.shape) != expected_shape:
        raise ValueError(
            "Predicted bbox shape does not match bbox size: "
            f"pred={tuple(predicted.shape)} expected={expected_shape} box_id={box.id!r}"
        )
    if predicted.ndim != 3:
        raise ValueError(
            f"Predicted bbox must be a 3D array, got ndim={predicted.ndim} for box_id={box.id!r}"
        )

    current_bbox = np.asarray(editor.array_view()[z0:z1, y0:y1, x0:x1])
    changed_mask = predicted != current_bbox
    if not np.any(changed_mask):
        return 0

    predicted_changed = np.asarray(predicted[changed_mask], dtype=np.int64)
    if predicted_changed.size == 0:
        return 0
    min_label = int(np.min(predicted_changed))
    max_label = int(np.max(predicted_changed))
    dtype_info = np.iinfo(editor.dtype)
    if min_label < 0 or max_label > int(dtype_info.max):
        raise ValueError(
            "Predicted labels cannot be represented in the active semantic dtype "
            f"{editor.dtype}: range=[{min_label}, {max_label}] allowed=[0, {int(dtype_info.max)}]."
        )

    changed_coords = np.argwhere(changed_mask)
    origin = np.asarray([[z0, y0, x0]], dtype=np.int64)
    changed_count = int(changed_coords.shape[0])

    for target_label in np.unique(predicted_changed):
        label_mask = predicted_changed == int(target_label)
        if not np.any(label_mask):
            continue
        label_coords = changed_coords[label_mask] + origin
        editor.assign(
            label_coords,
            label=int(target_label),
            operation_name="segment_inference_bboxes",
            ignore_out_of_bounds=False,
        )
    return changed_count


@dataclass
class MainWindowState:
    volume_loaded: bool = False
    annotation_mode_enabled: bool = False
    bbox_mode_enabled: bool = False
    annotation_tool: AnnotationTool = "brush"
    brush_radius: BrushRadius = 0
    eraser_target_label: Optional[int] = None
    picked_indices: Optional[Tuple[int, int, int]] = None
    picked_label: Optional[int] = None
    pending_bbox_corner: Optional[Tuple[int, int, int]] = None
    flood_fill_target_label: int = 1


@dataclass(frozen=True)
class _StagedBoundingBoxDragUpdate:
    before_box: BoundingBox
    after_box: BoundingBox
    before_selected_id: Optional[str]
    after_selected_id: Optional[str]


class _LearningTrainingWorker(QObject):
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

    def _maybe_save_completion_checkpoint(self, *, result: object) -> None:
        checkpoint_path = self._completion_checkpoint_save_path()
        if checkpoint_path is None:
            return
        if not isinstance(result, LearningTrainingLoopResult):
            return

        normalized_reason = str(result.stop_reason).strip().lower()
        if normalized_reason not in {"early_stop", "max_epoch"}:
            return

        runtime = get_current_learning_model_runtime()
        if runtime is None:
            raise RuntimeError(
                "No learning model runtime is available to save the completion checkpoint."
            )
        try:
            save_foundation_model_checkpoint(
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
            result = train_learning_model_with_validation_loop(
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
class _LearningInferencePrediction:
    box: BoundingBox
    predicted_bbox: np.ndarray


@dataclass(frozen=True)
class _LearningInferenceBackgroundResult:
    total_count: int
    predictions: Tuple[_LearningInferencePrediction, ...]
    failure_by_box_id: Dict[str, str]
    cleanup_errors_by_box_id: Dict[str, Tuple[str, ...]]


class _LearningInferenceStopRequested(RuntimeError):
    """Raised internally to stop inference at the next batch boundary."""


class _LearningInferenceWorker(QObject):
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
            raise _LearningInferenceStopRequested("Inference stop requested by user.")

    def _configured_model_runtime(self) -> object:
        if self._model_runtime is None:
            raise RuntimeError("Inference worker is not configured with a model runtime.")
        return self._model_runtime

    def _configured_raw_array(self) -> np.ndarray:
        if self._raw_array is None:
            raise RuntimeError("Inference worker is not configured with raw input array.")
        return self._raw_array

    def _run_inference(self) -> _LearningInferenceBackgroundResult:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                f"PyTorch is required to run Segment Inference Bbox: {exc}"
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
        predictions: list[_LearningInferencePrediction] = []

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
                        raw_context = extract_bbox_context_from_array(
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
                        runtime = build_inference_dataloader_runtime_from_entry(
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
                        get_pred_labels = getattr(runtime.buffer, "get_pred_labels", None)
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

                        context_plan = plan_bbox_context(
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
                            _LearningInferencePrediction(
                                box=box,
                                predicted_bbox=predicted_bbox,
                            )
                        )
                    except _LearningInferenceStopRequested:
                        raise
                    except Exception as exc:
                        failure_by_box_id[str(box.id)] = _exception_message(exc)
                    finally:
                        if runtime is not None:
                            dispose_errors = dispose_inference_runtime(runtime)
                            if dispose_errors:
                                cleanup_errors_by_box_id[str(box.id)] = tuple(dispose_errors)
        finally:
            if was_training:
                train_method = getattr(model, "train", None)
                if callable(train_method):
                    train_method()

        return _LearningInferenceBackgroundResult(
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
        except _LearningInferenceStopRequested as exc:
            self.canceled.emit(str(exc))
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class MainWindow(QMainWindow):
    _CONTROL_PANEL_MIN_WIDTH_FRACTION = 0.08
    _CONTROL_PANEL_MAX_WIDTH_FRACTION = 0.30
    _CONTROL_PANEL_INITIAL_WIDTH_FRACTION = 0.14

    def __init__(
        self,
        renderer: Renderer,
        sync_manager: SyncManager,
        input_handlers: InputHandlers,
        *,
        load_mode: str = "ram",
        cache_max_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        super().__init__()
        self.renderer = renderer
        self.sync_manager = sync_manager
        self.input_handlers = input_handlers
        self._load_mode = str(load_mode).strip().lower()
        if self._load_mode not in {"ram", "lazy"}:
            raise ValueError("load_mode must be 'ram' or 'lazy'")
        self._cache_max_bytes = int(cache_max_bytes)
        self._io_worker: Optional[object] = None
        self._semantic_worker: Optional[object] = None
        self._semantic_volume: Optional[VolumeData] = None
        self._instance_worker: Optional[object] = None
        self._instance_volume: Optional[VolumeData] = None
        self._training_running = False
        self._training_worker: Optional[object] = None
        self._training_thread: Optional[object] = None
        self._inference_running = False
        self._inference_stop_requested = False
        self._inference_worker: Optional[object] = None
        self._inference_thread: Optional[object] = None
        self._deferred_close_after_training = False
        self._deferred_close_training_mode: DeferredTrainingCloseMode = "none"
        self._deferred_close_checkpoint_path: Optional[str] = None
        self._segmentation_editor: Optional[SegmentationEditor] = None
        self._annotation_kind: SegmentationKind = "semantic"
        self._raw_volume: Optional[VolumeData] = None
        self._pending_render_view_ids: Set[ViewId] = set()
        self._render_flush_scheduled = False
        self._pending_annotation_peer_view_ids: Set[ViewId] = set()
        self._annotation_dirty_views: Set[ViewId] = set()
        self._annotation_peer_flush_scheduled = False
        self._bbox_drag_active = False
        self._bbox_drag_source_view_id: Optional[ViewId] = None
        self._bbox_pending_peer_view_ids: Set[ViewId] = set()
        self._bbox_peer_flush_scheduled = False
        self._bbox_drag_staged_history_updates: Dict[str, _StagedBoundingBoxDragUpdate] = {}
        self._annotation_modification_active = False
        self._annotation_modification_view_id: Optional[ViewId] = None
        self._annotation_labels_dirty = False
        self._deferred_hover_readout = False
        self._deferred_picked_readout = False
        # Keep the active painting view responsive by updating peer views less often.
        self._annotation_peer_redraw_interval_ms = 50
        # Keep drag source overlays responsive while throttling peer views.
        self._bbox_peer_redraw_interval_ms = 33
        # Cancel very large flood fills that would otherwise stall the UI.
        self._flood_fill_timeout_seconds = 30.0
        self._global_history = GlobalHistoryManager()
        self.bottom_panel = BottomPanel()
        self.state = MainWindowState()
        self._bbox_manager = BoundingBoxManager((1, 1, 1))
        self._bbox_manager.on_changed(self._on_bounding_boxes_changed)
        self.views: Dict[ViewId, OrthogonalView] = {
            "axial": OrthogonalView(
                "axial",
                axis=0,
                renderer=renderer,
                input_handlers=input_handlers,
                annotation_tool_getter=self._current_annotation_tool,
                bounding_box_mode_enabled_getter=self._bounding_box_mode_enabled,
                on_paint_voxel=self._handle_paint_voxel,
                on_paint_stroke=self._handle_paint_stroke,
                on_pick_voxel=self._handle_pick_voxel,
                on_annotation_finished=self._handle_annotation_finished,
                bounding_boxes_getter=self._overlay_bounding_boxes,
                selected_bounding_box_id_getter=self._overlay_selected_bounding_box_id,
                on_bounding_box_select=self._handle_bounding_box_selected,
                on_bounding_box_move_face=self._handle_bounding_box_face_moved,
                on_bounding_box_translate=self._handle_bounding_box_translated,
                on_bounding_box_drag_started=self._handle_bounding_box_drag_started,
                on_bounding_box_drag_finished=self._handle_bounding_box_drag_finished,
                on_bounding_box_delete_requested=self._handle_bounding_box_delete_shortcut_requested,
            ),
            "coronal": OrthogonalView(
                "coronal",
                axis=1,
                renderer=renderer,
                input_handlers=input_handlers,
                annotation_tool_getter=self._current_annotation_tool,
                bounding_box_mode_enabled_getter=self._bounding_box_mode_enabled,
                on_paint_voxel=self._handle_paint_voxel,
                on_paint_stroke=self._handle_paint_stroke,
                on_pick_voxel=self._handle_pick_voxel,
                on_annotation_finished=self._handle_annotation_finished,
                bounding_boxes_getter=self._overlay_bounding_boxes,
                selected_bounding_box_id_getter=self._overlay_selected_bounding_box_id,
                on_bounding_box_select=self._handle_bounding_box_selected,
                on_bounding_box_move_face=self._handle_bounding_box_face_moved,
                on_bounding_box_translate=self._handle_bounding_box_translated,
                on_bounding_box_drag_started=self._handle_bounding_box_drag_started,
                on_bounding_box_drag_finished=self._handle_bounding_box_drag_finished,
                on_bounding_box_delete_requested=self._handle_bounding_box_delete_shortcut_requested,
            ),
            "sagittal": OrthogonalView(
                "sagittal",
                axis=2,
                renderer=renderer,
                input_handlers=input_handlers,
                annotation_tool_getter=self._current_annotation_tool,
                bounding_box_mode_enabled_getter=self._bounding_box_mode_enabled,
                on_paint_voxel=self._handle_paint_voxel,
                on_paint_stroke=self._handle_paint_stroke,
                on_pick_voxel=self._handle_pick_voxel,
                on_annotation_finished=self._handle_annotation_finished,
                bounding_boxes_getter=self._overlay_bounding_boxes,
                selected_bounding_box_id_getter=self._overlay_selected_bounding_box_id,
                on_bounding_box_select=self._handle_bounding_box_selected,
                on_bounding_box_move_face=self._handle_bounding_box_face_moved,
                on_bounding_box_translate=self._handle_bounding_box_translated,
                on_bounding_box_drag_started=self._handle_bounding_box_drag_started,
                on_bounding_box_drag_finished=self._handle_bounding_box_drag_finished,
                on_bounding_box_delete_requested=self._handle_bounding_box_delete_shortcut_requested,
            ),
        }

        left_panel = QWidget()
        left_layout = QGridLayout()
        left_layout.addWidget(self.views["axial"], 0, 0)
        left_layout.addWidget(self.views["coronal"], 0, 1)
        left_layout.addWidget(self.views["sagittal"], 1, 0, 1, 2)
        left_layout.setColumnStretch(0, 1)
        left_layout.setColumnStretch(1, 1)
        left_layout.setRowStretch(0, 1)
        left_layout.setRowStretch(1, 1)
        left_panel.setLayout(left_layout)

        control_scroll_area = QScrollArea()
        control_scroll_area.setWidget(self.bottom_panel)
        control_scroll_area.setWidgetResizable(False)
        control_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        control_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        right_panel = QWidget()
        right_layout = QGridLayout()
        right_layout.addWidget(control_scroll_area, 0, 0)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_panel.setLayout(right_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setChildrenCollapsible(False)
        splitter.splitterMoved.connect(self._handle_main_splitter_moved)

        self._main_splitter = splitter
        self._left_panel = left_panel
        self._right_panel = right_panel
        self._main_splitter_initial_sizes_applied = False

        self.setCentralWidget(splitter)
        self.setWindowTitle("3D Volume Viewer")

        self.bottom_panel.on_open_requested(self._handle_open_request)
        self.bottom_panel.on_open_semantic_requested(self._handle_open_semantic_request)
        self.bottom_panel.on_open_instance_requested(self._handle_open_instance_request)
        self.bottom_panel.on_save_segmentation_requested(self._handle_save_segmentation_request)
        self.bottom_panel.on_cursor_changed(self.sync_manager.set_cursor_indices)
        self.bottom_panel.on_zoom_changed(self.sync_manager.set_zoom)
        self.bottom_panel.on_auto_level_mode_changed(self._handle_auto_level_mode_changed)
        self.bottom_panel.on_manual_level_requested(self._handle_manual_level_requested)
        self.bottom_panel.on_contrast_window_changed(self._handle_contrast_window_changed)
        self.bottom_panel.on_annotation_mode_changed(self._handle_annotation_mode_changed)
        self.bottom_panel.on_bounding_box_mode_changed(self._handle_bounding_box_mode_changed)
        self.bottom_panel.on_annotation_tool_changed(self._handle_annotation_tool_changed)
        self.bottom_panel.on_active_label_changed(self._handle_active_label_changed)
        self.bottom_panel.on_next_available_label_requested(self._handle_next_available_label_requested)
        self.bottom_panel.on_brush_radius_changed(self._handle_brush_radius_changed)
        self.bottom_panel.on_eraser_target_changed(self._handle_eraser_target_changed)
        self.bottom_panel.on_flood_fill_target_changed(self._handle_flood_fill_target_changed)
        self.bottom_panel.on_flood_fill_requested(self._handle_flood_fill_requested)
        self.bottom_panel.on_undo_requested(self._handle_undo_requested)
        self.bottom_panel.on_redo_requested(self._handle_redo_requested)
        self.bottom_panel.on_open_bounding_boxes_requested(
            self._handle_open_bounding_boxes_request
        )
        self.bottom_panel.on_save_bounding_boxes_requested(
            self._handle_save_bounding_boxes_request
        )
        self.bottom_panel.on_build_dataset_from_bboxes_requested(
            self._handle_build_dataset_from_bboxes_request
        )
        self.bottom_panel.on_load_model_requested(self._handle_load_model_request)
        self.bottom_panel.on_save_model_requested(self._handle_save_model_request)
        self.bottom_panel.on_segment_inference_requested(
            self._handle_segment_inference_request
        )
        self.bottom_panel.on_stop_inference_requested(self._handle_stop_inference_request)
        self.bottom_panel.on_train_model_requested(self._handle_train_model_request)
        self.bottom_panel.on_stop_training_requested(self._handle_stop_training_request)
        self.bottom_panel.on_median_filter_selected_requested(
            self._handle_median_filter_selected_request
        )
        self.bottom_panel.on_erosion_selected_requested(
            self._handle_erosion_selected_request
        )
        self.bottom_panel.on_dilation_selected_requested(
            self._handle_dilation_selected_request
        )
        self.bottom_panel.on_erase_bbox_segmentation_requested(
            self._handle_erase_bbox_segmentation_request
        )
        self.bottom_panel.on_bounding_box_double_clicked(
            self._handle_bounding_box_double_clicked
        )
        self.bottom_panel.on_bounding_boxes_selected(self._handle_bounding_boxes_selected)
        self.bottom_panel.on_bounding_boxes_delete_requested(self._handle_bounding_boxes_delete_requested)
        self.bottom_panel.on_bounding_boxes_label_changed(self._handle_bounding_boxes_label_changed)
        self._undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self._undo_shortcut.activated.connect(self._handle_undo_requested)
        self._redo_shortcut = QShortcut(QKeySequence("Ctrl+Y"), self)
        self._redo_shortcut.activated.connect(self._handle_redo_requested)
        self._save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        self._save_shortcut.activated.connect(self._handle_save_shortcut_requested)
        self._annotation_brush_shortcut = QShortcut(QKeySequence("Ctrl+B"), self)
        self._annotation_brush_shortcut.activated.connect(
            lambda: self._apply_annotation_tool_shortcut("brush")
        )
        self._annotation_eraser_shortcut = QShortcut(QKeySequence("Ctrl+E"), self)
        self._annotation_eraser_shortcut.activated.connect(
            lambda: self._apply_annotation_tool_shortcut("eraser")
        )
        self._annotation_flood_fill_shortcut = QShortcut(QKeySequence("Ctrl+F"), self)
        self._annotation_flood_fill_shortcut.activated.connect(
            lambda: self._apply_annotation_tool_shortcut("flood_filler")
        )
        self._app_event_filter_installed = False
        app_instance = QApplication.instance()
        if app_instance is not None:
            app_instance.installEventFilter(self)
            self._app_event_filter_installed = True
        self.sync_manager.on_state_changed(self._on_sync_state_changed)
        self._sync_bounding_boxes_ui()
        self._refresh_learning_training_ui_state()
        self._refresh_annotation_ui_state()
        self._apply_main_splitter_width_constraints()
        QTimer.singleShot(0, self._initialize_main_splitter_sizes)

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        if not self._maybe_resolve_unsaved_data_before_close():
            event.ignore()
            return
        if self._training_is_running():
            if not self._maybe_prepare_close_while_training():
                event.ignore()
                return
            app_instance = QApplication.instance()
            if app_instance is not None:
                app_instance.setQuitOnLastWindowClosed(False)
        else:
            self._clear_deferred_close_training_state()
        if self._app_event_filter_installed:
            app_instance = QApplication.instance()
            if app_instance is not None:
                app_instance.removeEventFilter(self)
            self._app_event_filter_installed = False
        event.accept()

    def resizeEvent(self, event: QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if not self._main_splitter_initial_sizes_applied:
            self._initialize_main_splitter_sizes()
        self._apply_main_splitter_width_constraints()

    def _main_splitter_total_width(self) -> int:
        sizes = tuple(int(size) for size in self._main_splitter.sizes())
        total_width = int(sum(sizes))
        if total_width > 0:
            return total_width
        return max(0, int(self._main_splitter.width()))

    def _main_splitter_control_panel_width_bounds(self) -> Tuple[int, int]:
        total_width = self._main_splitter_total_width()
        if total_width <= 1:
            return (0, total_width)

        min_width = max(
            1,
            int(round(total_width * self._CONTROL_PANEL_MIN_WIDTH_FRACTION)),
        )
        max_width = max(
            1,
            int(round(total_width * self._CONTROL_PANEL_MAX_WIDTH_FRACTION)),
        )
        max_width = min(max_width, total_width - 1)
        min_width = min(min_width, max_width)
        return (min_width, max_width)

    def _apply_main_splitter_width_constraints(self) -> None:
        sizes = tuple(int(size) for size in self._main_splitter.sizes())
        if len(sizes) < 2:
            return

        total_width = int(sum(sizes))
        if total_width <= 0:
            return

        min_width, max_width = self._main_splitter_control_panel_width_bounds()
        self._right_panel.setMinimumWidth(min_width)
        self._right_panel.setMaximumWidth(max_width)

        current_right_width = int(sizes[1])
        clamped_right_width = max(min_width, min(current_right_width, max_width))
        if clamped_right_width == current_right_width:
            return

        self._main_splitter.setSizes([total_width - clamped_right_width, clamped_right_width])

    def _initialize_main_splitter_sizes(self) -> None:
        if self._main_splitter_initial_sizes_applied:
            return

        total_width = self._main_splitter_total_width()
        if total_width <= 1:
            return

        min_width, max_width = self._main_splitter_control_panel_width_bounds()
        target_right_width = int(
            round(total_width * self._CONTROL_PANEL_INITIAL_WIDTH_FRACTION)
        )
        target_right_width = max(min_width, min(target_right_width, max_width))
        self._main_splitter.setSizes([total_width - target_right_width, target_right_width])
        self._main_splitter_initial_sizes_applied = True
        self._apply_main_splitter_width_constraints()

    def _handle_main_splitter_moved(self, _pos: int, _index: int) -> None:
        self._apply_main_splitter_width_constraints()

    def _maybe_resolve_unsaved_data_before_close(self) -> bool:
        if not self._maybe_resolve_unsaved_segmentation(context="closing the application"):
            return False
        if not self._maybe_resolve_unsaved_bounding_boxes(context="closing the application"):
            return False
        return True

    def _maybe_prepare_close_while_training(self) -> bool:
        decision = ask_training_running_close_decision(parent=self)
        if decision == TrainingCloseDecision.CANCEL:
            self._clear_deferred_close_training_state()
            return False
        if decision == TrainingCloseDecision.STOP_AND_CLOSE:
            self._set_deferred_close_after_stop_training()
            self._request_learning_training_stop()
            return True
        if decision == TrainingCloseDecision.CONTINUE_IN_BACKGROUND:
            dialog_result = open_save_model_checkpoint_dialog(
                self,
                retry_on_overwrite_decline=True,
            )
            if not dialog_result.accepted or not dialog_result.path:
                self._clear_deferred_close_training_state()
                return False
            checkpoint_path = str(Path(dialog_result.path).expanduser())
            self._set_deferred_close_with_background_training(
                checkpoint_path=checkpoint_path,
            )
            return True
        self._clear_deferred_close_training_state()
        return False

    def _annotation_tool_from_keypress_event(self, event: object) -> Optional[AnnotationTool]:
        key = getattr(event, "key", None)
        modifiers = getattr(event, "modifiers", None)
        if not callable(key) or not callable(modifiers):
            return None
        key_value = int(key())
        modifier_mask = modifiers() & (
            Qt.ControlModifier
            | Qt.ShiftModifier
            | Qt.AltModifier
            | Qt.MetaModifier
        )
        if modifier_mask != Qt.ControlModifier:
            return None
        return _ANNOTATION_TOOL_SHORTCUT_BY_KEY.get(key_value)

    def _maybe_consume_annotation_tool_shortcut_event(self, obj: object, event: object) -> bool:
        event_type = getattr(event, "type", None)
        if not callable(event_type):
            return False
        if event_type() != QEvent.Type.KeyPress:
            return False
        if not self.isActiveWindow() or not isinstance(obj, QWidget):
            return False
        if obj is not self and not self.isAncestorOf(obj):
            return False
        shortcut_tool = self._annotation_tool_from_keypress_event(event)
        if shortcut_tool is None:
            return False
        self._apply_annotation_tool_shortcut(shortcut_tool)
        accept = getattr(event, "accept", None)
        if callable(accept):
            accept()
        return True

    @staticmethod
    def _is_text_editing_widget(widget: object) -> bool:
        if isinstance(widget, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)):
            return True
        if isinstance(widget, QComboBox) and bool(widget.isEditable()):
            return True
        return False

    def _maybe_consume_bbox_delete_shortcut_event(self, obj: object, event: object) -> bool:
        event_type = getattr(event, "type", None)
        if not callable(event_type):
            return False
        if event_type() != QEvent.Type.KeyPress:
            return False

        key_getter = getattr(event, "key", None)
        if not callable(key_getter):
            return False
        key_value = int(key_getter())
        if key_value not in (int(Qt.Key_Backspace), int(Qt.Key_Delete)):
            return False

        candidate_widget: Optional[QWidget] = obj if isinstance(obj, QWidget) else None
        app_instance = QApplication.instance()
        focus_widget: Optional[QWidget] = None
        if app_instance is not None:
            candidate_focus = app_instance.focusWidget()
            if isinstance(candidate_focus, QWidget):
                focus_widget = candidate_focus
                if candidate_widget is None:
                    candidate_widget = candidate_focus

        if candidate_widget is None or not self.isActiveWindow():
            return False
        if candidate_widget is not self and not self.isAncestorOf(candidate_widget):
            return False
        if MainWindow._is_text_editing_widget(candidate_widget):
            return False
        if focus_widget is not None and MainWindow._is_text_editing_widget(focus_widget):
            return False

        left_panel = getattr(self, "_left_panel", None)
        if not isinstance(left_panel, QWidget):
            return False
        if candidate_widget is not left_panel and not left_panel.isAncestorOf(candidate_widget):
            return False

        self._handle_bounding_box_delete_shortcut_requested()
        accept = getattr(event, "accept", None)
        if callable(accept):
            accept()
        return True

    def eventFilter(self, obj, event) -> bool:  # type: ignore[override]
        if self._maybe_consume_bbox_delete_shortcut_event(obj, event):
            return True
        if self._maybe_consume_annotation_tool_shortcut_event(obj, event):
            return True
        return super().eventFilter(obj, event)

    def set_volume(self, volume: VolumeData, levels: Optional[Tuple[VolumeData, ...]] = None) -> bool:
        # Validate and bind the new raw volume first; if this fails (for example:
        # NaN/Inf scan rejection), keep the current window state untouched.
        self.renderer.attach_volume(volume, levels=levels)

        if self._semantic_volume is not None:
            self._semantic_volume = None
            self._semantic_worker = None
        if self._instance_volume is not None:
            self._instance_volume = None
            self._instance_worker = None
        self._segmentation_editor = None
        self._pending_render_view_ids.clear()
        self._render_flush_scheduled = False
        self._pending_annotation_peer_view_ids.clear()
        self._annotation_dirty_views.clear()
        self._annotation_peer_flush_scheduled = False
        self._bbox_drag_active = False
        self._bbox_drag_source_view_id = None
        self._bbox_pending_peer_view_ids.clear()
        self._bbox_peer_flush_scheduled = False
        self._bbox_drag_staged_history_updates.clear()
        self._annotation_modification_active = False
        self._annotation_modification_view_id = None
        self._annotation_labels_dirty = False
        self._deferred_hover_readout = False
        self._deferred_picked_readout = False
        self._global_history.clear()
        self._clear_picker_selection()
        self.renderer.detach_segmentation()
        self._raw_volume = volume
        self._bbox_manager = BoundingBoxManager(volume.info.shape)
        self._bbox_manager.on_changed(self._on_bounding_boxes_changed)
        self._sync_bounding_boxes_ui()
        self._sync_contrast_controls_from_renderer()
        self._sync_level_mode_controls_from_renderer()
        self.bottom_panel.set_cursor_range(volume.info.shape)
        self.sync_manager.set_volume_info(volume.info)
        self.state.volume_loaded = True
        self.bottom_panel.set_pyramid_levels(len(levels) if levels else 1, kind="Raw")
        self.bottom_panel.set_active_levels(
            axial=(0, 1),
            coronal=(0, 1),
            sagittal=(0, 1),
            forced=not self.renderer.is_auto_level_enabled(),
        )
        if self.state.annotation_mode_enabled:
            self._ensure_editable_segmentation_for_annotation()
        self._refresh_annotation_ui_state()
        return True

    def set_semantic_volume(self, volume: VolumeData, levels: Optional[Tuple[VolumeData, ...]] = None) -> bool:
        del levels  # Active editable segmentation is attached as a writable in-memory volume.
        if not self._is_valid_segmentation_dtype(volume):
            show_warning(
                "Semantic map dtype must be int8/16/32/64 or uint8/16/32/64.",
                parent=self,
            )
            return False
        if self._raw_volume is not None and self._raw_volume.info.shape != volume.info.shape:
            show_warning(
                "Semantic map shape does not match current raw image.",
                parent=self,
            )
            return False
        self._clear_picker_selection()
        self._annotation_kind = "semantic"
        self._bbox_drag_staged_history_updates.clear()
        self._global_history.clear()
        editor = SegmentationEditor.from_volume(volume, kind="semantic")
        self._attach_segmentation_editor(editor, kind="semantic")
        self.bottom_panel.set_pyramid_levels(1, kind="Semantic")
        self._refresh_annotation_ui_state()
        return True

    def set_instance_volume(self, volume: VolumeData, levels: Optional[Tuple[VolumeData, ...]] = None) -> bool:
        del levels  # Active editable segmentation is attached as a writable in-memory volume.
        if not self._is_valid_instance_dtype(volume):
            show_warning(
                "Instance map dtype must be int8/16/32/64 or uint8/16/32/64.",
                parent=self,
            )
            return False
        if self._raw_volume is not None and self._raw_volume.info.shape != volume.info.shape:
            show_warning(
                "Instance map shape does not match current raw image.",
                parent=self,
            )
            return False
        self._clear_picker_selection()
        self._annotation_kind = "instance"
        self._bbox_drag_staged_history_updates.clear()
        self._global_history.clear()
        editor = SegmentationEditor.from_volume(volume, kind="instance")
        self._attach_segmentation_editor(editor, kind="instance")
        self.bottom_panel.set_pyramid_levels(1, kind="Instance")
        self._refresh_annotation_ui_state()
        return True

    def set_annotation_mode(
        self,
        enabled: bool,
        *,
        kind: Optional[SegmentationKind] = None,
    ) -> bool:
        if kind is not None:
            self._annotation_kind = kind
        self.state.annotation_mode_enabled = bool(enabled)
        if self.state.annotation_mode_enabled:
            self.state.bbox_mode_enabled = False
            self.state.pending_bbox_corner = None
        if not self.state.annotation_mode_enabled:
            self._end_annotation_modification()
            self._sync_segmentation_volume_from_editor(reattach_renderer=True)
            self.render_all()
            self._refresh_annotation_ui_state()
            return True
        success = self._ensure_editable_segmentation_for_annotation()
        self._refresh_annotation_ui_state()
        return success

    def segmentation_editor(self) -> Optional[SegmentationEditor]:
        return self._segmentation_editor

    def semantic_volume(self) -> Optional[VolumeData]:
        return self._semantic_volume

    def instance_volume(self) -> Optional[VolumeData]:
        return self._instance_volume

    def bounding_box_manager(self) -> BoundingBoxManager:
        return self._bbox_manager

    def _overlay_bounding_boxes(self) -> Tuple[BoundingBox, ...]:
        return self._bbox_manager.boxes()

    def _overlay_selected_bounding_box_id(self) -> Optional[str]:
        return self._bbox_manager.selected_id

    def _on_bounding_boxes_changed(self, _change: BoundingBoxChange) -> None:
        self._sync_bounding_boxes_ui()
        if not self._bbox_drag_active:
            for view in self.views.values():
                view.refresh_overlay()
            return

        source_view_id = self._bbox_drag_source_view_id
        if source_view_id is None or source_view_id not in self.views:
            for view in self.views.values():
                view.refresh_overlay()
            return

        self.views[source_view_id].refresh_overlay()
        self._queue_bbox_peer_overlays(source_view_id=source_view_id)

    def _sync_bounding_boxes_ui(self) -> None:
        self.bottom_panel.set_bounding_boxes(self._bbox_manager.boxes())
        selected_id = self._bbox_manager.selected_id
        if selected_id is None:
            selected_ids_getter = getattr(self.bottom_panel, "selected_bounding_boxes", None)
            if callable(selected_ids_getter):
                selected_ids = tuple(selected_ids_getter())
                if len(selected_ids) > 1:
                    self.bottom_panel.set_selected_bounding_boxes(selected_ids)
                    return
        self.bottom_panel.set_selected_bounding_box(selected_id)

    def _handle_bounding_boxes_selected(self, box_ids: Tuple[str, ...]) -> None:
        normalized_ids = []
        seen_ids = set()
        for raw_box_id in tuple(box_ids):
            box_id = str(raw_box_id).strip()
            if not box_id or box_id in seen_ids:
                continue
            normalized_ids.append(box_id)
            seen_ids.add(box_id)
        selected_id = normalized_ids[0] if len(normalized_ids) == 1 else None
        try:
            self._bbox_manager.select(selected_id)
        except KeyError:
            self._sync_bounding_boxes_ui()

    def _handle_bounding_box_double_clicked(self, box_id: str) -> None:
        normalized_id = str(box_id).strip()
        if not normalized_id:
            return
        box_getter = getattr(self._bbox_manager, "get", None)
        if callable(box_getter):
            box = box_getter(normalized_id)
        else:
            boxes_getter = getattr(self._bbox_manager, "boxes", None)
            if not callable(boxes_getter):
                return
            boxes_by_id = {box.id: box for box in boxes_getter()}
            box = boxes_by_id.get(normalized_id)
        if box is None:
            return
        center = getattr(box, "center_index_space", None)
        if not isinstance(center, tuple) or len(center) != 3:
            return
        try:
            cursor_indices = [
                MainWindow._round_to_nearest_index(center[0]),
                MainWindow._round_to_nearest_index(center[1]),
                MainWindow._round_to_nearest_index(center[2]),
            ]
        except Exception:
            return

        volume_shape = getattr(self._bbox_manager, "volume_shape", None)
        if isinstance(volume_shape, tuple) and len(volume_shape) == 3:
            for axis in (0, 1, 2):
                try:
                    max_index = max(0, int(volume_shape[axis]) - 1)
                except Exception:
                    continue
                cursor_indices[axis] = max(0, min(int(cursor_indices[axis]), max_index))

        set_cursor_indices = getattr(self.sync_manager, "set_cursor_indices", None)
        if callable(set_cursor_indices):
            set_cursor_indices(
                (
                    int(cursor_indices[0]),
                    int(cursor_indices[1]),
                    int(cursor_indices[2]),
                )
            )

        select_box = getattr(self._bbox_manager, "select", None)
        if callable(select_box):
            try:
                select_box(normalized_id)
            except KeyError:
                return

        sync_bounding_boxes_ui = getattr(self, "_sync_bounding_boxes_ui", None)
        if callable(sync_bounding_boxes_ui):
            sync_bounding_boxes_ui()

        hover_updater = getattr(self, "_request_hover_readout", None)
        if not callable(hover_updater):
            hover_updater = getattr(self, "_refresh_hover_readout", None)
        if callable(hover_updater):
            hover_updater()

        picked_updater = getattr(self, "_request_picked_readout", None)
        if not callable(picked_updater):
            picked_updater = getattr(self, "_refresh_picked_readout", None)
        if callable(picked_updater):
            picked_updater()

    @staticmethod
    def _round_to_nearest_index(value: object) -> int:
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"Index value must be finite, got {numeric!r}")
        if numeric >= 0.0:
            return int(np.floor(numeric + 0.5))
        return int(np.ceil(numeric - 0.5))

    def _handle_bounding_box_selected(self, box_id: Optional[str]) -> None:
        try:
            self._bbox_manager.select(box_id)
        except KeyError:
            self._sync_bounding_boxes_ui()

    def _handle_bounding_boxes_label_changed(
        self,
        box_ids: Tuple[str, ...],
        label: BoundingBoxLabel,
    ) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            return
        normalized_label = str(label).strip().lower()
        if normalized_label not in ("train", "validation", "inference"):
            self._sync_bounding_boxes_ui()
            return
        next_label = cast(BoundingBoxLabel, normalized_label)

        normalized_ids = []
        seen_ids = set()
        for raw_box_id in tuple(box_ids):
            box_id = str(raw_box_id).strip()
            if not box_id or box_id in seen_ids:
                continue
            normalized_ids.append(box_id)
            seen_ids.add(box_id)
        if not normalized_ids:
            return
        self._finalize_bbox_history_transaction()
        transaction_started = False
        updated_any = False
        try:
            self._global_history.begin_transaction("bbox_label_selected")
            transaction_started = True
            for box_id in normalized_ids:
                before_box = self._bbox_manager.get(box_id)
                if before_box is None:
                    continue
                if before_box.label == next_label:
                    continue
                before_selected_id = self._bbox_manager.selected_id
                try:
                    after_box = BoundingBox(
                        id=before_box.id,
                        z0=before_box.z0,
                        z1=before_box.z1,
                        y0=before_box.y0,
                        y1=before_box.y1,
                        x0=before_box.x0,
                        x1=before_box.x1,
                        label=next_label,
                    )
                    self._bbox_manager.replace(box_id, after_box)
                except Exception as exc:
                    show_warning(str(exc), parent=self)
                    self._sync_bounding_boxes_ui()
                    continue
                after_selected_id = self._bbox_manager.selected_id
                self._global_history.push(
                    BoundingBoxUpdateCommand(
                        manager=self._bbox_manager,
                        before_box=before_box,
                        after_box=after_box,
                        before_selected_id=before_selected_id,
                        after_selected_id=after_selected_id,
                        bytes_used=estimate_bounding_box_history_bytes(
                            before_box=before_box,
                            after_box=after_box,
                        ),
                    )
                )
                updated_any = True
        finally:
            if transaction_started and self._global_history.in_transaction():
                self._global_history.commit_transaction()
        if updated_any:
            self._refresh_undo_ui_state()

    def _handle_bounding_box_label_changed(
        self,
        box_id: str,
        label: BoundingBoxLabel,
    ) -> None:
        normalized_box_id = str(box_id).strip()
        if not normalized_box_id:
            self._sync_bounding_boxes_ui()
            return
        self._handle_bounding_boxes_label_changed((normalized_box_id,), label)

    def _handle_bounding_boxes_delete_requested(self, box_ids: Tuple[str, ...]) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            return
        normalized_ids = []
        seen_ids = set()
        for raw_box_id in tuple(box_ids):
            box_id = str(raw_box_id).strip()
            if not box_id or box_id in seen_ids:
                continue
            normalized_ids.append(box_id)
            seen_ids.add(box_id)
        if not normalized_ids:
            return
        self._finalize_bbox_history_transaction()
        transaction_started = False
        deleted_any = False
        try:
            self._global_history.begin_transaction("bbox_delete_selected")
            transaction_started = True
            for box_id in normalized_ids:
                before_box = self._bbox_manager.get(box_id)
                if before_box is None:
                    continue
                before_selected_id = self._bbox_manager.selected_id
                if not self._bbox_manager.delete(box_id):
                    continue
                after_selected_id = self._bbox_manager.selected_id
                self._global_history.push(
                    BoundingBoxDeleteCommand(
                        manager=self._bbox_manager,
                        box=before_box,
                        before_selected_id=before_selected_id,
                        after_selected_id=after_selected_id,
                        bytes_used=estimate_bounding_box_history_bytes(before_box=before_box),
                    )
                )
                deleted_any = True
        finally:
            if transaction_started and self._global_history.in_transaction():
                self._global_history.commit_transaction()
        if not deleted_any:
            return
        self.bottom_panel.set_selected_bounding_boxes(tuple())
        self._bbox_manager.select(None)
        self._refresh_undo_ui_state()

    def _handle_bounding_box_delete_requested(self, box_id: str) -> None:
        normalized_box_id = str(box_id).strip()
        if not normalized_box_id:
            return
        self._handle_bounding_boxes_delete_requested((normalized_box_id,))

    def _handle_bounding_box_delete_shortcut_requested(self) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            return
        selected_ids: Tuple[str, ...] = tuple()
        selected_ids_getter = getattr(self.bottom_panel, "selected_bounding_boxes", None)
        if callable(selected_ids_getter):
            try:
                selected_ids = tuple(selected_ids_getter())
            except Exception:
                selected_ids = tuple()
        if not selected_ids:
            selected_id = self._bbox_manager.selected_id
            if selected_id:
                selected_ids = (selected_id,)
        if not selected_ids:
            return
        self._handle_bounding_boxes_delete_requested(selected_ids)

    def _handle_bounding_box_drag_started(self, source_view_id: ViewId) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            return
        self._bbox_drag_active = True
        self._bbox_drag_source_view_id = source_view_id
        self._bbox_pending_peer_view_ids.clear()
        self._bbox_peer_flush_scheduled = False
        self._finalize_bbox_history_transaction()
        self._bbox_drag_staged_history_updates.clear()
        self._global_history.begin_transaction("bbox_drag")

    def _handle_bounding_box_drag_finished(self, source_view_id: ViewId) -> None:
        if self._bbox_drag_source_view_id == source_view_id:
            self._bbox_drag_source_view_id = None
        self._bbox_drag_active = False
        self._finalize_bbox_history_transaction()
        self._flush_bbox_peer_overlays()
        for view in self.views.values():
            view.refresh_overlay()

    def _queue_bbox_peer_overlays(self, *, source_view_id: ViewId) -> None:
        for view_id in self.views:
            if view_id != source_view_id:
                self._bbox_pending_peer_view_ids.add(view_id)
        if self._bbox_peer_flush_scheduled:
            return
        self._bbox_peer_flush_scheduled = True
        QTimer.singleShot(
            self._bbox_peer_redraw_interval_ms,
            self._flush_bbox_peer_overlays,
        )

    def _flush_bbox_peer_overlays(self) -> None:
        self._bbox_peer_flush_scheduled = False
        if not self._bbox_pending_peer_view_ids:
            return
        pending = set(self._bbox_pending_peer_view_ids)
        self._bbox_pending_peer_view_ids.clear()
        for view_id in pending:
            view = self.views.get(view_id)
            if view is not None:
                view.refresh_overlay()
        if self._bbox_pending_peer_view_ids and not self._bbox_peer_flush_scheduled:
            self._bbox_peer_flush_scheduled = True
            QTimer.singleShot(
                self._bbox_peer_redraw_interval_ms,
                self._flush_bbox_peer_overlays,
            )

    def _handle_bounding_box_face_moved(
        self,
        box_id: str,
        face: FaceId,
        boundary: int,
    ) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            return
        before_box = self._bbox_manager.get(box_id)
        before_selected_id = self._bbox_manager.selected_id
        if before_box is None:
            self._sync_bounding_boxes_ui()
            return
        try:
            after_box = self._bbox_manager.move_face(box_id, face, int(boundary))
        except KeyError:
            self._sync_bounding_boxes_ui()
            return
        if after_box == before_box:
            return
        after_selected_id = self._bbox_manager.selected_id
        if self._bbox_drag_active and self._global_history.in_transaction():
            self._stage_bounding_box_drag_update(
                box_id=box_id,
                before_box=before_box,
                after_box=after_box,
                before_selected_id=before_selected_id,
                after_selected_id=after_selected_id,
            )
            return
        self._push_global_history_command(
            BoundingBoxUpdateCommand(
                manager=self._bbox_manager,
                before_box=before_box,
                after_box=after_box,
                before_selected_id=before_selected_id,
                after_selected_id=after_selected_id,
                bytes_used=estimate_bounding_box_history_bytes(
                    before_box=before_box,
                    after_box=after_box,
                ),
            )
        )

    def _handle_bounding_box_translated(
        self,
        box_id: str,
        dz: int,
        dy: int,
        dx: int,
    ) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            return
        step = (int(dz), int(dy), int(dx))
        if step == (0, 0, 0):
            return
        before_box = self._bbox_manager.get(box_id)
        before_selected_id = self._bbox_manager.selected_id
        if before_box is None:
            self._sync_bounding_boxes_ui()
            return
        try:
            after_box = self._bbox_manager.move(
                box_id,
                dz=step[0],
                dy=step[1],
                dx=step[2],
            )
        except KeyError:
            self._sync_bounding_boxes_ui()
            return
        if after_box == before_box:
            return
        after_selected_id = self._bbox_manager.selected_id
        if self._bbox_drag_active and self._global_history.in_transaction():
            self._stage_bounding_box_drag_update(
                box_id=box_id,
                before_box=before_box,
                after_box=after_box,
                before_selected_id=before_selected_id,
                after_selected_id=after_selected_id,
            )
            return
        self._push_global_history_command(
            BoundingBoxUpdateCommand(
                manager=self._bbox_manager,
                before_box=before_box,
                after_box=after_box,
                before_selected_id=before_selected_id,
                after_selected_id=after_selected_id,
                bytes_used=estimate_bounding_box_history_bytes(
                    before_box=before_box,
                    after_box=after_box,
                ),
            )
        )

    def _mark_bounding_boxes_clean(self) -> None:
        self._bbox_manager.mark_clean()

    def load_bounding_boxes_path(self, path: str, *, show_success: bool = False) -> bool:
        if not self.state.volume_loaded or self._raw_volume is None:
            show_warning(
                "Load a raw volume before opening bounding boxes.",
                parent=self,
            )
            return False

        normalized_path = str(Path(path).expanduser())
        try:
            payload = load_bounding_boxes(
                normalized_path,
                expected_shape=self._bbox_manager.volume_shape,
            )
            self._bbox_manager.replace_all(payload.boxes, selected_id=None, mark_clean=True)
            # External replacement invalidates bbox command replay assumptions.
            self._bbox_drag_staged_history_updates.clear()
            self._global_history.clear()
        except Exception as exc:
            show_warning(str(exc), parent=self)
            return False

        if show_success:
            show_info(
                f"Loaded {len(payload.boxes)} bounding box(es) from {normalized_path}",
                parent=self,
            )
        self._refresh_annotation_ui_state()
        return True

    def _handle_open_bounding_boxes_request(self) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            return
        if not self.state.volume_loaded or self._raw_volume is None:
            show_warning(
                "Load a raw volume before opening bounding boxes.",
                parent=self,
            )
            return

        had_unsaved_boxes = self._has_unsaved_bounding_box_changes()
        if had_unsaved_boxes and not self._maybe_resolve_unsaved_bounding_boxes(
            context="loading bounding boxes from a file"
        ):
            return

        if self._bbox_manager.boxes() and not had_unsaved_boxes:
            if not confirm_replace_bounding_boxes(parent=self):
                return

        result = open_bounding_boxes_dialog(self)
        if not result.accepted or not result.path:
            return
        self.load_bounding_boxes_path(result.path, show_success=True)

    def _handle_save_bounding_boxes_request(self) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            return
        self._save_bounding_boxes_with_dialog()

    def _abort_if_learning_training_running(self) -> bool:
        if not self._training_is_running():
            return False
        show_warning(
            "A training is running. Wait for it to finish before launching another learning action.",
            parent=self,
        )
        return True

    def _handle_build_dataset_from_bboxes_request(self) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            return
        if self._abort_if_learning_training_running():
            return
        self._build_dataset_from_bboxes_with_dialog()

    def _handle_load_model_request(self) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            return
        if self._abort_if_learning_training_running():
            return
        self._instantiate_foundation_model_with_dialog()

    def _handle_save_model_request(self) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            return
        if self._abort_if_learning_training_running():
            return
        self._save_model_with_dialog()

    def _save_model_with_dialog(self) -> bool:
        runtime = get_current_learning_model_runtime()
        if runtime is None:
            show_warning(
                "Load a model before saving.",
                parent=self,
            )
            return False

        dialog_result = open_save_model_checkpoint_dialog(self)
        if not dialog_result.accepted or not dialog_result.path:
            return False

        checkpoint_path = str(Path(dialog_result.path).expanduser())
        if Path(checkpoint_path).suffix.lower() != ".cp":
            show_warning(
                "Model checkpoints must use the .cp extension.",
                parent=self,
            )
            return False

        try:
            self._save_model_runtime_checkpoint(runtime, checkpoint_path=checkpoint_path)
        except Exception as exc:
            show_warning(_exception_message(exc), parent=self)
            return False

        show_info(
            (
                "Model checkpoint saved.\n"
                f"- checkpoint: {checkpoint_path}\n"
                f"- num_classes: {runtime.num_classes}\n"
                f"- device_ids: {runtime.device_ids}"
            ),
            parent=self,
        )
        return True

    def _save_model_runtime_checkpoint(
        self,
        runtime: object,
        *,
        checkpoint_path: str,
    ) -> None:
        save_foundation_model_checkpoint(
            runtime=runtime,
            checkpoint_path=checkpoint_path,
        )

    # Backward-compatible alias kept for existing tests/callers.
    def _handle_instantiate_model_request(self) -> None:
        if self._abort_if_learning_training_running():
            return
        self._instantiate_foundation_model_with_dialog()

    def _handle_train_model_request(self) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            return
        if self._abort_if_learning_training_running():
            return
        self._train_model_on_dataset_with_dialog()

    def _handle_segment_inference_request(self) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            return
        if self._abort_if_learning_training_running():
            return
        self._segment_inference_bboxes_with_dialog()

    def _handle_stop_inference_request(self) -> None:
        self._request_learning_inference_stop()

    def _handle_stop_training_request(self) -> None:
        self._request_learning_training_stop()

    def _handle_median_filter_selected_request(self) -> None:
        self._handle_selected_bbox_segmentation_processing_request("median_filter")

    def _handle_erosion_selected_request(self) -> None:
        self._handle_selected_bbox_segmentation_processing_request("erosion")

    def _handle_dilation_selected_request(self) -> None:
        self._handle_selected_bbox_segmentation_processing_request("dilation")

    def _handle_erase_bbox_segmentation_request(self) -> None:
        self._erase_selected_bbox_segmentation()

    def _handle_selected_bbox_segmentation_processing_request(
        self,
        operation: BBoxSegmentationOperation,
    ) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            return
        self._process_selected_bbox_segmentation_operation(operation)

    def _erase_selected_bbox_segmentation(self) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            return
        selected_ids_getter = getattr(self.bottom_panel, "selected_bounding_boxes", None)
        if callable(selected_ids_getter):
            raw_selected_ids = list(selected_ids_getter())
        else:
            panel_state = getattr(self.bottom_panel, "state", None)
            raw_selected_ids = list(getattr(panel_state, "bbox_selected_ids", tuple()))

        manager_selected_id = getattr(self._bbox_manager, "selected_id", None)
        if manager_selected_id is not None:
            raw_selected_ids.append(manager_selected_id)

        selected_ids = []
        seen_ids = set()
        for raw_box_id in tuple(raw_selected_ids):
            box_id = str(raw_box_id).strip()
            if not box_id or box_id in seen_ids:
                continue
            selected_ids.append(box_id)
            seen_ids.add(box_id)

        if not selected_ids:
            show_warning(
                "Select one or more bounding boxes before erasing bbox segmentation.",
                parent=self,
            )
            return

        editor = self._segmentation_editor
        if editor is None:
            if not self._ensure_editable_segmentation_for_annotation():
                show_warning(
                    "No semantic or instance segmentation map is available for bbox segmentation erasing.",
                    parent=self,
                )
                return
            editor = self._segmentation_editor
        if editor is None:
            show_warning(
                "No semantic or instance segmentation map is available for bbox segmentation erasing.",
                parent=self,
            )
            return

        selected_boxes = MainWindow._resolve_selected_bounding_boxes(self, selected_ids)
        if not selected_boxes:
            show_warning(
                "Selected bounding boxes are no longer available.",
                parent=self,
            )
            return

        z_bounds, y_bounds, x_bounds, union_mask = MainWindow._build_selected_bbox_union_domain(
            selected_boxes
        )
        erase_coordinates = MainWindow._mask_to_absolute_coordinates(
            union_mask,
            origin=(int(z_bounds[0]), int(y_bounds[0]), int(x_bounds[0])),
        )

        end_annotation_modification = getattr(self, "_end_annotation_modification", None)
        if callable(end_annotation_modification):
            end_annotation_modification()

        operation_name = "erase_bbox_segmentation_selected"
        editor.begin_modification(operation_name)
        try:
            if erase_coordinates.size > 0:
                editor.erase(
                    erase_coordinates,
                    operation_name=operation_name,
                    ignore_out_of_bounds=False,
                )
        except Exception as exc:
            editor.cancel_modification()
            show_warning(str(exc), parent=self)
            return

        committed_operation = editor.commit_modification()
        self._record_global_history_for_segmentation_operation(committed_operation)
        changed_voxels = int(
            getattr(
                committed_operation,
                "changed_voxels",
                int(erase_coordinates.shape[0]),
            )
        )
        if changed_voxels < 0:
            changed_voxels = 0

        if changed_voxels > 0:
            sync_renderer_labels = getattr(self, "_sync_renderer_segmentation_labels", None)
            if callable(sync_renderer_labels):
                sync_renderer_labels()
            hover_updater = getattr(self, "_request_hover_readout", None)
            if not callable(hover_updater):
                hover_updater = getattr(self, "_refresh_hover_readout", None)
            if callable(hover_updater):
                hover_updater()
            picked_updater = getattr(self, "_request_picked_readout", None)
            if not callable(picked_updater):
                picked_updater = getattr(self, "_refresh_picked_readout", None)
            if callable(picked_updater):
                picked_updater()
            render_all = getattr(self, "render_all", None)
            if callable(render_all):
                render_all()

        refresh_annotation_ui_state = getattr(self, "_refresh_annotation_ui_state", None)
        if callable(refresh_annotation_ui_state):
            refresh_annotation_ui_state()

    def _process_selected_bbox_segmentation_operation(
        self,
        operation: BBoxSegmentationOperation,
    ) -> None:
        selected_ids_getter = getattr(self.bottom_panel, "selected_bounding_boxes", None)
        if callable(selected_ids_getter):
            raw_selected_ids = tuple(selected_ids_getter())
        else:
            panel_state = getattr(self.bottom_panel, "state", None)
            raw_selected_ids = tuple(getattr(panel_state, "bbox_selected_ids", tuple()))

        selected_ids = []
        seen_ids = set()
        for raw_box_id in raw_selected_ids:
            box_id = str(raw_box_id).strip()
            if not box_id or box_id in seen_ids:
                continue
            selected_ids.append(box_id)
            seen_ids.add(box_id)

        if not selected_ids:
            show_warning(
                "Select one or more bounding boxes before processing selected bounding boxes.",
                parent=self,
            )
            return

        editor = self._segmentation_editor
        if editor is None:
            if not self._ensure_editable_segmentation_for_annotation():
                show_warning(
                    "No semantic or instance segmentation map is available for selected bbox processing.",
                    parent=self,
                )
                return
            editor = self._segmentation_editor
        if editor is None:
            show_warning(
                "No semantic or instance segmentation map is available for selected bbox processing.",
                parent=self,
            )
            return

        selected_boxes = MainWindow._resolve_selected_bounding_boxes(self, selected_ids)
        if not selected_boxes:
            show_warning(
                "Selected bounding boxes are no longer available.",
                parent=self,
            )
            return

        segmentation_view = editor.array_view()
        (
            core_z_bounds,
            core_y_bounds,
            core_x_bounds,
            union_mask,
            _extended_z_bounds,
            _extended_y_bounds,
            _extended_x_bounds,
        ) = MainWindow._build_selected_bbox_processing_regions(
            selected_boxes,
            volume_shape=np.shape(segmentation_view),
            halo_size=1,
        )
        segmentation_roi = np.asarray(
            segmentation_view[
                int(core_z_bounds[0]) : int(core_z_bounds[1]),
                int(core_y_bounds[0]) : int(core_y_bounds[1]),
                int(core_x_bounds[0]) : int(core_x_bounds[1]),
            ]
        )
        foreground_mask = segmentation_roi != 0
        processed_foreground_mask = MainWindow._compute_selected_bbox_binary_operation_with_halo_context(
            operation=operation,
            segmentation_volume=segmentation_view,
            core_z_bounds=core_z_bounds,
            core_y_bounds=core_y_bounds,
            core_x_bounds=core_x_bounds,
            halo_size=1,
        )
        before_foreground_mask = foreground_mask & union_mask
        after_foreground_mask = np.asarray(processed_foreground_mask, dtype=bool) & union_mask
        clear_mask = before_foreground_mask & np.logical_not(after_foreground_mask)
        set_mask = np.logical_not(before_foreground_mask) & after_foreground_mask
        clear_mask = np.asarray(clear_mask, dtype=bool) & union_mask
        set_mask = np.asarray(set_mask, dtype=bool) & union_mask
        origin = (int(core_z_bounds[0]), int(core_y_bounds[0]), int(core_x_bounds[0]))
        clear_coordinates = MainWindow._mask_to_absolute_coordinates(
            clear_mask,
            origin=origin,
        )
        set_coordinates = MainWindow._mask_to_absolute_coordinates(
            set_mask,
            origin=origin,
        )
        # Keep label propagation scoped to the original selected-union core region.
        # Halo context is only for the binary morphology stage above.
        set_labels = MainWindow._compute_set_mask_labels(
            segmentation_roi=segmentation_roi,
            set_mask=set_mask,
            union_mask=union_mask,
            fallback_label=int(editor.active_label),
        )

        end_annotation_modification = getattr(self, "_end_annotation_modification", None)
        if callable(end_annotation_modification):
            end_annotation_modification()

        operation_name = f"{operation}_selected"
        editor.begin_modification(operation_name)
        try:
            if clear_coordinates.size > 0:
                editor.erase(
                    clear_coordinates,
                    operation_name=operation_name,
                    ignore_out_of_bounds=False,
                )
            if set_coordinates.size > 0:
                if set_labels.shape[0] != set_coordinates.shape[0]:
                    raise ValueError(
                        "set_labels and set_coordinates must have the same length: "
                        f"labels={set_labels.shape[0]} coords={set_coordinates.shape[0]}"
                    )
                for label_value in np.unique(set_labels):
                    label_coordinates = set_coordinates[set_labels == label_value]
                    if label_coordinates.size == 0:
                        continue
                    editor.assign(
                        label_coordinates,
                        label=int(label_value),
                        operation_name=operation_name,
                        ignore_out_of_bounds=False,
                    )
        except Exception as exc:
            editor.cancel_modification()
            show_warning(str(exc), parent=self)
            return

        committed_operation = editor.commit_modification()
        self._record_global_history_for_segmentation_operation(committed_operation)
        changed_voxels = int(
            getattr(
                committed_operation,
                "changed_voxels",
                int(clear_coordinates.shape[0] + set_coordinates.shape[0]),
            )
        )
        if changed_voxels < 0:
            changed_voxels = 0

        if changed_voxels > 0:
            sync_renderer_labels = getattr(self, "_sync_renderer_segmentation_labels", None)
            if callable(sync_renderer_labels):
                sync_renderer_labels()
            hover_updater = getattr(self, "_request_hover_readout", None)
            if not callable(hover_updater):
                hover_updater = getattr(self, "_refresh_hover_readout", None)
            if callable(hover_updater):
                hover_updater()
            picked_updater = getattr(self, "_request_picked_readout", None)
            if not callable(picked_updater):
                picked_updater = getattr(self, "_refresh_picked_readout", None)
            if callable(picked_updater):
                picked_updater()
            render_all = getattr(self, "render_all", None)
            if callable(render_all):
                render_all()

        refresh_annotation_ui_state = getattr(self, "_refresh_annotation_ui_state", None)
        if callable(refresh_annotation_ui_state):
            refresh_annotation_ui_state()

        show_info(
            "\n".join(
                (
                    f"{MainWindow._bbox_segmentation_operation_display_name(operation)} processing is over.",
                    f"- selected bounding boxes: {len(selected_boxes)}",
                    f"- changed voxels: {changed_voxels}",
                )
            ),
            parent=self,
        )

    def _resolve_selected_bounding_boxes(
        self,
        selected_ids: Sequence[str],
    ) -> Tuple[BoundingBox, ...]:
        boxes_by_id = {box.id: box for box in self._bbox_manager.boxes()}
        resolved = []
        seen_ids = set()
        for raw_box_id in tuple(selected_ids):
            box_id = str(raw_box_id).strip()
            if not box_id or box_id in seen_ids:
                continue
            box = boxes_by_id.get(box_id)
            if box is None:
                continue
            resolved.append(box)
            seen_ids.add(box_id)
        return tuple(resolved)

    @staticmethod
    def _build_selected_bbox_union_domain(
        boxes: Sequence[BoundingBox],
    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], np.ndarray]:
        normalized_boxes = tuple(boxes)
        if not normalized_boxes:
            raise ValueError("At least one selected bounding box is required.")

        z_bounds = (
            min(int(box.z0) for box in normalized_boxes),
            max(int(box.z1) for box in normalized_boxes),
        )
        y_bounds = (
            min(int(box.y0) for box in normalized_boxes),
            max(int(box.y1) for box in normalized_boxes),
        )
        x_bounds = (
            min(int(box.x0) for box in normalized_boxes),
            max(int(box.x1) for box in normalized_boxes),
        )

        union_mask = np.zeros(
            (
                int(z_bounds[1] - z_bounds[0]),
                int(y_bounds[1] - y_bounds[0]),
                int(x_bounds[1] - x_bounds[0]),
            ),
            dtype=bool,
        )
        for box in normalized_boxes:
            union_mask[
                int(box.z0 - z_bounds[0]) : int(box.z1 - z_bounds[0]),
                int(box.y0 - y_bounds[0]) : int(box.y1 - y_bounds[0]),
                int(box.x0 - x_bounds[0]) : int(box.x1 - x_bounds[0]),
            ] = True

        return z_bounds, y_bounds, x_bounds, union_mask

    @staticmethod
    def _expand_axis_bounds_with_halo(
        bounds: Tuple[int, int],
        *,
        axis_length: int,
        halo_size: int,
    ) -> Tuple[int, int]:
        start = int(bounds[0])
        end = int(bounds[1])
        normalized_axis_length = int(axis_length)
        normalized_halo_size = int(halo_size)

        if normalized_axis_length <= 0:
            raise ValueError(
                "axis_length must be positive when expanding bounds with halo."
            )
        if start < 0 or end <= start or end > normalized_axis_length:
            raise ValueError(
                "bounds must satisfy 0 <= start < end <= axis_length: "
                f"bounds=({start}, {end}) axis_length={normalized_axis_length}"
            )
        if normalized_halo_size < 0:
            raise ValueError("halo_size must be >= 0")

        return (
            max(0, start - normalized_halo_size),
            min(normalized_axis_length, end + normalized_halo_size),
        )

    @staticmethod
    def _build_selected_bbox_processing_regions(
        boxes: Sequence[BoundingBox],
        *,
        volume_shape: Sequence[int],
        halo_size: int = 1,
    ) -> Tuple[
        Tuple[int, int],
        Tuple[int, int],
        Tuple[int, int],
        np.ndarray,
        Tuple[int, int],
        Tuple[int, int],
        Tuple[int, int],
    ]:
        normalized_volume_shape = tuple(int(dim) for dim in tuple(volume_shape))
        if len(normalized_volume_shape) != 3:
            raise ValueError(
                "volume_shape must be a 3D shape (z, y, x), "
                f"got {normalized_volume_shape!r}."
            )
        if any(dim <= 0 for dim in normalized_volume_shape):
            raise ValueError(
                "volume_shape dimensions must be positive, "
                f"got {normalized_volume_shape!r}."
            )

        core_z_bounds, core_y_bounds, core_x_bounds, union_mask = (
            MainWindow._build_selected_bbox_union_domain(boxes)
        )
        extended_z_bounds = MainWindow._expand_axis_bounds_with_halo(
            core_z_bounds,
            axis_length=normalized_volume_shape[0],
            halo_size=halo_size,
        )
        extended_y_bounds = MainWindow._expand_axis_bounds_with_halo(
            core_y_bounds,
            axis_length=normalized_volume_shape[1],
            halo_size=halo_size,
        )
        extended_x_bounds = MainWindow._expand_axis_bounds_with_halo(
            core_x_bounds,
            axis_length=normalized_volume_shape[2],
            halo_size=halo_size,
        )

        return (
            core_z_bounds,
            core_y_bounds,
            core_x_bounds,
            union_mask,
            extended_z_bounds,
            extended_y_bounds,
            extended_x_bounds,
        )

    @staticmethod
    def _reflect_axis_indices(
        indices: np.ndarray,
        *,
        axis_length: int,
    ) -> np.ndarray:
        normalized_axis_length = int(axis_length)
        if normalized_axis_length <= 0:
            raise ValueError("axis_length must be positive for reflect indexing.")
        if normalized_axis_length == 1:
            return np.zeros(np.shape(indices), dtype=np.int64)

        period = int(2 * (normalized_axis_length - 1))
        normalized = np.mod(np.asarray(indices, dtype=np.int64), period)
        return np.where(
            normalized <= (normalized_axis_length - 1),
            normalized,
            period - normalized,
        ).astype(np.int64, copy=False)

    @staticmethod
    def _build_extended_foreground_with_halo_padding(
        *,
        segmentation_volume: np.ndarray,
        core_z_bounds: Tuple[int, int],
        core_y_bounds: Tuple[int, int],
        core_x_bounds: Tuple[int, int],
        halo_size: int = 1,
    ) -> np.ndarray:
        volume = np.asarray(segmentation_volume)
        if volume.ndim != 3:
            raise ValueError(
                "segmentation_volume must be a 3D array, "
                f"got ndim={volume.ndim}."
            )
        depth, height, width = (int(dim) for dim in volume.shape)
        normalized_halo_size = int(halo_size)
        if normalized_halo_size < 0:
            raise ValueError("halo_size must be >= 0")

        z0, z1 = (int(core_z_bounds[0]), int(core_z_bounds[1]))
        y0, y1 = (int(core_y_bounds[0]), int(core_y_bounds[1]))
        x0, x1 = (int(core_x_bounds[0]), int(core_x_bounds[1]))
        if not (0 <= z0 < z1 <= depth):
            raise ValueError(
                "core_z_bounds must satisfy 0 <= z0 < z1 <= depth: "
                f"{(z0, z1)} for depth={depth}"
            )
        if not (0 <= y0 < y1 <= height):
            raise ValueError(
                "core_y_bounds must satisfy 0 <= y0 < y1 <= height: "
                f"{(y0, y1)} for height={height}"
            )
        if not (0 <= x0 < x1 <= width):
            raise ValueError(
                "core_x_bounds must satisfy 0 <= x0 < x1 <= width: "
                f"{(x0, x1)} for width={width}"
            )

        core_foreground = np.asarray(
            volume[z0:z1, y0:y1, x0:x1] != 0,
            dtype=bool,
        )
        z_size = int(z1 - z0)
        y_size = int(y1 - y0)
        x_size = int(x1 - x0)

        requested_z = np.arange(
            z0 - normalized_halo_size,
            z1 + normalized_halo_size,
            dtype=np.int64,
        )
        requested_y = np.arange(
            y0 - normalized_halo_size,
            y1 + normalized_halo_size,
            dtype=np.int64,
        )
        requested_x = np.arange(
            x0 - normalized_halo_size,
            x1 + normalized_halo_size,
            dtype=np.int64,
        )

        reflected_z = MainWindow._reflect_axis_indices(requested_z, axis_length=depth)
        reflected_y = MainWindow._reflect_axis_indices(requested_y, axis_length=height)
        reflected_x = MainWindow._reflect_axis_indices(requested_x, axis_length=width)

        sampled_z = np.clip(reflected_z, z0, z1 - 1) - z0
        sampled_y = np.clip(reflected_y, y0, y1 - 1) - y0
        sampled_x = np.clip(reflected_x, x0, x1 - 1) - x0
        sampled_z = np.clip(sampled_z, 0, z_size - 1)
        sampled_y = np.clip(sampled_y, 0, y_size - 1)
        sampled_x = np.clip(sampled_x, 0, x_size - 1)

        expanded = np.take(core_foreground, sampled_z, axis=0)
        expanded = np.take(expanded, sampled_y, axis=1)
        expanded = np.take(expanded, sampled_x, axis=2)
        return np.asarray(expanded, dtype=bool)

    @staticmethod
    def _mask_to_absolute_coordinates(
        mask: np.ndarray,
        *,
        origin: Tuple[int, int, int],
    ) -> np.ndarray:
        local_coordinates = np.argwhere(np.asarray(mask, dtype=bool))
        if local_coordinates.size == 0:
            return np.empty((0, 3), dtype=np.int64)
        origin_array = np.asarray(origin, dtype=np.int64).reshape(1, 3)
        return np.asarray(local_coordinates, dtype=np.int64) + origin_array

    @staticmethod
    def _bbox_segmentation_operation_display_name(operation: BBoxSegmentationOperation) -> str:
        if operation == "median_filter":
            return "Median Filter Selected"
        if operation == "erosion":
            return "Erosion Selected"
        if operation == "dilation":
            return "Dilation Selected"
        raise ValueError(f"Unsupported bbox segmentation operation: {operation!r}")

    @staticmethod
    def _compute_set_mask_labels(
        *,
        segmentation_roi: np.ndarray,
        set_mask: np.ndarray,
        union_mask: np.ndarray,
        fallback_label: int,
    ) -> np.ndarray:
        labels = np.asarray(segmentation_roi)
        pending_set = np.asarray(set_mask, dtype=bool)
        domain = np.asarray(union_mask, dtype=bool)
        if labels.shape != pending_set.shape or labels.shape != domain.shape:
            raise ValueError(
                "segmentation_roi, set_mask, and union_mask must share the same shape: "
                f"labels={tuple(labels.shape)} set={tuple(pending_set.shape)} union={tuple(domain.shape)}"
            )
        if labels.ndim != 3:
            raise ValueError(
                "set-mask label propagation expects 3D arrays, "
                f"got ndim={labels.ndim}"
            )
        local_coordinates = np.argwhere(pending_set)
        if local_coordinates.size == 0:
            return np.empty((0,), dtype=np.int64)

        fallback = int(fallback_label)
        resolved_labels = np.empty((local_coordinates.shape[0],), dtype=np.int64)
        depth, height, width = labels.shape
        for index, coordinate in enumerate(local_coordinates):
            z = int(coordinate[0])
            y = int(coordinate[1])
            x = int(coordinate[2])
            z0 = max(0, z - 1)
            z1 = min(depth, z + 2)
            y0 = max(0, y - 1)
            y1 = min(height, y + 2)
            x0 = max(0, x - 1)
            x1 = min(width, x + 2)

            neighborhood_labels = np.asarray(labels[z0:z1, y0:y1, x0:x1], dtype=np.int64)
            neighborhood_domain = domain[z0:z1, y0:y1, x0:x1]
            candidate_labels = neighborhood_labels[
                np.logical_and(neighborhood_domain, neighborhood_labels != 0)
            ]
            if candidate_labels.size == 0:
                resolved_labels[index] = fallback
                continue
            values, counts = np.unique(candidate_labels, return_counts=True)
            max_count = int(np.max(counts))
            winners = values[counts == max_count]
            resolved_labels[index] = int(np.min(winners))
        return resolved_labels

    @staticmethod
    def _compute_selected_bbox_binary_operation(
        *,
        operation: BBoxSegmentationOperation,
        foreground_mask: np.ndarray,
        union_mask: np.ndarray,
    ) -> np.ndarray:
        foreground = np.asarray(foreground_mask, dtype=bool)
        domain = np.asarray(union_mask, dtype=bool)
        if foreground.shape != domain.shape:
            raise ValueError(
                "foreground_mask and union_mask must share the same shape: "
                f"foreground={tuple(foreground.shape)} union={tuple(domain.shape)}"
            )
        constrained_foreground = foreground & domain
        neighbor_counts = MainWindow._count_true_neighbors_3x3x3(constrained_foreground)

        if operation == "median_filter":
            transformed = neighbor_counts >= 14
        elif operation == "erosion":
            transformed = neighbor_counts == 27
        elif operation == "dilation":
            transformed = neighbor_counts >= 1
        else:
            raise ValueError(f"Unsupported bbox segmentation operation: {operation!r}")
        return np.asarray(transformed, dtype=bool) & domain

    @staticmethod
    def _compute_selected_bbox_binary_operation_with_halo_context(
        *,
        operation: BBoxSegmentationOperation,
        segmentation_volume: np.ndarray,
        core_z_bounds: Tuple[int, int],
        core_y_bounds: Tuple[int, int],
        core_x_bounds: Tuple[int, int],
        halo_size: int = 1,
    ) -> np.ndarray:
        normalized_halo_size = int(halo_size)
        if normalized_halo_size < 0:
            raise ValueError("halo_size must be >= 0")

        core_shape = (
            int(core_z_bounds[1]) - int(core_z_bounds[0]),
            int(core_y_bounds[1]) - int(core_y_bounds[0]),
            int(core_x_bounds[1]) - int(core_x_bounds[0]),
        )
        if any(dim <= 0 for dim in core_shape):
            raise ValueError(
                "Core bounds must define a non-empty 3D region, "
                f"got shape={core_shape}."
            )

        extended_foreground = MainWindow._build_extended_foreground_with_halo_padding(
            segmentation_volume=segmentation_volume,
            core_z_bounds=core_z_bounds,
            core_y_bounds=core_y_bounds,
            core_x_bounds=core_x_bounds,
            halo_size=normalized_halo_size,
        )
        transformed_extended = MainWindow._compute_selected_bbox_binary_operation(
            operation=operation,
            foreground_mask=extended_foreground,
            union_mask=np.ones(np.shape(extended_foreground), dtype=bool),
        )
        transformed_arr = np.asarray(transformed_extended, dtype=bool)
        if tuple(int(dim) for dim in transformed_arr.shape) == core_shape:
            # Backward-compatible path for tests that patch
            # _compute_selected_bbox_binary_operation with core-shaped returns.
            return transformed_arr

        expected_extended_shape = tuple(
            int(dim) + int(2 * normalized_halo_size) for dim in core_shape
        )
        if tuple(int(dim) for dim in transformed_arr.shape) != expected_extended_shape:
            raise ValueError(
                "Unexpected transformed mask shape for halo-aware selected-bbox processing: "
                f"got={tuple(transformed_arr.shape)} expected_core={core_shape} "
                f"expected_extended={expected_extended_shape}"
            )

        z_slice = slice(normalized_halo_size, normalized_halo_size + int(core_shape[0]))
        y_slice = slice(normalized_halo_size, normalized_halo_size + int(core_shape[1]))
        x_slice = slice(normalized_halo_size, normalized_halo_size + int(core_shape[2]))
        return np.asarray(transformed_arr[z_slice, y_slice, x_slice], dtype=bool)

    @staticmethod
    def _count_true_neighbors_3x3x3(mask: np.ndarray) -> np.ndarray:
        data = np.asarray(mask, dtype=bool)
        if data.ndim != 3:
            raise ValueError(
                f"3x3x3 neighborhood counting expects a 3D mask, got ndim={data.ndim}"
            )
        padded = np.pad(data.astype(np.uint8, copy=False), pad_width=1, mode="constant")
        counts = np.zeros(data.shape, dtype=np.uint8)
        for z_off in range(3):
            z_slice = slice(z_off, z_off + data.shape[0])
            for y_off in range(3):
                y_slice = slice(y_off, y_off + data.shape[1])
                for x_off in range(3):
                    x_slice = slice(x_off, x_off + data.shape[2])
                    counts += padded[z_slice, y_slice, x_slice]
        return counts

    def _segment_inference_bboxes_with_dialog(self) -> bool:
        model_runtime = get_current_learning_model_runtime()
        if model_runtime is None:
            show_warning(
                "Instantiate a model before running Segment Inference Bbox.",
                parent=self,
            )
            return False

        ordered_box_ids = tuple(row.box_id for row in self.bottom_panel.state.bbox_rows)
        boxes_by_id = {box.id: box for box in self._bbox_manager.boxes()}
        inference_boxes = _ordered_inference_boxes(
            ordered_box_ids=ordered_box_ids,
            boxes_by_id=boxes_by_id,
        )
        if not inference_boxes:
            show_warning(
                (
                    "At least one bounding box labeled 'inference' is required to run "
                    "Segment Inference Bbox."
                ),
                parent=self,
            )
            return False

        overlapping_pairs = _find_overlapping_box_id_pairs(inference_boxes)
        if overlapping_pairs:
            pair_text = ", ".join(
                f"{first_box_id} <-> {second_box_id}"
                for first_box_id, second_box_id in overlapping_pairs
            )
            show_warning(
                (
                    "Inference bounding boxes overlap. Overlap is not supported for "
                    "Segment Inference Bbox.\n\n"
                    f"Overlapping pairs: {pair_text}"
                ),
                parent=self,
            )
            return False

        eval_runtimes_by_box_id = get_current_learning_eval_runtimes_by_box_id()
        if not eval_runtimes_by_box_id:
            show_warning(
                (
                    "Validation runtimes are missing. Click 'Build Dataset from Bbox' "
                    "first."
                ),
                parent=self,
            )
            return False

        try:
            label_values = _resolve_shared_eval_label_values(eval_runtimes_by_box_id)
        except Exception as exc:
            show_warning(str(exc), parent=self)
            return False

        active_segmentation = self._active_segmentation_volume()
        if active_segmentation is not None:
            active_kind, _active_volume = active_segmentation
            if active_kind == "instance" and self._semantic_volume is None:
                show_warning(
                    (
                        "Segment Inference Bbox requires a semantic map, but the active "
                        "map is instance and no semantic map is loaded."
                    ),
                    parent=self,
                )
                return False
        elif self._semantic_volume is None:
            if self._raw_volume is None:
                show_warning(
                    "Load a raw volume before running Segment Inference Bbox.",
                    parent=self,
                )
                return False
            self._annotation_kind = "semantic"
            if not self._ensure_editable_segmentation_for_annotation():
                show_warning(
                    "Could not auto-create an empty semantic map for Segment Inference Bbox.",
                    parent=self,
                )
                return False

        semantic_volume = self._semantic_volume
        if semantic_volume is None:
            show_warning(
                "A semantic map is required to run Segment Inference Bbox.",
                parent=self,
            )
            return False

        has_non_empty_inference_bbox = False
        try:
            for box in inference_boxes:
                bbox_values = np.asarray(
                    semantic_volume.get_chunk(
                        (
                            slice(int(box.z0), int(box.z1)),
                            slice(int(box.y0), int(box.y1)),
                            slice(int(box.x0), int(box.x1)),
                        )
                    )
                )
                if np.any((bbox_values != 0) & (bbox_values != -100)):
                    has_non_empty_inference_bbox = True
                    break
        except Exception as exc:
            show_warning(str(exc), parent=self)
            return False

        if has_non_empty_inference_bbox:
            if not confirm_replace_inference_bboxes(parent=self):
                return False

        if not label_values:
            show_warning(
                "Validation buffers did not provide any class label values.",
                parent=self,
            )
            return False

        raw_volume = self._raw_volume
        if raw_volume is None:
            show_warning(
                "Load a raw volume before running Segment Inference Bbox.",
                parent=self,
            )
            return False

        editor = self._segmentation_editor
        if editor is None or editor.kind != "semantic":
            show_warning(
                "A semantic map is required to run Segment Inference Bbox.",
                parent=self,
            )
            return False

        try:
            raw_array = np.asarray(
                raw_volume.get_chunk((slice(None), slice(None), slice(None)))
            )
        except Exception as exc:
            show_warning(_exception_message(exc), parent=self)
            return False

        show_navigation_only_notice = getattr(
            self,
            "_show_inference_navigation_only_notice",
            None,
        )
        if callable(show_navigation_only_notice):
            show_navigation_only_notice()
        else:
            MainWindow._show_inference_navigation_only_notice(self)

        try:
            self._start_learning_inference_background(
                model_runtime=model_runtime,
                inference_boxes=inference_boxes,
                raw_array=raw_array,
                label_values=label_values,
                volume_shape=self._bbox_manager.volume_shape,
            )
        except Exception as exc:
            self._exit_learning_inference_running_state()
            show_warning(_exception_message(exc), parent=self)
            return False
        return True

    def _show_inference_navigation_only_notice(self) -> None:
        parent = self if isinstance(self, QWidget) else None
        show_info(
            (
                "Segment Inference Bbox is starting.\n\n"
                "During inference, only navigation remains enabled:\n"
                "- Slice navigation, zoom/pan, contrast, and level controls\n"
                "- Bounding-box selection and double-click cursor jump\n\n"
                "Annotation/edit actions, undo/redo, and file/model operations are "
                "temporarily disabled.\n"
                "Use 'Stop Inference' to cancel the running inference."
            ),
            parent=parent,
        )

    def _start_learning_inference_background(
        self,
        *,
        model_runtime: object,
        inference_boxes: Sequence[BoundingBox],
        raw_array: np.ndarray,
        label_values: Sequence[int],
        volume_shape: Sequence[int],
    ) -> None:
        thread = QThread(self)
        worker = _LearningInferenceWorker()
        worker.configure(
            model_runtime=model_runtime,
            inference_boxes=inference_boxes,
            raw_array=raw_array,
            label_values=label_values,
            volume_shape=volume_shape,
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.completed.connect(self._on_learning_inference_completed)
        worker.canceled.connect(self._on_learning_inference_canceled)
        worker.failed.connect(self._on_learning_inference_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_learning_inference_thread_finished)
        thread.finished.connect(thread.deleteLater)

        try:
            self._enter_learning_inference_running_state(worker=worker, thread=thread)
            thread.start()
        except Exception:
            try:
                thread.quit()
            except Exception:
                pass
            try:
                worker.deleteLater()
            except Exception:
                pass
            try:
                thread.deleteLater()
            except Exception:
                pass
            raise

    def _apply_inference_predictions_in_single_commit(
        self,
        *,
        editor: SegmentationEditor,
        predictions: Sequence[_LearningInferencePrediction],
        initial_failure_by_box_id: Optional[Mapping[str, str]] = None,
    ) -> Tuple[int, Tuple[str, ...], Dict[str, str]]:
        if QThread.currentThread() is not self.thread():
            raise RuntimeError("Inference predictions must be applied on the main UI thread.")

        failure_by_box_id: Dict[str, str] = {}
        if isinstance(initial_failure_by_box_id, Mapping):
            for box_id, reason in tuple(initial_failure_by_box_id.items()):
                failure_by_box_id[str(box_id)] = str(reason)

        succeeded_box_ids: list[str] = []
        changed_voxel_count_total = 0

        self._end_annotation_modification()
        if self._annotation_labels_dirty:
            self._sync_renderer_segmentation_labels()
        if MainWindow._inference_stop_already_requested(self):
            raise _LearningInferenceStopRequested(
                "Inference canceled by user before applying predictions."
            )
        # Enforce one atomic history commit for the whole inference application phase.
        editor.begin_modification("segment_inference_bboxes")
        try:
            for prediction in tuple(predictions):
                if MainWindow._inference_stop_already_requested(self):
                    raise _LearningInferenceStopRequested(
                        "Inference canceled by user before commit."
                    )
                box = prediction.box
                try:
                    changed_count = _apply_predicted_bbox_to_editor(
                        editor=editor,
                        box=box,
                        predicted_bbox=prediction.predicted_bbox,
                    )
                    changed_voxel_count_total += int(changed_count)
                    succeeded_box_ids.append(str(box.id))
                except Exception as exc:
                    failure_by_box_id[str(box.id)] = _exception_message(exc)
        except _LearningInferenceStopRequested:
            cancel_modification = getattr(editor, "cancel_modification", None)
            if callable(cancel_modification):
                cancel_modification()
            raise
        else:
            committed_operation = editor.commit_modification()
            self._record_global_history_for_segmentation_operation(committed_operation)

        return (
            int(changed_voxel_count_total),
            tuple(succeeded_box_ids),
            dict(failure_by_box_id),
        )

    def _on_learning_inference_completed(self, result: object) -> None:
        try:
            if not isinstance(result, _LearningInferenceBackgroundResult):
                show_warning(
                    "Segment Inference Bbox completed with an invalid result payload.",
                    parent=self,
                )
                return
            if MainWindow._inference_stop_already_requested(self):
                self._on_learning_inference_canceled(
                    "Inference canceled by user before applying predictions."
                )
                return

            editor = self._segmentation_editor
            if editor is None or editor.kind != "semantic":
                show_warning(
                    (
                        "Segment Inference Bbox completed, but the semantic map is no longer "
                        "available. Predictions were discarded."
                    ),
                    parent=self,
                )
                return

            failure_by_box_id = dict(result.failure_by_box_id)
            cleanup_errors_by_box_id = {
                str(box_id): tuple(errors)
                for box_id, errors in tuple(result.cleanup_errors_by_box_id.items())
            }
            (
                changed_voxel_count_total,
                succeeded_box_ids,
                failure_by_box_id,
            ) = (0, tuple(), failure_by_box_id)
            (
                changed_voxel_count_total,
                succeeded_box_ids,
                failure_by_box_id,
            ) = self._apply_inference_predictions_in_single_commit(
                editor=editor,
                predictions=result.predictions,
                initial_failure_by_box_id=failure_by_box_id,
            )
        except _LearningInferenceStopRequested as exc:
            self._on_learning_inference_canceled(str(exc))
            return
        finally:
            clear_stop_requested_state = getattr(
                self,
                "_clear_learning_inference_stop_request_state",
                None,
            )
            if callable(clear_stop_requested_state):
                clear_stop_requested_state()
            else:
                MainWindow._clear_learning_inference_stop_request_state(self)

        if changed_voxel_count_total > 0:
            self._annotation_labels_dirty = True
            self._sync_renderer_segmentation_labels()
            self._request_hover_readout()
            self._request_picked_readout()
            self.render_all()
        self._refresh_annotation_ui_state()

        success_count = int(len(succeeded_box_ids))
        failure_count = int(len(failure_by_box_id))
        total_count = int(result.total_count)
        cleanup_warning_count = int(sum(len(errors) for errors in cleanup_errors_by_box_id.values()))

        if failure_count <= 0 and cleanup_warning_count <= 0:
            title_line = "Segment Inference Bbox completed: all inference bboxes succeeded."
        elif failure_count > 0 and success_count <= 0:
            title_line = "Segment Inference Bbox failed: no inference bbox was successfully processed."
        elif failure_count > 0:
            title_line = "Segment Inference Bbox completed with partial success."
        else:
            title_line = "Segment Inference Bbox completed with cleanup warnings."

        summary_lines = [
            title_line,
            f"- processed inference bboxes: {total_count}",
            f"- succeeded: {success_count}",
            f"- failed: {failure_count}",
            f"- changed voxels: {int(changed_voxel_count_total)}",
        ]
        if succeeded_box_ids:
            summary_lines.append("- succeeded bbox ids: " + ", ".join(succeeded_box_ids))
        if failure_by_box_id:
            summary_lines.append("- failed bbox reasons:")
            for box_id, reason in tuple(failure_by_box_id.items()):
                summary_lines.append(f"  - {box_id}: {reason}")
        if cleanup_errors_by_box_id:
            summary_lines.append("- cleanup warnings:")
            for box_id, errors in tuple(cleanup_errors_by_box_id.items()):
                for error in tuple(errors):
                    summary_lines.append(f"  - {box_id}: {error}")

        if failure_by_box_id or cleanup_errors_by_box_id:
            show_warning("\n".join(summary_lines), parent=self)
        else:
            show_info("\n".join(summary_lines), parent=self)

    def _on_learning_inference_canceled(self, message: str) -> None:
        try:
            normalized_message = str(message).strip()
            if not normalized_message:
                normalized_message = "Inference canceled by user."
            show_info(
                f"Segment Inference Bbox canceled: {normalized_message}",
                parent=self,
            )
        finally:
            clear_stop_requested_state = getattr(
                self,
                "_clear_learning_inference_stop_request_state",
                None,
            )
            if callable(clear_stop_requested_state):
                clear_stop_requested_state()
            else:
                MainWindow._clear_learning_inference_stop_request_state(self)

    def _on_learning_inference_failed(self, message: str) -> None:
        try:
            normalized_message = str(message).strip()
            if not normalized_message:
                normalized_message = "Unknown inference error."
            show_warning(
                f"Segment Inference Bbox aborted: {normalized_message}",
                parent=self,
            )
        finally:
            clear_stop_requested_state = getattr(
                self,
                "_clear_learning_inference_stop_request_state",
                None,
            )
            if callable(clear_stop_requested_state):
                clear_stop_requested_state()
            else:
                MainWindow._clear_learning_inference_stop_request_state(self)

    def _on_learning_inference_thread_finished(self) -> None:
        self._exit_learning_inference_running_state()

    def _request_learning_inference_stop(self) -> None:
        if not self._inference_is_running():
            return
        if MainWindow._inference_stop_already_requested(self):
            return
        worker = self._inference_worker
        request_stop = getattr(worker, "request_stop", None)
        if callable(request_stop):
            self._inference_stop_requested = True
            self._refresh_learning_inference_ui_state()
            request_stop()

    def _request_learning_training_stop(self) -> None:
        if not self._training_is_running():
            return
        worker = self._training_worker
        clear_completion_checkpoint_save_request = getattr(
            worker,
            "clear_completion_checkpoint_save_request",
            None,
        )
        if callable(clear_completion_checkpoint_save_request):
            clear_completion_checkpoint_save_request()
        request_stop = getattr(worker, "request_stop", None)
        if callable(request_stop):
            request_stop()

    def _runtime_training_provenance(
        self,
        runtime: object,
    ) -> Tuple[Optional[str], bool, int]:
        source_checkpoint_path = getattr(runtime, "checkpoint_path", None)
        trained_in_app = False
        training_run_count = 0

        hyperparameters_obj = getattr(runtime, "hyperparameters", None)
        if isinstance(hyperparameters_obj, Mapping):
            raw_source = hyperparameters_obj.get("source_checkpoint_path")
            if isinstance(raw_source, str) and raw_source.strip():
                source_checkpoint_path = raw_source.strip()

            raw_trained = hyperparameters_obj.get("trained_in_app")
            if isinstance(raw_trained, bool):
                trained_in_app = bool(raw_trained)

            raw_run_count = hyperparameters_obj.get("training_run_count")
            if isinstance(raw_run_count, Integral) and not isinstance(raw_run_count, bool):
                if int(raw_run_count) >= 0:
                    training_run_count = int(raw_run_count)

        if training_run_count > 0 and not trained_in_app:
            trained_in_app = True
        if not isinstance(source_checkpoint_path, str) or not source_checkpoint_path.strip():
            source_checkpoint_path = None

        return source_checkpoint_path, bool(trained_in_app), int(training_run_count)

    def _runtime_requires_training_reinitialization(self, runtime: object) -> bool:
        source_checkpoint_path, trained_in_app, training_run_count = self._runtime_training_provenance(
            runtime
        )
        if trained_in_app or training_run_count > 0:
            return True
        default_identity = _normalize_checkpoint_identity(
            _DEFAULT_TRAINING_FOUNDATION_CHECKPOINT_PATH
        )
        source_identity = _normalize_checkpoint_identity(source_checkpoint_path)
        return bool(default_identity is None or source_identity != default_identity)

    def _reinitialize_training_runtime_from_default_checkpoint(self) -> bool:
        checkpoint_path = _DEFAULT_TRAINING_FOUNDATION_CHECKPOINT_PATH
        try:
            preconditions = validate_foundation_model_instantiation_preconditions(
                require_min_gpu_count=2,
            )
            instantiate_foundation_model_runtime(
                num_classes=preconditions.num_classes,
                device_ids=preconditions.device_ids,
                checkpoint_path=checkpoint_path,
            )
        except Exception as exc:
            message = _exception_message(exc)
            show_warning(
                (
                    f"{message}\n\n"
                    "Training requires the default foundation checkpoint:\n"
                    f"{checkpoint_path}"
                ),
                parent=self,
            )
            return False
        return True

    def _ensure_training_runtime_for_new_training(self) -> bool:
        runtime = get_current_learning_model_runtime()
        if runtime is None:
            return self._reinitialize_training_runtime_from_default_checkpoint()
        if not self._runtime_requires_training_reinitialization(runtime):
            return True
        if not confirm_replace_training_model_with_default_checkpoint(
            checkpoint_path=_DEFAULT_TRAINING_FOUNDATION_CHECKPOINT_PATH,
            parent=self,
        ):
            return False
        return self._reinitialize_training_runtime_from_default_checkpoint()

    def _mark_current_model_runtime_as_trained(self, *, completed_epoch_count: int) -> None:
        if int(completed_epoch_count) <= 0:
            return
        runtime = get_current_learning_model_runtime()
        if runtime is None:
            return
        hyperparameters_obj = getattr(runtime, "hyperparameters", None)
        if not isinstance(hyperparameters_obj, dict):
            return
        previous_count = hyperparameters_obj.get("training_run_count", 0)
        try:
            normalized_count = int(previous_count)
        except Exception:
            normalized_count = 0
        if normalized_count < 0:
            normalized_count = 0
        hyperparameters_obj["trained_in_app"] = True
        hyperparameters_obj["training_run_count"] = normalized_count + 1
        if "source_checkpoint_path" not in hyperparameters_obj:
            checkpoint_path_obj = getattr(runtime, "checkpoint_path", None)
            if isinstance(checkpoint_path_obj, str) and checkpoint_path_obj.strip():
                hyperparameters_obj["source_checkpoint_path"] = checkpoint_path_obj

    def _train_model_on_dataset_with_dialog(self) -> bool:
        ensure_training_runtime = getattr(self, "_ensure_training_runtime_for_new_training", None)
        if callable(ensure_training_runtime):
            if not bool(ensure_training_runtime()):
                return False
        else:
            if not MainWindow._ensure_training_runtime_for_new_training(self):
                return False

        try:
            preconditions = validate_learning_model_training_preconditions(
                require_class_weights=True,
            )
        except Exception as exc:
            show_warning(str(exc), parent=self)
            return False

        try:
            self._start_learning_training_background(preconditions=preconditions)
        except Exception as exc:
            self._exit_learning_training_running_state()
            show_warning(str(exc), parent=self)
            return False
        return True

    def _start_learning_training_background(self, *, preconditions: object) -> None:
        thread = QThread(self)
        worker = _LearningTrainingWorker()
        worker.configure(preconditions=preconditions)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.completed.connect(self._on_learning_training_completed)
        worker.failed.connect(self._on_learning_training_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_learning_training_thread_finished)
        thread.finished.connect(thread.deleteLater)

        try:
            self._enter_learning_training_running_state(worker=worker, thread=thread)
            thread.start()
        except Exception:
            try:
                thread.quit()
            except Exception:
                pass
            try:
                worker.deleteLater()
            except Exception:
                pass
            try:
                thread.deleteLater()
            except Exception:
                pass
            raise

    def _on_learning_training_completed(self, result: object) -> None:
        background_close_mode = bool(
            getattr(self, "_deferred_close_after_training", False)
            and getattr(self, "_deferred_close_training_mode", "none")
            == "continue_in_background"
        )
        if not isinstance(result, LearningTrainingLoopResult):
            if background_close_mode:
                _LOGGER.error(
                    "Background training finished with an invalid result payload: %r",
                    result,
                )
            else:
                show_warning(
                    "Training finished with an invalid result payload.",
                    parent=self,
                )
            return
        normalized_reason = str(result.stop_reason).strip().lower()
        if normalized_reason == "early_stop":
            stop_reason_text = "early stop"
        elif normalized_reason == "max_epoch":
            stop_reason_text = "max epoch"
        elif normalized_reason == "user_stop":
            stop_reason_text = "stopped by user"
        else:
            if background_close_mode:
                _LOGGER.error(
                    "Background training finished with an invalid stop reason: %r",
                    result.stop_reason,
                )
            else:
                show_warning(
                    f"Training finished with an invalid stop reason: {result.stop_reason!r}.",
                    parent=self,
                )
            return
        best_epoch_text = (
            "N/A"
            if result.best_epoch_index is None
            else str(int(result.best_epoch_index))
        )
        best_accuracy_text = (
            "N/A"
            if result.best_weighted_mean_accuracy is None
            else f"{float(result.best_weighted_mean_accuracy):.6g}"
        )
        marker = getattr(self, "_mark_current_model_runtime_as_trained", None)
        if callable(marker):
            marker(completed_epoch_count=int(result.completed_epoch_count))
        else:
            MainWindow._mark_current_model_runtime_as_trained(
                self,
                completed_epoch_count=int(result.completed_epoch_count),
            )
        if background_close_mode:
            _LOGGER.info(
                "Background training completed: reason=%s, best_epoch=%s, best_weighted_accuracy=%s, checkpoint=%s",
                stop_reason_text,
                best_epoch_text,
                best_accuracy_text,
                getattr(self, "_deferred_close_checkpoint_path", None),
            )
            return
        show_info(
            (
                "Training is over.\n"
                f"- reason: {stop_reason_text}\n"
                f"- best epoch (0-based): {best_epoch_text}\n"
                f"- best weighted accuracy: {best_accuracy_text}"
            ),
            parent=self,
        )

    def _on_learning_training_failed(self, message: str) -> None:
        background_close_mode = bool(
            getattr(self, "_deferred_close_after_training", False)
            and getattr(self, "_deferred_close_training_mode", "none")
            == "continue_in_background"
        )
        normalized_message = str(message).strip()
        if not normalized_message:
            normalized_message = "Unknown training error."
        if background_close_mode:
            lowered = normalized_message.lower()
            if lowered.startswith("failed to save training completion checkpoint"):
                _LOGGER.error(
                    "Background training completion checkpoint save failed: %s",
                    normalized_message,
                )
            else:
                _LOGGER.error(
                    "Background training aborted: %s",
                    normalized_message,
                )
            return
        show_warning(
            f"Training aborted: {normalized_message}",
            parent=self,
        )

    def _on_learning_training_thread_finished(self) -> None:
        self._exit_learning_training_running_state()
        if not bool(getattr(self, "_deferred_close_after_training", False)):
            return
        clear_state = getattr(self, "_clear_deferred_close_training_state", None)
        if callable(clear_state):
            clear_state()
        else:
            MainWindow._clear_deferred_close_training_state(self)

        app_instance = QApplication.instance()
        if app_instance is None:
            return
        set_quit_on_last = getattr(app_instance, "setQuitOnLastWindowClosed", None)
        if callable(set_quit_on_last):
            try:
                set_quit_on_last(True)
            except Exception:
                pass
        quit_method = getattr(app_instance, "quit", None)
        if callable(quit_method):
            quit_method()

    def _save_bounding_boxes_with_dialog(self) -> bool:
        if not self.state.volume_loaded or self._raw_volume is None:
            show_warning(
                "Load a raw volume before saving bounding boxes.",
                parent=self,
            )
            return False

        while True:
            result = open_save_bounding_boxes_dialog(self)
            if not result.accepted or not result.path:
                return False

            target_path = str(Path(result.path).expanduser())
            should_overwrite = False
            if Path(target_path).exists():
                if not confirm_overwrite(target_path, parent=self):
                    continue
                should_overwrite = True

            try:
                save_path = save_bounding_boxes(
                    target_path,
                    volume_shape=self._bbox_manager.volume_shape,
                    boxes=self._bbox_manager.boxes(),
                    overwrite=should_overwrite,
                )
            except FileExistsError:
                show_warning(
                    f"Refusing to overwrite existing path: {target_path}",
                    parent=self,
                )
                continue
            except Exception as exc:
                show_warning(str(exc), parent=self)
                return False

            box_count = len(self._bbox_manager.boxes())
            self._mark_bounding_boxes_clean()
            show_info(
                f"Saved {box_count} bounding box(es) to {save_path}",
                parent=self,
            )
            return True

    def _build_dataset_from_bboxes_with_dialog(self) -> bool:
        if not self.state.volume_loaded or self._raw_volume is None:
            show_warning(
                "Load a raw volume before building datasets from bounding boxes.",
                parent=self,
            )
            return False

        ordered_box_ids = tuple(row.box_id for row in self.bottom_panel.state.bbox_rows)
        if not ordered_box_ids:
            show_warning(
                "There are no bounding boxes to build datasets from.",
                parent=self,
            )
            return False
        boxes_by_id = {box.id: box for box in self._bbox_manager.boxes()}
        learning_box_ids = tuple(
            box_id
            for box_id in ordered_box_ids
            if box_id in boxes_by_id and str(boxes_by_id[box_id].label) != "inference"
        )
        train_box_ids = tuple(
            box_id
            for box_id in learning_box_ids
            if box_id in boxes_by_id and str(boxes_by_id[box_id].label) == "train"
        )
        if not train_box_ids:
            show_warning(
                (
                    "At least one bounding box labeled 'train' is required to build "
                    "datasets from bboxes."
                ),
                parent=self,
            )
            return False

        active_segmentation = self._active_segmentation_volume()
        if active_segmentation is None:
            show_warning(
                (
                    "Load a semantic segmentation map before building datasets "
                    "from bounding boxes."
                ),
                parent=self,
            )
            return False
        seg_kind, seg_volume = active_segmentation
        if seg_kind != "semantic":
            show_warning(
                (
                    "Only semantic segmentation is supported for Build Dataset "
                    "from Bbox."
                ),
                parent=self,
            )
            return False
        validation_box_ids = tuple(
            box_id
            for box_id in learning_box_ids
            if box_id in boxes_by_id and str(boxes_by_id[box_id].label) == "validation"
        )
        if not validation_box_ids:
            show_warning(
                (
                    "At least one bounding box labeled 'validation' is required to "
                    "build datasets from bboxes."
                ),
                parent=self,
            )
            return False

        try:
            raw_array = np.asarray(
                self._raw_volume.get_chunk((slice(None), slice(None), slice(None)))
            )
            segmentation_array = np.asarray(
                seg_volume.get_chunk((slice(None), slice(None), slice(None)))
            )
            outcome = extract_learning_bboxes_in_memory(
                raw_array,
                segmentation_array,
                boxes_by_id=boxes_by_id,
                ordered_box_ids=learning_box_ids,
                learning_batch_size=4,
                learning_num_workers=8,
                learning_pin_memory=True,
                learning_drop_last=True,
                build_eval_dataloaders=True,
                eval_batch_size=4,
                eval_num_workers=8,
                eval_pin_memory=True,
                eval_drop_last=False,
            )
            class_weights = compute_and_store_current_learning_class_weights(
                max_weight=100.0,
                device="cuda:0",
            )
            clear_current_learning_bbox_batch()
            residual_batch = get_current_learning_bbox_batch()
            residual_entry_count = int(residual_batch.size) if residual_batch is not None else 0
        except Exception as exc:
            clear_current_learning_bbox_batch()
            show_warning(str(exc), parent=self)
            return False

        if residual_entry_count > 0:
            show_warning(
                (
                    "Dataset build completed, but temporary learning tensors were "
                    f"not fully released ({residual_entry_count} entries remain in session)."
                ),
                parent=self,
            )
            return False

        summary_lines = [
            "Built bounding box learning datasets and buffers in memory.",
            (
                "- Temporary tensor entries built then released: "
                f"{outcome.tensor_entry_count}"
            ),
        ]
        if outcome.learning_train_box_ids:
            summary_lines.append(
                (
                    "- Learning DataLoader: "
                    f"{len(outcome.learning_train_box_ids)} train bboxes, "
                    f"batch_size={outcome.learning_batch_size}, "
                    f"num_workers={outcome.learning_num_workers}"
                )
            )
        if outcome.eval_validation_box_ids:
            summary_lines.append(
                (
                    "- Evaluation DataLoaders: "
                    f"{len(outcome.eval_validation_box_ids)} validation bboxes, "
                    f"batch_size={outcome.eval_batch_size}, "
                    f"num_workers={outcome.eval_num_workers}"
                )
            )
        if class_weights is not None:
            formatted_weights = _format_class_weights_for_summary(class_weights)
            if formatted_weights is None:
                summary_lines.append("- Loss class weights initialized on cuda:0.")
            else:
                summary_lines.append(
                    "- Loss class weights initialized on cuda:0: "
                    f"{formatted_weights}"
                )

        show_info(
            "\n".join(summary_lines),
            parent=self,
        )
        return True

    def _instantiate_foundation_model_with_dialog(self) -> bool:
        try:
            preconditions = validate_foundation_model_instantiation_preconditions(
                require_min_gpu_count=2,
            )
            existing_runtime = get_current_learning_model_runtime()
            if existing_runtime is not None:
                if not confirm_reinitialize_model(parent=self):
                    return False

            dialog_result = open_model_checkpoint_dialog(self)
            if not dialog_result.accepted or not dialog_result.path:
                return False
            checkpoint_path = str(Path(dialog_result.path).expanduser())
            if Path(checkpoint_path).suffix.lower() != ".cp":
                show_warning(
                    "Model checkpoints must use the .cp extension.",
                    parent=self,
                )
                return False

            runtime = instantiate_foundation_model_runtime(
                num_classes=preconditions.num_classes,
                device_ids=preconditions.device_ids,
                checkpoint_path=checkpoint_path,
            )
        except Exception as exc:
            message = str(exc)
            if (
                isinstance(exc, ValueError)
                and (
                    "No training dataloader runtime" in message
                    or "No evaluation runtimes/buffers" in message
                )
            ):
                message = (
                    f"{message}\n\n"
                    "Click 'Build Dataset from Bbox' first to initialize learning datasets."
                )
            show_warning(message, parent=self)
            return False

        show_info(
            (
                "Foundation model loaded from checkpoint.\n"
                f"- checkpoint: {runtime.checkpoint_path}\n"
                f"- num_classes: {runtime.num_classes}\n"
                f"- device_ids: {runtime.device_ids}"
            ),
            parent=self,
        )
        return True

    def _is_valid_segmentation_dtype(self, volume: VolumeData) -> bool:
        dtype = np.dtype(volume.info.dtype)
        return dtype.kind in ("u", "i") and dtype.itemsize in (1, 2, 4, 8)

    def _is_valid_instance_dtype(self, volume: VolumeData) -> bool:
        dtype = np.dtype(volume.info.dtype)
        return dtype.kind in ("u", "i") and dtype.itemsize in (1, 2, 4, 8)

    def _attach_segmentation_editor(self, editor: SegmentationEditor, *, kind: SegmentationKind) -> None:
        editable_volume = editor.to_volume_data(path=f"{editor.source_path}::editable")
        self._segmentation_editor = editor
        if kind == "semantic":
            self._instance_volume = None
            self._instance_worker = None
            self._semantic_volume = editable_volume
            self._semantic_worker = None
        else:
            self._semantic_volume = None
            self._semantic_worker = None
            self._instance_volume = editable_volume
            self._instance_worker = None
        self.renderer.attach_segmentation(
            editable_volume,
            levels=self._editable_segmentation_levels(editable_volume),
        )
        self._sync_level_mode_controls_from_renderer()
        self.renderer.set_segmentation_labels(editor.labels_in_use(include_background=True))
        self._annotation_labels_dirty = False
        self._refresh_hover_readout()
        self._refresh_annotation_ui_state()

    def _ensure_editable_segmentation_for_annotation(self) -> bool:
        if self._segmentation_editor is not None:
            return True
        if self._raw_volume is None:
            return False
        kind = self._annotation_kind
        editor = SegmentationEditor.create_empty(
            self._raw_volume.info.shape,
            kind=kind,
            voxel_spacing=self._raw_volume.info.voxel_spacing,
            axes=self._raw_volume.info.axes,
            source_path=f"{self._raw_volume.loader.path}::generated-{kind}",
        )
        self._attach_segmentation_editor(editor, kind=kind)
        self.bottom_panel.set_pyramid_levels(1, kind="Semantic" if kind == "semantic" else "Instance")
        self._refresh_annotation_ui_state()
        return True

    def _annotation_label_ui_max(self) -> int:
        editor = self._segmentation_editor
        if editor is None:
            return 2_147_483_647
        dtype_max = int(np.iinfo(editor.dtype).max)
        return max(0, min(dtype_max, 2_147_483_647))

    def _current_annotation_tool(self) -> AnnotationTool:
        return self.state.annotation_tool

    def _normalize_annotation_tool(self, tool: object) -> AnnotationTool:
        normalized = str(tool).strip().lower()
        if normalized not in ("brush", "eraser", "flood_filler"):
            normalized = "brush"
        return cast(AnnotationTool, normalized)

    def _set_annotation_tool_from_action(self, tool: object) -> None:
        normalized_tool = self._normalize_annotation_tool(tool)
        self.state.annotation_tool = normalized_tool
        self._refresh_annotation_ui_state()
        for view in self.views.values():
            view.refresh_overlay()

    def _apply_annotation_tool_shortcut(self, tool: AnnotationTool) -> bool:
        """Apply annotation-tool shortcut behavior with explicit precedence.

        Precedence rules (locked by tests):
        - Shortcut intent wins over focused widgets/editors.
        - If manual annotation is disabled, attempt to enable it first.
        - If enabling annotation fails, ignore the shortcut silently.
        """
        normalized = str(tool).strip().lower()
        if normalized not in ("brush", "eraser", "flood_filler"):
            return False
        if MainWindow._inference_navigation_lock_active(self):
            return False
        target_tool = cast(AnnotationTool, normalized)

        if not self.state.annotation_mode_enabled:
            if not self.set_annotation_mode(True):
                return False
        apply_tool = getattr(self, "_set_annotation_tool_from_action", None)
        if callable(apply_tool):
            apply_tool(target_tool)
        else:
            # Compatibility fallback for lightweight test doubles.
            self._handle_annotation_tool_changed(target_tool)
        return True

    def _bounding_box_mode_enabled(self) -> bool:
        return bool(self.state.bbox_mode_enabled)

    def _picker_marker_active(self) -> bool:
        if self.state.bbox_mode_enabled:
            return self.state.pending_bbox_corner is not None
        return self.state.annotation_mode_enabled and self.state.annotation_tool == "flood_filler"

    def _apply_picker_state_to_views(self) -> None:
        active = self._picker_marker_active()
        marker_indices = (
            self.state.pending_bbox_corner
            if self.state.bbox_mode_enabled
            else self.state.picked_indices
        )
        for view in self.views.values():
            view.set_picker_selection(marker_indices, active=active)

    def _clear_picker_selection(self) -> None:
        self.state.picked_indices = None
        self.state.picked_label = None
        self.state.pending_bbox_corner = None
        self.bottom_panel.set_picked_info(None, None)
        self._apply_picker_state_to_views()

    def _refresh_picked_readout(self) -> None:
        indices = self.state.picked_indices
        picked_label = self._label_for_indices(indices)
        self.state.picked_label = picked_label
        if (
            self.bottom_panel.state.picked_position != indices
            or self.bottom_panel.state.picked_label != picked_label
        ):
            self.bottom_panel.set_picked_info(indices, picked_label)

    def _begin_annotation_modification(self, source_view_id: ViewId) -> None:
        editor = self._segmentation_editor
        if editor is None:
            return
        if self._annotation_modification_active:
            if self._annotation_modification_view_id == source_view_id:
                return
            self._end_annotation_modification()
        self._annotation_modification_active = True
        self._annotation_modification_view_id = source_view_id
        editor.begin_modification("annotation_stroke")

    def _end_annotation_modification(self) -> None:
        editor = self._segmentation_editor
        if editor is not None and self._annotation_modification_active:
            operation = editor.commit_modification()
            self._record_global_history_for_segmentation_operation(operation)
        self._annotation_modification_active = False
        self._annotation_modification_view_id = None
        self._flush_deferred_readout_updates()

    def _record_global_history_for_segmentation_operation(
        self,
        operation: Optional[EditOperation],
    ) -> None:
        editor = self._segmentation_editor
        if editor is None or operation is None:
            return
        if operation.changed_voxels <= 0:
            return
        if editor.latest_undo_operation_id() != operation.operation_id:
            return
        bytes_used = estimate_segmentation_history_bytes(editor, operation)
        command = SegmentationHistoryCommand(
            editor=editor,
            operation_id=operation.operation_id,
            bytes_used=bytes_used,
        )
        self._push_global_history_command(command)

    def _push_global_history_command(self, command: HistoryCommand) -> None:
        self._global_history.push(command)
        self._refresh_undo_ui_state()

    def _stage_bounding_box_drag_update(
        self,
        *,
        box_id: str,
        before_box: BoundingBox,
        after_box: BoundingBox,
        before_selected_id: Optional[str],
        after_selected_id: Optional[str],
    ) -> None:
        existing = self._bbox_drag_staged_history_updates.get(box_id)
        if existing is None:
            self._bbox_drag_staged_history_updates[box_id] = _StagedBoundingBoxDragUpdate(
                before_box=before_box,
                after_box=after_box,
                before_selected_id=before_selected_id,
                after_selected_id=after_selected_id,
            )
            return
        self._bbox_drag_staged_history_updates[box_id] = _StagedBoundingBoxDragUpdate(
            before_box=existing.before_box,
            after_box=after_box,
            before_selected_id=existing.before_selected_id,
            after_selected_id=after_selected_id,
        )

    def _flush_staged_bounding_box_drag_updates(self) -> None:
        if not self._bbox_drag_staged_history_updates:
            return
        pending = tuple(self._bbox_drag_staged_history_updates.items())
        self._bbox_drag_staged_history_updates.clear()
        for _box_id, update in pending:
            if update.before_box == update.after_box:
                continue
            self._global_history.push(
                BoundingBoxUpdateCommand(
                    manager=self._bbox_manager,
                    before_box=update.before_box,
                    after_box=update.after_box,
                    before_selected_id=update.before_selected_id,
                    after_selected_id=update.after_selected_id,
                    bytes_used=estimate_bounding_box_history_bytes(
                        before_box=update.before_box,
                        after_box=update.after_box,
                    ),
                )
            )

    def _finalize_bbox_history_transaction(self) -> None:
        if not self._global_history.in_transaction():
            self._bbox_drag_staged_history_updates.clear()
            return
        self._flush_staged_bounding_box_drag_updates()
        self._global_history.commit_transaction()
        self._refresh_undo_ui_state()

    def _deactivate_annotation_mode_for_interaction_switch(self) -> None:
        """Disable manual painting without forcing expensive segmentation reattach.

        This path is used when switching to the bounding-box tool so the UI can
        respond immediately while keeping the current editable segmentation attached.
        """
        if not self.state.annotation_mode_enabled:
            return
        self.state.annotation_mode_enabled = False
        self._end_annotation_modification()
        if self._annotation_labels_dirty:
            self._sync_renderer_segmentation_labels()
        self._annotation_dirty_views.clear()
        self._pending_annotation_peer_view_ids.clear()
        self._annotation_peer_flush_scheduled = False

    def _refresh_undo_ui_state(self) -> None:
        interaction_enabled = bool(self.state.volume_loaded) and not MainWindow._inference_navigation_lock_active(
            self
        )
        undo_depth = self._global_history.undo_depth()
        redo_depth = self._global_history.redo_depth()
        self.bottom_panel.set_undo_state(
            depth=undo_depth,
            enabled=interaction_enabled,
        )
        self.bottom_panel.set_redo_state(
            depth=redo_depth,
            enabled=interaction_enabled,
        )

    def _training_is_running(self) -> bool:
        return bool(self._training_running)

    def _inference_is_running(self) -> bool:
        return bool(self._inference_running)

    @staticmethod
    def _inference_navigation_lock_active(target: object) -> bool:
        state_getter = getattr(target, "_inference_is_running", None)
        if callable(state_getter):
            try:
                return bool(state_getter())
            except Exception:
                pass
        return bool(getattr(target, "_inference_running", False))

    @staticmethod
    def _inference_stop_already_requested(target: object) -> bool:
        return bool(getattr(target, "_inference_stop_requested", False))

    def _clear_learning_inference_stop_request_state(self) -> None:
        self._inference_stop_requested = False

    def _set_running_training_worker_completion_checkpoint_path(
        self,
        *,
        checkpoint_path: Optional[str],
    ) -> None:
        worker = getattr(self, "_training_worker", None)
        if worker is None:
            return
        if checkpoint_path is None:
            clear_request = getattr(worker, "clear_completion_checkpoint_save_request", None)
            if callable(clear_request):
                clear_request()
            return
        request_save = getattr(worker, "request_completion_checkpoint_save", None)
        if callable(request_save):
            request_save(checkpoint_path)

    def _clear_deferred_close_training_state(self) -> None:
        self._deferred_close_after_training = False
        self._deferred_close_training_mode = "none"
        self._deferred_close_checkpoint_path = None
        sync_method = getattr(
            self,
            "_set_running_training_worker_completion_checkpoint_path",
            None,
        )
        if callable(sync_method):
            sync_method(checkpoint_path=None)
            return
        MainWindow._set_running_training_worker_completion_checkpoint_path(
            self,
            checkpoint_path=None,
        )

    def _set_deferred_close_after_stop_training(self) -> None:
        self._deferred_close_after_training = True
        self._deferred_close_training_mode = "stop_and_close"
        self._deferred_close_checkpoint_path = None
        sync_method = getattr(
            self,
            "_set_running_training_worker_completion_checkpoint_path",
            None,
        )
        if callable(sync_method):
            sync_method(checkpoint_path=None)
            return
        MainWindow._set_running_training_worker_completion_checkpoint_path(
            self,
            checkpoint_path=None,
        )

    def _set_deferred_close_with_background_training(
        self,
        *,
        checkpoint_path: str,
    ) -> None:
        normalized_path = str(checkpoint_path).strip()
        if not normalized_path:
            raise ValueError("checkpoint_path must be a non-empty string")
        sync_method = getattr(
            self,
            "_set_running_training_worker_completion_checkpoint_path",
            None,
        )
        if callable(sync_method):
            sync_method(checkpoint_path=normalized_path)
        else:
            MainWindow._set_running_training_worker_completion_checkpoint_path(
                self,
                checkpoint_path=normalized_path,
            )
        self._deferred_close_after_training = True
        self._deferred_close_training_mode = "continue_in_background"
        self._deferred_close_checkpoint_path = normalized_path

    def _refresh_learning_training_ui_state(self) -> None:
        training_running = self._training_is_running()
        self.bottom_panel.set_learning_training_running(training_running)
        refresh_inference = getattr(self, "_refresh_learning_inference_ui_state", None)
        if callable(refresh_inference):
            refresh_inference()
        else:
            MainWindow._refresh_learning_inference_ui_state(self)
        self.bottom_panel.set_stop_training_enabled(training_running)

    def _refresh_learning_inference_ui_state(self) -> None:
        inference_running = MainWindow._inference_navigation_lock_active(self)
        learning_actions_enabled = not self._training_is_running() and not inference_running
        self.bottom_panel.set_segment_inference_enabled(learning_actions_enabled)
        self.bottom_panel.set_train_model_enabled(learning_actions_enabled)
        set_stop_inference_enabled = getattr(self.bottom_panel, "set_stop_inference_enabled", None)
        if callable(set_stop_inference_enabled):
            stop_enabled = bool(
                inference_running and not MainWindow._inference_stop_already_requested(self)
            )
            set_stop_inference_enabled(stop_enabled)
        set_navigation_only_mode = getattr(
            self.bottom_panel,
            "set_inference_navigation_only_mode",
            None,
        )
        if callable(set_navigation_only_mode):
            set_navigation_only_mode(inference_running)
        refresh_undo = getattr(self, "_refresh_undo_ui_state", None)
        if callable(refresh_undo):
            refresh_undo()

    def _enter_learning_training_running_state(
        self,
        *,
        worker: object,
        thread: object,
    ) -> None:
        self._training_running = True
        self._training_worker = worker
        self._training_thread = thread
        self._refresh_learning_training_ui_state()

    def _exit_learning_training_running_state(self) -> None:
        self._training_running = False
        self._training_worker = None
        self._training_thread = None
        self._refresh_learning_training_ui_state()

    def _enter_learning_inference_running_state(
        self,
        *,
        worker: object,
        thread: object,
    ) -> None:
        self._inference_running = True
        self._inference_stop_requested = False
        self._inference_worker = worker
        self._inference_thread = thread
        self._refresh_learning_inference_ui_state()

    def _exit_learning_inference_running_state(self) -> None:
        self._inference_running = False
        self._inference_worker = None
        self._inference_thread = None
        self._refresh_learning_inference_ui_state()

    def _refresh_annotation_ui_state(self) -> None:
        self.bottom_panel.set_interaction_tools_enabled(self.state.volume_loaded)
        self.bottom_panel.set_annotation_mode(self.state.annotation_mode_enabled)
        self.bottom_panel.set_bounding_box_mode(self.state.bbox_mode_enabled)
        self.bottom_panel.set_annotation_tool(self.state.annotation_tool)
        self.bottom_panel.set_brush_radius(self.state.brush_radius)
        editor = self._segmentation_editor
        if editor is not None and self.state.eraser_target_label is not None:
            max_label = int(np.iinfo(editor.dtype).max)
            if self.state.eraser_target_label > max_label:
                self.state.eraser_target_label = None
        eraser_target = (
            ""
            if self.state.eraser_target_label is None
            else str(self.state.eraser_target_label)
        )
        self.bottom_panel.set_eraser_target(eraser_target)
        if editor is None:
            self.bottom_panel.set_annotation_controls_enabled(False)
            self.bottom_panel.set_active_label_bounds(0, 2_147_483_647)
            self.bottom_panel.set_active_label(1)
            self.bottom_panel.set_flood_fill_target_bounds(0, 2_147_483_647)
            self.bottom_panel.set_flood_fill_target(self.state.flood_fill_target_label)
            self._refresh_picked_readout()
            self._apply_picker_state_to_views()
            self._refresh_undo_ui_state()
            return

        self.bottom_panel.set_annotation_controls_enabled(True)
        self.bottom_panel.set_active_label_bounds(0, self._annotation_label_ui_max())
        self.bottom_panel.set_active_label(editor.active_label)
        fill_max = self._annotation_label_ui_max()
        self.state.flood_fill_target_label = max(0, min(int(self.state.flood_fill_target_label), fill_max))
        self.bottom_panel.set_flood_fill_target_bounds(0, fill_max)
        self.bottom_panel.set_flood_fill_target(self.state.flood_fill_target_label)
        self._refresh_picked_readout()
        self._apply_picker_state_to_views()
        self._refresh_undo_ui_state()

    def _sync_renderer_segmentation_labels(self) -> None:
        editor = self._segmentation_editor
        if editor is None:
            self._annotation_labels_dirty = False
            return
        self.renderer.set_segmentation_labels(editor.labels_in_use(include_background=True))
        self._annotation_labels_dirty = False

    def _request_hover_readout(self) -> None:
        if self._annotation_modification_active:
            self._deferred_hover_readout = True
            return
        self._refresh_hover_readout()

    def _request_picked_readout(self) -> None:
        if self._annotation_modification_active:
            self._deferred_picked_readout = True
            return
        self._refresh_picked_readout()

    def _flush_deferred_readout_updates(self) -> None:
        if self._deferred_hover_readout:
            self._deferred_hover_readout = False
            self._refresh_hover_readout()
        if self._deferred_picked_readout:
            self._deferred_picked_readout = False
            self._refresh_picked_readout()

    def render_all(self) -> None:
        for view in self.views.values():
            view.render()
        self._update_active_levels_status()

    def _on_sync_state_changed(self) -> None:
        state = self.sync_manager.state
        cursor_changed = self.bottom_panel.state.cursor_position != state.slice_indices
        for view in self.views.values():
            axis = view.state.axis
            next_slice = state.slice_indices[axis]
            slice_changed = view.state.slice_index != next_slice
            zoom_changed = view.state.zoom != state.zoom

            view.set_slice_index(next_slice)
            if zoom_changed:
                view.set_zoom(state.zoom)
            if view.state.pan != state.pan:
                view.set_pan(state.pan)

            level_changed = False
            if zoom_changed:
                latest_result = self.renderer.latest_result(view.view_id)
                if latest_result is None:
                    level_changed = True
                else:
                    level_changed = latest_result.level != self.renderer.target_level_for_view(axis, state.zoom)

            if slice_changed or view.latest_image() is None or level_changed:
                self._queue_render(view.view_id)
            elif cursor_changed:
                view.refresh_overlay()

        if self.bottom_panel.state.cursor_position != state.slice_indices:
            self.bottom_panel.set_cursor_position(state.slice_indices)
        if self.bottom_panel.state.zoom != state.zoom:
            self.bottom_panel.set_zoom(state.zoom)
        self._request_hover_readout()
        self._request_picked_readout()

    def _label_for_indices(
        self,
        indices: Optional[Tuple[int, int, int]],
    ) -> Optional[int]:
        if indices is None:
            return None
        editor = self._segmentation_editor
        if editor is None:
            return None
        z, y, x = indices
        shape = editor.shape
        if not (
            0 <= z < shape[0]
            and 0 <= y < shape[1]
            and 0 <= x < shape[2]
        ):
            return None
        return int(editor.array_view()[z, y, x])

    def _hover_label_for_indices(
        self,
        indices: Optional[Tuple[int, int, int]],
    ) -> Optional[int]:
        return self._label_for_indices(indices)

    def _refresh_hover_readout(self) -> None:
        hover_indices = self.sync_manager.state.hover_indices
        hover_label = self._hover_label_for_indices(hover_indices)
        if (
            self.bottom_panel.state.hover_position != hover_indices
            or self.bottom_panel.state.hover_label != hover_label
        ):
            self.bottom_panel.set_hover_info(hover_indices, hover_label)

    def _sync_contrast_controls_from_renderer(self) -> None:
        self.bottom_panel.set_contrast_range(self.renderer.get_data_range())
        self.bottom_panel.set_contrast_window(self.renderer.get_window_range())

    def _sync_level_mode_controls_from_renderer(self) -> None:
        level_count = max(0, int(self.renderer.available_level_count()))
        max_level = max(0, level_count - 1)
        self.bottom_panel.set_level_mode(
            auto_enabled=self.renderer.is_auto_level_enabled(),
            manual_level=self.renderer.manual_level(),
            max_level=max_level,
        )

    def _handle_contrast_window_changed(self, vmin: float, vmax: float) -> None:
        if not self.state.volume_loaded:
            return
        try:
            self.renderer.set_window(float(vmin), float(vmax))
        except Exception as exc:
            self._sync_contrast_controls_from_renderer()
            show_warning(str(exc), parent=self)
            return
        self._queue_contrast_rerender()

    def _handle_auto_level_mode_changed(self, enabled: bool) -> None:
        if not self.state.volume_loaded:
            return
        auto_enabled = bool(enabled)
        self.renderer.set_auto_level_enabled(auto_enabled)
        if auto_enabled:
            # Recompute target levels from current zoom immediately when
            # returning to automatic level selection.
            self.render_all()
            return
        self._queue_contrast_rerender()

    def _handle_manual_level_requested(self, level: int) -> None:
        if not self.state.volume_loaded:
            return
        self.renderer.set_manual_level(int(level))
        self._queue_contrast_rerender()

    def _queue_contrast_rerender(self) -> None:
        if not self.views:
            self.render_all()
            return
        for view_id in self.views:
            self._queue_render(view_id)

    def _queue_render(self, view_id: ViewId) -> None:
        self._pending_render_view_ids.add(view_id)
        if self._render_flush_scheduled:
            return
        self._render_flush_scheduled = True
        QTimer.singleShot(0, self._flush_pending_renders)

    def _flush_pending_renders(self) -> None:
        self._render_flush_scheduled = False
        if not self._pending_render_view_ids:
            return

        pending = set(self._pending_render_view_ids)
        self._pending_render_view_ids.clear()
        for view_id, view in self.views.items():
            if view_id in pending:
                view.render()
        self._update_active_levels_status()

        if self._pending_render_view_ids and not self._render_flush_scheduled:
            self._render_flush_scheduled = True
            QTimer.singleShot(0, self._flush_pending_renders)

    def _update_active_levels_status(self) -> None:
        axial_result = self.renderer.latest_result("axial")
        coronal_result = self.renderer.latest_result("coronal")
        sagittal_result = self.renderer.latest_result("sagittal")
        axial = (axial_result.level, axial_result.level_scale) if axial_result is not None else (0, 1)
        coronal = (coronal_result.level, coronal_result.level_scale) if coronal_result is not None else (0, 1)
        sagittal = (sagittal_result.level, sagittal_result.level_scale) if sagittal_result is not None else (0, 1)
        self.bottom_panel.set_active_levels(
            axial=axial,
            coronal=coronal,
            sagittal=sagittal,
            forced=not self.renderer.is_auto_level_enabled(),
        )

    def _handle_open_request(self) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            return
        if not self._maybe_resolve_unsaved_segmentation(context="opening a new raw volume"):
            return
        if not self._maybe_resolve_unsaved_bounding_boxes(context="opening a new raw volume"):
            return
        result = open_file_dialog(self)
        if not result.accepted or not result.path:
            return
        try:
            from ..workers import IOWorker

            prepared = load_prepared_volume(
                result.path,
                kind="raw",
                load_mode=self._load_mode,
                cache_max_bytes=self._cache_max_bytes,
                pyramid_levels=4,
            )
            if self.set_volume(prepared.volume, levels=prepared.levels):
                self._io_worker = IOWorker(volume=prepared.volume, cache=prepared.cache)
                self.render_all()
        except Exception as exc:
            show_warning(str(exc), parent=self)

    def _handle_open_semantic_request(self) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            return
        if not self._maybe_resolve_unsaved_segmentation(context="opening a new segmentation map"):
            return
        result = open_file_dialog(self)
        if not result.accepted or not result.path:
            return
        try:
            prepared = load_prepared_volume(
                result.path,
                kind="semantic",
                load_mode=self._load_mode,
                cache_max_bytes=self._cache_max_bytes,
                pyramid_levels=4,
            )
            if self.set_semantic_volume(prepared.volume, levels=prepared.levels):
                from ..workers import IOWorker

                if self._semantic_volume is not None:
                    self._semantic_worker = IOWorker(volume=self._semantic_volume, cache=self._semantic_volume.cache)
                self.render_all()
        except Exception as exc:
            show_warning(str(exc), parent=self)

    def _handle_open_instance_request(self) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            return
        if not self._maybe_resolve_unsaved_segmentation(context="opening a new segmentation map"):
            return
        result = open_file_dialog(self)
        if not result.accepted or not result.path:
            return
        try:
            prepared = load_prepared_volume(
                result.path,
                kind="instance",
                load_mode=self._load_mode,
                cache_max_bytes=self._cache_max_bytes,
                pyramid_levels=4,
            )
            if self.set_instance_volume(prepared.volume, levels=prepared.levels):
                from ..workers import IOWorker

                if self._instance_volume is not None:
                    self._instance_worker = IOWorker(volume=self._instance_volume, cache=self._instance_volume.cache)
                self.render_all()
        except Exception as exc:
            show_warning(str(exc), parent=self)

    def _active_segmentation_volume(self) -> Optional[Tuple[str, VolumeData]]:
        synced = self._sync_segmentation_volume_from_editor(reattach_renderer=False)
        if synced is not None:
            return synced
        if self._instance_volume is not None:
            return ("instance", self._instance_volume)
        if self._semantic_volume is not None:
            return ("semantic", self._semantic_volume)
        return None

    def _handle_save_segmentation_request(self) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            return
        self._save_active_segmentation_with_dialog()

    def _handle_save_shortcut_requested(self) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            return
        self._save_active_segmentation_with_dialog()

    def _handle_annotation_mode_changed(self, enabled: bool) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            self._refresh_annotation_ui_state()
            return
        if enabled and self.state.bbox_mode_enabled:
            self._finalize_bbox_history_transaction()
            self.state.bbox_mode_enabled = False
            self.state.pending_bbox_corner = None
        if enabled and self._raw_volume is None:
            show_warning("Load a raw volume before enabling annotation mode.", parent=self)
            self.state.annotation_mode_enabled = False
            self.state.bbox_mode_enabled = False
            self._refresh_annotation_ui_state()
            for view in self.views.values():
                view.refresh_overlay()
            return
        success = self.set_annotation_mode(enabled)
        if enabled and not success:
            show_warning("Could not initialize an editable segmentation volume.", parent=self)
            self.state.annotation_mode_enabled = False
            self._refresh_annotation_ui_state()
            for view in self.views.values():
                view.refresh_overlay()
            return
        if enabled and self._segmentation_editor is not None:
            self.render_all()
        else:
            for view in self.views.values():
                view.refresh_overlay()

    def _handle_bounding_box_mode_changed(self, enabled: bool) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            self._refresh_annotation_ui_state()
            return
        if enabled and self._raw_volume is None:
            show_warning("Load a raw volume before enabling the bounding box tool.", parent=self)
            self.state.bbox_mode_enabled = False
            self._refresh_annotation_ui_state()
            for view in self.views.values():
                view.refresh_overlay()
            return

        if enabled and self.state.annotation_mode_enabled:
            self._deactivate_annotation_mode_for_interaction_switch()

        self.state.bbox_mode_enabled = bool(enabled)
        if not self.state.bbox_mode_enabled:
            self.state.pending_bbox_corner = None
            self._finalize_bbox_history_transaction()
        self._refresh_annotation_ui_state()
        for view in self.views.values():
            view.refresh_overlay()

    def _handle_active_label_changed(self, value: int) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            self._refresh_annotation_ui_state()
            return
        editor = self._segmentation_editor
        if editor is None:
            self._refresh_annotation_ui_state()
            return
        try:
            editor.set_active_label(int(value))
        except ValueError as exc:
            show_warning(str(exc), parent=self)
        self._refresh_annotation_ui_state()

    def _handle_next_available_label_requested(self) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            self._refresh_annotation_ui_state()
            return
        editor = self._segmentation_editor
        if editor is None:
            self._refresh_annotation_ui_state()
            return
        try:
            next_label = editor.next_available_label()
            editor.set_active_label(next_label)
        except ValueError as exc:
            show_warning(str(exc), parent=self)
        self._refresh_annotation_ui_state()

    def _handle_brush_radius_changed(self, brush_radius: BrushRadius) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            return
        value = int(brush_radius)
        if value < 0:
            value = 0
        elif value > 9:
            value = 9
        self.state.brush_radius = value

    def _handle_annotation_tool_changed(self, tool: AnnotationTool) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            self._refresh_annotation_ui_state()
            return
        self._set_annotation_tool_from_action(tool)

    def _handle_eraser_target_changed(self, value: str) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            self._refresh_annotation_ui_state()
            return
        text = str(value).strip()
        if text == "":
            self.state.eraser_target_label = None
            self._refresh_annotation_ui_state()
            return
        try:
            parsed = int(text)
        except ValueError:
            show_warning("Eraser ID must be a non-negative integer.", parent=self)
            self._refresh_annotation_ui_state()
            return
        if parsed < 0:
            show_warning("Eraser ID must be a non-negative integer.", parent=self)
            self._refresh_annotation_ui_state()
            return
        editor = self._segmentation_editor
        if editor is not None:
            max_label = int(np.iinfo(editor.dtype).max)
            if parsed > max_label:
                show_warning(
                    f"Eraser ID {parsed} exceeds max value {max_label} for dtype {editor.dtype}.",
                    parent=self,
                )
                self._refresh_annotation_ui_state()
                return
        self.state.eraser_target_label = parsed
        self._refresh_annotation_ui_state()

    def _handle_pick_voxel(
        self,
        _source_view_id: ViewId,
        indices: Tuple[int, int, int],
    ) -> None:
        if self.state.bbox_mode_enabled:
            self._handle_bounding_box_pick(indices)
            return
        if not self._picker_marker_active():
            return
        self.state.picked_indices = (int(indices[0]), int(indices[1]), int(indices[2]))
        self._refresh_picked_readout()
        self._apply_picker_state_to_views()

    def _handle_bounding_box_pick(self, indices: Tuple[int, int, int]) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            return
        if not self.state.volume_loaded:
            return
        corner = self.state.pending_bbox_corner
        picked = (int(indices[0]), int(indices[1]), int(indices[2]))
        if corner is None:
            self.state.pending_bbox_corner = picked
            self._apply_picker_state_to_views()
            return
        before_selected_id = self._bbox_manager.selected_id
        try:
            box = self._bbox_manager.add_from_corners(corner, picked, select=True)
        except Exception as exc:
            show_warning(str(exc), parent=self)
            return
        after_selected_id = self._bbox_manager.selected_id
        self._push_global_history_command(
            BoundingBoxAddCommand(
                manager=self._bbox_manager,
                box=box,
                before_selected_id=before_selected_id,
                after_selected_id=after_selected_id,
                bytes_used=estimate_bounding_box_history_bytes(after_box=box),
            )
        )
        self.state.pending_bbox_corner = None
        self._apply_picker_state_to_views()

    def _handle_flood_fill_target_changed(self, value: int) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            self._refresh_annotation_ui_state()
            return
        self.state.flood_fill_target_label = int(value)

    def _handle_flood_fill_requested(self, target_label: int) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            return
        if not self.state.annotation_mode_enabled or self.state.annotation_tool != "flood_filler":
            return
        editor = self._segmentation_editor
        if editor is None:
            show_warning("No editable segmentation is available for flood fill.", parent=self)
            return
        seed = self.state.picked_indices
        if seed is None:
            show_warning("Pick a voxel before running flood fill.", parent=self)
            return
        editor.begin_modification("flood_fill")
        try:
            target = int(target_label)
            self.state.flood_fill_target_label = target
            operation = editor.flood_fill_from_seed(
                seed,
                label=target,
                max_duration_seconds=self._flood_fill_timeout_seconds,
            )
        except ValueError as exc:
            editor.cancel_modification()
            show_warning(str(exc), parent=self)
            self._refresh_annotation_ui_state()
            return
        committed_operation = editor.commit_modification()
        self._record_global_history_for_segmentation_operation(committed_operation)

        if operation.changed_voxels <= 0:
            self._refresh_annotation_ui_state()
            return

        self.renderer.set_segmentation_labels(editor.labels_in_use(include_background=True))
        self._annotation_labels_dirty = False
        self._refresh_hover_readout()
        self._refresh_picked_readout()
        self.render_all()
        self._refresh_annotation_ui_state()

    def _handle_undo_requested(self) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            return
        self._finalize_bbox_history_transaction()
        self._end_annotation_modification()
        try:
            command = self._global_history.undo()
        except Exception as exc:
            show_warning(str(exc), parent=self)
            self._refresh_annotation_ui_state()
            return
        if command is None:
            self._refresh_annotation_ui_state()
            return
        self._after_global_history_navigation(command)
        self._refresh_annotation_ui_state()

    def _handle_redo_requested(self) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            return
        self._finalize_bbox_history_transaction()
        self._end_annotation_modification()
        try:
            command = self._global_history.redo()
        except Exception as exc:
            show_warning(str(exc), parent=self)
            self._refresh_annotation_ui_state()
            return
        if command is None:
            self._refresh_annotation_ui_state()
            return
        self._after_global_history_navigation(command)
        self._refresh_annotation_ui_state()

    def _after_global_history_navigation(self, command: object) -> None:
        if isinstance(command, SegmentationHistoryCommand):
            editor = self._segmentation_editor
            if editor is not None:
                self.renderer.set_segmentation_labels(editor.labels_in_use(include_background=True))
                self._annotation_labels_dirty = False
                self._refresh_hover_readout()
                self._refresh_picked_readout()
                self.render_all()
            return

        # Bounding-box command: keep overlays responsive and refresh readouts.
        for view in self.views.values():
            view.refresh_overlay()
        self._refresh_hover_readout()
        self._refresh_picked_readout()

    def _handle_paint_voxel(
        self,
        source_view_id: ViewId,
        indices: Tuple[int, int, int],
    ) -> AnnotationPaintOutcome:
        if MainWindow._inference_navigation_lock_active(self):
            return AnnotationPaintOutcome(accepted=False)
        editor = self._segmentation_editor
        if self.state.annotation_tool == "flood_filler":
            return AnnotationPaintOutcome(accepted=False)
        if not self.state.annotation_mode_enabled or editor is None:
            return AnnotationPaintOutcome(accepted=False)
        source_view = self.views.get(source_view_id)
        if source_view is None:
            return AnnotationPaintOutcome(accepted=False)
        self._begin_annotation_modification(source_view_id)
        try:
            if self.state.annotation_tool == "eraser":
                operation = editor.erase_brush_voxel(
                    indices,
                    axis=source_view.state.axis,
                    brush_radius=self.state.brush_radius,
                    target_label=self.state.eraser_target_label,
                )
            else:
                operation = editor.paint_brush_voxel(
                    indices,
                    axis=source_view.state.axis,
                    brush_radius=self.state.brush_radius,
                )
        except ValueError as exc:
            show_warning(str(exc), parent=self)
            return AnnotationPaintOutcome(accepted=False)
        if operation.changed_voxels > 0:
            self._annotation_labels_dirty = True
            self._annotation_dirty_views.add(source_view_id)
            self._request_hover_readout()
            self._request_picked_readout()
        return AnnotationPaintOutcome(
            accepted=True,
            changed_bounds=operation.bounds if operation.changed_voxels > 0 else None,
        )

    def _handle_paint_stroke(
        self,
        source_view_id: ViewId,
        start: Tuple[int, int, int],
        end: Tuple[int, int, int],
    ) -> AnnotationPaintOutcome:
        if MainWindow._inference_navigation_lock_active(self):
            return AnnotationPaintOutcome(accepted=False)
        editor = self._segmentation_editor
        if self.state.annotation_tool == "flood_filler":
            return AnnotationPaintOutcome(accepted=False)
        if not self.state.annotation_mode_enabled or editor is None:
            return AnnotationPaintOutcome(accepted=False)
        source_view = self.views.get(source_view_id)
        if source_view is None:
            return AnnotationPaintOutcome(accepted=False)
        self._begin_annotation_modification(source_view_id)
        try:
            if self.state.annotation_tool == "eraser":
                operation = editor.erase_brush_stroke(
                    (start, end),
                    axis=source_view.state.axis,
                    brush_radius=self.state.brush_radius,
                    target_label=self.state.eraser_target_label,
                )
            else:
                operation = editor.paint_brush_stroke(
                    (start, end),
                    axis=source_view.state.axis,
                    brush_radius=self.state.brush_radius,
                )
        except ValueError as exc:
            show_warning(str(exc), parent=self)
            return AnnotationPaintOutcome(accepted=False)
        if operation.changed_voxels > 0:
            self._annotation_labels_dirty = True
            self._annotation_dirty_views.add(source_view_id)
            self._request_hover_readout()
            self._request_picked_readout()
        return AnnotationPaintOutcome(
            accepted=True,
            changed_bounds=operation.bounds if operation.changed_voxels > 0 else None,
        )

    def _handle_annotation_finished(self, source_view_id: ViewId) -> None:
        if self._annotation_modification_view_id == source_view_id:
            self._end_annotation_modification()
            if self._annotation_labels_dirty:
                self._sync_renderer_segmentation_labels()
            self._refresh_annotation_ui_state()
        if source_view_id not in self._annotation_dirty_views:
            return
        self._annotation_dirty_views.discard(source_view_id)
        self._queue_annotation_peer_renders(source_view_id=source_view_id)

    def _queue_annotation_peer_renders(self, *, source_view_id: ViewId) -> None:
        for view_id in self.views:
            if view_id != source_view_id:
                self._pending_annotation_peer_view_ids.add(view_id)
        if self._annotation_peer_flush_scheduled:
            return
        self._annotation_peer_flush_scheduled = True
        QTimer.singleShot(
            self._annotation_peer_redraw_interval_ms,
            self._flush_annotation_peer_renders,
        )

    def _flush_annotation_peer_renders(self) -> None:
        self._annotation_peer_flush_scheduled = False
        if not self._pending_annotation_peer_view_ids:
            return
        pending = set(self._pending_annotation_peer_view_ids)
        self._pending_annotation_peer_view_ids.clear()
        for view_id in pending:
            self._queue_render(view_id)

    def _sync_segmentation_volume_from_editor(
        self,
        *,
        reattach_renderer: bool,
    ) -> Optional[Tuple[str, VolumeData]]:
        editor = self._segmentation_editor
        if editor is None:
            return None

        if editor.kind == "semantic":
            path = (
                self._semantic_volume.loader.path
                if self._semantic_volume is not None
                else f"{editor.source_path}::editable"
            )
            editable_volume = editor.to_volume_data(path=path)
            self._semantic_volume = editable_volume
            self._semantic_worker = None
            self._instance_volume = None
            self._instance_worker = None
            kind = "semantic"
        else:
            path = (
                self._instance_volume.loader.path
                if self._instance_volume is not None
                else f"{editor.source_path}::editable"
            )
            editable_volume = editor.to_volume_data(path=path)
            self._instance_volume = editable_volume
            self._instance_worker = None
            self._semantic_volume = None
            self._semantic_worker = None
            kind = "instance"

        if reattach_renderer:
            self.renderer.attach_segmentation(
                editable_volume,
                levels=self._editable_segmentation_levels(editable_volume),
            )
            self._sync_level_mode_controls_from_renderer()
            self.renderer.set_segmentation_labels(editor.labels_in_use(include_background=True))
            self._annotation_labels_dirty = False
        return (kind, editable_volume)

    def _editable_segmentation_levels(self, volume: VolumeData) -> Tuple[VolumeData, ...]:
        try:
            return build_segmentation_pyramid_lazy(volume, levels=4)
        except Exception:
            return (volume,)

    def _has_unsaved_segmentation_changes(self) -> bool:
        editor = self._segmentation_editor
        if editor is None:
            return False
        return editor.dirty

    def _mark_segmentation_clean(self) -> None:
        editor = self._segmentation_editor
        if editor is not None:
            editor.mark_clean()

    def _save_active_segmentation_with_dialog(self) -> bool:
        self._end_annotation_modification()
        if self._annotation_labels_dirty:
            self._sync_renderer_segmentation_labels()
        active = self._active_segmentation_volume()
        if active is None:
            show_warning("No semantic or instance segmentation map is loaded.", parent=self)
            return False
        kind, volume = active

        while True:
            result = open_save_segmentation_dialog(self)
            if not result.accepted or not result.path or not result.format:
                return False

            target_path = str(Path(result.path).expanduser())
            should_overwrite = False
            if Path(target_path).exists():
                if not confirm_overwrite(target_path, parent=self):
                    continue
                should_overwrite = True

            try:
                save_path = save_segmentation_volume(
                    volume,
                    target_path,
                    save_format=result.format,
                    overwrite=should_overwrite,
                )
            except FileExistsError:
                # Best-effort guard for races or callers that bypassed confirmation.
                show_warning(f"Refusing to overwrite existing path: {target_path}", parent=self)
                continue
            except Exception as exc:
                show_warning(str(exc), parent=self)
                return False

            self._mark_segmentation_clean()
            self._refresh_annotation_ui_state()
            show_info(f"Saved {kind} segmentation to {save_path}", parent=self)
            return True

    def _maybe_resolve_unsaved_segmentation(self, *, context: str) -> bool:
        if not self._has_unsaved_segmentation_changes():
            return True

        decision = ask_unsaved_changes(self, context=context, subject="segmentation")
        if decision == UnsavedChangesDecision.DISCARD:
            return True
        if decision == UnsavedChangesDecision.SAVE:
            return self._save_active_segmentation_with_dialog()
        return False

    def _has_unsaved_bounding_box_changes(self) -> bool:
        return bool(self._bbox_manager.dirty)

    def _maybe_resolve_unsaved_bounding_boxes(self, *, context: str) -> bool:
        if not self._has_unsaved_bounding_box_changes():
            return True

        decision = ask_unsaved_changes(self, context=context, subject="bounding boxes")
        if decision == UnsavedChangesDecision.DISCARD:
            return True
        if decision == UnsavedChangesDecision.SAVE:
            return self._save_bounding_boxes_with_dialog()
        return False
