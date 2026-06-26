from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from numbers import Integral
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, Literal, Mapping, Optional, Sequence, Set, Tuple, cast

import numpy as np

from PySide6.QtCore import QEvent, QThread, Qt, QTimer
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
from ..annotation import bbox_segmentation_ops as bbox_ops
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
from ..headless.job_spec import HeadlessJobSpec, save_headless_job_spec
from ..io import extract_learning_bboxes_in_memory
from ..io.saver import save_segmentation_volume
from ..loading import load_prepared_volume
from ..learning import (
    DEFAULT_FOUNDATION_CHECKPOINT_PATH,
    DEFAULT_FOUNDATION_MODEL_CONFIG,
    DEFAULT_TRAINING_PARAMETERS,
    LearningSession,
    TrainingParameters,
    clear_current_learning_bbox_batch,
    clear_current_learning_dataloader_runtime,
    clear_current_learning_eval_runtimes_by_box_id,
    clear_current_learning_label_space,
    clear_current_learning_model_runtime,
    compute_and_store_current_learning_class_weights,
    get_current_learning_dataloader_runtime,
    get_current_learning_eval_runtimes_by_box_id,
    get_current_learning_label_space,
    get_current_learning_model_runtime,
    get_current_learning_bbox_batch,
    instantiate_foundation_model_runtime,
    prepare_learning_state_from_volumes,
    save_foundation_model_checkpoint,
    set_current_learning_label_space,
    validate_foundation_checkpoint_load_preconditions,
    validate_foundation_model_instantiation_preconditions,
    validate_learning_model_training_preconditions,
    validate_training_parameters,
)
from ..learning.qt_workers import (
    LearningInferencePrediction,
    LearningInferenceWorker,
    LearningTrainingWorker,
)
from ..render import Renderer, ViewId
from ..utils import exception_message, get_logger
from .bottom_panel import BottomPanel
from .controllers import (
    InferenceController,
    InferenceControllerOperations,
    LearningStateController,
    LearningStateControllerOperations,
    ModelController,
    ModelControllerOperations,
    TrainingController,
    TrainingControllerOperations,
)
from .dialogs import (
    InferenceCloseDecision,
    TrainingCloseDecision,
    UnsavedChangesDecision,
    ask_inference_running_close_decision,
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
    open_training_parameters_dialog,
    show_info,
    show_warning,
)
from .orthogonal_view import AnnotationPaintOutcome, OrthogonalView


AnnotationTool = Literal["brush", "eraser", "flood_filler"]
BBoxSegmentationOperation = Literal["median_filter", "erosion", "dilation"]
LearningStateAction = Literal["load_model", "train", "inference"]
DeferredTrainingCloseMode = Literal[
    "none",
    "stop_and_close",
]
DeferredInferenceCloseMode = Literal[
    "none",
    "stop_and_close",
]
ViewLayoutMode = Literal["all", "axial", "coronal", "sagittal"]

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


def _resolve_inference_label_values_for_runtime(model_runtime: object) -> Tuple[int, ...]:
    if model_runtime is None:
        raise ValueError("No model runtime is available in session storage.")

    hyperparameters_obj = getattr(model_runtime, "hyperparameters", None)
    if isinstance(hyperparameters_obj, Mapping) and "label_values" in hyperparameters_obj:
        raw_label_values = hyperparameters_obj.get("label_values")
        try:
            resolved_label_values = _coerce_eval_label_values(raw_label_values)
        except Exception as exc:
            raise ValueError(
                "Loaded model metadata has invalid label_values for inference: "
                f"{_exception_message(exc)}. "
                "Load a compatible model checkpoint or rebuild training/validation learning state."
            ) from exc
        raw_num_classes = getattr(model_runtime, "num_classes", None)
        if (
            isinstance(raw_num_classes, Integral)
            and not isinstance(raw_num_classes, bool)
            and int(raw_num_classes) > 0
            and int(raw_num_classes) != int(len(resolved_label_values))
        ):
            raise ValueError(
                "Model runtime label_values length does not match num_classes: "
                f"len(label_values)={len(resolved_label_values)} num_classes={int(raw_num_classes)}."
            )
        return resolved_label_values

    raise ValueError(
        "Inference requires class label_values in loaded model metadata. "
        "Load a checkpoint saved with label_values metadata."
    )


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
    return exception_message(exc)


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
    tool_label_text: str = "1"
    shared_tool_numeric_label: int = 1
    eraser_target_label: Optional[int] = None
    picked_indices: Optional[Tuple[int, int, int]] = None
    picked_label: Optional[int] = None
    pending_bbox_corner: Optional[Tuple[int, int, int]] = None
    flood_fill_target_label: int = 1
    view_layout_mode: ViewLayoutMode = "all"


@dataclass(frozen=True)
class _StagedBoundingBoxDragUpdate:
    before_box: BoundingBox
    after_box: BoundingBox
    before_selected_id: Optional[str]
    after_selected_id: Optional[str]


class MainWindow(QMainWindow):
    _CONTROL_PANEL_MIN_WIDTH_FRACTION = 0.05
    _CONTROL_PANEL_MAX_WIDTH_FRACTION = 0.35
    _CONTROL_PANEL_INITIAL_WIDTH_FRACTION = 0.20

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
        self._semantic_volume: Optional[VolumeData] = None
        self._instance_volume: Optional[VolumeData] = None
        self._training_running = False
        self._training_worker: Optional[object] = None
        self._training_thread: Optional[object] = None
        self._inference_running = False
        self._inference_stop_requested = False
        self._inference_worker: Optional[object] = None
        self._inference_thread: Optional[object] = None
        self._learning_session = LearningSession()
        self._training_parameters: TrainingParameters = DEFAULT_TRAINING_PARAMETERS
        self._learning_state_signature: Optional[Tuple[object, ...]] = None
        self._learning_state_stale = True
        self._deferred_close_after_training = False
        self._deferred_close_training_mode: DeferredTrainingCloseMode = "none"
        self._deferred_close_after_inference = False
        self._deferred_close_inference_mode: DeferredInferenceCloseMode = "none"
        self._headless_close_requested = False
        self._last_saved_segmentation_path: Optional[str] = None
        self._last_saved_segmentation_kind: Optional[SegmentationKind] = None
        self._last_saved_bounding_boxes_path: Optional[str] = None
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
        self._left_layout = left_layout

        control_scroll_area = QScrollArea()
        control_scroll_area.setWidget(self.bottom_panel)
        control_scroll_area.setWidgetResizable(True)
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
        self.bottom_panel.on_view_layout_mode_changed(self._handle_view_layout_mode_changed)
        self.bottom_panel.on_contrast_window_changed(self._handle_contrast_window_changed)
        self.bottom_panel.on_segmentation_opacity_changed(self._handle_segmentation_opacity_changed)
        self.bottom_panel.on_annotation_mode_changed(self._handle_annotation_mode_changed)
        self.bottom_panel.on_bounding_box_mode_changed(self._handle_bounding_box_mode_changed)
        self.bottom_panel.on_annotation_tool_changed(self._handle_annotation_tool_changed)
        self.bottom_panel.on_tool_label_changed(self._handle_tool_label_changed)
        self.bottom_panel.on_next_available_label_requested(self._handle_next_available_label_requested)
        self.bottom_panel.on_brush_radius_changed(self._handle_brush_radius_changed)
        self.bottom_panel.on_flood_fill_requested(self._handle_flood_fill_requested)
        self.bottom_panel.on_undo_requested(self._handle_undo_requested)
        self.bottom_panel.on_redo_requested(self._handle_redo_requested)
        self.bottom_panel.on_open_bounding_boxes_requested(
            self._handle_open_bounding_boxes_request
        )
        self.bottom_panel.on_save_bounding_boxes_requested(
            self._handle_save_bounding_boxes_request
        )
        self.bottom_panel.on_load_model_requested(self._handle_load_model_request)
        self.bottom_panel.on_save_model_requested(self._handle_save_model_request)
        self.bottom_panel.on_segment_inference_requested(
            self._handle_segment_inference_request
        )
        self.bottom_panel.on_segment_inference_headless_close_requested(
            self._handle_segment_inference_headless_close_request
        )
        self.bottom_panel.on_stop_inference_requested(self._handle_stop_inference_request)
        self.bottom_panel.on_train_model_requested(self._handle_train_model_request)
        self.bottom_panel.on_train_model_headless_close_requested(
            self._handle_train_model_headless_close_request
        )
        self.bottom_panel.on_stop_training_requested(self._handle_stop_training_request)
        self.bottom_panel.on_change_training_parameters_requested(
            self._handle_change_training_parameters_request
        )
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
        set_view_layout_mode = getattr(self.bottom_panel, "set_view_layout_mode", None)
        if callable(set_view_layout_mode):
            set_view_layout_mode(self.state.view_layout_mode)
        self._apply_view_layout_mode()
        self.sync_manager.on_state_changed(self._on_sync_state_changed)
        self._handle_segmentation_opacity_changed(self.bottom_panel.segmentation_opacity())
        self._sync_bounding_boxes_ui()
        self._refresh_learning_training_ui_state()
        self._refresh_annotation_ui_state()
        self._apply_main_splitter_width_constraints()
        QTimer.singleShot(0, self._initialize_main_splitter_sizes)

    @staticmethod
    def _learning_session_for(owner: object) -> Optional[LearningSession]:
        session = getattr(owner, "_learning_session", None)
        if isinstance(session, LearningSession):
            return session
        return None

    @staticmethod
    def _learning_session_kwargs_for(owner: object) -> Dict[str, LearningSession]:
        session = MainWindow._learning_session_for(owner)
        if session is None:
            return {}
        return {"learning_session": session}

    @staticmethod
    def _get_learning_dataloader_runtime_for(owner: object) -> object:
        session = MainWindow._learning_session_for(owner)
        if session is not None:
            return session.get_dataloader_runtime()
        return get_current_learning_dataloader_runtime()

    @staticmethod
    def _get_learning_eval_runtimes_by_box_id_for(owner: object) -> Dict[str, object]:
        session = MainWindow._learning_session_for(owner)
        if session is not None:
            return session.get_eval_runtimes_by_box_id()
        return get_current_learning_eval_runtimes_by_box_id()

    @staticmethod
    def _get_learning_model_runtime_for(owner: object) -> object:
        session = MainWindow._learning_session_for(owner)
        if session is not None:
            return session.get_model_runtime()
        return get_current_learning_model_runtime()

    @staticmethod
    def _clear_learning_bbox_batch_for(owner: object) -> None:
        session = MainWindow._learning_session_for(owner)
        if session is not None:
            session.clear_bbox_batch()
            return
        clear_current_learning_bbox_batch()

    @staticmethod
    def _get_learning_bbox_batch_for(owner: object) -> object:
        session = MainWindow._learning_session_for(owner)
        if session is not None:
            return session.get_bbox_batch()
        return get_current_learning_bbox_batch()

    @staticmethod
    def _set_learning_label_space_for(owner: object, label_space: object) -> object:
        session = MainWindow._learning_session_for(owner)
        if session is not None:
            return session.set_label_space(label_space)
        return set_current_learning_label_space(label_space)

    @staticmethod
    def _clear_learning_label_space_for(owner: object) -> None:
        session = MainWindow._learning_session_for(owner)
        if session is not None:
            session.clear_label_space()
            return
        clear_current_learning_label_space()

    @staticmethod
    def _get_learning_label_space_for(owner: object) -> object:
        session = MainWindow._learning_session_for(owner)
        if session is not None:
            return session.get_label_space()
        return get_current_learning_label_space()

    @staticmethod
    def _training_parameters_for(owner: object) -> TrainingParameters:
        return validate_training_parameters(
            getattr(owner, "_training_parameters", DEFAULT_TRAINING_PARAMETERS)
        )

    @staticmethod
    def _instantiate_foundation_model_runtime_for(owner: object, **kwargs: object) -> object:
        training_parameters = MainWindow._training_parameters_for(owner)
        if (
            "config" not in kwargs
            and float(training_parameters.learning_rate)
            != float(DEFAULT_TRAINING_PARAMETERS.learning_rate)
        ):
            kwargs["config"] = replace(
                DEFAULT_FOUNDATION_MODEL_CONFIG,
                lr=float(training_parameters.learning_rate),
            )
        return instantiate_foundation_model_runtime(**kwargs)

    @staticmethod
    def _learning_state_controller_for(owner: object) -> LearningStateController:
        return LearningStateController(
            context=owner,
            operations=LearningStateControllerOperations(
                show_warning=show_warning,
                show_info=show_info,
                prepare_learning_state_from_volumes=prepare_learning_state_from_volumes,
                extract_learning_bboxes_in_memory=extract_learning_bboxes_in_memory,
                compute_and_store_current_learning_class_weights=(
                    compute_and_store_current_learning_class_weights
                ),
                get_learning_dataloader_runtime=(
                    MainWindow._get_learning_dataloader_runtime_for
                ),
                get_learning_eval_runtimes_by_box_id=(
                    MainWindow._get_learning_eval_runtimes_by_box_id_for
                ),
                clear_learning_bbox_batch=MainWindow._clear_learning_bbox_batch_for,
                get_learning_bbox_batch=MainWindow._get_learning_bbox_batch_for,
                set_learning_label_space=MainWindow._set_learning_label_space_for,
                learning_session_kwargs=MainWindow._learning_session_kwargs_for,
                format_class_weights_for_summary=_format_class_weights_for_summary,
            ),
        )

    @staticmethod
    def _model_controller_for(owner: object) -> ModelController:
        return ModelController(
            context=owner,
            operations=ModelControllerOperations(
                show_warning=show_warning,
                show_info=show_info,
                open_model_checkpoint_dialog=open_model_checkpoint_dialog,
                open_save_model_checkpoint_dialog=open_save_model_checkpoint_dialog,
                confirm_reinitialize_model=confirm_reinitialize_model,
                confirm_replace_training_model_with_default_checkpoint=(
                    confirm_replace_training_model_with_default_checkpoint
                ),
                validate_foundation_checkpoint_load_preconditions=(
                    validate_foundation_checkpoint_load_preconditions
                ),
                validate_foundation_model_instantiation_preconditions=(
                    validate_foundation_model_instantiation_preconditions
                ),
                instantiate_foundation_model_runtime=(
                    lambda **kwargs: MainWindow._instantiate_foundation_model_runtime_for(
                        owner,
                        **kwargs,
                    )
                ),
                save_foundation_model_checkpoint=save_foundation_model_checkpoint,
                get_learning_model_runtime=MainWindow._get_learning_model_runtime_for,
                get_learning_label_space=MainWindow._get_learning_label_space_for,
                learning_session_kwargs=MainWindow._learning_session_kwargs_for,
                exception_message=_exception_message,
                normalize_checkpoint_identity=_normalize_checkpoint_identity,
                resolve_shared_eval_label_values=_resolve_shared_eval_label_values,
                resolve_inference_label_values_for_runtime=(
                    _resolve_inference_label_values_for_runtime
                ),
                default_training_checkpoint_path=(
                    _DEFAULT_TRAINING_FOUNDATION_CHECKPOINT_PATH
                ),
            ),
        )

    @staticmethod
    def _training_controller_for(owner: object) -> TrainingController:
        return TrainingController(
            context=owner,
            operations=TrainingControllerOperations(
                show_warning=show_warning,
                show_info=show_info,
                validate_learning_model_training_preconditions=(
                    validate_learning_model_training_preconditions
                ),
                qthread_factory=QThread,
                training_worker_factory=LearningTrainingWorker,
                qapplication_instance=QApplication.instance,
                learning_session_kwargs=MainWindow._learning_session_kwargs_for,
                inference_navigation_lock_active=(
                    MainWindow._inference_navigation_lock_active
                ),
                mark_current_model_runtime_as_trained=(
                    lambda owner, *, completed_epoch_count: (
                        MainWindow._mark_current_model_runtime_as_trained(
                            owner,
                            completed_epoch_count=completed_epoch_count,
                        )
                    )
                ),
                refresh_learning_inference_ui_state=(
                    MainWindow._refresh_learning_inference_ui_state
                ),
                logger=_LOGGER,
            ),
        )

    @staticmethod
    def _inference_controller_for(owner: object) -> InferenceController:
        return InferenceController(
            context=owner,
            operations=InferenceControllerOperations(
                show_warning=show_warning,
                show_info=show_info,
                confirm_replace_inference_bboxes=confirm_replace_inference_bboxes,
                get_learning_model_runtime=MainWindow._get_learning_model_runtime_for,
                resolve_inference_label_values_for_runtime=(
                    _resolve_inference_label_values_for_runtime
                ),
                ordered_inference_boxes=_ordered_inference_boxes,
                find_overlapping_box_id_pairs=_find_overlapping_box_id_pairs,
                apply_predicted_bbox_to_editor=_apply_predicted_bbox_to_editor,
                exception_message=_exception_message,
                qthread_factory=QThread,
                qthread_current_thread=QThread.currentThread,
                inference_worker_factory=LearningInferenceWorker,
                qapplication_instance=QApplication.instance,
                inference_navigation_lock_active=(
                    MainWindow._inference_navigation_lock_active
                ),
                inference_stop_already_requested=(
                    MainWindow._inference_stop_already_requested
                ),
                logger=_LOGGER,
            ),
        )

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        if getattr(self, "_headless_close_requested", False):
            if self._app_event_filter_installed:
                app_instance = QApplication.instance()
                if app_instance is not None:
                    app_instance.removeEventFilter(self)
                self._app_event_filter_installed = False
            event.accept()
            return
        if not self._maybe_resolve_unsaved_data_before_close():
            event.ignore()
            return
        inference_close_prepared = False
        if MainWindow._inference_navigation_lock_active(self):
            if not self._maybe_prepare_close_while_inference():
                event.ignore()
                return
            inference_close_prepared = True

        if not inference_close_prepared:
            if self._training_is_running():
                if not self._maybe_prepare_close_while_training():
                    event.ignore()
                    return
                app_instance = QApplication.instance()
                if app_instance is not None:
                    app_instance.setQuitOnLastWindowClosed(False)
            else:
                self._clear_deferred_close_training_state()
        else:
            # Inference-close decisions take precedence when both are active.
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
        self._clear_deferred_close_training_state()
        return False

    def _maybe_prepare_close_while_inference(self) -> bool:
        decision = ask_inference_running_close_decision(parent=self)
        if decision == InferenceCloseDecision.CANCEL:
            self._clear_deferred_close_inference_state()
            return False
        if decision == InferenceCloseDecision.STOP_AND_CLOSE:
            self._set_deferred_close_after_stop_inference()
            self._request_learning_inference_stop()
            return True
        self._clear_deferred_close_inference_state()
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
        if self._instance_volume is not None:
            self._instance_volume = None
        self._last_saved_segmentation_path = None
        self._last_saved_segmentation_kind = None
        self._last_saved_bounding_boxes_path = None
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
        # New raw volumes always start in the default 3-view layout.
        self.state.view_layout_mode = "all"
        set_view_layout_mode = getattr(self.bottom_panel, "set_view_layout_mode", None)
        if callable(set_view_layout_mode):
            set_view_layout_mode("all")
        apply_layout_mode = getattr(self, "_apply_view_layout_mode", None)
        if callable(apply_layout_mode):
            apply_layout_mode()
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
        self._learning_state_stale = True
        MainWindow._clear_learning_label_space_for(self)
        return True

    def set_semantic_volume(self, volume: VolumeData) -> bool:
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
        self._last_saved_segmentation_path = volume.loader.path
        self._last_saved_segmentation_kind = "semantic"
        self._bbox_drag_staged_history_updates.clear()
        self._global_history.clear()
        editor = SegmentationEditor.from_volume(volume, kind="semantic")
        self._attach_segmentation_editor(editor, kind="semantic")
        self.bottom_panel.set_pyramid_levels(1, kind="Semantic")
        self._refresh_annotation_ui_state()
        self._learning_state_stale = True
        MainWindow._clear_learning_label_space_for(self)
        return True

    def set_instance_volume(self, volume: VolumeData) -> bool:
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
        self._last_saved_segmentation_path = volume.loader.path
        self._last_saved_segmentation_kind = "instance"
        self._bbox_drag_staged_history_updates.clear()
        self._global_history.clear()
        editor = SegmentationEditor.from_volume(volume, kind="instance")
        self._attach_segmentation_editor(editor, kind="instance")
        self.bottom_panel.set_pyramid_levels(1, kind="Instance")
        self._refresh_annotation_ui_state()
        self._learning_state_stale = True
        MainWindow._clear_learning_label_space_for(self)
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
        if _change.kind != "selection":
            self._learning_state_stale = True
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
            self._last_saved_bounding_boxes_path = normalized_path
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

    def _ensure_learning_state_for_action(self, action: LearningStateAction) -> bool:
        return MainWindow._learning_state_controller_for(self).ensure_for_action(action)

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
        return MainWindow._model_controller_for(self).save_model_with_dialog()

    def _save_model_runtime_checkpoint(
        self,
        runtime: object,
        *,
        checkpoint_path: str,
    ) -> None:
        MainWindow._model_controller_for(self).save_model_runtime_checkpoint(
            runtime,
            checkpoint_path=checkpoint_path,
        )

    def _handle_train_model_request(self) -> None:
        MainWindow._training_controller_for(self).handle_train_model_request()

    def _handle_train_model_headless_close_request(self) -> None:
        self._launch_headless_learning_job_and_close(kind="train")

    def _handle_segment_inference_request(self) -> None:
        MainWindow._inference_controller_for(self).handle_segment_inference_request()

    def _handle_segment_inference_headless_close_request(self) -> None:
        self._launch_headless_learning_job_and_close(kind="inference")

    def _handle_stop_inference_request(self) -> None:
        MainWindow._inference_controller_for(self).handle_stop_inference_request()

    def _handle_stop_training_request(self) -> None:
        MainWindow._training_controller_for(self).handle_stop_training_request()

    def _handle_change_training_parameters_request(self) -> None:
        result = open_training_parameters_dialog(
            self._training_parameters,
            parent=self,
        )
        if not bool(getattr(result, "accepted", False)):
            return
        parameters = getattr(result, "parameters", None)
        if parameters is None:
            return
        try:
            MainWindow._apply_training_parameters(self, parameters)
        except Exception as exc:
            show_warning(str(exc), parent=self)

    def _launch_headless_learning_job_and_close(self, *, kind: str) -> bool:
        normalized_kind = str(kind).strip().lower()
        if normalized_kind not in {"train", "inference"}:
            raise ValueError(f"Unsupported headless job kind: {kind!r}")
        if MainWindow._inference_navigation_lock_active(self):
            return False
        if self._abort_if_learning_training_running():
            return False

        job_dir = self._create_headless_job_dir(normalized_kind)
        try:
            common_inputs = self._prepare_headless_common_inputs()
            input_checkpoint_path = self._prepare_headless_input_checkpoint(
                kind=normalized_kind,
                job_dir=job_dir,
            )
            if normalized_kind == "train":
                spec = self._build_headless_training_spec(
                    job_dir=job_dir,
                    input_checkpoint_path=input_checkpoint_path,
                    **common_inputs,
                )
            else:
                spec = self._build_headless_inference_spec(
                    job_dir=job_dir,
                    input_checkpoint_path=input_checkpoint_path,
                    **common_inputs,
                )
            job_path = save_headless_job_spec(spec, str(job_dir / "job.json"))
            self._spawn_headless_after_ui_exit(job_path)
        except Exception as exc:
            show_warning(_exception_message(exc), parent=self)
            return False

        self._release_ui_state_for_headless_close()
        self._headless_close_requested = True
        self.close()
        return True

    def _release_ui_state_for_headless_close(self) -> None:
        self._release_learning_state_for_headless_close()
        self._close_loaded_volumes_for_headless_close()
        self._segmentation_editor = None
        self._pending_render_view_ids.clear()
        self._pending_annotation_peer_view_ids.clear()
        self._annotation_dirty_views.clear()
        self._bbox_pending_peer_view_ids.clear()
        self._bbox_drag_staged_history_updates.clear()
        self._global_history.clear()

    def _release_learning_state_for_headless_close(self) -> None:
        session = getattr(self, "_learning_session", None)
        for method_name in (
            "clear_dataloader_runtime",
            "clear_eval_runtimes_by_box_id",
            "clear_model_runtime",
            "clear_bbox_batch",
            "clear_label_space",
        ):
            method = getattr(session, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    _LOGGER.debug(
                        "Failed to clear learning session state during headless close: %s",
                        method_name,
                        exc_info=True,
                    )
        for clear_func in (
            clear_current_learning_dataloader_runtime,
            clear_current_learning_eval_runtimes_by_box_id,
            clear_current_learning_model_runtime,
            clear_current_learning_bbox_batch,
            clear_current_learning_label_space,
        ):
            try:
                clear_func()
            except Exception:
                _LOGGER.debug(
                    "Failed to clear global learning state during headless close",
                    exc_info=True,
                )

    def _close_loaded_volumes_for_headless_close(self) -> None:
        seen_ids = set()
        for attr_name in ("_raw_volume", "_semantic_volume", "_instance_volume"):
            volume = getattr(self, attr_name, None)
            if volume is None:
                setattr(self, attr_name, None)
                continue
            volume_id = id(volume)
            if volume_id not in seen_ids:
                seen_ids.add(volume_id)
                close = getattr(volume, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        _LOGGER.debug(
                            "Failed to close %s during headless close",
                            attr_name,
                            exc_info=True,
                        )
            setattr(self, attr_name, None)

    def _prepare_headless_common_inputs(self) -> Dict[str, str]:
        raw_path = self._headless_reopenable_volume_path(self._raw_volume, name="raw volume")
        active_segmentation = self._active_segmentation_volume()
        if active_segmentation is None:
            raise RuntimeError(
                "Headless learning requires a saved semantic segmentation map."
            )
        segmentation_kind, _segmentation_volume = active_segmentation
        if segmentation_kind != "semantic":
            raise RuntimeError(
                "Headless learning currently requires the active segmentation to be semantic."
            )
        segmentation_path = self._ensure_headless_segmentation_path()
        bbox_path = self._ensure_headless_bounding_boxes_path()
        return {
            "raw_volume_path": raw_path,
            "segmentation_path": segmentation_path,
            "segmentation_kind": segmentation_kind,
            "bbox_path": bbox_path,
        }

    def _build_headless_training_spec(
        self,
        *,
        job_dir: Path,
        raw_volume_path: str,
        segmentation_path: str,
        segmentation_kind: str,
        bbox_path: str,
        input_checkpoint_path: str,
    ) -> HeadlessJobSpec:
        self._require_headless_training_boxes()
        dialog_result = open_save_model_checkpoint_dialog(
            self,
            retry_on_overwrite_decline=True,
        )
        if not dialog_result.accepted or not dialog_result.path:
            raise RuntimeError("Headless training canceled: no output checkpoint selected.")
        return HeadlessJobSpec(
            kind="train",
            raw_volume_path=raw_volume_path,
            segmentation_path=segmentation_path,
            segmentation_kind=cast(SegmentationKind, segmentation_kind),
            bbox_path=bbox_path,
            load_mode=self._load_mode,
            cache_max_bytes=int(self._cache_max_bytes),
            training_parameters=validate_training_parameters(self._training_parameters),
            input_checkpoint_path=input_checkpoint_path,
            output_checkpoint_path=str(Path(dialog_result.path).expanduser()),
            job_dir=str(job_dir),
            source_pid=os.getpid(),
        )

    def _build_headless_inference_spec(
        self,
        *,
        job_dir: Path,
        raw_volume_path: str,
        segmentation_path: str,
        segmentation_kind: str,
        bbox_path: str,
        input_checkpoint_path: str,
    ) -> HeadlessJobSpec:
        self._require_headless_inference_boxes()
        while True:
            dialog_result = open_save_segmentation_dialog(self)
            if not dialog_result.accepted or not dialog_result.path or not dialog_result.format:
                raise RuntimeError("Headless inference canceled: no output segmentation selected.")
            output_path = str(Path(dialog_result.path).expanduser())
            if Path(output_path).exists() and not confirm_overwrite(output_path, parent=self):
                continue
            return HeadlessJobSpec(
                kind="inference",
                raw_volume_path=raw_volume_path,
                segmentation_path=segmentation_path,
                segmentation_kind=cast(SegmentationKind, segmentation_kind),
                bbox_path=bbox_path,
                load_mode=self._load_mode,
                cache_max_bytes=int(self._cache_max_bytes),
                training_parameters=validate_training_parameters(self._training_parameters),
                input_checkpoint_path=input_checkpoint_path,
                output_segmentation_path=output_path,
                output_segmentation_format=str(dialog_result.format),
                job_dir=str(job_dir),
                source_pid=os.getpid(),
            )

    def _prepare_headless_input_checkpoint(self, *, kind: str, job_dir: Path) -> str:
        runtime = MainWindow._get_learning_model_runtime_for(self)
        if runtime is None:
            if kind == "train":
                checkpoint_path = str(
                    Path(_DEFAULT_TRAINING_FOUNDATION_CHECKPOINT_PATH).expanduser()
                )
                if Path(checkpoint_path).exists():
                    return checkpoint_path
            raise RuntimeError(
                "Headless learning requires a loaded model runtime so the starting "
                "checkpoint can be saved for the job."
            )
        checkpoint_path = str(job_dir / "input_model.cp")
        self._save_model_runtime_checkpoint(runtime, checkpoint_path=checkpoint_path)
        return checkpoint_path

    def _ensure_headless_segmentation_path(self) -> str:
        active = self._active_segmentation_volume()
        if active is None:
            raise RuntimeError("No semantic segmentation map is loaded.")
        kind, volume = active
        tracked_path = (
            self._last_saved_segmentation_path
            if self._last_saved_segmentation_kind == kind
            else None
        )
        if (
            not self._has_unsaved_segmentation_changes()
            and self._is_headless_reopenable_path(tracked_path)
        ):
            return str(tracked_path)
        if (
            not self._has_unsaved_segmentation_changes()
            and self._is_headless_reopenable_path(volume.loader.path)
        ):
            return str(volume.loader.path)

        show_info(
            (
                "The current segmentation must be saved before launching a "
                "headless job and closing the UI because the headless job "
                "reloads the segmentation from disk after the UI exits."
            ),
            parent=self,
        )
        if not self._save_active_segmentation_with_dialog():
            raise RuntimeError("Headless job canceled: segmentation was not saved.")
        if not self._is_headless_reopenable_path(self._last_saved_segmentation_path):
            raise RuntimeError("Saved segmentation path cannot be reopened by the headless job.")
        return str(self._last_saved_segmentation_path)

    def _ensure_headless_bounding_boxes_path(self) -> str:
        if (
            not self._has_unsaved_bounding_box_changes()
            and self._is_headless_reopenable_path(self._last_saved_bounding_boxes_path)
        ):
            return str(self._last_saved_bounding_boxes_path)
        show_info(
            (
                "The current bounding boxes must be saved before launching a "
                "headless job and closing the UI because the headless job "
                "reloads the boxes from disk after the UI exits."
            ),
            parent=self,
        )
        if not self._save_bounding_boxes_with_dialog():
            raise RuntimeError("Headless job canceled: bounding boxes were not saved.")
        if not self._is_headless_reopenable_path(self._last_saved_bounding_boxes_path):
            raise RuntimeError("Saved bounding-box path cannot be reopened by the headless job.")
        return str(self._last_saved_bounding_boxes_path)

    def _require_headless_training_boxes(self) -> None:
        boxes = tuple(self._bbox_manager.boxes())
        if not any(str(box.label) == "train" for box in boxes):
            raise RuntimeError("Headless training requires at least one bbox labeled 'train'.")
        if not any(str(box.label) == "validation" for box in boxes):
            raise RuntimeError(
                "Headless training requires at least one bbox labeled 'validation'."
            )

    def _require_headless_inference_boxes(self) -> None:
        ordered_box_ids = tuple(row.box_id for row in self.bottom_panel.state.bbox_rows)
        boxes_by_id = {box.id: box for box in self._bbox_manager.boxes()}
        inference_boxes = _ordered_inference_boxes(
            ordered_box_ids=ordered_box_ids,
            boxes_by_id=boxes_by_id,
        )
        if not inference_boxes:
            raise RuntimeError(
                "Headless inference requires at least one bbox labeled 'inference'."
            )
        overlapping_pairs = _find_overlapping_box_id_pairs(inference_boxes)
        if overlapping_pairs:
            formatted = ", ".join(f"{a}/{b}" for a, b in overlapping_pairs)
            raise RuntimeError(
                "Headless inference bboxes must not overlap. Overlapping pairs: "
                f"{formatted}"
            )

    def _headless_reopenable_volume_path(
        self,
        volume: Optional[VolumeData],
        *,
        name: str,
    ) -> str:
        if volume is None:
            raise RuntimeError(f"Headless job requires a loaded {name}.")
        path = getattr(getattr(volume, "loader", None), "path", None)
        if not self._is_headless_reopenable_path(path):
            raise RuntimeError(f"The {name} path cannot be reopened by the headless job.")
        return str(path)

    @staticmethod
    def _is_headless_reopenable_path(path: object) -> bool:
        if not isinstance(path, str) or not path.strip():
            return False
        path_parts = str(path).split("::")
        base_path = path_parts[0]
        for qualifier in path_parts[1:]:
            normalized = qualifier.strip().lower()
            if normalized == "editable" or normalized.startswith("generated-"):
                return False
        return bool(base_path and Path(base_path).expanduser().exists())

    @staticmethod
    def _create_headless_job_dir(kind: str) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        job_dir = Path(".headless-job") / f"{timestamp}-{kind}-{os.getpid()}"
        job_dir.mkdir(parents=True, exist_ok=False)
        return job_dir

    @staticmethod
    def _spawn_headless_after_ui_exit(job_path: str) -> subprocess.Popen:
        repo_root = Path(__file__).resolve().parents[2]
        launcher_path = repo_root / "launch_headless_after_ui_exit.py"
        runner_path = repo_root / "run_headless_job.py"
        command = [
            sys.executable,
            str(launcher_path),
            "--wait-pid",
            str(os.getpid()),
            "--job",
            str(job_path),
            "--python",
            sys.executable,
            "--runner",
            str(runner_path),
            "--log-level",
            "INFO",
        ]
        return subprocess.Popen(command, close_fds=True, start_new_session=True)

    @staticmethod
    def _training_parameters_require_learning_state_rebuild(
        before: TrainingParameters,
        after: TrainingParameters,
    ) -> bool:
        return (
            int(before.training_batch_size) != int(after.training_batch_size)
            or int(before.validation_batch_size) != int(after.validation_batch_size)
            or int(before.patches_per_epoch) != int(after.patches_per_epoch)
        )

    def _apply_training_parameters(self, parameters: TrainingParameters) -> None:
        normalized = validate_training_parameters(parameters)
        previous = self._training_parameters
        if normalized == previous:
            return
        self._training_parameters = normalized
        if float(previous.learning_rate) != float(normalized.learning_rate):
            MainWindow._apply_training_learning_rate_to_current_runtime(
                self,
                float(normalized.learning_rate),
            )
        if MainWindow._training_parameters_require_learning_state_rebuild(
            previous,
            normalized,
        ):
            MainWindow._mark_learning_state_stale_for_training_parameter_change(self)

    def _mark_learning_state_stale_for_training_parameter_change(self) -> None:
        self._learning_state_stale = True

    def _apply_training_learning_rate_to_current_runtime(self, learning_rate: float) -> None:
        normalized_lr = float(
            validate_training_parameters(
                TrainingParameters(
                    learning_rate=learning_rate,
                    training_batch_size=self._training_parameters.training_batch_size,
                    validation_batch_size=self._training_parameters.validation_batch_size,
                    patches_per_epoch=self._training_parameters.patches_per_epoch,
                    early_stopping_patience=self._training_parameters.early_stopping_patience,
                )
            ).learning_rate
        )
        runtime = MainWindow._get_learning_model_runtime_for(self)
        if runtime is None:
            return
        hyperparameters_obj = getattr(runtime, "hyperparameters", None)
        if isinstance(hyperparameters_obj, dict):
            hyperparameters_obj["lr"] = normalized_lr

        optimizer = getattr(runtime, "optimizer", None)
        param_groups = getattr(optimizer, "param_groups", None)
        if not isinstance(param_groups, (list, tuple)):
            return
        for group in tuple(param_groups):
            if not isinstance(group, dict):
                continue
            try:
                decay_rate = float(group.get("lwise_lr_decay_rate", 1.0))
            except Exception:
                decay_rate = 1.0
            group["lr"] = normalized_lr * decay_rate

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

        end_annotation_modification = getattr(self, "_end_annotation_modification", None)
        if callable(end_annotation_modification):
            end_annotation_modification()

        operation_name = "erase_bbox_segmentation_selected"
        editor.begin_modification(operation_name)
        try:
            editor.erase_masked_region(
                z_bounds=(int(z_bounds[0]), int(z_bounds[1])),
                y_bounds=(int(y_bounds[0]), int(y_bounds[1])),
                x_bounds=(int(x_bounds[0]), int(x_bounds[1])),
                region_mask=np.asarray(union_mask, dtype=bool),
                operation_name=operation_name,
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
                int(np.count_nonzero(union_mask)),
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
        return bbox_ops.build_selected_bbox_union_domain(boxes).as_tuple()

    @staticmethod
    def _expand_axis_bounds_with_halo(
        bounds: Tuple[int, int],
        *,
        axis_length: int,
        halo_size: int,
    ) -> Tuple[int, int]:
        return bbox_ops.expand_axis_bounds_with_halo(
            bounds,
            axis_length=axis_length,
            halo_size=halo_size,
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
        return bbox_ops.build_selected_bbox_processing_regions(
            boxes,
            volume_shape=volume_shape,
            halo_size=halo_size,
        ).as_tuple()

    @staticmethod
    def _reflect_axis_indices(
        indices: np.ndarray,
        *,
        axis_length: int,
    ) -> np.ndarray:
        return bbox_ops.reflect_axis_indices(indices, axis_length=axis_length)

    @staticmethod
    def _build_extended_foreground_with_halo_padding(
        *,
        segmentation_volume: np.ndarray,
        core_z_bounds: Tuple[int, int],
        core_y_bounds: Tuple[int, int],
        core_x_bounds: Tuple[int, int],
        halo_size: int = 1,
    ) -> np.ndarray:
        return bbox_ops.build_extended_foreground_with_halo_padding(
            segmentation_volume=segmentation_volume,
            core_z_bounds=core_z_bounds,
            core_y_bounds=core_y_bounds,
            core_x_bounds=core_x_bounds,
            halo_size=halo_size,
        )

    @staticmethod
    def _mask_to_absolute_coordinates(
        mask: np.ndarray,
        *,
        origin: Tuple[int, int, int],
    ) -> np.ndarray:
        return bbox_ops.mask_to_absolute_coordinates(mask, origin=origin)

    @staticmethod
    def _bbox_segmentation_operation_display_name(operation: BBoxSegmentationOperation) -> str:
        return bbox_ops.bbox_segmentation_operation_display_name(operation)

    @staticmethod
    def _compute_set_mask_labels(
        *,
        segmentation_roi: np.ndarray,
        set_mask: np.ndarray,
        union_mask: np.ndarray,
        fallback_label: int,
    ) -> np.ndarray:
        return bbox_ops.compute_set_mask_labels(
            segmentation_roi=segmentation_roi,
            set_mask=set_mask,
            union_mask=union_mask,
            fallback_label=fallback_label,
        )

    @staticmethod
    def _compute_selected_bbox_binary_operation(
        *,
        operation: BBoxSegmentationOperation,
        foreground_mask: np.ndarray,
        union_mask: np.ndarray,
    ) -> np.ndarray:
        return bbox_ops.compute_selected_bbox_binary_operation(
            operation=operation,
            foreground_mask=foreground_mask,
            union_mask=union_mask,
        )

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
        return bbox_ops.compute_selected_bbox_binary_operation_with_halo_context(
            operation=operation,
            segmentation_volume=segmentation_volume,
            core_z_bounds=core_z_bounds,
            core_y_bounds=core_y_bounds,
            core_x_bounds=core_x_bounds,
            halo_size=halo_size,
            binary_operation_func=MainWindow._compute_selected_bbox_binary_operation,
        )

    @staticmethod
    def _count_true_neighbors_3x3x3(mask: np.ndarray) -> np.ndarray:
        return bbox_ops.count_true_neighbors_3x3x3(mask)

    def _segment_inference_bboxes_with_dialog(self) -> bool:
        return MainWindow._inference_controller_for(
            self
        ).segment_inference_bboxes_with_dialog()

    @staticmethod
    def _run_learning_inference_inline_compat(
        target: object,
        *,
        model_runtime: object,
        inference_boxes: Sequence[BoundingBox],
        raw_array: np.ndarray,
        label_values: Sequence[int],
        volume_shape: Sequence[int],
    ) -> bool:
        return MainWindow._inference_controller_for(
            target
        ).run_learning_inference_inline_compat(
            model_runtime=model_runtime,
            inference_boxes=inference_boxes,
            raw_array=raw_array,
            label_values=label_values,
            volume_shape=volume_shape,
        )

    def _show_inference_navigation_only_notice(self) -> None:
        MainWindow._inference_controller_for(self).show_inference_navigation_only_notice()

    def _start_learning_inference_background(
        self,
        *,
        model_runtime: object,
        inference_boxes: Sequence[BoundingBox],
        raw_array: np.ndarray,
        label_values: Sequence[int],
        volume_shape: Sequence[int],
    ) -> None:
        MainWindow._inference_controller_for(self).start_learning_inference_background(
            model_runtime=model_runtime,
            inference_boxes=inference_boxes,
            raw_array=raw_array,
            label_values=label_values,
            volume_shape=volume_shape,
        )

    def _apply_inference_predictions_in_single_commit(
        self,
        *,
        editor: SegmentationEditor,
        predictions: Sequence[LearningInferencePrediction],
        initial_failure_by_box_id: Optional[Mapping[str, str]] = None,
    ) -> Tuple[int, Tuple[str, ...], Dict[str, str]]:
        return MainWindow._inference_controller_for(
            self
        ).apply_inference_predictions_in_single_commit(
            editor=editor,
            predictions=predictions,
            initial_failure_by_box_id=initial_failure_by_box_id,
        )

    def _on_learning_inference_completed(self, result: object) -> None:
        MainWindow._inference_controller_for(self).on_learning_inference_completed(result)

    def _on_learning_inference_canceled(self, message: str) -> None:
        MainWindow._inference_controller_for(self).on_learning_inference_canceled(message)

    def _on_learning_inference_failed(self, message: str) -> None:
        MainWindow._inference_controller_for(self).on_learning_inference_failed(message)

    def _on_learning_inference_thread_finished(self) -> None:
        MainWindow._inference_controller_for(self).on_learning_inference_thread_finished()

    def _request_learning_inference_stop(self) -> None:
        MainWindow._inference_controller_for(self).request_learning_inference_stop()

    def _request_learning_training_stop(self) -> None:
        MainWindow._training_controller_for(self).request_learning_training_stop()

    def _runtime_training_provenance(
        self,
        runtime: object,
    ) -> Tuple[Optional[str], bool, int]:
        return MainWindow._model_controller_for(self).runtime_training_provenance(runtime)

    def _runtime_requires_training_reinitialization(self, runtime: object) -> bool:
        return MainWindow._model_controller_for(
            self
        ).runtime_requires_training_reinitialization(runtime)

    def _reinitialize_training_runtime_from_default_checkpoint(self) -> bool:
        return MainWindow._model_controller_for(
            self
        ).reinitialize_training_runtime_from_default_checkpoint()

    def _ensure_training_runtime_for_new_training(self) -> bool:
        return MainWindow._model_controller_for(
            self
        ).ensure_training_runtime_for_new_training()

    def _mark_current_model_runtime_as_trained(self, *, completed_epoch_count: int) -> None:
        MainWindow._model_controller_for(self).mark_current_model_runtime_as_trained(
            completed_epoch_count=completed_epoch_count
        )

    def _train_model_on_dataset_with_dialog(self) -> bool:
        return MainWindow._training_controller_for(
            self
        ).train_model_on_dataset_with_dialog()

    def _start_learning_training_background(self, *, preconditions: object) -> None:
        MainWindow._training_controller_for(self).start_learning_training_background(
            preconditions=preconditions
        )

    def _on_learning_training_completed(self, result: object) -> None:
        MainWindow._training_controller_for(self).on_learning_training_completed(result)

    def _on_learning_training_failed(self, message: str) -> None:
        MainWindow._training_controller_for(self).on_learning_training_failed(message)

    def _on_learning_training_thread_finished(self) -> None:
        MainWindow._training_controller_for(self).on_learning_training_thread_finished()

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
            self._last_saved_bounding_boxes_path = save_path
            self._mark_bounding_boxes_clean()
            show_info(
                f"Saved {box_count} bounding box(es) to {save_path}",
                parent=self,
            )
            return True

    def _prepare_learning_state(
        self,
        *,
        require_class_weights: bool,
        show_success_dialog: bool,
    ) -> bool:
        return MainWindow._learning_state_controller_for(self).prepare_learning_state(
            require_class_weights=require_class_weights,
            show_success_dialog=show_success_dialog,
        )

    def _persist_model_runtime_label_values_from_eval_runtimes(
        self,
        runtime: object,
        *,
        eval_runtimes_by_box_id: Mapping[str, object],
    ) -> None:
        MainWindow._model_controller_for(
            self
        ).persist_model_runtime_label_values_from_eval_runtimes(
            runtime,
            eval_runtimes_by_box_id=eval_runtimes_by_box_id,
        )

    def _current_learning_state_signature(self) -> Tuple[object, ...]:
        return MainWindow._learning_state_controller_for(self).current_signature()

    def _instantiate_foundation_model_with_dialog(self) -> bool:
        return MainWindow._model_controller_for(
            self
        ).instantiate_foundation_model_with_dialog()

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
            self._semantic_volume = editable_volume
        else:
            self._semantic_volume = None
            self._instance_volume = editable_volume
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
        previous_tool = self.state.annotation_tool
        normalized_tool = self._normalize_annotation_tool(tool)
        editor = self._segmentation_editor
        if normalized_tool == "eraser":
            if self.state.eraser_target_label is None:
                self.state.tool_label_text = "All"
            else:
                target_value = int(self.state.eraser_target_label)
                self.state.shared_tool_numeric_label = int(target_value)
                self.state.tool_label_text = str(int(target_value))
        else:
            try:
                shared_label = int(self.state.shared_tool_numeric_label)
            except (TypeError, ValueError):
                shared_label = 1
            if shared_label < 0:
                shared_label = 1
            if previous_tool == "eraser" and self.state.eraser_target_label is None:
                shared_label = 1
            elif previous_tool == "eraser" and self.state.eraser_target_label is not None:
                shared_label = int(self.state.eraser_target_label)
            if editor is not None:
                max_label = self._annotation_label_ui_max()
                shared_label = max(0, min(int(shared_label), int(max_label)))
                try:
                    editor.set_active_label(int(shared_label))
                except ValueError:
                    shared_label = 1 if max_label >= 1 else 0
                    editor.set_active_label(int(shared_label))
            self.state.shared_tool_numeric_label = int(shared_label)
            self.state.flood_fill_target_label = int(shared_label)
            self.state.tool_label_text = str(int(shared_label))
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
        if str(editor.kind) == "semantic":
            self._learning_state_stale = True
            MainWindow._clear_learning_label_space_for(self)
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
        return MainWindow._training_controller_for(self).training_is_running()

    def _inference_is_running(self) -> bool:
        return MainWindow._inference_controller_for(self).inference_is_running()

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
        MainWindow._inference_controller_for(
            self
        ).clear_learning_inference_stop_request_state()

    def _clear_deferred_close_inference_state(self) -> None:
        MainWindow._inference_controller_for(self).clear_deferred_close_inference_state()

    def _set_deferred_close_after_stop_inference(self) -> None:
        clear_training_state = getattr(self, "_clear_deferred_close_training_state", None)
        if callable(clear_training_state):
            clear_training_state()
        else:
            MainWindow._clear_deferred_close_training_state(self)
        MainWindow._inference_controller_for(
            self
        ).set_deferred_close_after_stop_inference()

    def _finalize_deferred_close_inference_and_quit(self) -> None:
        MainWindow._inference_controller_for(
            self
        ).finalize_deferred_close_inference_and_quit()

    def _clear_deferred_close_training_state(self) -> None:
        MainWindow._training_controller_for(
            self
        ).clear_deferred_close_training_state()

    def _set_deferred_close_after_stop_training(self) -> None:
        clear_inference_state = getattr(self, "_clear_deferred_close_inference_state", None)
        if callable(clear_inference_state):
            clear_inference_state()
        else:
            MainWindow._clear_deferred_close_inference_state(self)
        MainWindow._training_controller_for(
            self
        ).set_deferred_close_after_stop_training()

    def _refresh_learning_training_ui_state(self) -> None:
        MainWindow._training_controller_for(self).refresh_learning_training_ui_state()

    def _refresh_learning_inference_ui_state(self) -> None:
        MainWindow._inference_controller_for(self).refresh_learning_inference_ui_state()

    def _enter_learning_training_running_state(
        self,
        *,
        worker: object,
        thread: object,
    ) -> None:
        MainWindow._training_controller_for(
            self
        ).enter_learning_training_running_state(worker=worker, thread=thread)

    def _exit_learning_training_running_state(self) -> None:
        MainWindow._training_controller_for(self).exit_learning_training_running_state()

    def _enter_learning_inference_running_state(
        self,
        *,
        worker: object,
        thread: object,
    ) -> None:
        MainWindow._inference_controller_for(
            self
        ).enter_learning_inference_running_state(worker=worker, thread=thread)

    def _exit_learning_inference_running_state(self) -> None:
        MainWindow._inference_controller_for(self).exit_learning_inference_running_state()

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
        try:
            shared_label = int(self.state.shared_tool_numeric_label)
        except (TypeError, ValueError):
            shared_label = 1
        if shared_label < 0:
            shared_label = 1
        if editor is not None:
            shared_label = max(0, min(int(shared_label), self._annotation_label_ui_max()))
        self.state.shared_tool_numeric_label = int(shared_label)
        self.state.flood_fill_target_label = int(shared_label)
        if editor is None:
            self.bottom_panel.set_annotation_controls_enabled(False)
            if self.state.annotation_tool == "eraser":
                label_text = (
                    "All"
                    if self.state.eraser_target_label is None
                    else str(int(self.state.eraser_target_label))
                )
            else:
                label_text = str(int(self.state.shared_tool_numeric_label))
            self.state.tool_label_text = str(label_text)
            self.bottom_panel.set_tool_label(self.state.tool_label_text)
            self._refresh_picked_readout()
            self._apply_picker_state_to_views()
            self._refresh_undo_ui_state()
            return

        self.bottom_panel.set_annotation_controls_enabled(True)
        if self.state.annotation_tool == "eraser":
            label_text = (
                "All"
                if self.state.eraser_target_label is None
                else str(self.state.eraser_target_label)
            )
        else:
            label_text = str(int(self.state.shared_tool_numeric_label))
        self.state.tool_label_text = str(label_text)
        self.bottom_panel.set_tool_label(self.state.tool_label_text)
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

    def _handle_segmentation_opacity_changed(self, opacity: float) -> None:
        try:
            normalized = float(opacity)
        except (TypeError, ValueError):
            normalized = 0.3
        if not np.isfinite(normalized):
            normalized = 0.3
        normalized = max(0.0, min(1.0, normalized))
        for view in self.views.values():
            setter = getattr(view, "set_segmentation_opacity", None)
            if callable(setter):
                setter(normalized)

    def _handle_auto_level_mode_changed(self, enabled: bool) -> None:
        if not self.state.volume_loaded:
            return
        auto_enabled = bool(enabled)
        self.renderer.set_auto_level_enabled(auto_enabled)
        self._queue_contrast_rerender()

    def _handle_manual_level_requested(self, level: int) -> None:
        if not self.state.volume_loaded:
            return
        self.renderer.set_manual_level(int(level))
        self._queue_contrast_rerender()

    def _handle_view_layout_mode_changed(self, mode: str) -> None:
        normalized = str(mode).strip().lower()
        if normalized not in {"all", "axial", "coronal", "sagittal"}:
            normalized = "all"
        current = str(self.state.view_layout_mode).strip().lower()
        self.state.view_layout_mode = cast(ViewLayoutMode, normalized)
        if current != normalized:
            panel = getattr(self, "bottom_panel", None)
            setter = getattr(panel, "set_view_layout_mode", None)
            if callable(setter):
                setter(normalized)
        apply_layout_mode = getattr(self, "_apply_view_layout_mode", None)
        if callable(apply_layout_mode):
            apply_layout_mode()
        else:
            MainWindow._apply_view_layout_mode(self)

    def _apply_view_layout_mode(self) -> None:
        layout_obj = getattr(self, "_left_layout", None)
        if not isinstance(layout_obj, QGridLayout):
            return
        axial = self.views.get("axial")
        coronal = self.views.get("coronal")
        sagittal = self.views.get("sagittal")
        if not isinstance(axial, QWidget) or not isinstance(coronal, QWidget) or not isinstance(sagittal, QWidget):
            return

        mode = str(self.state.view_layout_mode).strip().lower()
        if mode not in {"all", "axial", "coronal", "sagittal"}:
            mode = "all"
            self.state.view_layout_mode = cast(ViewLayoutMode, mode)

        # Re-adding widgets to the grid is enough to update their positions/spans.
        layout_obj.removeWidget(axial)
        layout_obj.removeWidget(coronal)
        layout_obj.removeWidget(sagittal)

        if mode == "all":
            axial.show()
            coronal.show()
            sagittal.show()
            layout_obj.addWidget(axial, 0, 0, 1, 1)
            layout_obj.addWidget(coronal, 0, 1, 1, 1)
            layout_obj.addWidget(sagittal, 1, 0, 1, 2)
            if bool(getattr(self.state, "volume_loaded", False)):
                self._queue_visible_view_rerender()
            return

        selected_view = axial
        hidden_views = (coronal, sagittal)
        if mode == "coronal":
            selected_view = coronal
            hidden_views = (axial, sagittal)
        elif mode == "sagittal":
            selected_view = sagittal
            hidden_views = (axial, coronal)
        for view in hidden_views:
            view.hide()
        selected_view.show()
        layout_obj.addWidget(selected_view, 0, 0, 2, 2)

        if bool(getattr(self.state, "volume_loaded", False)):
            self._queue_visible_view_rerender()

    def _visible_view_ids(self) -> Tuple[ViewId, ...]:
        visible_ids = []
        for view_id, view in self.views.items():
            is_hidden = getattr(view, "isHidden", None)
            if callable(is_hidden):
                try:
                    if bool(is_hidden()):
                        continue
                except Exception:
                    pass
            visible_ids.append(view_id)
        return tuple(visible_ids)

    def _queue_visible_view_rerender(self) -> None:
        visible_ids = self._visible_view_ids()
        if not visible_ids:
            return
        for view_id in visible_ids:
            self._queue_render(view_id)

    def _queue_contrast_rerender(self) -> None:
        if not self.views:
            self.render_all()
            return
        visible_view_ids_getter = getattr(self, "_visible_view_ids", None)
        if callable(visible_view_ids_getter):
            visible_view_ids = tuple(visible_view_ids_getter())
        else:
            visible_view_ids = tuple(self.views.keys())
        if not visible_view_ids:
            visible_view_ids = tuple(self.views.keys())
        for view_id in visible_view_ids:
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
            prepared = load_prepared_volume(
                result.path,
                kind="raw",
                load_mode=self._load_mode,
                cache_max_bytes=self._cache_max_bytes,
                pyramid_levels=4,
            )
            if self.set_volume(prepared.volume, levels=prepared.levels):
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
                pyramid_levels=1,
            )
            if self.set_semantic_volume(prepared.volume):
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
                pyramid_levels=1,
            )
            if self.set_instance_volume(prepared.volume):
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
            self.state.shared_tool_numeric_label = int(editor.active_label)
            self.state.flood_fill_target_label = int(editor.active_label)
            self.state.tool_label_text = str(int(editor.active_label))
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
            self.state.shared_tool_numeric_label = int(next_label)
            self.state.flood_fill_target_label = int(next_label)
            self.state.tool_label_text = str(int(next_label))
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

    def _handle_tool_label_changed(self, value: str) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            self._refresh_annotation_ui_state()
            return
        text = str(value).strip()
        try:
            previous_numeric_label = int(self.state.shared_tool_numeric_label)
        except (TypeError, ValueError):
            previous_numeric_label = 1
        if previous_numeric_label < 0:
            previous_numeric_label = 1
        self.state.tool_label_text = text
        if self.state.annotation_tool == "eraser":
            self._handle_eraser_target_changed(text)
            return
        try:
            parsed = int(text)
        except ValueError:
            show_warning("Tool Label must be a non-negative integer.", parent=self)
            self.state.tool_label_text = str(previous_numeric_label)
            self._refresh_annotation_ui_state()
            return
        if parsed < 0:
            show_warning("Tool Label must be a non-negative integer.", parent=self)
            self.state.tool_label_text = str(previous_numeric_label)
            self._refresh_annotation_ui_state()
            return
        editor = self._segmentation_editor
        if editor is not None:
            max_label = int(np.iinfo(editor.dtype).max)
            if parsed > max_label:
                show_warning(
                    f"Tool Label {parsed} exceeds max value {max_label} for dtype {editor.dtype}.",
                    parent=self,
                )
                self.state.tool_label_text = str(previous_numeric_label)
                self._refresh_annotation_ui_state()
                return
            try:
                editor.set_active_label(int(parsed))
            except ValueError:
                self.state.tool_label_text = str(previous_numeric_label)
                self._refresh_annotation_ui_state()
                return
        self.state.shared_tool_numeric_label = int(parsed)
        self.state.flood_fill_target_label = int(parsed)
        self.state.tool_label_text = str(int(parsed))
        if self.state.annotation_tool == "flood_filler":
            self._refresh_annotation_ui_state()
            return
        self._refresh_annotation_ui_state()

    def _handle_eraser_target_changed(self, value: str) -> None:
        if MainWindow._inference_navigation_lock_active(self):
            self._refresh_annotation_ui_state()
            return
        text = str(value).strip()
        if text == "" or text.lower() == "all":
            self.state.eraser_target_label = None
            self.state.tool_label_text = "All"
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
        self.state.tool_label_text = str(parsed)
        self.state.shared_tool_numeric_label = int(parsed)
        self.state.flood_fill_target_label = int(parsed)
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

    def _handle_flood_fill_requested(self, _target_label: int) -> None:
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
            target = int(self.state.shared_tool_numeric_label)
            self.state.flood_fill_target_label = target
            self.state.shared_tool_numeric_label = target
            self.state.tool_label_text = str(target)
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
            self._instance_volume = None
            kind = "semantic"
        else:
            path = (
                self._instance_volume.loader.path
                if self._instance_volume is not None
                else f"{editor.source_path}::editable"
            )
            editable_volume = editor.to_volume_data(path=path)
            self._instance_volume = editable_volume
            self._semantic_volume = None
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
            self._last_saved_segmentation_path = save_path
            self._last_saved_segmentation_kind = kind
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
