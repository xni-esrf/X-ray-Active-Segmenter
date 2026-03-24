from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Optional, Tuple, cast

import numpy as np

from ..bbox import BoundingBox, BoundingBoxLabel

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


BrushRadius = int
AnnotationTool = Literal["brush", "eraser", "flood_filler"]
_BBOX_LABEL_OPTIONS = ("train", "validation", "inference")


def _normalize_brush_radius(value: object) -> BrushRadius:
    try:
        radius = int(value)
    except (TypeError, ValueError):
        return 0
    if radius < 0:
        return 0
    if radius > 9:
        return 9
    return radius


def _format_bbox_size(size_zyx: Tuple[int, int, int]) -> str:
    return f"{int(size_zyx[0])} x {int(size_zyx[1])} x {int(size_zyx[2])}"


def _format_bbox_center(center_zyx: Tuple[float, float, float]) -> str:
    return f"({center_zyx[0]:.2f}, {center_zyx[1]:.2f}, {center_zyx[2]:.2f})"


def _normalize_bbox_label(value: object) -> BoundingBoxLabel:
    if not isinstance(value, str):
        return "train"
    normalized = value.strip().lower()
    if normalized in _BBOX_LABEL_OPTIONS:
        return cast(BoundingBoxLabel, normalized)
    return "train"


def _primary_selected_bbox_id(selected_ids: Tuple[str, ...]) -> Optional[str]:
    if len(selected_ids) == 1:
        return selected_ids[0]
    return None


@dataclass(frozen=True)
class BoundingBoxRow:
    box_id: str
    label: BoundingBoxLabel
    size_text: str
    center_text: str


@dataclass
class BottomPanelState:
    zoom: float = 1.0
    cursor_position: Tuple[int, int, int] = (0, 0, 0)
    hover_position: Optional[Tuple[int, int, int]] = None
    hover_label: Optional[int] = None
    picked_position: Optional[Tuple[int, int, int]] = None
    picked_label: Optional[int] = None
    active_level: int = 0
    level_scale: int = 1
    pyramid_levels: int = 1
    annotation_enabled: bool = False
    bounding_box_mode_enabled: bool = False
    annotation_tool: AnnotationTool = "brush"
    active_label: int = 1
    brush_radius: BrushRadius = 0
    eraser_target: str = ""
    flood_fill_target: int = 1
    undo_depth: int = 0
    redo_depth: int = 0
    contrast_data_range: Optional[Tuple[float, float]] = None
    contrast_window: Optional[Tuple[float, float]] = None
    auto_level_enabled: bool = True
    manual_level: int = 0
    manual_level_max: int = 0
    bbox_rows: Tuple[BoundingBoxRow, ...] = tuple()
    bbox_selected_ids: Tuple[str, ...] = tuple()
    # Backward-compatible mirror of the first selected id.
    bbox_selected_id: Optional[str] = None
    bbox_selected_label: Optional[BoundingBoxLabel] = None
    learning_training_running: bool = False
    learning_inference_navigation_only: bool = False


class BottomPanel(QWidget):
    _CONTRAST_STEPS = 1_000
    _CONTRAST_MAX_STEP = _CONTRAST_STEPS - 1

    def __init__(self) -> None:
        super().__init__()
        self.state = BottomPanelState()
        self._file_path: Optional[str] = None
        self._on_open: Optional[Callable[[], None]] = None
        self._on_open_semantic: Optional[Callable[[], None]] = None
        self._on_open_instance: Optional[Callable[[], None]] = None
        self._on_save_segmentation: Optional[Callable[[], None]] = None
        self._on_cursor: Optional[Callable[[Tuple[int, int, int]], None]] = None
        self._on_zoom: Optional[Callable[[float], None]] = None
        self._on_auto_level_mode_changed: Optional[Callable[[bool], None]] = None
        self._on_manual_level_requested: Optional[Callable[[int], None]] = None
        self._on_contrast_window_changed: Optional[Callable[[float, float], None]] = None
        self._on_annotation_mode_changed: Optional[Callable[[bool], None]] = None
        self._on_bounding_box_mode_changed: Optional[Callable[[bool], None]] = None
        self._on_annotation_tool_changed: Optional[Callable[[AnnotationTool], None]] = None
        self._on_active_label_changed: Optional[Callable[[int], None]] = None
        self._on_next_available_label_requested: Optional[Callable[[], None]] = None
        self._on_brush_radius_changed: Optional[Callable[[BrushRadius], None]] = None
        self._on_eraser_target_changed: Optional[Callable[[str], None]] = None
        self._on_flood_fill_target_changed: Optional[Callable[[int], None]] = None
        self._on_flood_fill_requested: Optional[Callable[[int], None]] = None
        self._on_undo_requested: Optional[Callable[[], None]] = None
        self._on_redo_requested: Optional[Callable[[], None]] = None
        self._on_open_bounding_boxes_requested: Optional[Callable[[], None]] = None
        self._on_save_bounding_boxes_requested: Optional[Callable[[], None]] = None
        self._on_build_dataset_from_bboxes_requested: Optional[Callable[[], None]] = None
        self._on_load_model_requested: Optional[Callable[[], None]] = None
        self._on_save_model_requested: Optional[Callable[[], None]] = None
        self._on_segment_inference_requested: Optional[Callable[[], None]] = None
        self._on_stop_inference_requested: Optional[Callable[[], None]] = None
        self._on_train_model_requested: Optional[Callable[[], None]] = None
        self._on_stop_training_requested: Optional[Callable[[], None]] = None
        self._on_median_filter_selected_requested: Optional[Callable[[], None]] = None
        self._on_erosion_selected_requested: Optional[Callable[[], None]] = None
        self._on_dilation_selected_requested: Optional[Callable[[], None]] = None
        self._on_erase_bbox_segmentation_requested: Optional[Callable[[], None]] = None
        self._on_bounding_boxes_selected: Optional[Callable[[Tuple[str, ...]], None]] = None
        self._on_bounding_box_double_clicked: Optional[Callable[[str], None]] = None
        self._on_bounding_boxes_delete_requested: Optional[Callable[[Tuple[str, ...]], None]] = None
        self._on_bounding_boxes_label_changed: Optional[
            Callable[[Tuple[str, ...], BoundingBoxLabel], None]
        ] = None
        # Backward-compatible single-id callbacks kept until main-window migration.
        self._on_bounding_box_selected: Optional[Callable[[Optional[str]], None]] = None
        self._on_bounding_box_delete_requested: Optional[Callable[[str], None]] = None
        self._on_bounding_box_label_changed: Optional[Callable[[str, BoundingBoxLabel], None]] = None
        self._interaction_tools_enabled = False
        self._level_controls_enabled = True
        self._inference_navigation_only_mode = False
        self._segment_inference_enabled_requested = True
        self._train_model_enabled_requested = True
        self._stop_training_enabled_requested = False
        self._stop_inference_enabled_requested = False
        self._undo_enabled_requested = False
        self._redo_enabled_requested = False

        self._open_button = QPushButton("Open")
        self._open_semantic_button = QPushButton("Open Semantic")
        self._open_instance_button = QPushButton("Open Instance")
        self._save_segmentation_button = QPushButton("Save Segmentation")
        self._annotation_toggle = QCheckBox("Manual Segmentation")
        self._bounding_box_mode_toggle = QCheckBox("Bounding Box Tool")
        self._annotation_tool_label = QLabel("Tool")
        self._annotation_tool_combo = QComboBox()
        self._annotation_tool_combo.addItem("Brush (Ctrl+B)", "brush")
        self._annotation_tool_combo.addItem("Eraser (Ctrl+E)", "eraser")
        self._annotation_tool_combo.addItem("Flood Fill (Ctrl+F)", "flood_filler")
        annotation_shortcuts_hint = (
            "Shortcuts: Ctrl+B brush, Ctrl+E eraser, Ctrl+F flood fill. "
            "Using them auto-enables Manual Segmentation."
        )
        self._annotation_tool_label.setToolTip(annotation_shortcuts_hint)
        self._annotation_tool_combo.setToolTip(annotation_shortcuts_hint)
        self._annotation_toggle.setToolTip(
            "Enable manual segmentation. Ctrl+B/Ctrl+E/Ctrl+F also enables it automatically."
        )
        self._active_label_label = QLabel("Label")
        self._active_label_spin = QSpinBox()
        self._active_label_spin.setPrefix("L:")
        self._active_label_spin.setRange(0, 2_147_483_647)
        self._active_label_spin.setValue(1)
        self._brush_radius_label = QLabel("Brush Radius")
        self._brush_radius_spin = QSpinBox()
        self._brush_radius_spin.setRange(0, 9)
        self._brush_radius_spin.setValue(0)
        self._eraser_target_label = QLabel("Erase ID")
        self._eraser_target_edit = QLineEdit()
        self._eraser_target_edit.setPlaceholderText("All")
        self._eraser_target_edit.setClearButtonEnabled(True)
        self._flood_fill_target_label = QLabel("Fill ID")
        self._flood_fill_target_spin = QSpinBox()
        self._flood_fill_target_spin.setPrefix("L:")
        self._flood_fill_target_spin.setRange(0, 2_147_483_647)
        self._flood_fill_target_spin.setValue(1)
        self._flood_fill_button = QPushButton("Flood Fill")
        self._undo_button = QPushButton("Undo")
        self._redo_button = QPushButton("Redo")
        self._next_available_button = QPushButton("Next Available")
        self._cursor_label = QLabel("Cursor")
        self._cursor_z = QSpinBox()
        self._cursor_y = QSpinBox()
        self._cursor_x = QSpinBox()
        self._hover_label = QLabel("Hover")
        self._hover_value = QLabel("Z:- Y:- X:- | ID:-")
        self._picked_label = QLabel("Selected")
        self._picked_value = QLabel("Z:- Y:- X:- | ID:-")
        self._cursor_z.setPrefix("Z:")
        self._cursor_y.setPrefix("Y:")
        self._cursor_x.setPrefix("X:")
        self._cursor_z.setRange(0, 0)
        self._cursor_y.setRange(0, 0)
        self._cursor_x.setRange(0, 0)
        self._zoom_spin = QDoubleSpinBox()
        self._zoom_spin.setRange(0.1, 20.0)
        self._zoom_spin.setSingleStep(0.1)
        self._zoom_spin.setValue(1.0)
        self._auto_level_checkbox = QCheckBox("Auto Level")
        self._auto_level_checkbox.setChecked(True)
        self._manual_level_label = QLabel("Manual Level")
        self._manual_level_spin = QSpinBox()
        self._manual_level_spin.setPrefix("L:")
        self._manual_level_spin.setRange(0, 0)
        self._manual_level_spin.setValue(0)
        self._contrast_min_label = QLabel("Window Min")
        self._contrast_min_slider = QSlider(Qt.Orientation.Horizontal)
        self._contrast_min_slider.setRange(0, self._CONTRAST_MAX_STEP)
        self._contrast_min_slider.setSingleStep(1)
        self._contrast_min_slider.setPageStep(10)
        self._contrast_max_label = QLabel("Window Max")
        self._contrast_max_slider = QSlider(Qt.Orientation.Horizontal)
        self._contrast_max_slider.setRange(0, self._CONTRAST_MAX_STEP)
        self._contrast_max_slider.setSingleStep(1)
        self._contrast_max_slider.setPageStep(10)
        self._contrast_min_value = QLabel("Min: -")
        self._contrast_max_value = QLabel("Max: -")
        self._pyramid_status = QLabel("Pyramid: -")
        self._level_status = QLabel("Level: L0 (x1)")
        self._bbox_table = QTableWidget(0, 4)
        self._bbox_table.setHorizontalHeaderLabels(["ID", "Label", "Size (dz, dy, dx)", "Center (z, y, x)"])
        self._bbox_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._bbox_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._bbox_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._bbox_table.setAlternatingRowColors(True)
        self._bbox_table.verticalHeader().setVisible(False)
        bbox_header = self._bbox_table.horizontalHeader()
        bbox_header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        bbox_header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        bbox_header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        bbox_header.setSectionResizeMode(3, QHeaderView.Stretch)
        self._bbox_label_label = QLabel("Selected Label")
        self._bbox_label_combo = QComboBox()
        self._bbox_label_combo.addItem("Train", "train")
        self._bbox_label_combo.addItem("Validation", "validation")
        self._bbox_label_combo.addItem("Inference", "inference")
        self._bbox_label_combo.setEnabled(False)
        self._bbox_label_label.setEnabled(False)
        self._open_bounding_boxes_button = QPushButton("Open Boxes...")
        self._save_bounding_boxes_button = QPushButton("Save Boxes...")
        self._build_dataset_from_bboxes_button = QPushButton("Build Dataset from Bbox")
        self._load_model_button = QPushButton("Load Model")
        self._save_model_button = QPushButton("Save Model")
        # Backward-compatible alias kept for existing tests/callers.
        self._instantiate_model_button = self._load_model_button
        self._segment_inference_button = QPushButton("Segment Inference Bbox")
        self._stop_inference_button = QPushButton("Stop Inference")
        self._stop_inference_button.setEnabled(False)
        self._train_model_button = QPushButton("Train Model on Dataset")
        self._stop_training_button = QPushButton("Stop Training")
        self._stop_training_button.setEnabled(False)
        self._learning_training_status = QLabel("Training: Idle")
        self._delete_bbox_button = QPushButton("Delete Selected")
        self._delete_bbox_button.setEnabled(False)
        self._median_filter_selected_button = QPushButton("Median Filter Selected")
        self._erosion_selected_button = QPushButton("Erosion Selected")
        self._dilation_selected_button = QPushButton("Dilation Selected")
        self._erase_bbox_segmentation_button = QPushButton("Erase Bbox Segmentation")

        self._open_button.clicked.connect(self._handle_open)
        self._open_semantic_button.clicked.connect(self._handle_open_semantic)
        self._open_instance_button.clicked.connect(self._handle_open_instance)
        self._save_segmentation_button.clicked.connect(self._handle_save_segmentation)
        self._annotation_toggle.toggled.connect(self._handle_annotation_mode_changed)
        self._bounding_box_mode_toggle.toggled.connect(self._handle_bounding_box_mode_changed)
        self._annotation_tool_combo.currentIndexChanged.connect(self._handle_annotation_tool_changed)
        self._active_label_spin.valueChanged.connect(self._handle_active_label_changed)
        self._brush_radius_spin.valueChanged.connect(self._handle_brush_radius_changed)
        self._eraser_target_edit.editingFinished.connect(self._handle_eraser_target_changed)
        self._flood_fill_target_spin.valueChanged.connect(self._handle_flood_fill_target_changed)
        self._flood_fill_button.clicked.connect(self._handle_flood_fill_requested)
        self._undo_button.clicked.connect(self._handle_undo_requested)
        self._redo_button.clicked.connect(self._handle_redo_requested)
        self._next_available_button.clicked.connect(self._handle_next_available_label_requested)
        self._cursor_z.valueChanged.connect(self._handle_cursor)
        self._cursor_y.valueChanged.connect(self._handle_cursor)
        self._cursor_x.valueChanged.connect(self._handle_cursor)
        self._zoom_spin.valueChanged.connect(self._handle_zoom)
        self._auto_level_checkbox.toggled.connect(self._handle_auto_level_mode_changed)
        manual_level_line_edit = self._manual_level_spin.lineEdit()
        if manual_level_line_edit is not None:
            manual_level_line_edit.returnPressed.connect(self._handle_manual_level_requested)
        else:
            self._manual_level_spin.editingFinished.connect(self._handle_manual_level_requested)
        self._contrast_min_slider.valueChanged.connect(self._handle_contrast_min_changed)
        self._contrast_max_slider.valueChanged.connect(self._handle_contrast_max_changed)
        self._open_bounding_boxes_button.clicked.connect(self._handle_open_bounding_boxes_requested)
        self._save_bounding_boxes_button.clicked.connect(self._handle_save_bounding_boxes_requested)
        self._build_dataset_from_bboxes_button.clicked.connect(
            self._handle_build_dataset_from_bboxes_requested
        )
        self._load_model_button.clicked.connect(
            self._handle_load_model_requested
        )
        self._save_model_button.clicked.connect(
            self._handle_save_model_requested
        )
        self._segment_inference_button.clicked.connect(
            self._handle_segment_inference_requested
        )
        self._stop_inference_button.clicked.connect(
            self._handle_stop_inference_requested
        )
        self._train_model_button.clicked.connect(
            self._handle_train_model_requested
        )
        self._stop_training_button.clicked.connect(
            self._handle_stop_training_requested
        )
        self._median_filter_selected_button.clicked.connect(
            self._handle_median_filter_selected_requested
        )
        self._erosion_selected_button.clicked.connect(
            self._handle_erosion_selected_requested
        )
        self._dilation_selected_button.clicked.connect(
            self._handle_dilation_selected_requested
        )
        self._erase_bbox_segmentation_button.clicked.connect(
            self._handle_erase_bbox_segmentation_requested
        )
        self._bbox_table.itemSelectionChanged.connect(self._handle_bounding_box_selection_changed)
        self._bbox_table.itemDoubleClicked.connect(self._handle_bounding_box_double_clicked)
        self._delete_bbox_button.clicked.connect(self._handle_bounding_box_delete_requested)
        self._bbox_label_combo.currentIndexChanged.connect(self._handle_bounding_box_label_changed)

        files_group = QGroupBox("Files")
        files_layout = QVBoxLayout()
        files_layout.addWidget(self._open_button)
        files_layout.addWidget(self._open_semantic_button)
        files_layout.addWidget(self._open_instance_button)
        files_layout.addWidget(self._save_segmentation_button)
        files_group.setLayout(files_layout)

        navigation_group = QGroupBox("Navigation")
        navigation_layout = QGridLayout()
        cursor_row = QWidget()
        cursor_row_layout = QHBoxLayout()
        cursor_row_layout.setContentsMargins(0, 0, 0, 0)
        cursor_row_layout.addWidget(self._cursor_z)
        cursor_row_layout.addWidget(self._cursor_y)
        cursor_row_layout.addWidget(self._cursor_x)
        cursor_row.setLayout(cursor_row_layout)
        navigation_layout.addWidget(self._cursor_label, 0, 0)
        navigation_layout.addWidget(cursor_row, 0, 1)
        navigation_layout.addWidget(self._hover_label, 1, 0)
        navigation_layout.addWidget(self._hover_value, 1, 1)
        navigation_layout.addWidget(self._picked_label, 2, 0)
        navigation_layout.addWidget(self._picked_value, 2, 1)
        navigation_layout.addWidget(QLabel("Zoom"), 3, 0)
        navigation_layout.addWidget(self._zoom_spin, 3, 1)
        navigation_layout.addWidget(self._auto_level_checkbox, 4, 0, 1, 2)
        navigation_layout.addWidget(self._manual_level_label, 5, 0)
        navigation_layout.addWidget(self._manual_level_spin, 5, 1)
        navigation_layout.addWidget(self._pyramid_status, 6, 0, 1, 2)
        navigation_layout.addWidget(self._level_status, 7, 0, 1, 2)
        navigation_layout.setColumnStretch(1, 1)
        navigation_group.setLayout(navigation_layout)

        contrast_group = QGroupBox("Contrast")
        contrast_layout = QGridLayout()
        contrast_layout.addWidget(self._contrast_min_label, 0, 0)
        contrast_layout.addWidget(self._contrast_min_slider, 0, 1)
        contrast_layout.addWidget(self._contrast_min_value, 0, 2)
        contrast_layout.addWidget(self._contrast_max_label, 1, 0)
        contrast_layout.addWidget(self._contrast_max_slider, 1, 1)
        contrast_layout.addWidget(self._contrast_max_value, 1, 2)
        contrast_layout.setColumnStretch(1, 1)
        contrast_group.setLayout(contrast_layout)

        annotation_group = QGroupBox("Annotation")
        annotation_layout = QGridLayout()
        annotation_layout.addWidget(self._annotation_toggle, 0, 0, 1, 2)
        annotation_layout.addWidget(self._annotation_tool_label, 1, 0)
        annotation_layout.addWidget(self._annotation_tool_combo, 1, 1)
        annotation_layout.addWidget(self._active_label_label, 2, 0)
        annotation_layout.addWidget(self._active_label_spin, 2, 1)
        annotation_layout.addWidget(self._brush_radius_label, 3, 0)
        annotation_layout.addWidget(self._brush_radius_spin, 3, 1)
        annotation_layout.addWidget(self._eraser_target_label, 4, 0)
        annotation_layout.addWidget(self._eraser_target_edit, 4, 1)
        annotation_layout.addWidget(self._flood_fill_target_label, 5, 0)
        annotation_layout.addWidget(self._flood_fill_target_spin, 5, 1)
        annotation_layout.addWidget(self._flood_fill_button, 6, 0, 1, 2)
        annotation_layout.addWidget(self._next_available_button, 7, 0, 1, 2)
        annotation_layout.setColumnStretch(1, 1)
        annotation_group.setLayout(annotation_layout)

        bounding_boxes_group = QGroupBox("Bounding Boxes")
        bounding_boxes_layout = QVBoxLayout()
        bounding_boxes_layout.addWidget(self._bounding_box_mode_toggle)
        bounding_boxes_layout.addWidget(self._bbox_table)
        bbox_label_row = QWidget()
        bbox_label_layout = QHBoxLayout()
        bbox_label_layout.setContentsMargins(0, 0, 0, 0)
        bbox_label_layout.addWidget(self._bbox_label_label)
        bbox_label_layout.addWidget(self._bbox_label_combo)
        bbox_label_layout.addStretch(1)
        bbox_label_row.setLayout(bbox_label_layout)
        bounding_boxes_layout.addWidget(bbox_label_row)
        bbox_controls_row = QWidget()
        bbox_controls_layout = QHBoxLayout()
        bbox_controls_layout.setContentsMargins(0, 0, 0, 0)
        bbox_controls_layout.addWidget(self._open_bounding_boxes_button)
        bbox_controls_layout.addWidget(self._save_bounding_boxes_button)
        bbox_controls_layout.addWidget(self._build_dataset_from_bboxes_button)
        bbox_controls_layout.addWidget(self._delete_bbox_button)
        bbox_controls_layout.addStretch(1)
        bbox_controls_row.setLayout(bbox_controls_layout)
        bounding_boxes_layout.addWidget(bbox_controls_row)
        bbox_processing_row = QWidget()
        bbox_processing_layout = QHBoxLayout()
        bbox_processing_layout.setContentsMargins(0, 0, 0, 0)
        bbox_processing_layout.addWidget(self._median_filter_selected_button)
        bbox_processing_layout.addWidget(self._erosion_selected_button)
        bbox_processing_layout.addWidget(self._dilation_selected_button)
        bbox_processing_layout.addWidget(self._erase_bbox_segmentation_button)
        bbox_processing_layout.addStretch(1)
        bbox_processing_row.setLayout(bbox_processing_layout)
        bounding_boxes_layout.addWidget(bbox_processing_row)
        bounding_boxes_group.setLayout(bounding_boxes_layout)

        learning_group = QGroupBox("Learning")
        learning_layout = QVBoxLayout()
        learning_controls_row = QWidget()
        learning_controls_layout = QHBoxLayout()
        learning_controls_layout.setContentsMargins(0, 0, 0, 0)
        learning_controls_layout.addWidget(self._load_model_button)
        learning_controls_layout.addWidget(self._save_model_button)
        learning_controls_layout.addWidget(self._segment_inference_button)
        learning_controls_layout.addWidget(self._stop_inference_button)
        learning_controls_layout.addWidget(self._train_model_button)
        learning_controls_layout.addWidget(self._stop_training_button)
        learning_controls_layout.addWidget(self._learning_training_status)
        learning_controls_layout.addStretch(1)
        learning_controls_row.setLayout(learning_controls_layout)
        learning_layout.addWidget(learning_controls_row)
        learning_group.setLayout(learning_layout)

        history_group = QGroupBox("History")
        history_layout = QVBoxLayout()
        history_layout.addWidget(self._undo_button)
        history_layout.addWidget(self._redo_button)
        history_group.setLayout(history_layout)

        root_layout = QVBoxLayout()
        root_layout.addWidget(files_group)
        root_layout.addSpacing(8)
        root_layout.addWidget(navigation_group)
        root_layout.addSpacing(8)
        root_layout.addWidget(contrast_group)
        root_layout.addSpacing(8)
        root_layout.addWidget(annotation_group)
        root_layout.addSpacing(8)
        root_layout.addWidget(bounding_boxes_group)
        root_layout.addSpacing(8)
        root_layout.addWidget(learning_group)
        root_layout.addSpacing(8)
        root_layout.addWidget(history_group)
        root_layout.addStretch(1)
        root_layout.setContentsMargins(8, 8, 8, 8)
        self.setLayout(root_layout)
        self._annotation_controls_enabled = False
        self.set_annotation_controls_enabled(False)
        self._set_contrast_sliders_from_window()
        self._update_contrast_labels()
        self._update_file_controls_state()
        self._update_interaction_tool_controls_state()
        self._update_bounding_box_controls_state()
        self._update_learning_controls_state()
        self._update_history_controls_state()

    def set_file_path(self, path: str) -> None:
        self._file_path = path

    def file_path(self) -> Optional[str]:
        return self._file_path

    def set_zoom(self, zoom: float) -> None:
        self.state.zoom = zoom
        self._zoom_spin.blockSignals(True)
        self._zoom_spin.setValue(zoom)
        self._zoom_spin.blockSignals(False)

    def set_level_mode(
        self,
        *,
        auto_enabled: bool,
        manual_level: int,
        max_level: int,
    ) -> None:
        try:
            normalized_max = max(0, int(max_level))
        except (TypeError, ValueError):
            normalized_max = 0
        try:
            requested_manual = int(manual_level)
        except (TypeError, ValueError):
            requested_manual = 0
        normalized_manual = max(0, min(requested_manual, normalized_max))
        self.state.auto_level_enabled = bool(auto_enabled)
        self.state.manual_level = normalized_manual
        self.state.manual_level_max = normalized_max
        self._auto_level_checkbox.blockSignals(True)
        self._auto_level_checkbox.setChecked(self.state.auto_level_enabled)
        self._auto_level_checkbox.blockSignals(False)
        self._manual_level_spin.blockSignals(True)
        self._manual_level_spin.setRange(0, normalized_max)
        self._manual_level_spin.setValue(normalized_manual)
        self._manual_level_spin.blockSignals(False)
        self._update_level_controls_state()

    def set_level_controls_enabled(self, enabled: bool) -> None:
        self._level_controls_enabled = bool(enabled)
        self._update_level_controls_state()

    def set_contrast_range(self, data_range: Optional[Tuple[float, float]]) -> None:
        if data_range is None:
            self.state.contrast_data_range = None
            self.state.contrast_window = None
            self._set_contrast_sliders_from_window()
            self._update_contrast_labels()
            self._update_contrast_controls_state()
            return
        data_min = float(data_range[0])
        data_max = float(data_range[1])
        if not np.isfinite(data_min) or not np.isfinite(data_max) or data_min > data_max:
            self.state.contrast_data_range = None
            self.state.contrast_window = None
            self._set_contrast_sliders_from_window()
            self._update_contrast_labels()
            self._update_contrast_controls_state()
            return
        self.state.contrast_data_range = (data_min, data_max)
        if self.state.contrast_window is None:
            self.state.contrast_window = (data_min, data_max)
        else:
            self.set_contrast_window(self.state.contrast_window)
            return
        self._set_contrast_sliders_from_window()
        self._update_contrast_labels()
        self._update_contrast_controls_state()

    def set_contrast_window(self, window: Optional[Tuple[float, float]]) -> None:
        data_range = self.state.contrast_data_range
        if data_range is None:
            self.state.contrast_window = None
            self._set_contrast_sliders_from_window()
            self._update_contrast_labels()
            self._update_contrast_controls_state()
            return
        data_min, data_max = data_range
        if window is None:
            normalized = (data_min, data_max)
        else:
            requested_min = float(window[0])
            requested_max = float(window[1])
            if not np.isfinite(requested_min) or not np.isfinite(requested_max):
                normalized = (data_min, data_max)
            else:
                requested_min = max(data_min, min(requested_min, data_max))
                requested_max = max(data_min, min(requested_max, data_max))
                if data_max > data_min and requested_min >= requested_max:
                    requested_min, requested_max = data_min, data_max
                normalized = (requested_min, requested_max)
        self.state.contrast_window = normalized
        self._set_contrast_sliders_from_window()
        self._update_contrast_labels()
        self._update_contrast_controls_state()

    def contrast_window(self) -> Optional[Tuple[float, float]]:
        return self.state.contrast_window

    def set_annotation_mode(self, enabled: bool) -> None:
        self.state.annotation_enabled = bool(enabled)
        self._annotation_toggle.blockSignals(True)
        self._annotation_toggle.setChecked(self.state.annotation_enabled)
        self._annotation_toggle.blockSignals(False)
        self._update_eraser_controls_state()

    def set_bounding_box_mode(self, enabled: bool) -> None:
        self.state.bounding_box_mode_enabled = bool(enabled)
        self._bounding_box_mode_toggle.blockSignals(True)
        self._bounding_box_mode_toggle.setChecked(self.state.bounding_box_mode_enabled)
        self._bounding_box_mode_toggle.blockSignals(False)

    def set_annotation_controls_enabled(self, enabled: bool) -> None:
        self._annotation_controls_enabled = bool(enabled)
        editable_controls_enabled = bool(
            self._annotation_controls_enabled and not self._inference_navigation_only_mode
        )
        self._active_label_label.setEnabled(editable_controls_enabled)
        self._active_label_spin.setEnabled(editable_controls_enabled)
        self._brush_radius_label.setEnabled(editable_controls_enabled)
        self._brush_radius_spin.setEnabled(editable_controls_enabled)
        self._next_available_button.setEnabled(editable_controls_enabled)
        self._update_interaction_tool_controls_state()
        self._update_eraser_controls_state()

    def set_interaction_tools_enabled(self, enabled: bool) -> None:
        self._interaction_tools_enabled = bool(enabled)
        self._update_interaction_tool_controls_state()

    def set_annotation_tool(self, tool: AnnotationTool) -> None:
        normalized = str(tool).strip().lower()
        if normalized not in ("brush", "eraser", "flood_filler"):
            normalized = "brush"
        self.state.annotation_tool = cast(AnnotationTool, normalized)
        index = self._annotation_tool_combo.findData(normalized)
        if index < 0:
            index = 0
        self._annotation_tool_combo.blockSignals(True)
        self._annotation_tool_combo.setCurrentIndex(index)
        self._annotation_tool_combo.blockSignals(False)
        self._update_eraser_controls_state()

    def annotation_tool(self) -> AnnotationTool:
        return self.state.annotation_tool

    def set_active_label_bounds(self, minimum: int, maximum: int) -> None:
        minimum = int(minimum)
        maximum = int(maximum)
        if maximum < minimum:
            maximum = minimum
        self._active_label_spin.setRange(minimum, maximum)

    def set_active_label(self, label: int) -> None:
        self.state.active_label = int(label)
        self._active_label_spin.blockSignals(True)
        self._active_label_spin.setValue(self.state.active_label)
        self._active_label_spin.blockSignals(False)

    def active_label(self) -> int:
        return int(self._active_label_spin.value())

    def set_brush_radius(self, brush_radius: BrushRadius) -> None:
        normalized = _normalize_brush_radius(brush_radius)
        self.state.brush_radius = normalized
        self._brush_radius_spin.blockSignals(True)
        self._brush_radius_spin.setValue(normalized)
        self._brush_radius_spin.blockSignals(False)

    def brush_radius(self) -> BrushRadius:
        return int(self.state.brush_radius)

    def set_eraser_target(self, target: str) -> None:
        normalized = str(target).strip()
        self.state.eraser_target = normalized
        self._eraser_target_edit.blockSignals(True)
        self._eraser_target_edit.setText(normalized)
        self._eraser_target_edit.blockSignals(False)

    def eraser_target(self) -> str:
        return self.state.eraser_target

    def set_flood_fill_target_bounds(self, minimum: int, maximum: int) -> None:
        minimum = int(minimum)
        maximum = int(maximum)
        if maximum < minimum:
            maximum = minimum
        self._flood_fill_target_spin.setRange(minimum, maximum)

    def set_flood_fill_target(self, label: int) -> None:
        self.state.flood_fill_target = int(label)
        self._flood_fill_target_spin.blockSignals(True)
        self._flood_fill_target_spin.setValue(self.state.flood_fill_target)
        self._flood_fill_target_spin.blockSignals(False)

    def flood_fill_target(self) -> int:
        return int(self._flood_fill_target_spin.value())

    def set_undo_state(self, *, depth: int, enabled: bool) -> None:
        normalized_depth = max(0, int(depth))
        self.state.undo_depth = normalized_depth
        self._undo_enabled_requested = bool(enabled)
        self._undo_button.setText(f"Undo ({normalized_depth})")
        self._update_history_controls_state()

    def set_redo_state(self, *, depth: int, enabled: bool) -> None:
        normalized_depth = max(0, int(depth))
        self.state.redo_depth = normalized_depth
        self._redo_enabled_requested = bool(enabled)
        self._redo_button.setText(f"Redo ({normalized_depth})")
        self._update_history_controls_state()

    def set_cursor_range(self, shape: Tuple[int, int, int]) -> None:
        z_max = max(0, shape[0] - 1)
        y_max = max(0, shape[1] - 1)
        x_max = max(0, shape[2] - 1)
        self._cursor_z.setRange(0, z_max)
        self._cursor_y.setRange(0, y_max)
        self._cursor_x.setRange(0, x_max)

    def set_cursor_position(self, indices: Tuple[int, int, int]) -> None:
        self.state.cursor_position = indices
        z, y, x = indices
        self._cursor_z.blockSignals(True)
        self._cursor_y.blockSignals(True)
        self._cursor_x.blockSignals(True)
        self._cursor_z.setValue(z)
        self._cursor_y.setValue(y)
        self._cursor_x.setValue(x)
        self._cursor_z.blockSignals(False)
        self._cursor_y.blockSignals(False)
        self._cursor_x.blockSignals(False)

    def set_hover_info(
        self,
        indices: Optional[Tuple[int, int, int]],
        label: Optional[int],
    ) -> None:
        self.state.hover_position = indices
        self.state.hover_label = None if label is None else int(label)
        if indices is None:
            self._hover_value.setText("Z:- Y:- X:- | ID:-")
            return
        z, y, x = indices
        label_text = "-" if label is None else str(int(label))
        self._hover_value.setText(f"Z:{z} Y:{y} X:{x} | ID:{label_text}")

    def set_picked_info(
        self,
        indices: Optional[Tuple[int, int, int]],
        label: Optional[int],
    ) -> None:
        self.state.picked_position = indices
        self.state.picked_label = None if label is None else int(label)
        if indices is None:
            self._picked_value.setText("Z:- Y:- X:- | ID:-")
            self._update_eraser_controls_state()
            return
        z, y, x = indices
        label_text = "-" if label is None else str(int(label))
        self._picked_value.setText(f"Z:{z} Y:{y} X:{x} | ID:{label_text}")
        self._update_eraser_controls_state()

    def set_pyramid_levels(self, levels: int, kind: str = "Raw") -> None:
        levels = max(1, int(levels))
        self.state.pyramid_levels = levels
        self._pyramid_status.setText(f"{kind} levels computed: {levels}")

    def set_active_level(self, level: int, scale: int = 1) -> None:
        level = max(0, int(level))
        scale = max(1, int(scale))
        self.state.active_level = level
        self.state.level_scale = scale
        self.set_active_levels(
            axial=(level, scale),
            coronal=(level, scale),
            sagittal=(level, scale),
        )

    def set_active_levels(
        self,
        axial: Tuple[int, int],
        coronal: Tuple[int, int],
        sagittal: Tuple[int, int],
        *,
        forced: bool = False,
    ) -> None:
        ax_level = max(0, int(axial[0]))
        ax_scale = max(1, int(axial[1]))
        co_level = max(0, int(coronal[0]))
        co_scale = max(1, int(coronal[1]))
        sa_level = max(0, int(sagittal[0]))
        sa_scale = max(1, int(sagittal[1]))
        self.state.active_level = ax_level
        self.state.level_scale = ax_scale
        status_text = (
            f"Levels: Ax L{ax_level} (x{ax_scale}) | Co L{co_level} (x{co_scale}) | Sa L{sa_level} (x{sa_scale})"
        )
        if forced:
            status_text += " | Manual (forced)"
        self._level_status.setText(status_text)

    def set_bounding_boxes(self, boxes: Iterable[BoundingBox]) -> None:
        rows = []
        for box in boxes:
            if not isinstance(box, BoundingBox):
                continue
            rows.append(
                BoundingBoxRow(
                    box_id=box.id,
                    label=box.label,
                    size_text=_format_bbox_size(box.size_voxels),
                    center_text=_format_bbox_center(box.center_index_space),
                )
            )
        self.state.bbox_rows = tuple(rows)
        valid_ids = {row.box_id for row in self.state.bbox_rows}
        self.state.bbox_selected_ids = tuple(
            box_id for box_id in self.state.bbox_selected_ids if box_id in valid_ids
        )
        self.state.bbox_selected_id = _primary_selected_bbox_id(self.state.bbox_selected_ids)

        self._bbox_table.blockSignals(True)
        self._bbox_table.setRowCount(len(self.state.bbox_rows))
        for row_index, row in enumerate(self.state.bbox_rows):
            id_item = QTableWidgetItem(row.box_id)
            label_item = QTableWidgetItem(row.label)
            size_item = QTableWidgetItem(row.size_text)
            center_item = QTableWidgetItem(row.center_text)
            self._bbox_table.setItem(row_index, 0, id_item)
            self._bbox_table.setItem(row_index, 1, label_item)
            self._bbox_table.setItem(row_index, 2, size_item)
            self._bbox_table.setItem(row_index, 3, center_item)
        self._bbox_table.blockSignals(False)

        self.set_selected_bounding_boxes(self.state.bbox_selected_ids)
        self._update_bounding_box_controls_state()

    def set_selected_bounding_boxes(self, box_ids: Iterable[str]) -> None:
        seen_ids = set()
        for raw_box_id in tuple(box_ids):
            normalized = str(raw_box_id).strip()
            if not normalized or normalized in seen_ids:
                continue
            seen_ids.add(normalized)

        row_indices = []
        selected_ids = []
        for idx, row in enumerate(self.state.bbox_rows):
            if row.box_id not in seen_ids:
                continue
            row_indices.append(idx)
            selected_ids.append(row.box_id)

        self.state.bbox_selected_ids = tuple(selected_ids)
        self.state.bbox_selected_id = _primary_selected_bbox_id(self.state.bbox_selected_ids)
        self.state.bbox_selected_label = self._shared_bbox_label_for_ids(self.state.bbox_selected_ids)
        self._bbox_table.blockSignals(True)
        self._bbox_table.clearSelection()
        for row_index in row_indices:
            self._bbox_table.selectRow(row_index)
        self._bbox_table.blockSignals(False)
        self._set_selected_bbox_label_value(self.state.bbox_selected_label)
        self._update_bounding_box_controls_state()

    def set_selected_bounding_box(self, box_id: Optional[str]) -> None:
        if box_id is None:
            self.set_selected_bounding_boxes(tuple())
            return
        normalized = str(box_id).strip()
        if not normalized:
            self.set_selected_bounding_boxes(tuple())
            return
        self.set_selected_bounding_boxes((normalized,))

    def selected_bounding_boxes(self) -> Tuple[str, ...]:
        return self.state.bbox_selected_ids

    def selected_bounding_box(self) -> Optional[str]:
        return _primary_selected_bbox_id(self.state.bbox_selected_ids)

    def selected_bounding_box_label(self) -> Optional[BoundingBoxLabel]:
        return self.state.bbox_selected_label

    def set_train_model_enabled(self, enabled: bool) -> None:
        self._train_model_enabled_requested = bool(enabled)
        self._update_learning_controls_state()

    def set_segment_inference_enabled(self, enabled: bool) -> None:
        self._segment_inference_enabled_requested = bool(enabled)
        self._update_learning_controls_state()

    def set_stop_training_enabled(self, enabled: bool) -> None:
        self._stop_training_enabled_requested = bool(enabled)
        self._update_learning_controls_state()

    def set_stop_inference_enabled(self, enabled: bool) -> None:
        self._stop_inference_enabled_requested = bool(enabled)
        self._update_learning_controls_state()

    def set_inference_navigation_only_mode(self, enabled: bool) -> None:
        self._inference_navigation_only_mode = bool(enabled)
        self.state.learning_inference_navigation_only = self._inference_navigation_only_mode
        self._update_file_controls_state()
        self.set_annotation_controls_enabled(self._annotation_controls_enabled)
        self._update_bounding_box_controls_state()
        self._update_learning_controls_state()
        self._update_history_controls_state()

    def set_learning_training_running(self, running: bool) -> None:
        self.state.learning_training_running = bool(running)
        status = "Running" if self.state.learning_training_running else "Idle"
        self._learning_training_status.setText(f"Training: {status}")

    def learning_training_running(self) -> bool:
        return bool(self.state.learning_training_running)

    def on_open_requested(self, callback: Callable[[], None]) -> None:
        self._on_open = callback

    def on_open_semantic_requested(self, callback: Callable[[], None]) -> None:
        self._on_open_semantic = callback

    def on_open_instance_requested(self, callback: Callable[[], None]) -> None:
        self._on_open_instance = callback

    def on_save_segmentation_requested(self, callback: Callable[[], None]) -> None:
        self._on_save_segmentation = callback

    def on_cursor_changed(self, callback: Callable[[Tuple[int, int, int]], None]) -> None:
        self._on_cursor = callback

    def on_zoom_changed(self, callback: Callable[[float], None]) -> None:
        self._on_zoom = callback

    def on_auto_level_mode_changed(self, callback: Callable[[bool], None]) -> None:
        self._on_auto_level_mode_changed = callback

    def on_manual_level_requested(self, callback: Callable[[int], None]) -> None:
        self._on_manual_level_requested = callback

    def on_contrast_window_changed(self, callback: Callable[[float, float], None]) -> None:
        self._on_contrast_window_changed = callback

    def on_annotation_mode_changed(self, callback: Callable[[bool], None]) -> None:
        self._on_annotation_mode_changed = callback

    def on_bounding_box_mode_changed(self, callback: Callable[[bool], None]) -> None:
        self._on_bounding_box_mode_changed = callback

    def on_annotation_tool_changed(self, callback: Callable[[AnnotationTool], None]) -> None:
        self._on_annotation_tool_changed = callback

    def on_active_label_changed(self, callback: Callable[[int], None]) -> None:
        self._on_active_label_changed = callback

    def on_next_available_label_requested(self, callback: Callable[[], None]) -> None:
        self._on_next_available_label_requested = callback

    def on_brush_radius_changed(self, callback: Callable[[BrushRadius], None]) -> None:
        self._on_brush_radius_changed = callback

    def on_eraser_target_changed(self, callback: Callable[[str], None]) -> None:
        self._on_eraser_target_changed = callback

    def on_flood_fill_requested(self, callback: Callable[[int], None]) -> None:
        self._on_flood_fill_requested = callback

    def on_flood_fill_target_changed(self, callback: Callable[[int], None]) -> None:
        self._on_flood_fill_target_changed = callback

    def on_undo_requested(self, callback: Callable[[], None]) -> None:
        self._on_undo_requested = callback

    def on_redo_requested(self, callback: Callable[[], None]) -> None:
        self._on_redo_requested = callback

    def on_bounding_boxes_selected(
        self,
        callback: Callable[[Tuple[str, ...]], None],
    ) -> None:
        self._on_bounding_boxes_selected = callback

    def on_bounding_box_selected(
        self,
        callback: Callable[[Optional[str]], None],
    ) -> None:
        self._on_bounding_box_selected = callback

    def on_bounding_box_double_clicked(self, callback: Callable[[str], None]) -> None:
        self._on_bounding_box_double_clicked = callback

    def on_open_bounding_boxes_requested(self, callback: Callable[[], None]) -> None:
        self._on_open_bounding_boxes_requested = callback

    def on_save_bounding_boxes_requested(self, callback: Callable[[], None]) -> None:
        self._on_save_bounding_boxes_requested = callback

    def on_build_dataset_from_bboxes_requested(self, callback: Callable[[], None]) -> None:
        self._on_build_dataset_from_bboxes_requested = callback

    def on_load_model_requested(self, callback: Callable[[], None]) -> None:
        self._on_load_model_requested = callback

    def on_save_model_requested(self, callback: Callable[[], None]) -> None:
        self._on_save_model_requested = callback

    # Backward-compatible alias kept for existing tests/callers.
    def on_instantiate_model_requested(self, callback: Callable[[], None]) -> None:
        self.on_load_model_requested(callback)

    def on_segment_inference_requested(self, callback: Callable[[], None]) -> None:
        self._on_segment_inference_requested = callback

    def on_stop_inference_requested(self, callback: Callable[[], None]) -> None:
        self._on_stop_inference_requested = callback

    def on_train_model_requested(self, callback: Callable[[], None]) -> None:
        self._on_train_model_requested = callback

    def on_stop_training_requested(self, callback: Callable[[], None]) -> None:
        self._on_stop_training_requested = callback

    def on_median_filter_selected_requested(self, callback: Callable[[], None]) -> None:
        self._on_median_filter_selected_requested = callback

    def on_erosion_selected_requested(self, callback: Callable[[], None]) -> None:
        self._on_erosion_selected_requested = callback

    def on_dilation_selected_requested(self, callback: Callable[[], None]) -> None:
        self._on_dilation_selected_requested = callback

    def on_erase_bbox_segmentation_requested(self, callback: Callable[[], None]) -> None:
        self._on_erase_bbox_segmentation_requested = callback

    def on_bounding_boxes_delete_requested(
        self,
        callback: Callable[[Tuple[str, ...]], None],
    ) -> None:
        self._on_bounding_boxes_delete_requested = callback

    def on_bounding_box_delete_requested(self, callback: Callable[[str], None]) -> None:
        self._on_bounding_box_delete_requested = callback

    def on_bounding_boxes_label_changed(
        self,
        callback: Callable[[Tuple[str, ...], BoundingBoxLabel], None],
    ) -> None:
        self._on_bounding_boxes_label_changed = callback

    def on_bounding_box_label_changed(
        self,
        callback: Callable[[str, BoundingBoxLabel], None],
    ) -> None:
        self._on_bounding_box_label_changed = callback

    def _handle_open(self) -> None:
        if self._on_open:
            self._on_open()

    def _handle_open_semantic(self) -> None:
        if self._on_open_semantic:
            self._on_open_semantic()

    def _handle_open_instance(self) -> None:
        if self._on_open_instance:
            self._on_open_instance()

    def _handle_save_segmentation(self) -> None:
        if self._on_save_segmentation:
            self._on_save_segmentation()

    def _handle_zoom(self, value: float) -> None:
        self.state.zoom = value
        if self._on_zoom:
            self._on_zoom(value)

    def _handle_auto_level_mode_changed(self, enabled: bool) -> None:
        self.state.auto_level_enabled = bool(enabled)
        self._update_level_controls_state()
        if self._on_auto_level_mode_changed is not None:
            self._on_auto_level_mode_changed(self.state.auto_level_enabled)

    def _handle_manual_level_requested(self) -> None:
        if not self._manual_level_spin.isEnabled():
            return
        normalized = self._normalize_manual_level(int(self._manual_level_spin.value()))
        self.state.manual_level = normalized
        self._manual_level_spin.blockSignals(True)
        self._manual_level_spin.setValue(normalized)
        self._manual_level_spin.blockSignals(False)
        if self._on_manual_level_requested is not None:
            self._on_manual_level_requested(normalized)

    def _handle_contrast_min_changed(self, value: int) -> None:
        data_range = self.state.contrast_data_range
        if data_range is None:
            return
        min_step = int(value)
        max_step = int(self._contrast_max_slider.value())
        if self._can_adjust_contrast() and min_step >= max_step:
            min_step = max(0, max_step - 1)
            self._contrast_min_slider.blockSignals(True)
            self._contrast_min_slider.setValue(min_step)
            self._contrast_min_slider.blockSignals(False)
        self._set_contrast_window_from_steps(min_step, max_step, emit_change=True)

    def _handle_contrast_max_changed(self, value: int) -> None:
        data_range = self.state.contrast_data_range
        if data_range is None:
            return
        min_step = int(self._contrast_min_slider.value())
        max_step = int(value)
        if self._can_adjust_contrast() and max_step <= min_step:
            max_step = min(self._CONTRAST_MAX_STEP, min_step + 1)
            self._contrast_max_slider.blockSignals(True)
            self._contrast_max_slider.setValue(max_step)
            self._contrast_max_slider.blockSignals(False)
        self._set_contrast_window_from_steps(min_step, max_step, emit_change=True)

    def _handle_annotation_mode_changed(self, enabled: bool) -> None:
        self.state.annotation_enabled = bool(enabled)
        self._update_eraser_controls_state()
        if self._on_annotation_mode_changed:
            self._on_annotation_mode_changed(self.state.annotation_enabled)

    def _handle_annotation_tool_changed(self, _index: int) -> None:
        value = self._annotation_tool_combo.currentData()
        if value not in ("brush", "eraser", "flood_filler"):
            value = "brush"
        self.state.annotation_tool = cast(AnnotationTool, value)
        self._update_eraser_controls_state()
        if self._on_annotation_tool_changed:
            self._on_annotation_tool_changed(self.state.annotation_tool)

    def _handle_bounding_box_mode_changed(self, enabled: bool) -> None:
        self.state.bounding_box_mode_enabled = bool(enabled)
        if self._on_bounding_box_mode_changed:
            self._on_bounding_box_mode_changed(self.state.bounding_box_mode_enabled)

    def _handle_active_label_changed(self, value: int) -> None:
        self.state.active_label = int(value)
        if self._on_active_label_changed:
            self._on_active_label_changed(self.state.active_label)

    def _handle_next_available_label_requested(self) -> None:
        if self._on_next_available_label_requested:
            self._on_next_available_label_requested()

    def _handle_brush_radius_changed(self, value: int) -> None:
        self.state.brush_radius = _normalize_brush_radius(value)
        if self._on_brush_radius_changed:
            self._on_brush_radius_changed(self.state.brush_radius)

    def _handle_eraser_target_changed(self) -> None:
        value = self._eraser_target_edit.text().strip()
        self.state.eraser_target = value
        if self._on_eraser_target_changed:
            self._on_eraser_target_changed(value)

    def _handle_flood_fill_requested(self) -> None:
        self.state.flood_fill_target = int(self._flood_fill_target_spin.value())
        if self._on_flood_fill_requested:
            self._on_flood_fill_requested(self.state.flood_fill_target)

    def _handle_flood_fill_target_changed(self, value: int) -> None:
        self.state.flood_fill_target = int(value)
        if self._on_flood_fill_target_changed:
            self._on_flood_fill_target_changed(self.state.flood_fill_target)

    def _handle_undo_requested(self) -> None:
        if self._on_undo_requested:
            self._on_undo_requested()

    def _handle_redo_requested(self) -> None:
        if self._on_redo_requested:
            self._on_redo_requested()

    def _selected_bbox_ids_from_table_selection(self) -> Tuple[str, ...]:
        selected_row_indices = sorted({item.row() for item in self._bbox_table.selectedItems()})
        selected_ids = []
        for row in selected_row_indices:
            item = self._bbox_table.item(row, 0)
            if item is None:
                continue
            box_id = item.text().strip()
            if not box_id:
                continue
            selected_ids.append(box_id)
        return tuple(selected_ids)

    def _handle_bounding_box_selection_changed(self) -> None:
        selected_ids = self._selected_bbox_ids_from_table_selection()
        self.state.bbox_selected_ids = selected_ids
        selected_id = _primary_selected_bbox_id(selected_ids)
        self.state.bbox_selected_id = selected_id
        self.state.bbox_selected_label = self._shared_bbox_label_for_ids(selected_ids)
        self._set_selected_bbox_label_value(self.state.bbox_selected_label)
        self._update_bounding_box_controls_state()
        if self._on_bounding_boxes_selected:
            self._on_bounding_boxes_selected(selected_ids)
        if self._on_bounding_box_selected:
            self._on_bounding_box_selected(selected_id)

    def _handle_bounding_box_double_clicked(self, item: QTableWidgetItem) -> None:
        if self._on_bounding_box_double_clicked is None:
            return
        if not isinstance(item, QTableWidgetItem):
            return
        row_index = int(item.row())
        if row_index < 0 or row_index >= self._bbox_table.rowCount():
            return
        id_item = self._bbox_table.item(row_index, 0)
        if id_item is None:
            return
        box_id = id_item.text().strip()
        if not box_id:
            return
        self._on_bounding_box_double_clicked(box_id)

    def _handle_bounding_box_delete_requested(self) -> None:
        selected_ids = self.state.bbox_selected_ids
        if not selected_ids:
            return
        if self._on_bounding_boxes_delete_requested:
            self._on_bounding_boxes_delete_requested(selected_ids)
        selected_id = _primary_selected_bbox_id(selected_ids)
        if self._on_bounding_box_delete_requested and selected_id is not None:
            self._on_bounding_box_delete_requested(selected_id)

    def _handle_open_bounding_boxes_requested(self) -> None:
        if self._on_open_bounding_boxes_requested:
            self._on_open_bounding_boxes_requested()

    def _handle_save_bounding_boxes_requested(self) -> None:
        if self._on_save_bounding_boxes_requested:
            self._on_save_bounding_boxes_requested()

    def _handle_build_dataset_from_bboxes_requested(self) -> None:
        if self._on_build_dataset_from_bboxes_requested:
            self._on_build_dataset_from_bboxes_requested()

    def _handle_load_model_requested(self) -> None:
        if self._on_load_model_requested:
            self._on_load_model_requested()

    def _handle_save_model_requested(self) -> None:
        if self._on_save_model_requested:
            self._on_save_model_requested()

    # Backward-compatible alias kept for existing tests/callers.
    def _handle_instantiate_model_requested(self) -> None:
        self._handle_load_model_requested()

    def _handle_segment_inference_requested(self) -> None:
        if self._on_segment_inference_requested:
            self._on_segment_inference_requested()

    def _handle_stop_inference_requested(self) -> None:
        if self._on_stop_inference_requested:
            self._on_stop_inference_requested()

    def _handle_train_model_requested(self) -> None:
        if self._on_train_model_requested:
            self._on_train_model_requested()

    def _handle_stop_training_requested(self) -> None:
        if self._on_stop_training_requested:
            self._on_stop_training_requested()

    def _handle_median_filter_selected_requested(self) -> None:
        if self._on_median_filter_selected_requested:
            self._on_median_filter_selected_requested()

    def _handle_erosion_selected_requested(self) -> None:
        if self._on_erosion_selected_requested:
            self._on_erosion_selected_requested()

    def _handle_dilation_selected_requested(self) -> None:
        if self._on_dilation_selected_requested:
            self._on_dilation_selected_requested()

    def _handle_erase_bbox_segmentation_requested(self) -> None:
        if self._on_erase_bbox_segmentation_requested:
            self._on_erase_bbox_segmentation_requested()

    def _handle_bounding_box_label_changed(self, _index: int) -> None:
        selected_ids = self.state.bbox_selected_ids
        if not selected_ids:
            return
        selected_label = _normalize_bbox_label(self._bbox_label_combo.currentData())
        if self.state.bbox_selected_label is not None and self.state.bbox_selected_label == selected_label:
            return
        self.state.bbox_selected_label = selected_label
        if self._on_bounding_boxes_label_changed:
            self._on_bounding_boxes_label_changed(selected_ids, selected_label)
        selected_id = _primary_selected_bbox_id(selected_ids)
        if self._on_bounding_box_label_changed and selected_id is not None:
            self._on_bounding_box_label_changed(selected_id, selected_label)

    def _update_eraser_controls_state(self) -> None:
        eraser_active = (
            self._annotation_controls_enabled
            and not self._inference_navigation_only_mode
            and self.state.annotation_enabled
            and self.state.annotation_tool == "eraser"
        )
        self._eraser_target_label.setEnabled(eraser_active)
        self._eraser_target_edit.setEnabled(eraser_active)
        flood_fill_active = (
            self._annotation_controls_enabled
            and not self._inference_navigation_only_mode
            and self.state.annotation_enabled
            and self.state.annotation_tool == "flood_filler"
        )
        self._flood_fill_target_label.setEnabled(flood_fill_active)
        self._flood_fill_target_spin.setEnabled(flood_fill_active)
        self._flood_fill_button.setEnabled(flood_fill_active and self.state.picked_position is not None)
        self._update_bounding_box_controls_state()

    def _update_interaction_tool_controls_state(self) -> None:
        enabled = bool(self._interaction_tools_enabled)
        tool_controls_enabled = bool(enabled and not self._inference_navigation_only_mode)
        self._annotation_toggle.setEnabled(tool_controls_enabled)
        self._annotation_tool_label.setEnabled(tool_controls_enabled)
        self._annotation_tool_combo.setEnabled(tool_controls_enabled)
        self._bounding_box_mode_toggle.setEnabled(tool_controls_enabled)
        self._update_level_controls_state()
        self._update_contrast_controls_state()

    def _update_level_controls_state(self) -> None:
        enabled = bool(self._interaction_tools_enabled and self._level_controls_enabled)
        manual_enabled = enabled and (not self.state.auto_level_enabled)
        self._auto_level_checkbox.setEnabled(enabled)
        self._manual_level_label.setEnabled(manual_enabled)
        self._manual_level_spin.setEnabled(manual_enabled)

    def _update_contrast_controls_state(self) -> None:
        enabled = bool(self._interaction_tools_enabled)
        sliders_enabled = enabled and self._can_adjust_contrast()
        self._contrast_min_label.setEnabled(enabled)
        self._contrast_max_label.setEnabled(enabled)
        self._contrast_min_slider.setEnabled(sliders_enabled)
        self._contrast_max_slider.setEnabled(sliders_enabled)
        self._contrast_min_value.setEnabled(enabled)
        self._contrast_max_value.setEnabled(enabled)

    def _update_bounding_box_controls_state(self) -> None:
        editing_locked = bool(self._inference_navigation_only_mode)
        has_boxes = len(self.state.bbox_rows) > 0
        self._open_bounding_boxes_button.setEnabled(not editing_locked)
        self._save_bounding_boxes_button.setEnabled(has_boxes and not editing_locked)
        self._build_dataset_from_bboxes_button.setEnabled(has_boxes and not editing_locked)
        has_selected_box = bool(self.state.bbox_selected_ids)
        bbox_editing_enabled = bool(has_selected_box and not editing_locked)
        self._delete_bbox_button.setEnabled(bbox_editing_enabled)
        self._bbox_label_label.setEnabled(bbox_editing_enabled)
        self._bbox_label_combo.setEnabled(bbox_editing_enabled)
        self._median_filter_selected_button.setEnabled(not editing_locked)
        self._erosion_selected_button.setEnabled(not editing_locked)
        self._dilation_selected_button.setEnabled(not editing_locked)
        self._erase_bbox_segmentation_button.setEnabled(not editing_locked)

    def _update_file_controls_state(self) -> None:
        enabled = bool(not self._inference_navigation_only_mode)
        self._open_button.setEnabled(enabled)
        self._open_semantic_button.setEnabled(enabled)
        self._open_instance_button.setEnabled(enabled)
        self._save_segmentation_button.setEnabled(enabled)

    def _update_learning_controls_state(self) -> None:
        editing_locked = bool(self._inference_navigation_only_mode)
        self._load_model_button.setEnabled(not editing_locked)
        self._save_model_button.setEnabled(not editing_locked)
        self._segment_inference_button.setEnabled(
            bool(self._segment_inference_enabled_requested and not editing_locked)
        )
        self._train_model_button.setEnabled(
            bool(self._train_model_enabled_requested and not editing_locked)
        )
        self._stop_training_button.setEnabled(
            bool(self._stop_training_enabled_requested and not editing_locked)
        )
        self._stop_inference_button.setEnabled(self._stop_inference_enabled_requested)

    def _update_history_controls_state(self) -> None:
        editing_locked = bool(self._inference_navigation_only_mode)
        self._undo_button.setEnabled(
            bool(self._undo_enabled_requested and not editing_locked and self.state.undo_depth > 0)
        )
        self._redo_button.setEnabled(
            bool(self._redo_enabled_requested and not editing_locked and self.state.redo_depth > 0)
        )

    def _bbox_label_for_id(self, box_id: Optional[str]) -> Optional[BoundingBoxLabel]:
        if box_id is None:
            return None
        for row in self.state.bbox_rows:
            if row.box_id == box_id:
                return row.label
        return None

    def _shared_bbox_label_for_ids(self, box_ids: Tuple[str, ...]) -> Optional[BoundingBoxLabel]:
        if not box_ids:
            return None
        labels = []
        for box_id in box_ids:
            label = self._bbox_label_for_id(box_id)
            if label is None:
                return None
            labels.append(label)
        first_label = labels[0]
        for label in labels[1:]:
            if label != first_label:
                return None
        return first_label

    def _set_selected_bbox_label_value(self, label: Optional[BoundingBoxLabel]) -> None:
        if label is None:
            index = -1
        else:
            index = self._bbox_label_combo.findData(label)
            if index < 0:
                index = -1
        self._bbox_label_combo.blockSignals(True)
        self._bbox_label_combo.setCurrentIndex(index)
        self._bbox_label_combo.blockSignals(False)

    def _handle_cursor(self, _value: int) -> None:
        indices = (self._cursor_z.value(), self._cursor_y.value(), self._cursor_x.value())
        self.state.cursor_position = indices
        if self._on_cursor:
            self._on_cursor(indices)

    def _normalize_manual_level(self, level: int) -> int:
        max_level = max(0, int(self.state.manual_level_max))
        normalized = int(level)
        if normalized < 0:
            return 0
        if normalized > max_level:
            return max_level
        return normalized

    def _can_adjust_contrast(self) -> bool:
        data_range = self.state.contrast_data_range
        if data_range is None:
            return False
        return float(data_range[1]) > float(data_range[0])

    def _set_contrast_sliders_from_window(self) -> None:
        min_step = 0
        max_step = self._CONTRAST_MAX_STEP
        window = self.state.contrast_window
        if window is not None and self.state.contrast_data_range is not None:
            min_step = self._value_to_contrast_step(float(window[0]))
            max_step = self._value_to_contrast_step(float(window[1]))
            if self._can_adjust_contrast() and min_step >= max_step:
                if max_step >= self._CONTRAST_MAX_STEP:
                    min_step = max(0, max_step - 1)
                else:
                    max_step = min(self._CONTRAST_MAX_STEP, min_step + 1)
        self._contrast_min_slider.blockSignals(True)
        self._contrast_max_slider.blockSignals(True)
        self._contrast_min_slider.setValue(min_step)
        self._contrast_max_slider.setValue(max_step)
        self._contrast_min_slider.blockSignals(False)
        self._contrast_max_slider.blockSignals(False)

    def _set_contrast_window_from_steps(
        self,
        min_step: int,
        max_step: int,
        *,
        emit_change: bool,
    ) -> None:
        data_range = self.state.contrast_data_range
        if data_range is None:
            self.state.contrast_window = None
            self._update_contrast_labels()
            return
        data_min, data_max = data_range
        normalized_min_step = max(0, min(self._CONTRAST_MAX_STEP, int(min_step)))
        normalized_max_step = max(0, min(self._CONTRAST_MAX_STEP, int(max_step)))
        if self._can_adjust_contrast():
            if normalized_min_step >= normalized_max_step:
                normalized_min_step = max(0, normalized_max_step - 1)
                if normalized_min_step >= normalized_max_step:
                    normalized_max_step = min(self._CONTRAST_MAX_STEP, normalized_min_step + 1)
        value_min = self._contrast_step_to_value(normalized_min_step)
        value_max = self._contrast_step_to_value(normalized_max_step)
        if self._can_adjust_contrast() and value_min >= value_max:
            value_min, value_max = data_min, data_max
            normalized_min_step = self._value_to_contrast_step(value_min)
            normalized_max_step = self._value_to_contrast_step(value_max)
        self.state.contrast_window = (value_min, value_max)
        self._set_contrast_sliders_from_window()
        self._update_contrast_labels()
        if emit_change and self._on_contrast_window_changed is not None:
            self._on_contrast_window_changed(value_min, value_max)

    def _contrast_step_to_value(self, step: int) -> float:
        data_range = self.state.contrast_data_range
        if data_range is None:
            return 0.0
        data_min, data_max = data_range
        if data_max <= data_min:
            return float(data_min)
        ratio = float(max(0, min(self._CONTRAST_MAX_STEP, int(step)))) / float(self._CONTRAST_MAX_STEP)
        return float(data_min + ratio * (data_max - data_min))

    def _value_to_contrast_step(self, value: float) -> int:
        data_range = self.state.contrast_data_range
        if data_range is None:
            return 0
        data_min, data_max = data_range
        if data_max <= data_min:
            return 0
        clamped = max(data_min, min(float(value), data_max))
        ratio = (clamped - data_min) / (data_max - data_min)
        return int(round(ratio * float(self._CONTRAST_MAX_STEP)))

    def _update_contrast_labels(self) -> None:
        window = self.state.contrast_window
        if window is None:
            self._contrast_min_value.setText("Min: -")
            self._contrast_max_value.setText("Max: -")
            return
        self._contrast_min_value.setText(f"Min: {window[0]:.6g}")
        self._contrast_max_value.setText(f"Max: {window[1]:.6g}")
