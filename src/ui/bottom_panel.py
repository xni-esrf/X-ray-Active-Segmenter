from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Optional, Tuple, cast

import numpy as np

from ..bbox import BoundingBox, BoundingBoxLabel

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QSizePolicy,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .bottom_panel_subpanels import (
    AnnotationPanel,
    BoundingBoxesPanel,
    ContrastPanel,
    FilesPanel,
    HistoryPanel,
    LearningPanel,
    NavigationPanel,
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
    tool_label_text: str = "1"
    brush_radius: BrushRadius = 0
    undo_depth: int = 0
    redo_depth: int = 0
    contrast_data_range: Optional[Tuple[float, float]] = None
    contrast_window: Optional[Tuple[float, float]] = None
    segmentation_opacity: float = 0.3
    auto_level_enabled: bool = True
    manual_level: int = 0
    manual_level_max: int = 0
    view_layout_mode: str = "all"
    bbox_rows: Tuple[BoundingBoxRow, ...] = tuple()
    bbox_selected_ids: Tuple[str, ...] = tuple()
    # Backward-compatible mirror of the first selected id.
    bbox_selected_id: Optional[str] = None
    bbox_selected_label: Optional[BoundingBoxLabel] = None
    learning_training_running: bool = False
    learning_inference_navigation_only: bool = False


class BottomPanel(QWidget):
    _ZOOM_MIN = 0.1
    _ZOOM_MAX = 1.0
    _CONTRAST_STEPS = 1_000
    _CONTRAST_MAX_STEP = _CONTRAST_STEPS - 1
    _SEGMENTATION_OPACITY_DEFAULT = 0.3
    _SEGMENTATION_OPACITY_MAX_STEP = 100
    _COMPACT_BUTTON_MAX_WIDTH = 170
    _COMPACT_INPUT_MAX_WIDTH = 130
    _COMPACT_SLIDER_MAX_WIDTH = 180
    _COMPACT_BBOX_TABLE_MIN_WIDTH = 176
    _SECTION_COMPACT_WIDTH = 336

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
        self._on_segmentation_opacity_changed: Optional[Callable[[float], None]] = None
        self._on_view_layout_mode_changed: Optional[Callable[[str], None]] = None
        self._on_annotation_mode_changed: Optional[Callable[[bool], None]] = None
        self._on_bounding_box_mode_changed: Optional[Callable[[bool], None]] = None
        self._on_annotation_tool_changed: Optional[Callable[[AnnotationTool], None]] = None
        self._on_tool_label_changed: Optional[Callable[[str], None]] = None
        self._on_next_available_label_requested: Optional[Callable[[], None]] = None
        self._on_brush_radius_changed: Optional[Callable[[BrushRadius], None]] = None
        self._on_flood_fill_requested: Optional[Callable[[int], None]] = None
        self._on_undo_requested: Optional[Callable[[], None]] = None
        self._on_redo_requested: Optional[Callable[[], None]] = None
        self._on_open_bounding_boxes_requested: Optional[Callable[[], None]] = None
        self._on_save_bounding_boxes_requested: Optional[Callable[[], None]] = None
        self._on_load_model_requested: Optional[Callable[[], None]] = None
        self._on_save_model_requested: Optional[Callable[[], None]] = None
        self._on_segment_inference_requested: Optional[Callable[[], None]] = None
        self._on_segment_inference_headless_close_requested: Optional[
            Callable[[], None]
        ] = None
        self._on_stop_inference_requested: Optional[Callable[[], None]] = None
        self._on_train_model_requested: Optional[Callable[[], None]] = None
        self._on_train_model_headless_close_requested: Optional[Callable[[], None]] = None
        self._on_stop_training_requested: Optional[Callable[[], None]] = None
        self._on_change_training_parameters_requested: Optional[Callable[[], None]] = None
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

        self._files_panel = FilesPanel()
        self._open_button = self._files_panel.open_button
        self._open_semantic_button = self._files_panel.open_semantic_button
        self._open_instance_button = self._files_panel.open_instance_button
        self._save_segmentation_button = self._files_panel.save_segmentation_button
        self._files_panel.on_open_requested(self._handle_open)
        self._files_panel.on_open_semantic_requested(self._handle_open_semantic)
        self._files_panel.on_open_instance_requested(self._handle_open_instance)
        self._files_panel.on_save_segmentation_requested(self._handle_save_segmentation)

        self._history_panel = HistoryPanel()
        self._undo_button = self._history_panel.undo_button
        self._redo_button = self._history_panel.redo_button
        self._history_panel.on_undo_requested(self._handle_undo_requested)
        self._history_panel.on_redo_requested(self._handle_redo_requested)

        self._learning_panel = LearningPanel()
        self._load_model_button = self._learning_panel.load_model_button
        self._save_model_button = self._learning_panel.save_model_button
        self._segment_inference_button = self._learning_panel.segment_inference_button
        self._stop_inference_button = self._learning_panel.stop_inference_button
        self._train_model_button = self._learning_panel.train_model_button
        self._stop_training_button = self._learning_panel.stop_training_button
        self._change_training_parameters_button = (
            self._learning_panel.change_training_parameters_button
        )
        self._learning_training_status = self._learning_panel.training_status
        self._learning_panel.on_load_model_requested(self._handle_load_model_requested)
        self._learning_panel.on_save_model_requested(self._handle_save_model_requested)
        self._learning_panel.on_segment_inference_requested(
            self._handle_segment_inference_requested
        )
        self._learning_panel.on_segment_inference_headless_close_requested(
            self._handle_segment_inference_headless_close_requested
        )
        self._learning_panel.on_stop_inference_requested(
            self._handle_stop_inference_requested
        )
        self._learning_panel.on_train_model_requested(self._handle_train_model_requested)
        self._learning_panel.on_train_model_headless_close_requested(
            self._handle_train_model_headless_close_requested
        )
        self._learning_panel.on_stop_training_requested(
            self._handle_stop_training_requested
        )
        self._learning_panel.on_change_training_parameters_requested(
            self._handle_change_training_parameters_requested
        )

        self._navigation_panel = NavigationPanel(
            zoom_min=self._ZOOM_MIN,
            zoom_max=self._ZOOM_MAX,
        )
        self._cursor_label = self._navigation_panel.cursor_label
        self._cursor_z = self._navigation_panel.cursor_z
        self._cursor_y = self._navigation_panel.cursor_y
        self._cursor_x = self._navigation_panel.cursor_x
        self._hover_label = self._navigation_panel.hover_label
        self._hover_value = self._navigation_panel.hover_value
        self._picked_label = self._navigation_panel.picked_label
        self._picked_value = self._navigation_panel.picked_value
        self._zoom_spin = self._navigation_panel.zoom_spin
        self._auto_level_checkbox = self._navigation_panel.auto_level_checkbox
        self._manual_level_label = self._navigation_panel.manual_level_label
        self._manual_level_spin = self._navigation_panel.manual_level_spin
        self._view_layout_label = self._navigation_panel.view_layout_label
        self._view_layout_button_group = (
            self._navigation_panel.view_layout_button_group
        )
        self._view_layout_all_radio = self._navigation_panel.view_layout_all_radio
        self._view_layout_axial_radio = self._navigation_panel.view_layout_axial_radio
        self._view_layout_coronal_radio = (
            self._navigation_panel.view_layout_coronal_radio
        )
        self._view_layout_sagittal_radio = (
            self._navigation_panel.view_layout_sagittal_radio
        )
        self._pyramid_status = self._navigation_panel.pyramid_status
        self._level_status = self._navigation_panel.level_status
        self._navigation_panel.on_cursor_changed(self._handle_cursor)
        self._navigation_panel.on_zoom_changed(self._handle_zoom)
        self._navigation_panel.on_auto_level_mode_changed(
            self._handle_auto_level_mode_changed
        )
        self._navigation_panel.on_manual_level_requested(
            self._handle_manual_level_requested
        )
        self._navigation_panel.on_view_layout_mode_changed(
            self._handle_view_layout_mode_changed
        )

        self._contrast_panel = ContrastPanel(
            contrast_max_step=self._CONTRAST_MAX_STEP,
            segmentation_opacity_max_step=self._SEGMENTATION_OPACITY_MAX_STEP,
        )
        self._contrast_min_label = self._contrast_panel.contrast_min_label
        self._contrast_min_slider = self._contrast_panel.contrast_min_slider
        self._contrast_max_label = self._contrast_panel.contrast_max_label
        self._contrast_max_slider = self._contrast_panel.contrast_max_slider
        self._contrast_min_value = self._contrast_panel.contrast_min_value
        self._contrast_max_value = self._contrast_panel.contrast_max_value
        self._segmentation_opacity_label = (
            self._contrast_panel.segmentation_opacity_label
        )
        self._segmentation_opacity_slider = (
            self._contrast_panel.segmentation_opacity_slider
        )
        self._segmentation_opacity_value = (
            self._contrast_panel.segmentation_opacity_value
        )
        self._contrast_panel.on_contrast_min_changed(
            self._handle_contrast_min_changed
        )
        self._contrast_panel.on_contrast_max_changed(
            self._handle_contrast_max_changed
        )
        self._contrast_panel.on_segmentation_opacity_changed(
            self._handle_segmentation_opacity_changed
        )

        self._annotation_panel = AnnotationPanel()
        self._annotation_toggle = self._annotation_panel.annotation_toggle
        self._annotation_tool_label = self._annotation_panel.annotation_tool_label
        self._annotation_tool_combo = self._annotation_panel.annotation_tool_combo
        self._tool_label_label = self._annotation_panel.tool_label_label
        self._tool_label_edit = self._annotation_panel.tool_label_edit
        self._brush_radius_label = self._annotation_panel.brush_radius_label
        self._brush_radius_spin = self._annotation_panel.brush_radius_spin
        self._flood_fill_button = self._annotation_panel.flood_fill_button
        self._next_available_button = self._annotation_panel.next_available_button
        self._annotation_panel.on_annotation_mode_changed(
            self._handle_annotation_mode_changed
        )
        self._annotation_panel.on_annotation_tool_changed(
            self._handle_annotation_tool_changed
        )
        self._annotation_panel.on_tool_label_changed(self._handle_tool_label_changed)
        self._annotation_panel.on_brush_radius_changed(
            self._handle_brush_radius_changed
        )
        self._annotation_panel.on_flood_fill_requested(
            self._handle_flood_fill_requested
        )
        self._annotation_panel.on_next_available_label_requested(
            self._handle_next_available_label_requested
        )

        self._bounding_boxes_panel = BoundingBoxesPanel()
        self._bounding_boxes_group = self._bounding_boxes_panel
        self._bounding_box_mode_toggle = (
            self._bounding_boxes_panel.bounding_box_mode_toggle
        )
        self._bbox_table = self._bounding_boxes_panel.bbox_table
        self._bbox_label_label = self._bounding_boxes_panel.bbox_label_label
        self._bbox_label_combo = self._bounding_boxes_panel.bbox_label_combo
        self._open_bounding_boxes_button = (
            self._bounding_boxes_panel.open_bounding_boxes_button
        )
        self._save_bounding_boxes_button = (
            self._bounding_boxes_panel.save_bounding_boxes_button
        )
        self._delete_bbox_button = self._bounding_boxes_panel.delete_bbox_button
        self._median_filter_selected_button = (
            self._bounding_boxes_panel.median_filter_selected_button
        )
        self._erosion_selected_button = (
            self._bounding_boxes_panel.erosion_selected_button
        )
        self._dilation_selected_button = (
            self._bounding_boxes_panel.dilation_selected_button
        )
        self._erase_bbox_segmentation_button = (
            self._bounding_boxes_panel.erase_bbox_segmentation_button
        )
        self._bounding_boxes_panel.on_bounding_box_mode_changed(
            self._handle_bounding_box_mode_changed
        )
        self._bounding_boxes_panel.on_selection_changed(
            self._handle_bounding_box_selection_changed
        )
        self._bounding_boxes_panel.on_item_double_clicked(
            self._handle_bounding_box_double_clicked
        )
        self._bounding_boxes_panel.on_open_requested(
            self._handle_open_bounding_boxes_requested
        )
        self._bounding_boxes_panel.on_save_requested(
            self._handle_save_bounding_boxes_requested
        )
        self._bounding_boxes_panel.on_delete_requested(
            self._handle_bounding_box_delete_requested
        )
        self._bounding_boxes_panel.on_label_changed(
            self._handle_bounding_box_label_changed
        )
        self._bounding_boxes_panel.on_median_filter_requested(
            self._handle_median_filter_selected_requested
        )
        self._bounding_boxes_panel.on_erosion_requested(
            self._handle_erosion_selected_requested
        )
        self._bounding_boxes_panel.on_dilation_requested(
            self._handle_dilation_selected_requested
        )
        self._bounding_boxes_panel.on_erase_segmentation_requested(
            self._handle_erase_bbox_segmentation_requested
        )

        files_group = self._files_panel

        navigation_group = self._navigation_panel

        contrast_group = self._contrast_panel

        annotation_group = self._annotation_panel

        bounding_boxes_group = self._bounding_boxes_panel

        learning_group = self._learning_panel

        history_group = self._history_panel

        root_layout = QVBoxLayout()
        root_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        root_layout.addWidget(files_group)
        root_layout.addSpacing(8)
        root_layout.addWidget(navigation_group)
        root_layout.addSpacing(8)
        root_layout.addWidget(contrast_group)
        root_layout.addSpacing(8)
        root_layout.addWidget(annotation_group)
        root_layout.addSpacing(8)
        # Keep this section responsive to right-panel width so bbox metadata can expand.
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
        self.set_segmentation_opacity(self.state.segmentation_opacity)
        self._update_file_controls_state()
        self._update_interaction_tool_controls_state()
        self._update_bounding_box_controls_state()
        self._update_learning_controls_state()
        self._update_history_controls_state()
        self._apply_compact_right_panel_widths()
        self._update_bbox_table_width()
        self._apply_compact_section_widths()

    def _apply_compact_section_widths(self) -> None:
        for group in self.findChildren(QGroupBox):
            if group is getattr(self, "_bounding_boxes_group", None):
                group.setMinimumWidth(self._SECTION_COMPACT_WIDTH)
                group.setMaximumWidth(16_777_215)
                group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
                continue
            group.setMaximumWidth(self._SECTION_COMPACT_WIDTH)
            group.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)

    def _apply_compact_right_panel_widths(self) -> None:
        # Allow section containers (not individual controls) to use extra width
        # when the right splitter pane is expanded.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        file_buttons = (
            self._open_button,
            self._open_semantic_button,
            self._open_instance_button,
            self._save_segmentation_button,
        )
        for button in file_buttons:
            button.setMaximumWidth(self._COMPACT_BUTTON_MAX_WIDTH)

        navigation_inputs = (
            self._cursor_z,
            self._cursor_y,
            self._cursor_x,
            self._zoom_spin,
            self._manual_level_spin,
        )
        for widget in navigation_inputs:
            widget.setMaximumWidth(self._COMPACT_INPUT_MAX_WIDTH)

        self._contrast_min_slider.setMaximumWidth(self._COMPACT_SLIDER_MAX_WIDTH)
        self._contrast_max_slider.setMaximumWidth(self._COMPACT_SLIDER_MAX_WIDTH)
        self._segmentation_opacity_slider.setMaximumWidth(self._COMPACT_SLIDER_MAX_WIDTH)

        annotation_compact_widgets = (
            self._annotation_tool_label,
            self._annotation_tool_combo,
            self._tool_label_label,
            self._tool_label_edit,
            self._brush_radius_label,
            self._brush_radius_spin,
            self._flood_fill_button,
            self._next_available_button,
        )
        for widget in annotation_compact_widgets:
            widget.setMaximumWidth(self._COMPACT_BUTTON_MAX_WIDTH)

        self._bbox_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._bbox_table.setMinimumWidth(self._COMPACT_BBOX_TABLE_MIN_WIDTH)
        self._bbox_table.setMaximumWidth(16_777_215)

        history_buttons = (
            self._undo_button,
            self._redo_button,
        )
        for button in history_buttons:
            button.setMaximumWidth(self._COMPACT_BUTTON_MAX_WIDTH)

    def _update_bbox_table_width(self) -> None:
        self._bounding_boxes_panel.update_table_width(self._COMPACT_BBOX_TABLE_MIN_WIDTH)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_bbox_table_width()

    def set_file_path(self, path: str) -> None:
        self._file_path = path

    def file_path(self) -> Optional[str]:
        return self._file_path

    def set_zoom(self, zoom: float) -> None:
        normalized_zoom = max(self._ZOOM_MIN, min(self._ZOOM_MAX, float(zoom)))
        self.state.zoom = normalized_zoom
        self._navigation_panel.set_zoom(normalized_zoom)

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
        self._navigation_panel.set_level_mode(
            auto_enabled=self.state.auto_level_enabled,
            manual_level=normalized_manual,
            max_level=normalized_max,
        )
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

    def set_segmentation_opacity(self, opacity: float) -> None:
        try:
            normalized = float(opacity)
        except (TypeError, ValueError):
            normalized = self._SEGMENTATION_OPACITY_DEFAULT
        if not np.isfinite(normalized):
            normalized = self._SEGMENTATION_OPACITY_DEFAULT
        normalized = max(0.0, min(1.0, normalized))
        self.state.segmentation_opacity = normalized
        self._contrast_panel.set_segmentation_opacity(normalized)

    def segmentation_opacity(self) -> float:
        return float(self.state.segmentation_opacity)

    def set_view_layout_mode(self, mode: str) -> None:
        normalized = str(mode).strip().lower()
        if normalized not in {"all", "axial", "coronal", "sagittal"}:
            normalized = "all"
        self.state.view_layout_mode = normalized
        self._navigation_panel.set_view_layout_mode(normalized)

    def view_layout_mode(self) -> str:
        return str(self.state.view_layout_mode)

    def set_annotation_mode(self, enabled: bool) -> None:
        self.state.annotation_enabled = bool(enabled)
        self._annotation_panel.set_annotation_mode(self.state.annotation_enabled)
        self._update_eraser_controls_state()

    def set_bounding_box_mode(self, enabled: bool) -> None:
        self.state.bounding_box_mode_enabled = bool(enabled)
        self._bounding_boxes_panel.set_bounding_box_mode(
            self.state.bounding_box_mode_enabled
        )

    def set_annotation_controls_enabled(self, enabled: bool) -> None:
        self._annotation_controls_enabled = bool(enabled)
        editable_controls_enabled = bool(
            self._annotation_controls_enabled and not self._inference_navigation_only_mode
        )
        self._annotation_panel.set_editing_controls_enabled(editable_controls_enabled)
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
        self._annotation_panel.set_annotation_tool(normalized)
        self._update_eraser_controls_state()

    def annotation_tool(self) -> AnnotationTool:
        return self.state.annotation_tool

    def set_tool_label(self, value: str) -> None:
        normalized = str(value).strip()
        self.state.tool_label_text = normalized
        self._annotation_panel.set_tool_label(normalized)

    def tool_label(self) -> str:
        return self._annotation_panel.tool_label()

    def set_brush_radius(self, brush_radius: BrushRadius) -> None:
        normalized = _normalize_brush_radius(brush_radius)
        self.state.brush_radius = normalized
        self._annotation_panel.set_brush_radius(normalized)

    def brush_radius(self) -> BrushRadius:
        return int(self.state.brush_radius)

    def set_tool_label_placeholder(self, value: str) -> None:
        self._annotation_panel.set_tool_label_placeholder(str(value))

    def set_undo_state(self, *, depth: int, enabled: bool) -> None:
        normalized_depth = max(0, int(depth))
        self.state.undo_depth = normalized_depth
        self._undo_enabled_requested = bool(enabled)
        self._update_history_controls_state()

    def set_redo_state(self, *, depth: int, enabled: bool) -> None:
        normalized_depth = max(0, int(depth))
        self.state.redo_depth = normalized_depth
        self._redo_enabled_requested = bool(enabled)
        self._update_history_controls_state()

    def set_cursor_range(self, shape: Tuple[int, int, int]) -> None:
        self._navigation_panel.set_cursor_range(shape)

    def set_cursor_position(self, indices: Tuple[int, int, int]) -> None:
        self.state.cursor_position = indices
        self._navigation_panel.set_cursor_position(indices)

    def set_hover_info(
        self,
        indices: Optional[Tuple[int, int, int]],
        label: Optional[int],
    ) -> None:
        self.state.hover_position = indices
        self.state.hover_label = None if label is None else int(label)
        self._navigation_panel.set_hover_info(indices, label)

    def set_picked_info(
        self,
        indices: Optional[Tuple[int, int, int]],
        label: Optional[int],
    ) -> None:
        self.state.picked_position = indices
        self.state.picked_label = None if label is None else int(label)
        self._navigation_panel.set_picked_info(indices, label)
        self._update_eraser_controls_state()

    def set_pyramid_levels(self, levels: int, kind: str = "Raw") -> None:
        levels = max(1, int(levels))
        self.state.pyramid_levels = levels
        self._navigation_panel.set_pyramid_levels(levels, kind)

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
        self._navigation_panel.set_active_levels(
            axial=(ax_level, ax_scale),
            coronal=(co_level, co_scale),
            sagittal=(sa_level, sa_scale),
            forced=forced,
        )

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

        table_rows = tuple(
            (
                str(row_index + 1),
                row.box_id,
                row.label,
                row.size_text,
                row.center_text,
            )
            for row_index, row in enumerate(self.state.bbox_rows)
        )
        self._bounding_boxes_panel.set_rows(table_rows)
        self._update_bbox_table_width()

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
        self._bounding_boxes_panel.set_selected_rows(tuple(row_indices))
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
        self._learning_panel.set_training_running(self.state.learning_training_running)

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

    def on_segmentation_opacity_changed(self, callback: Callable[[float], None]) -> None:
        self._on_segmentation_opacity_changed = callback

    def on_view_layout_mode_changed(self, callback: Callable[[str], None]) -> None:
        self._on_view_layout_mode_changed = callback

    def on_annotation_mode_changed(self, callback: Callable[[bool], None]) -> None:
        self._on_annotation_mode_changed = callback

    def on_bounding_box_mode_changed(self, callback: Callable[[bool], None]) -> None:
        self._on_bounding_box_mode_changed = callback

    def on_annotation_tool_changed(self, callback: Callable[[AnnotationTool], None]) -> None:
        self._on_annotation_tool_changed = callback

    def on_tool_label_changed(self, callback: Callable[[str], None]) -> None:
        self._on_tool_label_changed = callback

    def on_next_available_label_requested(self, callback: Callable[[], None]) -> None:
        self._on_next_available_label_requested = callback

    def on_brush_radius_changed(self, callback: Callable[[BrushRadius], None]) -> None:
        self._on_brush_radius_changed = callback

    def on_flood_fill_requested(self, callback: Callable[[int], None]) -> None:
        self._on_flood_fill_requested = callback

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

    def on_load_model_requested(self, callback: Callable[[], None]) -> None:
        self._on_load_model_requested = callback

    def on_save_model_requested(self, callback: Callable[[], None]) -> None:
        self._on_save_model_requested = callback

    def on_segment_inference_requested(self, callback: Callable[[], None]) -> None:
        self._on_segment_inference_requested = callback

    def on_segment_inference_headless_close_requested(
        self,
        callback: Callable[[], None],
    ) -> None:
        self._on_segment_inference_headless_close_requested = callback

    def on_stop_inference_requested(self, callback: Callable[[], None]) -> None:
        self._on_stop_inference_requested = callback

    def on_train_model_requested(self, callback: Callable[[], None]) -> None:
        self._on_train_model_requested = callback

    def on_train_model_headless_close_requested(
        self,
        callback: Callable[[], None],
    ) -> None:
        self._on_train_model_headless_close_requested = callback

    def on_stop_training_requested(self, callback: Callable[[], None]) -> None:
        self._on_stop_training_requested = callback

    def on_change_training_parameters_requested(self, callback: Callable[[], None]) -> None:
        self._on_change_training_parameters_requested = callback

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

    def _handle_manual_level_requested(self, level: Optional[int] = None) -> None:
        if not self._manual_level_spin.isEnabled():
            return
        requested = self._navigation_panel.manual_level_value() if level is None else int(level)
        normalized = self._normalize_manual_level(requested)
        self.state.manual_level = normalized
        self._navigation_panel.set_level_mode(
            auto_enabled=self.state.auto_level_enabled,
            manual_level=normalized,
            max_level=self.state.manual_level_max,
        )
        if self._on_manual_level_requested is not None:
            self._on_manual_level_requested(normalized)

    def _handle_view_layout_mode_changed(self, mode: object) -> None:
        if isinstance(mode, bool):
            if not mode:
                return
            next_mode = self._navigation_panel.current_view_layout_mode()
        else:
            next_mode = str(mode).strip().lower()
        if next_mode not in {"all", "axial", "coronal", "sagittal"}:
            next_mode = "all"
        previous_mode = self.state.view_layout_mode
        self.state.view_layout_mode = next_mode
        if next_mode != previous_mode and self._on_view_layout_mode_changed is not None:
            self._on_view_layout_mode_changed(next_mode)

    def _handle_contrast_min_changed(self, value: int) -> None:
        data_range = self.state.contrast_data_range
        if data_range is None:
            return
        min_step = int(value)
        max_step = self._contrast_panel.contrast_max_step()
        if self._can_adjust_contrast() and min_step >= max_step:
            min_step = max(0, max_step - 1)
            self._contrast_panel.set_slider_steps(
                min_step=min_step,
                max_step=max_step,
            )
        self._set_contrast_window_from_steps(min_step, max_step, emit_change=True)

    def _handle_contrast_max_changed(self, value: int) -> None:
        data_range = self.state.contrast_data_range
        if data_range is None:
            return
        min_step = self._contrast_panel.contrast_min_step()
        max_step = int(value)
        if self._can_adjust_contrast() and max_step <= min_step:
            max_step = min(self._CONTRAST_MAX_STEP, min_step + 1)
            self._contrast_panel.set_slider_steps(
                min_step=min_step,
                max_step=max_step,
            )
        self._set_contrast_window_from_steps(min_step, max_step, emit_change=True)

    def _handle_segmentation_opacity_changed(self, value: int) -> None:
        normalized = self._contrast_panel.set_segmentation_opacity_step(value)
        self.state.segmentation_opacity = normalized
        if self._on_segmentation_opacity_changed is not None:
            self._on_segmentation_opacity_changed(normalized)

    def _handle_annotation_mode_changed(self, enabled: bool) -> None:
        self.state.annotation_enabled = bool(enabled)
        self._update_eraser_controls_state()
        if self._on_annotation_mode_changed:
            self._on_annotation_mode_changed(self.state.annotation_enabled)

    def _handle_annotation_tool_changed(self, tool: str) -> None:
        value = str(tool)
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

    def _handle_tool_label_changed(self, value: str) -> None:
        value = str(value).strip()
        self.state.tool_label_text = value
        if self._on_tool_label_changed:
            self._on_tool_label_changed(value)

    def _handle_next_available_label_requested(self) -> None:
        if self._on_next_available_label_requested:
            self._on_next_available_label_requested()

    def _handle_brush_radius_changed(self, value: int) -> None:
        self.state.brush_radius = _normalize_brush_radius(value)
        if self._on_brush_radius_changed:
            self._on_brush_radius_changed(self.state.brush_radius)

    def _handle_flood_fill_requested(self, value: int) -> None:
        if self._on_flood_fill_requested:
            self._on_flood_fill_requested(int(value))

    def _handle_undo_requested(self) -> None:
        if self._on_undo_requested:
            self._on_undo_requested()

    def _handle_redo_requested(self) -> None:
        if self._on_redo_requested:
            self._on_redo_requested()

    def _selected_bbox_ids_from_table_selection(self) -> Tuple[str, ...]:
        selected_row_indices = self._bounding_boxes_panel.selected_row_indices()
        selected_ids = []
        for row in selected_row_indices:
            if row < 0 or row >= len(self.state.bbox_rows):
                continue
            box_id = str(self.state.bbox_rows[row].box_id).strip()
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
        if row_index >= len(self.state.bbox_rows):
            return
        box_id = str(self.state.bbox_rows[row_index].box_id).strip()
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

    def _handle_load_model_requested(self) -> None:
        if self._on_load_model_requested:
            self._on_load_model_requested()

    def _handle_save_model_requested(self) -> None:
        if self._on_save_model_requested:
            self._on_save_model_requested()

    def _handle_segment_inference_requested(self) -> None:
        if self._on_segment_inference_requested:
            self._on_segment_inference_requested()

    def _handle_segment_inference_headless_close_requested(self) -> None:
        if self._on_segment_inference_headless_close_requested:
            self._on_segment_inference_headless_close_requested()

    def _handle_stop_inference_requested(self) -> None:
        if self._on_stop_inference_requested:
            self._on_stop_inference_requested()

    def _handle_train_model_requested(self) -> None:
        if self._on_train_model_requested:
            self._on_train_model_requested()

    def _handle_train_model_headless_close_requested(self) -> None:
        if self._on_train_model_headless_close_requested:
            self._on_train_model_headless_close_requested()

    def _handle_stop_training_requested(self) -> None:
        if self._on_stop_training_requested:
            self._on_stop_training_requested()

    def _handle_change_training_parameters_requested(self) -> None:
        if self._on_change_training_parameters_requested:
            self._on_change_training_parameters_requested()

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
        selected_label = _normalize_bbox_label(
            self._bounding_boxes_panel.selected_label_value()
        )
        if self.state.bbox_selected_label is not None and self.state.bbox_selected_label == selected_label:
            return
        self.state.bbox_selected_label = selected_label
        if self._on_bounding_boxes_label_changed:
            self._on_bounding_boxes_label_changed(selected_ids, selected_label)
        selected_id = _primary_selected_bbox_id(selected_ids)
        if self._on_bounding_box_label_changed and selected_id is not None:
            self._on_bounding_box_label_changed(selected_id, selected_label)

    def _update_eraser_controls_state(self) -> None:
        tool_label_active = (
            self._annotation_controls_enabled
            and not self._inference_navigation_only_mode
            and self.state.annotation_enabled
        )
        placeholder = "All" if self.state.annotation_tool == "eraser" else "1"
        flood_fill_active = (
            self._annotation_controls_enabled
            and not self._inference_navigation_only_mode
            and self.state.annotation_enabled
            and self.state.annotation_tool == "flood_filler"
        )
        self._annotation_panel.set_tool_controls_state(
            tool_label_active=tool_label_active,
            flood_fill_active=(
                flood_fill_active and self.state.picked_position is not None
            ),
            placeholder=placeholder,
        )
        self._update_bounding_box_controls_state()

    def _update_interaction_tool_controls_state(self) -> None:
        enabled = bool(self._interaction_tools_enabled)
        tool_controls_enabled = bool(enabled and not self._inference_navigation_only_mode)
        self._annotation_panel.set_interaction_controls_enabled(tool_controls_enabled)
        self._bounding_boxes_panel.set_mode_control_enabled(tool_controls_enabled)
        self._update_level_controls_state()
        self._update_contrast_controls_state()

    def _update_level_controls_state(self) -> None:
        enabled = bool(self._interaction_tools_enabled and self._level_controls_enabled)
        self._navigation_panel.set_level_controls_state(
            enabled=enabled,
            auto_enabled=self.state.auto_level_enabled,
        )

    def _update_contrast_controls_state(self) -> None:
        enabled = bool(self._interaction_tools_enabled)
        sliders_enabled = enabled and self._can_adjust_contrast()
        self._contrast_panel.set_controls_state(
            enabled=enabled,
            sliders_enabled=sliders_enabled,
        )

    def _update_bounding_box_controls_state(self) -> None:
        editing_locked = bool(self._inference_navigation_only_mode)
        has_boxes = len(self.state.bbox_rows) > 0
        has_selected_box = bool(self.state.bbox_selected_ids)
        self._bounding_boxes_panel.set_controls_state(
            editing_locked=editing_locked,
            has_boxes=has_boxes,
            has_selected_box=has_selected_box,
        )

    def _update_file_controls_state(self) -> None:
        enabled = bool(not self._inference_navigation_only_mode)
        self._files_panel.set_controls_enabled(enabled)

    def _update_learning_controls_state(self) -> None:
        editing_locked = bool(self._inference_navigation_only_mode)
        self._learning_panel.set_controls_state(
            editing_locked=editing_locked,
            segment_inference_enabled=self._segment_inference_enabled_requested,
            train_model_enabled=self._train_model_enabled_requested,
            stop_training_enabled=self._stop_training_enabled_requested,
            stop_inference_enabled=self._stop_inference_enabled_requested,
        )

    def _update_history_controls_state(self) -> None:
        editing_locked = bool(self._inference_navigation_only_mode)
        self._history_panel.set_undo_state(
            depth=self.state.undo_depth,
            requested_enabled=self._undo_enabled_requested,
            editing_locked=editing_locked,
        )
        self._history_panel.set_redo_state(
            depth=self.state.redo_depth,
            requested_enabled=self._redo_enabled_requested,
            editing_locked=editing_locked,
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
        self._bounding_boxes_panel.set_selected_label_value(label)

    def _handle_cursor(self, indices: object) -> None:
        if not isinstance(indices, tuple) or len(indices) != 3:
            indices = (
                self._cursor_z.value(),
                self._cursor_y.value(),
                self._cursor_x.value(),
            )
        indices = (int(indices[0]), int(indices[1]), int(indices[2]))
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
        self._contrast_panel.set_slider_steps(min_step=min_step, max_step=max_step)

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
        self._contrast_panel.set_contrast_labels(self.state.contrast_window)
