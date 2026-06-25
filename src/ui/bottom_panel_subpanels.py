from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class BottomPanelSubpanelSpec:
    name: str
    title: str
    responsibility: str
    public_methods: Tuple[str, ...]


BOTTOM_PANEL_SUBPANEL_SPECS: Tuple[BottomPanelSubpanelSpec, ...] = (
    BottomPanelSubpanelSpec(
        name="files",
        title="Files",
        responsibility="Raw/segmentation file actions and current file path.",
        public_methods=(
            "set_file_path",
            "file_path",
            "on_open_requested",
            "on_open_semantic_requested",
            "on_open_instance_requested",
            "on_save_segmentation_requested",
        ),
    ),
    BottomPanelSubpanelSpec(
        name="navigation",
        title="Navigation",
        responsibility="Cursor, zoom, pyramid level, and view-layout controls.",
        public_methods=(
            "set_zoom",
            "set_level_mode",
            "set_level_controls_enabled",
            "set_cursor_range",
            "set_cursor_position",
            "set_hover_info",
            "set_picked_info",
            "set_pyramid_levels",
            "set_active_level",
            "set_active_levels",
            "set_view_layout_mode",
            "view_layout_mode",
            "on_cursor_changed",
            "on_zoom_changed",
            "on_auto_level_mode_changed",
            "on_manual_level_requested",
            "on_view_layout_mode_changed",
        ),
    ),
    BottomPanelSubpanelSpec(
        name="contrast",
        title="Contrast",
        responsibility="Raw intensity window and segmentation opacity controls.",
        public_methods=(
            "set_contrast_range",
            "set_contrast_window",
            "contrast_window",
            "set_segmentation_opacity",
            "segmentation_opacity",
            "on_contrast_window_changed",
            "on_segmentation_opacity_changed",
        ),
    ),
    BottomPanelSubpanelSpec(
        name="annotation",
        title="Annotation",
        responsibility="Manual segmentation tool, label, brush, and flood-fill controls.",
        public_methods=(
            "set_annotation_mode",
            "set_annotation_controls_enabled",
            "set_interaction_tools_enabled",
            "set_annotation_tool",
            "annotation_tool",
            "set_tool_label",
            "tool_label",
            "set_brush_radius",
            "brush_radius",
            "set_tool_label_placeholder",
            "on_annotation_mode_changed",
            "on_annotation_tool_changed",
            "on_tool_label_changed",
            "on_next_available_label_requested",
            "on_brush_radius_changed",
            "on_flood_fill_requested",
        ),
    ),
    BottomPanelSubpanelSpec(
        name="bounding_boxes",
        title="Bounding Boxes",
        responsibility="BBox table, selection, labels, persistence, and bbox-local processing actions.",
        public_methods=(
            "set_bounding_box_mode",
            "set_bounding_boxes",
            "set_selected_bounding_boxes",
            "set_selected_bounding_box",
            "selected_bounding_boxes",
            "selected_bounding_box",
            "selected_bounding_box_label",
            "on_bounding_box_mode_changed",
            "on_bounding_boxes_selected",
            "on_bounding_box_selected",
            "on_bounding_box_double_clicked",
            "on_open_bounding_boxes_requested",
            "on_save_bounding_boxes_requested",
            "on_median_filter_selected_requested",
            "on_erosion_selected_requested",
            "on_dilation_selected_requested",
            "on_erase_bbox_segmentation_requested",
            "on_bounding_boxes_delete_requested",
            "on_bounding_box_delete_requested",
            "on_bounding_boxes_label_changed",
            "on_bounding_box_label_changed",
        ),
    ),
    BottomPanelSubpanelSpec(
        name="learning",
        title="Learning",
        responsibility="Model load/save, training, inference, stop controls, and learning status.",
        public_methods=(
            "set_train_model_enabled",
            "set_segment_inference_enabled",
            "set_stop_training_enabled",
            "set_stop_inference_enabled",
            "set_inference_navigation_only_mode",
            "set_learning_training_running",
            "learning_training_running",
            "on_load_model_requested",
            "on_save_model_requested",
            "on_segment_inference_requested",
            "on_segment_inference_headless_close_requested",
            "on_stop_inference_requested",
            "on_train_model_requested",
            "on_train_model_headless_close_requested",
            "on_stop_training_requested",
            "on_change_training_parameters_requested",
        ),
    ),
    BottomPanelSubpanelSpec(
        name="history",
        title="History",
        responsibility="Undo and redo controls.",
        public_methods=(
            "set_undo_state",
            "set_redo_state",
            "on_undo_requested",
            "on_redo_requested",
        ),
    ),
)

BOTTOM_PANEL_SUBPANEL_ORDER: Tuple[str, ...] = tuple(
    spec.name for spec in BOTTOM_PANEL_SUBPANEL_SPECS
)


class FilesPanel(QGroupBox):
    def __init__(self) -> None:
        super().__init__("Files")
        self._on_open_requested: Optional[Callable[[], None]] = None
        self._on_open_semantic_requested: Optional[Callable[[], None]] = None
        self._on_open_instance_requested: Optional[Callable[[], None]] = None
        self._on_save_segmentation_requested: Optional[Callable[[], None]] = None

        self.open_button = QPushButton("Open")
        self.open_semantic_button = QPushButton("Open Semantic")
        self.open_instance_button = QPushButton("Open Instance")
        self.save_segmentation_button = QPushButton("Save Segmentation")

        layout = QGridLayout()
        layout.addWidget(self.open_button, 0, 0)
        layout.addWidget(self.open_semantic_button, 0, 1)
        layout.addWidget(self.open_instance_button, 1, 0)
        layout.addWidget(self.save_segmentation_button, 1, 1)
        self.setLayout(layout)

        self.open_button.clicked.connect(self._emit_open_requested)
        self.open_semantic_button.clicked.connect(self._emit_open_semantic_requested)
        self.open_instance_button.clicked.connect(self._emit_open_instance_requested)
        self.save_segmentation_button.clicked.connect(
            self._emit_save_segmentation_requested
        )

    def on_open_requested(self, callback: Callable[[], None]) -> None:
        self._on_open_requested = callback

    def on_open_semantic_requested(self, callback: Callable[[], None]) -> None:
        self._on_open_semantic_requested = callback

    def on_open_instance_requested(self, callback: Callable[[], None]) -> None:
        self._on_open_instance_requested = callback

    def on_save_segmentation_requested(self, callback: Callable[[], None]) -> None:
        self._on_save_segmentation_requested = callback

    def set_controls_enabled(self, enabled: bool) -> None:
        normalized = bool(enabled)
        self.open_button.setEnabled(normalized)
        self.open_semantic_button.setEnabled(normalized)
        self.open_instance_button.setEnabled(normalized)
        self.save_segmentation_button.setEnabled(normalized)

    def _emit_open_requested(self) -> None:
        if self._on_open_requested is not None:
            self._on_open_requested()

    def _emit_open_semantic_requested(self) -> None:
        if self._on_open_semantic_requested is not None:
            self._on_open_semantic_requested()

    def _emit_open_instance_requested(self) -> None:
        if self._on_open_instance_requested is not None:
            self._on_open_instance_requested()

    def _emit_save_segmentation_requested(self) -> None:
        if self._on_save_segmentation_requested is not None:
            self._on_save_segmentation_requested()


class NavigationPanel(QGroupBox):
    def __init__(self, *, zoom_min: float, zoom_max: float) -> None:
        super().__init__("Navigation")
        self._on_cursor_changed: Optional[Callable[[Tuple[int, int, int]], None]] = None
        self._on_zoom_changed: Optional[Callable[[float], None]] = None
        self._on_auto_level_mode_changed: Optional[Callable[[bool], None]] = None
        self._on_manual_level_requested: Optional[Callable[[int], None]] = None
        self._on_view_layout_mode_changed: Optional[Callable[[str], None]] = None

        self.cursor_label = QLabel("Cursor")
        self.cursor_z = QSpinBox()
        self.cursor_y = QSpinBox()
        self.cursor_x = QSpinBox()
        self.hover_label = QLabel("Hover")
        self.hover_value = QLabel("Z:- Y:- X:- | ID:-")
        self.picked_label = QLabel("Selected")
        self.picked_value = QLabel("Z:- Y:- X:- | ID:-")
        self.cursor_z.setPrefix("Z:")
        self.cursor_y.setPrefix("Y:")
        self.cursor_x.setPrefix("X:")
        self.cursor_z.setRange(0, 0)
        self.cursor_y.setRange(0, 0)
        self.cursor_x.setRange(0, 0)

        self.zoom_spin = QDoubleSpinBox()
        self.zoom_spin.setRange(float(zoom_min), float(zoom_max))
        self.zoom_spin.setSingleStep(0.1)
        self.zoom_spin.setValue(1.0)

        self.auto_level_checkbox = QCheckBox("Auto Level")
        self.auto_level_checkbox.setChecked(True)
        self.manual_level_label = QLabel("Manual Level")
        self.manual_level_spin = QSpinBox()
        self.manual_level_spin.setPrefix("L:")
        self.manual_level_spin.setRange(0, 0)
        self.manual_level_spin.setValue(0)

        self.view_layout_label = QLabel("View Layout")
        self.view_layout_button_group = QButtonGroup(self)
        self.view_layout_button_group.setExclusive(True)
        self.view_layout_all_radio = QRadioButton("All (3 views)")
        self.view_layout_axial_radio = QRadioButton("Axial only")
        self.view_layout_coronal_radio = QRadioButton("Coronal only")
        self.view_layout_sagittal_radio = QRadioButton("Sagittal only")
        self.view_layout_all_radio.setChecked(True)
        self.view_layout_button_group.addButton(self.view_layout_all_radio)
        self.view_layout_button_group.addButton(self.view_layout_axial_radio)
        self.view_layout_button_group.addButton(self.view_layout_coronal_radio)
        self.view_layout_button_group.addButton(self.view_layout_sagittal_radio)

        self.pyramid_status = QLabel("Pyramid: -")
        self.level_status = QLabel("Level: L0 (x1)")

        layout = QFormLayout()
        cursor_row = QWidget()
        cursor_row_layout = QHBoxLayout()
        cursor_row_layout.setContentsMargins(0, 0, 0, 0)
        cursor_row_layout.addWidget(self.cursor_z)
        cursor_row_layout.addWidget(self.cursor_y)
        cursor_row_layout.addWidget(self.cursor_x)
        cursor_row.setLayout(cursor_row_layout)
        layout.addRow(self.cursor_label, cursor_row)
        layout.addRow(self.hover_label, self.hover_value)
        layout.addRow(self.picked_label, self.picked_value)
        layout.addRow(QLabel("Zoom"), self.zoom_spin)
        layout.addRow(self.auto_level_checkbox)
        layout.addRow(self.manual_level_label, self.manual_level_spin)

        view_layout_row = QWidget()
        view_layout_grid = QGridLayout()
        view_layout_grid.setContentsMargins(0, 0, 0, 0)
        view_layout_grid.addWidget(self.view_layout_all_radio, 0, 0, 1, 2)
        view_layout_grid.addWidget(self.view_layout_axial_radio, 1, 0)
        view_layout_grid.addWidget(self.view_layout_coronal_radio, 1, 1)
        view_layout_grid.addWidget(self.view_layout_sagittal_radio, 2, 0)
        view_layout_row.setLayout(view_layout_grid)
        layout.addRow(self.view_layout_label, view_layout_row)
        layout.addRow(self.pyramid_status)
        layout.addRow(self.level_status)
        self.setLayout(layout)

        self.cursor_z.valueChanged.connect(self._emit_cursor_changed)
        self.cursor_y.valueChanged.connect(self._emit_cursor_changed)
        self.cursor_x.valueChanged.connect(self._emit_cursor_changed)
        self.zoom_spin.valueChanged.connect(self._emit_zoom_changed)
        self.auto_level_checkbox.toggled.connect(self._emit_auto_level_mode_changed)
        manual_level_line_edit = self.manual_level_spin.lineEdit()
        if manual_level_line_edit is not None:
            manual_level_line_edit.returnPressed.connect(
                self._emit_manual_level_requested
            )
        else:
            self.manual_level_spin.editingFinished.connect(
                self._emit_manual_level_requested
            )
        self.view_layout_all_radio.toggled.connect(
            self._emit_view_layout_mode_changed
        )
        self.view_layout_axial_radio.toggled.connect(
            self._emit_view_layout_mode_changed
        )
        self.view_layout_coronal_radio.toggled.connect(
            self._emit_view_layout_mode_changed
        )
        self.view_layout_sagittal_radio.toggled.connect(
            self._emit_view_layout_mode_changed
        )

    def on_cursor_changed(
        self,
        callback: Callable[[Tuple[int, int, int]], None],
    ) -> None:
        self._on_cursor_changed = callback

    def on_zoom_changed(self, callback: Callable[[float], None]) -> None:
        self._on_zoom_changed = callback

    def on_auto_level_mode_changed(self, callback: Callable[[bool], None]) -> None:
        self._on_auto_level_mode_changed = callback

    def on_manual_level_requested(self, callback: Callable[[int], None]) -> None:
        self._on_manual_level_requested = callback

    def on_view_layout_mode_changed(self, callback: Callable[[str], None]) -> None:
        self._on_view_layout_mode_changed = callback

    def set_zoom(self, zoom: float) -> None:
        self.zoom_spin.blockSignals(True)
        self.zoom_spin.setValue(float(zoom))
        self.zoom_spin.blockSignals(False)

    def set_level_mode(
        self,
        *,
        auto_enabled: bool,
        manual_level: int,
        max_level: int,
    ) -> None:
        self.auto_level_checkbox.blockSignals(True)
        self.auto_level_checkbox.setChecked(bool(auto_enabled))
        self.auto_level_checkbox.blockSignals(False)
        self.manual_level_spin.blockSignals(True)
        self.manual_level_spin.setRange(0, max(0, int(max_level)))
        self.manual_level_spin.setValue(max(0, int(manual_level)))
        self.manual_level_spin.blockSignals(False)

    def set_level_controls_state(
        self,
        *,
        enabled: bool,
        auto_enabled: bool,
    ) -> None:
        controls_enabled = bool(enabled)
        manual_enabled = bool(controls_enabled and not auto_enabled)
        self.auto_level_checkbox.setEnabled(controls_enabled)
        self.manual_level_label.setEnabled(manual_enabled)
        self.manual_level_spin.setEnabled(manual_enabled)

    def set_cursor_range(self, shape: Tuple[int, int, int]) -> None:
        z_max = max(0, int(shape[0]) - 1)
        y_max = max(0, int(shape[1]) - 1)
        x_max = max(0, int(shape[2]) - 1)
        self.cursor_z.setRange(0, z_max)
        self.cursor_y.setRange(0, y_max)
        self.cursor_x.setRange(0, x_max)

    def set_cursor_position(self, indices: Tuple[int, int, int]) -> None:
        z, y, x = indices
        self.cursor_z.blockSignals(True)
        self.cursor_y.blockSignals(True)
        self.cursor_x.blockSignals(True)
        self.cursor_z.setValue(int(z))
        self.cursor_y.setValue(int(y))
        self.cursor_x.setValue(int(x))
        self.cursor_z.blockSignals(False)
        self.cursor_y.blockSignals(False)
        self.cursor_x.blockSignals(False)

    def set_hover_info(
        self,
        indices: Optional[Tuple[int, int, int]],
        label: Optional[int],
    ) -> None:
        self.hover_value.setText(self._format_position_label(indices, label))

    def set_picked_info(
        self,
        indices: Optional[Tuple[int, int, int]],
        label: Optional[int],
    ) -> None:
        self.picked_value.setText(self._format_position_label(indices, label))

    def set_pyramid_levels(self, levels: int, kind: str) -> None:
        self.pyramid_status.setText(f"{kind} levels computed: {max(1, int(levels))}")

    def set_active_levels(
        self,
        *,
        axial: Tuple[int, int],
        coronal: Tuple[int, int],
        sagittal: Tuple[int, int],
        forced: bool = False,
    ) -> None:
        ax_level = max(0, int(axial[0]))
        ax_scale = max(1, int(axial[1]))
        co_level = max(0, int(coronal[0]))
        co_scale = max(1, int(coronal[1]))
        sa_level = max(0, int(sagittal[0]))
        sa_scale = max(1, int(sagittal[1]))
        status_text = (
            f"Levels: Ax L{ax_level} (x{ax_scale}) | "
            f"Co L{co_level} (x{co_scale}) | Sa L{sa_level} (x{sa_scale})"
        )
        if forced:
            status_text += " | Manual (forced)"
        self.level_status.setText(status_text)

    def set_view_layout_mode(self, mode: str) -> None:
        normalized = str(mode).strip().lower()
        target_button = self.view_layout_all_radio
        if normalized == "axial":
            target_button = self.view_layout_axial_radio
        elif normalized == "coronal":
            target_button = self.view_layout_coronal_radio
        elif normalized == "sagittal":
            target_button = self.view_layout_sagittal_radio
        buttons = (
            self.view_layout_all_radio,
            self.view_layout_axial_radio,
            self.view_layout_coronal_radio,
            self.view_layout_sagittal_radio,
        )
        for button in buttons:
            button.blockSignals(True)
        target_button.setChecked(True)
        for button in buttons:
            button.blockSignals(False)

    def current_view_layout_mode(self) -> str:
        if self.view_layout_axial_radio.isChecked():
            return "axial"
        if self.view_layout_coronal_radio.isChecked():
            return "coronal"
        if self.view_layout_sagittal_radio.isChecked():
            return "sagittal"
        return "all"

    def manual_level_value(self) -> int:
        return int(self.manual_level_spin.value())

    def _emit_cursor_changed(self, _value: int) -> None:
        if self._on_cursor_changed is None:
            return
        self._on_cursor_changed(
            (self.cursor_z.value(), self.cursor_y.value(), self.cursor_x.value())
        )

    def _emit_zoom_changed(self, value: float) -> None:
        if self._on_zoom_changed is not None:
            self._on_zoom_changed(float(value))

    def _emit_auto_level_mode_changed(self, enabled: bool) -> None:
        if self._on_auto_level_mode_changed is not None:
            self._on_auto_level_mode_changed(bool(enabled))

    def _emit_manual_level_requested(self) -> None:
        if not self.manual_level_spin.isEnabled():
            return
        if self._on_manual_level_requested is not None:
            self._on_manual_level_requested(self.manual_level_value())

    def _emit_view_layout_mode_changed(self, checked: bool) -> None:
        if not checked:
            return
        if self._on_view_layout_mode_changed is not None:
            self._on_view_layout_mode_changed(self.current_view_layout_mode())

    @staticmethod
    def _format_position_label(
        indices: Optional[Tuple[int, int, int]],
        label: Optional[int],
    ) -> str:
        if indices is None:
            return "Z:- Y:- X:- | ID:-"
        z, y, x = indices
        label_text = "-" if label is None else str(int(label))
        return f"Z:{z} Y:{y} X:{x} | ID:{label_text}"


class ContrastPanel(QGroupBox):
    def __init__(
        self,
        *,
        contrast_max_step: int,
        segmentation_opacity_max_step: int,
    ) -> None:
        super().__init__("Contrast")
        self._contrast_max_step = int(contrast_max_step)
        self._segmentation_opacity_max_step = int(segmentation_opacity_max_step)
        self._on_contrast_min_changed: Optional[Callable[[int], None]] = None
        self._on_contrast_max_changed: Optional[Callable[[int], None]] = None
        self._on_segmentation_opacity_changed: Optional[Callable[[int], None]] = None

        self.contrast_min_label = QLabel("Window Min")
        self.contrast_min_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_min_slider.setRange(0, self._contrast_max_step)
        self.contrast_min_slider.setSingleStep(1)
        self.contrast_min_slider.setPageStep(10)
        self.contrast_max_label = QLabel("Window Max")
        self.contrast_max_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_max_slider.setRange(0, self._contrast_max_step)
        self.contrast_max_slider.setSingleStep(1)
        self.contrast_max_slider.setPageStep(10)
        self.contrast_min_value = QLabel("Min: -")
        self.contrast_max_value = QLabel("Max: -")

        self.segmentation_opacity_label = QLabel("Seg Alpha")
        self.segmentation_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.segmentation_opacity_slider.setRange(0, self._segmentation_opacity_max_step)
        self.segmentation_opacity_slider.setSingleStep(1)
        self.segmentation_opacity_slider.setPageStep(5)
        self.segmentation_opacity_value = QLabel("30%")

        layout = QFormLayout()
        contrast_min_row = QWidget()
        contrast_min_row_layout = QHBoxLayout()
        contrast_min_row_layout.setContentsMargins(0, 0, 0, 0)
        contrast_min_row_layout.addWidget(self.contrast_min_slider)
        contrast_min_row_layout.addWidget(self.contrast_min_value)
        contrast_min_row.setLayout(contrast_min_row_layout)

        contrast_max_row = QWidget()
        contrast_max_row_layout = QHBoxLayout()
        contrast_max_row_layout.setContentsMargins(0, 0, 0, 0)
        contrast_max_row_layout.addWidget(self.contrast_max_slider)
        contrast_max_row_layout.addWidget(self.contrast_max_value)
        contrast_max_row.setLayout(contrast_max_row_layout)

        segmentation_alpha_row = QWidget()
        segmentation_alpha_row_layout = QHBoxLayout()
        segmentation_alpha_row_layout.setContentsMargins(0, 0, 0, 0)
        segmentation_alpha_row_layout.addWidget(self.segmentation_opacity_slider)
        segmentation_alpha_row_layout.addWidget(self.segmentation_opacity_value)
        segmentation_alpha_row.setLayout(segmentation_alpha_row_layout)

        layout.addRow(self.contrast_min_label, contrast_min_row)
        layout.addRow(self.contrast_max_label, contrast_max_row)
        layout.addRow(self.segmentation_opacity_label, segmentation_alpha_row)
        self.setLayout(layout)

        self.contrast_min_slider.valueChanged.connect(
            self._emit_contrast_min_changed
        )
        self.contrast_max_slider.valueChanged.connect(
            self._emit_contrast_max_changed
        )
        self.segmentation_opacity_slider.valueChanged.connect(
            self._emit_segmentation_opacity_changed
        )

    def on_contrast_min_changed(self, callback: Callable[[int], None]) -> None:
        self._on_contrast_min_changed = callback

    def on_contrast_max_changed(self, callback: Callable[[int], None]) -> None:
        self._on_contrast_max_changed = callback

    def on_segmentation_opacity_changed(
        self,
        callback: Callable[[int], None],
    ) -> None:
        self._on_segmentation_opacity_changed = callback

    def set_slider_steps(self, *, min_step: int, max_step: int) -> None:
        normalized_min = max(0, min(self._contrast_max_step, int(min_step)))
        normalized_max = max(0, min(self._contrast_max_step, int(max_step)))
        self.contrast_min_slider.blockSignals(True)
        self.contrast_max_slider.blockSignals(True)
        self.contrast_min_slider.setValue(normalized_min)
        self.contrast_max_slider.setValue(normalized_max)
        self.contrast_min_slider.blockSignals(False)
        self.contrast_max_slider.blockSignals(False)

    def contrast_min_step(self) -> int:
        return int(self.contrast_min_slider.value())

    def contrast_max_step(self) -> int:
        return int(self.contrast_max_slider.value())

    def set_contrast_labels(self, window: Optional[Tuple[float, float]]) -> None:
        if window is None:
            self.contrast_min_value.setText("Min: -")
            self.contrast_max_value.setText("Max: -")
            return
        self.contrast_min_value.setText(f"Min: {window[0]:.6g}")
        self.contrast_max_value.setText(f"Max: {window[1]:.6g}")

    def set_segmentation_opacity(self, opacity: float) -> None:
        normalized = max(0.0, min(1.0, float(opacity)))
        step = int(round(normalized * float(self._segmentation_opacity_max_step)))
        step = max(0, min(self._segmentation_opacity_max_step, step))
        self.segmentation_opacity_slider.blockSignals(True)
        self.segmentation_opacity_slider.setValue(step)
        self.segmentation_opacity_slider.blockSignals(False)
        self.segmentation_opacity_value.setText(f"{int(round(normalized * 100.0))}%")

    def set_segmentation_opacity_step(self, step: int) -> float:
        normalized_step = max(0, min(self._segmentation_opacity_max_step, int(step)))
        normalized = float(normalized_step) / float(self._segmentation_opacity_max_step)
        self.segmentation_opacity_value.setText(f"{int(round(normalized * 100.0))}%")
        return normalized

    def set_controls_state(self, *, enabled: bool, sliders_enabled: bool) -> None:
        controls_enabled = bool(enabled)
        self.contrast_min_label.setEnabled(controls_enabled)
        self.contrast_max_label.setEnabled(controls_enabled)
        self.contrast_min_slider.setEnabled(bool(sliders_enabled))
        self.contrast_max_slider.setEnabled(bool(sliders_enabled))
        self.contrast_min_value.setEnabled(controls_enabled)
        self.contrast_max_value.setEnabled(controls_enabled)
        self.segmentation_opacity_label.setEnabled(controls_enabled)
        self.segmentation_opacity_slider.setEnabled(controls_enabled)
        self.segmentation_opacity_value.setEnabled(controls_enabled)

    def _emit_contrast_min_changed(self, value: int) -> None:
        if self._on_contrast_min_changed is not None:
            self._on_contrast_min_changed(int(value))

    def _emit_contrast_max_changed(self, value: int) -> None:
        if self._on_contrast_max_changed is not None:
            self._on_contrast_max_changed(int(value))

    def _emit_segmentation_opacity_changed(self, value: int) -> None:
        self.set_segmentation_opacity_step(int(value))
        if self._on_segmentation_opacity_changed is not None:
            self._on_segmentation_opacity_changed(int(value))


class AnnotationPanel(QGroupBox):
    def __init__(self) -> None:
        super().__init__("Annotation")
        self._on_annotation_mode_changed: Optional[Callable[[bool], None]] = None
        self._on_annotation_tool_changed: Optional[Callable[[str], None]] = None
        self._on_tool_label_changed: Optional[Callable[[str], None]] = None
        self._on_next_available_label_requested: Optional[Callable[[], None]] = None
        self._on_brush_radius_changed: Optional[Callable[[int], None]] = None
        self._on_flood_fill_requested: Optional[Callable[[int], None]] = None

        self.annotation_toggle = QCheckBox("Manual Segmentation")
        self.annotation_tool_label = QLabel("Tool")
        self.annotation_tool_combo = QComboBox()
        self.annotation_tool_combo.addItem("Brush (Ctrl+B)", "brush")
        self.annotation_tool_combo.addItem("Eraser (Ctrl+E)", "eraser")
        self.annotation_tool_combo.addItem("Flood Fill (Ctrl+F)", "flood_filler")
        annotation_shortcuts_hint = (
            "Shortcuts: Ctrl+B brush, Ctrl+E eraser, Ctrl+F flood fill. "
            "Using them auto-enables Manual Segmentation."
        )
        self.annotation_tool_label.setToolTip(annotation_shortcuts_hint)
        self.annotation_tool_combo.setToolTip(annotation_shortcuts_hint)
        self.annotation_toggle.setToolTip(
            "Enable manual segmentation. Ctrl+B/Ctrl+E/Ctrl+F also enables it automatically."
        )

        self.tool_label_label = QLabel("Tool Label")
        self.tool_label_edit = QLineEdit()
        self.tool_label_edit.setPlaceholderText("1")
        self.tool_label_edit.setText("1")
        self.tool_label_edit.setClearButtonEnabled(True)
        self.brush_radius_label = QLabel("Brush Radius")
        self.brush_radius_spin = QSpinBox()
        self.brush_radius_spin.setRange(0, 9)
        self.brush_radius_spin.setValue(0)
        self.flood_fill_button = QPushButton("Flood Fill")
        self.next_available_button = QPushButton("Next Available")

        layout = QFormLayout()
        layout.addRow(self.annotation_toggle)
        layout.addRow(self.annotation_tool_label, self.annotation_tool_combo)
        layout.addRow(self.tool_label_label, self.tool_label_edit)
        layout.addRow(self.brush_radius_label, self.brush_radius_spin)
        layout.addRow(self.flood_fill_button)
        layout.addRow(self.next_available_button)
        self.setLayout(layout)

        self.annotation_toggle.toggled.connect(self._emit_annotation_mode_changed)
        self.annotation_tool_combo.currentIndexChanged.connect(
            self._emit_annotation_tool_changed
        )
        self.tool_label_edit.editingFinished.connect(self._emit_tool_label_changed)
        self.brush_radius_spin.valueChanged.connect(self._emit_brush_radius_changed)
        self.flood_fill_button.clicked.connect(self._emit_flood_fill_requested)
        self.next_available_button.clicked.connect(
            self._emit_next_available_label_requested
        )

    def on_annotation_mode_changed(self, callback: Callable[[bool], None]) -> None:
        self._on_annotation_mode_changed = callback

    def on_annotation_tool_changed(self, callback: Callable[[str], None]) -> None:
        self._on_annotation_tool_changed = callback

    def on_tool_label_changed(self, callback: Callable[[str], None]) -> None:
        self._on_tool_label_changed = callback

    def on_next_available_label_requested(self, callback: Callable[[], None]) -> None:
        self._on_next_available_label_requested = callback

    def on_brush_radius_changed(self, callback: Callable[[int], None]) -> None:
        self._on_brush_radius_changed = callback

    def on_flood_fill_requested(self, callback: Callable[[int], None]) -> None:
        self._on_flood_fill_requested = callback

    def set_annotation_mode(self, enabled: bool) -> None:
        self.annotation_toggle.blockSignals(True)
        self.annotation_toggle.setChecked(bool(enabled))
        self.annotation_toggle.blockSignals(False)

    def set_annotation_tool(self, tool: str) -> None:
        index = self.annotation_tool_combo.findData(str(tool))
        if index < 0:
            index = 0
        self.annotation_tool_combo.blockSignals(True)
        self.annotation_tool_combo.setCurrentIndex(index)
        self.annotation_tool_combo.blockSignals(False)

    def current_annotation_tool(self) -> str:
        value = self.annotation_tool_combo.currentData()
        if value not in ("brush", "eraser", "flood_filler"):
            return "brush"
        return str(value)

    def set_tool_label(self, value: str) -> None:
        self.tool_label_edit.blockSignals(True)
        self.tool_label_edit.setText(str(value))
        self.tool_label_edit.blockSignals(False)

    def tool_label(self) -> str:
        return str(self.tool_label_edit.text().strip())

    def set_brush_radius(self, radius: int) -> None:
        self.brush_radius_spin.blockSignals(True)
        self.brush_radius_spin.setValue(int(radius))
        self.brush_radius_spin.blockSignals(False)

    def set_tool_label_placeholder(self, value: str) -> None:
        self.tool_label_edit.setPlaceholderText(str(value))

    def set_editing_controls_enabled(self, enabled: bool) -> None:
        normalized = bool(enabled)
        self.tool_label_label.setEnabled(normalized)
        self.tool_label_edit.setEnabled(normalized)
        self.brush_radius_label.setEnabled(normalized)
        self.brush_radius_spin.setEnabled(normalized)
        self.next_available_button.setEnabled(normalized)

    def set_interaction_controls_enabled(self, enabled: bool) -> None:
        normalized = bool(enabled)
        self.annotation_toggle.setEnabled(normalized)
        self.annotation_tool_label.setEnabled(normalized)
        self.annotation_tool_combo.setEnabled(normalized)

    def set_tool_controls_state(
        self,
        *,
        tool_label_active: bool,
        flood_fill_active: bool,
        placeholder: str,
    ) -> None:
        self.tool_label_label.setEnabled(bool(tool_label_active))
        self.tool_label_edit.setEnabled(bool(tool_label_active))
        self.tool_label_edit.setPlaceholderText(str(placeholder))
        self.flood_fill_button.setEnabled(bool(flood_fill_active))

    def _emit_annotation_mode_changed(self, enabled: bool) -> None:
        if self._on_annotation_mode_changed is not None:
            self._on_annotation_mode_changed(bool(enabled))

    def _emit_annotation_tool_changed(self, _index: int) -> None:
        if self._on_annotation_tool_changed is not None:
            self._on_annotation_tool_changed(self.current_annotation_tool())

    def _emit_tool_label_changed(self) -> None:
        if self._on_tool_label_changed is not None:
            self._on_tool_label_changed(self.tool_label())

    def _emit_next_available_label_requested(self) -> None:
        if self._on_next_available_label_requested is not None:
            self._on_next_available_label_requested()

    def _emit_brush_radius_changed(self, value: int) -> None:
        if self._on_brush_radius_changed is not None:
            self._on_brush_radius_changed(int(value))

    def _emit_flood_fill_requested(self) -> None:
        text = self.tool_label()
        try:
            value = int(text)
        except ValueError:
            value = 1
        if self._on_flood_fill_requested is not None:
            self._on_flood_fill_requested(value)


class BoundingBoxesPanel(QGroupBox):
    def __init__(self) -> None:
        super().__init__("Bounding Boxes")
        self._on_bounding_box_mode_changed: Optional[Callable[[bool], None]] = None
        self._on_selection_changed: Optional[Callable[[], None]] = None
        self._on_item_double_clicked: Optional[Callable[[QTableWidgetItem], None]] = None
        self._on_open_requested: Optional[Callable[[], None]] = None
        self._on_save_requested: Optional[Callable[[], None]] = None
        self._on_delete_requested: Optional[Callable[[], None]] = None
        self._on_label_changed: Optional[Callable[[int], None]] = None
        self._on_median_filter_requested: Optional[Callable[[], None]] = None
        self._on_erosion_requested: Optional[Callable[[], None]] = None
        self._on_dilation_requested: Optional[Callable[[], None]] = None
        self._on_erase_segmentation_requested: Optional[Callable[[], None]] = None

        self.bounding_box_mode_toggle = QCheckBox("Bounding Box Tool")
        self.bbox_table = QTableWidget(0, 5)
        self.bbox_table.setHorizontalHeaderLabels(
            ["ID", "bbox_name", "Label", "Size (dz, dy, dx)", "Center (z, y, x)"]
        )
        self.bbox_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.bbox_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.bbox_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.bbox_table.setAlternatingRowColors(True)
        self.bbox_table.verticalHeader().setVisible(False)
        bbox_header = self.bbox_table.horizontalHeader()
        bbox_header.setStretchLastSection(False)
        bbox_header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        bbox_header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        bbox_header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        bbox_header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        bbox_header.setSectionResizeMode(4, QHeaderView.ResizeToContents)

        self.bbox_label_label = QLabel("Selected Label")
        self.bbox_label_combo = QComboBox()
        self.bbox_label_combo.addItem("Train", "train")
        self.bbox_label_combo.addItem("Validation", "validation")
        self.bbox_label_combo.addItem("Inference", "inference")
        self.bbox_label_combo.setEnabled(False)
        self.bbox_label_label.setEnabled(False)

        self.open_bounding_boxes_button = QPushButton("Open Boxes...")
        self.save_bounding_boxes_button = QPushButton("Save Boxes...")
        self.delete_bbox_button = QPushButton("Delete Selected")
        self.delete_bbox_button.setEnabled(False)
        self.median_filter_selected_button = QPushButton("Median Filter Selected")
        self.erosion_selected_button = QPushButton("Erosion Selected")
        self.dilation_selected_button = QPushButton("Dilation Selected")
        self.erase_bbox_segmentation_button = QPushButton("Erase BBox Segmentation")

        layout = QVBoxLayout()
        layout.addWidget(self.bounding_box_mode_toggle)
        layout.addWidget(self.bbox_table)

        label_row = QWidget()
        label_layout = QFormLayout()
        label_layout.setContentsMargins(0, 0, 0, 0)
        label_layout.addRow(self.bbox_label_label, self.bbox_label_combo)
        label_row.setLayout(label_layout)
        layout.addWidget(label_row)

        controls_row = QWidget()
        controls_layout = QGridLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addWidget(self.open_bounding_boxes_button, 0, 0)
        controls_layout.addWidget(self.save_bounding_boxes_button, 0, 1)
        controls_layout.addWidget(self.delete_bbox_button, 1, 0)
        controls_row.setLayout(controls_layout)
        layout.addWidget(controls_row)

        processing_row = QWidget()
        processing_layout = QGridLayout()
        processing_layout.setContentsMargins(0, 0, 0, 0)
        processing_layout.addWidget(self.median_filter_selected_button, 0, 0)
        processing_layout.addWidget(self.erosion_selected_button, 0, 1)
        processing_layout.addWidget(self.dilation_selected_button, 1, 0)
        processing_layout.addWidget(self.erase_bbox_segmentation_button, 1, 1)
        processing_row.setLayout(processing_layout)
        layout.addWidget(processing_row)
        self.setLayout(layout)

        self.bounding_box_mode_toggle.toggled.connect(
            self._emit_bounding_box_mode_changed
        )
        self.bbox_table.itemSelectionChanged.connect(self._emit_selection_changed)
        self.bbox_table.itemDoubleClicked.connect(self._emit_item_double_clicked)
        self.open_bounding_boxes_button.clicked.connect(self._emit_open_requested)
        self.save_bounding_boxes_button.clicked.connect(self._emit_save_requested)
        self.delete_bbox_button.clicked.connect(self._emit_delete_requested)
        self.bbox_label_combo.currentIndexChanged.connect(self._emit_label_changed)
        self.median_filter_selected_button.clicked.connect(
            self._emit_median_filter_requested
        )
        self.erosion_selected_button.clicked.connect(self._emit_erosion_requested)
        self.dilation_selected_button.clicked.connect(self._emit_dilation_requested)
        self.erase_bbox_segmentation_button.clicked.connect(
            self._emit_erase_segmentation_requested
        )

    def on_bounding_box_mode_changed(
        self,
        callback: Callable[[bool], None],
    ) -> None:
        self._on_bounding_box_mode_changed = callback

    def on_selection_changed(self, callback: Callable[[], None]) -> None:
        self._on_selection_changed = callback

    def on_item_double_clicked(
        self,
        callback: Callable[[QTableWidgetItem], None],
    ) -> None:
        self._on_item_double_clicked = callback

    def on_open_requested(self, callback: Callable[[], None]) -> None:
        self._on_open_requested = callback

    def on_save_requested(self, callback: Callable[[], None]) -> None:
        self._on_save_requested = callback

    def on_delete_requested(self, callback: Callable[[], None]) -> None:
        self._on_delete_requested = callback

    def on_label_changed(self, callback: Callable[[int], None]) -> None:
        self._on_label_changed = callback

    def on_median_filter_requested(self, callback: Callable[[], None]) -> None:
        self._on_median_filter_requested = callback

    def on_erosion_requested(self, callback: Callable[[], None]) -> None:
        self._on_erosion_requested = callback

    def on_dilation_requested(self, callback: Callable[[], None]) -> None:
        self._on_dilation_requested = callback

    def on_erase_segmentation_requested(self, callback: Callable[[], None]) -> None:
        self._on_erase_segmentation_requested = callback

    def set_bounding_box_mode(self, enabled: bool) -> None:
        self.bounding_box_mode_toggle.blockSignals(True)
        self.bounding_box_mode_toggle.setChecked(bool(enabled))
        self.bounding_box_mode_toggle.blockSignals(False)

    def set_rows(self, rows: Tuple[Tuple[str, str, str, str, str], ...]) -> None:
        self.bbox_table.blockSignals(True)
        self.bbox_table.setRowCount(len(rows))
        for row_index, row_values in enumerate(rows):
            for column_index, value in enumerate(row_values):
                self.bbox_table.setItem(
                    row_index,
                    column_index,
                    QTableWidgetItem(str(value)),
                )
        self.bbox_table.blockSignals(False)

    def set_selected_rows(self, row_indices: Tuple[int, ...]) -> None:
        self.bbox_table.blockSignals(True)
        self.bbox_table.clearSelection()
        selection_model = self.bbox_table.selectionModel()
        for row_index in row_indices:
            if selection_model is None:
                self.bbox_table.selectRow(row_index)
                continue
            index = self.bbox_table.model().index(row_index, 0)
            selection_model.select(
                index,
                QItemSelectionModel.SelectionFlag.Select
                | QItemSelectionModel.SelectionFlag.Rows,
            )
        self.bbox_table.blockSignals(False)

    def selected_row_indices(self) -> Tuple[int, ...]:
        return tuple(sorted({item.row() for item in self.bbox_table.selectedItems()}))

    def set_selected_label_value(self, label: Optional[str]) -> None:
        if label is None:
            index = -1
        else:
            index = self.bbox_label_combo.findData(str(label))
            if index < 0:
                index = -1
        self.bbox_label_combo.blockSignals(True)
        self.bbox_label_combo.setCurrentIndex(index)
        self.bbox_label_combo.blockSignals(False)

    def selected_label_value(self) -> object:
        return self.bbox_label_combo.currentData()

    def set_mode_control_enabled(self, enabled: bool) -> None:
        self.bounding_box_mode_toggle.setEnabled(bool(enabled))

    def set_controls_state(
        self,
        *,
        editing_locked: bool,
        has_boxes: bool,
        has_selected_box: bool,
    ) -> None:
        locked = bool(editing_locked)
        self.open_bounding_boxes_button.setEnabled(not locked)
        self.save_bounding_boxes_button.setEnabled(bool(has_boxes and not locked))
        bbox_editing_enabled = bool(has_selected_box and not locked)
        self.delete_bbox_button.setEnabled(bbox_editing_enabled)
        self.bbox_label_label.setEnabled(bbox_editing_enabled)
        self.bbox_label_combo.setEnabled(bbox_editing_enabled)
        self.median_filter_selected_button.setEnabled(not locked)
        self.erosion_selected_button.setEnabled(not locked)
        self.dilation_selected_button.setEnabled(not locked)
        self.erase_bbox_segmentation_button.setEnabled(not locked)

    def update_table_width(self, minimum_table_width: int) -> None:
        self.bbox_table.resizeColumnsToContents()
        frame = int(self.bbox_table.frameWidth()) * 2
        vertical_header = int(self.bbox_table.verticalHeader().width())
        column_widths = sum(
            int(self.bbox_table.columnWidth(i))
            for i in range(self.bbox_table.columnCount())
        )
        padding = 12
        content_width = frame + vertical_header + column_widths + padding
        minimum_width = max(int(minimum_table_width), int(content_width))
        self.bbox_table.setMinimumWidth(minimum_width)

    def _emit_bounding_box_mode_changed(self, enabled: bool) -> None:
        if self._on_bounding_box_mode_changed is not None:
            self._on_bounding_box_mode_changed(bool(enabled))

    def _emit_selection_changed(self) -> None:
        if self._on_selection_changed is not None:
            self._on_selection_changed()

    def _emit_item_double_clicked(self, item: QTableWidgetItem) -> None:
        if self._on_item_double_clicked is not None:
            self._on_item_double_clicked(item)

    def _emit_open_requested(self) -> None:
        if self._on_open_requested is not None:
            self._on_open_requested()

    def _emit_save_requested(self) -> None:
        if self._on_save_requested is not None:
            self._on_save_requested()

    def _emit_delete_requested(self) -> None:
        if self._on_delete_requested is not None:
            self._on_delete_requested()

    def _emit_label_changed(self, index: int) -> None:
        if self._on_label_changed is not None:
            self._on_label_changed(int(index))

    def _emit_median_filter_requested(self) -> None:
        if self._on_median_filter_requested is not None:
            self._on_median_filter_requested()

    def _emit_erosion_requested(self) -> None:
        if self._on_erosion_requested is not None:
            self._on_erosion_requested()

    def _emit_dilation_requested(self) -> None:
        if self._on_dilation_requested is not None:
            self._on_dilation_requested()

    def _emit_erase_segmentation_requested(self) -> None:
        if self._on_erase_segmentation_requested is not None:
            self._on_erase_segmentation_requested()


class LearningPanel(QGroupBox):
    def __init__(self) -> None:
        super().__init__("Learning")
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

        self.load_model_button = QPushButton("Load Model")
        self.save_model_button = QPushButton("Save Model")
        self.segment_inference_button = QPushButton("Segment Inference BBox")
        self.segment_inference_headless_close_button = QPushButton(
            "Segment Inference Headless and Close"
        )
        self.stop_inference_button = QPushButton("Stop Inference")
        self.stop_inference_button.setEnabled(False)
        self.train_model_button = QPushButton("Train Model")
        self.train_model_headless_close_button = QPushButton("Train Headless and Close")
        self.stop_training_button = QPushButton("Stop Training")
        self.stop_training_button.setEnabled(False)
        self.change_training_parameters_button = QPushButton(
            "Change default training parameters"
        )
        self.training_status = QLabel("Training: Idle")

        layout = QVBoxLayout()
        controls_layout = QGridLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addWidget(self.load_model_button, 0, 0)
        controls_layout.addWidget(self.save_model_button, 0, 1)
        controls_layout.addWidget(self.segment_inference_button, 1, 0)
        controls_layout.addWidget(self.stop_inference_button, 1, 1)
        controls_layout.addWidget(self.segment_inference_headless_close_button, 2, 0, 1, 2)
        controls_layout.addWidget(self.train_model_button, 3, 0)
        controls_layout.addWidget(self.stop_training_button, 3, 1)
        controls_layout.addWidget(self.train_model_headless_close_button, 4, 0, 1, 2)
        controls_layout.addWidget(self.change_training_parameters_button, 5, 0, 1, 2)
        controls_layout.addWidget(self.training_status, 6, 0, 1, 2)
        layout.addLayout(controls_layout)
        self.setLayout(layout)

        self.load_model_button.clicked.connect(self._emit_load_model_requested)
        self.save_model_button.clicked.connect(self._emit_save_model_requested)
        self.segment_inference_button.clicked.connect(
            self._emit_segment_inference_requested
        )
        self.segment_inference_headless_close_button.clicked.connect(
            self._emit_segment_inference_headless_close_requested
        )
        self.stop_inference_button.clicked.connect(self._emit_stop_inference_requested)
        self.train_model_button.clicked.connect(self._emit_train_model_requested)
        self.train_model_headless_close_button.clicked.connect(
            self._emit_train_model_headless_close_requested
        )
        self.stop_training_button.clicked.connect(self._emit_stop_training_requested)
        self.change_training_parameters_button.clicked.connect(
            self._emit_change_training_parameters_requested
        )

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

    def on_change_training_parameters_requested(
        self,
        callback: Callable[[], None],
    ) -> None:
        self._on_change_training_parameters_requested = callback

    def set_controls_state(
        self,
        *,
        editing_locked: bool,
        segment_inference_enabled: bool,
        train_model_enabled: bool,
        stop_training_enabled: bool,
        stop_inference_enabled: bool,
    ) -> None:
        locked = bool(editing_locked)
        self.load_model_button.setEnabled(not locked)
        self.save_model_button.setEnabled(not locked)
        self.change_training_parameters_button.setEnabled(not locked)
        self.segment_inference_button.setEnabled(
            bool(segment_inference_enabled and not locked)
        )
        self.segment_inference_headless_close_button.setEnabled(
            bool(segment_inference_enabled and not locked)
        )
        self.train_model_button.setEnabled(bool(train_model_enabled and not locked))
        self.train_model_headless_close_button.setEnabled(
            bool(train_model_enabled and not locked)
        )
        self.stop_training_button.setEnabled(bool(stop_training_enabled and not locked))
        self.stop_inference_button.setEnabled(bool(stop_inference_enabled))

    def set_training_running(self, running: bool) -> None:
        status = "Running" if bool(running) else "Idle"
        self.training_status.setText(f"Training: {status}")

    def _emit_load_model_requested(self) -> None:
        if self._on_load_model_requested is not None:
            self._on_load_model_requested()

    def _emit_save_model_requested(self) -> None:
        if self._on_save_model_requested is not None:
            self._on_save_model_requested()

    def _emit_segment_inference_requested(self) -> None:
        if self._on_segment_inference_requested is not None:
            self._on_segment_inference_requested()

    def _emit_segment_inference_headless_close_requested(self) -> None:
        if self._on_segment_inference_headless_close_requested is not None:
            self._on_segment_inference_headless_close_requested()

    def _emit_stop_inference_requested(self) -> None:
        if self._on_stop_inference_requested is not None:
            self._on_stop_inference_requested()

    def _emit_train_model_requested(self) -> None:
        if self._on_train_model_requested is not None:
            self._on_train_model_requested()

    def _emit_train_model_headless_close_requested(self) -> None:
        if self._on_train_model_headless_close_requested is not None:
            self._on_train_model_headless_close_requested()

    def _emit_stop_training_requested(self) -> None:
        if self._on_stop_training_requested is not None:
            self._on_stop_training_requested()

    def _emit_change_training_parameters_requested(self) -> None:
        if self._on_change_training_parameters_requested is not None:
            self._on_change_training_parameters_requested()


class HistoryPanel(QGroupBox):
    def __init__(self) -> None:
        super().__init__("History")
        self._on_undo_requested: Optional[Callable[[], None]] = None
        self._on_redo_requested: Optional[Callable[[], None]] = None

        self.undo_button = QPushButton("Undo")
        self.redo_button = QPushButton("Redo")

        layout = QGridLayout()
        layout.addWidget(self.undo_button, 0, 0)
        layout.addWidget(self.redo_button, 0, 1)
        self.setLayout(layout)

        self.undo_button.clicked.connect(self._emit_undo_requested)
        self.redo_button.clicked.connect(self._emit_redo_requested)

    def on_undo_requested(self, callback: Callable[[], None]) -> None:
        self._on_undo_requested = callback

    def on_redo_requested(self, callback: Callable[[], None]) -> None:
        self._on_redo_requested = callback

    def set_undo_state(
        self,
        *,
        depth: int,
        requested_enabled: bool,
        editing_locked: bool,
    ) -> None:
        normalized_depth = max(0, int(depth))
        self.undo_button.setText(f"Undo ({normalized_depth})")
        self.undo_button.setEnabled(
            bool(requested_enabled and not editing_locked and normalized_depth > 0)
        )

    def set_redo_state(
        self,
        *,
        depth: int,
        requested_enabled: bool,
        editing_locked: bool,
    ) -> None:
        normalized_depth = max(0, int(depth))
        self.redo_button.setText(f"Redo ({normalized_depth})")
        self.redo_button.setEnabled(
            bool(requested_enabled and not editing_locked and normalized_depth > 0)
        )

    def _emit_undo_requested(self) -> None:
        if self._on_undo_requested is not None:
            self._on_undo_requested()

    def _emit_redo_requested(self) -> None:
        if self._on_redo_requested is not None:
            self._on_redo_requested()

__all__ = [
    "BOTTOM_PANEL_SUBPANEL_ORDER",
    "BOTTOM_PANEL_SUBPANEL_SPECS",
    "AnnotationPanel",
    "BoundingBoxesPanel",
    "BottomPanelSubpanelSpec",
    "ContrastPanel",
    "FilesPanel",
    "HistoryPanel",
    "LearningPanel",
    "NavigationPanel",
]
