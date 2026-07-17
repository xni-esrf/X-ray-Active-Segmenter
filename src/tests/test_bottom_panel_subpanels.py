from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication
except Exception:  # pragma: no cover - environment dependent
    QApplication = None  # type: ignore[assignment]

try:
    from src.ui.bottom_panel_subpanels import (
        AnnotationPanel,
        BoundingBoxesPanel,
        ContrastPanel,
        NavigationPanel,
    )
except Exception:  # pragma: no cover - environment dependent
    AnnotationPanel = None  # type: ignore[assignment]
    BoundingBoxesPanel = None  # type: ignore[assignment]
    ContrastPanel = None  # type: ignore[assignment]
    NavigationPanel = None  # type: ignore[assignment]


@unittest.skipUnless(
    QApplication is not None
    and AnnotationPanel is not None
    and BoundingBoxesPanel is not None
    and ContrastPanel is not None
    and NavigationPanel is not None,
    "PySide6 is not available",
)
class BottomPanelSubpanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_navigation_panel_emits_cursor_zoom_and_view_layout(self) -> None:
        panel = NavigationPanel(zoom_min=0.1, zoom_max=1.0)
        cursor_events = []
        zoom_events = []
        layout_events = []
        panel.on_cursor_changed(cursor_events.append)
        panel.on_zoom_changed(zoom_events.append)
        panel.on_view_layout_mode_changed(layout_events.append)

        panel.set_cursor_range((3, 4, 5))
        panel.cursor_z.setValue(2)
        panel.cursor_y.setValue(3)
        panel.cursor_x.setValue(4)
        panel.zoom_spin.setValue(0.5)
        panel.view_layout_coronal_radio.setChecked(True)
        QApplication.processEvents()

        self.assertEqual(cursor_events[-1], (2, 3, 4))
        self.assertAlmostEqual(zoom_events[-1], 0.5)
        self.assertEqual(layout_events[-1], "coronal")

    def test_annotation_panel_emits_normalized_local_values(self) -> None:
        panel = AnnotationPanel()
        mode_events = []
        tool_events = []
        label_events = []
        radius_events = []
        flood_events = []
        panel.on_annotation_mode_changed(mode_events.append)
        panel.on_annotation_tool_changed(tool_events.append)
        panel.on_tool_label_changed(label_events.append)
        panel.on_brush_radius_changed(radius_events.append)
        panel.on_flood_fill_requested(flood_events.append)

        panel.annotation_toggle.setChecked(True)
        panel.annotation_tool_combo.setCurrentIndex(
            panel.annotation_tool_combo.findData("flood_filler")
        )
        panel.tool_label_edit.setText(" 42 ")
        panel.tool_label_edit.editingFinished.emit()
        panel.brush_radius_spin.setValue(5)
        panel.flood_fill_button.click()
        QApplication.processEvents()

        self.assertEqual(mode_events, [True])
        self.assertEqual(tool_events[-1], "flood_filler")
        self.assertEqual(label_events, ["42"])
        self.assertEqual(radius_events[-1], 5)
        self.assertEqual(flood_events, [42])

    def test_contrast_panel_updates_labels_and_emits_steps(self) -> None:
        panel = ContrastPanel(
            contrast_max_step=999,
            segmentation_opacity_max_step=100,
        )
        min_events = []
        max_events = []
        opacity_events = []
        panel.on_contrast_min_changed(min_events.append)
        panel.on_contrast_max_changed(max_events.append)
        panel.on_segmentation_opacity_changed(opacity_events.append)

        panel.set_contrast_labels((1.25, 9.5))
        panel.contrast_min_slider.setValue(123)
        panel.contrast_max_slider.setValue(777)
        panel.segmentation_opacity_slider.setValue(65)
        QApplication.processEvents()

        self.assertEqual(panel.contrast_min_value.text(), "Min: 1.25")
        self.assertEqual(panel.contrast_max_value.text(), "Max: 9.5")
        self.assertEqual(min_events[-1], 123)
        self.assertEqual(max_events[-1], 777)
        self.assertEqual(opacity_events[-1], 65)
        self.assertEqual(panel.segmentation_opacity_value.text(), "65%")

    def test_bounding_boxes_panel_emits_local_actions_and_selection(self) -> None:
        panel = BoundingBoxesPanel()
        mode_events = []
        selection_events = []
        double_click_events = []
        open_events = []
        save_events = []
        delete_events = []
        label_events = []
        median_events = []
        erosion_events = []
        dilation_events = []
        erase_events = []
        panel.on_bounding_box_mode_changed(mode_events.append)
        panel.on_selection_changed(lambda: selection_events.append(panel.selected_row_indices()))
        panel.on_item_double_clicked(lambda item: double_click_events.append(item.row()))
        panel.on_open_requested(lambda: open_events.append("open"))
        panel.on_save_requested(lambda: save_events.append("save"))
        panel.on_delete_requested(lambda: delete_events.append("delete"))
        panel.on_label_changed(label_events.append)
        panel.on_median_filter_requested(lambda: median_events.append("median"))
        panel.on_erosion_requested(lambda: erosion_events.append("erosion"))
        panel.on_dilation_requested(lambda: dilation_events.append("dilation"))
        panel.on_erase_segmentation_requested(lambda: erase_events.append("erase"))

        panel.set_rows(
            (
                ("1", "bbox_0001", "train", "3 x 4 x 5", "(2.00, 3.50, 5.00)"),
                ("2", "bbox_0002", "validation", "4 x 6 x 5", "(7.00, 9.00, 12.00)"),
            )
        )
        panel.set_controls_state(
            editing_locked=False,
            has_boxes=True,
            has_selected_box=True,
        )
        panel.bounding_box_mode_toggle.setChecked(True)
        panel.bbox_table.selectRow(1)
        panel.bbox_table.itemDoubleClicked.emit(panel.bbox_table.item(1, 1))
        panel.open_bounding_boxes_button.click()
        panel.save_bounding_boxes_button.click()
        panel.delete_bbox_button.click()
        panel.bbox_label_combo.setCurrentIndex(panel.bbox_label_combo.findData("validation"))
        panel.median_filter_selected_button.click()
        panel.erosion_selected_button.click()
        panel.dilation_selected_button.click()
        panel.erase_bbox_segmentation_button.click()
        QApplication.processEvents()

        self.assertEqual(mode_events, [True])
        self.assertEqual(selection_events[-1], (1,))
        self.assertEqual(double_click_events, [1])
        self.assertEqual(open_events, ["open"])
        self.assertEqual(save_events, ["save"])
        self.assertEqual(delete_events, ["delete"])
        self.assertEqual(label_events[-1], panel.bbox_label_combo.findData("validation"))
        self.assertEqual(median_events, ["median"])
        self.assertEqual(erosion_events, ["erosion"])
        self.assertEqual(dilation_events, ["dilation"])
        self.assertEqual(erase_events, ["erase"])

    def test_bbox_table_resize_handle_drags_table_height_within_bounds(self) -> None:
        from PySide6.QtCore import QEvent, QPointF, Qt
        from PySide6.QtGui import QMouseEvent

        panel = BoundingBoxesPanel()
        table = panel.bbox_table
        handle = panel.bbox_table_resize_handle
        self.assertEqual(table.height(), panel._DEFAULT_BBOX_TABLE_HEIGHT)

        def _mouse_event(event_type, y, buttons):
            point = QPointF(2, y)
            return QMouseEvent(event_type, point, point, Qt.LeftButton, buttons, Qt.NoModifier)

        handle.mousePressEvent(_mouse_event(QEvent.MouseButtonPress, 2, Qt.LeftButton))
        handle.mouseMoveEvent(_mouse_event(QEvent.MouseMove, 82, Qt.LeftButton))
        self.assertEqual(table.height(), panel._DEFAULT_BBOX_TABLE_HEIGHT + 80)
        handle.mouseReleaseEvent(_mouse_event(QEvent.MouseButtonRelease, 82, Qt.NoButton))

        handle.mousePressEvent(_mouse_event(QEvent.MouseButtonPress, 2, Qt.LeftButton))
        handle.mouseMoveEvent(_mouse_event(QEvent.MouseMove, -10_000, Qt.LeftButton))
        self.assertEqual(table.height(), panel._MIN_BBOX_TABLE_HEIGHT)
        handle.mouseMoveEvent(_mouse_event(QEvent.MouseMove, 10_000, Qt.LeftButton))
        self.assertEqual(table.height(), panel._MAX_BBOX_TABLE_HEIGHT)


if __name__ == "__main__":
    unittest.main()
