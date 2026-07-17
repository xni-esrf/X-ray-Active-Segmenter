from __future__ import annotations

import unittest

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent

from src.bbox import BoundingBox
from src.tests.bottom_panel_test_utils import (
    QApplication,
    BottomPanelTestCase,
    QAbstractItemView,
)


def _make_wheel_event(angle_delta_y: int) -> QWheelEvent:
    return QWheelEvent(
        QPointF(5, 5),
        QPointF(5, 5),
        QPoint(0, 0),
        QPoint(0, angle_delta_y),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.ScrollUpdate,
        False,
    )


class BottomPanelBoundingBoxesTests(BottomPanelTestCase):
    def test_set_bounding_boxes_populates_table(self) -> None:
        box1, box2 = self._boxes()
        self.panel.set_bounding_boxes((box1, box2))
        table = self.panel._bounding_boxes_panel.bbox_table

        self.assertEqual(table.rowCount(), 2)
        self.assertEqual(table.item(0, 0).text(), "1")
        self.assertEqual(table.item(0, 1).text(), "bbox_0001")
        self.assertEqual(table.item(0, 2).text(), "train")
        self.assertEqual(table.item(0, 3).text(), "3 x 4 x 5")
        self.assertEqual(table.item(0, 4).text(), "(2.00, 3.50, 5.00)")
        self.assertEqual(table.item(1, 0).text(), "2")
        self.assertEqual(table.item(1, 1).text(), "bbox_0002")
        self.assertEqual(table.item(1, 2).text(), "train")

    def test_bbox_table_uses_extended_row_selection_mode(self) -> None:
        self.assertIsNotNone(QAbstractItemView)
        self.assertEqual(
            self.panel._bounding_boxes_panel.bbox_table.selectionMode(),
            QAbstractItemView.ExtendedSelection,
        )
        self.assertEqual(
            self.panel._bounding_boxes_panel.bbox_table.selectionBehavior(),
            QAbstractItemView.SelectRows,
        )

    def test_row_selection_emits_callback_and_delete_requests_selected_ids(self) -> None:
        box1, box2 = self._boxes()
        selected_many_events = []
        delete_many_events = []
        self.panel.on_bounding_boxes_selected(selected_many_events.append)
        self.panel.on_bounding_boxes_delete_requested(delete_many_events.append)
        self.panel.set_bounding_boxes((box1, box2))

        self.panel._bounding_boxes_panel.bbox_table.selectRow(1)
        QApplication.processEvents()
        self.assertEqual(self.panel.state.bbox_selected_ids, ("bbox_0002",))
        self.assertEqual(selected_many_events[-1], ("bbox_0002",))
        self.assertTrue(self.panel._bounding_boxes_panel.delete_bbox_button.isEnabled())

        self.panel._bounding_boxes_panel.delete_bbox_button.click()
        QApplication.processEvents()
        self.assertEqual(delete_many_events, [("bbox_0002",)])

    def test_selection_is_cleared_when_selected_box_disappears(self) -> None:
        box1, box2 = self._boxes()
        self.panel.set_bounding_boxes((box1, box2))
        self.panel.set_selected_bounding_box("bbox_0002")

        self.assertEqual(self.panel.selected_bounding_box(), "bbox_0002")
        self.assertEqual(self.panel.selected_bounding_boxes(), ("bbox_0002",))
        self.panel.set_bounding_boxes((box1,))
        self.assertIsNone(self.panel.selected_bounding_box())
        self.assertEqual(self.panel.selected_bounding_boxes(), tuple())
        self.assertFalse(self.panel._bounding_boxes_panel.delete_bbox_button.isEnabled())

    def test_multi_selection_has_no_primary_selected_box(self) -> None:
        box1, box2 = self._boxes()
        self.panel.set_bounding_boxes((box1, box2))
        self.panel.set_selected_bounding_boxes(("bbox_0002", "bbox_0001", "bbox_0002"))

        self.assertEqual(
            self.panel.selected_bounding_boxes(),
            ("bbox_0001", "bbox_0002"),
        )
        self.assertIsNone(self.panel.selected_bounding_box())
        self.assertEqual(self.panel.selected_bounding_box_label(), "train")

    def test_multi_selection_emits_plural_callback(self) -> None:
        box1, box2 = self._boxes()
        selected_many_events = []
        self.panel.on_bounding_boxes_selected(selected_many_events.append)
        self.panel.set_bounding_boxes((box1, box2))
        self.panel.set_selected_bounding_boxes(("bbox_0001", "bbox_0002"))

        self.panel._handle_bounding_box_selection_changed()

        self.assertEqual(selected_many_events[-1], ("bbox_0001", "bbox_0002"))

    def test_bbox_table_item_double_click_emits_clicked_row_id(self) -> None:
        box1, box2 = self._boxes()
        double_click_events = []
        self.panel.on_bounding_box_double_clicked(double_click_events.append)
        self.panel.set_bounding_boxes((box1, box2))

        item = self.panel._bounding_boxes_panel.bbox_table.item(1, 3)
        self.assertIsNotNone(item)
        self.panel._bounding_boxes_panel.bbox_table.itemDoubleClicked.emit(item)
        QApplication.processEvents()

        self.assertEqual(double_click_events, ["bbox_0002"])

    def test_bbox_table_item_double_click_uses_clicked_row_not_current_selection(self) -> None:
        box1, box2 = self._boxes()
        double_click_events = []
        self.panel.on_bounding_box_double_clicked(double_click_events.append)
        self.panel.set_bounding_boxes((box1, box2))
        self.panel._bounding_boxes_panel.bbox_table.selectRow(0)
        QApplication.processEvents()

        item = self.panel._bounding_boxes_panel.bbox_table.item(1, 1)
        self.assertIsNotNone(item)
        self.panel._bounding_boxes_panel.bbox_table.itemDoubleClicked.emit(item)
        QApplication.processEvents()

        self.assertEqual(double_click_events, ["bbox_0002"])

    def test_bbox_table_item_double_click_ignores_invalid_item(self) -> None:
        double_click_events = []
        self.panel.on_bounding_box_double_clicked(double_click_events.append)

        self.panel._handle_bounding_box_double_clicked(None)  # type: ignore[arg-type]

        self.assertEqual(double_click_events, [])

    def test_multi_selection_with_mixed_labels_shows_neutral_label_and_stays_editable(self) -> None:
        box1, box2 = self._boxes()
        box2 = BoundingBox.from_bounds(
            box_id=box2.id,
            z0=box2.z0,
            z1=box2.z1,
            y0=box2.y0,
            y1=box2.y1,
            x0=box2.x0,
            x1=box2.x1,
            label="validation",
            volume_shape=(20, 30, 40),
        )
        self.panel.set_bounding_boxes((box1, box2))
        self.panel.set_selected_bounding_boxes(("bbox_0001", "bbox_0002"))

        self.assertIsNone(self.panel.selected_bounding_box())
        self.assertIsNone(self.panel.selected_bounding_box_label())
        self.assertTrue(self.panel._bounding_boxes_panel.bbox_label_combo.isEnabled())
        self.assertEqual(self.panel._bounding_boxes_panel.bbox_label_combo.currentIndex(), -1)

    def test_label_editor_emits_plural_callback_for_mixed_multi_selection(self) -> None:
        box1, box2 = self._boxes()
        box2 = BoundingBox.from_bounds(
            box_id=box2.id,
            z0=box2.z0,
            z1=box2.z1,
            y0=box2.y0,
            y1=box2.y1,
            x0=box2.x0,
            x1=box2.x1,
            label="validation",
            volume_shape=(20, 30, 40),
        )
        label_many_events = []
        self.panel.set_bounding_boxes((box1, box2))
        self.panel.on_bounding_boxes_label_changed(
            lambda box_ids, label: label_many_events.append((box_ids, label))
        )
        self.panel.set_selected_bounding_boxes(("bbox_0001", "bbox_0002"))

        inference_index = self.panel._bounding_boxes_panel.bbox_label_combo.findData("inference")
        self.assertGreaterEqual(inference_index, 0)
        self.panel._bounding_boxes_panel.bbox_label_combo.setCurrentIndex(inference_index)
        QApplication.processEvents()

        self.assertEqual(
            label_many_events,
            [(("bbox_0001", "bbox_0002"), "inference")],
        )

    def test_delete_button_emits_plural_callback_for_multi_selection(self) -> None:
        box1, box2 = self._boxes()
        delete_many_events = []
        self.panel.set_bounding_boxes((box1, box2))
        self.panel.on_bounding_boxes_delete_requested(delete_many_events.append)
        self.panel.set_selected_bounding_boxes(("bbox_0001", "bbox_0002"))

        self.panel._bounding_boxes_panel.delete_bbox_button.click()
        QApplication.processEvents()

        self.assertEqual(delete_many_events, [("bbox_0001", "bbox_0002")])

    def test_metadata_updates_after_geometry_change(self) -> None:
        box1, _ = self._boxes()
        self.panel.set_bounding_boxes((box1,))
        table = self.panel._bounding_boxes_panel.bbox_table
        self.assertEqual(table.item(0, 3).text(), "3 x 4 x 5")

        updated = box1.move_face("x_max", 12, volume_shape=(20, 30, 40))
        self.panel.set_bounding_boxes((updated,))
        self.assertEqual(table.item(0, 3).text(), "3 x 4 x 9")
        self.assertEqual(table.item(0, 4).text(), "(2.00, 3.50, 7.00)")

    def test_label_editor_emits_plural_callback_for_selected_box(self) -> None:
        label_many_events = []
        box1, _ = self._boxes()
        self.panel.set_bounding_boxes((box1,))
        self.panel.on_bounding_boxes_label_changed(
            lambda box_ids, label: label_many_events.append((box_ids, label))
        )
        self.panel.set_selected_bounding_box("bbox_0001")

        validation_index = self.panel._bounding_boxes_panel.bbox_label_combo.findData("validation")
        self.assertGreaterEqual(validation_index, 0)
        self.panel._bounding_boxes_panel.bbox_label_combo.setCurrentIndex(validation_index)
        QApplication.processEvents()

        self.assertEqual(label_many_events, [(("bbox_0001",), "validation")])
        self.assertEqual(self.panel.selected_bounding_box_label(), "validation")

    def test_label_combo_ignores_mouse_wheel(self) -> None:
        label_many_events = []
        box1, _ = self._boxes()
        self.panel.set_bounding_boxes((box1,))
        self.panel.on_bounding_boxes_label_changed(
            lambda box_ids, label: label_many_events.append((box_ids, label))
        )
        self.panel.set_selected_bounding_box("bbox_0001")

        combo = self.panel._bounding_boxes_panel.bbox_label_combo
        combo.show()
        combo.setFocus()
        QApplication.processEvents()
        train_index = combo.findData("train")
        self.assertEqual(combo.currentIndex(), train_index)

        event = _make_wheel_event(angle_delta_y=-120)
        QApplication.sendEvent(combo, event)
        QApplication.processEvents()

        self.assertFalse(event.isAccepted())
        self.assertEqual(combo.currentIndex(), train_index)
        self.assertEqual(label_many_events, [])

    def test_bbox_file_and_processing_buttons_emit_callbacks(self) -> None:
        open_events = []
        save_events = []
        median_filter_events = []
        erosion_events = []
        dilation_events = []
        erase_bbox_segmentation_events = []
        self.assertEqual(
            self.panel._bounding_boxes_panel.median_filter_selected_button.text(),
            "Median Filter Selected",
        )
        self.assertEqual(
            self.panel._bounding_boxes_panel.erosion_selected_button.text(),
            "Erosion Selected",
        )
        self.assertEqual(
            self.panel._bounding_boxes_panel.dilation_selected_button.text(),
            "Dilation Selected",
        )
        self.assertEqual(
            self.panel._bounding_boxes_panel.erase_bbox_segmentation_button.text(),
            "Erase BBox Segmentation",
        )
        self.panel.on_open_bounding_boxes_requested(lambda: open_events.append("open"))
        self.panel.on_save_bounding_boxes_requested(lambda: save_events.append("save"))
        self.panel.on_median_filter_selected_requested(
            lambda: median_filter_events.append("median")
        )
        self.panel.on_erosion_selected_requested(
            lambda: erosion_events.append("erosion")
        )
        self.panel.on_dilation_selected_requested(
            lambda: dilation_events.append("dilation")
        )
        self.panel.on_erase_bbox_segmentation_requested(
            lambda: erase_bbox_segmentation_events.append("erase_bbox_segmentation")
        )

        box1, _ = self._boxes()
        self.panel.set_bounding_boxes((box1,))
        self.panel._bounding_boxes_panel.open_bounding_boxes_button.click()
        self.panel._bounding_boxes_panel.save_bounding_boxes_button.click()
        self.panel._bounding_boxes_panel.median_filter_selected_button.click()
        self.panel._bounding_boxes_panel.erosion_selected_button.click()
        self.panel._bounding_boxes_panel.dilation_selected_button.click()
        self.panel._bounding_boxes_panel.erase_bbox_segmentation_button.click()
        QApplication.processEvents()

        self.assertEqual(open_events, ["open"])
        self.assertEqual(save_events, ["save"])
        self.assertEqual(median_filter_events, ["median"])
        self.assertEqual(erosion_events, ["erosion"])
        self.assertEqual(dilation_events, ["dilation"])
        self.assertEqual(erase_bbox_segmentation_events, ["erase_bbox_segmentation"])

    def test_save_bbox_button_is_disabled_without_boxes(self) -> None:
        self.panel.set_bounding_boxes(tuple())
        self.assertFalse(self.panel._bounding_boxes_panel.save_bounding_boxes_button.isEnabled())

        box1, _ = self._boxes()
        self.panel.set_bounding_boxes((box1,))
        self.assertTrue(self.panel._bounding_boxes_panel.save_bounding_boxes_button.isEnabled())

    def test_bounding_box_tool_uses_dedicated_checkbox(self) -> None:
        index = self.panel._annotation_panel.annotation_tool_combo.findData("bbox")
        self.assertEqual(index, -1)

        events = []
        self.panel.on_bounding_box_mode_changed(events.append)
        self.panel.set_interaction_tools_enabled(True)
        self.panel._bounding_boxes_panel.bounding_box_mode_toggle.setChecked(True)
        QApplication.processEvents()

        self.assertEqual(events, [True])
        self.assertTrue(self.panel.state.bounding_box_mode_enabled)

    def test_inference_navigation_only_mode_disables_mutations_but_keeps_navigation(self) -> None:
        box1, box2 = self._boxes()
        self.panel.set_interaction_tools_enabled(True)
        self.panel.set_annotation_mode(True)
        self.panel.set_annotation_controls_enabled(True)
        self.panel.set_contrast_range((0.0, 255.0))
        self.panel.set_bounding_boxes((box1, box2))
        self.panel.set_selected_bounding_boxes(("bbox_0001",))
        self.panel.set_segment_inference_enabled(True)
        self.panel.set_train_model_enabled(True)
        self.panel.set_stop_inference_enabled(True)
        self.panel.set_undo_state(depth=2, enabled=True)
        self.panel.set_redo_state(depth=1, enabled=True)

        self.panel.set_inference_navigation_only_mode(True)

        self.assertFalse(self.panel._files_panel.open_button.isEnabled())
        self.assertFalse(self.panel._files_panel.open_semantic_button.isEnabled())
        self.assertFalse(self.panel._files_panel.open_instance_button.isEnabled())
        self.assertFalse(self.panel._files_panel.save_segmentation_button.isEnabled())
        self.assertFalse(self.panel._annotation_panel.annotation_toggle.isEnabled())
        self.assertFalse(self.panel._annotation_panel.annotation_tool_combo.isEnabled())
        self.assertFalse(self.panel._annotation_panel.tool_label_edit.isEnabled())
        self.assertFalse(self.panel._bounding_boxes_panel.open_bounding_boxes_button.isEnabled())
        self.assertFalse(self.panel._bounding_boxes_panel.save_bounding_boxes_button.isEnabled())
        self.assertFalse(self.panel._bounding_boxes_panel.delete_bbox_button.isEnabled())
        self.assertFalse(self.panel._bounding_boxes_panel.bbox_label_combo.isEnabled())
        self.assertFalse(self.panel._learning_panel.load_model_button.isEnabled())
        self.assertFalse(self.panel._learning_panel.save_model_button.isEnabled())
        self.assertFalse(self.panel._learning_panel.segment_inference_button.isEnabled())
        self.assertFalse(self.panel._learning_panel.train_model_button.isEnabled())
        self.assertFalse(self.panel._learning_panel.change_training_parameters_button.isEnabled())
        self.assertFalse(self.panel._history_panel.undo_button.isEnabled())
        self.assertFalse(self.panel._history_panel.redo_button.isEnabled())
        self.assertTrue(self.panel._learning_panel.stop_inference_button.isEnabled())

        self.assertTrue(self.panel._navigation_panel.cursor_z.isEnabled())
        self.assertTrue(self.panel._navigation_panel.cursor_y.isEnabled())
        self.assertTrue(self.panel._navigation_panel.cursor_x.isEnabled())
        self.assertTrue(self.panel._navigation_panel.zoom_spin.isEnabled())
        self.assertTrue(self.panel._navigation_panel.auto_level_checkbox.isEnabled())
        self.assertTrue(self.panel._contrast_panel.contrast_min_slider.isEnabled())
        self.assertTrue(self.panel._contrast_panel.contrast_max_slider.isEnabled())
        self.assertTrue(self.panel._bounding_boxes_panel.bbox_table.isEnabled())

        self.panel._bounding_boxes_panel.bbox_table.clearSelection()
        self.panel._bounding_boxes_panel.bbox_table.selectRow(1)
        QApplication.processEvents()
        self.assertEqual(self.panel.selected_bounding_boxes(), ("bbox_0002",))
        self.assertFalse(self.panel._bounding_boxes_panel.delete_bbox_button.isEnabled())

        self.panel.set_inference_navigation_only_mode(False)
        self.assertTrue(self.panel._files_panel.open_button.isEnabled())
        self.assertTrue(self.panel._bounding_boxes_panel.save_bounding_boxes_button.isEnabled())
        self.assertTrue(self.panel._learning_panel.load_model_button.isEnabled())
        self.assertTrue(self.panel._learning_panel.segment_inference_button.isEnabled())
        self.assertTrue(self.panel._learning_panel.train_model_button.isEnabled())
        self.assertTrue(self.panel._learning_panel.change_training_parameters_button.isEnabled())
        self.assertTrue(self.panel._history_panel.undo_button.isEnabled())
        self.assertTrue(self.panel._history_panel.redo_button.isEnabled())


if __name__ == "__main__":
    unittest.main()
