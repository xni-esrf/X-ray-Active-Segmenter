from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication, QAbstractItemView
except Exception:  # pragma: no cover - environment dependent
    QApplication = None  # type: ignore[assignment]
    QAbstractItemView = None  # type: ignore[assignment]

from src.bbox import BoundingBox
try:
    from src.ui.bottom_panel import BottomPanel
except Exception:  # pragma: no cover - environment dependent
    BottomPanel = None  # type: ignore[assignment]


@unittest.skipUnless(QApplication is not None and BottomPanel is not None, "PySide6 is not available")
class BottomPanelBoundingBoxesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.panel = BottomPanel()

    def _boxes(self) -> tuple[BoundingBox, BoundingBox]:
        box1 = BoundingBox.from_bounds(
            box_id="bbox_0001",
            z0=1,
            z1=4,
            y0=2,
            y1=6,
            x0=3,
            x1=8,
            volume_shape=(20, 30, 40),
        )
        box2 = BoundingBox.from_bounds(
            box_id="bbox_0002",
            z0=5,
            z1=9,
            y0=6,
            y1=12,
            x0=10,
            x1=15,
            volume_shape=(20, 30, 40),
        )
        return (box1, box2)

    def test_set_bounding_boxes_populates_table(self) -> None:
        box1, box2 = self._boxes()
        self.panel.set_bounding_boxes((box1, box2))

        self.assertEqual(self.panel._bbox_table.rowCount(), 2)
        self.assertEqual(self.panel._bbox_table.item(0, 0).text(), "bbox_0001")
        self.assertEqual(self.panel._bbox_table.item(0, 1).text(), "train")
        self.assertEqual(self.panel._bbox_table.item(0, 2).text(), "3 x 4 x 5")
        self.assertEqual(self.panel._bbox_table.item(0, 3).text(), "(2.00, 3.50, 5.00)")
        self.assertEqual(self.panel._bbox_table.item(1, 0).text(), "bbox_0002")
        self.assertEqual(self.panel._bbox_table.item(1, 1).text(), "train")

    def test_bbox_table_uses_extended_row_selection_mode(self) -> None:
        self.assertIsNotNone(QAbstractItemView)
        self.assertEqual(
            self.panel._bbox_table.selectionMode(),
            QAbstractItemView.ExtendedSelection,
        )
        self.assertEqual(
            self.panel._bbox_table.selectionBehavior(),
            QAbstractItemView.SelectRows,
        )

    def test_row_selection_emits_callback_and_delete_requests_selected_id(self) -> None:
        box1, box2 = self._boxes()
        selected_events = []
        selected_many_events = []
        delete_events = []
        delete_many_events = []
        self.panel.on_bounding_box_selected(selected_events.append)
        self.panel.on_bounding_boxes_selected(selected_many_events.append)
        self.panel.on_bounding_box_delete_requested(delete_events.append)
        self.panel.on_bounding_boxes_delete_requested(delete_many_events.append)
        self.panel.set_bounding_boxes((box1, box2))

        self.panel._bbox_table.selectRow(1)
        QApplication.processEvents()
        self.assertEqual(self.panel.state.bbox_selected_id, "bbox_0002")
        self.assertEqual(self.panel.state.bbox_selected_ids, ("bbox_0002",))
        self.assertEqual(selected_events[-1], "bbox_0002")
        self.assertEqual(selected_many_events[-1], ("bbox_0002",))
        self.assertTrue(self.panel._delete_bbox_button.isEnabled())

        self.panel._delete_bbox_button.click()
        QApplication.processEvents()
        self.assertEqual(delete_events, ["bbox_0002"])
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
        self.assertFalse(self.panel._delete_bbox_button.isEnabled())

    def test_multi_selection_has_no_primary_selected_box(self) -> None:
        box1, box2 = self._boxes()
        self.panel.set_bounding_boxes((box1, box2))
        self.panel.set_selected_bounding_boxes(("bbox_0002", "bbox_0001", "bbox_0002"))

        self.assertEqual(
            self.panel.selected_bounding_boxes(),
            ("bbox_0001", "bbox_0002"),
        )
        self.assertIsNone(self.panel.state.bbox_selected_id)
        self.assertIsNone(self.panel.selected_bounding_box())
        self.assertEqual(self.panel.selected_bounding_box_label(), "train")

    def test_multi_selection_emits_plural_callback_and_no_single_primary(self) -> None:
        box1, box2 = self._boxes()
        selected_events = []
        selected_many_events = []
        self.panel.on_bounding_box_selected(selected_events.append)
        self.panel.on_bounding_boxes_selected(selected_many_events.append)
        self.panel.set_bounding_boxes((box1, box2))
        self.panel.set_selected_bounding_boxes(("bbox_0001", "bbox_0002"))

        self.panel._handle_bounding_box_selection_changed()

        self.assertEqual(selected_many_events[-1], ("bbox_0001", "bbox_0002"))
        self.assertIsNone(selected_events[-1])

    def test_bbox_table_item_double_click_emits_clicked_row_id(self) -> None:
        box1, box2 = self._boxes()
        double_click_events = []
        self.panel.on_bounding_box_double_clicked(double_click_events.append)
        self.panel.set_bounding_boxes((box1, box2))

        item = self.panel._bbox_table.item(1, 3)
        self.assertIsNotNone(item)
        self.panel._bbox_table.itemDoubleClicked.emit(item)
        QApplication.processEvents()

        self.assertEqual(double_click_events, ["bbox_0002"])

    def test_bbox_table_item_double_click_uses_clicked_row_not_current_selection(self) -> None:
        box1, box2 = self._boxes()
        double_click_events = []
        self.panel.on_bounding_box_double_clicked(double_click_events.append)
        self.panel.set_bounding_boxes((box1, box2))
        self.panel._bbox_table.selectRow(0)
        QApplication.processEvents()

        item = self.panel._bbox_table.item(1, 1)
        self.assertIsNotNone(item)
        self.panel._bbox_table.itemDoubleClicked.emit(item)
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
        self.assertTrue(self.panel._bbox_label_combo.isEnabled())
        self.assertEqual(self.panel._bbox_label_combo.currentIndex(), -1)

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
        label_single_events = []
        self.panel.set_bounding_boxes((box1, box2))
        self.panel.on_bounding_boxes_label_changed(
            lambda box_ids, label: label_many_events.append((box_ids, label))
        )
        self.panel.on_bounding_box_label_changed(
            lambda box_id, label: label_single_events.append((box_id, label))
        )
        self.panel.set_selected_bounding_boxes(("bbox_0001", "bbox_0002"))

        inference_index = self.panel._bbox_label_combo.findData("inference")
        self.assertGreaterEqual(inference_index, 0)
        self.panel._bbox_label_combo.setCurrentIndex(inference_index)
        QApplication.processEvents()

        self.assertEqual(
            label_many_events,
            [(("bbox_0001", "bbox_0002"), "inference")],
        )
        self.assertEqual(label_single_events, [])

    def test_delete_button_emits_plural_callback_for_multi_selection_only(self) -> None:
        box1, box2 = self._boxes()
        delete_many_events = []
        delete_single_events = []
        self.panel.set_bounding_boxes((box1, box2))
        self.panel.on_bounding_boxes_delete_requested(delete_many_events.append)
        self.panel.on_bounding_box_delete_requested(delete_single_events.append)
        self.panel.set_selected_bounding_boxes(("bbox_0001", "bbox_0002"))

        self.panel._delete_bbox_button.click()
        QApplication.processEvents()

        self.assertEqual(delete_many_events, [(("bbox_0001", "bbox_0002"))])
        self.assertEqual(delete_single_events, [])

    def test_metadata_updates_after_geometry_change(self) -> None:
        box1, _ = self._boxes()
        self.panel.set_bounding_boxes((box1,))
        self.assertEqual(self.panel._bbox_table.item(0, 2).text(), "3 x 4 x 5")

        updated = box1.move_face("x_max", 12, volume_shape=(20, 30, 40))
        self.panel.set_bounding_boxes((updated,))
        self.assertEqual(self.panel._bbox_table.item(0, 2).text(), "3 x 4 x 9")
        self.assertEqual(self.panel._bbox_table.item(0, 3).text(), "(2.00, 3.50, 7.00)")

    def test_label_editor_emits_callback_for_selected_box(self) -> None:
        label_events = []
        label_many_events = []
        box1, _ = self._boxes()
        self.panel.set_bounding_boxes((box1,))
        self.panel.on_bounding_box_label_changed(
            lambda box_id, label: label_events.append((box_id, label))
        )
        self.panel.on_bounding_boxes_label_changed(
            lambda box_ids, label: label_many_events.append((box_ids, label))
        )
        self.panel.set_selected_bounding_box("bbox_0001")

        validation_index = self.panel._bbox_label_combo.findData("validation")
        self.assertGreaterEqual(validation_index, 0)
        self.panel._bbox_label_combo.setCurrentIndex(validation_index)
        QApplication.processEvents()

        self.assertEqual(label_events, [("bbox_0001", "validation")])
        self.assertEqual(label_many_events, [(("bbox_0001",), "validation")])
        self.assertEqual(self.panel.selected_bounding_box_label(), "validation")

    def test_bbox_file_buttons_emit_callbacks(self) -> None:
        open_events = []
        save_events = []
        build_events = []
        instantiate_events = []
        save_model_events = []
        segment_inference_events = []
        stop_inference_events = []
        train_events = []
        stop_events = []
        median_filter_events = []
        erosion_events = []
        dilation_events = []
        erase_bbox_segmentation_events = []
        self.assertEqual(
            self.panel._build_dataset_from_bboxes_button.text(),
            "Build Dataset from Bbox",
        )
        self.assertEqual(
            self.panel._instantiate_model_button.text(),
            "Load Model",
        )
        self.assertEqual(
            self.panel._save_model_button.text(),
            "Save Model",
        )
        self.assertEqual(
            self.panel._train_model_button.text(),
            "Train Model on Dataset",
        )
        self.assertEqual(
            self.panel._segment_inference_button.text(),
            "Segment Inference Bbox",
        )
        self.assertEqual(
            self.panel._stop_inference_button.text(),
            "Stop Inference",
        )
        self.assertEqual(
            self.panel._stop_training_button.text(),
            "Stop Training",
        )
        self.assertEqual(
            self.panel._median_filter_selected_button.text(),
            "Median Filter Selected",
        )
        self.assertEqual(
            self.panel._erosion_selected_button.text(),
            "Erosion Selected",
        )
        self.assertEqual(
            self.panel._dilation_selected_button.text(),
            "Dilation Selected",
        )
        self.assertEqual(
            self.panel._erase_bbox_segmentation_button.text(),
            "Erase Bbox Segmentation",
        )
        self.assertFalse(self.panel._stop_training_button.isEnabled())
        self.assertFalse(self.panel._stop_inference_button.isEnabled())
        self.assertEqual(
            self.panel._learning_training_status.text(),
            "Training: Idle",
        )
        self.panel.on_open_bounding_boxes_requested(lambda: open_events.append("open"))
        self.panel.on_save_bounding_boxes_requested(lambda: save_events.append("save"))
        self.panel.on_build_dataset_from_bboxes_requested(
            lambda: build_events.append("build")
        )
        self.panel.on_load_model_requested(
            lambda: instantiate_events.append("instantiate")
        )
        self.panel.on_save_model_requested(
            lambda: save_model_events.append("save_model")
        )
        self.panel.on_segment_inference_requested(
            lambda: segment_inference_events.append("segment_inference")
        )
        self.panel.on_stop_inference_requested(
            lambda: stop_inference_events.append("stop_inference")
        )
        self.panel.on_train_model_requested(
            lambda: train_events.append("train")
        )
        self.panel.on_stop_training_requested(
            lambda: stop_events.append("stop")
        )
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
        self.panel.set_stop_inference_enabled(True)
        self.panel.set_stop_training_enabled(True)
        self.panel._open_bounding_boxes_button.click()
        self.panel._save_bounding_boxes_button.click()
        self.panel._build_dataset_from_bboxes_button.click()
        self.panel._instantiate_model_button.click()
        self.panel._save_model_button.click()
        self.panel._segment_inference_button.click()
        self.panel._stop_inference_button.click()
        self.panel._train_model_button.click()
        self.panel._stop_training_button.click()
        self.panel._median_filter_selected_button.click()
        self.panel._erosion_selected_button.click()
        self.panel._dilation_selected_button.click()
        self.panel._erase_bbox_segmentation_button.click()
        QApplication.processEvents()

        self.assertEqual(open_events, ["open"])
        self.assertEqual(save_events, ["save"])
        self.assertEqual(build_events, ["build"])
        self.assertEqual(instantiate_events, ["instantiate"])
        self.assertEqual(save_model_events, ["save_model"])
        self.assertEqual(segment_inference_events, ["segment_inference"])
        self.assertEqual(stop_inference_events, ["stop_inference"])
        self.assertEqual(train_events, ["train"])
        self.assertEqual(stop_events, ["stop"])
        self.assertEqual(median_filter_events, ["median"])
        self.assertEqual(erosion_events, ["erosion"])
        self.assertEqual(dilation_events, ["dilation"])
        self.assertEqual(erase_bbox_segmentation_events, ["erase_bbox_segmentation"])

    def test_learning_training_status_display_updates(self) -> None:
        self.assertFalse(self.panel.learning_training_running())
        self.assertEqual(self.panel._learning_training_status.text(), "Training: Idle")
        self.assertFalse(self.panel._stop_training_button.isEnabled())
        self.assertFalse(self.panel._stop_inference_button.isEnabled())

        self.panel.set_learning_training_running(True)
        self.assertTrue(self.panel.learning_training_running())
        self.assertEqual(self.panel._learning_training_status.text(), "Training: Running")

        self.panel.set_learning_training_running(False)
        self.assertFalse(self.panel.learning_training_running())
        self.assertEqual(self.panel._learning_training_status.text(), "Training: Idle")

        self.panel.set_stop_training_enabled(True)
        self.assertTrue(self.panel._stop_training_button.isEnabled())
        self.panel.set_stop_training_enabled(False)
        self.assertFalse(self.panel._stop_training_button.isEnabled())
        self.panel.set_stop_inference_enabled(True)
        self.assertTrue(self.panel._stop_inference_button.isEnabled())
        self.panel.set_stop_inference_enabled(False)
        self.assertFalse(self.panel._stop_inference_button.isEnabled())

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

        self.assertFalse(self.panel._open_button.isEnabled())
        self.assertFalse(self.panel._open_semantic_button.isEnabled())
        self.assertFalse(self.panel._open_instance_button.isEnabled())
        self.assertFalse(self.panel._save_segmentation_button.isEnabled())
        self.assertFalse(self.panel._annotation_toggle.isEnabled())
        self.assertFalse(self.panel._annotation_tool_combo.isEnabled())
        self.assertFalse(self.panel._active_label_spin.isEnabled())
        self.assertFalse(self.panel._open_bounding_boxes_button.isEnabled())
        self.assertFalse(self.panel._save_bounding_boxes_button.isEnabled())
        self.assertFalse(self.panel._build_dataset_from_bboxes_button.isEnabled())
        self.assertFalse(self.panel._delete_bbox_button.isEnabled())
        self.assertFalse(self.panel._bbox_label_combo.isEnabled())
        self.assertFalse(self.panel._load_model_button.isEnabled())
        self.assertFalse(self.panel._save_model_button.isEnabled())
        self.assertFalse(self.panel._segment_inference_button.isEnabled())
        self.assertFalse(self.panel._train_model_button.isEnabled())
        self.assertFalse(self.panel._undo_button.isEnabled())
        self.assertFalse(self.panel._redo_button.isEnabled())
        self.assertTrue(self.panel._stop_inference_button.isEnabled())

        self.assertTrue(self.panel._cursor_z.isEnabled())
        self.assertTrue(self.panel._cursor_y.isEnabled())
        self.assertTrue(self.panel._cursor_x.isEnabled())
        self.assertTrue(self.panel._zoom_spin.isEnabled())
        self.assertTrue(self.panel._auto_level_checkbox.isEnabled())
        self.assertTrue(self.panel._contrast_min_slider.isEnabled())
        self.assertTrue(self.panel._contrast_max_slider.isEnabled())
        self.assertTrue(self.panel._bbox_table.isEnabled())

        self.panel._bbox_table.clearSelection()
        self.panel._bbox_table.selectRow(1)
        QApplication.processEvents()
        self.assertEqual(self.panel.selected_bounding_boxes(), ("bbox_0002",))
        self.assertFalse(self.panel._delete_bbox_button.isEnabled())

        self.panel.set_inference_navigation_only_mode(False)
        self.assertTrue(self.panel._open_button.isEnabled())
        self.assertTrue(self.panel._save_bounding_boxes_button.isEnabled())
        self.assertTrue(self.panel._load_model_button.isEnabled())
        self.assertTrue(self.panel._segment_inference_button.isEnabled())
        self.assertTrue(self.panel._train_model_button.isEnabled())
        self.assertTrue(self.panel._undo_button.isEnabled())
        self.assertTrue(self.panel._redo_button.isEnabled())

    def test_save_bbox_button_is_disabled_without_boxes(self) -> None:
        self.panel.set_bounding_boxes(tuple())
        self.assertFalse(self.panel._save_bounding_boxes_button.isEnabled())
        self.assertFalse(self.panel._build_dataset_from_bboxes_button.isEnabled())

        box1, _ = self._boxes()
        self.panel.set_bounding_boxes((box1,))
        self.assertTrue(self.panel._save_bounding_boxes_button.isEnabled())
        self.assertTrue(self.panel._build_dataset_from_bboxes_button.isEnabled())

    def test_bounding_box_tool_uses_dedicated_checkbox(self) -> None:
        index = self.panel._annotation_tool_combo.findData("bbox")
        self.assertEqual(index, -1)

        events = []
        self.panel.on_bounding_box_mode_changed(events.append)
        self.panel.set_interaction_tools_enabled(True)
        self.panel._bounding_box_mode_toggle.setChecked(True)
        QApplication.processEvents()

        self.assertEqual(events, [True])
        self.assertTrue(self.panel.state.bounding_box_mode_enabled)

    def test_annotation_tool_controls_show_shortcut_hints(self) -> None:
        brush_index = self.panel._annotation_tool_combo.findData("brush")
        eraser_index = self.panel._annotation_tool_combo.findData("eraser")
        flood_index = self.panel._annotation_tool_combo.findData("flood_filler")
        self.assertGreaterEqual(brush_index, 0)
        self.assertGreaterEqual(eraser_index, 0)
        self.assertGreaterEqual(flood_index, 0)
        self.assertEqual(
            self.panel._annotation_tool_combo.itemText(brush_index),
            "Brush (Ctrl+B)",
        )
        self.assertEqual(
            self.panel._annotation_tool_combo.itemText(eraser_index),
            "Eraser (Ctrl+E)",
        )
        self.assertEqual(
            self.panel._annotation_tool_combo.itemText(flood_index),
            "Flood Fill (Ctrl+F)",
        )
        hint = self.panel._annotation_tool_combo.toolTip()
        self.assertIn("Ctrl+B", hint)
        self.assertIn("Ctrl+E", hint)
        self.assertIn("Ctrl+F", hint)

    def test_history_buttons_are_not_reset_by_annotation_controls(self) -> None:
        self.panel.set_undo_state(depth=2, enabled=True)
        self.panel.set_redo_state(depth=1, enabled=True)
        self.assertTrue(self.panel._undo_button.isEnabled())
        self.assertTrue(self.panel._redo_button.isEnabled())

        self.panel.set_annotation_controls_enabled(False)

        self.assertTrue(self.panel._undo_button.isEnabled())
        self.assertTrue(self.panel._redo_button.isEnabled())

    def test_level_controls_are_disabled_until_interaction_tools_enabled(self) -> None:
        self.panel.set_level_mode(auto_enabled=True, manual_level=0, max_level=4)
        self.assertFalse(self.panel._auto_level_checkbox.isEnabled())
        self.assertFalse(self.panel._manual_level_spin.isEnabled())

        self.panel.set_interaction_tools_enabled(True)
        self.assertTrue(self.panel._auto_level_checkbox.isEnabled())
        self.assertTrue(self.panel._auto_level_checkbox.isChecked())
        self.assertFalse(self.panel._manual_level_spin.isEnabled())

    def test_level_controls_emit_mode_change_and_manual_level_only_on_enter(self) -> None:
        mode_changes = []
        manual_changes = []
        self.panel.on_auto_level_mode_changed(mode_changes.append)
        self.panel.on_manual_level_requested(manual_changes.append)
        self.panel.set_level_mode(auto_enabled=True, manual_level=0, max_level=4)
        self.panel.set_interaction_tools_enabled(True)

        self.panel._auto_level_checkbox.setChecked(False)
        QApplication.processEvents()
        self.assertEqual(mode_changes, [False])
        self.assertTrue(self.panel._manual_level_spin.isEnabled())

        self.panel._manual_level_spin.setValue(3)
        QApplication.processEvents()
        self.assertEqual(manual_changes, [])

        line_edit = self.panel._manual_level_spin.lineEdit()
        self.assertIsNotNone(line_edit)
        line_edit.returnPressed.emit()
        QApplication.processEvents()
        self.assertEqual(manual_changes, [3])

    def test_level_controls_clamp_manual_level_silently(self) -> None:
        self.panel.set_level_mode(auto_enabled=False, manual_level=99, max_level=2)
        self.assertFalse(self.panel.state.auto_level_enabled)
        self.assertEqual(self.panel.state.manual_level, 2)
        self.assertEqual(self.panel.state.manual_level_max, 2)
        self.assertEqual(self.panel._manual_level_spin.value(), 2)

        self.panel.set_level_mode(auto_enabled=False, manual_level=-5, max_level=2)
        self.assertEqual(self.panel.state.manual_level, 0)
        self.assertEqual(self.panel._manual_level_spin.value(), 0)

    def test_level_controls_can_be_disabled_explicitly(self) -> None:
        self.panel.set_level_mode(auto_enabled=False, manual_level=1, max_level=3)
        self.panel.set_interaction_tools_enabled(True)
        self.assertTrue(self.panel._auto_level_checkbox.isEnabled())
        self.assertTrue(self.panel._manual_level_spin.isEnabled())

        self.panel.set_level_controls_enabled(False)
        self.assertFalse(self.panel._auto_level_checkbox.isEnabled())
        self.assertFalse(self.panel._manual_level_spin.isEnabled())

        self.panel.set_level_controls_enabled(True)
        self.assertTrue(self.panel._auto_level_checkbox.isEnabled())
        self.assertTrue(self.panel._manual_level_spin.isEnabled())

    def test_active_levels_status_indicates_manual_forced_mode(self) -> None:
        self.panel.set_active_levels(
            axial=(1, 2),
            coronal=(1, 2),
            sagittal=(1, 2),
            forced=False,
        )
        self.assertNotIn("Manual (forced)", self.panel._level_status.text())

        self.panel.set_active_levels(
            axial=(1, 2),
            coronal=(1, 2),
            sagittal=(1, 2),
            forced=True,
        )
        self.assertIn("Manual (forced)", self.panel._level_status.text())

    def test_contrast_controls_use_1000_steps_and_emit_values(self) -> None:
        changes = []
        self.panel.on_contrast_window_changed(lambda vmin, vmax: changes.append((vmin, vmax)))

        self.panel.set_interaction_tools_enabled(True)
        self.panel.set_contrast_range((0.0, 999.0))

        self.assertEqual(self.panel._contrast_min_slider.minimum(), 0)
        self.assertEqual(self.panel._contrast_min_slider.maximum(), 999)
        self.assertEqual(self.panel._contrast_max_slider.minimum(), 0)
        self.assertEqual(self.panel._contrast_max_slider.maximum(), 999)
        self.assertTrue(self.panel._contrast_min_slider.isEnabled())
        self.assertTrue(self.panel._contrast_max_slider.isEnabled())
        self.assertEqual(self.panel._contrast_min_value.text(), "Min: 0")
        self.assertEqual(self.panel._contrast_max_value.text(), "Max: 999")

        self.panel._contrast_min_slider.setValue(123)
        QApplication.processEvents()
        self.assertEqual(changes[-1], (123.0, 999.0))
        self.assertEqual(self.panel._contrast_min_value.text(), "Min: 123")
        self.assertEqual(self.panel._contrast_max_value.text(), "Max: 999")

        self.panel._contrast_max_slider.setValue(777)
        QApplication.processEvents()
        self.assertEqual(changes[-1], (123.0, 777.0))
        self.assertEqual(self.panel._contrast_min_value.text(), "Min: 123")
        self.assertEqual(self.panel._contrast_max_value.text(), "Max: 777")

    def test_contrast_sliders_enforce_min_less_than_max(self) -> None:
        self.panel.set_interaction_tools_enabled(True)
        self.panel.set_contrast_range((0.0, 999.0))
        self.panel._contrast_max_slider.setValue(200)
        QApplication.processEvents()

        self.panel._contrast_min_slider.setValue(500)
        QApplication.processEvents()
        self.assertEqual(self.panel._contrast_min_slider.value(), 199)
        self.assertEqual(self.panel.contrast_window(), (199.0, 200.0))

        self.panel._contrast_max_slider.setValue(10)
        QApplication.processEvents()
        self.assertEqual(self.panel._contrast_max_slider.value(), 200)
        self.assertEqual(self.panel.contrast_window(), (199.0, 200.0))

    def test_contrast_sliders_are_disabled_for_constant_range(self) -> None:
        self.panel.set_interaction_tools_enabled(True)
        self.panel.set_contrast_range((7.0, 7.0))

        self.assertFalse(self.panel._contrast_min_slider.isEnabled())
        self.assertFalse(self.panel._contrast_max_slider.isEnabled())
        self.assertEqual(self.panel._contrast_min_value.text(), "Min: 7")
        self.assertEqual(self.panel._contrast_max_value.text(), "Max: 7")

    def test_segmentation_opacity_slider_updates_state_and_emits_callback(self) -> None:
        changes: list[float] = []
        self.panel.on_segmentation_opacity_changed(lambda opacity: changes.append(float(opacity)))

        self.assertAlmostEqual(self.panel.segmentation_opacity(), 0.3, places=6)
        self.assertEqual(self.panel._segmentation_opacity_value.text(), "30%")

        self.panel._segmentation_opacity_slider.setValue(65)
        QApplication.processEvents()

        self.assertAlmostEqual(self.panel.segmentation_opacity(), 0.65, places=6)
        self.assertEqual(self.panel._segmentation_opacity_value.text(), "65%")
        self.assertAlmostEqual(changes[-1], 0.65, places=6)


if __name__ == "__main__":
    unittest.main()
