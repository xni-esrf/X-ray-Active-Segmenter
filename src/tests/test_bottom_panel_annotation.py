from __future__ import annotations

import unittest

from src.tests.bottom_panel_test_utils import QApplication, BottomPanelTestCase


class BottomPanelAnnotationTests(BottomPanelTestCase):
    def test_annotation_tool_controls_show_shortcut_hints(self) -> None:
        brush_index = self.panel._annotation_panel.annotation_tool_combo.findData("brush")
        eraser_index = self.panel._annotation_panel.annotation_tool_combo.findData("eraser")
        flood_index = self.panel._annotation_panel.annotation_tool_combo.findData("flood_filler")
        self.assertGreaterEqual(brush_index, 0)
        self.assertGreaterEqual(eraser_index, 0)
        self.assertGreaterEqual(flood_index, 0)
        self.assertEqual(
            self.panel._annotation_panel.annotation_tool_combo.itemText(brush_index),
            "Brush (Ctrl+B)",
        )
        self.assertEqual(
            self.panel._annotation_panel.annotation_tool_combo.itemText(eraser_index),
            "Eraser (Ctrl+E)",
        )
        self.assertEqual(
            self.panel._annotation_panel.annotation_tool_combo.itemText(flood_index),
            "Flood Fill (Ctrl+F)",
        )
        hint = self.panel._annotation_panel.annotation_tool_combo.toolTip()
        self.assertIn("Ctrl+B", hint)
        self.assertIn("Ctrl+E", hint)
        self.assertIn("Ctrl+F", hint)

    def test_tool_label_change_emits_unified_callback(self) -> None:
        changes: list[str] = []
        self.panel.on_tool_label_changed(changes.append)
        self.panel.set_interaction_tools_enabled(True)
        self.panel.set_annotation_mode(True)
        self.panel.set_annotation_controls_enabled(True)
        self.panel._annotation_panel.tool_label_edit.setText(" 17 ")
        self.panel._annotation_panel.tool_label_edit.editingFinished.emit()
        QApplication.processEvents()

        self.assertEqual(changes, ["17"])
        self.assertEqual(self.panel.state.tool_label_text, "17")

    def test_tool_label_placeholder_tracks_annotation_tool(self) -> None:
        self.panel.set_interaction_tools_enabled(True)
        self.panel.set_annotation_mode(True)
        self.panel.set_annotation_controls_enabled(True)

        self.panel.set_annotation_tool("eraser")
        self.assertEqual(self.panel._annotation_panel.tool_label_edit.placeholderText(), "All")

        self.panel.set_annotation_tool("brush")
        self.assertEqual(self.panel._annotation_panel.tool_label_edit.placeholderText(), "1")

        self.panel.set_annotation_tool("flood_filler")
        self.assertEqual(self.panel._annotation_panel.tool_label_edit.placeholderText(), "1")

    def test_history_buttons_are_not_reset_by_annotation_controls(self) -> None:
        self.panel.set_undo_state(depth=2, enabled=True)
        self.panel.set_redo_state(depth=1, enabled=True)
        self.assertTrue(self.panel._history_panel.undo_button.isEnabled())
        self.assertTrue(self.panel._history_panel.redo_button.isEnabled())

        self.panel.set_annotation_controls_enabled(False)

        self.assertTrue(self.panel._history_panel.undo_button.isEnabled())
        self.assertTrue(self.panel._history_panel.redo_button.isEnabled())


if __name__ == "__main__":
    unittest.main()
