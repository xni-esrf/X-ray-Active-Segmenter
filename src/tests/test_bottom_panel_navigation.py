from __future__ import annotations

import unittest

from src.tests.bottom_panel_test_utils import QApplication, BottomPanelTestCase


class BottomPanelNavigationTests(BottomPanelTestCase):
    def test_level_controls_are_disabled_until_interaction_tools_enabled(self) -> None:
        self.panel.set_level_mode(auto_enabled=True, manual_level=0, max_level=4)
        self.assertFalse(self.panel._navigation_panel.auto_level_checkbox.isEnabled())
        self.assertFalse(self.panel._navigation_panel.manual_level_spin.isEnabled())

        self.panel.set_interaction_tools_enabled(True)
        self.assertTrue(self.panel._navigation_panel.auto_level_checkbox.isEnabled())
        self.assertTrue(self.panel._navigation_panel.auto_level_checkbox.isChecked())
        self.assertFalse(self.panel._navigation_panel.manual_level_spin.isEnabled())

    def test_zoom_spin_range_matches_core_zoom_bounds(self) -> None:
        self.assertAlmostEqual(self.panel._navigation_panel.zoom_spin.minimum(), 0.1)
        self.assertAlmostEqual(self.panel._navigation_panel.zoom_spin.maximum(), 1.0)

    def test_set_zoom_clamps_to_zoom_spin_range(self) -> None:
        self.panel.set_zoom(2.5)
        self.assertAlmostEqual(self.panel.state.zoom, 1.0)
        self.assertAlmostEqual(self.panel._navigation_panel.zoom_spin.value(), 1.0)

        self.panel.set_zoom(-3.0)
        self.assertAlmostEqual(self.panel.state.zoom, 0.1)
        self.assertAlmostEqual(self.panel._navigation_panel.zoom_spin.value(), 0.1)

    def test_level_controls_emit_mode_change_and_manual_level_only_on_enter(self) -> None:
        mode_changes = []
        manual_changes = []
        self.panel.on_auto_level_mode_changed(mode_changes.append)
        self.panel.on_manual_level_requested(manual_changes.append)
        self.panel.set_level_mode(auto_enabled=True, manual_level=0, max_level=4)
        self.panel.set_interaction_tools_enabled(True)

        self.panel._navigation_panel.auto_level_checkbox.setChecked(False)
        QApplication.processEvents()
        self.assertEqual(mode_changes, [False])
        self.assertTrue(self.panel._navigation_panel.manual_level_spin.isEnabled())

        self.panel._navigation_panel.manual_level_spin.setValue(3)
        QApplication.processEvents()
        self.assertEqual(manual_changes, [])

        line_edit = self.panel._navigation_panel.manual_level_spin.lineEdit()
        self.assertIsNotNone(line_edit)
        line_edit.returnPressed.emit()
        QApplication.processEvents()
        self.assertEqual(manual_changes, [3])

    def test_level_controls_clamp_manual_level_silently(self) -> None:
        self.panel.set_level_mode(auto_enabled=False, manual_level=99, max_level=2)
        self.assertFalse(self.panel.state.auto_level_enabled)
        self.assertEqual(self.panel.state.manual_level, 2)
        self.assertEqual(self.panel.state.manual_level_max, 2)
        self.assertEqual(self.panel._navigation_panel.manual_level_spin.value(), 2)

        self.panel.set_level_mode(auto_enabled=False, manual_level=-5, max_level=2)
        self.assertEqual(self.panel.state.manual_level, 0)
        self.assertEqual(self.panel._navigation_panel.manual_level_spin.value(), 0)

    def test_level_controls_can_be_disabled_explicitly(self) -> None:
        self.panel.set_level_mode(auto_enabled=False, manual_level=1, max_level=3)
        self.panel.set_interaction_tools_enabled(True)
        self.assertTrue(self.panel._navigation_panel.auto_level_checkbox.isEnabled())
        self.assertTrue(self.panel._navigation_panel.manual_level_spin.isEnabled())

        self.panel.set_level_controls_enabled(False)
        self.assertFalse(self.panel._navigation_panel.auto_level_checkbox.isEnabled())
        self.assertFalse(self.panel._navigation_panel.manual_level_spin.isEnabled())

        self.panel.set_level_controls_enabled(True)
        self.assertTrue(self.panel._navigation_panel.auto_level_checkbox.isEnabled())
        self.assertTrue(self.panel._navigation_panel.manual_level_spin.isEnabled())

    def test_active_levels_status_indicates_manual_forced_mode(self) -> None:
        self.panel.set_active_levels(
            axial=(1, 2),
            coronal=(1, 2),
            sagittal=(1, 2),
            forced=False,
        )
        self.assertNotIn("Manual (forced)", self.panel._navigation_panel.level_status.text())

        self.panel.set_active_levels(
            axial=(1, 2),
            coronal=(1, 2),
            sagittal=(1, 2),
            forced=True,
        )
        self.assertIn("Manual (forced)", self.panel._navigation_panel.level_status.text())

    def test_view_layout_mode_defaults_to_all(self) -> None:
        self.assertEqual(self.panel.view_layout_mode(), "all")
        self.assertTrue(self.panel._navigation_panel.view_layout_all_radio.isChecked())

    def test_set_view_layout_mode_updates_selected_radio(self) -> None:
        self.panel.set_view_layout_mode("coronal")
        self.assertEqual(self.panel.view_layout_mode(), "coronal")
        self.assertTrue(self.panel._navigation_panel.view_layout_coronal_radio.isChecked())
        self.assertFalse(self.panel._navigation_panel.view_layout_all_radio.isChecked())

        self.panel.set_view_layout_mode("not-a-mode")
        self.assertEqual(self.panel.view_layout_mode(), "all")
        self.assertTrue(self.panel._navigation_panel.view_layout_all_radio.isChecked())

    def test_view_layout_mode_callback_emits_on_user_toggle(self) -> None:
        changes: list[str] = []
        self.panel.on_view_layout_mode_changed(lambda mode: changes.append(str(mode)))

        self.panel._navigation_panel.view_layout_axial_radio.click()
        QApplication.processEvents()
        self.assertEqual(changes[-1], "axial")

        # Re-clicking checked button should not emit a redundant change.
        count_after_first_toggle = len(changes)
        self.panel._navigation_panel.view_layout_axial_radio.click()
        QApplication.processEvents()
        self.assertEqual(len(changes), count_after_first_toggle)

    def test_set_view_layout_mode_does_not_emit_callback(self) -> None:
        changes: list[str] = []
        self.panel.on_view_layout_mode_changed(lambda mode: changes.append(str(mode)))

        self.panel.set_view_layout_mode("sagittal")
        QApplication.processEvents()

        self.assertEqual(changes, [])


if __name__ == "__main__":
    unittest.main()
