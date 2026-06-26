from __future__ import annotations

import unittest

from src.tests.bottom_panel_test_utils import QApplication, BottomPanelTestCase


class BottomPanelContrastTests(BottomPanelTestCase):
    def test_contrast_controls_use_1000_steps_and_emit_values(self) -> None:
        changes = []
        self.panel.on_contrast_window_changed(lambda vmin, vmax: changes.append((vmin, vmax)))

        self.panel.set_interaction_tools_enabled(True)
        self.panel.set_contrast_range((0.0, 999.0))

        self.assertEqual(self.panel._contrast_panel.contrast_min_slider.minimum(), 0)
        self.assertEqual(self.panel._contrast_panel.contrast_min_slider.maximum(), 999)
        self.assertEqual(self.panel._contrast_panel.contrast_max_slider.minimum(), 0)
        self.assertEqual(self.panel._contrast_panel.contrast_max_slider.maximum(), 999)
        self.assertTrue(self.panel._contrast_panel.contrast_min_slider.isEnabled())
        self.assertTrue(self.panel._contrast_panel.contrast_max_slider.isEnabled())
        self.assertEqual(self.panel._contrast_panel.contrast_min_value.text(), "Min: 0")
        self.assertEqual(self.panel._contrast_panel.contrast_max_value.text(), "Max: 999")

        self.panel._contrast_panel.contrast_min_slider.setValue(123)
        QApplication.processEvents()
        self.assertEqual(changes[-1], (123.0, 999.0))
        self.assertEqual(self.panel._contrast_panel.contrast_min_value.text(), "Min: 123")
        self.assertEqual(self.panel._contrast_panel.contrast_max_value.text(), "Max: 999")

        self.panel._contrast_panel.contrast_max_slider.setValue(777)
        QApplication.processEvents()
        self.assertEqual(changes[-1], (123.0, 777.0))
        self.assertEqual(self.panel._contrast_panel.contrast_min_value.text(), "Min: 123")
        self.assertEqual(self.panel._contrast_panel.contrast_max_value.text(), "Max: 777")

    def test_contrast_sliders_enforce_min_less_than_max(self) -> None:
        self.panel.set_interaction_tools_enabled(True)
        self.panel.set_contrast_range((0.0, 999.0))
        self.panel._contrast_panel.contrast_max_slider.setValue(200)
        QApplication.processEvents()

        self.panel._contrast_panel.contrast_min_slider.setValue(500)
        QApplication.processEvents()
        self.assertEqual(self.panel._contrast_panel.contrast_min_slider.value(), 199)
        self.assertEqual(self.panel.contrast_window(), (199.0, 200.0))

        self.panel._contrast_panel.contrast_max_slider.setValue(10)
        QApplication.processEvents()
        self.assertEqual(self.panel._contrast_panel.contrast_max_slider.value(), 200)
        self.assertEqual(self.panel.contrast_window(), (199.0, 200.0))

    def test_contrast_sliders_are_disabled_for_constant_range(self) -> None:
        self.panel.set_interaction_tools_enabled(True)
        self.panel.set_contrast_range((7.0, 7.0))

        self.assertFalse(self.panel._contrast_panel.contrast_min_slider.isEnabled())
        self.assertFalse(self.panel._contrast_panel.contrast_max_slider.isEnabled())
        self.assertEqual(self.panel._contrast_panel.contrast_min_value.text(), "Min: 7")
        self.assertEqual(self.panel._contrast_panel.contrast_max_value.text(), "Max: 7")

    def test_segmentation_opacity_slider_updates_state_and_emits_callback(self) -> None:
        changes: list[float] = []
        self.panel.on_segmentation_opacity_changed(lambda opacity: changes.append(float(opacity)))

        self.assertAlmostEqual(self.panel.segmentation_opacity(), 0.3, places=6)
        self.assertEqual(self.panel._contrast_panel.segmentation_opacity_value.text(), "30%")

        self.panel._contrast_panel.segmentation_opacity_slider.setValue(65)
        QApplication.processEvents()

        self.assertAlmostEqual(self.panel.segmentation_opacity(), 0.65, places=6)
        self.assertEqual(self.panel._contrast_panel.segmentation_opacity_value.text(), "65%")
        self.assertAlmostEqual(changes[-1], 0.65, places=6)


if __name__ == "__main__":
    unittest.main()
