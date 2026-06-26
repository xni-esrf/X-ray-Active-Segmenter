from __future__ import annotations

import unittest

from src.tests.bottom_panel_test_utils import BottomPanelTestCase, QGroupBox, QSizePolicy


class BottomPanelLayoutTests(BottomPanelTestCase):
    def test_only_bounding_boxes_group_uses_expanding_width_policy(self) -> None:
        self.assertIsNotNone(QGroupBox)
        self.assertIsNotNone(QSizePolicy)
        groups = self.panel.findChildren(QGroupBox)
        self.assertGreater(len(groups), 0)

        for group in groups:
            horizontal_policy = group.sizePolicy().horizontalPolicy()
            if group.title() == "Bounding Boxes":
                self.assertEqual(horizontal_policy, QSizePolicy.Expanding)
                self.assertGreaterEqual(group.minimumWidth(), 336)
            else:
                self.assertEqual(horizontal_policy, QSizePolicy.Maximum)

    def test_bbox_table_width_is_expanding_with_content_minimum(self) -> None:
        self.assertIsNotNone(QSizePolicy)
        table = self.panel._bounding_boxes_panel.bbox_table
        horizontal_policy = table.sizePolicy().horizontalPolicy()
        self.assertEqual(horizontal_policy, QSizePolicy.Expanding)
        self.assertGreaterEqual(table.minimumWidth(), 176)
        self.assertGreaterEqual(table.maximumWidth(), 16_777_215)

    def test_compact_per_control_width_limits_remain_unchanged(self) -> None:
        self.assertEqual(self.panel._files_panel.open_button.maximumWidth(), 170)
        self.assertEqual(self.panel._files_panel.open_semantic_button.maximumWidth(), 170)
        self.assertEqual(self.panel._files_panel.open_instance_button.maximumWidth(), 170)
        self.assertEqual(self.panel._files_panel.save_segmentation_button.maximumWidth(), 170)

        self.assertEqual(self.panel._navigation_panel.cursor_z.maximumWidth(), 130)
        self.assertEqual(self.panel._navigation_panel.cursor_y.maximumWidth(), 130)
        self.assertEqual(self.panel._navigation_panel.cursor_x.maximumWidth(), 130)
        self.assertEqual(self.panel._navigation_panel.zoom_spin.maximumWidth(), 130)
        self.assertEqual(self.panel._navigation_panel.manual_level_spin.maximumWidth(), 130)

        self.assertEqual(self.panel._contrast_panel.contrast_min_slider.maximumWidth(), 180)
        self.assertEqual(self.panel._contrast_panel.contrast_max_slider.maximumWidth(), 180)
        self.assertEqual(self.panel._contrast_panel.segmentation_opacity_slider.maximumWidth(), 180)

        self.assertEqual(self.panel._history_panel.undo_button.maximumWidth(), 170)
        self.assertEqual(self.panel._history_panel.redo_button.maximumWidth(), 170)


if __name__ == "__main__":
    unittest.main()
