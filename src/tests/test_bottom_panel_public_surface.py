from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from src.ui.bottom_panel import BottomPanel
    from src.ui.bottom_panel_subpanels import (
        BOTTOM_PANEL_SUBPANEL_ORDER,
        BOTTOM_PANEL_SUBPANEL_SPECS,
    )
except Exception:  # pragma: no cover - environment dependent
    BottomPanel = None  # type: ignore[assignment]
    BOTTOM_PANEL_SUBPANEL_ORDER = tuple()  # type: ignore[assignment]
    BOTTOM_PANEL_SUBPANEL_SPECS = tuple()  # type: ignore[assignment]


EXPECTED_BOTTOM_PANEL_PUBLIC_METHODS = {
    "annotation_tool",
    "brush_radius",
    "contrast_window",
    "file_path",
    "learning_training_running",
    "on_annotation_mode_changed",
    "on_annotation_tool_changed",
    "on_auto_level_mode_changed",
    "on_bounding_box_delete_requested",
    "on_bounding_box_double_clicked",
    "on_bounding_box_label_changed",
    "on_bounding_box_mode_changed",
    "on_bounding_box_selected",
    "on_bounding_boxes_delete_requested",
    "on_bounding_boxes_label_changed",
    "on_bounding_boxes_selected",
    "on_brush_radius_changed",
    "on_contrast_window_changed",
    "on_change_training_parameters_requested",
    "on_cursor_changed",
    "on_dilation_selected_requested",
    "on_erase_bbox_segmentation_requested",
    "on_erosion_selected_requested",
    "on_flood_fill_requested",
    "on_load_model_requested",
    "on_manual_level_requested",
    "on_median_filter_selected_requested",
    "on_next_available_label_requested",
    "on_open_bounding_boxes_requested",
    "on_open_instance_requested",
    "on_open_requested",
    "on_open_semantic_requested",
    "on_redo_requested",
    "on_save_bounding_boxes_requested",
    "on_save_model_requested",
    "on_save_segmentation_requested",
    "on_segment_inference_requested",
    "on_segmentation_opacity_changed",
    "on_stop_inference_requested",
    "on_stop_training_requested",
    "on_tool_label_changed",
    "on_train_model_requested",
    "on_change_training_parameters_requested",
    "on_undo_requested",
    "on_view_layout_mode_changed",
    "on_zoom_changed",
    "segmentation_opacity",
    "selected_bounding_box",
    "selected_bounding_box_label",
    "selected_bounding_boxes",
    "set_active_level",
    "set_active_levels",
    "set_annotation_controls_enabled",
    "set_annotation_mode",
    "set_annotation_tool",
    "set_bounding_box_mode",
    "set_bounding_boxes",
    "set_brush_radius",
    "set_contrast_range",
    "set_contrast_window",
    "set_cursor_position",
    "set_cursor_range",
    "set_file_path",
    "set_hover_info",
    "set_inference_navigation_only_mode",
    "set_interaction_tools_enabled",
    "set_learning_training_running",
    "set_level_controls_enabled",
    "set_level_mode",
    "set_picked_info",
    "set_pyramid_levels",
    "set_redo_state",
    "set_segment_inference_enabled",
    "set_segmentation_opacity",
    "set_selected_bounding_box",
    "set_selected_bounding_boxes",
    "set_stop_inference_enabled",
    "set_stop_training_enabled",
    "set_tool_label",
    "set_tool_label_placeholder",
    "set_train_model_enabled",
    "set_undo_state",
    "set_view_layout_mode",
    "set_zoom",
    "tool_label",
    "view_layout_mode",
}


@unittest.skipUnless(BottomPanel is not None, "BottomPanel is not available")
class BottomPanelPublicSurfaceTests(unittest.TestCase):
    def test_public_bottom_panel_facade_methods_are_explicitly_frozen(self) -> None:
        current_methods = {
            name
            for name, value in BottomPanel.__dict__.items()
            if callable(value) and not name.startswith("_") and name != "resizeEvent"
        }

        self.assertEqual(current_methods, EXPECTED_BOTTOM_PANEL_PUBLIC_METHODS)

    def test_future_subpanel_boundaries_cover_public_facade_once(self) -> None:
        self.assertEqual(
            BOTTOM_PANEL_SUBPANEL_ORDER,
            (
                "files",
                "navigation",
                "contrast",
                "annotation",
                "bounding_boxes",
                "learning",
                "history",
            ),
        )

        assigned_methods: list[str] = []
        for spec in BOTTOM_PANEL_SUBPANEL_SPECS:
            self.assertTrue(spec.name)
            self.assertTrue(spec.title)
            self.assertTrue(spec.responsibility)
            self.assertTrue(spec.public_methods)
            assigned_methods.extend(spec.public_methods)

        self.assertEqual(len(assigned_methods), len(set(assigned_methods)))
        self.assertEqual(set(assigned_methods), EXPECTED_BOTTOM_PANEL_PUBLIC_METHODS)


if __name__ == "__main__":
    unittest.main()
