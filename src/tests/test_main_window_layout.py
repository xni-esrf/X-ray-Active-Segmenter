from __future__ import annotations

import os
import unittest
from importlib import import_module
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QGridLayout, QScrollArea, QSplitter, QWidget
from src.bbox import BoundingBox
try:
    from src.ui.bottom_panel import BottomPanel as RealBottomPanel
except Exception:  # pragma: no cover - environment dependent
    RealBottomPanel = None  # type: ignore[assignment]


class _BottomPanelStub(QWidget):
    def __init__(self) -> None:
        super().__init__()

    def __getattr__(self, _name: str):
        return lambda *_args, **_kwargs: None


class _OrthogonalViewStub(QWidget):
    def __init__(self, view_id: str, axis: int, *args, **kwargs) -> None:
        super().__init__()
        self.view_id = view_id
        self.axis = axis


class _LargeBottomPanelStub(_BottomPanelStub):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(1200, 1200)


class _TrackingBottomPanelStub(_BottomPanelStub):
    def __init__(self) -> None:
        super().__init__()
        self.view_layout_mode_calls: list[str] = []
        self._view_layout_mode = "all"

    def set_view_layout_mode(self, mode: str) -> None:
        normalized = str(mode).strip().lower()
        if normalized not in {"all", "axial", "coronal", "sagittal"}:
            normalized = "all"
        self._view_layout_mode = normalized
        self.view_layout_mode_calls.append(normalized)

    def view_layout_mode(self) -> str:
        return self._view_layout_mode


class _SyncManagerStub:
    def set_cursor_indices(self, _indices) -> None:
        return None

    def set_zoom(self, _zoom: float) -> None:
        return None

    def on_state_changed(self, _callback) -> None:
        return None


class MainWindowLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def _build_window(self, *, bottom_panel_cls=_BottomPanelStub):
        main_window_module = import_module("src.ui.main_window")
        MainWindow = main_window_module.MainWindow

        with patch.object(main_window_module, "BottomPanel", bottom_panel_cls), patch.object(
            main_window_module, "OrthogonalView", _OrthogonalViewStub
        ), patch.object(MainWindow, "_sync_bounding_boxes_ui", lambda self: None), patch.object(
            MainWindow, "_refresh_learning_training_ui_state", lambda self: None
        ), patch.object(
            MainWindow, "_refresh_annotation_ui_state", lambda self: None
        ):
            window = MainWindow(
                renderer=object(),
                sync_manager=_SyncManagerStub(),
                input_handlers=object(),
            )

        self.addCleanup(window.close)
        return window

    def test_layout_uses_splitter_scroll_area_and_three_views_without_unused_placeholder(self) -> None:
        window = self._build_window()

        splitter = window.centralWidget()
        self.assertIsInstance(splitter, QSplitter)
        self.assertIs(splitter, window._main_splitter)
        self.assertEqual(splitter.count(), 2)

        left_panel = splitter.widget(0)
        right_panel = splitter.widget(1)
        self.assertIs(left_panel, window._left_panel)
        self.assertIs(right_panel, window._right_panel)

        layout = left_panel.layout()
        self.assertIsInstance(layout, QGridLayout)

        self.assertEqual(layout.indexOf(window.views["axial"]), 0)
        axial_row, axial_col, axial_row_span, axial_col_span = layout.getItemPosition(0)
        self.assertEqual((axial_row, axial_col, axial_row_span, axial_col_span), (0, 0, 1, 1))

        self.assertEqual(layout.indexOf(window.views["coronal"]), 1)
        coronal_row, coronal_col, coronal_row_span, coronal_col_span = layout.getItemPosition(1)
        self.assertEqual((coronal_row, coronal_col, coronal_row_span, coronal_col_span), (0, 1, 1, 1))

        self.assertEqual(layout.indexOf(window.views["sagittal"]), 2)
        sagittal_row, sagittal_col, sagittal_row_span, sagittal_col_span = layout.getItemPosition(2)
        self.assertEqual((sagittal_row, sagittal_col, sagittal_row_span, sagittal_col_span), (1, 0, 1, 2))

        right_layout = right_panel.layout()
        self.assertIsInstance(right_layout, QGridLayout)
        self.assertEqual(right_layout.count(), 1)
        scroll_area = right_layout.itemAt(0).widget()
        self.assertIsInstance(scroll_area, QScrollArea)
        self.assertIs(scroll_area.widget(), window.bottom_panel)
        self.assertEqual(scroll_area.horizontalScrollBarPolicy(), Qt.ScrollBarAsNeeded)
        self.assertEqual(scroll_area.verticalScrollBarPolicy(), Qt.ScrollBarAsNeeded)
        self.assertTrue(scroll_area.widgetResizable())

        self.assertFalse(hasattr(window, "_unused_view"))

    def test_control_panel_scroll_area_can_overflow_both_directions(self) -> None:
        window = self._build_window(bottom_panel_cls=_LargeBottomPanelStub)
        window.resize(800, 500)
        window.show()
        self._app.processEvents()
        window._initialize_main_splitter_sizes()
        self._app.processEvents()

        splitter = window._main_splitter
        total_width = max(1, splitter.width())
        right_width = max(1, int(round(total_width * 0.25)))
        splitter.setSizes([max(1, total_width - right_width), right_width])
        self._app.processEvents()

        scroll_area = window._right_panel.layout().itemAt(0).widget()
        self.assertIsInstance(scroll_area, QScrollArea)
        self.assertGreater(scroll_area.horizontalScrollBar().maximum(), 0)
        self.assertGreater(scroll_area.verticalScrollBar().maximum(), 0)

    def test_control_panel_width_is_clamped_to_five_and_thirty_five_percent(self) -> None:
        window = self._build_window()
        window.resize(1000, 700)
        window.show()
        self._app.processEvents()
        window._initialize_main_splitter_sizes()
        self._app.processEvents()

        min_width, max_width = window._main_splitter_control_panel_width_bounds()
        total_width = window._main_splitter_total_width()
        self.assertEqual(
            (min_width, max_width),
            (
                int(round(total_width * window._CONTROL_PANEL_MIN_WIDTH_FRACTION)),
                int(round(total_width * window._CONTROL_PANEL_MAX_WIDTH_FRACTION)),
            ),
        )

        window._main_splitter.setSizes([950, 50])
        window._apply_main_splitter_width_constraints()
        self.assertEqual(tuple(window._main_splitter.sizes()), (total_width - min_width, min_width))

        window._main_splitter.setSizes([400, 600])
        window._apply_main_splitter_width_constraints()
        self.assertEqual(tuple(window._main_splitter.sizes()), (total_width - max_width, max_width))

    def test_initial_splitter_size_defaults_to_twenty_percent_without_persistence(self) -> None:
        first_window = self._build_window()
        first_window.resize(1000, 700)
        first_window.show()
        self._app.processEvents()
        first_window._initialize_main_splitter_sizes()
        self._app.processEvents()
        first_total_width = first_window._main_splitter_total_width()
        first_target_width = int(
            round(first_total_width * first_window._CONTROL_PANEL_INITIAL_WIDTH_FRACTION)
        )
        self.assertEqual(
            tuple(first_window._main_splitter.sizes()),
            (first_total_width - first_target_width, first_target_width),
        )

        first_window._main_splitter.setSizes([600, 400])
        first_window._apply_main_splitter_width_constraints()
        resized_sizes = tuple(first_window._main_splitter.sizes())
        self.assertNotEqual(
            resized_sizes,
            (first_total_width - first_target_width, first_target_width),
        )
        self.assertGreater(resized_sizes[1], first_target_width)

        second_window = self._build_window()
        second_window.resize(1000, 700)
        second_window.show()
        self._app.processEvents()
        second_window._initialize_main_splitter_sizes()
        self._app.processEvents()
        second_total_width = second_window._main_splitter_total_width()
        second_target_width = int(
            round(second_total_width * second_window._CONTROL_PANEL_INITIAL_WIDTH_FRACTION)
        )
        self.assertEqual(
            tuple(second_window._main_splitter.sizes()),
            (second_total_width - second_target_width, second_target_width),
        )

    def test_view_layout_single_mode_hides_two_views_and_expands_selected_view(self) -> None:
        window = self._build_window()
        layout = window._left_panel.layout()
        self.assertIsInstance(layout, QGridLayout)

        window._handle_view_layout_mode_changed("coronal")
        self._app.processEvents()

        self.assertFalse(window.views["coronal"].isHidden())
        self.assertTrue(window.views["axial"].isHidden())
        self.assertTrue(window.views["sagittal"].isHidden())

        coronal_index = layout.indexOf(window.views["coronal"])
        self.assertGreaterEqual(coronal_index, 0)
        coronal_row, coronal_col, coronal_row_span, coronal_col_span = layout.getItemPosition(coronal_index)
        self.assertEqual((coronal_row, coronal_col, coronal_row_span, coronal_col_span), (0, 0, 2, 2))

    def test_view_layout_all_mode_restores_three_view_grid(self) -> None:
        window = self._build_window()
        layout = window._left_panel.layout()
        self.assertIsInstance(layout, QGridLayout)

        window._handle_view_layout_mode_changed("axial")
        self._app.processEvents()
        window._handle_view_layout_mode_changed("all")
        self._app.processEvents()

        self.assertFalse(window.views["axial"].isHidden())
        self.assertFalse(window.views["coronal"].isHidden())
        self.assertFalse(window.views["sagittal"].isHidden())

        axial_index = layout.indexOf(window.views["axial"])
        coronal_index = layout.indexOf(window.views["coronal"])
        sagittal_index = layout.indexOf(window.views["sagittal"])
        self.assertEqual(layout.getItemPosition(axial_index), (0, 0, 1, 1))
        self.assertEqual(layout.getItemPosition(coronal_index), (0, 1, 1, 1))
        self.assertEqual(layout.getItemPosition(sagittal_index), (1, 0, 1, 2))

    def test_view_layout_switch_does_not_replace_or_mutate_view_objects(self) -> None:
        window = self._build_window()
        axial = window.views["axial"]
        coronal = window.views["coronal"]
        sagittal = window.views["sagittal"]
        axial.custom_state = {"slice": 10, "zoom": 1.7}
        coronal.custom_state = {"slice": 20, "zoom": 0.8}
        sagittal.custom_state = {"slice": 30, "zoom": 2.1}

        window._handle_view_layout_mode_changed("sagittal")
        self._app.processEvents()
        window._handle_view_layout_mode_changed("all")
        self._app.processEvents()

        self.assertIs(window.views["axial"], axial)
        self.assertIs(window.views["coronal"], coronal)
        self.assertIs(window.views["sagittal"], sagittal)
        self.assertEqual(axial.custom_state, {"slice": 10, "zoom": 1.7})
        self.assertEqual(coronal.custom_state, {"slice": 20, "zoom": 0.8})
        self.assertEqual(sagittal.custom_state, {"slice": 30, "zoom": 2.1})

    def test_view_layout_initialization_syncs_bottom_panel_and_defaults_to_all(self) -> None:
        window = self._build_window(bottom_panel_cls=_TrackingBottomPanelStub)
        layout = window._left_panel.layout()
        self.assertIsInstance(layout, QGridLayout)

        self.assertEqual(window.state.view_layout_mode, "all")
        self.assertEqual(window.bottom_panel.view_layout_mode(), "all")
        self.assertEqual(window.bottom_panel.view_layout_mode_calls, ["all"])

        self.assertFalse(window.views["axial"].isHidden())
        self.assertFalse(window.views["coronal"].isHidden())
        self.assertFalse(window.views["sagittal"].isHidden())
        self.assertEqual(layout.getItemPosition(layout.indexOf(window.views["axial"])), (0, 0, 1, 1))
        self.assertEqual(layout.getItemPosition(layout.indexOf(window.views["coronal"])), (0, 1, 1, 1))
        self.assertEqual(layout.getItemPosition(layout.indexOf(window.views["sagittal"])), (1, 0, 1, 2))

    def test_view_layout_mode_change_rerenders_only_visible_views_when_volume_loaded(self) -> None:
        window = self._build_window()
        window.state.volume_loaded = True
        queued_view_ids: list[str] = []
        window._queue_render = lambda view_id: queued_view_ids.append(str(view_id))  # type: ignore[method-assign]

        window._handle_view_layout_mode_changed("coronal")
        self.assertEqual(queued_view_ids, ["coronal"])

        queued_view_ids.clear()
        window._handle_view_layout_mode_changed("all")
        self.assertEqual(queued_view_ids, ["axial", "coronal", "sagittal"])

    def test_bbox_table_width_grows_when_control_panel_expands(self) -> None:
        if RealBottomPanel is None:
            self.skipTest("Real BottomPanel is not available")
        window = self._build_window(bottom_panel_cls=RealBottomPanel)
        window.resize(2200, 1200)
        window.show()
        self._app.processEvents()
        window._initialize_main_splitter_sizes()
        self._app.processEvents()

        box = BoundingBox.from_bounds(
            box_id="bbox_very_long",
            z0=123,
            z1=999,
            y0=10,
            y1=998,
            x0=20,
            x1=997,
            volume_shape=(2000, 2000, 2000),
        )
        window.bottom_panel.set_bounding_boxes((box,))
        self._app.processEvents()

        total_width = window._main_splitter_total_width()
        min_width, max_width = window._main_splitter_control_panel_width_bounds()
        window._main_splitter.setSizes([total_width - min_width, min_width])
        window._apply_main_splitter_width_constraints()
        self._app.processEvents()
        narrow_width = window.bottom_panel._bbox_table.width()

        window._main_splitter.setSizes([total_width - max_width, max_width])
        window._apply_main_splitter_width_constraints()
        self._app.processEvents()
        wide_width = window.bottom_panel._bbox_table.width()

        self.assertGreaterEqual(max_width, min_width)
        self.assertGreater(wide_width, narrow_width)

    def test_compact_controls_remain_compact_at_minimum_control_panel_width(self) -> None:
        if RealBottomPanel is None:
            self.skipTest("Real BottomPanel is not available")
        window = self._build_window(bottom_panel_cls=RealBottomPanel)
        window.resize(1600, 900)
        window.show()
        self._app.processEvents()
        window._initialize_main_splitter_sizes()
        self._app.processEvents()

        total_width = window._main_splitter_total_width()
        min_width, _max_width = window._main_splitter_control_panel_width_bounds()
        window._main_splitter.setSizes([total_width - min_width, min_width])
        window._apply_main_splitter_width_constraints()
        self._app.processEvents()

        self.assertLessEqual(window.bottom_panel._open_button.width(), 170)
        self.assertLessEqual(window.bottom_panel._zoom_spin.width(), 130)
        self.assertLessEqual(window.bottom_panel._contrast_min_slider.width(), 180)
        self.assertLessEqual(window.bottom_panel._undo_button.width(), 170)


if __name__ == "__main__":
    unittest.main()
