from __future__ import annotations

import os
import unittest
from importlib import import_module
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QGridLayout, QScrollArea, QSplitter, QWidget


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
        self.assertFalse(scroll_area.widgetResizable())

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

    def test_control_panel_width_is_clamped_to_ten_and_fifty_percent(self) -> None:
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

    def test_initial_splitter_size_defaults_to_twenty_five_percent_without_persistence(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
