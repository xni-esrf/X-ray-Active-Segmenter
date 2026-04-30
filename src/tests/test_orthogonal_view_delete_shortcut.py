from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QApplication, QLineEdit

try:
    from src.ui.orthogonal_view import OrthogonalView
except Exception:  # pragma: no cover - environment dependent
    OrthogonalView = None  # type: ignore[assignment]


@unittest.skipUnless(OrthogonalView is not None, "OrthogonalView is not available")
class OrthogonalViewDeleteShortcutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_keypress_backspace_triggers_bbox_delete_callback_and_accepts_event(self) -> None:
        delete_calls: list[str] = []
        accepted = {"value": False}

        class _Event:
            def key(self):
                return Qt.Key_Backspace

            def accept(self):
                accepted["value"] = True

        view_like = SimpleNamespace(
            _on_bounding_box_delete_requested=lambda: delete_calls.append("delete"),
        )

        OrthogonalView.keyPressEvent(view_like, _Event())

        self.assertEqual(delete_calls, ["delete"])
        self.assertTrue(accepted["value"])

    def test_keypress_delete_is_accepted_without_callback(self) -> None:
        accepted = {"value": False}

        class _Event:
            def key(self):
                return Qt.Key_Delete

            def accept(self):
                accepted["value"] = True

        view_like = SimpleNamespace(
            _on_bounding_box_delete_requested=None,
        )

        OrthogonalView.keyPressEvent(view_like, _Event())

        self.assertTrue(accepted["value"])

    def test_eventfilter_canvas_backspace_consumes_event_and_triggers_delete_callback(self) -> None:
        delete_calls: list[str] = []
        accepted = {"value": False}
        canvas = object()

        class _Event:
            def type(self):
                return QEvent.KeyPress

            def key(self):
                return Qt.Key_Backspace

            def accept(self):
                accepted["value"] = True

        view_like = SimpleNamespace(
            _canvas_widget=canvas,
            _on_bounding_box_delete_requested=lambda: delete_calls.append("delete"),
        )

        consumed = OrthogonalView.eventFilter(view_like, canvas, _Event())

        self.assertTrue(consumed)
        self.assertEqual(delete_calls, ["delete"])
        self.assertTrue(accepted["value"])

    def test_eventfilter_canvas_backspace_ignored_when_focus_is_line_edit(self) -> None:
        delete_calls: list[str] = []
        accepted = {"value": False}
        canvas = object()

        class _Event:
            def type(self):
                return QEvent.KeyPress

            def key(self):
                return Qt.Key_Backspace

            def accept(self):
                accepted["value"] = True

        line_edit = QLineEdit()
        line_edit.show()
        line_edit.setFocus()
        QApplication.processEvents()

        view_like = SimpleNamespace(
            _canvas_widget=canvas,
            _on_bounding_box_delete_requested=lambda: delete_calls.append("delete"),
        )

        consumed = OrthogonalView.eventFilter(view_like, canvas, _Event())

        self.assertFalse(consumed)
        self.assertEqual(delete_calls, [])
        self.assertFalse(accepted["value"])

    def test_eventfilter_canvas_non_delete_key_is_not_consumed(self) -> None:
        delete_calls: list[str] = []
        accepted = {"value": False}
        canvas = object()

        class _Event:
            def type(self):
                return QEvent.KeyPress

            def key(self):
                return Qt.Key_A

            def accept(self):
                accepted["value"] = True

        view_like = SimpleNamespace(
            _canvas_widget=canvas,
            _on_bounding_box_delete_requested=lambda: delete_calls.append("delete"),
        )

        consumed = OrthogonalView.eventFilter(view_like, canvas, _Event())

        self.assertFalse(consumed)
        self.assertEqual(delete_calls, [])
        self.assertFalse(accepted["value"])


if __name__ == "__main__":
    unittest.main()
