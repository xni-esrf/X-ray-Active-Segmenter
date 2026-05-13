from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, Qt

try:
    from src.events import InputHandlers, SyncManager
    from src.ui.orthogonal_view import OrthogonalView
except Exception:  # pragma: no cover - environment dependent
    InputHandlers = None  # type: ignore[assignment]
    SyncManager = None  # type: ignore[assignment]
    OrthogonalView = None  # type: ignore[assignment]


class _KeyEvent:
    def __init__(self, key: int) -> None:
        self._key = int(key)
        self.accepted = False

    def key(self) -> int:
        return self._key

    def accept(self) -> None:
        self.accepted = True


class _Delta:
    def __init__(self, y_value: int) -> None:
        self._y_value = int(y_value)

    def y(self) -> int:
        return self._y_value


class _WheelEvent:
    def __init__(self, *, y_value: int, modifiers: Qt.KeyboardModifier = Qt.NoModifier) -> None:
        self._y_value = int(y_value)
        self._modifiers = modifiers
        self.accepted = False

    def type(self) -> QEvent.Type:
        return QEvent.Wheel

    def modifiers(self) -> Qt.KeyboardModifier:
        return self._modifiers

    def angleDelta(self) -> _Delta:
        return _Delta(self._y_value)

    def accept(self) -> None:
        self.accepted = True


@unittest.skipUnless(
    OrthogonalView is not None and InputHandlers is not None and SyncManager is not None,
    "OrthogonalView and event stack are not available",
)
class OrthogonalViewZoomInputParityTests(unittest.TestCase):
    def _build_view(self, *, initial_zoom: float):
        sync_manager = SyncManager()
        sync_manager.set_zoom(float(initial_zoom))
        view_like = SimpleNamespace(
            _canvas_widget=object(),
            state=SimpleNamespace(axis=0, zoom=float(initial_zoom)),
            input_handlers=InputHandlers(sync_manager),
        )
        view_like._apply_zoom_action = (
            lambda action, target=view_like: OrthogonalView._apply_zoom_action(target, action)
        )
        return view_like, sync_manager

    def test_plus_matches_wheel_up(self) -> None:
        key_view, key_sync = self._build_view(initial_zoom=0.6)
        wheel_view, wheel_sync = self._build_view(initial_zoom=0.6)
        key_event = _KeyEvent(Qt.Key_Plus)
        wheel_event = _WheelEvent(y_value=120)

        OrthogonalView.keyPressEvent(key_view, key_event)
        consumed = OrthogonalView.eventFilter(wheel_view, wheel_view._canvas_widget, wheel_event)

        self.assertTrue(key_event.accepted)
        self.assertTrue(wheel_event.accepted)
        self.assertTrue(consumed)
        self.assertAlmostEqual(key_sync.state.zoom, wheel_sync.state.zoom)
        self.assertAlmostEqual(key_sync.state.zoom, 0.5)

    def test_minus_matches_wheel_down(self) -> None:
        key_view, key_sync = self._build_view(initial_zoom=0.6)
        wheel_view, wheel_sync = self._build_view(initial_zoom=0.6)
        key_event = _KeyEvent(Qt.Key_Minus)
        wheel_event = _WheelEvent(y_value=-120)

        OrthogonalView.keyPressEvent(key_view, key_event)
        consumed = OrthogonalView.eventFilter(wheel_view, wheel_view._canvas_widget, wheel_event)

        self.assertTrue(key_event.accepted)
        self.assertTrue(wheel_event.accepted)
        self.assertTrue(consumed)
        self.assertAlmostEqual(key_sync.state.zoom, wheel_sync.state.zoom)
        self.assertAlmostEqual(key_sync.state.zoom, 0.7)

    def test_upper_clamp_matches_for_minus_and_wheel_down(self) -> None:
        key_view, key_sync = self._build_view(initial_zoom=1.0)
        wheel_view, wheel_sync = self._build_view(initial_zoom=1.0)
        key_event = _KeyEvent(Qt.Key_Minus)
        wheel_event = _WheelEvent(y_value=-120)

        OrthogonalView.keyPressEvent(key_view, key_event)
        OrthogonalView.eventFilter(wheel_view, wheel_view._canvas_widget, wheel_event)

        self.assertAlmostEqual(key_sync.state.zoom, 1.0)
        self.assertAlmostEqual(wheel_sync.state.zoom, 1.0)

    def test_lower_clamp_matches_for_plus_and_wheel_up(self) -> None:
        key_view, key_sync = self._build_view(initial_zoom=0.1)
        wheel_view, wheel_sync = self._build_view(initial_zoom=0.1)
        key_event = _KeyEvent(Qt.Key_Plus)
        wheel_event = _WheelEvent(y_value=120)

        OrthogonalView.keyPressEvent(key_view, key_event)
        OrthogonalView.eventFilter(wheel_view, wheel_view._canvas_widget, wheel_event)

        self.assertAlmostEqual(key_sync.state.zoom, 0.1)
        self.assertAlmostEqual(wheel_sync.state.zoom, 0.1)


if __name__ == "__main__":
    unittest.main()
