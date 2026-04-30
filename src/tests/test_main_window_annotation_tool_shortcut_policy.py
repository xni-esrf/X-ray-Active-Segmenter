from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    import src.ui.main_window as main_window_module
    from src.ui.main_window import MainWindow
except Exception:  # pragma: no cover - environment dependent
    main_window_module = None  # type: ignore[assignment]
    MainWindow = None  # type: ignore[assignment]

try:
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtWidgets import QApplication, QLineEdit, QWidget
except Exception:  # pragma: no cover - environment dependent
    QEvent = None  # type: ignore[assignment]
    Qt = None  # type: ignore[assignment]
    QApplication = None  # type: ignore[assignment]
    QLineEdit = None  # type: ignore[assignment]
    QWidget = None  # type: ignore[assignment]


@unittest.skipUnless(MainWindow is not None, "MainWindow is not available")
class MainWindowAnnotationToolShortcutPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if QApplication is not None:
            cls._app = QApplication.instance() or QApplication([])

    def test_shortcut_switches_tool_when_annotation_mode_is_already_enabled(self) -> None:
        enable_calls: list[bool] = []
        tool_calls: list[str] = []
        window_like = SimpleNamespace(
            state=SimpleNamespace(annotation_mode_enabled=True),
            set_annotation_mode=lambda enabled: enable_calls.append(bool(enabled)),
            _handle_annotation_tool_changed=lambda tool: tool_calls.append(str(tool)),
        )

        applied = MainWindow._apply_annotation_tool_shortcut(window_like, "eraser")

        self.assertTrue(applied)
        self.assertEqual(enable_calls, [])
        self.assertEqual(tool_calls, ["eraser"])

    def test_shortcut_auto_enables_annotation_mode_before_switching_tool(self) -> None:
        enable_calls: list[bool] = []
        tool_calls: list[str] = []
        state = SimpleNamespace(annotation_mode_enabled=False)

        def _enable_annotation(enabled: bool) -> bool:
            enable_calls.append(bool(enabled))
            state.annotation_mode_enabled = bool(enabled)
            return True

        window_like = SimpleNamespace(
            state=state,
            set_annotation_mode=_enable_annotation,
            _handle_annotation_tool_changed=lambda tool: tool_calls.append(str(tool)),
        )

        applied = MainWindow._apply_annotation_tool_shortcut(window_like, "flood_filler")

        self.assertTrue(applied)
        self.assertEqual(enable_calls, [True])
        self.assertEqual(tool_calls, ["flood_filler"])
        self.assertTrue(state.annotation_mode_enabled)

    def test_shortcut_is_ignored_when_annotation_mode_cannot_be_enabled(self) -> None:
        enable_calls: list[bool] = []
        tool_calls: list[str] = []
        state = SimpleNamespace(annotation_mode_enabled=False)
        window_like = SimpleNamespace(
            state=state,
            set_annotation_mode=lambda enabled: enable_calls.append(bool(enabled)) or False,
            _handle_annotation_tool_changed=lambda tool: tool_calls.append(str(tool)),
        )

        applied = MainWindow._apply_annotation_tool_shortcut(window_like, "brush")

        self.assertFalse(applied)
        self.assertEqual(enable_calls, [True])
        self.assertEqual(tool_calls, [])
        self.assertFalse(state.annotation_mode_enabled)

    def test_shortcut_rejects_unknown_tool(self) -> None:
        enable_calls: list[bool] = []
        tool_calls: list[str] = []
        state = SimpleNamespace(annotation_mode_enabled=True)
        window_like = SimpleNamespace(
            state=state,
            set_annotation_mode=lambda enabled: enable_calls.append(bool(enabled)) or True,
            _handle_annotation_tool_changed=lambda tool: tool_calls.append(str(tool)),
        )

        applied = MainWindow._apply_annotation_tool_shortcut(window_like, "not_a_tool")

        self.assertFalse(applied)
        self.assertEqual(enable_calls, [])
        self.assertEqual(tool_calls, [])
        self.assertTrue(state.annotation_mode_enabled)

    @unittest.skipUnless(Qt is not None, "Qt key enums are unavailable")
    def test_keypress_mapping_resolves_ctrl_b_e_f_tools(self) -> None:
        class _Event:
            def __init__(self, key: int, modifiers: object) -> None:
                self._key = key
                self._modifiers = modifiers

            def key(self) -> int:
                return self._key

            def modifiers(self):
                return self._modifiers

        window_like = SimpleNamespace()
        self.assertEqual(
            MainWindow._annotation_tool_from_keypress_event(
                window_like, _Event(int(Qt.Key_B), Qt.ControlModifier)
            ),
            "brush",
        )
        self.assertEqual(
            MainWindow._annotation_tool_from_keypress_event(
                window_like, _Event(int(Qt.Key_E), Qt.ControlModifier)
            ),
            "eraser",
        )
        self.assertEqual(
            MainWindow._annotation_tool_from_keypress_event(
                window_like, _Event(int(Qt.Key_F), Qt.ControlModifier)
            ),
            "flood_filler",
        )

    @unittest.skipUnless(Qt is not None, "Qt key enums are unavailable")
    def test_keypress_mapping_rejects_non_ctrl_or_extra_modifiers(self) -> None:
        class _Event:
            def __init__(self, key: int, modifiers: object) -> None:
                self._key = key
                self._modifiers = modifiers

            def key(self) -> int:
                return self._key

            def modifiers(self):
                return self._modifiers

        window_like = SimpleNamespace()
        self.assertIsNone(
            MainWindow._annotation_tool_from_keypress_event(
                window_like, _Event(int(Qt.Key_E), Qt.NoModifier)
            )
        )
        self.assertIsNone(
            MainWindow._annotation_tool_from_keypress_event(
                window_like, _Event(int(Qt.Key_E), Qt.ControlModifier | Qt.ShiftModifier)
            )
        )
        self.assertIsNone(
            MainWindow._annotation_tool_from_keypress_event(
                window_like, _Event(int(Qt.Key_X), Qt.ControlModifier)
            )
        )

    @unittest.skipUnless(Qt is not None, "Qt key enums are unavailable")
    def test_keypress_mapping_does_not_override_undo_redo_save_shortcuts(self) -> None:
        class _Event:
            def __init__(self, key: int, modifiers: object) -> None:
                self._key = key
                self._modifiers = modifiers

            def key(self) -> int:
                return self._key

            def modifiers(self):
                return self._modifiers

        window_like = SimpleNamespace()
        self.assertIsNone(
            MainWindow._annotation_tool_from_keypress_event(
                window_like, _Event(int(Qt.Key_Z), Qt.ControlModifier)
            )
        )
        self.assertIsNone(
            MainWindow._annotation_tool_from_keypress_event(
                window_like, _Event(int(Qt.Key_Y), Qt.ControlModifier)
            )
        )
        self.assertIsNone(
            MainWindow._annotation_tool_from_keypress_event(
                window_like, _Event(int(Qt.Key_S), Qt.ControlModifier)
            )
        )

    @unittest.skipUnless(
        Qt is not None and QEvent is not None and QWidget is not None,
        "Qt widgets/key enums are unavailable",
    )
    def test_keypress_consumer_applies_ctrl_e_and_consumes_event(self) -> None:
        class _Event:
            def __init__(self, key: int, modifiers: object) -> None:
                self._key = key
                self._modifiers = modifiers
                self.accept_count = 0

            def type(self):
                return QEvent.Type.KeyPress

            def key(self) -> int:
                return self._key

            def modifiers(self):
                return self._modifiers

            def accept(self) -> None:
                self.accept_count += 1

        applied: list[str] = []
        window_like = SimpleNamespace(
            isActiveWindow=lambda: True,
            isAncestorOf=lambda _obj: True,
            _annotation_tool_from_keypress_event=(
                lambda event: MainWindow._annotation_tool_from_keypress_event(
                    SimpleNamespace(), event
                )
            ),
            _apply_annotation_tool_shortcut=lambda tool: applied.append(str(tool)) or True,
        )
        source_widget = QWidget()
        event = _Event(int(Qt.Key_E), Qt.ControlModifier)

        consumed = MainWindow._maybe_consume_annotation_tool_shortcut_event(
            window_like, source_widget, event
        )

        self.assertTrue(consumed)
        self.assertEqual(applied, ["eraser"])
        self.assertEqual(event.accept_count, 1)

    @unittest.skipUnless(
        Qt is not None and QEvent is not None and QWidget is not None,
        "Qt widgets/key enums are unavailable",
    )
    def test_keypress_consumer_leaves_ctrl_s_for_existing_save_shortcut(self) -> None:
        class _Event:
            def __init__(self, key: int, modifiers: object) -> None:
                self._key = key
                self._modifiers = modifiers
                self.accept_count = 0

            def type(self):
                return QEvent.Type.KeyPress

            def key(self) -> int:
                return self._key

            def modifiers(self):
                return self._modifiers

            def accept(self) -> None:
                self.accept_count += 1

        applied: list[str] = []
        window_like = SimpleNamespace(
            isActiveWindow=lambda: True,
            isAncestorOf=lambda _obj: True,
            _annotation_tool_from_keypress_event=(
                lambda event: MainWindow._annotation_tool_from_keypress_event(
                    SimpleNamespace(), event
                )
            ),
            _apply_annotation_tool_shortcut=lambda tool: applied.append(str(tool)) or True,
        )
        source_widget = QWidget()
        event = _Event(int(Qt.Key_S), Qt.ControlModifier)

        consumed = MainWindow._maybe_consume_annotation_tool_shortcut_event(
            window_like, source_widget, event
        )

        self.assertFalse(consumed)
        self.assertEqual(applied, [])
        self.assertEqual(event.accept_count, 0)

    @unittest.skipUnless(
        Qt is not None and QEvent is not None and QWidget is not None,
        "Qt widgets/key enums are unavailable",
    )
    def test_keypress_consumer_ignores_inactive_window(self) -> None:
        class _Event:
            def __init__(self, key: int, modifiers: object) -> None:
                self._key = key
                self._modifiers = modifiers
                self.accept_count = 0

            def type(self):
                return QEvent.Type.KeyPress

            def key(self) -> int:
                return self._key

            def modifiers(self):
                return self._modifiers

            def accept(self) -> None:
                self.accept_count += 1

        applied: list[str] = []
        window_like = SimpleNamespace(
            isActiveWindow=lambda: False,
            isAncestorOf=lambda _obj: True,
            _annotation_tool_from_keypress_event=(
                lambda event: MainWindow._annotation_tool_from_keypress_event(
                    SimpleNamespace(), event
                )
            ),
            _apply_annotation_tool_shortcut=lambda tool: applied.append(str(tool)) or True,
        )
        source_widget = QWidget()
        event = _Event(int(Qt.Key_E), Qt.ControlModifier)

        consumed = MainWindow._maybe_consume_annotation_tool_shortcut_event(
            window_like, source_widget, event
        )

        self.assertFalse(consumed)
        self.assertEqual(applied, [])
        self.assertEqual(event.accept_count, 0)

    @unittest.skipUnless(
        Qt is not None and QEvent is not None and QWidget is not None,
        "Qt widgets/key enums are unavailable",
    )
    def test_keypress_consumer_ignores_widgets_outside_main_window(self) -> None:
        class _Event:
            def __init__(self, key: int, modifiers: object) -> None:
                self._key = key
                self._modifiers = modifiers
                self.accept_count = 0

            def type(self):
                return QEvent.Type.KeyPress

            def key(self) -> int:
                return self._key

            def modifiers(self):
                return self._modifiers

            def accept(self) -> None:
                self.accept_count += 1

        applied: list[str] = []
        window_like = SimpleNamespace(
            isActiveWindow=lambda: True,
            isAncestorOf=lambda _obj: False,
            _annotation_tool_from_keypress_event=(
                lambda event: MainWindow._annotation_tool_from_keypress_event(
                    SimpleNamespace(), event
                )
            ),
            _apply_annotation_tool_shortcut=lambda tool: applied.append(str(tool)) or True,
        )
        source_widget = QWidget()
        event = _Event(int(Qt.Key_E), Qt.ControlModifier)

        consumed = MainWindow._maybe_consume_annotation_tool_shortcut_event(
            window_like, source_widget, event
        )

        self.assertFalse(consumed)
        self.assertEqual(applied, [])
        self.assertEqual(event.accept_count, 0)

    @unittest.skipUnless(
        Qt is not None and QEvent is not None and QWidget is not None,
        "Qt widgets/key enums are unavailable",
    )
    def test_bbox_delete_key_consumer_consumes_backspace_for_left_panel_widgets(self) -> None:
        class _Event:
            def __init__(self, key: int) -> None:
                self._key = key
                self.accept_count = 0

            def type(self):
                return QEvent.Type.KeyPress

            def key(self) -> int:
                return self._key

            def accept(self) -> None:
                self.accept_count += 1

        delete_calls: list[str] = []
        left_panel = QWidget()
        child = QWidget(left_panel)
        window_like = SimpleNamespace(
            _left_panel=left_panel,
            isActiveWindow=lambda: True,
            isAncestorOf=lambda _obj: True,
            _handle_bounding_box_delete_shortcut_requested=lambda: delete_calls.append("delete"),
        )
        event = _Event(int(Qt.Key_Backspace))

        consumed = MainWindow._maybe_consume_bbox_delete_shortcut_event(
            window_like,
            child,
            event,
        )

        self.assertTrue(consumed)
        self.assertEqual(delete_calls, ["delete"])
        self.assertEqual(event.accept_count, 1)

    @unittest.skipUnless(
        Qt is not None and QEvent is not None and QWidget is not None and QLineEdit is not None,
        "Qt widgets/key enums are unavailable",
    )
    def test_bbox_delete_key_consumer_ignores_text_input_widgets(self) -> None:
        class _Event:
            def __init__(self, key: int) -> None:
                self._key = key
                self.accept_count = 0

            def type(self):
                return QEvent.Type.KeyPress

            def key(self) -> int:
                return self._key

            def accept(self) -> None:
                self.accept_count += 1

        delete_calls: list[str] = []
        left_panel = QWidget()
        line_edit = QLineEdit(left_panel)
        window_like = SimpleNamespace(
            _left_panel=left_panel,
            isActiveWindow=lambda: True,
            isAncestorOf=lambda _obj: True,
            _handle_bounding_box_delete_shortcut_requested=lambda: delete_calls.append("delete"),
        )
        event = _Event(int(Qt.Key_Delete))

        consumed = MainWindow._maybe_consume_bbox_delete_shortcut_event(
            window_like,
            line_edit,
            event,
        )

        self.assertFalse(consumed)
        self.assertEqual(delete_calls, [])
        self.assertEqual(event.accept_count, 0)

    @unittest.skipUnless(
        Qt is not None and QEvent is not None and QWidget is not None,
        "Qt widgets/key enums are unavailable",
    )
    def test_bbox_delete_key_consumer_ignores_widgets_outside_left_panel(self) -> None:
        class _Event:
            def __init__(self, key: int) -> None:
                self._key = key
                self.accept_count = 0

            def type(self):
                return QEvent.Type.KeyPress

            def key(self) -> int:
                return self._key

            def accept(self) -> None:
                self.accept_count += 1

        delete_calls: list[str] = []
        left_panel = QWidget()
        external_widget = QWidget()
        window_like = SimpleNamespace(
            _left_panel=left_panel,
            isActiveWindow=lambda: True,
            isAncestorOf=lambda _obj: True,
            _handle_bounding_box_delete_shortcut_requested=lambda: delete_calls.append("delete"),
        )
        event = _Event(int(Qt.Key_Backspace))

        consumed = MainWindow._maybe_consume_bbox_delete_shortcut_event(
            window_like,
            external_widget,
            event,
        )

        self.assertFalse(consumed)
        self.assertEqual(delete_calls, [])
        self.assertEqual(event.accept_count, 0)

    def test_eventfilter_prioritizes_bbox_delete_consumer(self) -> None:
        calls: list[str] = []
        window_like = SimpleNamespace(
            _maybe_consume_bbox_delete_shortcut_event=lambda obj, event: calls.append("bbox") or True,
            _maybe_consume_annotation_tool_shortcut_event=lambda obj, event: calls.append("annot") or True,
        )

        consumed = MainWindow.eventFilter(window_like, object(), object())

        self.assertTrue(consumed)
        self.assertEqual(calls, ["bbox"])

    def test_eventfilter_uses_annotation_consumer_when_bbox_not_consumed(self) -> None:
        calls: list[str] = []
        window_like = SimpleNamespace(
            _maybe_consume_bbox_delete_shortcut_event=lambda obj, event: calls.append("bbox") or False,
            _maybe_consume_annotation_tool_shortcut_event=lambda obj, event: calls.append("annot") or True,
        )

        consumed = MainWindow.eventFilter(window_like, object(), object())

        self.assertTrue(consumed)
        self.assertEqual(calls, ["bbox", "annot"])

    @unittest.skipUnless(MainWindow is not None and QApplication is not None, "Qt/MainWindow unavailable")
    def test_close_event_removes_app_event_filter_when_close_is_accepted(self) -> None:
        removed_filters: list[object] = []
        quit_on_last_closed_values: list[bool] = []
        app_like = SimpleNamespace(
            removeEventFilter=lambda obj: removed_filters.append(obj),
            setQuitOnLastWindowClosed=lambda value: quit_on_last_closed_values.append(bool(value)),
        )
        event = SimpleNamespace(
            accept_calls=0,
            ignore_calls=0,
            accept=lambda: None,
            ignore=lambda: None,
        )
        event.accept = lambda: setattr(event, "accept_calls", event.accept_calls + 1)
        event.ignore = lambda: setattr(event, "ignore_calls", event.ignore_calls + 1)

        window_like = SimpleNamespace(
            _maybe_resolve_unsaved_data_before_close=lambda: True,
            _training_is_running=lambda: False,
            _clear_deferred_close_training_state=lambda: None,
            _app_event_filter_installed=True,
        )

        with patch("src.ui.main_window.QApplication.instance", return_value=app_like):
            MainWindow.closeEvent(window_like, event)

        self.assertEqual(removed_filters, [window_like])
        self.assertEqual(quit_on_last_closed_values, [])
        self.assertFalse(window_like._app_event_filter_installed)
        self.assertEqual(event.accept_calls, 1)
        self.assertEqual(event.ignore_calls, 0)

    @unittest.skipUnless(MainWindow is not None, "MainWindow is not available")
    def test_unsaved_data_resolution_before_close_runs_segmentation_then_bboxes(self) -> None:
        calls: list[str] = []

        window_like = SimpleNamespace(
            _maybe_resolve_unsaved_segmentation=lambda context: calls.append("segmentation") or True,
            _maybe_resolve_unsaved_bounding_boxes=lambda context: calls.append("bboxes") or True,
        )

        allowed = MainWindow._maybe_resolve_unsaved_data_before_close(window_like)

        self.assertTrue(allowed)
        self.assertEqual(calls, ["segmentation", "bboxes"])

    @unittest.skipUnless(MainWindow is not None, "MainWindow is not available")
    def test_unsaved_data_resolution_before_close_short_circuits_when_segmentation_cancels(self) -> None:
        calls: list[str] = []

        window_like = SimpleNamespace(
            _maybe_resolve_unsaved_segmentation=lambda context: calls.append("segmentation") or False,
            _maybe_resolve_unsaved_bounding_boxes=lambda context: calls.append("bboxes") or True,
        )

        allowed = MainWindow._maybe_resolve_unsaved_data_before_close(window_like)

        self.assertFalse(allowed)
        self.assertEqual(calls, ["segmentation"])

    @unittest.skipUnless(MainWindow is not None, "MainWindow is not available")
    def test_unsaved_data_resolution_before_close_returns_false_when_bboxes_cancel(self) -> None:
        calls: list[str] = []

        window_like = SimpleNamespace(
            _maybe_resolve_unsaved_segmentation=lambda context: calls.append("segmentation") or True,
            _maybe_resolve_unsaved_bounding_boxes=lambda context: calls.append("bboxes") or False,
        )

        allowed = MainWindow._maybe_resolve_unsaved_data_before_close(window_like)

        self.assertFalse(allowed)
        self.assertEqual(calls, ["segmentation", "bboxes"])

    @unittest.skipUnless(MainWindow is not None, "MainWindow is not available")
    def test_close_event_keeps_app_event_filter_when_close_is_blocked(self) -> None:
        event = SimpleNamespace(
            accept_calls=0,
            ignore_calls=0,
            accept=lambda: None,
            ignore=lambda: None,
        )
        event.accept = lambda: setattr(event, "accept_calls", event.accept_calls + 1)
        event.ignore = lambda: setattr(event, "ignore_calls", event.ignore_calls + 1)
        window_like = SimpleNamespace(
            _maybe_resolve_unsaved_data_before_close=lambda: False,
            _app_event_filter_installed=True,
        )

        MainWindow.closeEvent(window_like, event)

        self.assertTrue(window_like._app_event_filter_installed)
        self.assertEqual(event.accept_calls, 0)
        self.assertEqual(event.ignore_calls, 1)

    @unittest.skipUnless(MainWindow is not None and QApplication is not None, "Qt/MainWindow unavailable")
    def test_close_event_training_running_defers_app_quit_when_close_is_prepared(self) -> None:
        removed_filters: list[object] = []
        quit_on_last_closed_values: list[bool] = []
        app_like = SimpleNamespace(
            removeEventFilter=lambda obj: removed_filters.append(obj),
            setQuitOnLastWindowClosed=lambda value: quit_on_last_closed_values.append(bool(value)),
        )
        event = SimpleNamespace(
            accept_calls=0,
            ignore_calls=0,
            accept=lambda: None,
            ignore=lambda: None,
        )
        event.accept = lambda: setattr(event, "accept_calls", event.accept_calls + 1)
        event.ignore = lambda: setattr(event, "ignore_calls", event.ignore_calls + 1)

        window_like = SimpleNamespace(
            _maybe_resolve_unsaved_data_before_close=lambda: True,
            _training_is_running=lambda: True,
            _maybe_prepare_close_while_training=lambda: True,
            _app_event_filter_installed=True,
        )

        with patch("src.ui.main_window.QApplication.instance", return_value=app_like):
            MainWindow.closeEvent(window_like, event)

        self.assertEqual(quit_on_last_closed_values, [False])
        self.assertEqual(removed_filters, [window_like])
        self.assertFalse(window_like._app_event_filter_installed)
        self.assertEqual(event.accept_calls, 1)
        self.assertEqual(event.ignore_calls, 0)

    @unittest.skipUnless(MainWindow is not None, "MainWindow is not available")
    def test_close_event_training_running_ignores_when_training_close_preparation_fails(self) -> None:
        event = SimpleNamespace(
            accept_calls=0,
            ignore_calls=0,
            accept=lambda: None,
            ignore=lambda: None,
        )
        event.accept = lambda: setattr(event, "accept_calls", event.accept_calls + 1)
        event.ignore = lambda: setattr(event, "ignore_calls", event.ignore_calls + 1)

        window_like = SimpleNamespace(
            _maybe_resolve_unsaved_data_before_close=lambda: True,
            _training_is_running=lambda: True,
            _maybe_prepare_close_while_training=lambda: False,
            _app_event_filter_installed=True,
        )

        MainWindow.closeEvent(window_like, event)

        self.assertTrue(window_like._app_event_filter_installed)
        self.assertEqual(event.accept_calls, 0)
        self.assertEqual(event.ignore_calls, 1)

    @unittest.skipUnless(MainWindow is not None and main_window_module is not None, "MainWindow is not available")
    def test_prepare_close_while_training_cancel_keeps_window_open(self) -> None:
        clears: list[str] = []
        window_like = SimpleNamespace(
            _clear_deferred_close_training_state=lambda: clears.append("clear"),
            _set_deferred_close_after_stop_training=lambda: None,
            _request_learning_training_stop=lambda: None,
            _set_deferred_close_with_background_training=lambda **_: None,
        )

        with patch(
            "src.ui.main_window.ask_training_running_close_decision",
            return_value=main_window_module.TrainingCloseDecision.CANCEL,
        ):
            should_close = MainWindow._maybe_prepare_close_while_training(window_like)

        self.assertFalse(should_close)
        self.assertEqual(clears, ["clear"])

    @unittest.skipUnless(MainWindow is not None and main_window_module is not None, "MainWindow is not available")
    def test_prepare_close_while_training_stop_requests_graceful_stop(self) -> None:
        actions: list[str] = []
        window_like = SimpleNamespace(
            _clear_deferred_close_training_state=lambda: actions.append("clear"),
            _set_deferred_close_after_stop_training=lambda: actions.append("set_stop_mode"),
            _request_learning_training_stop=lambda: actions.append("request_stop"),
            _set_deferred_close_with_background_training=lambda **_: actions.append("set_bg_mode"),
        )

        with patch(
            "src.ui.main_window.ask_training_running_close_decision",
            return_value=main_window_module.TrainingCloseDecision.STOP_AND_CLOSE,
        ):
            should_close = MainWindow._maybe_prepare_close_while_training(window_like)

        self.assertTrue(should_close)
        self.assertEqual(actions, ["set_stop_mode", "request_stop"])

    @unittest.skipUnless(MainWindow is not None and main_window_module is not None, "MainWindow is not available")
    def test_prepare_close_while_training_continue_configures_background_checkpoint(self) -> None:
        actions: list[tuple[str, object]] = []
        window_like = SimpleNamespace(
            _clear_deferred_close_training_state=lambda: actions.append(("clear", None)),
            _set_deferred_close_after_stop_training=lambda: actions.append(("set_stop_mode", None)),
            _request_learning_training_stop=lambda: actions.append(("request_stop", None)),
            _set_deferred_close_with_background_training=lambda **kwargs: actions.append(
                ("set_bg_mode", kwargs.get("checkpoint_path"))
            ),
        )

        with patch(
            "src.ui.main_window.ask_training_running_close_decision",
            return_value=main_window_module.TrainingCloseDecision.CONTINUE_IN_BACKGROUND,
        ), patch(
            "src.ui.main_window.open_save_model_checkpoint_dialog",
            return_value=SimpleNamespace(accepted=True, path="/tmp/background_best.cp"),
        ) as save_dialog_mock:
            should_close = MainWindow._maybe_prepare_close_while_training(window_like)

        self.assertTrue(should_close)
        self.assertEqual(actions, [("set_bg_mode", "/tmp/background_best.cp")])
        save_dialog_mock.assert_called_once_with(
            window_like,
            retry_on_overwrite_decline=True,
        )

    @unittest.skipUnless(MainWindow is not None and main_window_module is not None, "MainWindow is not available")
    def test_prepare_close_while_training_continue_cancelled_checkpoint_picker_keeps_window_open(self) -> None:
        actions: list[str] = []
        window_like = SimpleNamespace(
            _clear_deferred_close_training_state=lambda: actions.append("clear"),
            _set_deferred_close_after_stop_training=lambda: actions.append("set_stop_mode"),
            _request_learning_training_stop=lambda: actions.append("request_stop"),
            _set_deferred_close_with_background_training=lambda **_: actions.append("set_bg_mode"),
        )

        with patch(
            "src.ui.main_window.ask_training_running_close_decision",
            return_value=main_window_module.TrainingCloseDecision.CONTINUE_IN_BACKGROUND,
        ), patch(
            "src.ui.main_window.open_save_model_checkpoint_dialog",
            return_value=SimpleNamespace(accepted=False, path=None),
        ):
            should_close = MainWindow._maybe_prepare_close_while_training(window_like)

        self.assertFalse(should_close)
        self.assertEqual(actions, ["clear"])


if __name__ == "__main__":
    unittest.main()
