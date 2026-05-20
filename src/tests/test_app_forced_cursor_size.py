from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    from src import app as app_module
except Exception:  # pragma: no cover - environment dependent
    app_module = None  # type: ignore[assignment]


@unittest.skipUnless(app_module is not None, "app module is not available")
class AppForcedCursorSizeTests(unittest.TestCase):
    def test_resolve_forced_cursor_size_prefers_cli_value(self) -> None:
        logs: list[tuple[str, str]] = []
        logger = SimpleNamespace(
            info=lambda msg, *args: logs.append(("info", msg % args if args else msg)),
            warning=lambda msg, *args: logs.append(("warning", msg % args if args else msg)),
        )

        with patch.dict(os.environ, {"XRA_CURSOR_SIZE": "64"}, clear=False):
            resolved = app_module._resolve_forced_cursor_size(24, logger=logger)

        self.assertEqual(resolved, 24)
        self.assertEqual(logs, [])

    def test_resolve_forced_cursor_size_uses_env_when_cli_missing(self) -> None:
        logger = SimpleNamespace(info=lambda *_a, **_k: None, warning=lambda *_a, **_k: None)
        with patch.dict(os.environ, {"XRA_CURSOR_SIZE": "40"}, clear=False):
            resolved = app_module._resolve_forced_cursor_size(None, logger=logger)
        self.assertEqual(resolved, 40)

    def test_resolve_forced_cursor_size_clamps_and_validates(self) -> None:
        logs: list[tuple[str, str]] = []
        logger = SimpleNamespace(
            info=lambda msg, *args: logs.append(("info", msg % args if args else msg)),
            warning=lambda msg, *args: logs.append(("warning", msg % args if args else msg)),
        )

        self.assertEqual(app_module._resolve_forced_cursor_size(5, logger=logger), 12)
        self.assertEqual(app_module._resolve_forced_cursor_size(300, logger=logger), 128)
        self.assertIsNone(app_module._resolve_forced_cursor_size(0, logger=logger))
        self.assertIsNone(app_module._resolve_forced_cursor_size(-1, logger=logger))
        self.assertIsNone(app_module._resolve_forced_cursor_size("bad", logger=logger))

        warning_lines = [line for level, line in logs if level == "warning"]
        self.assertTrue(any("non-positive" in line for line in warning_lines))
        self.assertTrue(any("invalid" in line for line in warning_lines))

    def test_apply_forced_cursor_size_sets_override_cursor_when_resolved(self) -> None:
        calls: list[object] = []
        app = SimpleNamespace(setOverrideCursor=lambda cursor: calls.append(cursor))
        logger = SimpleNamespace(info=lambda *_a, **_k: None, warning=lambda *_a, **_k: None)

        with patch.object(app_module, "_build_forced_arrow_cursor", return_value="cursor-obj"):
            app_module._apply_forced_cursor_size(app, 24, logger=logger)

        self.assertEqual(calls, ["cursor-obj"])

    def test_apply_forced_cursor_size_is_noop_when_unset(self) -> None:
        calls: list[object] = []
        app = SimpleNamespace(setOverrideCursor=lambda cursor: calls.append(cursor))
        logger = SimpleNamespace(info=lambda *_a, **_k: None, warning=lambda *_a, **_k: None)

        with patch.dict(os.environ, {}, clear=True):
            app_module._apply_forced_cursor_size(app, None, logger=logger)

        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
