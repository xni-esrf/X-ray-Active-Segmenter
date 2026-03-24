from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    from src import app as app_module
except Exception:  # pragma: no cover - environment dependent
    app_module = None  # type: ignore[assignment]


@unittest.skipUnless(app_module is not None, "app module is not available")
class AppStartupLoadingTests(unittest.TestCase):
    def test_run_loads_raw_volume_before_showing_window(self) -> None:
        call_order: list[str] = []
        fake_app = SimpleNamespace(exec=lambda: None)
        fake_renderer = SimpleNamespace()
        fake_sync_manager = SimpleNamespace()
        fake_input_handlers = SimpleNamespace()
        fake_volume = SimpleNamespace()
        fake_cache = SimpleNamespace()
        fake_prepared = SimpleNamespace(
            volume=fake_volume,
            levels=(SimpleNamespace(),),
            cache=fake_cache,
        )
        fake_worker = SimpleNamespace()

        class _MainWindowStub:
            def __init__(self) -> None:
                self._bbox_manager = SimpleNamespace()

            def set_volume(self, volume: object, *, levels: object = None) -> bool:
                call_order.append("set_volume")
                self.last_set_volume = (volume, levels)
                return True

            def render_all(self) -> None:
                call_order.append("render_all")

            def show(self) -> None:
                call_order.append("show")

            def bounding_box_manager(self) -> object:
                return self._bbox_manager

        fake_main_window = _MainWindowStub()
        fake_logger = SimpleNamespace(
            info=lambda *_args, **_kwargs: None,
            warning=lambda *_args, **_kwargs: None,
            exception=lambda *_args, **_kwargs: None,
        )
        fake_config = SimpleNamespace(
            log_level="INFO",
            load_mode="lazy",
            cache_max_bytes=4096,
        )

        def _load_prepared(*_args, **_kwargs):
            call_order.append("load_prepared")
            return fake_prepared

        with patch.object(app_module, "setup_logging") as setup_logging_mock, patch.object(
            app_module, "get_logger", return_value=fake_logger
        ) as get_logger_mock, patch.object(app_module, "QApplication") as qapplication_mock, patch.object(
            app_module, "Renderer", return_value=fake_renderer
        ) as renderer_mock, patch.object(
            app_module, "SyncManager", return_value=fake_sync_manager
        ) as sync_manager_mock, patch.object(
            app_module, "InputHandlers", return_value=fake_input_handlers
        ) as input_handlers_mock, patch.object(
            app_module, "MainWindow", return_value=fake_main_window
        ) as main_window_mock, patch.object(
            app_module, "load_prepared_volume", side_effect=_load_prepared
        ) as load_prepared_mock, patch.object(
            app_module, "IOWorker", return_value=fake_worker
        ) as io_worker_mock, patch.object(
            app_module, "show_warning"
        ) as warning_mock:
            qapplication_mock.instance.return_value = fake_app
            qapplication_mock.return_value = fake_app

            context = app_module.run(
                config=fake_config,
                volume_path="/tmp/good.raw",
                run_event_loop=False,
            )

        setup_logging_mock.assert_called_once_with("INFO")
        get_logger_mock.assert_called_once()
        renderer_mock.assert_called_once()
        sync_manager_mock.assert_called_once()
        input_handlers_mock.assert_called_once_with(sync_manager=fake_sync_manager)
        main_window_mock.assert_called_once()
        load_prepared_mock.assert_called_once()
        io_worker_mock.assert_called_once_with(volume=fake_volume, cache=fake_cache)
        warning_mock.assert_not_called()
        self.assertEqual(
            call_order,
            [
                "load_prepared",
                "set_volume",
                "render_all",
                "show",
            ],
        )
        self.assertIs(context.volume, fake_volume)
        self.assertIs(context.io_worker, fake_worker)

    def test_run_shows_warning_when_startup_raw_load_fails(self) -> None:
        fake_app = SimpleNamespace(exec=lambda: None)
        fake_renderer = SimpleNamespace()
        fake_sync_manager = SimpleNamespace()
        fake_input_handlers = SimpleNamespace()
        fake_main_window = SimpleNamespace(
            show=lambda: None,
            bounding_box_manager=lambda: None,
        )
        fake_logger = SimpleNamespace(
            info=lambda *_args, **_kwargs: None,
            warning=lambda *_args, **_kwargs: None,
            exception=lambda *_args, **_kwargs: None,
        )
        fake_config = SimpleNamespace(
            log_level="INFO",
            load_mode="lazy",
            cache_max_bytes=1024,
        )

        with patch.object(app_module, "setup_logging") as setup_logging_mock, patch.object(
            app_module, "get_logger", return_value=fake_logger
        ) as get_logger_mock, patch.object(app_module, "QApplication") as qapplication_mock, patch.object(
            app_module, "Renderer", return_value=fake_renderer
        ) as renderer_mock, patch.object(
            app_module, "SyncManager", return_value=fake_sync_manager
        ) as sync_manager_mock, patch.object(
            app_module, "InputHandlers", return_value=fake_input_handlers
        ) as input_handlers_mock, patch.object(
            app_module, "MainWindow", return_value=fake_main_window
        ) as main_window_mock, patch.object(
            app_module, "load_prepared_volume", side_effect=RuntimeError("bad raw volume")
        ) as load_prepared_mock, patch.object(
            app_module, "show_warning"
        ) as warning_mock:
            qapplication_mock.instance.return_value = fake_app
            qapplication_mock.return_value = fake_app

            context = app_module.run(
                config=fake_config,
                volume_path="/tmp/bad.raw",
                run_event_loop=False,
            )

        setup_logging_mock.assert_called_once_with("INFO")
        get_logger_mock.assert_called_once()
        renderer_mock.assert_called_once()
        sync_manager_mock.assert_called_once()
        input_handlers_mock.assert_called_once_with(sync_manager=fake_sync_manager)
        main_window_mock.assert_called_once()
        load_prepared_mock.assert_called_once()
        warning_mock.assert_called_once()
        self.assertIn("bad raw volume", warning_mock.call_args.args[0])
        self.assertIs(warning_mock.call_args.kwargs.get("parent"), fake_main_window)
        self.assertIsNone(context.volume)


if __name__ == "__main__":
    unittest.main()
