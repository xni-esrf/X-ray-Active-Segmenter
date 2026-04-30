from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from src.ui.main_window import MainWindow
except Exception:  # pragma: no cover - environment dependent
    MainWindow = None  # type: ignore[assignment]


@unittest.skipUnless(MainWindow is not None, "MainWindow is not available")
class MainWindowContrastFlowTests(unittest.TestCase):
    def test_sync_contrast_controls_from_renderer_uses_renderer_ranges(self) -> None:
        calls: list[tuple[str, object]] = []
        renderer = SimpleNamespace(
            get_data_range=lambda: (10.0, 200.0),
            get_window_range=lambda: (25.0, 175.0),
        )
        bottom_panel = SimpleNamespace(
            set_contrast_range=lambda value: calls.append(("range", value)),
            set_contrast_window=lambda value: calls.append(("window", value)),
        )
        window_like = SimpleNamespace(renderer=renderer, bottom_panel=bottom_panel)

        MainWindow._sync_contrast_controls_from_renderer(window_like)

        self.assertEqual(
            calls,
            [
                ("range", (10.0, 200.0)),
                ("window", (25.0, 175.0)),
            ],
        )

    def test_sync_level_mode_controls_from_renderer_uses_renderer_state(self) -> None:
        calls: list[dict[str, object]] = []
        renderer = SimpleNamespace(
            available_level_count=lambda: 5,
            is_auto_level_enabled=lambda: False,
            manual_level=lambda: 3,
        )
        bottom_panel = SimpleNamespace(
            set_level_mode=lambda **kwargs: calls.append(dict(kwargs)),
        )
        window_like = SimpleNamespace(renderer=renderer, bottom_panel=bottom_panel)

        MainWindow._sync_level_mode_controls_from_renderer(window_like)

        self.assertEqual(
            calls,
            [
                {
                    "auto_enabled": False,
                    "manual_level": 3,
                    "max_level": 4,
                }
            ],
        )

    def test_handle_contrast_window_changed_applies_window_and_queues_rerender(self) -> None:
        set_window_calls: list[tuple[float, float]] = []
        queue_calls: list[str] = []
        sync_calls: list[str] = []
        renderer = SimpleNamespace(
            set_window=lambda vmin, vmax: set_window_calls.append((float(vmin), float(vmax)))
        )
        window_like = SimpleNamespace(
            state=SimpleNamespace(volume_loaded=True),
            renderer=renderer,
            _queue_contrast_rerender=lambda: queue_calls.append("queue"),
            _sync_contrast_controls_from_renderer=lambda: sync_calls.append("sync"),
        )

        with patch("src.ui.main_window.show_warning") as warning_mock:
            MainWindow._handle_contrast_window_changed(window_like, 12.0, 34.0)

        self.assertEqual(set_window_calls, [(12.0, 34.0)])
        self.assertEqual(queue_calls, ["queue"])
        self.assertEqual(sync_calls, [])
        warning_mock.assert_not_called()

    def test_handle_contrast_window_changed_resyncs_and_warns_on_error(self) -> None:
        queue_calls: list[str] = []
        sync_calls: list[str] = []

        def _raise(_vmin: float, _vmax: float) -> None:
            raise ValueError("window error")

        renderer = SimpleNamespace(set_window=_raise)
        window_like = SimpleNamespace(
            state=SimpleNamespace(volume_loaded=True),
            renderer=renderer,
            _queue_contrast_rerender=lambda: queue_calls.append("queue"),
            _sync_contrast_controls_from_renderer=lambda: sync_calls.append("sync"),
        )

        with patch("src.ui.main_window.show_warning") as warning_mock:
            MainWindow._handle_contrast_window_changed(window_like, 12.0, 34.0)

        self.assertEqual(sync_calls, ["sync"])
        self.assertEqual(queue_calls, [])
        warning_mock.assert_called_once()
        self.assertIn("window error", warning_mock.call_args.args[0])
        self.assertIs(warning_mock.call_args.kwargs["parent"], window_like)

    def test_handle_contrast_window_changed_is_noop_when_no_volume_loaded(self) -> None:
        set_window_calls: list[tuple[float, float]] = []
        queue_calls: list[str] = []
        renderer = SimpleNamespace(
            set_window=lambda vmin, vmax: set_window_calls.append((float(vmin), float(vmax)))
        )
        window_like = SimpleNamespace(
            state=SimpleNamespace(volume_loaded=False),
            renderer=renderer,
            _queue_contrast_rerender=lambda: queue_calls.append("queue"),
            _sync_contrast_controls_from_renderer=lambda: None,
        )

        with patch("src.ui.main_window.show_warning") as warning_mock:
            MainWindow._handle_contrast_window_changed(window_like, 12.0, 34.0)

        self.assertEqual(set_window_calls, [])
        self.assertEqual(queue_calls, [])
        warning_mock.assert_not_called()

    def test_queue_contrast_rerender_queues_each_view(self) -> None:
        queued_view_ids: list[str] = []
        render_calls: list[str] = []
        window_like = SimpleNamespace(
            views={
                "axial": object(),
                "coronal": object(),
                "sagittal": object(),
            },
            _queue_render=lambda view_id: queued_view_ids.append(str(view_id)),
            render_all=lambda: render_calls.append("render"),
        )

        MainWindow._queue_contrast_rerender(window_like)

        self.assertEqual(queued_view_ids, ["axial", "coronal", "sagittal"])
        self.assertEqual(render_calls, [])

    def test_queue_contrast_rerender_falls_back_when_views_missing(self) -> None:
        queued_view_ids: list[str] = []
        render_calls: list[str] = []
        window_like = SimpleNamespace(
            views={},
            _queue_render=lambda view_id: queued_view_ids.append(str(view_id)),
            render_all=lambda: render_calls.append("render"),
        )

        MainWindow._queue_contrast_rerender(window_like)

        self.assertEqual(queued_view_ids, [])
        self.assertEqual(render_calls, ["render"])

    def test_handle_auto_level_mode_changed_updates_renderer_and_rerenders_when_reenabled(self) -> None:
        mode_calls: list[bool] = []
        queue_calls: list[str] = []
        render_calls: list[str] = []
        renderer = SimpleNamespace(set_auto_level_enabled=lambda enabled: mode_calls.append(bool(enabled)))
        window_like = SimpleNamespace(
            state=SimpleNamespace(volume_loaded=True),
            renderer=renderer,
            _queue_contrast_rerender=lambda: queue_calls.append("queue"),
            render_all=lambda: render_calls.append("render"),
        )

        MainWindow._handle_auto_level_mode_changed(window_like, False)
        MainWindow._handle_auto_level_mode_changed(window_like, True)

        self.assertEqual(mode_calls, [False, True])
        self.assertEqual(queue_calls, ["queue"])
        self.assertEqual(render_calls, ["render"])

    def test_handle_auto_level_mode_changed_is_noop_when_no_volume_loaded(self) -> None:
        mode_calls: list[bool] = []
        queue_calls: list[str] = []
        render_calls: list[str] = []
        renderer = SimpleNamespace(set_auto_level_enabled=lambda enabled: mode_calls.append(bool(enabled)))
        window_like = SimpleNamespace(
            state=SimpleNamespace(volume_loaded=False),
            renderer=renderer,
            _queue_contrast_rerender=lambda: queue_calls.append("queue"),
            render_all=lambda: render_calls.append("render"),
        )

        MainWindow._handle_auto_level_mode_changed(window_like, False)
        MainWindow._handle_auto_level_mode_changed(window_like, True)

        self.assertEqual(mode_calls, [])
        self.assertEqual(queue_calls, [])
        self.assertEqual(render_calls, [])

    def test_handle_manual_level_requested_updates_renderer_and_queues(self) -> None:
        manual_calls: list[int] = []
        queue_calls: list[str] = []
        renderer = SimpleNamespace(set_manual_level=lambda level: manual_calls.append(int(level)))
        window_like = SimpleNamespace(
            state=SimpleNamespace(volume_loaded=True),
            renderer=renderer,
            _queue_contrast_rerender=lambda: queue_calls.append("queue"),
        )

        MainWindow._handle_manual_level_requested(window_like, 3)

        self.assertEqual(manual_calls, [3])
        self.assertEqual(queue_calls, ["queue"])

    def test_handle_manual_level_requested_is_noop_when_no_volume_loaded(self) -> None:
        manual_calls: list[int] = []
        queue_calls: list[str] = []
        renderer = SimpleNamespace(set_manual_level=lambda level: manual_calls.append(int(level)))
        window_like = SimpleNamespace(
            state=SimpleNamespace(volume_loaded=False),
            renderer=renderer,
            _queue_contrast_rerender=lambda: queue_calls.append("queue"),
        )

        MainWindow._handle_manual_level_requested(window_like, 3)

        self.assertEqual(manual_calls, [])
        self.assertEqual(queue_calls, [])

    def test_handle_segmentation_opacity_changed_applies_to_all_views(self) -> None:
        applied: list[tuple[str, float]] = []
        window_like = SimpleNamespace(
            views={
                "axial": SimpleNamespace(
                    set_segmentation_opacity=lambda value: applied.append(("axial", float(value)))
                ),
                "coronal": SimpleNamespace(
                    set_segmentation_opacity=lambda value: applied.append(("coronal", float(value)))
                ),
                "sagittal": SimpleNamespace(
                    set_segmentation_opacity=lambda value: applied.append(("sagittal", float(value)))
                ),
            }
        )

        MainWindow._handle_segmentation_opacity_changed(window_like, 0.42)

        self.assertEqual(
            applied,
            [("axial", 0.42), ("coronal", 0.42), ("sagittal", 0.42)],
        )

    def test_update_active_levels_status_marks_manual_forced_mode(self) -> None:
        calls: list[dict[str, object]] = []
        renderer = SimpleNamespace(
            latest_result=lambda view_id: {
                "axial": SimpleNamespace(level=2, level_scale=4),
                "coronal": SimpleNamespace(level=2, level_scale=4),
                "sagittal": SimpleNamespace(level=2, level_scale=4),
            }.get(view_id),
            is_auto_level_enabled=lambda: False,
        )
        bottom_panel = SimpleNamespace(set_active_levels=lambda **kwargs: calls.append(dict(kwargs)))
        window_like = SimpleNamespace(renderer=renderer, bottom_panel=bottom_panel)

        MainWindow._update_active_levels_status(window_like)

        self.assertEqual(
            calls,
            [
                {
                    "axial": (2, 4),
                    "coronal": (2, 4),
                    "sagittal": (2, 4),
                    "forced": True,
                }
            ],
        )

    def test_attach_segmentation_editor_syncs_level_mode_controls(self) -> None:
        sync_calls: list[str] = []
        attach_calls: list[tuple[object, object]] = []
        label_calls: list[tuple[int, ...]] = []
        editable_volume = SimpleNamespace(loader=SimpleNamespace(path="/tmp/semantic.editable"))
        editor = SimpleNamespace(
            source_path="/tmp/semantic.raw",
            to_volume_data=lambda path: editable_volume,
            labels_in_use=lambda **_kwargs: (0, 1, 2),
        )
        renderer = SimpleNamespace(
            attach_segmentation=lambda volume, *, levels=None: attach_calls.append((volume, levels)),
            set_segmentation_labels=lambda labels: label_calls.append(tuple(labels)),
        )
        window_like = SimpleNamespace(
            _segmentation_editor=None,
            _instance_volume=None,
            _instance_worker=None,
            _semantic_volume=None,
            _semantic_worker=None,
            renderer=renderer,
            _editable_segmentation_levels=lambda _volume: ("level0", "level1"),
            _sync_level_mode_controls_from_renderer=lambda: sync_calls.append("sync"),
            _refresh_hover_readout=lambda: None,
            _refresh_annotation_ui_state=lambda: None,
            _annotation_labels_dirty=True,
        )

        MainWindow._attach_segmentation_editor(window_like, editor, kind="semantic")

        self.assertIs(window_like._segmentation_editor, editor)
        self.assertIs(window_like._semantic_volume, editable_volume)
        self.assertIsNone(window_like._instance_volume)
        self.assertEqual(attach_calls, [(editable_volume, ("level0", "level1"))])
        self.assertEqual(sync_calls, ["sync"])
        self.assertEqual(label_calls, [(0, 1, 2)])
        self.assertFalse(window_like._annotation_labels_dirty)

    def test_sync_segmentation_volume_from_editor_reattach_syncs_level_mode_controls(self) -> None:
        sync_calls: list[str] = []
        attach_calls: list[tuple[object, object]] = []
        label_calls: list[tuple[int, ...]] = []
        editor = SimpleNamespace(
            kind="semantic",
            source_path="/tmp/semantic.raw",
            to_volume_data=lambda path: SimpleNamespace(loader=SimpleNamespace(path=path)),
            labels_in_use=lambda **_kwargs: (0, 4),
        )
        renderer = SimpleNamespace(
            attach_segmentation=lambda volume, *, levels=None: attach_calls.append((volume, levels)),
            set_segmentation_labels=lambda labels: label_calls.append(tuple(labels)),
        )
        window_like = SimpleNamespace(
            _segmentation_editor=editor,
            _semantic_volume=None,
            _semantic_worker=object(),
            _instance_volume=object(),
            _instance_worker=object(),
            renderer=renderer,
            _editable_segmentation_levels=lambda volume: (volume, "next"),
            _sync_level_mode_controls_from_renderer=lambda: sync_calls.append("sync"),
            _annotation_labels_dirty=True,
        )

        result = MainWindow._sync_segmentation_volume_from_editor(window_like, reattach_renderer=True)

        self.assertEqual(result[0], "semantic")
        self.assertEqual(len(attach_calls), 1)
        attached_volume, attached_levels = attach_calls[0]
        self.assertIs(window_like._semantic_volume, attached_volume)
        self.assertIsNone(window_like._instance_volume)
        self.assertEqual(attached_levels, (attached_volume, "next"))
        self.assertEqual(sync_calls, ["sync"])
        self.assertEqual(label_calls, [(0, 4)])
        self.assertFalse(window_like._annotation_labels_dirty)


if __name__ == "__main__":
    unittest.main()
