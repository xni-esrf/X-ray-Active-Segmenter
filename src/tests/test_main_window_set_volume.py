from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from src.ui.main_window import MainWindow
except Exception:  # pragma: no cover - environment dependent
    MainWindow = None  # type: ignore[assignment]


@unittest.skipUnless(MainWindow is not None, "MainWindow is not available")
class MainWindowSetVolumeTests(unittest.TestCase):
    def test_set_volume_resets_contrast_window_on_new_raw_load(self) -> None:
        contrast_calls: list[tuple[str, object]] = []
        level_mode_calls: list[dict[str, object]] = []
        history_calls: list[str] = []
        picker_calls: list[str] = []
        sync_bbox_calls: list[str] = []
        cursor_calls: list[tuple[int, int, int]] = []
        info_calls: list[object] = []
        refresh_calls: list[str] = []

        class _RendererStub:
            def __init__(self) -> None:
                self._data_range: tuple[float, float] | None = None
                self._window_range: tuple[float, float] | None = None
                self._auto_level_enabled = True
                self._manual_level = 0
                self._available_level_count = 0
                self.attach_calls: list[tuple[object, object]] = []

            def attach_volume(self, volume: object, *, levels: object = None) -> None:
                self.attach_calls.append((volume, levels))
                data_range = getattr(volume, "data_range", None)
                if data_range is None:
                    raise RuntimeError("Missing data range")
                data_min, data_max = data_range
                self._data_range = (float(data_min), float(data_max))
                self._window_range = self._data_range
                if levels is None:
                    self._available_level_count = 1
                else:
                    self._available_level_count = max(1, len(tuple(levels)))

            def detach_segmentation(self) -> None:
                return None

            def set_window(self, vmin: float, vmax: float) -> None:
                self._window_range = (float(vmin), float(vmax))

            def get_data_range(self) -> tuple[float, float] | None:
                return self._data_range

            def get_window_range(self) -> tuple[float, float] | None:
                return self._window_range

            def is_auto_level_enabled(self) -> bool:
                return self._auto_level_enabled

            def manual_level(self) -> int:
                return self._manual_level

            def available_level_count(self) -> int:
                return self._available_level_count

        renderer = _RendererStub()
        window_like = SimpleNamespace(
            _semantic_volume=None,
            _semantic_worker=None,
            _instance_volume=None,
            _instance_worker=None,
            _segmentation_editor=None,
            _pending_render_view_ids=set(),
            _render_flush_scheduled=False,
            _pending_annotation_peer_view_ids=set(),
            _annotation_dirty_views=set(),
            _annotation_peer_flush_scheduled=False,
            _bbox_drag_active=False,
            _bbox_drag_source_view_id=None,
            _bbox_pending_peer_view_ids=set(),
            _bbox_peer_flush_scheduled=False,
            _bbox_drag_staged_history_updates={},
            _annotation_modification_active=False,
            _annotation_modification_view_id=None,
            _annotation_labels_dirty=False,
            _deferred_hover_readout=False,
            _deferred_picked_readout=False,
            _global_history=SimpleNamespace(clear=lambda: history_calls.append("clear")),
            _clear_picker_selection=lambda: picker_calls.append("clear"),
            renderer=renderer,
            _raw_volume=None,
            _bbox_manager=object(),
            _on_bounding_boxes_changed=lambda _change: None,
            _sync_bounding_boxes_ui=lambda: sync_bbox_calls.append("sync"),
            bottom_panel=SimpleNamespace(
                set_contrast_range=lambda value: contrast_calls.append(("range", value)),
                set_contrast_window=lambda value: contrast_calls.append(("window", value)),
                set_level_mode=lambda **kwargs: level_mode_calls.append(dict(kwargs)),
                set_cursor_range=lambda shape: cursor_calls.append(tuple(shape)),
                set_pyramid_levels=lambda *_args, **_kwargs: None,
                set_active_levels=lambda **_kwargs: None,
            ),
            sync_manager=SimpleNamespace(
                set_volume_info=lambda info: info_calls.append(info),
            ),
            state=SimpleNamespace(volume_loaded=False, annotation_mode_enabled=False),
            _ensure_editable_segmentation_for_annotation=lambda: None,
            _refresh_annotation_ui_state=lambda: refresh_calls.append("refresh"),
        )
        window_like._sync_contrast_controls_from_renderer = lambda: MainWindow._sync_contrast_controls_from_renderer(
            window_like
        )

        first_volume = SimpleNamespace(
            info=SimpleNamespace(shape=(10, 11, 12)),
            loader=SimpleNamespace(path="/tmp/first.raw"),
            data_range=(0.0, 50.0),
        )
        second_volume = SimpleNamespace(
            info=SimpleNamespace(shape=(4, 5, 6)),
            loader=SimpleNamespace(path="/tmp/second.raw"),
            data_range=(100.0, 200.0),
        )

        self.assertTrue(MainWindow.set_volume(window_like, first_volume, levels=None))
        renderer.set_window(10.0, 20.0)
        contrast_calls.clear()
        level_mode_calls.clear()

        self.assertTrue(MainWindow.set_volume(window_like, second_volume, levels=None))

        self.assertEqual(renderer.get_data_range(), (100.0, 200.0))
        self.assertEqual(renderer.get_window_range(), (100.0, 200.0))
        self.assertEqual(
            contrast_calls,
            [
                ("range", (100.0, 200.0)),
                ("window", (100.0, 200.0)),
            ],
        )
        self.assertEqual(
            level_mode_calls,
            [
                {
                    "auto_enabled": True,
                    "manual_level": 0,
                    "max_level": 0,
                }
            ],
        )
        self.assertEqual(len(renderer.attach_calls), 2)
        self.assertEqual(sync_bbox_calls, ["sync", "sync"])
        self.assertEqual(cursor_calls, [(10, 11, 12), (4, 5, 6)])
        self.assertEqual(len(info_calls), 2)
        self.assertEqual(history_calls, ["clear", "clear"])
        self.assertEqual(picker_calls, ["clear", "clear"])
        self.assertEqual(refresh_calls, ["refresh", "refresh"])
        self.assertIs(window_like._raw_volume, second_volume)
        self.assertTrue(window_like.state.volume_loaded)

    def test_set_volume_failure_keeps_existing_window_state(self) -> None:
        attach_calls: list[tuple[object, object]] = []
        detach_calls: list[str] = []
        history_calls: list[str] = []
        picker_calls: list[str] = []
        sync_bbox_calls: list[str] = []
        sync_contrast_calls: list[str] = []
        cursor_calls: list[tuple[int, int, int]] = []
        info_calls: list[object] = []
        refresh_calls: list[str] = []
        ensure_calls: list[str] = []

        def _attach_raise(volume: object, *, levels: object = None) -> None:
            attach_calls.append((volume, levels))
            raise ValueError("raw volume rejected")

        renderer = SimpleNamespace(
            attach_volume=_attach_raise,
            detach_segmentation=lambda: detach_calls.append("detach"),
        )
        previous_bbox_manager = object()
        window_like = SimpleNamespace(
            _semantic_volume="semantic",
            _semantic_worker="semantic_worker",
            _instance_volume="instance",
            _instance_worker="instance_worker",
            _segmentation_editor="editor",
            _pending_render_view_ids={"axial"},
            _render_flush_scheduled=True,
            _pending_annotation_peer_view_ids={"coronal"},
            _annotation_dirty_views={"sagittal"},
            _annotation_peer_flush_scheduled=True,
            _bbox_drag_active=True,
            _bbox_drag_source_view_id="axial",
            _bbox_pending_peer_view_ids={"coronal"},
            _bbox_peer_flush_scheduled=True,
            _bbox_drag_staged_history_updates={"bbox_0001": object()},
            _annotation_modification_active=True,
            _annotation_modification_view_id="axial",
            _annotation_labels_dirty=True,
            _deferred_hover_readout=True,
            _deferred_picked_readout=True,
            _global_history=SimpleNamespace(clear=lambda: history_calls.append("clear")),
            _clear_picker_selection=lambda: picker_calls.append("clear"),
            renderer=renderer,
            _raw_volume="existing_raw",
            _bbox_manager=previous_bbox_manager,
            _on_bounding_boxes_changed=lambda _change: None,
            _sync_bounding_boxes_ui=lambda: sync_bbox_calls.append("sync"),
            _sync_contrast_controls_from_renderer=lambda: sync_contrast_calls.append("sync"),
            bottom_panel=SimpleNamespace(
                set_cursor_range=lambda shape: cursor_calls.append(tuple(shape)),
                set_pyramid_levels=lambda *_args, **_kwargs: None,
                set_active_levels=lambda **_kwargs: None,
            ),
            sync_manager=SimpleNamespace(
                set_volume_info=lambda info: info_calls.append(info),
            ),
            state=SimpleNamespace(volume_loaded=True, annotation_mode_enabled=False),
            _ensure_editable_segmentation_for_annotation=lambda: ensure_calls.append("ensure"),
            _refresh_annotation_ui_state=lambda: refresh_calls.append("refresh"),
        )
        next_volume = SimpleNamespace(
            info=SimpleNamespace(shape=(10, 11, 12)),
            loader=SimpleNamespace(path="/tmp/rejected.raw"),
        )

        with self.assertRaisesRegex(ValueError, "raw volume rejected"):
            MainWindow.set_volume(window_like, next_volume, levels=None)

        self.assertEqual(len(attach_calls), 1)
        self.assertEqual(attach_calls[0], (next_volume, None))
        self.assertEqual(detach_calls, [])
        self.assertEqual(history_calls, [])
        self.assertEqual(picker_calls, [])
        self.assertEqual(sync_bbox_calls, [])
        self.assertEqual(sync_contrast_calls, [])
        self.assertEqual(cursor_calls, [])
        self.assertEqual(info_calls, [])
        self.assertEqual(refresh_calls, [])
        self.assertEqual(ensure_calls, [])
        self.assertEqual(window_like._raw_volume, "existing_raw")
        self.assertIs(window_like._bbox_manager, previous_bbox_manager)
        self.assertTrue(window_like.state.volume_loaded)


if __name__ == "__main__":
    unittest.main()
