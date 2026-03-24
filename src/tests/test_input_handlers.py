from __future__ import annotations

import unittest

from src.events import InputHandlers, PointerDelta, SyncManager


class InputHandlersTests(unittest.TestCase):
    def test_on_drag_pan_updates_only_target_view(self) -> None:
        sync_manager = SyncManager()
        sync_manager.set_pan((1.0, 2.0))
        handlers = InputHandlers(sync_manager)

        handlers.on_drag_pan("coronal", PointerDelta(dx=3.0, dy=-1.5))

        self.assertEqual(sync_manager.pan_for_view("axial"), (1.0, 2.0))
        self.assertEqual(sync_manager.pan_for_view("coronal"), (4.0, 0.5))
        self.assertEqual(sync_manager.pan_for_view("sagittal"), (1.0, 2.0))

    def test_on_drag_pan_rejects_unknown_view(self) -> None:
        sync_manager = SyncManager()
        handlers = InputHandlers(sync_manager)

        with self.assertRaises(ValueError):
            handlers.on_drag_pan("unknown", PointerDelta(dx=1.0, dy=1.0))

    def test_scroll_and_zoom_do_not_mutate_per_view_pan(self) -> None:
        sync_manager = SyncManager()
        sync_manager.set_pan_for_view("axial", (4.0, 5.0))
        sync_manager.set_pan_for_view("coronal", (1.0, -2.0))
        sync_manager.set_pan_for_view("sagittal", (7.5, 3.25))
        handlers = InputHandlers(sync_manager)
        before = dict(sync_manager.state.pan_by_view)

        handlers.on_scroll(axis=0, delta=2)
        handlers.on_scroll(axis=1, delta=-1)
        handlers.on_zoom(0.6)

        self.assertEqual(sync_manager.state.pan_by_view, before)


if __name__ == "__main__":
    unittest.main()
