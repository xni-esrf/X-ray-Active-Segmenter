from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .sync_manager import SyncManager


@dataclass
class PointerDelta:
    dx: float
    dy: float


class InputHandlers:
    def __init__(self, sync_manager: SyncManager) -> None:
        self.sync_manager = sync_manager

    def on_scroll(self, axis: int, delta: int) -> None:
        current = self.sync_manager.state.slice_indices[axis]
        self.sync_manager.set_slice_index(axis, current + delta)

    def on_drag_pan(self, *args) -> None:
        # Backward-compatible signatures:
        # - on_drag_pan(delta)
        # - on_drag_pan(view_id, delta)
        if len(args) == 1:
            delta = args[0]
            if not isinstance(delta, PointerDelta):
                raise TypeError("on_drag_pan(delta) expects a PointerDelta argument")
            pan_x, pan_y = self.sync_manager.state.pan
            self.sync_manager.set_pan((pan_x + delta.dx, pan_y + delta.dy))
            return
        if len(args) == 2:
            view_id, delta = args
            if not isinstance(delta, PointerDelta):
                raise TypeError(
                    "on_drag_pan(view_id, delta) expects delta to be PointerDelta"
                )
            current_x, current_y = self.sync_manager.pan_for_view(str(view_id))
            self.sync_manager.set_pan_for_view(
                str(view_id),
                (current_x + delta.dx, current_y + delta.dy),
            )
            return
        raise TypeError(
            "on_drag_pan accepts either (delta) or (view_id, delta)"
        )

    def on_drag_cursor(self, indices: Tuple[int, int, int]) -> None:
        self.sync_manager.set_cursor_indices(indices)

    def on_hover_cursor(self, indices: Optional[Tuple[int, int, int]]) -> None:
        self.sync_manager.set_hover_indices(indices)

    def on_zoom(self, zoom: float) -> None:
        self.sync_manager.set_zoom(zoom)

    def jump_to(self, axis: int, index: int) -> None:
        self.sync_manager.set_slice_index(axis, index)
