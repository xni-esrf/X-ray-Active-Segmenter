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

    def on_drag_pan(self, delta: PointerDelta) -> None:
        pan_x, pan_y = self.sync_manager.state.pan
        self.sync_manager.set_pan((pan_x + delta.dx, pan_y + delta.dy))

    def on_drag_cursor(self, indices: Tuple[int, int, int]) -> None:
        self.sync_manager.set_cursor_indices(indices)

    def on_hover_cursor(self, indices: Optional[Tuple[int, int, int]]) -> None:
        self.sync_manager.set_hover_indices(indices)

    def on_zoom(self, zoom: float) -> None:
        self.sync_manager.set_zoom(zoom)

    def jump_to(self, axis: int, index: int) -> None:
        self.sync_manager.set_slice_index(axis, index)
