from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from ..io.loader import VolumeInfo


@dataclass
class SyncState:
    slice_indices: Tuple[int, int, int] = (0, 0, 0)
    hover_indices: Optional[Tuple[int, int, int]] = None
    zoom: float = 1.0
    pan: Tuple[float, float] = (0.0, 0.0)


class SyncManager:
    def __init__(self) -> None:
        self.state = SyncState()
        self._volume_info: VolumeInfo | None = None
        self._listeners: List[Callable[[], None]] = []

    def set_volume_info(self, info: VolumeInfo) -> None:
        self._volume_info = info
        center = tuple(max(0, (dim - 1) // 2) for dim in info.shape)
        self.state = SyncState(
            slice_indices=center,
            hover_indices=None,
            zoom=self.state.zoom,
            pan=self.state.pan,
        )
        self._notify()

    def set_slice_index(self, axis: int, index: int) -> None:
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2")
        max_index = None
        if self._volume_info is not None:
            max_index = self._volume_info.shape[axis] - 1
        if max_index is not None:
            index = max(0, min(index, max_index))
        indices = list(self.state.slice_indices)
        indices[axis] = index
        self.state = SyncState(
            slice_indices=tuple(indices),
            hover_indices=self.state.hover_indices,
            zoom=self.state.zoom,
            pan=self.state.pan,
        )
        self._notify()

    def set_cursor_indices(self, indices: Tuple[int, int, int]) -> None:
        if len(indices) != 3:
            raise ValueError("cursor indices must have 3 values (z, y, x)")
        clamped = list(indices)
        if self._volume_info is not None:
            for axis in (0, 1, 2):
                max_index = self._volume_info.shape[axis] - 1
                clamped[axis] = max(0, min(clamped[axis], max_index))
        self.state = SyncState(
            slice_indices=tuple(clamped),
            hover_indices=self.state.hover_indices,
            zoom=self.state.zoom,
            pan=self.state.pan,
        )
        self._notify()

    def set_hover_indices(self, indices: Optional[Tuple[int, int, int]]) -> None:
        if indices is None:
            normalized = None
        else:
            if len(indices) != 3:
                raise ValueError("hover indices must have 3 values (z, y, x)")
            normalized = [int(indices[0]), int(indices[1]), int(indices[2])]
            if self._volume_info is not None:
                for axis in (0, 1, 2):
                    max_index = self._volume_info.shape[axis] - 1
                    normalized[axis] = max(0, min(normalized[axis], max_index))
            normalized = tuple(normalized)
        if normalized == self.state.hover_indices:
            return
        self.state = SyncState(
            slice_indices=self.state.slice_indices,
            hover_indices=normalized,
            zoom=self.state.zoom,
            pan=self.state.pan,
        )
        self._notify()

    def set_zoom(self, zoom: float) -> None:
        zoom = max(0.1, min(1.0, zoom))
        self.state = SyncState(
            slice_indices=self.state.slice_indices,
            hover_indices=self.state.hover_indices,
            zoom=zoom,
            pan=self.state.pan,
        )
        self._notify()

    def set_pan(self, pan: Tuple[float, float]) -> None:
        self.state = SyncState(
            slice_indices=self.state.slice_indices,
            hover_indices=self.state.hover_indices,
            zoom=self.state.zoom,
            pan=pan,
        )
        self._notify()

    def on_state_changed(self, callback: Callable[[], None]) -> None:
        self._listeners.append(callback)

    def _notify(self) -> None:
        for callback in list(self._listeners):
            callback()
