from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Callable, Iterable, Literal, Optional, Tuple, cast

import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt, QTimer
from PySide6.QtWidgets import QApplication, QLineEdit, QWidget, QVBoxLayout

from ..bbox import (
    BoundingBox,
    BoundingBoxHandleHit,
    FaceId,
    face_updates_for_handle_drag,
    hit_test_projected_box_handles,
    project_boxes_to_slice,
    translation_delta_for_edge_drag,
)
from ..events import InputHandlers, PointerDelta
from ..render import RenderResult, Renderer, ViewId
from ..render.gl_backend import GLBackend
from ..utils import maybe_profile


@dataclass
class ViewState:
    axis: int
    slice_index: int
    zoom: float = 1.0
    pan: Tuple[float, float] = (0.0, 0.0)


Bounds3D = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]


@dataclass(frozen=True)
class AnnotationPaintOutcome:
    accepted: bool
    changed_bounds: Optional[Bounds3D] = None


logger = logging.getLogger(__name__)
AnnotationTool = Literal["brush", "eraser", "flood_filler"]


class OrthogonalView(QWidget):
    def __init__(
        self,
        view_id: ViewId,
        axis: int,
        renderer: Renderer,
        input_handlers: InputHandlers,
        *,
        annotation_tool_getter: Optional[Callable[[], AnnotationTool]] = None,
        bounding_box_mode_enabled_getter: Optional[Callable[[], bool]] = None,
        on_paint_voxel: Optional[
            Callable[[ViewId, Tuple[int, int, int]], AnnotationPaintOutcome]
        ] = None,
        on_paint_stroke: Optional[
            Callable[[ViewId, Tuple[int, int, int], Tuple[int, int, int]], AnnotationPaintOutcome]
        ] = None,
        on_pick_voxel: Optional[Callable[[ViewId, Tuple[int, int, int]], None]] = None,
        on_annotation_finished: Optional[Callable[[ViewId], None]] = None,
        bounding_boxes_getter: Optional[Callable[[], Iterable[BoundingBox]]] = None,
        selected_bounding_box_id_getter: Optional[Callable[[], Optional[str]]] = None,
        on_bounding_box_select: Optional[Callable[[Optional[str]], None]] = None,
        on_bounding_box_move_face: Optional[Callable[[str, FaceId, int], None]] = None,
        on_bounding_box_translate: Optional[Callable[[str, int, int, int], None]] = None,
        on_bounding_box_drag_started: Optional[Callable[[ViewId], None]] = None,
        on_bounding_box_drag_finished: Optional[Callable[[ViewId], None]] = None,
        on_bounding_box_delete_requested: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__()
        self.view_id = view_id
        self.renderer = renderer
        self.input_handlers = input_handlers
        self._annotation_tool_getter = annotation_tool_getter
        self._bounding_box_mode_enabled_getter = bounding_box_mode_enabled_getter
        self._on_paint_voxel = on_paint_voxel
        self._on_paint_stroke = on_paint_stroke
        self._on_pick_voxel = on_pick_voxel
        self._on_annotation_finished = on_annotation_finished
        self._bounding_boxes_getter = bounding_boxes_getter
        self._selected_bounding_box_id_getter = selected_bounding_box_id_getter
        self._on_bounding_box_select = on_bounding_box_select
        self._on_bounding_box_move_face = on_bounding_box_move_face
        self._on_bounding_box_translate = on_bounding_box_translate
        self._on_bounding_box_drag_started = on_bounding_box_drag_started
        self._on_bounding_box_drag_finished = on_bounding_box_drag_finished
        self._on_bounding_box_delete_requested = on_bounding_box_delete_requested
        self.state = ViewState(axis=axis, slice_index=0)
        self._latest: Optional[RenderResult] = None
        self._recenter_on_next_render = False
        self._drag_active = False
        self._pan_active = False
        self._pan_last: Optional[QPointF] = None
        self._annotation_drag_active = False
        self._annotation_last_coord: Optional[Tuple[int, int, int]] = None
        self._annotation_pending_coord: Optional[Tuple[int, int, int]] = None
        self._annotation_tick_base_interval_ms = 16
        self._annotation_tick_interval_ms = self._annotation_tick_base_interval_ms
        self._annotation_tick_max_interval_ms = 120
        self._annotation_backpressure_active = False
        self._bbox_hover_hit: Optional[BoundingBoxHandleHit] = None
        self._bbox_drag_hit: Optional[BoundingBoxHandleHit] = None
        self._bbox_drag_last_updates: Optional[Tuple[Tuple[FaceId, int], ...]] = None
        self._bbox_drag_pending_updates: Optional[Tuple[Tuple[FaceId, int], ...]] = None
        self._bbox_drag_last_boundaries: Optional[Tuple[int, int]] = None
        self._bbox_drag_pending_boundaries: Optional[Tuple[int, int]] = None
        self._bbox_drag_tick_interval_ms = 16
        self._picked_indices: Optional[Tuple[int, int, int]] = None
        self._picker_marker_active = False
        self._annotation_backpressure_timer = QTimer(self)
        self._annotation_backpressure_timer.setSingleShot(True)
        self._annotation_backpressure_timer.timeout.connect(
            self._release_annotation_backpressure
        )
        self._overlay_flush_scheduled = False
        self._pending_overlay_result: Optional[RenderResult] = None
        self._annotation_tick_timer = QTimer(self)
        self._annotation_tick_timer.setSingleShot(True)
        self._annotation_tick_timer.timeout.connect(self._flush_annotation_drag_tick)
        self._bbox_drag_tick_timer = QTimer(self)
        self._bbox_drag_tick_timer.setSingleShot(True)
        self._bbox_drag_tick_timer.timeout.connect(self._flush_bounding_box_drag_tick)
        self._gl_backend = GLBackend()
        self._gl_backend.initialize()

        layout = QVBoxLayout()
        self._canvas_widget = self._gl_backend.widget()
        self.setMouseTracking(True)
        self._canvas_widget.setMouseTracking(True)
        self._canvas_widget.installEventFilter(self)
        layout.addWidget(self._canvas_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.renderer.set_output_handler(self.view_id, self._handle_render_result)

    def set_slice_index(self, index: int) -> None:
        self.state.slice_index = index

    def set_zoom(self, zoom: float) -> None:
        self.state.zoom = zoom
        if self._latest is None:
            self._recenter_on_next_render = False
            return
        target_level = self.renderer.target_level_for_view(self.state.axis, self.state.zoom)
        self._recenter_on_next_render = self._latest.level != target_level
        self._apply_zoom(self._latest.image, recenter=True)

    def set_pan(self, pan: Tuple[float, float]) -> None:
        self.state.pan = pan
        if self._latest is not None:
            self._apply_pan(self._latest.image)

    def render(self) -> RenderResult:
        result = self.renderer.render_slice(
            view_id=self.view_id,
            axis=self.state.axis,
            slice_index=self.state.slice_index,
            zoom=self.state.zoom,
            pan=self.state.pan,
        )
        return result

    def latest_image(self) -> Optional[np.ndarray]:
        if self._latest is None:
            return None
        return self._latest.image

    def refresh_overlay(self) -> None:
        if self._latest is None:
            return
        self._update_crosshair(self._latest.image)
        self._update_picker_marker(self._latest.image)
        self._update_bounding_boxes_overlay(self._latest)

    def set_picker_selection(
        self,
        indices: Optional[Tuple[int, int, int]],
        *,
        active: bool,
    ) -> None:
        self._picked_indices = None if indices is None else (int(indices[0]), int(indices[1]), int(indices[2]))
        self._picker_marker_active = bool(active)
        if self._latest is None:
            self._gl_backend.set_selection_marker(0.0, 0.0, visible=False)
            return
        self._update_picker_marker(self._latest.image)

    def _handle_render_result(self, result: RenderResult) -> None:
        with maybe_profile(
            "orthogonal_view_handle_render_result",
            logger=logger,
            details=f"view={self.view_id} axis={result.axis} level={result.level}",
        ):
            previous = self._latest
            overlay_only_refresh = self._is_overlay_only_refresh(previous, result)
            self._latest = result
            if overlay_only_refresh:
                self._pending_overlay_result = result
                if not self._overlay_flush_scheduled:
                    self._overlay_flush_scheduled = True
                    QTimer.singleShot(0, self._flush_overlay_update)
            else:
                self._pending_overlay_result = None
                self._gl_backend.upload_texture(
                    result.image,
                    segmentation=result.segmentation,
                    segmentation_range=result.segmentation_range,
                    segmentation_labels=result.segmentation_labels,
                    segmentation_roi=result.segmentation_roi,
                )
            recenter = self._recenter_on_next_render and not overlay_only_refresh
            self._recenter_on_next_render = False
            self._update_crosshair(result.image)
            self._update_picker_marker(result.image)
            self._update_bounding_boxes_overlay(result)
            if not overlay_only_refresh:
                self._apply_pan(result.image)
                self._apply_zoom(result.image, recenter=recenter)

    def _flush_overlay_update(self) -> None:
        self._overlay_flush_scheduled = False
        result = self._pending_overlay_result
        self._pending_overlay_result = None
        if result is None:
            return
        self._gl_backend.update_segmentation_overlay(
            result.segmentation,
            result.segmentation_range,
            result.segmentation_labels,
            segmentation_roi=result.segmentation_roi,
            segmentation_patch=result.segmentation_patch,
        )
        self._update_bounding_boxes_overlay(result)

    def _is_overlay_only_refresh(
        self,
        previous: Optional[RenderResult],
        result: RenderResult,
    ) -> bool:
        if previous is None:
            return False
        if previous.axis != result.axis:
            return False
        if previous.slice_index != result.slice_index:
            return False
        if previous.level != result.level:
            return False
        if previous.level_scale != result.level_scale:
            return False
        return previous.image is result.image

    def _apply_pan(self, image: np.ndarray) -> None:
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            return
        pan_x, pan_y = self.state.pan
        self._gl_backend.set_pan(pan_x, pan_y, width, height)

    def _apply_zoom(self, image: np.ndarray, *, recenter: bool) -> None:
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            return
        scale = self._current_level_scale()
        center = self._cursor_xy(scale) if recenter else None
        self._gl_backend.set_zoom(self.state.zoom, width, height, center=center)

    def _update_crosshair(self, image: np.ndarray) -> None:
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            return
        x, y = self._cursor_xy(self._current_level_scale())
        self._gl_backend.set_crosshair(x, y, width, height)

    def _update_picker_marker(self, image: np.ndarray) -> None:
        marker = self._picker_marker_xy(image)
        if marker is None:
            self._gl_backend.set_selection_marker(0.0, 0.0, visible=False)
            return
        self._gl_backend.set_selection_marker(marker[0], marker[1], visible=True)

    def _update_bounding_boxes_overlay(self, result: RenderResult) -> None:
        if self._bounding_boxes_getter is None:
            self._gl_backend.clear_bounding_boxes_overlay()
            return
        image = result.image
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            self._gl_backend.clear_bounding_boxes_overlay()
            return
        try:
            boxes = tuple(self._bounding_boxes_getter())
        except Exception:
            logger.exception("Failed to retrieve bounding boxes for overlay")
            self._gl_backend.clear_bounding_boxes_overlay()
            return

        selected_id: Optional[str] = None
        if self._selected_bounding_box_id_getter is not None:
            try:
                selected_id = self._selected_bounding_box_id_getter()
            except Exception:
                logger.exception("Failed to retrieve selected bounding-box id")
                selected_id = None

        projected = project_boxes_to_slice(
            boxes,
            axis=self.state.axis,
            slice_index=result.slice_index,
            level_scale=max(1, int(result.level_scale)),
            image_shape=(int(height), int(width)),
        )
        active_hit = self._bbox_drag_hit
        hover_hit: Optional[BoundingBoxHandleHit]
        if active_hit is not None or self._bounding_box_interaction_enabled():
            hover_hit = self._bbox_hover_hit
        else:
            hover_hit = None
        self._gl_backend.update_bounding_boxes_overlay(
            projected,
            selected_id=selected_id,
            hover_hit=hover_hit,
            active_hit=active_hit,
        )

    def _picker_marker_xy(self, image: np.ndarray) -> Optional[Tuple[float, float]]:
        if not self._picker_marker_active or self._picked_indices is None or self._latest is None:
            return None

        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            return None

        scale = self._current_level_scale()
        z, y, x = self._picked_indices
        level_slice_index = int(self._latest.slice_index) // scale
        axis = self.state.axis

        if axis == 0:
            if z // scale != level_slice_index:
                return None
            row = int(y // scale)
            col = int(x // scale)
        elif axis == 1:
            if y // scale != level_slice_index:
                return None
            row = int(z // scale)
            col = int(x // scale)
        else:
            if x // scale != level_slice_index:
                return None
            row = int(z // scale)
            col = int(y // scale)

        if row < 0 or col < 0 or row >= height or col >= width:
            return None
        return (float(col), float(row))

    def _map_canvas_to_image_coords(
        self, position: QPointF, width: int, height: int
    ) -> Optional[Tuple[float, float]]:
        mapped = self._gl_backend.map_canvas_to_image(position.x(), position.y())
        if mapped is not None:
            return mapped
        widget_size = self._canvas_widget.size()
        widget_w = widget_size.width()
        widget_h = widget_size.height()
        if widget_w <= 0 or widget_h <= 0:
            return None
        col = position.x() / widget_w * width
        row = position.y() / widget_h * height
        return (col, row)

    def _update_cursor_from_position(self, position: QPointF) -> None:
        indices = self._indices_from_position(position)
        if indices is None:
            return
        self.input_handlers.on_drag_cursor(indices)

    def _update_hover_from_position(self, position: QPointF) -> None:
        indices = self._indices_from_position(position)
        self.input_handlers.on_hover_cursor(indices)

    def _clear_hover(self) -> None:
        self.input_handlers.on_hover_cursor(None)

    def _indices_from_position(self, position: QPointF) -> Optional[Tuple[int, int, int]]:
        if self._latest is None:
            return None
        image = self._latest.image
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            return None
        mapped = self._map_canvas_to_image_coords(position, width, height)
        if mapped is None:
            return None
        col = int(np.clip(mapped[0], 0, width - 1))
        row = int(np.clip(mapped[1], 0, height - 1))
        scale = self._current_level_scale()
        col_full = int(round(col * scale))
        row_full = int(round(row * scale))
        current = self.input_handlers.sync_manager.state.slice_indices
        axis = self.state.axis
        if axis == 0:
            indices = (current[0], row_full, col_full)
        elif axis == 1:
            indices = (row_full, current[1], col_full)
        else:
            indices = (row_full, col_full, current[2])
        return indices

    def _current_level_scale(self) -> int:
        if self._latest is None:
            return 1
        return max(1, int(self._latest.level_scale))

    def _cursor_xy(self, scale: int) -> Tuple[float, float]:
        cursor = self.input_handlers.sync_manager.state.slice_indices
        axis = self.state.axis
        scale = float(max(1, scale))
        if axis == 0:
            return (cursor[2] / scale, cursor[1] / scale)
        if axis == 1:
            return (cursor[2] / scale, cursor[0] / scale)
        return (cursor[1] / scale, cursor[0] / scale)

    def _handle_canvas_press(self, position: QPointF) -> None:
        self._drag_active = True
        self._update_cursor_from_position(position)

    def _handle_canvas_move(self, position: QPointF) -> None:
        if not self._drag_active:
            return
        self._update_cursor_from_position(position)

    def _handle_canvas_release(self) -> None:
        self._drag_active = False

    def _handle_annotation_press(self, position: QPointF) -> None:
        self._annotation_drag_active = False
        self._annotation_last_coord = None
        self._annotation_pending_coord = None
        self._annotation_tick_interval_ms = self._annotation_tick_base_interval_ms
        self._annotation_backpressure_active = False
        self._annotation_backpressure_timer.stop()
        self._annotation_tick_timer.stop()
        indices = self._indices_from_position(position)
        if indices is None:
            return
        if self._on_paint_voxel is None:
            return
        outcome = self._on_paint_voxel(self.view_id, indices)
        if not bool(outcome.accepted):
            return
        # Keep the active view responsive by rendering it immediately.
        self._render_annotation_update(changed_bounds=outcome.changed_bounds)
        self._annotation_drag_active = True
        self._annotation_last_coord = indices

    def _annotation_tool(self) -> AnnotationTool:
        if self._annotation_tool_getter is None:
            return "brush"
        tool = str(self._annotation_tool_getter()).strip().lower()
        if tool not in ("brush", "eraser", "flood_filler"):
            return "brush"
        return cast(AnnotationTool, tool)

    def _bounding_box_interaction_enabled(self) -> bool:
        if self._bounding_box_mode_enabled_getter is None:
            return False
        return bool(self._bounding_box_mode_enabled_getter())

    def _refresh_bounding_box_hover_for_position(self, position: QPointF) -> None:
        if self._bbox_drag_hit is not None or self._bounding_box_interaction_enabled():
            self._update_bounding_box_hover_from_position(position)
            return
        self._clear_bounding_box_hover()

    def _handle_picker_press(self, position: QPointF) -> None:
        if self._annotation_drag_active:
            self._handle_annotation_release()
        if self._on_pick_voxel is None:
            return
        indices = self._indices_from_position(position)
        if indices is None:
            return
        self._on_pick_voxel(self.view_id, indices)

    def _handle_bounding_box_press(self, position: QPointF) -> bool:
        if (
            self._on_bounding_box_select is None
            and self._on_bounding_box_move_face is None
            and self._on_bounding_box_translate is None
        ):
            return False
        hit = self._bounding_box_hit_at_position(position)
        if hit is None:
            return False
        self._bbox_drag_hit = hit
        self._bbox_drag_last_updates = None
        self._bbox_drag_pending_updates = None
        self._bbox_drag_last_boundaries = self._boundary_coordinates_from_position(position)
        self._bbox_drag_pending_boundaries = None
        self._bbox_drag_tick_timer.stop()
        self._set_bounding_box_hover_hit(hit)
        if self._on_bounding_box_select is not None:
            try:
                self._on_bounding_box_select(hit.box_id)
            except Exception:
                logger.exception("Failed to select bounding box from interactive reshape")
        if self._on_bounding_box_drag_started is not None:
            try:
                self._on_bounding_box_drag_started(self.view_id)
            except Exception:
                logger.exception("Failed to notify bounding-box drag start")
        if self._latest is not None:
            self._update_bounding_boxes_overlay(self._latest)
        return True

    def _handle_bounding_box_move(self, position: QPointF) -> bool:
        hit = self._bbox_drag_hit
        if hit is None:
            return False
        self._set_bounding_box_hover_hit(hit)
        boundaries = self._boundary_coordinates_from_position(position)
        if boundaries is None:
            return True

        if hit.kind == "corner":
            if self._on_bounding_box_move_face is None:
                return True
            try:
                face_updates = face_updates_for_handle_drag(
                    hit,
                    axis=self.state.axis,
                    row_boundary=boundaries[0],
                    col_boundary=boundaries[1],
                )
            except Exception:
                logger.exception("Failed to compute bounding-box corner drag updates")
                return True
            if (
                face_updates == self._bbox_drag_last_updates
                and self._bbox_drag_pending_updates is None
            ):
                return True
            self._bbox_drag_pending_updates = face_updates
            self._bbox_drag_pending_boundaries = None
            self._schedule_bounding_box_drag_tick()
            return True

        if hit.kind == "edge":
            if self._on_bounding_box_translate is None:
                return True
            if self._bbox_drag_last_boundaries is None:
                self._bbox_drag_last_boundaries = boundaries
                return True
            if (
                boundaries == self._bbox_drag_last_boundaries
                and self._bbox_drag_pending_boundaries is None
            ):
                return True
            self._bbox_drag_pending_boundaries = boundaries
            self._bbox_drag_pending_updates = None
            self._schedule_bounding_box_drag_tick()
            return True

        logger.warning("Ignoring unknown bounding-box handle kind: %s", hit.kind)
        self._schedule_bounding_box_drag_tick()
        return True

    def _handle_bounding_box_release(self) -> bool:
        if self._bbox_drag_hit is None:
            return False
        if self._bbox_drag_pending_updates is not None or self._bbox_drag_pending_boundaries is not None:
            self._flush_bounding_box_drag_tick()
        self._bbox_drag_tick_timer.stop()
        if self._on_bounding_box_drag_finished is not None:
            try:
                self._on_bounding_box_drag_finished(self.view_id)
            except Exception:
                logger.exception("Failed to notify bounding-box drag finish")
        self._bbox_drag_hit = None
        self._bbox_drag_last_updates = None
        self._bbox_drag_pending_updates = None
        self._bbox_drag_last_boundaries = None
        self._bbox_drag_pending_boundaries = None
        if self._latest is not None:
            self._update_bounding_boxes_overlay(self._latest)
        return True

    def _schedule_bounding_box_drag_tick(self) -> None:
        if self._bbox_drag_tick_timer.isActive():
            return
        self._bbox_drag_tick_timer.start(max(1, int(self._bbox_drag_tick_interval_ms)))

    def _flush_bounding_box_drag_tick(self) -> None:
        hit = self._bbox_drag_hit
        if hit is None:
            self._bbox_drag_pending_updates = None
            self._bbox_drag_pending_boundaries = None
            return

        if hit.kind == "corner":
            if self._on_bounding_box_move_face is None:
                self._bbox_drag_pending_updates = None
                return
            face_updates = self._bbox_drag_pending_updates
            if face_updates is None:
                return
            self._bbox_drag_pending_updates = None
            if face_updates == self._bbox_drag_last_updates:
                return
            for face, boundary in face_updates:
                try:
                    self._on_bounding_box_move_face(hit.box_id, face, boundary)
                except Exception:
                    logger.exception("Failed to apply bounding-box corner resize update")
                    return
            self._bbox_drag_last_updates = face_updates
            if self._bbox_drag_pending_updates is not None and self._bbox_drag_hit is not None:
                self._schedule_bounding_box_drag_tick()
            return

        if hit.kind == "edge":
            if self._on_bounding_box_translate is None:
                self._bbox_drag_pending_boundaries = None
                return
            current_boundaries = self._bbox_drag_pending_boundaries
            previous_boundaries = self._bbox_drag_last_boundaries
            if current_boundaries is None or previous_boundaries is None:
                return
            self._bbox_drag_pending_boundaries = None
            row_delta = int(current_boundaries[0] - previous_boundaries[0])
            col_delta = int(current_boundaries[1] - previous_boundaries[1])
            if row_delta == 0 and col_delta == 0:
                self._bbox_drag_last_boundaries = current_boundaries
                return
            try:
                dz, dy, dx = translation_delta_for_edge_drag(
                    hit,
                    axis=self.state.axis,
                    row_delta=row_delta,
                    col_delta=col_delta,
                )
                self._on_bounding_box_translate(hit.box_id, dz, dy, dx)
            except Exception:
                logger.exception("Failed to apply bounding-box edge translation update")
                return
            self._bbox_drag_last_boundaries = current_boundaries
            if self._bbox_drag_pending_boundaries is not None and self._bbox_drag_hit is not None:
                self._schedule_bounding_box_drag_tick()
            return

        self._bbox_drag_pending_updates = None
        self._bbox_drag_pending_boundaries = None

    def _update_bounding_box_hover_from_position(self, position: QPointF) -> None:
        if self._bbox_drag_hit is not None:
            self._set_bounding_box_hover_hit(self._bbox_drag_hit)
            return
        hit = self._bounding_box_hit_at_position(position)
        self._set_bounding_box_hover_hit(hit)

    def _clear_bounding_box_hover(self) -> None:
        self._set_bounding_box_hover_hit(None)

    def _set_bounding_box_hover_hit(
        self,
        hit: Optional[BoundingBoxHandleHit],
    ) -> None:
        if self._bbox_hover_hit == hit:
            return
        self._bbox_hover_hit = hit
        if self._latest is not None:
            self._update_bounding_boxes_overlay(self._latest)

    def _bounding_box_hit_at_position(self, position: QPointF) -> Optional[BoundingBoxHandleHit]:
        if self._latest is None or self._bounding_boxes_getter is None:
            return None
        image = self._latest.image
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            return None
        mapped = self._map_canvas_to_image_coords(position, width, height)
        if mapped is None:
            return None
        try:
            boxes = tuple(self._bounding_boxes_getter())
        except Exception:
            logger.exception("Failed to retrieve bounding boxes for hit-testing")
            return None
        if not boxes:
            return None

        selected_id: Optional[str] = None
        if self._selected_bounding_box_id_getter is not None:
            try:
                selected_id = self._selected_bounding_box_id_getter()
            except Exception:
                logger.exception("Failed to retrieve selected bounding-box id")
                selected_id = None

        projected = project_boxes_to_slice(
            boxes,
            axis=self.state.axis,
            slice_index=self._latest.slice_index,
            level_scale=self._current_level_scale(),
            image_shape=(int(height), int(width)),
        )
        if not projected:
            return None
        return hit_test_projected_box_handles(
            projected,
            row=float(mapped[1]),
            col=float(mapped[0]),
            tolerance=self._bounding_box_hit_tolerance(position, width=width, height=height),
            selected_id=selected_id,
        )

    def _boundary_coordinates_from_position(self, position: QPointF) -> Optional[Tuple[int, int]]:
        if self._latest is None:
            return None
        image = self._latest.image
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            return None
        mapped = self._map_canvas_to_image_coords(position, width, height)
        if mapped is None:
            return None
        scale = self._current_level_scale()
        row = int(round(mapped[1] * scale))
        col = int(round(mapped[0] * scale))
        return (row, col)

    def _bounding_box_hit_tolerance(self, position: QPointF, *, width: int, height: int) -> float:
        base = self._map_canvas_to_image_coords(position, width, height)
        if base is None:
            return 2.5
        probe = self._map_canvas_to_image_coords(
            QPointF(position.x() + 6.0, position.y()),
            width,
            height,
        )
        if probe is None:
            return 2.5
        tolerance = abs(float(probe[0]) - float(base[0]))
        if tolerance <= 0.0:
            return 2.5
        return max(1.5, min(8.0, tolerance))

    def _handle_annotation_move(self, position: QPointF) -> None:
        if not self._annotation_drag_active or self._annotation_last_coord is None:
            return
        indices = self._indices_from_position(position)
        if indices is None or indices == self._annotation_last_coord:
            return
        self._annotation_pending_coord = indices
        self._schedule_annotation_drag_tick()

    def _handle_annotation_release(self) -> None:
        notify_finished = self._annotation_drag_active
        if self._annotation_drag_active and self._annotation_pending_coord is not None:
            self._flush_annotation_drag_tick()
        self._annotation_drag_active = False
        self._annotation_last_coord = None
        self._annotation_pending_coord = None
        self._annotation_backpressure_active = False
        self._annotation_backpressure_timer.stop()
        self._annotation_tick_timer.stop()
        if notify_finished and self._on_annotation_finished is not None:
            self._on_annotation_finished(self.view_id)

    def _schedule_annotation_drag_tick(self) -> None:
        if self._annotation_backpressure_active:
            return
        if self._annotation_tick_timer.isActive():
            return
        self._annotation_tick_timer.start(max(1, int(self._annotation_tick_interval_ms)))

    def _flush_annotation_drag_tick(self) -> None:
        tick_start = time.perf_counter()
        if not self._annotation_drag_active or self._annotation_last_coord is None:
            return
        pending = self._annotation_pending_coord
        if pending is None or pending == self._annotation_last_coord:
            elapsed_ms = (time.perf_counter() - tick_start) * 1000.0
            self._update_annotation_tick_interval(elapsed_ms)
            return
        self._annotation_pending_coord = None

        if self._on_paint_stroke is not None:
            outcome = self._on_paint_stroke(
                self.view_id,
                self._annotation_last_coord,
                pending,
            )
        elif self._on_paint_voxel is not None:
            outcome = self._on_paint_voxel(self.view_id, pending)
        else:
            outcome = AnnotationPaintOutcome(accepted=False)

        if outcome.accepted:
            self._annotation_last_coord = pending
            self._render_annotation_update(changed_bounds=outcome.changed_bounds)

        elapsed_ms = (time.perf_counter() - tick_start) * 1000.0
        slow_tick = self._update_annotation_tick_interval(elapsed_ms)
        if slow_tick:
            self._annotation_backpressure_active = True
            self._annotation_backpressure_timer.start(
                max(1, int(self._annotation_tick_interval_ms))
            )

        if (
            self._annotation_pending_coord is not None
            and self._annotation_pending_coord != self._annotation_last_coord
        ):
            if not slow_tick:
                self._schedule_annotation_drag_tick()

    def _render_annotation_update(self, *, changed_bounds: Optional[Bounds3D] = None) -> None:
        fast_result = self.renderer.refresh_segmentation_overlay(
            view_id=self.view_id,
            axis=self.state.axis,
            slice_index=self.state.slice_index,
            zoom=self.state.zoom,
            changed_bounds=changed_bounds,
        )
        if fast_result is None:
            self.render()

    def _update_annotation_tick_interval(self, elapsed_ms: float) -> bool:
        target = float(self._annotation_tick_base_interval_ms)
        current = float(self._annotation_tick_interval_ms)
        over_budget = elapsed_ms > target
        if over_budget:
            # Back off when a tick exceeds the target frame budget.
            next_interval = max(current + 1.0, elapsed_ms * 1.2)
            next_interval = min(float(self._annotation_tick_max_interval_ms), next_interval)
        else:
            # Recover toward the default cadence once workload drops.
            next_interval = max(target, current - 1.0)
        self._annotation_tick_interval_ms = int(round(next_interval))
        return over_budget

    def _release_annotation_backpressure(self) -> None:
        self._annotation_backpressure_active = False
        if not self._annotation_drag_active or self._annotation_last_coord is None:
            return
        if (
            self._annotation_pending_coord is None
            or self._annotation_pending_coord == self._annotation_last_coord
        ):
            return
        self._schedule_annotation_drag_tick()

    def _handle_pan_press(self, position: QPointF) -> None:
        self._pan_active = True
        self._pan_last = position

    def _handle_pan_move(self, position: QPointF) -> None:
        if not self._pan_active or self._pan_last is None:
            return
        if self._latest is None:
            return
        image = self._latest.image
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            return
        current = self._map_canvas_to_image_coords(position, width, height)
        previous = self._map_canvas_to_image_coords(self._pan_last, width, height)
        if current is None or previous is None:
            return
        delta_x = current[0] - previous[0]
        delta_y = current[1] - previous[1]
        self._pan_last = position
        self.input_handlers.on_drag_pan(PointerDelta(dx=delta_x, dy=delta_y))

    def _handle_pan_release(self) -> None:
        self._pan_active = False
        self._pan_last = None

    def eventFilter(self, obj, event) -> bool:  # type: ignore[override]
        if obj is self._canvas_widget:
            if event.type() == QEvent.KeyPress:
                if event.key() in (Qt.Key_Backspace, Qt.Key_Delete):
                    focus_widget = QApplication.focusWidget()
                    if isinstance(focus_widget, QLineEdit):
                        return False
                    if self._on_bounding_box_delete_requested is not None:
                        self._on_bounding_box_delete_requested()
                    # Block VisPy camera BACKSPACE reset on the focused canvas.
                    event.accept()
                    return True
                return False
            if event.type() == QEvent.Leave:
                self._clear_hover()
                self._clear_bounding_box_hover()
                return False
            if event.type() == QEvent.Wheel:
                if event.modifiers() & Qt.ControlModifier:
                    delta = 1 if event.angleDelta().y() > 0 else -1
                    self.input_handlers.on_scroll(self.state.axis, delta)
                    event.accept()
                    return True
                delta = 1 if event.angleDelta().y() > 0 else -1
                step = 0.1
                next_zoom = self.state.zoom - (delta * step)
                next_zoom = max(0.1, min(1.0, next_zoom))
                if next_zoom != self.state.zoom:
                    self.input_handlers.on_zoom(next_zoom)
                event.accept()
                return True
            if event.type() == QEvent.MouseButtonPress:
                self._update_hover_from_position(event.position())
                self._refresh_bounding_box_hover_for_position(event.position())
                if event.button() == Qt.LeftButton:
                    if self._bounding_box_interaction_enabled():
                        if self._handle_bounding_box_press(event.position()):
                            event.accept()
                            return True
                        self._handle_picker_press(event.position())
                        event.accept()
                        return True
                    if self._annotation_tool() == "flood_filler":
                        self._handle_picker_press(event.position())
                    else:
                        self._handle_annotation_press(event.position())
                    event.accept()
                    return True
                if event.button() == Qt.RightButton:
                    if event.modifiers() & Qt.ShiftModifier:
                        self._handle_canvas_press(event.position())
                    else:
                        self._handle_pan_press(event.position())
                    event.accept()
                    return True
            if event.type() == QEvent.MouseMove:
                self._update_hover_from_position(event.position())
                self._refresh_bounding_box_hover_for_position(event.position())
                if event.buttons() & Qt.LeftButton:
                    if (
                        self._bbox_drag_hit is not None or self._bounding_box_interaction_enabled()
                    ) and self._handle_bounding_box_move(event.position()):
                        event.accept()
                        return True
                    if self._annotation_tool() != "flood_filler":
                        self._handle_annotation_move(event.position())
                    event.accept()
                    return True
                if event.buttons() & Qt.RightButton:
                    if event.modifiers() & Qt.ShiftModifier:
                        self._handle_canvas_move(event.position())
                    else:
                        self._handle_pan_move(event.position())
                    event.accept()
                    return True
                return False
            if event.type() == QEvent.MouseButtonRelease:
                self._update_hover_from_position(event.position())
                self._refresh_bounding_box_hover_for_position(event.position())
                if event.button() == Qt.LeftButton:
                    if self._bbox_drag_hit is not None and self._handle_bounding_box_release():
                        event.accept()
                        return True
                    if self._annotation_drag_active or (
                        not self._bounding_box_interaction_enabled()
                        and self._annotation_tool() != "flood_filler"
                    ):
                        self._handle_annotation_release()
                    event.accept()
                    return True
                if event.button() == Qt.RightButton:
                    if event.modifiers() & Qt.ShiftModifier:
                        self._handle_canvas_release()
                    else:
                        self._handle_pan_release()
                    event.accept()
                    return True
        return super().eventFilter(obj, event)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        canvas_pos = self._canvas_widget.mapFrom(self, event.position().toPoint())
        self._update_hover_from_position(QPointF(canvas_pos))
        self._refresh_bounding_box_hover_for_position(QPointF(canvas_pos))
        if event.button() == Qt.LeftButton:
            if self._bounding_box_interaction_enabled():
                if self._handle_bounding_box_press(QPointF(canvas_pos)):
                    event.accept()
                    return
                self._handle_picker_press(QPointF(canvas_pos))
                event.accept()
                return
            if self._annotation_tool() == "flood_filler":
                self._handle_picker_press(QPointF(canvas_pos))
            else:
                self._handle_annotation_press(QPointF(canvas_pos))
            event.accept()
            return
        if event.button() == Qt.RightButton:
            if event.modifiers() & Qt.ShiftModifier:
                self._handle_canvas_press(QPointF(canvas_pos))
            else:
                self._handle_pan_press(QPointF(canvas_pos))
            event.accept()
            return

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        canvas_pos = self._canvas_widget.mapFrom(self, event.position().toPoint())
        self._update_hover_from_position(QPointF(canvas_pos))
        self._refresh_bounding_box_hover_for_position(QPointF(canvas_pos))
        if event.buttons() & Qt.LeftButton:
            if (
                self._bbox_drag_hit is not None or self._bounding_box_interaction_enabled()
            ) and self._handle_bounding_box_move(QPointF(canvas_pos)):
                event.accept()
                return
            if self._annotation_tool() != "flood_filler":
                self._handle_annotation_move(QPointF(canvas_pos))
            event.accept()
            return
        if event.buttons() & Qt.RightButton:
            if event.modifiers() & Qt.ShiftModifier:
                self._handle_canvas_move(QPointF(canvas_pos))
            else:
                self._handle_pan_move(QPointF(canvas_pos))
            event.accept()
            return

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        canvas_pos = self._canvas_widget.mapFrom(self, event.position().toPoint())
        self._update_hover_from_position(QPointF(canvas_pos))
        self._refresh_bounding_box_hover_for_position(QPointF(canvas_pos))
        if event.button() == Qt.LeftButton:
            if self._bbox_drag_hit is not None and self._handle_bounding_box_release():
                event.accept()
                return
            if self._annotation_drag_active or (
                not self._bounding_box_interaction_enabled()
                and self._annotation_tool() != "flood_filler"
            ):
                self._handle_annotation_release()
            event.accept()
            return
        if event.button() == Qt.RightButton:
            if event.modifiers() & Qt.ShiftModifier:
                self._handle_canvas_release()
            else:
                self._handle_pan_release()
            event.accept()
            return

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() in (Qt.Key_Backspace, Qt.Key_Delete):
            if self._on_bounding_box_delete_requested is not None:
                self._on_bounding_box_delete_requested()
            # Prevent fallthrough to VisPy's camera reset bound to Backspace.
            event.accept()
            return
        if event.key() in (Qt.Key_Up, Qt.Key_Right):
            self.input_handlers.on_scroll(self.state.axis, 1)
            event.accept()
            return
        if event.key() in (Qt.Key_Down, Qt.Key_Left):
            self.input_handlers.on_scroll(self.state.axis, -1)
            event.accept()
            return
        if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self.input_handlers.on_zoom(self.state.zoom + 0.1)
            event.accept()
            return
        if event.key() == Qt.Key_Minus:
            self.input_handlers.on_zoom(max(0.1, self.state.zoom - 0.1))
            event.accept()
            return
        super().keyPressEvent(event)
