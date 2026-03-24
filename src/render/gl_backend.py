from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import colorsys
import logging

import numpy as np

from ..bbox.interaction import BoundingBoxHandleHit
from ..bbox.model import BoundingBoxLabel
from ..bbox.projection import ProjectedBoundingBox2D
from ..utils import maybe_profile


logger = logging.getLogger(__name__)
SegmentationROI = Tuple[int, int, int, int]
_SEGMENTATION_ALPHA = 0.3
_SEGMENTATION_ALPHA_U8 = int((_SEGMENTATION_ALPHA * 255.0) + 0.5)


@dataclass
class GLTexture:
    width: int
    height: int
    dtype: str
    data: Optional[np.ndarray] = None
    segmentation: Optional[np.ndarray] = None
    segmentation_patch: Optional[np.ndarray] = None
    segmentation_range: Optional[Tuple[int, int]] = None
    segmentation_labels: Optional[np.ndarray] = None
    segmentation_roi: Optional[SegmentationROI] = None


class GLBackend:
    def __init__(self) -> None:
        self._ready = False
        self._canvas = None
        self._view = None
        self._image_node = None
        self._seg_node = None
        self._selection_marker_node = None
        self._bbox_line_node = None
        self._bbox_selected_line_node = None
        self._bbox_hover_handle_node = None
        self._bbox_active_handle_node = None
        self._image_dtype: Optional[np.dtype] = None
        self._crosshair_h = None
        self._crosshair_v = None
        self._zoom = 1.0
        self._pan = (0.0, 0.0)
        self._seg_range: Optional[Tuple[int, int]] = None
        self._seg_labels: Optional[np.ndarray] = None
        self._seg_palette: Optional[np.ndarray] = None
        self._seg_rgba_cache: Optional[np.ndarray] = None
        self._seg_texture = None
        self._seg_subupload_supported: Optional[bool] = None
        self._seg_subupload_mode: Optional[str] = None
        self._seg_subupload_mapping_verified = False
        self._seg_roi_upload_attempts = 0
        self._seg_roi_subupload_success = 0
        self._seg_roi_full_fallback = 0
        self._seg_roi_fallback_reasons: Dict[str, int] = {}
        self._seg_roi_stats_log_every = 200
        self._label_rgba_cache: Dict[int, np.ndarray] = {
            0: np.array([0, 0, 0, 0], dtype=np.uint8),
        }
        self._label_rgba_cache_limit = 131_072
        self._selection_marker_visible = False
        self._selection_marker_xy: Optional[Tuple[float, float]] = None
        self._bbox_label_colors: Dict[BoundingBoxLabel, np.ndarray] = {
            "train": np.array([0.0, 0.95, 0.35, 0.95], dtype=np.float32),
            "validation": np.array([0.05, 0.75, 1.0, 0.95], dtype=np.float32),
            "inference": np.array([1.0, 0.2, 0.2, 0.95], dtype=np.float32),
        }
        self._bbox_default_color = self._bbox_label_colors["train"]
        self._bbox_selected_color = np.array([1.0, 0.95, 0.2, 1.0], dtype=np.float32)
        self._bbox_hover_color = np.array([0.15, 0.7, 1.0, 1.0], dtype=np.float32)
        self._bbox_active_color = np.array([1.0, 0.45, 0.1, 1.0], dtype=np.float32)
        self._fit_done = False
        self._gl_info_logged = False

    def initialize(self) -> None:
        if self._ready:
            return
        try:
            from vispy import scene
        except ImportError as exc:
            raise ImportError("vispy is required for GL backend") from exc

        self._canvas = scene.SceneCanvas(keys=None, show=False, bgcolor="black")
        self._view = self._canvas.central_widget.add_view()
        self._image_node = scene.Image(
            np.zeros((2, 2), dtype=np.uint8),
            parent=self._view.scene,
            cmap="grays",
        )
        self._seg_node = scene.Image(
            np.zeros((2, 2), dtype=np.uint8),
            parent=self._view.scene,
            cmap="grays",
            interpolation="nearest",
        )
        try:
            self._seg_node.order = 10
        except Exception:
            pass
        try:
            self._image_node.order = 0
        except Exception:
            pass
        if hasattr(self._seg_node, "set_gl_state"):
            self._seg_node.set_gl_state(blend=True, depth_test=False)
        self._seg_node.visible = False
        self._image_dtype = np.dtype(np.uint8)
        self._crosshair_h = scene.Line(
            np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            parent=self._view.scene,
            color=(1.0, 0.0, 0.0, 0.8),
            width=1.0,
        )
        self._crosshair_v = scene.Line(
            np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            parent=self._view.scene,
            color=(1.0, 0.0, 0.0, 0.8),
            width=1.0,
        )
        try:
            self._selection_marker_node = scene.Markers(parent=self._view.scene)
        except Exception:
            self._selection_marker_node = scene.visuals.Markers(parent=self._view.scene)
        self._selection_marker_node.set_data(
            pos=self._selection_marker_position(0.0, 0.0),
            face_color=np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32),
            edge_color=np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            size=8.0,
        )
        self._selection_marker_node.visible = False
        if hasattr(self._selection_marker_node, "set_gl_state"):
            self._selection_marker_node.set_gl_state(blend=True, depth_test=False)
        try:
            self._selection_marker_node.order = 20
        except Exception:
            pass
        self._bbox_line_node = scene.Line(
            np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            parent=self._view.scene,
            color=tuple(float(channel) for channel in self._bbox_default_color),
            width=1.25,
        )
        self._bbox_line_node.visible = False
        if hasattr(self._bbox_line_node, "set_gl_state"):
            self._bbox_line_node.set_gl_state(blend=True, depth_test=False)
        try:
            self._bbox_line_node.order = 15
        except Exception:
            pass
        self._bbox_selected_line_node = scene.Line(
            np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            parent=self._view.scene,
            color=tuple(float(channel) for channel in self._bbox_selected_color),
            width=2.0,
        )
        self._bbox_selected_line_node.visible = False
        if hasattr(self._bbox_selected_line_node, "set_gl_state"):
            self._bbox_selected_line_node.set_gl_state(blend=True, depth_test=False)
        try:
            self._bbox_selected_line_node.order = 16
        except Exception:
            pass
        try:
            self._bbox_hover_handle_node = scene.Markers(parent=self._view.scene)
        except Exception:
            self._bbox_hover_handle_node = scene.visuals.Markers(parent=self._view.scene)
        self._bbox_hover_handle_node.set_data(
            pos=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            face_color=self._bbox_hover_color[np.newaxis, :],
            edge_color=np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            size=7.0,
        )
        self._bbox_hover_handle_node.visible = False
        if hasattr(self._bbox_hover_handle_node, "set_gl_state"):
            self._bbox_hover_handle_node.set_gl_state(blend=True, depth_test=False)
        try:
            self._bbox_hover_handle_node.order = 17
        except Exception:
            pass
        try:
            self._bbox_active_handle_node = scene.Markers(parent=self._view.scene)
        except Exception:
            self._bbox_active_handle_node = scene.visuals.Markers(parent=self._view.scene)
        self._bbox_active_handle_node.set_data(
            pos=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            face_color=self._bbox_active_color[np.newaxis, :],
            edge_color=np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            size=9.0,
        )
        self._bbox_active_handle_node.visible = False
        if hasattr(self._bbox_active_handle_node, "set_gl_state"):
            self._bbox_active_handle_node.set_gl_state(blend=True, depth_test=False)
        try:
            self._bbox_active_handle_node.order = 18
        except Exception:
            pass
        self._view.camera = "panzoom"
        if hasattr(self._view.camera, "aspect"):
            self._view.camera.aspect = 1
        self._log_gl_context_info()
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def widget(self):
        if not self._ready:
            self.initialize()
        return self._canvas.native  # Qt widget

    def _ensure_float_image_node(self, image: np.ndarray) -> None:
        if self._image_node is None:
            return
        if self._image_dtype is None:
            self._image_dtype = image.dtype
            return
        if image.dtype.kind != "f" or self._image_dtype.kind == "f":
            return

        try:
            from vispy import scene
        except ImportError as exc:
            raise ImportError("vispy is required for GL backend") from exc

        parent = self._image_node.parent
        self._image_node.parent = None
        self._image_node = scene.Image(
            np.zeros((2, 2), dtype=image.dtype),
            parent=parent,
            cmap="grays",
        )
        self._image_dtype = image.dtype

    def upload_texture(
        self,
        image: np.ndarray,
        segmentation: Optional[np.ndarray] = None,
        segmentation_patch: Optional[np.ndarray] = None,
        segmentation_range: Optional[Tuple[int, int]] = None,
        segmentation_labels: Optional[np.ndarray] = None,
        segmentation_roi: Optional[SegmentationROI] = None,
    ) -> GLTexture:
        if not self._ready:
            self.initialize()
        height, width = image.shape[:2]
        texture = GLTexture(
            width=width,
            height=height,
            dtype=str(image.dtype),
            data=image,
            segmentation=segmentation,
            segmentation_patch=segmentation_patch,
            segmentation_range=segmentation_range,
            segmentation_labels=segmentation_labels,
            segmentation_roi=segmentation_roi,
        )
        self._ensure_float_image_node(image)
        self._image_node.set_data(image)
        self._image_node.clim = (0.0, 1.0)
        self._update_segmentation(
            segmentation,
            segmentation_range,
            segmentation_labels,
            segmentation_roi=segmentation_roi,
            segmentation_patch=segmentation_patch,
        )
        if not self._fit_done:
            if hasattr(self._view.camera, "set_range"):
                self._view.camera.set_range(x=(0, width), y=(0, height), margin=0)
            if hasattr(self._view.camera, "center"):
                self._view.camera.center = (width / 2, height / 2)
            self._fit_done = True
        self._canvas.update()
        return texture

    def draw_texture(self, texture: GLTexture) -> None:
        if not self._ready:
            self.initialize()
        if texture.data is None:
            return
        self._ensure_float_image_node(texture.data)
        self._image_node.set_data(texture.data)
        self._image_node.clim = (0.0, 1.0)
        self._update_segmentation(
            texture.segmentation,
            texture.segmentation_range,
            texture.segmentation_labels,
            segmentation_roi=texture.segmentation_roi,
            segmentation_patch=texture.segmentation_patch,
        )
        self._canvas.update()

    def update_segmentation_overlay(
        self,
        segmentation: Optional[np.ndarray],
        segmentation_range: Optional[Tuple[int, int]],
        segmentation_labels: Optional[np.ndarray],
        segmentation_roi: Optional[SegmentationROI] = None,
        segmentation_patch: Optional[np.ndarray] = None,
    ) -> None:
        if not self._ready:
            self.initialize()
        self._update_segmentation(
            segmentation,
            segmentation_range,
            segmentation_labels,
            segmentation_roi=segmentation_roi,
            segmentation_patch=segmentation_patch,
        )
        self._canvas.update()

    def clear_bounding_boxes_overlay(self) -> None:
        if not self._ready:
            self.initialize()
        if self._bbox_line_node is not None:
            self._bbox_line_node.visible = False
        if self._bbox_selected_line_node is not None:
            self._bbox_selected_line_node.visible = False
        if self._bbox_hover_handle_node is not None:
            self._bbox_hover_handle_node.visible = False
        if self._bbox_active_handle_node is not None:
            self._bbox_active_handle_node.visible = False
        if self._canvas is not None:
            self._canvas.update()

    def update_bounding_boxes_overlay(
        self,
        boxes: Iterable[ProjectedBoundingBox2D],
        *,
        selected_id: Optional[str] = None,
        hover_hit: Optional[BoundingBoxHandleHit] = None,
        active_hit: Optional[BoundingBoxHandleHit] = None,
    ) -> None:
        if not self._ready:
            self.initialize()
        if self._bbox_line_node is None or self._bbox_selected_line_node is None:
            return

        default_parts = []
        default_part_colors = []
        selected_parts = []
        by_id: Dict[str, ProjectedBoundingBox2D] = {}
        for box in boxes:
            if not isinstance(box, ProjectedBoundingBox2D):
                continue
            x0 = float(box.col0)
            x1 = float(box.col1)
            y0 = float(box.row0)
            y1 = float(box.row1)
            if x1 <= x0 or y1 <= y0:
                continue
            by_id[box.box_id] = box
            points = self._projected_box_segments(box)
            if selected_id is not None and box.box_id == selected_id:
                selected_parts.append(points)
            else:
                default_parts.append(points)
                default_part_colors.append(self._bbox_color_for_label(box.label))

        if not default_parts and not selected_parts:
            self._bbox_line_node.visible = False
            self._bbox_selected_line_node.visible = False
            if self._bbox_hover_handle_node is not None:
                self._bbox_hover_handle_node.visible = False
            if self._bbox_active_handle_node is not None:
                self._bbox_active_handle_node.visible = False
            if self._canvas is not None:
                self._canvas.update()
            return

        self._set_bounding_box_line_node(
            self._bbox_line_node,
            default_parts,
            color=self._bbox_default_color,
            width=1.25,
            part_colors=default_part_colors,
        )
        self._set_bounding_box_line_node(
            self._bbox_selected_line_node,
            selected_parts,
            color=self._bbox_selected_color,
            width=2.0,
        )

        active_marker = self._handle_marker_xy(active_hit, by_id=by_id)
        hover_marker = self._handle_marker_xy(hover_hit, by_id=by_id)
        if (
            active_hit is not None
            and hover_hit is not None
            and active_hit.box_id == hover_hit.box_id
            and active_hit.handle == hover_hit.handle
            and active_hit.kind == hover_hit.kind
        ):
            hover_marker = None
        self._set_handle_marker_node(
            self._bbox_hover_handle_node,
            hover_marker,
            color=self._bbox_hover_color,
            size=7.0,
        )
        self._set_handle_marker_node(
            self._bbox_active_handle_node,
            active_marker,
            color=self._bbox_active_color,
            size=9.0,
        )
        if self._canvas is not None:
            self._canvas.update()

    def _set_bounding_box_line_node(
        self,
        node,
        parts: Iterable[np.ndarray],
        *,
        color: np.ndarray,
        width: float,
        part_colors: Optional[Iterable[np.ndarray]] = None,
    ) -> None:
        if node is None:
            return
        items = tuple(parts)
        if not items:
            node.visible = False
            return
        pos = np.concatenate(items, axis=0)
        if part_colors is None:
            colors = np.repeat(color[np.newaxis, :], pos.shape[0], axis=0)
        else:
            color_items = tuple(part_colors)
            if len(color_items) != len(items):
                color_items = tuple(color for _ in items)
            expanded = []
            for points, part_color in zip(items, color_items):
                expanded.append(
                    np.repeat(
                        np.asarray(part_color, dtype=np.float32)[np.newaxis, :],
                        points.shape[0],
                        axis=0,
                    )
                )
            colors = np.concatenate(expanded, axis=0)
        try:
            node.set_data(
                pos=pos,
                color=colors,
                connect="segments",
                width=float(width),
            )
        except TypeError:
            node.set_data(
                pos=pos,
                color=colors,
                connect="segments",
            )
        node.visible = True

    def _bbox_color_for_label(self, label: object) -> np.ndarray:
        if not isinstance(label, str):
            return self._bbox_default_color
        normalized = label.strip().lower()
        if normalized == "train":
            return self._bbox_label_colors["train"]
        if normalized == "validation":
            return self._bbox_label_colors["validation"]
        if normalized == "inference":
            return self._bbox_label_colors["inference"]
        return self._bbox_default_color

    def _set_handle_marker_node(
        self,
        node,
        marker_xy: Optional[Tuple[float, float]],
        *,
        color: np.ndarray,
        size: float,
    ) -> None:
        if node is None:
            return
        if marker_xy is None:
            node.visible = False
            return
        node.set_data(
            pos=np.array([[marker_xy[0], marker_xy[1], 0.0]], dtype=np.float32),
            face_color=color[np.newaxis, :],
            edge_color=np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            size=float(size),
        )
        node.visible = True

    def _projected_box_segments(self, box: ProjectedBoundingBox2D) -> np.ndarray:
        x0 = float(box.col0)
        x1 = float(box.col1)
        y0 = float(box.row0)
        y1 = float(box.row1)
        return np.array(
            [
                [x0, y0],
                [x1, y0],
                [x1, y0],
                [x1, y1],
                [x1, y1],
                [x0, y1],
                [x0, y1],
                [x0, y0],
            ],
            dtype=np.float32,
        )

    def _handle_marker_xy(
        self,
        hit: Optional[BoundingBoxHandleHit],
        *,
        by_id: Dict[str, ProjectedBoundingBox2D],
    ) -> Optional[Tuple[float, float]]:
        if hit is None:
            return None
        box = by_id.get(hit.box_id)
        if box is None:
            return None
        x0 = float(box.col0)
        x1 = float(box.col1)
        y0 = float(box.row0)
        y1 = float(box.row1)

        if hit.handle == "top_left":
            return (x0, y0)
        if hit.handle == "top_right":
            return (x1, y0)
        if hit.handle == "bottom_left":
            return (x0, y1)
        if hit.handle == "bottom_right":
            return (x1, y1)
        if hit.handle == "top":
            return ((x0 + x1) / 2.0, y0)
        if hit.handle == "bottom":
            return ((x0 + x1) / 2.0, y1)
        if hit.handle == "left":
            return (x0, (y0 + y1) / 2.0)
        if hit.handle == "right":
            return (x1, (y0 + y1) / 2.0)
        return None

    def _reset_segmentation_texture_state(self) -> None:
        self._seg_texture = None
        self._seg_subupload_supported = None
        self._seg_subupload_mode = None
        self._seg_subupload_mapping_verified = False

    def _update_segmentation(
        self,
        segmentation: Optional[np.ndarray],
        segmentation_range: Optional[Tuple[int, int]],
        segmentation_labels: Optional[np.ndarray],
        segmentation_roi: Optional[SegmentationROI] = None,
        segmentation_patch: Optional[np.ndarray] = None,
    ) -> None:
        seg_shape = None if segmentation is None else tuple(np.asarray(segmentation).shape)
        label_count = 0 if segmentation_labels is None else int(np.asarray(segmentation_labels).size)
        with maybe_profile(
            "gl_backend_update_segmentation",
            logger=logger,
            details=f"shape={seg_shape} labels={label_count} roi={segmentation_roi}",
        ):
            if self._seg_node is None:
                return
            if segmentation is None:
                self._seg_node.visible = False
                self._seg_range = None
                self._seg_labels = None
                self._seg_palette = None
                self._seg_rgba_cache = None
                self._reset_segmentation_texture_state()
                return
            if segmentation_labels is not None:
                labels = np.asarray(segmentation_labels, dtype=np.int64).reshape(-1)
                if labels.size > 0:
                    if self._seg_labels is None:
                        labels_changed = True
                    elif self._seg_labels is labels:
                        labels_changed = False
                    elif self._seg_labels.shape != labels.shape:
                        labels_changed = True
                    else:
                        labels_changed = not np.array_equal(self._seg_labels, labels)
                    if labels_changed:
                        self._seg_labels = labels.copy()
                        self._seg_palette = self._build_seg_palette(self._seg_labels)
                        self._seg_range = None
                    self._seg_node.visible = True
                    if self._seg_palette is None or self._seg_palette.shape[0] != labels.shape[0]:
                        self._seg_palette = self._build_seg_palette(labels)
                    seg_array = np.asarray(segmentation)
                    roi = self._normalize_segmentation_roi(segmentation_roi, seg_array.shape[:2])
                    if (
                        roi is not None
                        and self._seg_rgba_cache is not None
                        and self._seg_rgba_cache.shape[:2] == seg_array.shape[:2]
                    ):
                        row0, row1, col0, col1 = roi
                        patch = self._normalize_segmentation_patch(
                            segmentation_patch,
                            roi=roi,
                        )
                        if patch is None:
                            patch = seg_array[row0:row1, col0:col1]
                        rgba_patch = self._colorize_labeled_segmentation(patch)
                        self._seg_rgba_cache[row0:row1, col0:col1] = rgba_patch
                        uploaded, reason = self._upload_segmentation_roi(rgba_patch, roi=roi)
                        self._record_seg_roi_upload_result(uploaded, reason=reason)
                        if not uploaded:
                            self._seg_node.set_data(self._seg_rgba_cache)
                        return
                    dense = self._remap_segmentation_to_labels(seg_array, labels)
                    rgba = self._colorize_segmentation(dense, self._seg_palette)
                    self._seg_rgba_cache = np.array(rgba, copy=True)
                    self._seg_node.set_data(self._seg_rgba_cache)
                    self._reset_segmentation_texture_state()
                    return
            if segmentation_range is None:
                seg_min = int(np.min(segmentation))
                seg_max = int(np.max(segmentation))
                segmentation_range = (seg_min, seg_max)
            seg_min, seg_max = segmentation_range
            if self._seg_range != segmentation_range:
                self._seg_node.cmap = self._build_seg_colormap(seg_min, seg_max)
                self._seg_range = segmentation_range
            self._seg_node.visible = True
            if seg_min == seg_max:
                self._seg_node.clim = (seg_min - 0.5, seg_max + 0.5)
            else:
                self._seg_node.clim = (seg_min, seg_max)
            self._seg_node.set_data(segmentation)
            self._seg_labels = None
            self._seg_palette = None
            self._seg_rgba_cache = None
            self._reset_segmentation_texture_state()

    def _normalize_segmentation_roi(
        self,
        segmentation_roi: Optional[SegmentationROI],
        shape: Tuple[int, int],
    ) -> Optional[SegmentationROI]:
        if segmentation_roi is None:
            return None
        height, width = int(shape[0]), int(shape[1])
        row0, row1, col0, col1 = segmentation_roi
        row0 = max(0, min(int(row0), height))
        row1 = max(0, min(int(row1), height))
        col0 = max(0, min(int(col0), width))
        col1 = max(0, min(int(col1), width))
        if row0 >= row1 or col0 >= col1:
            return None
        return (row0, row1, col0, col1)

    def _build_seg_colormap(self, seg_min: int, seg_max: int):
        try:
            from vispy.color import Colormap
        except Exception:
            return "grays"
        count = max(1, seg_max - seg_min + 1)
        colors = []
        for idx in range(count):
            label = seg_min + idx
            if label == 0:
                colors.append((0.0, 0.0, 0.0, 0.0))
                continue
            hue = self._label_hue(label)
            r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
            colors.append((r, g, b, _SEGMENTATION_ALPHA))
        return Colormap(colors)

    def _build_seg_palette(self, labels: np.ndarray) -> np.ndarray:
        palette = np.zeros((labels.size, 4), dtype=np.uint8)
        for idx, label in enumerate(labels):
            palette[idx] = self._label_to_rgba(int(label))
        return palette

    def _normalize_segmentation_patch(
        self,
        patch: Optional[np.ndarray],
        *,
        roi: SegmentationROI,
    ) -> Optional[np.ndarray]:
        if patch is None:
            return None
        row0, row1, col0, col1 = roi
        expected_shape = (int(row1 - row0), int(col1 - col0))
        array = np.asarray(patch)
        if array.ndim < 2:
            return None
        if array.shape[0] != expected_shape[0] or array.shape[1] != expected_shape[1]:
            return None
        return array

    def _upload_segmentation_roi(
        self,
        rgba_patch: np.ndarray,
        *,
        roi: SegmentationROI,
    ) -> Tuple[bool, str]:
        if self._seg_node is None:
            return (False, "seg_node_missing")
        if self._seg_subupload_supported is False:
            return (False, "subupload_disabled")
        row0, row1, col0, col1 = roi
        if row1 <= row0 or col1 <= col0:
            return (True, "empty_roi")
        expected_shape = (int(row1 - row0), int(col1 - col0))
        if rgba_patch.shape[0] != expected_shape[0] or rgba_patch.shape[1] != expected_shape[1]:
            return (False, "shape_mismatch")

        texture = self._seg_texture_handle()
        if texture is None:
            self._seg_subupload_supported = False
            return (False, "texture_unavailable")

        texture_shape = None
        if self._seg_rgba_cache is not None and self._seg_rgba_cache.ndim >= 2:
            texture_shape = (
                int(self._seg_rgba_cache.shape[0]),
                int(self._seg_rgba_cache.shape[1]),
            )
        if texture_shape is None:
            self._seg_subupload_supported = False
            return (False, "texture_shape_unavailable")

        if not self._seg_subupload_mapping_verified or self._seg_subupload_mode is None:
            verified, reason = self._verify_subupload_mapping(texture, texture_shape=texture_shape)
            if not verified:
                self._seg_subupload_supported = False
                self._seg_subupload_mode = None
                self._seg_subupload_mapping_verified = False
                return (False, reason)

        offset = self._roi_texture_offset(
            roi,
            mode=self._seg_subupload_mode,
            texture_shape=texture_shape,
        )
        if offset is None:
            self._seg_subupload_supported = False
            self._seg_subupload_mode = None
            self._seg_subupload_mapping_verified = False
            return (False, "mapping_offset_invalid")

        patch_data = np.ascontiguousarray(rgba_patch)
        try:
            texture.set_data(patch_data, offset=offset)
        except Exception:
            self._seg_subupload_supported = False
            return (False, "texture_set_data_failed")

        self._seg_subupload_supported = True
        if hasattr(self._seg_node, "update"):
            try:
                self._seg_node.update()
            except Exception:
                pass
        return (True, "ok")

    def _roi_texture_offset(
        self,
        roi: SegmentationROI,
        *,
        mode: Optional[str],
        texture_shape: Tuple[int, int],
    ) -> Optional[Tuple[int, int]]:
        if mode is None:
            return None
        row0, row1, col0, col1 = roi
        height, width = int(texture_shape[0]), int(texture_shape[1])
        patch_h = int(row1 - row0)
        patch_w = int(col1 - col0)
        if patch_h <= 0 or patch_w <= 0 or height <= 0 or width <= 0:
            return None

        if mode == "xy_top_left":
            offset = (int(col0), int(row0))
            max0 = width - patch_w
            max1 = height - patch_h
        elif mode == "xy_bottom_left":
            offset = (int(col0), int(height - row1))
            max0 = width - patch_w
            max1 = height - patch_h
        elif mode == "yx_top_left":
            offset = (int(row0), int(col0))
            max0 = height - patch_h
            max1 = width - patch_w
        elif mode == "yx_bottom_left":
            offset = (int(height - row1), int(col0))
            max0 = height - patch_h
            max1 = width - patch_w
        else:
            return None

        if max0 < 0 or max1 < 0:
            return None
        if offset[0] < 0 or offset[1] < 0:
            return None
        if offset[0] > max0 or offset[1] > max1:
            return None
        return offset

    def _find_single_pixel_delta(
        self,
        before: np.ndarray,
        after: np.ndarray,
        probe_pixel: np.ndarray,
    ) -> Optional[Tuple[int, int]]:
        if before.shape != after.shape or before.ndim < 2:
            return None
        if before.ndim == 2:
            diff = before != after
        else:
            diff = np.any(before != after, axis=-1)
        changed_coords = np.argwhere(diff)
        if changed_coords.shape[0] == 1:
            row, col = changed_coords[0]
            return (int(row), int(col))
        if changed_coords.shape[0] == 0 or before.ndim < 3:
            return None

        channels = min(int(after.shape[-1]), int(probe_pixel.shape[0]))
        if channels <= 0:
            return None
        probe = np.asarray(probe_pixel[:channels]).reshape((1, 1, channels))
        probe_match = np.all(after[..., :channels] == probe, axis=-1)
        targeted_coords = np.argwhere(diff & probe_match)
        if targeted_coords.shape[0] == 1:
            row, col = targeted_coords[0]
            return (int(row), int(col))
        return None

    def _verify_subupload_mapping(
        self,
        texture,
        *,
        texture_shape: Tuple[int, int],
    ) -> Tuple[bool, str]:
        if self._seg_rgba_cache is None:
            return (False, "mapping_cache_missing")
        if not hasattr(texture, "get_data"):
            return (False, "mapping_readback_unavailable")

        cache = np.asarray(self._seg_rgba_cache)
        if cache.ndim < 3 or cache.shape[0] <= 0 or cache.shape[1] <= 0:
            return (False, "mapping_cache_invalid")
        if cache.shape[0] != int(texture_shape[0]) or cache.shape[1] != int(texture_shape[1]):
            return (False, "mapping_shape_mismatch")

        base_cache = np.ascontiguousarray(cache)
        height, width = int(texture_shape[0]), int(texture_shape[1])
        probe_row = min(max(height // 3, 0), height - 1)
        probe_col = min(max(width // 3, 0), width - 1)

        base_pixel = np.asarray(base_cache[probe_row, probe_col], dtype=np.uint8).copy()
        channels = int(base_pixel.shape[0]) if base_pixel.ndim == 1 else int(base_cache.shape[2])
        if channels <= 0:
            return (False, "mapping_channels_invalid")
        probe_pixel = base_pixel.copy()
        probe_pixel[0] = np.uint8(int(probe_pixel[0]) ^ 0xFF)
        if channels > 1:
            probe_pixel[1] = np.uint8(int(probe_pixel[1]) ^ 0xA5)
        if channels > 2:
            probe_pixel[2] = np.uint8(int(probe_pixel[2]) ^ 0x5A)
        if channels > 3:
            probe_pixel[3] = np.uint8(255)
        if np.array_equal(probe_pixel, base_pixel):
            probe_pixel[0] = np.uint8((int(probe_pixel[0]) + 1) % 256)

        try:
            texture.set_data(base_cache)
            baseline = np.asarray(texture.get_data())
        except Exception:
            return (False, "mapping_baseline_readback_failed")

        reference_cache = np.array(base_cache, copy=True)
        reference_cache[probe_row, probe_col] = probe_pixel
        try:
            texture.set_data(reference_cache)
            reference = np.asarray(texture.get_data())
        except Exception:
            try:
                texture.set_data(base_cache)
            except Exception:
                pass
            return (False, "mapping_reference_probe_failed")

        reference_location = self._find_single_pixel_delta(baseline, reference, probe_pixel)
        if reference_location is None:
            try:
                texture.set_data(base_cache)
            except Exception:
                pass
            return (False, "mapping_reference_location_unknown")

        try:
            texture.set_data(base_cache)
            baseline = np.asarray(texture.get_data())
        except Exception:
            return (False, "mapping_restore_failed")

        probe_patch = np.asarray(probe_pixel, dtype=np.uint8).reshape((1, 1, channels))
        probe_patch = np.ascontiguousarray(probe_patch)
        probe_roi = (probe_row, probe_row + 1, probe_col, probe_col + 1)
        candidate_modes = (
            "xy_top_left",
            "xy_bottom_left",
            "yx_top_left",
            "yx_bottom_left",
        )
        for mode in candidate_modes:
            offset = self._roi_texture_offset(
                probe_roi,
                mode=mode,
                texture_shape=texture_shape,
            )
            if offset is None:
                continue
            try:
                texture.set_data(probe_patch, offset=offset)
                attempted = np.asarray(texture.get_data())
            except Exception:
                try:
                    texture.set_data(base_cache)
                    baseline = np.asarray(texture.get_data())
                except Exception:
                    return (False, "mapping_restore_failed")
                continue

            candidate_location = self._find_single_pixel_delta(baseline, attempted, probe_pixel)
            try:
                texture.set_data(base_cache)
                baseline = np.asarray(texture.get_data())
            except Exception:
                return (False, "mapping_restore_failed")

            if candidate_location == reference_location:
                self._seg_subupload_mode = mode
                self._seg_subupload_mapping_verified = True
                self._seg_subupload_supported = True
                return (True, "mapping_verified")

        self._seg_subupload_mapping_verified = False
        self._seg_subupload_mode = None
        return (False, "mapping_mode_unknown")

    def _record_seg_roi_upload_result(self, success: bool, *, reason: str) -> None:
        self._seg_roi_upload_attempts += 1
        if success:
            self._seg_roi_subupload_success += 1
        else:
            self._seg_roi_full_fallback += 1
            self._seg_roi_fallback_reasons[reason] = self._seg_roi_fallback_reasons.get(reason, 0) + 1

        attempts = self._seg_roi_upload_attempts
        if (
            attempts > 0
            and self._seg_roi_stats_log_every > 0
            and (attempts % self._seg_roi_stats_log_every == 0)
            and logger.isEnabledFor(logging.DEBUG)
        ):
            success_rate = (100.0 * self._seg_roi_subupload_success) / float(attempts)
            if self._seg_roi_fallback_reasons:
                top_reason = max(
                    self._seg_roi_fallback_reasons.items(),
                    key=lambda item: item[1],
                )
                reason_text = f"{top_reason[0]} ({top_reason[1]})"
            else:
                reason_text = "none"
            logger.debug(
                "ROI upload stats: attempts=%d success=%d fallback=%d success_rate=%.1f%% top_fallback=%s",
                attempts,
                self._seg_roi_subupload_success,
                self._seg_roi_full_fallback,
                success_rate,
                reason_text,
            )

    def segmentation_upload_stats(self) -> Dict[str, object]:
        attempts = int(self._seg_roi_upload_attempts)
        success = int(self._seg_roi_subupload_success)
        fallback = int(self._seg_roi_full_fallback)
        success_rate = 0.0 if attempts <= 0 else (100.0 * float(success) / float(attempts))
        return {
            "roi_upload_attempts": attempts,
            "roi_subupload_success": success,
            "roi_full_fallback": fallback,
            "roi_subupload_success_rate": success_rate,
            "roi_fallback_reasons": dict(self._seg_roi_fallback_reasons),
            "subupload_supported_flag": self._seg_subupload_supported,
            "subupload_mode": self._seg_subupload_mode,
            "subupload_mapping_verified": bool(self._seg_subupload_mapping_verified),
        }

    def reset_segmentation_upload_stats(self) -> None:
        self._seg_roi_upload_attempts = 0
        self._seg_roi_subupload_success = 0
        self._seg_roi_full_fallback = 0
        self._seg_roi_fallback_reasons = {}

    def _seg_texture_handle(self):
        if self._seg_texture is not None and hasattr(self._seg_texture, "set_data"):
            return self._seg_texture
        if self._seg_node is None:
            return None

        candidates = []
        for attr_name in ("_texture", "texture"):
            candidate = getattr(self._seg_node, attr_name, None)
            if candidate is not None:
                candidates.append(candidate)

        data_obj = getattr(self._seg_node, "_data", None)
        if data_obj is not None:
            for attr_name in ("_texture", "texture"):
                candidate = getattr(data_obj, attr_name, None)
                if candidate is not None:
                    candidates.append(candidate)

        for candidate in candidates:
            if hasattr(candidate, "set_data"):
                self._seg_texture = candidate
                return candidate
        return None

    def _label_to_rgba(self, label: int) -> Tuple[int, int, int, int]:
        if label == 0:
            return (0, 0, 0, 0)
        hue = self._label_hue(label)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
        return (
            int(np.clip(round(r * 255), 0, 255)),
            int(np.clip(round(g * 255), 0, 255)),
            int(np.clip(round(b * 255), 0, 255)),
            _SEGMENTATION_ALPHA_U8,
        )

    def _label_rgba(self, label: int) -> np.ndarray:
        normalized = int(label)
        cached = self._label_rgba_cache.get(normalized)
        if cached is not None:
            return cached
        if len(self._label_rgba_cache) >= self._label_rgba_cache_limit:
            self._label_rgba_cache.clear()
            self._label_rgba_cache[0] = np.array([0, 0, 0, 0], dtype=np.uint8)
        rgba = np.asarray(self._label_to_rgba(normalized), dtype=np.uint8)
        self._label_rgba_cache[normalized] = rgba
        return rgba

    def _palette_for_unique_labels(self, labels: np.ndarray) -> np.ndarray:
        palette = np.empty((labels.size, 4), dtype=np.uint8)
        for idx, label in enumerate(labels):
            palette[idx] = self._label_rgba(int(label))
        return palette

    def _colorize_labeled_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        seg = np.asarray(segmentation)
        if seg.size == 0:
            return np.zeros(seg.shape + (4,), dtype=np.uint8)
        unique_labels, inverse = np.unique(seg.reshape(-1), return_inverse=True)
        palette = self._palette_for_unique_labels(unique_labels)
        rgba = palette[inverse]
        return rgba.reshape(seg.shape + (4,))

    def _label_hue(self, label: int) -> float:
        # Bit-reversed ordering (Van der Corput base-2) spreads consecutive
        # labels across the hue circle to improve visual separation.
        return self._bit_reverse32(label) / 2**32

    def _bit_reverse32(self, value: int) -> int:
        x = value & 0xFFFFFFFF
        x = ((x & 0x55555555) << 1) | ((x >> 1) & 0x55555555)
        x = ((x & 0x33333333) << 2) | ((x >> 2) & 0x33333333)
        x = ((x & 0x0F0F0F0F) << 4) | ((x >> 4) & 0x0F0F0F0F)
        x = ((x & 0x00FF00FF) << 8) | ((x >> 8) & 0x00FF00FF)
        x = ((x & 0x0000FFFF) << 16) | ((x >> 16) & 0x0000FFFF)
        return x

    def _colorize_segmentation(self, dense: np.ndarray, palette: np.ndarray) -> np.ndarray:
        dense_indices = np.asarray(dense, dtype=np.int32)
        return palette[dense_indices]

    def _remap_segmentation_to_labels(self, segmentation: np.ndarray, labels: np.ndarray) -> np.ndarray:
        if labels.size == 0:
            return np.zeros_like(segmentation, dtype=np.int32)
        seg = np.asarray(segmentation).astype(np.int64, copy=False)
        dense = np.searchsorted(labels, seg, side="left")
        dense = np.clip(dense, 0, labels.size - 1)
        matches = labels[dense] == seg
        if not np.all(matches):
            fallback_idx = int(np.searchsorted(labels, 0))
            if fallback_idx >= labels.size or labels[fallback_idx] != 0:
                fallback_idx = 0
            dense = np.where(matches, dense, fallback_idx)
        return dense.astype(np.int32, copy=False)

    def set_crosshair(self, x: float, y: float, width: int, height: int) -> None:
        if not self._ready:
            self.initialize()
        if self._crosshair_h is None or self._crosshair_v is None:
            return
        if width <= 0 or height <= 0:
            return
        x = float(np.clip(x, 0, width - 1))
        y = float(np.clip(y, 0, height - 1))
        self._crosshair_h.set_data(
            pos=np.array([[0.0, y], [float(width), y]], dtype=np.float32)
        )
        self._crosshair_v.set_data(
            pos=np.array([[x, 0.0], [x, float(height)]], dtype=np.float32)
        )
        self._canvas.update()

    def set_selection_marker(self, x: float, y: float, *, visible: bool) -> None:
        if not self._ready:
            self.initialize()
        if self._selection_marker_node is None:
            return

        if not visible:
            if self._selection_marker_visible:
                self._selection_marker_node.visible = False
                self._selection_marker_visible = False
                self._selection_marker_xy = None
                self._canvas.update()
            return

        x = float(x)
        y = float(y)
        if (
            self._selection_marker_visible
            and self._selection_marker_xy is not None
            and self._selection_marker_xy[0] == x
            and self._selection_marker_xy[1] == y
        ):
            return

        self._selection_marker_node.set_data(
            pos=self._selection_marker_position(x, y),
            face_color=np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32),
            edge_color=np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            size=8.0,
        )
        self._selection_marker_node.visible = True
        self._selection_marker_visible = True
        self._selection_marker_xy = (x, y)
        self._canvas.update()

    def _selection_marker_position(self, x: float, y: float) -> np.ndarray:
        return np.array([[float(x), float(y), 0.0]], dtype=np.float32)

    def set_pan(self, pan_x: float, pan_y: float, width: int, height: int) -> None:
        if not self._ready:
            self.initialize()
        if self._view is None or self._view.camera is None:
            return
        if width <= 0 or height <= 0:
            return
        delta_x = pan_x - self._pan[0]
        delta_y = pan_y - self._pan[1]
        if delta_x == 0 and delta_y == 0:
            return
        if hasattr(self._view.camera, "center"):
            center = self._view.camera.center
            if center is None:
                center = (width / 2, height / 2)
            self._view.camera.center = (center[0] - delta_x, center[1] - delta_y)
            self._canvas.update()
        self._pan = (pan_x, pan_y)

    def _map_image_to_scene(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        if not self._ready:
            self.initialize()
        if self._image_node is None:
            return None
        try:
            transform = self._image_node.get_transform(map_from="visual", map_to="scene")
        except TypeError:
            try:
                transform = self._image_node.get_transform("visual", "scene")
            except Exception:
                return None
        except Exception:
            return None
        if transform is None:
            return None
        mapped = None
        for coords in (
            np.array([[x, y]], dtype=np.float32),
            np.array([[x, y, 0.0]], dtype=np.float32),
            np.array([[x, y, 0.0, 1.0]], dtype=np.float32),
        ):
            try:
                mapped = transform.map(coords)
                break
            except Exception:
                continue
        if mapped is None or mapped.size == 0:
            return None
        return float(mapped[0][0]), float(mapped[0][1])

    def set_zoom(
        self,
        zoom: float,
        width: int,
        height: int,
        center: Optional[Tuple[float, float]] = None,
    ) -> None:
        if not self._ready:
            self.initialize()
        if self._view is None or self._view.camera is None:
            return
        if width <= 0 or height <= 0:
            return
        zoom = max(0.1, min(1.0, zoom))
        if not hasattr(self._view.camera, "zoom"):
            return
        if self._zoom <= 0:
            self._zoom = 1.0
        factor = zoom / self._zoom
        center_point = None
        if center is not None:
            center_point = self._map_image_to_scene(center[0], center[1])
        if factor == 1.0:
            if center_point is not None and hasattr(self._view.camera, "center"):
                self._view.camera.center = center_point
                self._canvas.update()
            return
        if center_point is None and hasattr(self._view.camera, "center"):
            center_point = self._view.camera.center
        try:
            self._view.camera.zoom(factor, center=center_point)
        except TypeError:
            self._view.camera.zoom(factor)
        self._zoom = zoom
        self._canvas.update()

    def map_canvas_to_image(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        if not self._ready:
            self.initialize()
        if self._canvas is None or self._image_node is None:
            return None
        scale = float(getattr(self._canvas, "pixel_scale", 1.0) or 1.0)
        x_canvas = x * scale
        y_canvas = y * scale
        try:
            transform = self._image_node.get_transform(map_from="canvas", map_to="visual")
        except TypeError:
            try:
                transform = self._image_node.get_transform("canvas", "visual")
            except Exception:
                return None
        except Exception:
            return None
        if transform is None:
            return None
        mapped = None
        for coords in (
            np.array([[x_canvas, y_canvas]], dtype=np.float32),
            np.array([[x_canvas, y_canvas, 0.0]], dtype=np.float32),
            np.array([[x_canvas, y_canvas, 0.0, 1.0]], dtype=np.float32),
        ):
            try:
                mapped = transform.map(coords)
                break
            except Exception:
                continue
        if mapped is None or mapped.size == 0:
            return None
        return float(mapped[0][0]), float(mapped[0][1])

    def _log_gl_context_info(self) -> None:
        if self._gl_info_logged:
            return
        self._gl_info_logged = True
        try:
            from vispy import gloo
        except Exception as exc:
            logger.info("OpenGL context details unavailable (vispy gloo import failed): %s", exc)
            return

        context = getattr(self._canvas, "context", None)
        if context is not None and hasattr(context, "set_current"):
            try:
                context.set_current()
            except Exception:
                pass

        gl = gloo.gl
        vendor = self._query_gl_string(gl, "GL_VENDOR")
        renderer = self._query_gl_string(gl, "GL_RENDERER")
        version = self._query_gl_string(gl, "GL_VERSION")
        logger.info(
            "OpenGL context: vendor=%s | renderer=%s | version=%s",
            vendor or "unknown",
            renderer or "unknown",
            version or "unknown",
        )

    def _query_gl_string(self, gl, token_name: str) -> Optional[str]:
        token = getattr(gl, token_name, None)
        if token is None:
            return None
        value = None

        get_string = getattr(gl, "glGetString", None)
        if callable(get_string):
            try:
                value = get_string(token)
            except Exception:
                value = None

        if value is None:
            get_parameter = getattr(gl, "glGetParameter", None)
            if callable(get_parameter):
                try:
                    value = get_parameter(token)
                except Exception:
                    value = None

        return self._normalize_gl_value(value)

    def _normalize_gl_value(self, value) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, bytes):
            text = value.decode("utf-8", errors="replace").strip()
            return text or None
        raw = getattr(value, "value", None)
        if isinstance(raw, bytes):
            text = raw.decode("utf-8", errors="replace").strip()
            return text or None
        text = str(value).strip()
        return text or None
