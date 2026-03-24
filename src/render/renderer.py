from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np

from ..data.volume import VolumeData
from ..utils import maybe_profile

ViewId = str
logger = logging.getLogger(__name__)
Bounds3D = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
SegmentationROI = Tuple[int, int, int, int]


@dataclass
class RenderRequest:
    view_id: ViewId
    axis: int
    slice_index: int
    zoom: float = 1.0
    pan: Tuple[float, float] = (0.0, 0.0)


@dataclass
class RenderResult:
    view_id: ViewId
    axis: int
    slice_index: int
    image: np.ndarray
    segmentation: Optional[np.ndarray] = None
    segmentation_patch: Optional[np.ndarray] = None
    segmentation_range: Optional[Tuple[int, int]] = None
    segmentation_labels: Optional[np.ndarray] = None
    segmentation_roi: Optional[SegmentationROI] = None
    level: int = 0
    level_scale: int = 1


class Renderer:
    _LEVEL_THRESHOLDS = (1000, 2000, 4000)

    def __init__(self, *, eager_statistics: bool = True) -> None:
        self._volume: Optional[VolumeData] = None
        self._volume_levels: Tuple[VolumeData, ...] = tuple()
        self._segmentation: Optional[VolumeData] = None
        self._segmentation_levels: Tuple[VolumeData, ...] = tuple()
        self._latest: Dict[ViewId, RenderResult] = {}
        self._output_handlers: Dict[ViewId, Callable[[RenderResult], None]] = {}
        self._data_range: Optional[Tuple[float, float]] = None
        self._window_range: Optional[Tuple[float, float]] = None
        self._seg_range: Optional[Tuple[int, int]] = None
        self._seg_labels: Optional[np.ndarray] = None
        self._auto_level_enabled = True
        self._manual_level = 0
        self._eager_statistics = eager_statistics

    def attach_volume(self, volume: VolumeData, levels: Optional[Tuple[VolumeData, ...]] = None) -> None:
        volume_levels = self._normalize_levels(volume, levels)
        data_range = self._compute_full_volume_data_range(volume)
        self._volume = volume
        self._volume_levels = volume_levels
        self._latest.clear()
        self._data_range = data_range
        self._window_range = data_range
        self._auto_level_enabled = True
        self._manual_level = 0

    def detach_volume(self) -> None:
        self._volume = None
        self._volume_levels = tuple()
        self._latest.clear()
        self._data_range = None
        self._window_range = None

    def attach_segmentation(
        self,
        volume: Optional[VolumeData],
        levels: Optional[Tuple[VolumeData, ...]] = None,
    ) -> None:
        self._segmentation = volume
        self._segmentation_levels = tuple() if volume is None else self._normalize_levels(volume, levels)
        self._latest.clear()
        self.set_manual_level(self._manual_level)
        if volume is None:
            self._seg_range = None
            self._seg_labels = None
            return
        if self._eager_statistics:
            self._seg_labels = self._compute_segmentation_labels(volume)
            if self._seg_labels is None or self._seg_labels.size == 0:
                self._seg_range = None
            else:
                self._seg_range = (int(self._seg_labels[0]), int(self._seg_labels[-1]))
        else:
            self._seg_labels = None
            self._seg_range = None

    def detach_segmentation(self) -> None:
        self._segmentation = None
        self._segmentation_levels = tuple()
        self._latest.clear()
        self.set_manual_level(self._manual_level)
        self._seg_range = None
        self._seg_labels = None

    def render_slice(
        self,
        view_id: ViewId,
        axis: int,
        slice_index: int,
        zoom: float = 1.0,
        pan: Tuple[float, float] = (0.0, 0.0),
    ) -> RenderResult:
        if self._volume is None:
            raise RuntimeError("No volume attached to renderer")
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2")

        base_volume = self._volume_levels[0] if self._volume_levels else self._volume
        slice_index = int(np.clip(slice_index, 0, base_volume.shape[axis] - 1))
        level = self.target_level_for_view(axis, zoom)
        level_volume = self._volume_levels[level] if self._volume_levels else self._volume
        level_scale = 1 << level
        level_slice_index = int(np.clip(slice_index // level_scale, 0, level_volume.shape[axis] - 1))
        image = level_volume.get_slice(axis, level_slice_index)
        image = self._apply_normalization(image)
        segmentation = None
        segmentation_labels = self._seg_labels
        segmentation_range = self._seg_range
        if self._segmentation is not None:
            seg_volume = self._segmentation_levels[level] if self._segmentation_levels else self._segmentation
            seg_slice_index = int(np.clip(slice_index // level_scale, 0, seg_volume.shape[axis] - 1))
            segmentation = seg_volume.get_slice(axis, seg_slice_index)
            if segmentation.shape != image.shape:
                height = min(segmentation.shape[0], image.shape[0])
                width = min(segmentation.shape[1], image.shape[1])
                segmentation = segmentation[:height, :width]
                image = image[:height, :width]
            if segmentation_labels is None:
                segmentation_labels = self._compute_slice_segmentation_labels(segmentation)
                segmentation_range = None
        result = RenderResult(
            view_id=view_id,
            axis=axis,
            slice_index=slice_index,
            image=image,
            segmentation=segmentation,
            segmentation_patch=None,
            segmentation_range=segmentation_range,
            segmentation_labels=segmentation_labels,
            level=level,
            level_scale=level_scale,
        )
        self._latest[view_id] = result
        handler = self._output_handlers.get(view_id)
        if handler is not None:
            handler(result)
        return result

    def queue_request(self, request: RenderRequest) -> RenderResult:
        return self.render_slice(
            view_id=request.view_id,
            axis=request.axis,
            slice_index=request.slice_index,
            zoom=request.zoom,
            pan=request.pan,
        )

    def refresh_segmentation_overlay(
        self,
        *,
        view_id: ViewId,
        axis: int,
        slice_index: int,
        zoom: float = 1.0,
        changed_bounds: Optional[Bounds3D] = None,
    ) -> Optional[RenderResult]:
        """Refresh only the segmentation overlay for the latest rendered image.

        Returns None when a full re-render is required (for example: missing latest
        frame, changed mip level, or missing segmentation).
        """
        with maybe_profile(
            "refresh_segmentation_overlay",
            logger=logger,
            details=f"view={view_id} axis={axis} slice={slice_index} zoom={zoom:.3f}",
        ):
            if self._volume is None or self._segmentation is None:
                return None
            if axis not in (0, 1, 2):
                return None

            latest = self._latest.get(view_id)
            if latest is None:
                return None

            base_volume = self._volume_levels[0] if self._volume_levels else self._volume
            clamped_index = int(np.clip(slice_index, 0, base_volume.shape[axis] - 1))
            target_level = self.target_level_for_view(axis, zoom)
            target_level_scale = 1 << target_level

            if (
                latest.axis != axis
                or latest.slice_index != clamped_index
                or latest.level != target_level
                or latest.level_scale != target_level_scale
            ):
                return None

            seg_volume = self._segmentation_levels[target_level] if self._segmentation_levels else self._segmentation
            seg_slice_index = int(np.clip(clamped_index // target_level_scale, 0, seg_volume.shape[axis] - 1))
            segmentation: Optional[np.ndarray] = None
            segmentation_roi: Optional[SegmentationROI] = None
            segmentation_patch: Optional[np.ndarray] = None
            patch_labels: Optional[np.ndarray] = None
            if changed_bounds is not None and latest.segmentation is not None:
                roi = self._segmentation_roi_for_bounds(
                    changed_bounds,
                    axis=axis,
                    seg_slice_index=seg_slice_index,
                    level_scale=target_level_scale,
                    segmentation_shape=latest.segmentation.shape,
                )
                if roi is None:
                    # No intersection with this slice/plane: keep latest frame unchanged.
                    return latest
                # Avoid copying full slices for small paint updates; patch in place when possible.
                segmentation = latest.segmentation
                if not segmentation.flags.writeable:
                    segmentation = np.array(segmentation, copy=True)
                patch = self._read_segmentation_roi(
                    seg_volume,
                    axis=axis,
                    seg_slice_index=seg_slice_index,
                    roi=roi,
                )
                row0, row1, col0, col1 = roi
                segmentation[row0:row1, col0:col1] = patch
                segmentation_roi = roi
                segmentation_patch = patch
                patch_labels = np.unique(patch).astype(np.int64, copy=False)
            if segmentation is None:
                segmentation = seg_volume.get_slice(axis, seg_slice_index)
            image = latest.image
            if segmentation.shape != image.shape:
                height = min(segmentation.shape[0], image.shape[0])
                width = min(segmentation.shape[1], image.shape[1])
                segmentation = segmentation[:height, :width]
                image = image[:height, :width]
                if segmentation_roi is not None:
                    old_row0, old_row1, old_col0, old_col1 = segmentation_roi
                    segmentation_roi = (
                        max(0, min(old_row0, height)),
                        max(0, min(old_row1, height)),
                        max(0, min(old_col0, width)),
                        max(0, min(old_col1, width)),
                    )
                    if segmentation_roi[0] >= segmentation_roi[1] or segmentation_roi[2] >= segmentation_roi[3]:
                        segmentation_roi = None
                        segmentation_patch = None
                    elif segmentation_patch is not None:
                        row0, row1, col0, col1 = segmentation_roi
                        patch_row0 = max(0, row0 - old_row0)
                        patch_row1 = patch_row0 + (row1 - row0)
                        patch_col0 = max(0, col0 - old_col0)
                        patch_col1 = patch_col0 + (col1 - col0)
                        segmentation_patch = np.asarray(
                            segmentation_patch[patch_row0:patch_row1, patch_col0:patch_col1]
                        )

            segmentation_labels = self._seg_labels
            segmentation_range = self._seg_range
            if segmentation_labels is None and segmentation_roi is not None:
                previous_labels = latest.segmentation_labels
                if previous_labels is not None and patch_labels is not None:
                    segmentation_labels = np.union1d(
                        previous_labels.astype(np.int64, copy=False),
                        patch_labels,
                    ).astype(np.int64, copy=False)
                    segmentation_range = None
            if segmentation_labels is None:
                segmentation_labels = self._compute_slice_segmentation_labels(segmentation)
                segmentation_range = None

            result = RenderResult(
                view_id=view_id,
                axis=axis,
                slice_index=clamped_index,
                image=image,
                segmentation=segmentation,
                segmentation_patch=segmentation_patch,
                segmentation_range=segmentation_range,
                segmentation_labels=segmentation_labels,
                segmentation_roi=segmentation_roi,
                level=target_level,
                level_scale=target_level_scale,
            )
            self._latest[view_id] = result
            handler = self._output_handlers.get(view_id)
            if handler is not None:
                handler(result)
            return result

    def _segmentation_roi_for_bounds(
        self,
        bounds: Bounds3D,
        *,
        axis: int,
        seg_slice_index: int,
        level_scale: int,
        segmentation_shape: Tuple[int, int],
    ) -> Optional[SegmentationROI]:
        (z_bounds, y_bounds, x_bounds) = bounds
        if level_scale < 1:
            level_scale = 1
        z0 = z_bounds[0] // level_scale
        z1 = (z_bounds[1] + level_scale - 1) // level_scale
        y0 = y_bounds[0] // level_scale
        y1 = (y_bounds[1] + level_scale - 1) // level_scale
        x0 = x_bounds[0] // level_scale
        x1 = (x_bounds[1] + level_scale - 1) // level_scale

        if axis == 0:
            if seg_slice_index < z0 or seg_slice_index >= z1:
                return None
            row0, row1, col0, col1 = y0, y1, x0, x1
        elif axis == 1:
            if seg_slice_index < y0 or seg_slice_index >= y1:
                return None
            row0, row1, col0, col1 = z0, z1, x0, x1
        elif axis == 2:
            if seg_slice_index < x0 or seg_slice_index >= x1:
                return None
            row0, row1, col0, col1 = z0, z1, y0, y1
        else:
            return None

        height, width = segmentation_shape
        row0 = max(0, min(int(row0), int(height)))
        row1 = max(0, min(int(row1), int(height)))
        col0 = max(0, min(int(col0), int(width)))
        col1 = max(0, min(int(col1), int(width)))
        if row0 >= row1 or col0 >= col1:
            return None
        return (row0, row1, col0, col1)

    def _read_segmentation_roi(
        self,
        seg_volume: VolumeData,
        *,
        axis: int,
        seg_slice_index: int,
        roi: SegmentationROI,
    ) -> np.ndarray:
        row0, row1, col0, col1 = roi
        if axis == 0:
            chunk = seg_volume.get_chunk(
                (
                    slice(seg_slice_index, seg_slice_index + 1),
                    slice(row0, row1),
                    slice(col0, col1),
                )
            )
            return chunk.squeeze(axis=0)
        if axis == 1:
            chunk = seg_volume.get_chunk(
                (
                    slice(row0, row1),
                    slice(seg_slice_index, seg_slice_index + 1),
                    slice(col0, col1),
                )
            )
            return chunk.squeeze(axis=1)
        chunk = seg_volume.get_chunk(
            (
                slice(row0, row1),
                slice(col0, col1),
                slice(seg_slice_index, seg_slice_index + 1),
            )
        )
        return chunk.squeeze(axis=2)

    def latest_result(self, view_id: ViewId) -> Optional[RenderResult]:
        return self._latest.get(view_id)

    def clear_cache(self) -> None:
        self._latest.clear()

    def set_output_handler(self, view_id: ViewId, handler: Callable[[RenderResult], None]) -> None:
        self._output_handlers[view_id] = handler

    def clear_output_handler(self, view_id: ViewId) -> None:
        self._output_handlers.pop(view_id, None)

    def get_data_range(self) -> Optional[Tuple[float, float]]:
        return self._data_range

    def get_window_range(self) -> Optional[Tuple[float, float]]:
        return self._window_range

    def set_window(self, vmin: float, vmax: float) -> Tuple[float, float]:
        if self._volume is None or self._data_range is None:
            raise RuntimeError("No volume attached to renderer")
        normalized_min = float(vmin)
        normalized_max = float(vmax)
        if not np.isfinite(normalized_min) or not np.isfinite(normalized_max):
            raise ValueError("Window bounds must be finite real numbers")
        if normalized_min >= normalized_max:
            raise ValueError("Window bounds must satisfy vmin < vmax")
        data_min, data_max = self._data_range
        if normalized_min < data_min or normalized_max > data_max:
            raise ValueError(
                "Window bounds must stay within the raw data range: "
                f"[{data_min}, {data_max}]"
            )
        self._window_range = (normalized_min, normalized_max)
        return self._window_range

    def reset_window(self) -> Optional[Tuple[float, float]]:
        if self._data_range is None:
            self._window_range = None
            return None
        self._window_range = self._data_range
        return self._window_range

    def is_auto_level_enabled(self) -> bool:
        return self._auto_level_enabled

    def set_auto_level_enabled(self, enabled: bool) -> None:
        self._auto_level_enabled = bool(enabled)

    def manual_level(self) -> int:
        return int(self._manual_level)

    def set_manual_level(self, level: int) -> int:
        try:
            requested = int(level)
        except (TypeError, ValueError):
            requested = 0
        max_level = max(0, self.available_level_count() - 1)
        if requested < 0:
            requested = 0
        if requested > max_level:
            requested = max_level
        self._manual_level = requested
        return self._manual_level

    def available_level_count(self) -> int:
        if self._volume is None:
            return 0
        raw_levels = len(self._volume_levels) if self._volume_levels else 1
        if self._segmentation is None:
            return max(0, raw_levels)
        seg_levels = len(self._segmentation_levels) if self._segmentation_levels else 1
        return max(0, min(raw_levels, seg_levels))

    def invalidate_segmentation_metadata(self) -> None:
        # Force per-slice label extraction on subsequent renders.
        self._seg_range = None
        self._seg_labels = None

    def set_segmentation_labels(self, labels: Iterable[int]) -> None:
        if self._segmentation is None:
            self._seg_labels = None
            self._seg_range = None
            return
        normalized = np.asarray([int(label) for label in labels], dtype=np.int64).reshape(-1)
        if normalized.size == 0:
            self._seg_labels = None
            self._seg_range = None
            return
        normalized = np.unique(normalized).astype(np.int64, copy=False)
        self._seg_labels = normalized
        self._seg_range = (int(normalized[0]), int(normalized[-1]))

    def register_segmentation_labels(self, labels: Iterable[int]) -> None:
        if self._segmentation is None or self._seg_labels is None:
            return
        additions = np.asarray([int(label) for label in labels], dtype=np.int64).reshape(-1)
        if additions.size == 0:
            return
        additions = np.unique(additions)
        merged = np.union1d(self._seg_labels, additions)
        if np.array_equal(merged, self._seg_labels):
            return
        self._seg_labels = merged.astype(np.int64, copy=False)
        if self._seg_labels.size == 0:
            self._seg_range = None
            return
        self._seg_range = (int(self._seg_labels[0]), int(self._seg_labels[-1]))

    def _compute_full_volume_data_range(self, volume: VolumeData) -> Tuple[float, float]:
        shape = tuple(int(dim) for dim in volume.shape)
        if len(shape) != 3 or any(dim <= 0 for dim in shape):
            raise ValueError(
                "Raw volume must have a strictly positive 3D shape (z, y, x), "
                f"got {shape}."
            )

        chunk_shape = self._scan_chunk_shape(shape, volume.chunk_shape)
        declared_dtype = np.dtype(volume.dtype)
        if np.issubdtype(declared_dtype, np.complexfloating):
            raise ValueError(
                "Complex-valued raw volumes are not supported for rendering."
            )
        global_min: Optional[float] = None
        global_max: Optional[float] = None

        with maybe_profile(
            "compute_full_volume_data_range",
            logger=logger,
            details=(
                f"path={volume.loader.path} shape={shape} "
                f"chunk_shape={chunk_shape} dtype={declared_dtype}"
            ),
        ):
            for zyx_slices in self._iter_chunk_slices(shape, chunk_shape):
                chunk = np.asarray(volume.loader.get_chunk(zyx_slices))
                if chunk.size == 0:
                    continue
                chunk_dtype = np.dtype(chunk.dtype)
                if np.issubdtype(chunk_dtype, np.complexfloating):
                    raise ValueError(
                        "Complex-valued raw volumes are not supported for rendering."
                    )
                if np.issubdtype(chunk_dtype, np.floating) and not np.all(np.isfinite(chunk)):
                    raise ValueError(
                        "Raw volume contains NaN or Inf values and cannot be displayed."
                    )
                local_min = float(np.min(chunk))
                local_max = float(np.max(chunk))
                if global_min is None or local_min < global_min:
                    global_min = local_min
                if global_max is None or local_max > global_max:
                    global_max = local_max

        if global_min is None or global_max is None:
            raise ValueError("Raw volume has no voxels to render.")

        return (global_min, global_max)

    def _scan_chunk_shape(
        self,
        shape: Tuple[int, int, int],
        chunk_shape: Optional[Tuple[int, int, int]],
    ) -> Tuple[int, int, int]:
        if (
            chunk_shape is not None
            and len(chunk_shape) == 3
            and all(int(dim) > 0 for dim in chunk_shape)
        ):
            return (int(chunk_shape[0]), int(chunk_shape[1]), int(chunk_shape[2]))

        max_elements = 4_000_000
        base = max(1, int(round(max_elements ** (1.0 / 3.0))))
        candidate = [max(1, min(int(dim), base)) for dim in shape]
        while candidate[0] * candidate[1] * candidate[2] > max_elements:
            largest_axis = max(range(3), key=lambda axis: candidate[axis])
            if candidate[largest_axis] <= 1:
                break
            candidate[largest_axis] = max(1, candidate[largest_axis] // 2)
        return (candidate[0], candidate[1], candidate[2])

    def _iter_chunk_slices(
        self,
        shape: Tuple[int, int, int],
        chunk_shape: Tuple[int, int, int],
    ) -> Iterable[Tuple[slice, slice, slice]]:
        for z_start in range(0, shape[0], chunk_shape[0]):
            z_stop = min(z_start + chunk_shape[0], shape[0])
            for y_start in range(0, shape[1], chunk_shape[1]):
                y_stop = min(y_start + chunk_shape[1], shape[1])
                for x_start in range(0, shape[2], chunk_shape[2]):
                    x_stop = min(x_start + chunk_shape[2], shape[2])
                    yield (slice(z_start, z_stop), slice(y_start, y_stop), slice(x_start, x_stop))

    def _apply_normalization(self, image: np.ndarray) -> np.ndarray:
        if self._window_range is None:
            return image
        vmin, vmax = self._window_range
        if vmin == vmax:
            return np.zeros_like(image, dtype=np.float32)
        scaled = (image.astype(np.float32) - vmin) / (vmax - vmin)
        return np.clip(scaled, 0.0, 1.0)

    def _compute_segmentation_labels(self, volume: VolumeData) -> Optional[np.ndarray]:
        data = volume.loader.get_chunk((slice(None), slice(None), slice(None)))
        array = np.asarray(data)
        if array.size == 0:
            return None
        labels = np.unique(array)
        if labels.size == 0:
            return None
        return labels.astype(np.int64, copy=False)

    def _compute_slice_segmentation_labels(self, segmentation: np.ndarray) -> Optional[np.ndarray]:
        array = np.asarray(segmentation)
        if array.size == 0:
            return None
        labels = np.unique(array)
        if labels.size == 0:
            return None
        return labels.astype(np.int64, copy=False)

    def _normalize_levels(
        self,
        base_volume: VolumeData,
        levels: Optional[Tuple[VolumeData, ...]],
    ) -> Tuple[VolumeData, ...]:
        if not levels:
            return (base_volume,)
        normalized = tuple(levels)
        if not normalized:
            return (base_volume,)
        if normalized[0].shape != base_volume.shape:
            return (base_volume,) + normalized
        return normalized

    def _base_slice_shape(self, axis: int) -> Tuple[int, int]:
        if self._volume is None:
            raise RuntimeError("No volume attached to renderer")
        base_shape = self._volume.shape
        if axis == 0:
            return (base_shape[1], base_shape[2])
        if axis == 1:
            return (base_shape[0], base_shape[2])
        if axis == 2:
            return (base_shape[0], base_shape[1])
        raise ValueError("axis must be 0, 1, or 2")

    def _requested_level_from_zoom(self, axis: int, zoom: float) -> int:
        del axis
        if self._volume is None:
            return 0
        base_extent = max(self._volume.shape)
        zoom = max(0.1, min(1.0, float(zoom)))
        visible_extent = int(base_extent * zoom)
        if visible_extent < self._LEVEL_THRESHOLDS[0]:
            return 0
        if visible_extent < self._LEVEL_THRESHOLDS[1]:
            return 1
        if visible_extent < self._LEVEL_THRESHOLDS[2]:
            return 2
        return 3

    def target_level_for_view(self, axis: int, zoom: float) -> int:
        if self._volume is None:
            return 0
        if not self._auto_level_enabled:
            return self.set_manual_level(self._manual_level)
        requested = self._requested_level_from_zoom(axis, zoom)
        if self._volume_levels:
            requested = min(requested, len(self._volume_levels) - 1)
        else:
            requested = 0
        level = requested
        if self._segmentation is not None:
            seg_levels = len(self._segmentation_levels) if self._segmentation_levels else 1
            level = min(level, seg_levels - 1)
        return max(0, level)
