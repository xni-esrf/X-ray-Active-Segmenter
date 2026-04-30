from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import time
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np

from ..data.volume import VolumeData, open_volume
from ..io.loader import InMemoryVolumeLoader, VoxelSpacing
from ..utils import maybe_profile


SegmentationKind = Literal["semantic", "instance"]
BrushRadius = int
Coordinate = Tuple[int, int, int]
Bounds = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EditOperation:
    operation_id: int
    name: str
    label: int
    changed_voxels: int
    bounds: Optional[Bounds] = None


@dataclass(frozen=True)
class OperationDelta:
    coordinates: np.ndarray
    previous_values: np.ndarray
    new_values: np.ndarray


@dataclass(frozen=True)
class HistoryEntry:
    operation: EditOperation
    delta: OperationDelta


@dataclass
class _PendingModification:
    name: str
    flat_index_chunks: List[np.ndarray]
    previous_value_chunks: List[np.ndarray]


@dataclass(frozen=True)
class _UndoEntry:
    operation: EditOperation
    flat_indices: np.ndarray
    previous_values: np.ndarray
    bytes_used: int
    before_state_id: int
    after_state_id: int


class SegmentationEditor:
    _MAX_UNDO_ENTRIES = 10
    _DEFAULT_MAX_UNDO_BYTES = 5 * 1024 * 1024 * 1024
    _MAX_BRUSH_RADIUS = 9
    _BRUSH_OFFSETS_CACHE: Dict[Tuple[int, int], np.ndarray] = {}

    def __init__(
        self,
        array: np.ndarray,
        *,
        kind: SegmentationKind,
        voxel_spacing: VoxelSpacing = (1.0, 1.0, 1.0),
        axes: str = "zyx",
        source_path: str = "<in-memory-segmentation>",
        active_label: Optional[int] = None,
        max_history_entries: int = _MAX_UNDO_ENTRIES,
        max_undo_bytes: int = _DEFAULT_MAX_UNDO_BYTES,
    ) -> None:
        if kind not in ("semantic", "instance"):
            raise ValueError("kind must be 'semantic' or 'instance'")

        data = np.asarray(array)
        if data.ndim != 3:
            raise ValueError("Segmentation editor expects a 3D array with shape (z, y, x)")
        if not np.issubdtype(data.dtype, np.integer):
            raise ValueError("Segmentation editor expects an integer array dtype")

        self._array = np.array(data, copy=True)
        self._kind = kind
        self._voxel_spacing = voxel_spacing
        self._axes = axes
        self._source_path = source_path
        self._dirty = False
        self._has_uncommitted_changes = False
        self._state_id = 0
        self._clean_state_id = 0
        self._operation_counter = 0
        self._max_undo_entries = max(
            0,
            min(self._MAX_UNDO_ENTRIES, int(max_history_entries)),
        )
        self._max_undo_bytes = max(0, int(max_undo_bytes))
        self._undo_stack: List[_UndoEntry] = []
        self._undo_total_bytes = 0
        self._redo_stack: List[_UndoEntry] = []
        self._redo_total_bytes = 0
        self._active_modification: Optional[_PendingModification] = None
        self._label_counts = self._build_label_counts(self._array)
        self._active_label = 1

        if active_label is None:
            self._active_label = self.next_available_label()
        else:
            self.set_active_label(active_label)

    @classmethod
    def from_volume(
        cls,
        volume: VolumeData,
        *,
        kind: SegmentationKind,
        active_label: Optional[int] = None,
        max_history_entries: int = _MAX_UNDO_ENTRIES,
        max_undo_bytes: int = _DEFAULT_MAX_UNDO_BYTES,
    ) -> "SegmentationEditor":
        array = np.asarray(volume.get_chunk((slice(None), slice(None), slice(None))))
        return cls(
            array,
            kind=kind,
            voxel_spacing=volume.info.voxel_spacing,
            axes=volume.info.axes,
            source_path=volume.loader.path,
            active_label=active_label,
            max_history_entries=max_history_entries,
            max_undo_bytes=max_undo_bytes,
        )

    @classmethod
    def create_empty(
        cls,
        shape: Tuple[int, int, int],
        *,
        kind: SegmentationKind,
        dtype: np.dtype = np.dtype(np.uint32),
        voxel_spacing: VoxelSpacing = (1.0, 1.0, 1.0),
        axes: str = "zyx",
        source_path: str = "<generated-empty-segmentation>",
        active_label: Optional[int] = None,
        max_history_entries: int = _MAX_UNDO_ENTRIES,
        max_undo_bytes: int = _DEFAULT_MAX_UNDO_BYTES,
    ) -> "SegmentationEditor":
        normalized_dtype = np.dtype(dtype)
        if not np.issubdtype(normalized_dtype, np.integer):
            raise ValueError("Empty segmentation dtype must be an integer dtype")
        if len(shape) != 3:
            raise ValueError("shape must contain exactly 3 dimensions (z, y, x)")
        array = np.zeros(shape, dtype=normalized_dtype)
        return cls(
            array,
            kind=kind,
            voxel_spacing=voxel_spacing,
            axes=axes,
            source_path=source_path,
            active_label=active_label,
            max_history_entries=max_history_entries,
            max_undo_bytes=max_undo_bytes,
        )

    @property
    def kind(self) -> SegmentationKind:
        return self._kind

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self._array.shape)

    @property
    def dtype(self) -> np.dtype:
        return self._array.dtype

    @property
    def axes(self) -> str:
        return self._axes

    @property
    def voxel_spacing(self) -> VoxelSpacing:
        return self._voxel_spacing

    @property
    def source_path(self) -> str:
        return self._source_path

    @property
    def active_label(self) -> int:
        return self._active_label

    @property
    def dirty(self) -> bool:
        return self._dirty

    @property
    def history(self) -> Tuple[HistoryEntry, ...]:
        entries: List[HistoryEntry] = []
        flat = self._array.reshape(-1)
        for undo_entry in self._undo_stack:
            coordinates = self._coordinates_from_flat_indices(undo_entry.flat_indices)
            current_values = np.asarray(
                flat[undo_entry.flat_indices],
                dtype=self._array.dtype,
            ).copy()
            current_values.setflags(write=False)
            entries.append(
                HistoryEntry(
                    operation=undo_entry.operation,
                    delta=OperationDelta(
                        coordinates=coordinates,
                        previous_values=undo_entry.previous_values,
                        new_values=current_values,
                    ),
                )
            )
        return tuple(entries)

    def mark_clean(self) -> None:
        self._clean_state_id = self._state_id
        self._has_uncommitted_changes = False
        self._update_dirty_from_state()

    def clear_history(self) -> None:
        self._undo_stack.clear()
        self._undo_total_bytes = 0
        self._redo_stack.clear()
        self._redo_total_bytes = 0

    def undo_depth(self) -> int:
        return len(self._undo_stack)

    def redo_depth(self) -> int:
        return len(self._redo_stack)

    def latest_undo_operation_id(self) -> Optional[int]:
        if not self._undo_stack:
            return None
        return int(self._undo_stack[-1].operation.operation_id)

    def latest_redo_operation_id(self) -> Optional[int]:
        if not self._redo_stack:
            return None
        return int(self._redo_stack[-1].operation.operation_id)

    def discard_undo_operation(self, operation_id: int) -> bool:
        normalized = int(operation_id)
        for index, entry in enumerate(self._undo_stack):
            if entry.operation.operation_id != normalized:
                continue
            removed = self._undo_stack.pop(index)
            self._undo_total_bytes = max(0, self._undo_total_bytes - removed.bytes_used)
            return True
        return False

    def discard_redo_operation(self, operation_id: int) -> bool:
        normalized = int(operation_id)
        for index, entry in enumerate(self._redo_stack):
            if entry.operation.operation_id != normalized:
                continue
            removed = self._redo_stack.pop(index)
            self._redo_total_bytes = max(0, self._redo_total_bytes - removed.bytes_used)
            return True
        return False

    def begin_modification(self, name: str = "modification") -> None:
        if self._active_modification is not None:
            return
        normalized = str(name).strip() or "modification"
        self._active_modification = _PendingModification(
            name=normalized,
            flat_index_chunks=[],
            previous_value_chunks=[],
        )

    def commit_modification(self) -> Optional[EditOperation]:
        pending = self._active_modification
        if pending is None:
            return None
        self._active_modification = None
        operation = self._finalize_modification(pending, store_undo=True)
        if operation is not None:
            self._clear_redo_stack()
        self._has_uncommitted_changes = False
        self._update_dirty_from_state()
        return operation

    def cancel_modification(self) -> None:
        pending = self._active_modification
        if pending is None:
            return
        self._active_modification = None
        if pending.flat_index_chunks:
            self._finalize_modification(pending, store_undo=False)
            self._clear_redo_stack()
        self._has_uncommitted_changes = False
        self._update_dirty_from_state()

    def undo_last_modification(self) -> Optional[EditOperation]:
        self.commit_modification()
        if not self._undo_stack:
            self._update_dirty_from_state()
            return None

        entry = self._undo_stack.pop()
        self._undo_total_bytes = max(0, self._undo_total_bytes - entry.bytes_used)
        flat = self._array.reshape(-1)
        current_values = np.asarray(flat[entry.flat_indices], dtype=self._array.dtype).copy()
        if current_values.size > 0:
            flat[entry.flat_indices] = entry.previous_values
            self._update_counts_after_reassignment(
                source_values=current_values,
                target_values=entry.previous_values,
            )
            current_values.setflags(write=False)
            redo_entry = _UndoEntry(
                operation=entry.operation,
                flat_indices=entry.flat_indices,
                previous_values=current_values,
                bytes_used=entry.bytes_used,
                before_state_id=entry.before_state_id,
                after_state_id=entry.after_state_id,
            )
            self._redo_stack.append(redo_entry)
            self._redo_total_bytes += redo_entry.bytes_used

        self._state_id = entry.before_state_id
        self._has_uncommitted_changes = False
        self._update_dirty_from_state()
        return entry.operation

    def redo_last_modification(self) -> Optional[EditOperation]:
        self.commit_modification()
        if not self._redo_stack:
            self._update_dirty_from_state()
            return None

        entry = self._redo_stack.pop()
        self._redo_total_bytes = max(0, self._redo_total_bytes - entry.bytes_used)
        flat = self._array.reshape(-1)
        current_values = np.asarray(flat[entry.flat_indices], dtype=self._array.dtype).copy()
        if current_values.size > 0:
            flat[entry.flat_indices] = entry.previous_values
            self._update_counts_after_reassignment(
                source_values=current_values,
                target_values=entry.previous_values,
            )
            current_values.setflags(write=False)
            undo_entry = _UndoEntry(
                operation=entry.operation,
                flat_indices=entry.flat_indices,
                previous_values=current_values,
                bytes_used=entry.bytes_used,
                before_state_id=entry.before_state_id,
                after_state_id=entry.after_state_id,
            )
            self._undo_stack.append(undo_entry)
            self._undo_total_bytes += undo_entry.bytes_used

        self._state_id = entry.after_state_id
        self._has_uncommitted_changes = False
        self._update_dirty_from_state()
        return entry.operation

    def set_active_label(self, label: int) -> int:
        normalized = int(label)
        self._validate_label(normalized)
        self._active_label = normalized
        return self._active_label

    def next_available_label(self, *, start_at: int = 1) -> int:
        candidate = max(1, int(start_at))
        max_value = self._max_supported_label()
        while candidate <= max_value:
            if self._label_counts.get(candidate, 0) == 0:
                return candidate
            candidate += 1
        raise ValueError(
            f"No available positive label fits dtype {self.dtype} (max value {max_value})"
        )

    def labels_in_use(self, *, include_background: bool = False) -> Tuple[int, ...]:
        labels = sorted(self._label_counts.keys())
        if include_background:
            return tuple(labels)
        return tuple(label for label in labels if label > 0)

    def array_view(self) -> np.ndarray:
        return self._array

    def to_volume_data(self, *, path: Optional[str] = None) -> VolumeData:
        loader = InMemoryVolumeLoader(
            path=path or self._source_path,
            array=self._array,
            voxel_spacing=self._voxel_spacing,
            axes=self._axes,
        )
        return open_volume(loader, cache=None)

    def paint_voxel(self, coordinate: Coordinate, *, label: Optional[int] = None) -> EditOperation:
        coord = self._coerce_coordinate(coordinate)
        if not self._is_in_bounds(coord):
            raise ValueError(f"Coordinate {coord} is out of bounds for shape {self.shape}")
        return self.assign(
            [coord],
            label=label,
            operation_name="paint_voxel",
            ignore_out_of_bounds=False,
        )

    def paint_stroke(
        self,
        coordinates: Sequence[Coordinate],
        *,
        label: Optional[int] = None,
    ) -> EditOperation:
        if not coordinates:
            return self._create_operation(
                name="paint_stroke",
                label=self._resolve_label(label),
                changed_voxels=0,
                bounds=None,
            )
        rasterized: List[Coordinate] = []
        previous = self._coerce_coordinate(coordinates[0])
        rasterized.append(previous)
        for raw_coord in coordinates[1:]:
            current = self._coerce_coordinate(raw_coord)
            rasterized.extend(self._rasterize_line(previous, current))
            previous = current
        return self.assign(
            rasterized,
            label=label,
            operation_name="paint_stroke",
            ignore_out_of_bounds=True,
        )

    def paint_brush_voxel(
        self,
        coordinate: Coordinate,
        *,
        axis: int,
        brush_radius: BrushRadius = 0,
        label: Optional[int] = None,
    ) -> EditOperation:
        center = self._coerce_coordinate(coordinate)
        if not self._is_in_bounds(center):
            raise ValueError(f"Coordinate {center} is out of bounds for shape {self.shape}")
        radius = self._normalize_brush_radius(brush_radius)
        coords = self._brush_coordinates_array(center, axis=axis, radius=radius)
        return self.assign(
            coords,
            label=label,
            operation_name="paint_brush_voxel",
            ignore_out_of_bounds=True,
        )

    def paint_brush_stroke(
        self,
        coordinates: Sequence[Coordinate],
        *,
        axis: int,
        brush_radius: BrushRadius = 0,
        label: Optional[int] = None,
    ) -> EditOperation:
        with maybe_profile(
            "paint_brush_stroke",
            logger=logger,
            details=f"points={len(coordinates)} axis={axis} brush_radius={brush_radius}",
        ):
            if not coordinates:
                return self._create_operation(
                    name="paint_brush_stroke",
                    label=self._resolve_label(label),
                    changed_voxels=0,
                    bounds=None,
                )

            radius = self._normalize_brush_radius(brush_radius)
            centers = self._stroke_centers(coordinates)
            offsets = self._brush_offsets(axis=axis, radius=radius)
            rasterized = (centers[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
            return self.assign(
                rasterized,
                label=label,
                operation_name="paint_brush_stroke",
                ignore_out_of_bounds=True,
            )

    def erase_brush_voxel(
        self,
        coordinate: Coordinate,
        *,
        axis: int,
        brush_radius: BrushRadius = 0,
        target_label: Optional[int] = None,
    ) -> EditOperation:
        center = self._coerce_coordinate(coordinate)
        if not self._is_in_bounds(center):
            raise ValueError(f"Coordinate {center} is out of bounds for shape {self.shape}")
        radius = self._normalize_brush_radius(brush_radius)
        coords = self._brush_coordinates_array(center, axis=axis, radius=radius)
        return self.erase(
            coords,
            target_label=target_label,
            operation_name="erase_brush_voxel",
            ignore_out_of_bounds=True,
        )

    def erase_brush_stroke(
        self,
        coordinates: Sequence[Coordinate],
        *,
        axis: int,
        brush_radius: BrushRadius = 0,
        target_label: Optional[int] = None,
    ) -> EditOperation:
        with maybe_profile(
            "erase_brush_stroke",
            logger=logger,
            details=f"points={len(coordinates)} axis={axis} brush_radius={brush_radius}",
        ):
            if not coordinates:
                return self._create_operation(
                    name="erase_brush_stroke",
                    label=0,
                    changed_voxels=0,
                    bounds=None,
                )

            radius = self._normalize_brush_radius(brush_radius)
            centers = self._stroke_centers(coordinates)
            offsets = self._brush_offsets(axis=axis, radius=radius)
            rasterized = (centers[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
            return self.erase(
                rasterized,
                target_label=target_label,
                operation_name="erase_brush_stroke",
                ignore_out_of_bounds=True,
            )

    def erase(
        self,
        coordinates: Sequence[Coordinate],
        *,
        target_label: Optional[int] = None,
        operation_name: str = "erase",
        ignore_out_of_bounds: bool = True,
    ) -> EditOperation:
        normalized_target: Optional[int] = None
        if target_label is not None:
            normalized_target = int(target_label)
            self._validate_label(normalized_target)

        coords_array = self._normalize_coordinates(
            coordinates,
            ignore_out_of_bounds=ignore_out_of_bounds,
        )
        if coords_array.size == 0:
            return self._create_operation(
                name=operation_name,
                label=0,
                changed_voxels=0,
                bounds=None,
            )

        coords_array = self._deduplicate_coordinates(coords_array)
        z_idx = coords_array[:, 0]
        y_idx = coords_array[:, 1]
        x_idx = coords_array[:, 2]
        old_values = self._array[z_idx, y_idx, x_idx]

        if normalized_target is None:
            changed_mask = old_values != 0
        else:
            changed_mask = (old_values == normalized_target) & (old_values != 0)

        if not np.any(changed_mask):
            return self._create_operation(
                name=operation_name,
                label=0,
                changed_voxels=0,
                bounds=None,
            )

        changed_coords = coords_array[changed_mask]
        changed_old = old_values[changed_mask].astype(self._array.dtype, copy=True)
        changed_count = int(changed_coords.shape[0])
        self._array[
            changed_coords[:, 0],
            changed_coords[:, 1],
            changed_coords[:, 2],
        ] = 0
        self._update_counts_after_assignment(changed_old, 0)
        self._mark_uncommitted_change()

        bounds = self._bounds_from_coordinates(changed_coords)
        operation = self._create_operation(
            name=operation_name,
            label=0,
            changed_voxels=changed_count,
            bounds=bounds,
        )
        self._record_modification_delta(
            name=operation_name,
            coordinates=changed_coords,
            previous_values=changed_old,
        )
        return operation

    def erase_masked_region(
        self,
        *,
        z_bounds: Tuple[int, int],
        y_bounds: Tuple[int, int],
        x_bounds: Tuple[int, int],
        region_mask: np.ndarray,
        target_label: Optional[int] = None,
        operation_name: str = "erase_masked_region",
    ) -> EditOperation:
        normalized_target: Optional[int] = None
        if target_label is not None:
            normalized_target = int(target_label)
            self._validate_label(normalized_target)

        z0, z1 = int(z_bounds[0]), int(z_bounds[1])
        y0, y1 = int(y_bounds[0]), int(y_bounds[1])
        x0, x1 = int(x_bounds[0]), int(x_bounds[1])
        if not (0 <= z0 <= z1 <= self.shape[0]):
            raise ValueError(f"z_bounds {z_bounds} are out of bounds for shape {self.shape}")
        if not (0 <= y0 <= y1 <= self.shape[1]):
            raise ValueError(f"y_bounds {y_bounds} are out of bounds for shape {self.shape}")
        if not (0 <= x0 <= x1 <= self.shape[2]):
            raise ValueError(f"x_bounds {x_bounds} are out of bounds for shape {self.shape}")

        region_shape = (z1 - z0, y1 - y0, x1 - x0)
        mask_array = np.asarray(region_mask, dtype=bool)
        if mask_array.shape != region_shape:
            raise ValueError(
                f"region_mask shape {mask_array.shape} does not match region shape {region_shape}"
            )
        if mask_array.size == 0 or not np.any(mask_array):
            return self._create_operation(
                name=operation_name,
                label=0,
                changed_voxels=0,
                bounds=None,
            )

        subarray = self._array[z0:z1, y0:y1, x0:x1]
        subarray_flat = subarray.ravel(order="C")
        mask_flat = mask_array.reshape(-1)
        masked_local_flat_indices = np.flatnonzero(mask_flat).astype(np.int64, copy=False)
        if masked_local_flat_indices.size == 0:
            return self._create_operation(
                name=operation_name,
                label=0,
                changed_voxels=0,
                bounds=None,
            )

        masked_old_values = subarray_flat[masked_local_flat_indices]
        if normalized_target is None:
            changed_mask = masked_old_values != 0
        else:
            changed_mask = (masked_old_values == normalized_target) & (masked_old_values != 0)
        if not np.any(changed_mask):
            return self._create_operation(
                name=operation_name,
                label=0,
                changed_voxels=0,
                bounds=None,
            )

        changed_local_flat_indices = masked_local_flat_indices[changed_mask]
        local_y_size = int(region_shape[1])
        local_x_size = int(region_shape[2])
        local_plane = int(local_y_size * local_x_size)
        local_z = changed_local_flat_indices // local_plane
        local_remainder = changed_local_flat_indices - (local_z * local_plane)
        local_y = local_remainder // local_x_size
        local_x = local_remainder - (local_y * local_x_size)
        changed_old = masked_old_values[changed_mask].astype(self._array.dtype, copy=True)
        changed_count = int(changed_local_flat_indices.shape[0])
        subarray[local_z, local_y, local_x] = 0
        self._update_counts_after_assignment(changed_old, 0)
        self._mark_uncommitted_change()

        global_z = local_z + int(z0)
        global_y = local_y + int(y0)
        global_x = local_x + int(x0)
        global_x_size = int(self.shape[2])
        global_plane = int(self.shape[1] * global_x_size)
        global_flat_indices = (
            (global_z * global_plane)
            + (global_y * global_x_size)
            + global_x
        ).astype(np.int64, copy=False)

        bounds = self._bounds_from_flat_indices(global_flat_indices)
        operation = self._create_operation(
            name=operation_name,
            label=0,
            changed_voxels=changed_count,
            bounds=bounds,
        )
        self._record_modification_flat_delta(
            name=operation_name,
            flat_indices=global_flat_indices,
            previous_values=changed_old,
        )
        return operation

    def assign(
        self,
        coordinates: Sequence[Coordinate],
        *,
        label: Optional[int] = None,
        operation_name: str = "assign",
        ignore_out_of_bounds: bool = True,
    ) -> EditOperation:
        target_label = self._resolve_label(label)
        coords_array = self._normalize_coordinates(
            coordinates,
            ignore_out_of_bounds=ignore_out_of_bounds,
        )
        if coords_array.size == 0:
            return self._create_operation(
                name=operation_name,
                label=target_label,
                changed_voxels=0,
                bounds=None,
            )

        coords_array = self._deduplicate_coordinates(coords_array)
        z_idx = coords_array[:, 0]
        y_idx = coords_array[:, 1]
        x_idx = coords_array[:, 2]
        old_values = self._array[z_idx, y_idx, x_idx]
        changed_mask = old_values != target_label

        if not np.any(changed_mask):
            return self._create_operation(
                name=operation_name,
                label=target_label,
                changed_voxels=0,
                bounds=None,
            )

        changed_coords = coords_array[changed_mask]
        changed_old = old_values[changed_mask].astype(self._array.dtype, copy=True)
        changed_count = int(changed_coords.shape[0])
        self._array[
            changed_coords[:, 0],
            changed_coords[:, 1],
            changed_coords[:, 2],
        ] = target_label
        self._update_counts_after_assignment(changed_old, target_label)
        self._mark_uncommitted_change()

        bounds = self._bounds_from_coordinates(changed_coords)
        operation = self._create_operation(
            name=operation_name,
            label=target_label,
            changed_voxels=changed_count,
            bounds=bounds,
        )
        self._record_modification_delta(
            name=operation_name,
            coordinates=changed_coords,
            previous_values=changed_old,
        )
        return operation

    def flood_fill_from_seed(
        self,
        coordinate: Coordinate,
        *,
        label: Optional[int] = None,
        connectivity: int = 6,
        max_duration_seconds: Optional[float] = None,
    ) -> EditOperation:
        seed = self._coerce_coordinate(coordinate)
        if not self._is_in_bounds(seed):
            raise ValueError(f"Coordinate {seed} is out of bounds for shape {self.shape}")
        if connectivity != 6:
            raise ValueError("Only 6-connectivity flood fill is currently supported")

        timeout_seconds: Optional[float] = None
        timeout_deadline: Optional[float] = None
        if max_duration_seconds is not None:
            timeout_seconds = float(max_duration_seconds)
            if not np.isfinite(timeout_seconds):
                raise ValueError("Flood fill timeout must be a finite number of seconds.")
            if timeout_seconds < 0.0:
                raise ValueError("Flood fill timeout must be >= 0 seconds.")
            if timeout_seconds == 0.0:
                raise ValueError("Flood fill exceeded time limit (0.00s) and was canceled.")
            timeout_deadline = time.perf_counter() + timeout_seconds

        target_label = self._resolve_label(label)
        source_label = int(self._array[seed])
        if source_label == target_label:
            return self._create_operation(
                name="flood_fill",
                label=target_label,
                changed_voxels=0,
                bounds=None,
            )

        z_max, y_max, x_max = self.shape
        queue = deque([seed])
        changed_coords: List[Coordinate] = []
        source_value = self._array.dtype.type(source_label)
        target_value = self._array.dtype.type(target_label)
        timeout_check_interval = 2048
        steps_since_timeout_check = 0

        while queue:
            if timeout_deadline is not None:
                steps_since_timeout_check += 1
                if steps_since_timeout_check >= timeout_check_interval:
                    steps_since_timeout_check = 0
                    if time.perf_counter() >= timeout_deadline:
                        if changed_coords:
                            changed_array = np.asarray(changed_coords, dtype=np.int64)
                            self._array[
                                changed_array[:, 0],
                                changed_array[:, 1],
                                changed_array[:, 2],
                            ] = source_value
                        raise ValueError(
                            f"Flood fill exceeded time limit ({timeout_seconds:.2f}s) and was canceled."
                        )
            z, y, x = queue.popleft()
            if self._array[z, y, x] != source_value:
                continue

            self._array[z, y, x] = target_value
            changed_coords.append((z, y, x))

            if z > 0:
                queue.append((z - 1, y, x))
            if z + 1 < z_max:
                queue.append((z + 1, y, x))
            if y > 0:
                queue.append((z, y - 1, x))
            if y + 1 < y_max:
                queue.append((z, y + 1, x))
            if x > 0:
                queue.append((z, y, x - 1))
            if x + 1 < x_max:
                queue.append((z, y, x + 1))

        if not changed_coords:
            return self._create_operation(
                name="flood_fill",
                label=target_label,
                changed_voxels=0,
                bounds=None,
            )

        changed_array = np.asarray(changed_coords, dtype=np.int64)
        changed_count = int(changed_array.shape[0])
        changed_old = np.full((changed_count,), source_value, dtype=self._array.dtype)
        self._update_counts_after_assignment(changed_old, target_label)
        self._mark_uncommitted_change()

        bounds = self._bounds_from_coordinates(changed_array)
        operation = self._create_operation(
            name="flood_fill",
            label=target_label,
            changed_voxels=changed_count,
            bounds=bounds,
        )
        self._record_modification_delta(
            name="flood_fill",
            coordinates=changed_array,
            previous_values=changed_old,
        )
        return operation

    def merge_labels(
        self,
        source_labels: Iterable[int],
        *,
        target_label: Optional[int] = None,
    ) -> EditOperation:
        resolved_target = self._resolve_label(target_label)
        source_set = {int(label) for label in source_labels}
        source_set.discard(resolved_target)
        if not source_set:
            return self._create_operation(
                name="merge_labels",
                label=resolved_target,
                changed_voxels=0,
                bounds=None,
            )

        mask = np.isin(self._array, np.array(sorted(source_set), dtype=np.int64))
        coords = np.argwhere(mask)
        if coords.size == 0:
            return self._create_operation(
                name="merge_labels",
                label=resolved_target,
                changed_voxels=0,
                bounds=None,
            )

        coord_tuples: List[Coordinate] = [
            (int(item[0]), int(item[1]), int(item[2])) for item in coords
        ]
        return self.assign(
            coord_tuples,
            label=resolved_target,
            operation_name="merge_labels",
            ignore_out_of_bounds=True,
        )

    def split_label(self, _label: int) -> None:
        raise NotImplementedError("split_label is planned but not implemented yet")

    def _resolve_label(self, label: Optional[int]) -> int:
        if label is None:
            return self._active_label
        normalized = int(label)
        self._validate_label(normalized)
        return normalized

    def _validate_label(self, label: int) -> None:
        if label < 0:
            raise ValueError("Labels must be >= 0")
        max_value = self._max_supported_label()
        if label > max_value:
            raise ValueError(
                f"Label {label} cannot be represented with dtype {self.dtype} (max {max_value})"
            )

    def _max_supported_label(self) -> int:
        info = np.iinfo(self._array.dtype)
        return int(info.max)

    def _coerce_coordinate(self, coordinate: Coordinate) -> Coordinate:
        if len(coordinate) != 3:
            raise ValueError("Coordinate must contain exactly 3 values (z, y, x)")
        return (int(coordinate[0]), int(coordinate[1]), int(coordinate[2]))

    def _is_in_bounds(self, coord: Coordinate) -> bool:
        return (
            0 <= coord[0] < self.shape[0]
            and 0 <= coord[1] < self.shape[1]
            and 0 <= coord[2] < self.shape[2]
        )

    def _normalize_coordinates(
        self,
        coordinates: Sequence[Coordinate],
        *,
        ignore_out_of_bounds: bool,
    ) -> np.ndarray:
        if len(coordinates) == 0:
            return np.empty((0, 3), dtype=np.int64)

        data = np.asarray(coordinates, dtype=np.int64)
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError("Coordinates must be a sequence of (z, y, x) triples")

        in_bounds_mask = (
            (data[:, 0] >= 0)
            & (data[:, 0] < self.shape[0])
            & (data[:, 1] >= 0)
            & (data[:, 1] < self.shape[1])
            & (data[:, 2] >= 0)
            & (data[:, 2] < self.shape[2])
        )
        if np.all(in_bounds_mask):
            return data

        if not ignore_out_of_bounds:
            first_invalid = data[np.logical_not(in_bounds_mask)][0]
            coord = (int(first_invalid[0]), int(first_invalid[1]), int(first_invalid[2]))
            raise ValueError(f"Coordinate {coord} is out of bounds for shape {self.shape}")

        return data[in_bounds_mask]

    def _deduplicate_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        if coordinates.shape[0] <= 1:
            return coordinates
        contiguous = np.ascontiguousarray(coordinates, dtype=np.int64)
        packed = contiguous.view(
            dtype=[
                ("z", np.int64),
                ("y", np.int64),
                ("x", np.int64),
            ]
        ).reshape(-1)
        _, first_indices = np.unique(packed, return_index=True)
        if first_indices.size == contiguous.shape[0]:
            return contiguous
        return contiguous[np.sort(first_indices)]

    def _rasterize_line(self, start: Coordinate, end: Coordinate) -> List[Coordinate]:
        dz = end[0] - start[0]
        dy = end[1] - start[1]
        dx = end[2] - start[2]
        steps = max(abs(dz), abs(dy), abs(dx))
        if steps == 0:
            return [start]
        points: List[Coordinate] = []
        for index in range(steps + 1):
            ratio = float(index) / float(steps)
            z = int(round(start[0] + dz * ratio))
            y = int(round(start[1] + dy * ratio))
            x = int(round(start[2] + dx * ratio))
            points.append((z, y, x))
        return points

    def _normalize_brush_radius(self, brush_radius: BrushRadius) -> int:
        radius = int(brush_radius)
        if radius < 0 or radius > self._MAX_BRUSH_RADIUS:
            raise ValueError(
                f"Brush radius must be between 0 and {self._MAX_BRUSH_RADIUS} (got {radius})."
            )
        return radius

    def _brush_coordinates(
        self,
        center: Coordinate,
        *,
        axis: int,
        radius: int,
    ) -> List[Coordinate]:
        coords = self._brush_coordinates_array(center, axis=axis, radius=radius)
        return [
            (int(coord[0]), int(coord[1]), int(coord[2]))
            for coord in coords
        ]

    def _brush_coordinates_array(
        self,
        center: Coordinate,
        *,
        axis: int,
        radius: int,
    ) -> np.ndarray:
        base = np.asarray(self._coerce_coordinate(center), dtype=np.int64)
        offsets = self._brush_offsets(axis=axis, radius=radius)
        return base[None, :] + offsets

    def _brush_offsets(self, *, axis: int, radius: int) -> np.ndarray:
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2")
        radius = max(0, int(radius))
        cache_key = (axis, radius)
        cached = self._BRUSH_OFFSETS_CACHE.get(cache_key)
        if cached is not None:
            return cached

        if radius == 0:
            offsets = np.zeros((1, 3), dtype=np.int64)
            offsets.setflags(write=False)
            self._BRUSH_OFFSETS_CACHE[cache_key] = offsets
            return offsets

        grid = np.arange(-radius, radius + 1, dtype=np.int64)
        dz, dy, dx = np.meshgrid(grid, grid, grid, indexing="ij")
        mask = (dz * dz) + (dy * dy) + (dx * dx) <= (radius * radius)
        offsets = np.stack((dz[mask], dy[mask], dx[mask]), axis=1).astype(
            np.int64,
            copy=False,
        )
        offsets.setflags(write=False)
        self._BRUSH_OFFSETS_CACHE[cache_key] = offsets
        return offsets

    def _stroke_centers(self, coordinates: Sequence[Coordinate]) -> np.ndarray:
        if len(coordinates) == 0:
            return np.empty((0, 3), dtype=np.int64)
        normalized = np.asarray(
            [self._coerce_coordinate(coordinate) for coordinate in coordinates],
            dtype=np.int64,
        )
        if normalized.shape[0] == 1:
            return normalized

        chunks: List[np.ndarray] = [normalized[:1]]
        previous = normalized[0]
        for current in normalized[1:]:
            chunks.append(self._rasterize_line_array(previous, current))
            previous = current
        return np.concatenate(chunks, axis=0)

    def _rasterize_line_array(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        delta = end - start
        steps = int(np.max(np.abs(delta)))
        if steps <= 0:
            return start.reshape(1, 3).astype(np.int64, copy=False)
        ratios = np.linspace(0.0, 1.0, steps + 1, dtype=np.float64)[:, None]
        points = start[None, :] + (ratios * delta[None, :])
        return np.rint(points).astype(np.int64, copy=False)

    def _create_operation(
        self,
        *,
        name: str,
        label: int,
        changed_voxels: int,
        bounds: Optional[Bounds],
    ) -> EditOperation:
        self._operation_counter += 1
        return EditOperation(
            operation_id=self._operation_counter,
            name=name,
            label=label,
            changed_voxels=changed_voxels,
            bounds=bounds,
        )

    def _mark_uncommitted_change(self) -> None:
        self._has_uncommitted_changes = True
        self._update_dirty_from_state()

    def _record_modification_delta(
        self,
        *,
        name: str,
        coordinates: np.ndarray,
        previous_values: np.ndarray,
    ) -> None:
        if coordinates.size == 0 or previous_values.size == 0:
            return

        indices = np.ravel_multi_index(
            (
                coordinates[:, 0],
                coordinates[:, 1],
                coordinates[:, 2],
            ),
            self.shape,
        ).astype(np.int64, copy=False)
        self._record_modification_flat_delta(
            name=name,
            flat_indices=indices,
            previous_values=previous_values,
        )

    def _record_modification_flat_delta(
        self,
        *,
        name: str,
        flat_indices: np.ndarray,
        previous_values: np.ndarray,
    ) -> None:
        indices_copy = np.asarray(flat_indices, dtype=np.int64).reshape(-1).copy()
        values_copy = np.asarray(previous_values, dtype=self._array.dtype).reshape(-1).copy()
        if indices_copy.size == 0 or values_copy.size == 0:
            return
        if indices_copy.shape[0] != values_copy.shape[0]:
            raise ValueError("flat_indices and previous_values must have the same length")
        indices_copy.setflags(write=False)
        values_copy.setflags(write=False)

        auto_commit = False
        if self._active_modification is None:
            self.begin_modification(name)
            auto_commit = True
        pending = self._active_modification
        if pending is None:
            return

        pending.flat_index_chunks.append(indices_copy)
        pending.previous_value_chunks.append(values_copy)
        self._has_uncommitted_changes = True
        self._update_dirty_from_state()

        if auto_commit:
            self.commit_modification()

    def _finalize_modification(
        self,
        pending: _PendingModification,
        *,
        store_undo: bool,
    ) -> Optional[EditOperation]:
        if not pending.flat_index_chunks:
            return None

        if len(pending.flat_index_chunks) == 1:
            flat_indices = pending.flat_index_chunks[0]
            previous_values = pending.previous_value_chunks[0]
        else:
            flat_indices = np.concatenate(pending.flat_index_chunks, axis=0)
            previous_values = np.concatenate(pending.previous_value_chunks, axis=0)
            _, first_indices = np.unique(flat_indices, return_index=True)
            if first_indices.size < flat_indices.size:
                keep = np.sort(first_indices)
                flat_indices = flat_indices[keep]
                previous_values = previous_values[keep]

        flat = self._array.reshape(-1)
        current_values = np.asarray(flat[flat_indices], dtype=self._array.dtype).copy()
        changed_mask = current_values != previous_values
        if not np.any(changed_mask):
            return None

        kept_indices = flat_indices[changed_mask].copy()
        kept_previous = previous_values[changed_mask].copy()
        kept_indices.setflags(write=False)
        kept_previous.setflags(write=False)

        before_state_id = self._state_id
        self._state_id += 1
        after_state_id = self._state_id
        bounds = self._bounds_from_flat_indices(kept_indices)
        final_values = np.asarray(flat[kept_indices], dtype=self._array.dtype)
        if final_values.size > 0 and np.all(final_values == final_values[0]):
            operation_label = int(final_values[0])
        else:
            operation_label = self._active_label
        operation = self._create_operation(
            name=pending.name,
            label=operation_label,
            changed_voxels=int(kept_indices.size),
            bounds=bounds,
        )

        if store_undo:
            bytes_used = int(kept_indices.nbytes + kept_previous.nbytes)
            if self._max_undo_entries <= 0 or self._max_undo_bytes <= 0:
                self._undo_stack.clear()
                self._undo_total_bytes = 0
                self._clear_redo_stack()
            elif bytes_used <= self._max_undo_bytes:
                entry = _UndoEntry(
                    operation=operation,
                    flat_indices=kept_indices,
                    previous_values=kept_previous,
                    bytes_used=bytes_used,
                    before_state_id=before_state_id,
                    after_state_id=after_state_id,
                )
                self._undo_stack.append(entry)
                self._undo_total_bytes += bytes_used
                self._trim_undo_stack()

        return operation

    def _trim_undo_stack(self) -> None:
        while self._undo_stack and (
            len(self._undo_stack) > self._max_undo_entries
            or self._undo_total_bytes > self._max_undo_bytes
        ):
            dropped = self._undo_stack.pop(0)
            self._undo_total_bytes = max(0, self._undo_total_bytes - dropped.bytes_used)

    def _clear_redo_stack(self) -> None:
        self._redo_stack.clear()
        self._redo_total_bytes = 0

    def _update_dirty_from_state(self) -> None:
        self._dirty = self._has_uncommitted_changes or (self._state_id != self._clean_state_id)

    def _coordinates_from_flat_indices(self, flat_indices: np.ndarray) -> np.ndarray:
        if flat_indices.size == 0:
            return np.empty((0, 3), dtype=np.int64)
        y_size = self.shape[1]
        x_size = self.shape[2]
        plane = y_size * x_size
        z = flat_indices // plane
        remainder = flat_indices - (z * plane)
        y = remainder // x_size
        x = remainder - (y * x_size)
        coords = np.stack((z, y, x), axis=1).astype(np.int64, copy=False)
        coords.setflags(write=False)
        return coords

    def _bounds_from_flat_indices(self, flat_indices: np.ndarray) -> Optional[Bounds]:
        if flat_indices.size == 0:
            return None
        y_size = self.shape[1]
        x_size = self.shape[2]
        plane = y_size * x_size
        z = flat_indices // plane
        remainder = flat_indices - (z * plane)
        y = remainder // x_size
        x = remainder - (y * x_size)
        return (
            (int(np.min(z)), int(np.max(z)) + 1),
            (int(np.min(y)), int(np.max(y)) + 1),
            (int(np.min(x)), int(np.max(x)) + 1),
        )

    def _build_label_counts(self, array: np.ndarray) -> Dict[int, int]:
        if array.size == 0:
            return {}
        labels, counts = np.unique(array, return_counts=True)
        return {int(label): int(count) for label, count in zip(labels, counts)}

    def _update_counts_after_assignment(self, previous_values: np.ndarray, target_label: int) -> None:
        if previous_values.size == 0:
            return
        old_labels, old_counts = np.unique(previous_values, return_counts=True)
        for label, count in zip(old_labels, old_counts):
            normalized = int(label)
            remaining = self._label_counts.get(normalized, 0) - int(count)
            if remaining > 0:
                self._label_counts[normalized] = remaining
            else:
                self._label_counts.pop(normalized, None)
        self._label_counts[target_label] = self._label_counts.get(target_label, 0) + int(
            previous_values.size
        )

    def _update_counts_after_reassignment(
        self,
        *,
        source_values: np.ndarray,
        target_values: np.ndarray,
    ) -> None:
        if source_values.size == 0:
            return

        source_labels, source_counts = np.unique(source_values, return_counts=True)
        for label, count in zip(source_labels, source_counts):
            normalized = int(label)
            remaining = self._label_counts.get(normalized, 0) - int(count)
            if remaining > 0:
                self._label_counts[normalized] = remaining
            else:
                self._label_counts.pop(normalized, None)

        target_labels, target_counts = np.unique(target_values, return_counts=True)
        for label, count in zip(target_labels, target_counts):
            normalized = int(label)
            self._label_counts[normalized] = self._label_counts.get(normalized, 0) + int(count)

    def _bounds_from_coordinates(self, coordinates: np.ndarray) -> Optional[Bounds]:
        if coordinates.size == 0:
            return None
        z_min = int(np.min(coordinates[:, 0]))
        z_max = int(np.max(coordinates[:, 0])) + 1
        y_min = int(np.min(coordinates[:, 1]))
        y_max = int(np.max(coordinates[:, 1])) + 1
        x_min = int(np.min(coordinates[:, 2]))
        x_max = int(np.max(coordinates[:, 2])) + 1
        return ((z_min, z_max), (y_min, y_max), (x_min, x_max))
