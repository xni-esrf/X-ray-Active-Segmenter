from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence, Tuple

import numpy as np

from ..bbox.model import BoundingBox

BBoxSegmentationOperation = Literal["median_filter", "erosion", "dilation"]


@dataclass(frozen=True)
class BBoxUnionDomain:
    z_bounds: Tuple[int, int]
    y_bounds: Tuple[int, int]
    x_bounds: Tuple[int, int]
    union_mask: np.ndarray

    def as_tuple(
        self,
    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], np.ndarray]:
        return (self.z_bounds, self.y_bounds, self.x_bounds, self.union_mask)

    def __iter__(self):
        return iter(self.as_tuple())


@dataclass(frozen=True)
class BBoxProcessingRegions:
    core_z_bounds: Tuple[int, int]
    core_y_bounds: Tuple[int, int]
    core_x_bounds: Tuple[int, int]
    union_mask: np.ndarray
    extended_z_bounds: Tuple[int, int]
    extended_y_bounds: Tuple[int, int]
    extended_x_bounds: Tuple[int, int]

    def as_tuple(
        self,
    ) -> Tuple[
        Tuple[int, int],
        Tuple[int, int],
        Tuple[int, int],
        np.ndarray,
        Tuple[int, int],
        Tuple[int, int],
        Tuple[int, int],
    ]:
        return (
            self.core_z_bounds,
            self.core_y_bounds,
            self.core_x_bounds,
            self.union_mask,
            self.extended_z_bounds,
            self.extended_y_bounds,
            self.extended_x_bounds,
        )

    def __iter__(self):
        return iter(self.as_tuple())


def build_selected_bbox_union_domain(
    boxes: Sequence[BoundingBox],
) -> BBoxUnionDomain:
    normalized_boxes = tuple(boxes)
    if not normalized_boxes:
        raise ValueError("At least one selected bounding box is required.")

    z_bounds = (
        min(int(box.z0) for box in normalized_boxes),
        max(int(box.z1) for box in normalized_boxes),
    )
    y_bounds = (
        min(int(box.y0) for box in normalized_boxes),
        max(int(box.y1) for box in normalized_boxes),
    )
    x_bounds = (
        min(int(box.x0) for box in normalized_boxes),
        max(int(box.x1) for box in normalized_boxes),
    )

    union_mask = np.zeros(
        (
            int(z_bounds[1] - z_bounds[0]),
            int(y_bounds[1] - y_bounds[0]),
            int(x_bounds[1] - x_bounds[0]),
        ),
        dtype=bool,
    )
    for box in normalized_boxes:
        union_mask[
            int(box.z0 - z_bounds[0]) : int(box.z1 - z_bounds[0]),
            int(box.y0 - y_bounds[0]) : int(box.y1 - y_bounds[0]),
            int(box.x0 - x_bounds[0]) : int(box.x1 - x_bounds[0]),
        ] = True

    return BBoxUnionDomain(
        z_bounds=z_bounds,
        y_bounds=y_bounds,
        x_bounds=x_bounds,
        union_mask=union_mask,
    )


def expand_axis_bounds_with_halo(
    bounds: Tuple[int, int],
    *,
    axis_length: int,
    halo_size: int,
) -> Tuple[int, int]:
    start = int(bounds[0])
    end = int(bounds[1])
    normalized_axis_length = int(axis_length)
    normalized_halo_size = int(halo_size)

    if normalized_axis_length <= 0:
        raise ValueError(
            "axis_length must be positive when expanding bounds with halo."
        )
    if start < 0 or end <= start or end > normalized_axis_length:
        raise ValueError(
            "bounds must satisfy 0 <= start < end <= axis_length: "
            f"bounds=({start}, {end}) axis_length={normalized_axis_length}"
        )
    if normalized_halo_size < 0:
        raise ValueError("halo_size must be >= 0")

    return (
        max(0, start - normalized_halo_size),
        min(normalized_axis_length, end + normalized_halo_size),
    )


def build_selected_bbox_processing_regions(
    boxes: Sequence[BoundingBox],
    *,
    volume_shape: Sequence[int],
    halo_size: int = 1,
) -> BBoxProcessingRegions:
    normalized_volume_shape = tuple(int(dim) for dim in tuple(volume_shape))
    if len(normalized_volume_shape) != 3:
        raise ValueError(
            "volume_shape must be a 3D shape (z, y, x), "
            f"got {normalized_volume_shape!r}."
        )
    if any(dim <= 0 for dim in normalized_volume_shape):
        raise ValueError(
            "volume_shape dimensions must be positive, "
            f"got {normalized_volume_shape!r}."
        )

    union_domain = build_selected_bbox_union_domain(boxes)
    extended_z_bounds = expand_axis_bounds_with_halo(
        union_domain.z_bounds,
        axis_length=normalized_volume_shape[0],
        halo_size=halo_size,
    )
    extended_y_bounds = expand_axis_bounds_with_halo(
        union_domain.y_bounds,
        axis_length=normalized_volume_shape[1],
        halo_size=halo_size,
    )
    extended_x_bounds = expand_axis_bounds_with_halo(
        union_domain.x_bounds,
        axis_length=normalized_volume_shape[2],
        halo_size=halo_size,
    )

    return BBoxProcessingRegions(
        core_z_bounds=union_domain.z_bounds,
        core_y_bounds=union_domain.y_bounds,
        core_x_bounds=union_domain.x_bounds,
        union_mask=union_domain.union_mask,
        extended_z_bounds=extended_z_bounds,
        extended_y_bounds=extended_y_bounds,
        extended_x_bounds=extended_x_bounds,
    )


def reflect_axis_indices(
    indices: np.ndarray,
    *,
    axis_length: int,
) -> np.ndarray:
    normalized_axis_length = int(axis_length)
    if normalized_axis_length <= 0:
        raise ValueError("axis_length must be positive for reflect indexing.")
    if normalized_axis_length == 1:
        return np.zeros(np.shape(indices), dtype=np.int64)

    period = int(2 * (normalized_axis_length - 1))
    normalized = np.mod(np.asarray(indices, dtype=np.int64), period)
    return np.where(
        normalized <= (normalized_axis_length - 1),
        normalized,
        period - normalized,
    ).astype(np.int64, copy=False)


def build_extended_foreground_with_halo_padding(
    *,
    segmentation_volume: np.ndarray,
    core_z_bounds: Tuple[int, int],
    core_y_bounds: Tuple[int, int],
    core_x_bounds: Tuple[int, int],
    halo_size: int = 1,
) -> np.ndarray:
    volume = np.asarray(segmentation_volume)
    if volume.ndim != 3:
        raise ValueError(
            "segmentation_volume must be a 3D array, " f"got ndim={volume.ndim}."
        )
    depth, height, width = (int(dim) for dim in volume.shape)
    normalized_halo_size = int(halo_size)
    if normalized_halo_size < 0:
        raise ValueError("halo_size must be >= 0")

    z0, z1 = (int(core_z_bounds[0]), int(core_z_bounds[1]))
    y0, y1 = (int(core_y_bounds[0]), int(core_y_bounds[1]))
    x0, x1 = (int(core_x_bounds[0]), int(core_x_bounds[1]))
    if not (0 <= z0 < z1 <= depth):
        raise ValueError(
            "core_z_bounds must satisfy 0 <= z0 < z1 <= depth: "
            f"{(z0, z1)} for depth={depth}"
        )
    if not (0 <= y0 < y1 <= height):
        raise ValueError(
            "core_y_bounds must satisfy 0 <= y0 < y1 <= height: "
            f"{(y0, y1)} for height={height}"
        )
    if not (0 <= x0 < x1 <= width):
        raise ValueError(
            "core_x_bounds must satisfy 0 <= x0 < x1 <= width: "
            f"{(x0, x1)} for width={width}"
        )

    core_foreground = np.asarray(
        volume[z0:z1, y0:y1, x0:x1] != 0,
        dtype=bool,
    )
    z_size = int(z1 - z0)
    y_size = int(y1 - y0)
    x_size = int(x1 - x0)

    requested_z = np.arange(
        z0 - normalized_halo_size,
        z1 + normalized_halo_size,
        dtype=np.int64,
    )
    requested_y = np.arange(
        y0 - normalized_halo_size,
        y1 + normalized_halo_size,
        dtype=np.int64,
    )
    requested_x = np.arange(
        x0 - normalized_halo_size,
        x1 + normalized_halo_size,
        dtype=np.int64,
    )

    reflected_z = reflect_axis_indices(requested_z, axis_length=depth)
    reflected_y = reflect_axis_indices(requested_y, axis_length=height)
    reflected_x = reflect_axis_indices(requested_x, axis_length=width)

    sampled_z = np.clip(reflected_z, z0, z1 - 1) - z0
    sampled_y = np.clip(reflected_y, y0, y1 - 1) - y0
    sampled_x = np.clip(reflected_x, x0, x1 - 1) - x0
    sampled_z = np.clip(sampled_z, 0, z_size - 1)
    sampled_y = np.clip(sampled_y, 0, y_size - 1)
    sampled_x = np.clip(sampled_x, 0, x_size - 1)

    expanded = np.take(core_foreground, sampled_z, axis=0)
    expanded = np.take(expanded, sampled_y, axis=1)
    expanded = np.take(expanded, sampled_x, axis=2)
    return np.asarray(expanded, dtype=bool)


def mask_to_absolute_coordinates(
    mask: np.ndarray,
    *,
    origin: Tuple[int, int, int],
) -> np.ndarray:
    local_coordinates = np.argwhere(np.asarray(mask, dtype=bool))
    if local_coordinates.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    origin_array = np.asarray(origin, dtype=np.int64).reshape(1, 3)
    return np.asarray(local_coordinates, dtype=np.int64) + origin_array


def bbox_segmentation_operation_display_name(
    operation: BBoxSegmentationOperation,
) -> str:
    if operation == "median_filter":
        return "Median Filter Selected"
    if operation == "erosion":
        return "Erosion Selected"
    if operation == "dilation":
        return "Dilation Selected"
    raise ValueError(f"Unsupported bbox segmentation operation: {operation!r}")


def compute_set_mask_labels(
    *,
    segmentation_roi: np.ndarray,
    set_mask: np.ndarray,
    union_mask: np.ndarray,
    fallback_label: int,
) -> np.ndarray:
    labels = np.asarray(segmentation_roi)
    pending_set = np.asarray(set_mask, dtype=bool)
    domain = np.asarray(union_mask, dtype=bool)
    if labels.shape != pending_set.shape or labels.shape != domain.shape:
        raise ValueError(
            "segmentation_roi, set_mask, and union_mask must share the same shape: "
            f"labels={tuple(labels.shape)} set={tuple(pending_set.shape)} "
            f"union={tuple(domain.shape)}"
        )
    if labels.ndim != 3:
        raise ValueError(
            "set-mask label propagation expects 3D arrays, " f"got ndim={labels.ndim}"
        )
    local_coordinates = np.argwhere(pending_set)
    if local_coordinates.size == 0:
        return np.empty((0,), dtype=np.int64)

    fallback = int(fallback_label)
    resolved_labels = np.empty((local_coordinates.shape[0],), dtype=np.int64)
    depth, height, width = labels.shape
    for index, coordinate in enumerate(local_coordinates):
        z = int(coordinate[0])
        y = int(coordinate[1])
        x = int(coordinate[2])
        z0 = max(0, z - 1)
        z1 = min(depth, z + 2)
        y0 = max(0, y - 1)
        y1 = min(height, y + 2)
        x0 = max(0, x - 1)
        x1 = min(width, x + 2)

        neighborhood_labels = np.asarray(labels[z0:z1, y0:y1, x0:x1], dtype=np.int64)
        neighborhood_domain = domain[z0:z1, y0:y1, x0:x1]
        candidate_labels = neighborhood_labels[
            np.logical_and(neighborhood_domain, neighborhood_labels != 0)
        ]
        if candidate_labels.size == 0:
            resolved_labels[index] = fallback
            continue
        values, counts = np.unique(candidate_labels, return_counts=True)
        max_count = int(np.max(counts))
        winners = values[counts == max_count]
        resolved_labels[index] = int(np.min(winners))
    return resolved_labels


def compute_selected_bbox_binary_operation(
    *,
    operation: BBoxSegmentationOperation,
    foreground_mask: np.ndarray,
    union_mask: np.ndarray,
) -> np.ndarray:
    foreground = np.asarray(foreground_mask, dtype=bool)
    domain = np.asarray(union_mask, dtype=bool)
    if foreground.shape != domain.shape:
        raise ValueError(
            "foreground_mask and union_mask must share the same shape: "
            f"foreground={tuple(foreground.shape)} union={tuple(domain.shape)}"
        )
    constrained_foreground = foreground & domain
    neighbor_counts = count_true_neighbors_3x3x3(constrained_foreground)

    if operation == "median_filter":
        transformed = neighbor_counts >= 14
    elif operation == "erosion":
        transformed = neighbor_counts == 27
    elif operation == "dilation":
        transformed = neighbor_counts >= 1
    else:
        raise ValueError(f"Unsupported bbox segmentation operation: {operation!r}")
    return np.asarray(transformed, dtype=bool) & domain


def compute_selected_bbox_binary_operation_with_halo_context(
    *,
    operation: BBoxSegmentationOperation,
    segmentation_volume: np.ndarray,
    core_z_bounds: Tuple[int, int],
    core_y_bounds: Tuple[int, int],
    core_x_bounds: Tuple[int, int],
    halo_size: int = 1,
    binary_operation_func: Callable[
        ..., np.ndarray
    ] = compute_selected_bbox_binary_operation,
) -> np.ndarray:
    normalized_halo_size = int(halo_size)
    if normalized_halo_size < 0:
        raise ValueError("halo_size must be >= 0")

    core_shape = (
        int(core_z_bounds[1]) - int(core_z_bounds[0]),
        int(core_y_bounds[1]) - int(core_y_bounds[0]),
        int(core_x_bounds[1]) - int(core_x_bounds[0]),
    )
    if any(dim <= 0 for dim in core_shape):
        raise ValueError(
            "Core bounds must define a non-empty 3D region, " f"got shape={core_shape}."
        )

    extended_foreground = build_extended_foreground_with_halo_padding(
        segmentation_volume=segmentation_volume,
        core_z_bounds=core_z_bounds,
        core_y_bounds=core_y_bounds,
        core_x_bounds=core_x_bounds,
        halo_size=normalized_halo_size,
    )
    transformed_extended = binary_operation_func(
        operation=operation,
        foreground_mask=extended_foreground,
        union_mask=np.ones(np.shape(extended_foreground), dtype=bool),
    )
    transformed_arr = np.asarray(transformed_extended, dtype=bool)
    if tuple(int(dim) for dim in transformed_arr.shape) == core_shape:
        return transformed_arr

    expected_extended_shape = tuple(
        int(dim) + int(2 * normalized_halo_size) for dim in core_shape
    )
    if tuple(int(dim) for dim in transformed_arr.shape) != expected_extended_shape:
        raise ValueError(
            "Unexpected transformed mask shape for halo-aware selected-bbox processing: "
            f"got={tuple(transformed_arr.shape)} expected_core={core_shape} "
            f"expected_extended={expected_extended_shape}"
        )

    z_slice = slice(normalized_halo_size, normalized_halo_size + int(core_shape[0]))
    y_slice = slice(normalized_halo_size, normalized_halo_size + int(core_shape[1]))
    x_slice = slice(normalized_halo_size, normalized_halo_size + int(core_shape[2]))
    return np.asarray(transformed_arr[z_slice, y_slice, x_slice], dtype=bool)


def count_true_neighbors_3x3x3(mask: np.ndarray) -> np.ndarray:
    data = np.asarray(mask, dtype=bool)
    if data.ndim != 3:
        raise ValueError(
            f"3x3x3 neighborhood counting expects a 3D mask, got ndim={data.ndim}"
        )
    padded = np.pad(data.astype(np.uint8, copy=False), pad_width=1, mode="constant")
    counts = np.zeros(data.shape, dtype=np.uint8)
    for z_off in range(3):
        z_slice = slice(z_off, z_off + data.shape[0])
        for y_off in range(3):
            y_slice = slice(y_off, y_off + data.shape[1])
            for x_off in range(3):
                x_slice = slice(x_off, x_off + data.shape[2])
                counts += padded[z_slice, y_slice, x_slice]
    return counts


__all__ = [
    "BBoxSegmentationOperation",
    "BBoxUnionDomain",
    "BBoxProcessingRegions",
    "build_selected_bbox_union_domain",
    "expand_axis_bounds_with_halo",
    "build_selected_bbox_processing_regions",
    "reflect_axis_indices",
    "build_extended_foreground_with_halo_padding",
    "mask_to_absolute_coordinates",
    "bbox_segmentation_operation_display_name",
    "compute_set_mask_labels",
    "compute_selected_bbox_binary_operation",
    "compute_selected_bbox_binary_operation_with_halo_context",
    "count_true_neighbors_3x3x3",
]
