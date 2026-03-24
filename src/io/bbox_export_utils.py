from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Optional, Tuple

import numpy as np


def _coerce_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    return int(value)


def _coerce_positive_size(value: object, *, name: str) -> int:
    size = _coerce_int(value, name=name)
    if size <= 0:
        raise ValueError(f"{name} must be >= 1")
    return size


def _coerce_non_negative_int(value: object, *, name: str) -> int:
    number = _coerce_int(value, name=name)
    if number < 0:
        raise ValueError(f"{name} must be >= 0")
    return number


@dataclass(frozen=True)
class AxisContextPlan:
    start: int
    stop: int
    volume_size: int
    target_size: int
    extend_before: int
    extend_after: int
    planned_start: int
    planned_stop: int
    clipped_start: int
    clipped_stop: int
    pad_before: int
    pad_after: int

    @property
    def original_size(self) -> int:
        return int(self.stop - self.start)

    @property
    def clipped_size(self) -> int:
        return int(self.clipped_stop - self.clipped_start)

    @property
    def final_size(self) -> int:
        return int(self.clipped_size + self.pad_before + self.pad_after)


@dataclass(frozen=True)
class BBoxContextPlan:
    z: AxisContextPlan
    y: AxisContextPlan
    x: AxisContextPlan

    @property
    def clipped_bounds(self) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        return (
            (self.z.clipped_start, self.z.clipped_stop),
            (self.y.clipped_start, self.y.clipped_stop),
            (self.x.clipped_start, self.x.clipped_stop),
        )

    @property
    def pad_width(self) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        return (
            (self.z.pad_before, self.z.pad_after),
            (self.y.pad_before, self.y.pad_after),
            (self.x.pad_before, self.x.pad_after),
        )


def target_size_for_bbox_dimension(size: int) -> int:
    """Return the target axis size after adding context.

    Rule:
    - size <= 250 -> 300
    - 250 < size <= 350 -> 400
    - 350 < size <= 450 -> 500
    - etc.
    """
    normalized_size = _coerce_positive_size(size, name="size")
    delta = normalized_size - 250
    if delta <= 0:
        return 300
    # 300 + 100 * ceil(max(0, size - 250) / 100)
    return 300 + 100 * ((delta + 99) // 100)


def split_centered_extension(total_extension: int) -> Tuple[int, int]:
    """Split a total extension between before/after sides.

    When odd, the extra voxel is assigned to the `before` side.
    """
    extension = _coerce_int(total_extension, name="total_extension")
    if extension < 0:
        raise ValueError("total_extension must be >= 0")
    before = (extension + 1) // 2
    after = extension // 2
    return (before, after)


def plan_axis_context(
    *,
    start: int,
    stop: int,
    volume_size: int,
    target_size: Optional[int] = None,
) -> AxisContextPlan:
    normalized_start = _coerce_non_negative_int(start, name="start")
    normalized_stop = _coerce_non_negative_int(stop, name="stop")
    normalized_volume_size = _coerce_positive_size(volume_size, name="volume_size")
    if normalized_stop <= normalized_start:
        raise ValueError("Axis bounds must satisfy start < stop")
    if normalized_stop > normalized_volume_size:
        raise ValueError(
            "Axis bounds exceed volume size: "
            f"stop={normalized_stop} volume_size={normalized_volume_size}"
        )

    original_size = normalized_stop - normalized_start
    if target_size is None:
        normalized_target = target_size_for_bbox_dimension(original_size)
    else:
        normalized_target = _coerce_positive_size(target_size, name="target_size")
    if normalized_target < original_size:
        raise ValueError(
            "target_size must be >= original axis size: "
            f"target_size={normalized_target} original_size={original_size}"
        )

    extend_before, extend_after = split_centered_extension(normalized_target - original_size)
    planned_start = normalized_start - extend_before
    planned_stop = normalized_stop + extend_after
    clipped_start = max(0, planned_start)
    clipped_stop = min(normalized_volume_size, planned_stop)
    pad_before = clipped_start - planned_start
    pad_after = planned_stop - clipped_stop

    return AxisContextPlan(
        start=normalized_start,
        stop=normalized_stop,
        volume_size=normalized_volume_size,
        target_size=normalized_target,
        extend_before=extend_before,
        extend_after=extend_after,
        planned_start=planned_start,
        planned_stop=planned_stop,
        clipped_start=clipped_start,
        clipped_stop=clipped_stop,
        pad_before=pad_before,
        pad_after=pad_after,
    )


def plan_bbox_context(
    *,
    z_bounds: Tuple[int, int],
    y_bounds: Tuple[int, int],
    x_bounds: Tuple[int, int],
    volume_shape: Tuple[int, int, int],
) -> BBoxContextPlan:
    if len(z_bounds) != 2:
        raise ValueError("z_bounds must have exactly 2 values")
    if len(y_bounds) != 2:
        raise ValueError("y_bounds must have exactly 2 values")
    if len(x_bounds) != 2:
        raise ValueError("x_bounds must have exactly 2 values")
    if len(volume_shape) != 3:
        raise ValueError("volume_shape must have exactly 3 dimensions (z, y, x)")

    z_size = _coerce_positive_size(volume_shape[0], name="volume_shape[0]")
    y_size = _coerce_positive_size(volume_shape[1], name="volume_shape[1]")
    x_size = _coerce_positive_size(volume_shape[2], name="volume_shape[2]")

    z_plan = plan_axis_context(
        start=z_bounds[0],
        stop=z_bounds[1],
        volume_size=z_size,
    )
    y_plan = plan_axis_context(
        start=y_bounds[0],
        stop=y_bounds[1],
        volume_size=y_size,
    )
    x_plan = plan_axis_context(
        start=x_bounds[0],
        stop=x_bounds[1],
        volume_size=x_size,
    )
    return BBoxContextPlan(
        z=z_plan,
        y=y_plan,
        x=x_plan,
    )


def _coerce_volume_array(volume_array: np.ndarray) -> np.ndarray:
    array = np.asarray(volume_array)
    if array.ndim != 3:
        raise ValueError(
            f"volume_array must be a 3D array (z, y, x), got ndim={array.ndim}"
        )
    return array


def _axis_supports_reflect_padding(size: int, pad_before: int, pad_after: int) -> bool:
    if pad_before <= 0 and pad_after <= 0:
        return True
    return int(size) > 1


def extract_planned_bbox_context_from_array(
    volume_array: np.ndarray,
    *,
    plan: BBoxContextPlan,
) -> np.ndarray:
    """Extract one bbox crop from `volume_array` using a precomputed context plan."""
    array = _coerce_volume_array(volume_array)
    expected_shape = (plan.z.volume_size, plan.y.volume_size, plan.x.volume_size)
    if tuple(int(v) for v in array.shape) != expected_shape:
        raise ValueError(
            "volume_array shape does not match plan volume sizes: "
            f"array.shape={tuple(array.shape)} plan_shape={expected_shape}"
        )

    clipped = array[
        plan.z.clipped_start : plan.z.clipped_stop,
        plan.y.clipped_start : plan.y.clipped_stop,
        plan.x.clipped_start : plan.x.clipped_stop,
    ]
    pad_width = plan.pad_width
    if any((before > 0 or after > 0) for before, after in pad_width):
        if not _axis_supports_reflect_padding(clipped.shape[0], plan.z.pad_before, plan.z.pad_after):
            raise ValueError(
                "Cannot apply reflect padding on z axis with length <= 1."
            )
        if not _axis_supports_reflect_padding(clipped.shape[1], plan.y.pad_before, plan.y.pad_after):
            raise ValueError(
                "Cannot apply reflect padding on y axis with length <= 1."
            )
        if not _axis_supports_reflect_padding(clipped.shape[2], plan.x.pad_before, plan.x.pad_after):
            raise ValueError(
                "Cannot apply reflect padding on x axis with length <= 1."
            )
        clipped = np.pad(clipped, pad_width, mode="reflect")

    expected_output_shape = (plan.z.target_size, plan.y.target_size, plan.x.target_size)
    if tuple(int(v) for v in clipped.shape) != expected_output_shape:
        raise RuntimeError(
            "Unexpected bbox context output shape: "
            f"got={tuple(clipped.shape)} expected={expected_output_shape}"
        )
    return clipped


def extract_bbox_context_from_array(
    volume_array: np.ndarray,
    *,
    z_bounds: Tuple[int, int],
    y_bounds: Tuple[int, int],
    x_bounds: Tuple[int, int],
) -> np.ndarray:
    """Extract one bbox crop from `volume_array` with context and reflect padding."""
    array = _coerce_volume_array(volume_array)
    volume_shape = (
        int(array.shape[0]),
        int(array.shape[1]),
        int(array.shape[2]),
    )
    plan = plan_bbox_context(
        z_bounds=z_bounds,
        y_bounds=y_bounds,
        x_bounds=x_bounds,
        volume_shape=volume_shape,
    )
    return extract_planned_bbox_context_from_array(array, plan=plan)
