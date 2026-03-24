from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, Optional, Tuple

from .model import Axis, BoundingBox, BoundingBoxLabel


Shape2D = Tuple[int, int]


def _coerce_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    return int(value)


def _coerce_axis(value: object) -> Axis:
    axis = _coerce_int(value, name="axis")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    return axis  # type: ignore[return-value]


def _coerce_level_scale(value: object) -> int:
    scale = _coerce_int(value, name="level_scale")
    if scale <= 0:
        raise ValueError("level_scale must be >= 1")
    return scale


def _coerce_shape2d(shape: Shape2D) -> Shape2D:
    if len(shape) != 2:
        raise ValueError("image_shape must have exactly 2 dimensions (height, width)")
    height = _coerce_int(shape[0], name="image_shape[0]")
    width = _coerce_int(shape[1], name="image_shape[1]")
    if height < 0 or width < 0:
        raise ValueError("image_shape dimensions must be >= 0")
    return (height, width)


def _to_level_bounds(start: int, stop: int, scale: int) -> Tuple[int, int]:
    level_start = start // scale
    level_stop = (stop + scale - 1) // scale
    return (level_start, level_stop)


def _clamp(value: int, lower: int, upper: int) -> int:
    if lower > upper:
        raise ValueError("Invalid clamp interval")
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


@dataclass(frozen=True)
class ProjectedBoundingBox2D:
    box_id: str
    row0: int
    row1: int
    col0: int
    col1: int
    label: BoundingBoxLabel = "train"

    @property
    def row_bounds(self) -> Tuple[int, int]:
        return (self.row0, self.row1)

    @property
    def col_bounds(self) -> Tuple[int, int]:
        return (self.col0, self.col1)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.row1 - self.row0, self.col1 - self.col0)


def project_box_to_slice(
    box: BoundingBox,
    *,
    axis: Axis,
    slice_index: int,
    level_scale: int = 1,
    image_shape: Optional[Shape2D] = None,
) -> Optional[ProjectedBoundingBox2D]:
    normalized_axis = _coerce_axis(axis)
    normalized_scale = _coerce_level_scale(level_scale)
    normalized_slice = _coerce_int(slice_index, name="slice_index")
    if not isinstance(box, BoundingBox):
        raise TypeError(f"box must be a BoundingBox, got {type(box).__name__}")

    z0, z1 = _to_level_bounds(box.z0, box.z1, normalized_scale)
    y0, y1 = _to_level_bounds(box.y0, box.y1, normalized_scale)
    x0, x1 = _to_level_bounds(box.x0, box.x1, normalized_scale)
    level_slice_index = normalized_slice // normalized_scale

    if normalized_axis == 0:
        intersects = z0 <= level_slice_index < z1
        row0, row1, col0, col1 = y0, y1, x0, x1
    elif normalized_axis == 1:
        intersects = y0 <= level_slice_index < y1
        row0, row1, col0, col1 = z0, z1, x0, x1
    else:
        intersects = x0 <= level_slice_index < x1
        row0, row1, col0, col1 = z0, z1, y0, y1

    if not intersects:
        return None

    if image_shape is not None:
        height, width = _coerce_shape2d(image_shape)
        row0 = _clamp(row0, 0, height)
        row1 = _clamp(row1, 0, height)
        col0 = _clamp(col0, 0, width)
        col1 = _clamp(col1, 0, width)

    if row0 >= row1 or col0 >= col1:
        return None

    return ProjectedBoundingBox2D(
        box_id=box.id,
        row0=row0,
        row1=row1,
        col0=col0,
        col1=col1,
        label=box.label,
    )


def project_boxes_to_slice(
    boxes: Iterable[BoundingBox],
    *,
    axis: Axis,
    slice_index: int,
    level_scale: int = 1,
    image_shape: Optional[Shape2D] = None,
) -> Tuple[ProjectedBoundingBox2D, ...]:
    projected = []
    for box in boxes:
        item = project_box_to_slice(
            box,
            axis=axis,
            slice_index=slice_index,
            level_scale=level_scale,
            image_shape=image_shape,
        )
        if item is not None:
            projected.append(item)
    return tuple(projected)
