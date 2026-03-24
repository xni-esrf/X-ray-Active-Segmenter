from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Dict, Literal, Optional, Tuple


Axis = Literal[0, 1, 2]
BoundingBoxLabel = Literal["train", "validation", "inference"]
VoxelIndex = Tuple[int, int, int]
BoundaryIndex = Tuple[int, int, int]
Shape3D = Tuple[int, int, int]
Bounds3D = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
Projection2D = Tuple[Tuple[int, int], Tuple[int, int]]

FaceId = Literal[
    "z_min",
    "z_max",
    "y_min",
    "y_max",
    "x_min",
    "x_max",
]

CornerId = Literal[
    "z_min_y_min_x_min",
    "z_min_y_min_x_max",
    "z_min_y_max_x_min",
    "z_min_y_max_x_max",
    "z_max_y_min_x_min",
    "z_max_y_min_x_max",
    "z_max_y_max_x_min",
    "z_max_y_max_x_max",
]

_FACE_TO_AXIS: Dict[FaceId, int] = {
    "z_min": 0,
    "z_max": 0,
    "y_min": 1,
    "y_max": 1,
    "x_min": 2,
    "x_max": 2,
}

_CORNER_ORIENTATION: Dict[CornerId, Tuple[str, str, str]] = {
    "z_min_y_min_x_min": ("min", "min", "min"),
    "z_min_y_min_x_max": ("min", "min", "max"),
    "z_min_y_max_x_min": ("min", "max", "min"),
    "z_min_y_max_x_max": ("min", "max", "max"),
    "z_max_y_min_x_min": ("max", "min", "min"),
    "z_max_y_min_x_max": ("max", "min", "max"),
    "z_max_y_max_x_min": ("max", "max", "min"),
    "z_max_y_max_x_max": ("max", "max", "max"),
}

_ALLOWED_LABELS = {"train", "validation", "inference"}


def _coerce_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    return int(value)


def _coerce_axis(value: object) -> Axis:
    axis = _coerce_int(value, name="axis")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    return axis  # type: ignore[return-value]


def _coerce_shape(shape: Shape3D) -> Shape3D:
    if len(shape) != 3:
        raise ValueError("volume_shape must have exactly 3 dimensions (z, y, x)")
    z, y, x = (
        _coerce_int(shape[0], name="volume_shape[0]"),
        _coerce_int(shape[1], name="volume_shape[1]"),
        _coerce_int(shape[2], name="volume_shape[2]"),
    )
    if z <= 0 or y <= 0 or x <= 0:
        raise ValueError("volume_shape dimensions must be strictly positive")
    return (z, y, x)


def _coerce_index3(value: Tuple[int, int, int], *, name: str) -> Tuple[int, int, int]:
    if len(value) != 3:
        raise ValueError(f"{name} must have exactly 3 coordinates (z, y, x)")
    return (
        _coerce_int(value[0], name=f"{name}[0]"),
        _coerce_int(value[1], name=f"{name}[1]"),
        _coerce_int(value[2], name=f"{name}[2]"),
    )


def _clamp(value: int, lower: int, upper: int) -> int:
    if lower > upper:
        raise ValueError("Invalid clamp interval")
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def _translate_axis(start: int, stop: int, delta: int, extent: int) -> Tuple[int, int]:
    size = stop - start
    if size <= 0:
        raise ValueError("Bounding-box axis size must be positive")
    if size > extent:
        raise ValueError("Bounding-box axis size cannot exceed volume extent")
    next_start = start + delta
    max_start = extent - size
    clamped_start = _clamp(next_start, 0, max_start)
    return (clamped_start, clamped_start + size)


def _normalize_label(label: object) -> BoundingBoxLabel:
    if not isinstance(label, str):
        raise TypeError(f"Bounding-box label must be a string, got {type(label).__name__}")
    normalized = label.strip().lower()
    if normalized not in _ALLOWED_LABELS:
        allowed = ", ".join(sorted(_ALLOWED_LABELS))
        raise ValueError(f"Bounding-box label must be one of: {allowed}")
    return normalized  # type: ignore[return-value]


@dataclass(frozen=True)
class BoundingBox:
    id: str
    z0: int
    z1: int
    y0: int
    y1: int
    x0: int
    x1: int
    label: BoundingBoxLabel = "train"

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Bounding-box id must be a non-empty string")

        object.__setattr__(self, "z0", _coerce_int(self.z0, name="z0"))
        object.__setattr__(self, "z1", _coerce_int(self.z1, name="z1"))
        object.__setattr__(self, "y0", _coerce_int(self.y0, name="y0"))
        object.__setattr__(self, "y1", _coerce_int(self.y1, name="y1"))
        object.__setattr__(self, "x0", _coerce_int(self.x0, name="x0"))
        object.__setattr__(self, "x1", _coerce_int(self.x1, name="x1"))
        object.__setattr__(self, "label", _normalize_label(self.label))

        if self.z0 < 0 or self.y0 < 0 or self.x0 < 0:
            raise ValueError("Bounding-box lower bounds must be >= 0")
        if self.z1 <= self.z0:
            raise ValueError("Bounding-box bounds must satisfy z0 < z1")
        if self.y1 <= self.y0:
            raise ValueError("Bounding-box bounds must satisfy y0 < y1")
        if self.x1 <= self.x0:
            raise ValueError("Bounding-box bounds must satisfy x0 < x1")

    @classmethod
    def from_bounds(
        cls,
        *,
        box_id: str,
        z0: int,
        z1: int,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        label: BoundingBoxLabel = "train",
        volume_shape: Optional[Shape3D] = None,
    ) -> "BoundingBox":
        box = cls(id=box_id, z0=z0, z1=z1, y0=y0, y1=y1, x0=x0, x1=x1, label=label)
        if volume_shape is not None:
            box.validate_within(volume_shape)
        return box

    @classmethod
    def from_voxel_corners(
        cls,
        *,
        box_id: str,
        p0: VoxelIndex,
        p1: VoxelIndex,
        label: BoundingBoxLabel = "train",
        volume_shape: Optional[Shape3D] = None,
    ) -> "BoundingBox":
        z0, y0, x0 = _coerce_index3(p0, name="p0")
        z1, y1, x1 = _coerce_index3(p1, name="p1")
        box = cls(
            id=box_id,
            z0=min(z0, z1),
            z1=max(z0, z1) + 1,
            y0=min(y0, y1),
            y1=max(y0, y1) + 1,
            x0=min(x0, x1),
            x1=max(x0, x1) + 1,
            label=label,
        )
        if volume_shape is not None:
            box.validate_within(volume_shape)
        return box

    @property
    def bounds(self) -> Bounds3D:
        return ((self.z0, self.z1), (self.y0, self.y1), (self.x0, self.x1))

    @property
    def size_voxels(self) -> Tuple[int, int, int]:
        return (self.z1 - self.z0, self.y1 - self.y0, self.x1 - self.x0)

    @property
    def center_index_space(self) -> Tuple[float, float, float]:
        # Centers are expressed in voxel-index space (voxel centers).
        return (
            (self.z0 + self.z1 - 1) / 2.0,
            (self.y0 + self.y1 - 1) / 2.0,
            (self.x0 + self.x1 - 1) / 2.0,
        )

    def as_tuple(self) -> Tuple[int, int, int, int, int, int]:
        return (self.z0, self.z1, self.y0, self.y1, self.x0, self.x1)

    def validate_within(self, volume_shape: Shape3D) -> None:
        z_size, y_size, x_size = _coerce_shape(volume_shape)
        if self.z1 > z_size or self.y1 > y_size or self.x1 > x_size:
            raise ValueError(
                "Bounding box exceeds volume shape "
                f"{volume_shape}: box={self.as_tuple()}"
            )

    def contains_voxel(self, voxel: VoxelIndex) -> bool:
        z, y, x = _coerce_index3(voxel, name="voxel")
        return (
            self.z0 <= z < self.z1
            and self.y0 <= y < self.y1
            and self.x0 <= x < self.x1
        )

    def intersects_slice(self, axis: Axis, index: int) -> bool:
        normalized_axis = _coerce_axis(axis)
        slice_index = _coerce_int(index, name="index")
        if normalized_axis == 0:
            return self.z0 <= slice_index < self.z1
        if normalized_axis == 1:
            return self.y0 <= slice_index < self.y1
        return self.x0 <= slice_index < self.x1

    def slice_projection(self, axis: Axis, index: int) -> Optional[Projection2D]:
        normalized_axis = _coerce_axis(axis)
        if not self.intersects_slice(normalized_axis, index):
            return None
        if normalized_axis == 0:
            return ((self.y0, self.y1), (self.x0, self.x1))
        if normalized_axis == 1:
            return ((self.z0, self.z1), (self.x0, self.x1))
        return ((self.z0, self.z1), (self.y0, self.y1))

    def corner_coordinate(self, corner: CornerId) -> BoundaryIndex:
        orientation = _CORNER_ORIENTATION[corner]
        return (
            self.z0 if orientation[0] == "min" else self.z1,
            self.y0 if orientation[1] == "min" else self.y1,
            self.x0 if orientation[2] == "min" else self.x1,
        )

    def move(
        self,
        *,
        dz: int = 0,
        dy: int = 0,
        dx: int = 0,
        volume_shape: Shape3D,
    ) -> "BoundingBox":
        z_size, y_size, x_size = _coerce_shape(volume_shape)
        self.validate_within((z_size, y_size, x_size))
        dz_i = _coerce_int(dz, name="dz")
        dy_i = _coerce_int(dy, name="dy")
        dx_i = _coerce_int(dx, name="dx")

        z0, z1 = _translate_axis(self.z0, self.z1, dz_i, z_size)
        y0, y1 = _translate_axis(self.y0, self.y1, dy_i, y_size)
        x0, x1 = _translate_axis(self.x0, self.x1, dx_i, x_size)
        return self._rebuild(z0=z0, z1=z1, y0=y0, y1=y1, x0=x0, x1=x1)

    def move_face(
        self,
        face: FaceId,
        new_boundary: int,
        *,
        volume_shape: Shape3D,
    ) -> "BoundingBox":
        z_size, y_size, x_size = _coerce_shape(volume_shape)
        self.validate_within((z_size, y_size, x_size))
        if face not in _FACE_TO_AXIS:
            raise ValueError(f"Unknown face identifier: {face}")
        target = _coerce_int(new_boundary, name="new_boundary")

        z0, z1 = self.z0, self.z1
        y0, y1 = self.y0, self.y1
        x0, x1 = self.x0, self.x1

        if face == "z_min":
            z0 = _clamp(target, 0, z1 - 1)
        elif face == "z_max":
            z1 = _clamp(target, z0 + 1, z_size)
        elif face == "y_min":
            y0 = _clamp(target, 0, y1 - 1)
        elif face == "y_max":
            y1 = _clamp(target, y0 + 1, y_size)
        elif face == "x_min":
            x0 = _clamp(target, 0, x1 - 1)
        elif face == "x_max":
            x1 = _clamp(target, x0 + 1, x_size)
        return self._rebuild(z0=z0, z1=z1, y0=y0, y1=y1, x0=x0, x1=x1)

    def move_corner(
        self,
        corner: CornerId,
        new_corner: BoundaryIndex,
        *,
        volume_shape: Shape3D,
    ) -> "BoundingBox":
        orientation = _CORNER_ORIENTATION[corner]
        z_new, y_new, x_new = _coerce_index3(new_corner, name="new_corner")

        box = self
        box = box.move_face(
            "z_min" if orientation[0] == "min" else "z_max",
            z_new,
            volume_shape=volume_shape,
        )
        box = box.move_face(
            "y_min" if orientation[1] == "min" else "y_max",
            y_new,
            volume_shape=volume_shape,
        )
        box = box.move_face(
            "x_min" if orientation[2] == "min" else "x_max",
            x_new,
            volume_shape=volume_shape,
        )
        return box

    def _rebuild(
        self,
        *,
        z0: int,
        z1: int,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        label: Optional[BoundingBoxLabel] = None,
    ) -> "BoundingBox":
        next_label = self.label if label is None else _normalize_label(label)
        if (
            z0 == self.z0
            and z1 == self.z1
            and y0 == self.y0
            and y1 == self.y1
            and x0 == self.x0
            and x1 == self.x1
            and next_label == self.label
        ):
            return self
        return BoundingBox(
            id=self.id,
            z0=z0,
            z1=z1,
            y0=y0,
            y1=y1,
            x0=x0,
            x1=x1,
            label=next_label,
        )
