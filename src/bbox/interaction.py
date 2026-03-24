from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from numbers import Integral, Real
from typing import Dict, Iterable, List, Literal, Optional, Tuple

from .model import Axis, FaceId
from .projection import ProjectedBoundingBox2D


EdgeHandle = Literal["top", "bottom", "left", "right"]
CornerHandle = Literal["top_left", "top_right", "bottom_left", "bottom_right"]
HandleName = Literal[
    "top",
    "bottom",
    "left",
    "right",
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
]
HandleKind = Literal["edge", "corner"]
FaceUpdate = Tuple[FaceId, int]

_EDGE_HANDLES: Tuple[EdgeHandle, ...] = ("top", "bottom", "left", "right")
_CORNER_HANDLES: Tuple[CornerHandle, ...] = (
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
)
_CORNER_TO_EDGES: Dict[CornerHandle, Tuple[EdgeHandle, EdgeHandle]] = {
    "top_left": ("top", "left"),
    "top_right": ("top", "right"),
    "bottom_left": ("bottom", "left"),
    "bottom_right": ("bottom", "right"),
}
_EDGE_TO_FACE_BY_AXIS: Dict[int, Dict[EdgeHandle, FaceId]] = {
    0: {
        "top": "y_min",
        "bottom": "y_max",
        "left": "x_min",
        "right": "x_max",
    },
    1: {
        "top": "z_min",
        "bottom": "z_max",
        "left": "x_min",
        "right": "x_max",
    },
    2: {
        "top": "z_min",
        "bottom": "z_max",
        "left": "y_min",
        "right": "y_max",
    },
}


def _coerce_axis(value: object) -> Axis:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"axis must be an integer, got {type(value).__name__}")
    axis = int(value)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    return axis  # type: ignore[return-value]


def _coerce_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    return int(value)


def _coerce_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    return float(value)


def _coerce_edge_handle(handle: str) -> EdgeHandle:
    if handle in _EDGE_HANDLES:
        return handle  # type: ignore[return-value]
    raise ValueError(f"Unknown edge handle: {handle}")


def _coerce_corner_handle(handle: str) -> CornerHandle:
    if handle in _CORNER_HANDLES:
        return handle  # type: ignore[return-value]
    raise ValueError(f"Unknown corner handle: {handle}")


def _distance_point_to_segment(
    *,
    x: float,
    y: float,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> float:
    dx = x1 - x0
    dy = y1 - y0
    denom = (dx * dx) + (dy * dy)
    if denom <= 0.0:
        return hypot(x - x0, y - y0)
    t = ((x - x0) * dx + (y - y0) * dy) / denom
    t = max(0.0, min(1.0, t))
    proj_x = x0 + (t * dx)
    proj_y = y0 + (t * dy)
    return hypot(x - proj_x, y - proj_y)


@dataclass(frozen=True)
class BoundingBoxHandleHit:
    box_id: str
    kind: HandleKind
    handle: HandleName


def hit_test_projected_box_handles(
    boxes: Iterable[ProjectedBoundingBox2D],
    *,
    row: float,
    col: float,
    tolerance: float = 2.5,
    selected_id: Optional[str] = None,
) -> Optional[BoundingBoxHandleHit]:
    row_f = _coerce_float(row, name="row")
    col_f = _coerce_float(col, name="col")
    tolerance_f = _coerce_float(tolerance, name="tolerance")
    if tolerance_f < 0.0:
        raise ValueError("tolerance must be >= 0")
    if selected_id is not None and not isinstance(selected_id, str):
        raise TypeError(
            f"selected_id must be a string or None, got {type(selected_id).__name__}"
        )

    candidates: List[
        Tuple[float, int, int, int, BoundingBoxHandleHit]
    ] = []
    projected_items = list(boxes)
    for draw_rank, box in enumerate(reversed(projected_items)):
        if not isinstance(box, ProjectedBoundingBox2D):
            continue
        col0 = float(box.col0)
        col1 = float(box.col1)
        row0 = float(box.row0)
        row1 = float(box.row1)
        if col1 <= col0 or row1 <= row0:
            continue
        selected_penalty = 0 if selected_id is not None and box.box_id == selected_id else 1

        corners: Tuple[Tuple[CornerHandle, Tuple[float, float]], ...] = (
            ("top_left", (row0, col0)),
            ("top_right", (row0, col1)),
            ("bottom_left", (row1, col0)),
            ("bottom_right", (row1, col1)),
        )
        for handle, (corner_row, corner_col) in corners:
            distance = hypot(row_f - corner_row, col_f - corner_col)
            if distance > tolerance_f:
                continue
            candidates.append(
                (
                    distance,
                    0,
                    selected_penalty,
                    draw_rank,
                    BoundingBoxHandleHit(
                        box_id=box.box_id,
                        kind="corner",
                        handle=handle,
                    ),
                )
            )

        edges: Tuple[Tuple[EdgeHandle, Tuple[float, float, float, float]], ...] = (
            ("top", (col0, row0, col1, row0)),
            ("bottom", (col0, row1, col1, row1)),
            ("left", (col0, row0, col0, row1)),
            ("right", (col1, row0, col1, row1)),
        )
        for handle, (x0, y0, x1, y1) in edges:
            distance = _distance_point_to_segment(
                x=col_f,
                y=row_f,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
            )
            if distance > tolerance_f:
                continue
            candidates.append(
                (
                    distance,
                    1,
                    selected_penalty,
                    draw_rank,
                    BoundingBoxHandleHit(
                        box_id=box.box_id,
                        kind="edge",
                        handle=handle,
                    ),
                )
            )

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return candidates[0][4]


def face_updates_for_handle_drag(
    hit: BoundingBoxHandleHit,
    *,
    axis: Axis,
    row_boundary: int,
    col_boundary: int,
) -> Tuple[FaceUpdate, ...]:
    if not isinstance(hit, BoundingBoxHandleHit):
        raise TypeError(f"hit must be BoundingBoxHandleHit, got {type(hit).__name__}")
    normalized_axis = _coerce_axis(axis)
    row_i = _coerce_int(row_boundary, name="row_boundary")
    col_i = _coerce_int(col_boundary, name="col_boundary")
    edge_faces = _EDGE_TO_FACE_BY_AXIS[normalized_axis]

    if hit.kind == "edge":
        raise ValueError(
            "Edge handles do not produce face updates; use translation_delta_for_edge_drag"
        )

    if hit.kind != "corner":
        raise ValueError(f"Unknown handle kind: {hit.kind}")
    corner = _coerce_corner_handle(hit.handle)
    row_edge, col_edge = _CORNER_TO_EDGES[corner]
    return (
        (edge_faces[row_edge], row_i),
        (edge_faces[col_edge], col_i),
    )


def translation_delta_for_edge_drag(
    hit: BoundingBoxHandleHit,
    *,
    axis: Axis,
    row_delta: int,
    col_delta: int,
) -> Tuple[int, int, int]:
    if not isinstance(hit, BoundingBoxHandleHit):
        raise TypeError(f"hit must be BoundingBoxHandleHit, got {type(hit).__name__}")
    if hit.kind != "edge":
        raise ValueError("Only edge handles can translate a bounding box")
    _coerce_edge_handle(hit.handle)
    normalized_axis = _coerce_axis(axis)
    row_i = _coerce_int(row_delta, name="row_delta")
    col_i = _coerce_int(col_delta, name="col_delta")

    if normalized_axis == 0:
        return (0, row_i, col_i)
    if normalized_axis == 1:
        return (row_i, 0, col_i)
    return (row_i, col_i, 0)


__all__ = [
    "EdgeHandle",
    "CornerHandle",
    "HandleName",
    "HandleKind",
    "FaceUpdate",
    "BoundingBoxHandleHit",
    "hit_test_projected_box_handles",
    "face_updates_for_handle_drag",
    "translation_delta_for_edge_drag",
]
