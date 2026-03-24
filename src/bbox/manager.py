from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Callable, Dict, Iterable, List, Literal, Optional, Tuple

from .model import BoundingBox, CornerId, FaceId, Shape3D, VoxelIndex


ChangeKind = Literal[
    "added",
    "updated",
    "deleted",
    "cleared",
    "selection",
    "loaded",
    "cleaned",
]


@dataclass(frozen=True)
class BoundingBoxChange:
    kind: ChangeKind
    box_id: Optional[str]
    revision: int
    selected_id: Optional[str]
    box_count: int
    dirty: bool


ChangeCallback = Callable[[BoundingBoxChange], None]


def _coerce_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    return int(value)


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


def _coerce_non_empty_id(box_id: str) -> str:
    if not isinstance(box_id, str):
        raise TypeError(f"box id must be a string, got {type(box_id).__name__}")
    normalized = box_id.strip()
    if not normalized:
        raise ValueError("box id must be a non-empty string")
    if any(char.isspace() for char in normalized):
        raise ValueError("box id must not contain whitespace")
    return normalized


class BoundingBoxManager:
    def __init__(self, volume_shape: Shape3D) -> None:
        self._volume_shape = _coerce_shape(volume_shape)
        self._boxes: Dict[str, BoundingBox] = {}
        self._selected_id: Optional[str] = None
        self._listeners: List[ChangeCallback] = []
        self._revision = 0
        self._clean_boxes_signature = self._boxes_signature()
        self._next_auto_id = 1

    @property
    def volume_shape(self) -> Shape3D:
        return self._volume_shape

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def dirty(self) -> bool:
        return self._boxes_signature() != self._clean_boxes_signature

    @property
    def selected_id(self) -> Optional[str]:
        return self._selected_id

    def on_changed(self, callback: ChangeCallback) -> None:
        self._listeners.append(callback)

    def boxes(self) -> Tuple[BoundingBox, ...]:
        return tuple(self._boxes.values())

    def box_ids(self) -> Tuple[str, ...]:
        return tuple(self._boxes.keys())

    def get(self, box_id: str) -> Optional[BoundingBox]:
        normalized = _coerce_non_empty_id(box_id)
        return self._boxes.get(normalized)

    def add(self, box: BoundingBox, *, select: bool = False) -> BoundingBox:
        if not isinstance(box, BoundingBox):
            raise TypeError(f"box must be a BoundingBox, got {type(box).__name__}")
        box.validate_within(self._volume_shape)
        box_id = _coerce_non_empty_id(box.id)
        if box_id in self._boxes:
            raise ValueError(f"Bounding box id already exists: {box_id}")
        self._boxes[box_id] = box
        if select:
            self._selected_id = box_id
        self._touch("added", box_id)
        return box

    def add_from_corners(
        self,
        p0: VoxelIndex,
        p1: VoxelIndex,
        *,
        box_id: Optional[str] = None,
        select: bool = False,
    ) -> BoundingBox:
        candidate_id = self._next_id() if box_id is None else _coerce_non_empty_id(box_id)
        box = BoundingBox.from_voxel_corners(
            box_id=candidate_id,
            p0=p0,
            p1=p1,
            volume_shape=self._volume_shape,
        )
        return self.add(box, select=select)

    def replace(self, box_id: str, box: BoundingBox) -> BoundingBox:
        normalized = _coerce_non_empty_id(box_id)
        if normalized not in self._boxes:
            raise KeyError(f"Unknown bounding box id: {normalized}")
        if not isinstance(box, BoundingBox):
            raise TypeError(f"box must be a BoundingBox, got {type(box).__name__}")
        if box.id != normalized:
            raise ValueError(
                f"Replacement box id mismatch: expected {normalized}, got {box.id}"
            )
        box.validate_within(self._volume_shape)
        previous = self._boxes[normalized]
        if previous == box:
            return previous
        self._boxes[normalized] = box
        self._touch("updated", normalized)
        return box

    def move(
        self,
        box_id: str,
        *,
        dz: int = 0,
        dy: int = 0,
        dx: int = 0,
    ) -> BoundingBox:
        current = self._require(box_id)
        moved = current.move(
            dz=dz,
            dy=dy,
            dx=dx,
            volume_shape=self._volume_shape,
        )
        return self.replace(current.id, moved)

    def move_face(
        self,
        box_id: str,
        face: FaceId,
        new_boundary: int,
    ) -> BoundingBox:
        current = self._require(box_id)
        moved = current.move_face(
            face,
            new_boundary,
            volume_shape=self._volume_shape,
        )
        return self.replace(current.id, moved)

    def move_corner(
        self,
        box_id: str,
        corner: CornerId,
        new_corner: Tuple[int, int, int],
    ) -> BoundingBox:
        current = self._require(box_id)
        moved = current.move_corner(
            corner,
            new_corner,
            volume_shape=self._volume_shape,
        )
        return self.replace(current.id, moved)

    def delete(self, box_id: str) -> bool:
        normalized = _coerce_non_empty_id(box_id)
        if normalized not in self._boxes:
            return False
        self._boxes.pop(normalized, None)
        if self._selected_id == normalized:
            self._selected_id = None
        self._touch("deleted", normalized)
        return True

    def clear(self) -> None:
        if not self._boxes and self._selected_id is None:
            return
        self._boxes.clear()
        self._selected_id = None
        self._touch("cleared", None)

    def select(self, box_id: Optional[str]) -> Optional[BoundingBox]:
        normalized: Optional[str]
        if box_id is None:
            normalized = None
        else:
            normalized = _coerce_non_empty_id(box_id)
            if normalized not in self._boxes:
                raise KeyError(f"Unknown bounding box id: {normalized}")
        if normalized == self._selected_id:
            return self._boxes.get(normalized) if normalized is not None else None
        self._selected_id = normalized
        self._emit("selection", normalized)
        return self._boxes.get(normalized) if normalized is not None else None

    def mark_clean(self) -> None:
        if not self.dirty:
            return
        self._clean_boxes_signature = self._boxes_signature()
        self._emit("cleaned", None)

    def replace_all(
        self,
        boxes: Iterable[BoundingBox],
        *,
        selected_id: Optional[str] = None,
        mark_clean: bool = False,
    ) -> Tuple[BoundingBox, ...]:
        normalized_boxes = self._normalize_boxes(boxes)
        self._boxes = {box.id: box for box in normalized_boxes}
        if selected_id is None:
            if self._selected_id not in self._boxes:
                self._selected_id = None
        else:
            normalized_selected = _coerce_non_empty_id(selected_id)
            if normalized_selected not in self._boxes:
                raise KeyError(
                    f"selected_id references unknown box id: {normalized_selected}"
                )
            self._selected_id = normalized_selected
        self._sync_next_auto_id()
        self._revision += 1
        if mark_clean:
            self._clean_boxes_signature = self._boxes_signature()
        self._emit("loaded", self._selected_id)
        return tuple(self._boxes.values())

    def _boxes_signature(self) -> Tuple[Tuple[str, str, int, int, int, int, int, int], ...]:
        signature: List[Tuple[str, str, int, int, int, int, int, int]] = []
        for box in self._boxes.values():
            signature.append(
                (
                    box.id,
                    box.label,
                    int(box.z0),
                    int(box.z1),
                    int(box.y0),
                    int(box.y1),
                    int(box.x0),
                    int(box.x1),
                )
            )
        signature.sort(key=lambda item: item[0])
        return tuple(signature)

    def _normalize_boxes(self, boxes: Iterable[BoundingBox]) -> Tuple[BoundingBox, ...]:
        normalized: List[BoundingBox] = []
        seen_ids: Dict[str, None] = {}
        for box in boxes:
            if not isinstance(box, BoundingBox):
                raise TypeError(
                    f"All items must be BoundingBox instances, got {type(box).__name__}"
                )
            box_id = _coerce_non_empty_id(box.id)
            if box_id in seen_ids:
                raise ValueError(f"Duplicate bounding box id: {box_id}")
            box.validate_within(self._volume_shape)
            seen_ids[box_id] = None
            normalized.append(box)
        return tuple(normalized)

    def _require(self, box_id: str) -> BoundingBox:
        normalized = _coerce_non_empty_id(box_id)
        box = self._boxes.get(normalized)
        if box is None:
            raise KeyError(f"Unknown bounding box id: {normalized}")
        return box

    def _next_id(self) -> str:
        while True:
            candidate = f"bbox_{self._next_auto_id:04d}"
            self._next_auto_id += 1
            if candidate not in self._boxes:
                return candidate

    def _sync_next_auto_id(self) -> None:
        max_used = 0
        for box_id in self._boxes:
            if not box_id.startswith("bbox_"):
                continue
            suffix = box_id[5:]
            if suffix.isdigit():
                max_used = max(max_used, int(suffix))
        self._next_auto_id = max(1, max_used + 1)

    def _touch(self, kind: ChangeKind, box_id: Optional[str]) -> None:
        self._revision += 1
        self._emit(kind, box_id)

    def _emit(self, kind: ChangeKind, box_id: Optional[str]) -> None:
        event = BoundingBoxChange(
            kind=kind,
            box_id=box_id,
            revision=self._revision,
            selected_id=self._selected_id,
            box_count=len(self._boxes),
            dirty=self.dirty,
        )
        for callback in tuple(self._listeners):
            callback(event)
