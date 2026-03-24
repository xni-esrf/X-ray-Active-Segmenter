from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from numbers import Integral
from typing import Dict, Iterable, List, Optional, Tuple

from .model import BoundingBox, Shape3D


FORMAT_MAGIC = "segmentation_tool_bboxes"
FORMAT_VERSION = 2
_SUPPORTED_FORMAT_VERSIONS = {1, FORMAT_VERSION}
_HEADER_PREFIX = f"# {FORMAT_MAGIC} v"
HEADER_LINE = f"{_HEADER_PREFIX}{FORMAT_VERSION}"


@dataclass(frozen=True)
class BoundingBoxFileData:
    version: int
    volume_shape: Shape3D
    boxes: Tuple[BoundingBox, ...]


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


def _normalize_boxes(
    boxes: Iterable[BoundingBox],
    *,
    volume_shape: Shape3D,
) -> Tuple[BoundingBox, ...]:
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
        box.validate_within(volume_shape)
        seen_ids[box_id] = None
        normalized.append(box)
    return tuple(normalized)


def serialize_bounding_boxes(data: BoundingBoxFileData) -> str:
    if data.version != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported bounding-box format version for serialization: {data.version}"
        )
    volume_shape = _coerce_shape(data.volume_shape)
    boxes = _normalize_boxes(data.boxes, volume_shape=volume_shape)

    lines = [HEADER_LINE, f"shape {volume_shape[0]} {volume_shape[1]} {volume_shape[2]}"]
    for box in boxes:
        z0, z1, y0, y1, x0, x1 = box.as_tuple()
        lines.append(f"box {box.id} {z0} {z1} {y0} {y1} {x0} {x1} {box.label}")
    return "\n".join(lines) + "\n"


def parse_bounding_boxes_text(
    text: str,
    *,
    expected_shape: Optional[Shape3D] = None,
) -> BoundingBoxFileData:
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")

    expected: Optional[Shape3D] = None
    if expected_shape is not None:
        expected = _coerce_shape(expected_shape)

    lines = text.splitlines()
    header_seen = False
    parsed_version: Optional[int] = None
    shape: Optional[Shape3D] = None
    boxes: List[BoundingBox] = []
    seen_ids: Dict[str, None] = {}

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        if not header_seen:
            if line.startswith(_HEADER_PREFIX):
                version_str = line[len(_HEADER_PREFIX) :].strip()
                if not version_str.isdigit():
                    raise ValueError(
                        f"Invalid bounding-box header version at line {line_number}: {line!r}"
                    )
                parsed_version = int(version_str)
                if parsed_version not in _SUPPORTED_FORMAT_VERSIONS:
                    supported = ", ".join(str(version) for version in sorted(_SUPPORTED_FORMAT_VERSIONS))
                    raise ValueError(
                        "Unsupported bounding-box file format version: "
                        f"{parsed_version} (supported: {supported})"
                    )
                header_seen = True
                continue
            if line.startswith("#"):
                continue
            raise ValueError(
                f"Missing bounding-box header before data at line {line_number}"
            )

        if line.startswith("#"):
            continue

        tokens = line.split()
        if not tokens:
            continue

        directive = tokens[0]
        if directive == "shape":
            if shape is not None:
                raise ValueError(f"Duplicate shape directive at line {line_number}")
            if len(tokens) != 4:
                raise ValueError(
                    f"Invalid shape directive at line {line_number}: expected 3 integers"
                )
            try:
                parsed_shape = (
                    int(tokens[1]),
                    int(tokens[2]),
                    int(tokens[3]),
                )
            except ValueError as exc:
                raise ValueError(
                    f"Invalid shape directive at line {line_number}: non-integer value"
                ) from exc
            shape = _coerce_shape(parsed_shape)
            if expected is not None and shape != expected:
                raise ValueError(
                    "Bounding-box file shape does not match expected shape: "
                    f"file={shape} expected={expected}"
                )
            continue

        if directive == "box":
            if shape is None:
                raise ValueError(
                    f"Box directive requires a preceding shape directive (line {line_number})"
                )
            if len(tokens) not in (8, 9):
                raise ValueError(
                    f"Invalid box directive at line {line_number}: expected id, 6 bounds, and optional label"
                )
            box_id = _coerce_non_empty_id(tokens[1])
            if box_id in seen_ids:
                raise ValueError(f"Duplicate bounding box id '{box_id}' at line {line_number}")
            try:
                z0, z1, y0, y1, x0, x1 = (
                    int(tokens[2]),
                    int(tokens[3]),
                    int(tokens[4]),
                    int(tokens[5]),
                    int(tokens[6]),
                    int(tokens[7]),
                )
            except ValueError as exc:
                raise ValueError(
                    f"Invalid box directive at line {line_number}: non-integer bound"
                ) from exc
            label = "train" if len(tokens) == 8 else tokens[8]
            box = BoundingBox.from_bounds(
                box_id=box_id,
                z0=z0,
                z1=z1,
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
                label=label,
                volume_shape=shape,
            )
            seen_ids[box_id] = None
            boxes.append(box)
            continue

        raise ValueError(f"Unknown directive '{directive}' at line {line_number}")

    if not header_seen:
        raise ValueError("Missing bounding-box header")
    if shape is None:
        raise ValueError("Missing shape directive in bounding-box file")
    if parsed_version is None:
        raise ValueError("Missing bounding-box header version")

    return BoundingBoxFileData(
        version=parsed_version,
        volume_shape=shape,
        boxes=tuple(boxes),
    )


def save_bounding_boxes(
    path: str,
    *,
    volume_shape: Shape3D,
    boxes: Iterable[BoundingBox],
    overwrite: bool = False,
) -> str:
    normalized_path = str(Path(path).expanduser())
    target = Path(normalized_path)
    if target.exists() and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing path without explicit overwrite=True: "
            f"{normalized_path}"
        )
    shape = _coerce_shape(volume_shape)
    normalized_boxes = _normalize_boxes(boxes, volume_shape=shape)
    payload = BoundingBoxFileData(
        version=FORMAT_VERSION,
        volume_shape=shape,
        boxes=normalized_boxes,
    )
    text = serialize_bounding_boxes(payload)
    target.write_text(text, encoding="utf-8")
    return normalized_path


def load_bounding_boxes(
    path: str,
    *,
    expected_shape: Optional[Shape3D] = None,
) -> BoundingBoxFileData:
    normalized_path = str(Path(path).expanduser())
    text = Path(normalized_path).read_text(encoding="utf-8")
    return parse_bounding_boxes_text(text, expected_shape=expected_shape)
