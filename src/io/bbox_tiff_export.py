from __future__ import annotations

from dataclasses import dataclass
import re
from numbers import Integral
from pathlib import Path
from typing import Callable, Final, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..bbox.model import BoundingBox
from ..learning import (
    LearningBBoxTensorBatch,
    LearningBBoxTensorEntry,
    build_eval_dataloader_runtimes_from_batch,
    build_learning_dataloader_from_batch,
    clear_current_learning_dataloader_runtime,
    clear_current_learning_eval_runtimes_by_box_id,
    set_current_learning_bbox_entries,
)
from ..utils import torch_from_numpy_safe
from .bbox_export_utils import extract_bbox_context_from_array, plan_bbox_context


_ALLOWED_LABELS: Final = ("train", "validation", "inference")
FIXED_LEARNING_MINIVOL_SIZE: Final[int] = 200


@dataclass(frozen=True)
class SegmentationBBoxSaveFailure:
    box_id: str
    index: int
    label: str
    error: str


@dataclass(frozen=True)
class SegmentationBBoxExportOutcome:
    saved_paths: Tuple[str, ...] = tuple()
    skipped_zero_box_ids: Tuple[str, ...] = tuple()
    failed_boxes: Tuple[SegmentationBBoxSaveFailure, ...] = tuple()


@dataclass(frozen=True)
class RawBBoxSaveFailure:
    box_id: str
    index: int
    label: str
    error: str


@dataclass(frozen=True)
class RawBBoxExportOutcome:
    saved_paths: Tuple[str, ...] = tuple()
    failed_boxes: Tuple[RawBBoxSaveFailure, ...] = tuple()


@dataclass(frozen=True)
class LearningBBoxExtractionOutcome:
    raw_saved_paths: Tuple[str, ...] = tuple()
    raw_failed_boxes: Tuple[RawBBoxSaveFailure, ...] = tuple()
    segmentation_saved_paths: Tuple[str, ...] = tuple()
    segmentation_skipped_zero_box_ids: Tuple[str, ...] = tuple()
    segmentation_failed_boxes: Tuple[SegmentationBBoxSaveFailure, ...] = tuple()
    tensor_entry_count: int = 0
    learning_train_box_ids: Tuple[str, ...] = tuple()
    learning_batch_size: Optional[int] = None
    learning_num_workers: Optional[int] = None
    eval_validation_box_ids: Tuple[str, ...] = tuple()
    eval_batch_size: Optional[int] = None
    eval_num_workers: Optional[int] = None


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("PyTorch is required to build learning tensors") from exc
    return torch


def _torch_signed_integer_dtype_for_numpy(np_dtype: np.dtype):
    torch = _require_torch()
    normalized = np.dtype(np_dtype)
    mapping = {
        np.dtype(np.int8): torch.int8,
        np.dtype(np.int16): torch.int16,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.int64): torch.int64,
    }
    torch_dtype = mapping.get(normalized)
    if torch_dtype is None:
        raise ValueError(
            "Segmentation tensors require signed integer dtype matching saved arrays, "
            f"got {normalized}"
        )
    return torch_dtype


def _coerce_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    return int(value)


def _coerce_positive_int(value: object, *, name: str) -> int:
    integer = _coerce_int(value, name=name)
    if integer <= 0:
        raise ValueError(f"{name} must be >= 1")
    return integer


def _resolve_fixed_learning_minivol_size(value: Optional[object], *, name: str) -> int:
    if value is None:
        return int(FIXED_LEARNING_MINIVOL_SIZE)
    normalized = _coerce_positive_int(value, name=name)
    if normalized != int(FIXED_LEARNING_MINIVOL_SIZE):
        raise ValueError(
            f"{name} is fixed to {FIXED_LEARNING_MINIVOL_SIZE} for now, got {normalized}."
        )
    return int(FIXED_LEARNING_MINIVOL_SIZE)


def _normalize_label(label: object) -> str:
    if not isinstance(label, str):
        raise TypeError(f"label must be a string, got {type(label).__name__}")
    normalized = label.strip().lower()
    if normalized not in _ALLOWED_LABELS:
        allowed = ", ".join(_ALLOWED_LABELS)
        raise ValueError(f"label must be one of: {allowed}")
    return normalized


def _coerce_3d_array(array: np.ndarray) -> np.ndarray:
    normalized = np.asarray(array)
    if normalized.ndim != 3:
        raise ValueError(
            f"array must be a 3D volume (z, y, x), got ndim={normalized.ndim}"
        )
    return normalized


def smallest_signed_integer_dtype_for_range(min_value: int, max_value: int) -> np.dtype:
    normalized_min = _coerce_int(min_value, name="min_value")
    normalized_max = _coerce_int(max_value, name="max_value")
    if normalized_min > normalized_max:
        raise ValueError("min_value must be <= max_value")
    for candidate in (np.int8, np.int16, np.int32, np.int64):
        info = np.iinfo(candidate)
        if normalized_min >= int(info.min) and normalized_max <= int(info.max):
            return np.dtype(candidate)
    raise ValueError(
        "Signed integer range exceeds int64 limits: "
        f"[{normalized_min}, {normalized_max}]"
    )


def choose_segmentation_export_dtype(
    array: np.ndarray,
    *,
    context_fill_value: int = -100,
) -> np.dtype:
    normalized = np.asarray(array)
    if not np.issubdtype(normalized.dtype, np.integer):
        raise ValueError("Segmentation export dtype selection expects an integer array")
    context_value = _coerce_int(context_fill_value, name="context_fill_value")
    if normalized.size == 0:
        min_value = context_value
        max_value = context_value
    else:
        min_value = min(context_value, int(np.min(normalized)))
        max_value = max(context_value, int(np.max(normalized)))
    return smallest_signed_integer_dtype_for_range(min_value, max_value)


def _ensure_no_singleton_axis(shape: Tuple[int, int, int]) -> None:
    axis_names = ("z", "y", "x")
    singleton_axes = [
        axis_name
        for axis_name, axis_size in zip(axis_names, shape)
        if int(axis_size) == 1
    ]
    if not singleton_axes:
        return
    joined_axes = ", ".join(singleton_axes)
    raise ValueError(
        "Cannot export bounding boxes as TIFF when raw volume has an axis of length 1 "
        f"(singleton axes: {joined_axes})."
    )


def build_bbox_tiff_filename(index: int, label: str) -> str:
    normalized_index = _coerce_positive_int(index, name="index")
    normalized_label = _normalize_label(label)
    return f"bbox{normalized_index}_{normalized_label}.tif"


def build_segmentation_bbox_tiff_filename(index: int, label: str) -> str:
    normalized_index = _coerce_positive_int(index, name="index")
    normalized_label = _normalize_label(label)
    return f"bbox{normalized_index}_{normalized_label}_seg.tif"


def extract_segmentation_bbox_region_from_array(
    segmentation_array: np.ndarray,
    *,
    z_bounds: Tuple[int, int],
    y_bounds: Tuple[int, int],
    x_bounds: Tuple[int, int],
) -> np.ndarray:
    normalized = _coerce_3d_array(segmentation_array)
    if not np.issubdtype(normalized.dtype, np.integer):
        raise ValueError("segmentation_array must have an integer dtype")
    plan = plan_bbox_context(
        z_bounds=z_bounds,
        y_bounds=y_bounds,
        x_bounds=x_bounds,
        volume_shape=(
            int(normalized.shape[0]),
            int(normalized.shape[1]),
            int(normalized.shape[2]),
        ),
    )
    return normalized[
        int(plan.z.start) : int(plan.z.stop),
        int(plan.y.start) : int(plan.y.stop),
        int(plan.x.start) : int(plan.x.stop),
    ]


def is_segmentation_bbox_zero_only(
    segmentation_array: np.ndarray,
    *,
    z_bounds: Tuple[int, int],
    y_bounds: Tuple[int, int],
    x_bounds: Tuple[int, int],
) -> bool:
    region = extract_segmentation_bbox_region_from_array(
        segmentation_array,
        z_bounds=z_bounds,
        y_bounds=y_bounds,
        x_bounds=x_bounds,
    )
    return bool(np.all(region == 0))


def build_segmentation_bbox_context_from_array(
    segmentation_array: np.ndarray,
    *,
    z_bounds: Tuple[int, int],
    y_bounds: Tuple[int, int],
    x_bounds: Tuple[int, int],
    context_fill_value: int = -100,
    output_dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    normalized = _coerce_3d_array(segmentation_array)
    if not np.issubdtype(normalized.dtype, np.integer):
        raise ValueError("segmentation_array must have an integer dtype")

    plan = plan_bbox_context(
        z_bounds=z_bounds,
        y_bounds=y_bounds,
        x_bounds=x_bounds,
        volume_shape=(
            int(normalized.shape[0]),
            int(normalized.shape[1]),
            int(normalized.shape[2]),
        ),
    )
    bbox_region = extract_segmentation_bbox_region_from_array(
        normalized,
        z_bounds=z_bounds,
        y_bounds=y_bounds,
        x_bounds=x_bounds,
    )
    fill_value = _coerce_int(context_fill_value, name="context_fill_value")

    if output_dtype is None:
        target_dtype = choose_segmentation_export_dtype(
            bbox_region,
            context_fill_value=fill_value,
        )
    else:
        target_dtype = np.dtype(output_dtype)
        if not np.issubdtype(target_dtype, np.integer):
            raise ValueError(f"output_dtype must be an integer dtype, got {target_dtype}")
        target_info = np.iinfo(target_dtype)
        range_min = min(fill_value, int(np.min(bbox_region)))
        range_max = max(fill_value, int(np.max(bbox_region)))
        if range_min < int(target_info.min) or range_max > int(target_info.max):
            raise ValueError(
                "output_dtype cannot represent segmentation values and context fill: "
                f"dtype={target_dtype} required_range=[{range_min}, {range_max}]"
            )

    out_shape = (
        int(plan.z.target_size),
        int(plan.y.target_size),
        int(plan.x.target_size),
    )
    context = np.full(out_shape, fill_value=fill_value, dtype=target_dtype)

    z_start = int(plan.z.extend_before)
    y_start = int(plan.y.extend_before)
    x_start = int(plan.x.extend_before)
    z_stop = z_start + int(plan.z.original_size)
    y_stop = y_start + int(plan.y.original_size)
    x_stop = x_start + int(plan.x.original_size)
    context[z_start:z_stop, y_start:y_stop, x_start:x_stop] = bbox_region.astype(
        target_dtype,
        copy=False,
    )
    return context


def _save_array_as_tiff(
    array: np.ndarray,
    *,
    output_dir: str,
    filename: str,
    overwrite: bool = True,
) -> str:
    normalized_array = _coerce_3d_array(array)
    output_path = Path(str(output_dir).strip()).expanduser()
    if output_path.exists() and not output_path.is_dir():
        raise NotADirectoryError(f"output_dir is not a directory: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    target_path = output_path / str(filename).strip()
    if target_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file without overwrite=True: {target_path}"
        )

    try:
        import tifffile
    except ImportError as exc:
        raise ImportError("tifffile is required to save TIFF volumes") from exc

    tifffile.imwrite(str(target_path), normalized_array, compression=None)
    return str(target_path)


def _normalize_box_id(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"box id must be a string, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError("box id must be a non-empty string")
    return normalized


def _box_id_suffix_index(box_id: str) -> int:
    normalized = _normalize_box_id(box_id)
    match = re.search(r"(\d+)$", normalized)
    if match is None:
        raise ValueError(
            "Bounding box id must end with digits for TIFF naming, "
            f"got: {normalized}"
        )
    return _coerce_positive_int(int(match.group(1)), name=f"numeric suffix in box id '{normalized}'")


def export_bboxes_as_tiff(
    volume_array: np.ndarray,
    *,
    boxes_by_id: Mapping[str, BoundingBox],
    ordered_box_ids: Sequence[str],
    output_dir: str,
    overwrite: bool = True,
    extractor: Optional[Callable[..., np.ndarray]] = None,
    saver: Optional[Callable[..., str]] = None,
) -> Tuple[str, ...]:
    """Export all requested bounding boxes as TIFF files in UI order.

    `ordered_box_ids` defines export order. Output numbering is derived from each
    box id numeric suffix (e.g. `bbox_0007` -> `bbox7_<label>.tif`).
    """
    normalized_volume = _coerce_3d_array(volume_array)
    _ensure_no_singleton_axis(
        (
            int(normalized_volume.shape[0]),
            int(normalized_volume.shape[1]),
            int(normalized_volume.shape[2]),
        )
    )
    if extractor is None:
        def extractor_impl(
            array: np.ndarray,
            *,
            z_bounds: Tuple[int, int],
            y_bounds: Tuple[int, int],
            x_bounds: Tuple[int, int],
        ) -> np.ndarray:
            return extract_bbox_context_from_array(
                array,
                z_bounds=z_bounds,
                y_bounds=y_bounds,
                x_bounds=x_bounds,
            )
        extractor = extractor_impl
    if saver is None:
        def saver_impl(
            array: np.ndarray,
            *,
            output_dir: str,
            index: int,
            label: str,
            overwrite: bool,
        ) -> str:
            filename = build_bbox_tiff_filename(index, label)
            return _save_array_as_tiff(
                array,
                output_dir=output_dir,
                filename=filename,
                overwrite=overwrite,
            )
        saver = saver_impl

    saved_paths = []
    seen_ids = set()
    seen_indices = set()
    for raw_box_id in ordered_box_ids:
        box_id = _normalize_box_id(raw_box_id)
        if box_id in seen_ids:
            raise ValueError(f"Duplicate box id in ordered_box_ids: {box_id}")
        seen_ids.add(box_id)
        box = boxes_by_id.get(box_id)
        if box is None:
            raise KeyError(f"Unknown bounding box id in ordered_box_ids: {box_id}")
        if not isinstance(box, BoundingBox):
            raise TypeError(
                f"boxes_by_id must map ids to BoundingBox, got {type(box).__name__} for id={box_id}"
            )
        index = _box_id_suffix_index(box_id)
        if index in seen_indices:
            raise ValueError(
                "Duplicate numeric suffix in ordered_box_ids: "
                f"index={index} (box id: {box_id})"
            )
        seen_indices.add(index)

        cropped = extractor(
            normalized_volume,
            z_bounds=(int(box.z0), int(box.z1)),
            y_bounds=(int(box.y0), int(box.y1)),
            x_bounds=(int(box.x0), int(box.x1)),
        )
        saved_path = saver(
            cropped,
            output_dir=output_dir,
            index=index,
            label=box.label,
            overwrite=bool(overwrite),
        )
        saved_paths.append(str(saved_path))
    return tuple(saved_paths)


def export_raw_bboxes_as_tiff(
    volume_array: np.ndarray,
    *,
    boxes_by_id: Mapping[str, BoundingBox],
    ordered_box_ids: Sequence[str],
    output_dir: str,
    overwrite: bool = True,
    extractor: Optional[Callable[..., np.ndarray]] = None,
    saver: Optional[Callable[..., str]] = None,
) -> RawBBoxExportOutcome:
    normalized_volume = _coerce_3d_array(volume_array)
    _ensure_no_singleton_axis(
        (
            int(normalized_volume.shape[0]),
            int(normalized_volume.shape[1]),
            int(normalized_volume.shape[2]),
        )
    )
    if extractor is None:
        def extractor_impl(
            array: np.ndarray,
            *,
            z_bounds: Tuple[int, int],
            y_bounds: Tuple[int, int],
            x_bounds: Tuple[int, int],
        ) -> np.ndarray:
            return extract_bbox_context_from_array(
                array,
                z_bounds=z_bounds,
                y_bounds=y_bounds,
                x_bounds=x_bounds,
            )
        extractor = extractor_impl
    if saver is None:
        def saver_impl(
            array: np.ndarray,
            *,
            output_dir: str,
            index: int,
            label: str,
            overwrite: bool,
        ) -> str:
            filename = build_bbox_tiff_filename(index, label)
            return _save_array_as_tiff(
                array,
                output_dir=output_dir,
                filename=filename,
                overwrite=overwrite,
            )
        saver = saver_impl

    saved_paths = []
    failed_boxes = []
    seen_ids = set()
    seen_indices = set()

    for raw_box_id in ordered_box_ids:
        box_id = ""
        index = -1
        label = "train"
        try:
            box_id = _normalize_box_id(raw_box_id)
        except Exception as exc:
            failed_boxes.append(
                RawBBoxSaveFailure(
                    box_id=str(raw_box_id),
                    index=index,
                    label=label,
                    error=str(exc),
                )
            )
            continue

        if box_id in seen_ids:
            failed_boxes.append(
                RawBBoxSaveFailure(
                    box_id=box_id,
                    index=index,
                    label=label,
                    error=f"Duplicate box id in ordered_box_ids: {box_id}",
                )
            )
            continue
        seen_ids.add(box_id)

        box = boxes_by_id.get(box_id)
        if box is None:
            try:
                index = _box_id_suffix_index(box_id)
            except Exception:
                index = -1
            failed_boxes.append(
                RawBBoxSaveFailure(
                    box_id=box_id,
                    index=index,
                    label=label,
                    error=f"Unknown bounding box id in ordered_box_ids: {box_id}",
                )
            )
            continue
        if not isinstance(box, BoundingBox):
            failed_boxes.append(
                RawBBoxSaveFailure(
                    box_id=box_id,
                    index=index,
                    label=label,
                    error=(
                        "boxes_by_id must map ids to BoundingBox, got "
                        f"{type(box).__name__} for id={box_id}"
                    ),
                )
            )
            continue
        label = str(box.label)

        try:
            index = _box_id_suffix_index(box_id)
        except Exception as exc:
            failed_boxes.append(
                RawBBoxSaveFailure(
                    box_id=box_id,
                    index=index,
                    label=label,
                    error=str(exc),
                )
            )
            continue
        if index in seen_indices:
            failed_boxes.append(
                RawBBoxSaveFailure(
                    box_id=box_id,
                    index=index,
                    label=label,
                    error=(
                        "Duplicate numeric suffix in ordered_box_ids: "
                        f"index={index} (box id: {box_id})"
                    ),
                )
            )
            continue
        seen_indices.add(index)

        z_bounds = (int(box.z0), int(box.z1))
        y_bounds = (int(box.y0), int(box.y1))
        x_bounds = (int(box.x0), int(box.x1))

        try:
            context_array = extractor(
                normalized_volume,
                z_bounds=z_bounds,
                y_bounds=y_bounds,
                x_bounds=x_bounds,
            )
            saved_path = saver(
                context_array,
                output_dir=output_dir,
                index=index,
                label=label,
                overwrite=bool(overwrite),
            )
            saved_paths.append(str(saved_path))
        except Exception as exc:
            failed_boxes.append(
                RawBBoxSaveFailure(
                    box_id=box_id,
                    index=index,
                    label=label,
                    error=str(exc),
                )
            )
            continue

    return RawBBoxExportOutcome(
        saved_paths=tuple(saved_paths),
        failed_boxes=tuple(failed_boxes),
    )


def export_segmentation_bboxes_as_tiff(
    segmentation_array: np.ndarray,
    *,
    boxes_by_id: Mapping[str, BoundingBox],
    ordered_box_ids: Sequence[str],
    output_dir: str,
    context_fill_value: int = -100,
    overwrite: bool = True,
    zero_only_checker: Optional[Callable[..., bool]] = None,
    context_builder: Optional[Callable[..., np.ndarray]] = None,
    saver: Optional[Callable[..., str]] = None,
) -> SegmentationBBoxExportOutcome:
    normalized_segmentation = _coerce_3d_array(segmentation_array)
    if not np.issubdtype(normalized_segmentation.dtype, np.integer):
        raise ValueError("segmentation_array must have an integer dtype")

    if zero_only_checker is None:
        def zero_only_checker_impl(
            array: np.ndarray,
            *,
            z_bounds: Tuple[int, int],
            y_bounds: Tuple[int, int],
            x_bounds: Tuple[int, int],
        ) -> bool:
            return is_segmentation_bbox_zero_only(
                array,
                z_bounds=z_bounds,
                y_bounds=y_bounds,
                x_bounds=x_bounds,
            )
        zero_only_checker = zero_only_checker_impl

    if context_builder is None:
        def context_builder_impl(
            array: np.ndarray,
            *,
            z_bounds: Tuple[int, int],
            y_bounds: Tuple[int, int],
            x_bounds: Tuple[int, int],
            context_fill_value: int,
        ) -> np.ndarray:
            return build_segmentation_bbox_context_from_array(
                array,
                z_bounds=z_bounds,
                y_bounds=y_bounds,
                x_bounds=x_bounds,
                context_fill_value=context_fill_value,
            )
        context_builder = context_builder_impl

    if saver is None:
        def saver_impl(
            array: np.ndarray,
            *,
            output_dir: str,
            index: int,
            label: str,
            overwrite: bool,
        ) -> str:
            filename = build_segmentation_bbox_tiff_filename(index, label)
            return _save_array_as_tiff(
                array,
                output_dir=output_dir,
                filename=filename,
                overwrite=overwrite,
            )
        saver = saver_impl

    saved_paths = []
    skipped_zero_box_ids = []
    failed_boxes = []
    seen_ids = set()
    seen_indices = set()

    for raw_box_id in ordered_box_ids:
        box_id = ""
        index = -1
        label = "train"
        try:
            box_id = _normalize_box_id(raw_box_id)
        except Exception as exc:
            failed_boxes.append(
                SegmentationBBoxSaveFailure(
                    box_id=str(raw_box_id),
                    index=index,
                    label=label,
                    error=str(exc),
                )
            )
            continue

        if box_id in seen_ids:
            failed_boxes.append(
                SegmentationBBoxSaveFailure(
                    box_id=box_id,
                    index=index,
                    label=label,
                    error=f"Duplicate box id in ordered_box_ids: {box_id}",
                )
            )
            continue
        seen_ids.add(box_id)

        box = boxes_by_id.get(box_id)
        if box is None:
            try:
                index = _box_id_suffix_index(box_id)
            except Exception:
                index = -1
            failed_boxes.append(
                SegmentationBBoxSaveFailure(
                    box_id=box_id,
                    index=index,
                    label=label,
                    error=f"Unknown bounding box id in ordered_box_ids: {box_id}",
                )
            )
            continue
        if not isinstance(box, BoundingBox):
            failed_boxes.append(
                SegmentationBBoxSaveFailure(
                    box_id=box_id,
                    index=index,
                    label=label,
                    error=(
                        "boxes_by_id must map ids to BoundingBox, got "
                        f"{type(box).__name__} for id={box_id}"
                    ),
                )
            )
            continue
        label = str(box.label)

        try:
            index = _box_id_suffix_index(box_id)
        except Exception as exc:
            failed_boxes.append(
                SegmentationBBoxSaveFailure(
                    box_id=box_id,
                    index=index,
                    label=label,
                    error=str(exc),
                )
            )
            continue
        if index in seen_indices:
            failed_boxes.append(
                SegmentationBBoxSaveFailure(
                    box_id=box_id,
                    index=index,
                    label=label,
                    error=(
                        "Duplicate numeric suffix in ordered_box_ids: "
                        f"index={index} (box id: {box_id})"
                    ),
                )
            )
            continue
        seen_indices.add(index)

        z_bounds = (int(box.z0), int(box.z1))
        y_bounds = (int(box.y0), int(box.y1))
        x_bounds = (int(box.x0), int(box.x1))

        try:
            if zero_only_checker(
                normalized_segmentation,
                z_bounds=z_bounds,
                y_bounds=y_bounds,
                x_bounds=x_bounds,
            ):
                skipped_zero_box_ids.append(box_id)
                continue

            context_array = context_builder(
                normalized_segmentation,
                z_bounds=z_bounds,
                y_bounds=y_bounds,
                x_bounds=x_bounds,
                context_fill_value=int(context_fill_value),
            )
            saved_path = saver(
                context_array,
                output_dir=output_dir,
                index=index,
                label=label,
                overwrite=bool(overwrite),
            )
            saved_paths.append(str(saved_path))
        except Exception as exc:
            failed_boxes.append(
                SegmentationBBoxSaveFailure(
                    box_id=box_id,
                    index=index,
                    label=label,
                    error=str(exc),
                )
            )
            continue

    return SegmentationBBoxExportOutcome(
        saved_paths=tuple(saved_paths),
        skipped_zero_box_ids=tuple(skipped_zero_box_ids),
        failed_boxes=tuple(failed_boxes),
    )


def extract_learning_bbox_tensors(
    raw_array: np.ndarray,
    segmentation_array: np.ndarray,
    *,
    boxes_by_id: Mapping[str, BoundingBox],
    ordered_box_ids: Sequence[str],
    context_fill_value: int = -100,
    store_in_session: bool = True,
    raw_context_extractor: Optional[Callable[..., np.ndarray]] = None,
    segmentation_context_builder: Optional[Callable[..., np.ndarray]] = None,
) -> LearningBBoxTensorBatch:
    normalized_raw = _coerce_3d_array(raw_array)
    normalized_segmentation = _coerce_3d_array(segmentation_array)
    if tuple(int(v) for v in normalized_raw.shape) != tuple(
        int(v) for v in normalized_segmentation.shape
    ):
        raise ValueError(
            "raw_array and segmentation_array must share the same shape: "
            f"raw_shape={tuple(normalized_raw.shape)} "
            f"seg_shape={tuple(normalized_segmentation.shape)}"
        )
    if not np.issubdtype(normalized_segmentation.dtype, np.integer):
        raise ValueError("segmentation_array must have an integer dtype")

    if raw_context_extractor is None:
        def raw_extractor_impl(
            array: np.ndarray,
            *,
            z_bounds: Tuple[int, int],
            y_bounds: Tuple[int, int],
            x_bounds: Tuple[int, int],
        ) -> np.ndarray:
            return extract_bbox_context_from_array(
                array,
                z_bounds=z_bounds,
                y_bounds=y_bounds,
                x_bounds=x_bounds,
            )
        raw_context_extractor = raw_extractor_impl

    if segmentation_context_builder is None:
        def seg_builder_impl(
            array: np.ndarray,
            *,
            z_bounds: Tuple[int, int],
            y_bounds: Tuple[int, int],
            x_bounds: Tuple[int, int],
            context_fill_value: int,
        ) -> np.ndarray:
            return build_segmentation_bbox_context_from_array(
                array,
                z_bounds=z_bounds,
                y_bounds=y_bounds,
                x_bounds=x_bounds,
                context_fill_value=context_fill_value,
            )
        segmentation_context_builder = seg_builder_impl

    torch = _require_torch()
    normalized_fill = _coerce_int(context_fill_value, name="context_fill_value")
    seen_ids = set()
    seen_indices = set()
    entries = []

    for raw_box_id in ordered_box_ids:
        box_id = _normalize_box_id(raw_box_id)
        box = boxes_by_id.get(box_id)
        if box is None:
            raise KeyError(f"Unknown bounding box id in ordered_box_ids: {box_id}")
        if not isinstance(box, BoundingBox):
            raise TypeError(
                f"boxes_by_id must map ids to BoundingBox, got {type(box).__name__} for id={box_id}"
            )
        if str(box.label) == "inference":
            continue
        if box_id in seen_ids:
            raise ValueError(f"Duplicate box id in ordered_box_ids: {box_id}")
        seen_ids.add(box_id)

        index = _box_id_suffix_index(box_id)
        if index in seen_indices:
            raise ValueError(
                "Duplicate numeric suffix in ordered_box_ids: "
                f"index={index} (box id: {box_id})"
            )
        seen_indices.add(index)

        z_bounds = (int(box.z0), int(box.z1))
        y_bounds = (int(box.y0), int(box.y1))
        x_bounds = (int(box.x0), int(box.x1))

        raw_context = raw_context_extractor(
            normalized_raw,
            z_bounds=z_bounds,
            y_bounds=y_bounds,
            x_bounds=x_bounds,
        )
        seg_context = segmentation_context_builder(
            normalized_segmentation,
            z_bounds=z_bounds,
            y_bounds=y_bounds,
            x_bounds=x_bounds,
            context_fill_value=normalized_fill,
        )

        raw_tensor = torch_from_numpy_safe(raw_context, torch_module=torch).to(
            dtype=torch.float16
        )
        seg_torch_dtype = _torch_signed_integer_dtype_for_numpy(np.asarray(seg_context).dtype)
        segmentation_tensor = torch_from_numpy_safe(
            seg_context,
            torch_module=torch,
        ).to(dtype=seg_torch_dtype)
        entry = LearningBBoxTensorEntry(
            box_id=box_id,
            index=index,
            label=box.label,
            raw_tensor=raw_tensor,
            segmentation_tensor=segmentation_tensor,
        )
        entries.append(entry)

    if store_in_session:
        return set_current_learning_bbox_entries(tuple(entries))
    return LearningBBoxTensorBatch(entries=tuple(entries))


def extract_learning_bboxes_in_memory(
    raw_array: np.ndarray,
    segmentation_array: np.ndarray,
    *,
    boxes_by_id: Mapping[str, BoundingBox],
    ordered_box_ids: Sequence[str],
    context_fill_value: int = -100,
    store_in_session: bool = True,
    build_learning_dataloader: bool = True,
    build_eval_dataloaders: bool = False,
    learning_minivol_size: Optional[int] = None,
    learning_minivol_per_epoch: Optional[int] = None,
    learning_batch_size: int = 4,
    learning_num_workers: int = 8,
    learning_pin_memory: bool = True,
    learning_drop_last: bool = True,
    eval_minivol_size: Optional[int] = None,
    eval_batch_size: int = 4,
    eval_num_workers: int = 8,
    eval_pin_memory: bool = True,
    eval_drop_last: bool = False,
    learning_dataset_factory: Optional[Callable[..., object]] = None,
    learning_sampler_factory: Optional[Callable[..., object]] = None,
    learning_dataloader_factory: Optional[Callable[..., object]] = None,
    eval_dataset_factory: Optional[Callable[..., object]] = None,
    eval_dataloader_factory: Optional[Callable[..., object]] = None,
    eval_buffer_factory: Optional[Callable[..., object]] = None,
    raw_context_extractor: Optional[Callable[..., np.ndarray]] = None,
    segmentation_context_builder: Optional[Callable[..., np.ndarray]] = None,
) -> LearningBBoxExtractionOutcome:
    if bool(store_in_session) and bool(build_learning_dataloader):
        clear_current_learning_dataloader_runtime()
    if bool(store_in_session) and bool(build_eval_dataloaders):
        clear_current_learning_eval_runtimes_by_box_id()

    batch = extract_learning_bbox_tensors(
        raw_array,
        segmentation_array,
        boxes_by_id=boxes_by_id,
        ordered_box_ids=ordered_box_ids,
        context_fill_value=context_fill_value,
        store_in_session=store_in_session,
        raw_context_extractor=raw_context_extractor,
        segmentation_context_builder=segmentation_context_builder,
    )

    learning_runtime = None
    eval_runtimes_by_box_id: Mapping[str, object] = {}
    if build_learning_dataloader:
        train_entries = tuple(entry for entry in batch.entries if str(entry.label) == "train")
        if not train_entries:
            raise ValueError(
                "No bounding boxes labeled 'train' are available to build a learning dataloader."
            )
        resolved_minivol_size = _resolve_fixed_learning_minivol_size(
            learning_minivol_size,
            name="learning_minivol_size",
        )

        if learning_minivol_per_epoch is None:
            resolved_minivol_per_epoch = 1000
        else:
            resolved_minivol_per_epoch = int(learning_minivol_per_epoch)

        learning_runtime = build_learning_dataloader_from_batch(
            batch,
            minivol_size=resolved_minivol_size,
            minivol_per_epoch=resolved_minivol_per_epoch,
            batch_size=learning_batch_size,
            num_workers=learning_num_workers,
            pin_memory=learning_pin_memory,
            drop_last=learning_drop_last,
            dataset_factory=learning_dataset_factory,
            sampler_factory=learning_sampler_factory,
            dataloader_factory=learning_dataloader_factory,
            store_in_session=store_in_session,
        )

    if build_eval_dataloaders:
        resolved_eval_minivol_size = _resolve_fixed_learning_minivol_size(
            eval_minivol_size,
            name="eval_minivol_size",
        )
        try:
            eval_runtimes_by_box_id = build_eval_dataloader_runtimes_from_batch(
                batch,
                minivol_size=resolved_eval_minivol_size,
                batch_size=eval_batch_size,
                num_workers=eval_num_workers,
                pin_memory=eval_pin_memory,
                drop_last=eval_drop_last,
                dataset_factory=eval_dataset_factory,
                dataloader_factory=eval_dataloader_factory,
                buffer_factory=eval_buffer_factory,
                store_in_session=store_in_session,
            )
        except Exception:
            # Eval runtime build is part of one atomic extraction action:
            # if it fails, rollback any newly built training runtime.
            if bool(store_in_session):
                if bool(build_learning_dataloader):
                    clear_current_learning_dataloader_runtime()
                clear_current_learning_eval_runtimes_by_box_id()
            raise

    return LearningBBoxExtractionOutcome(
        raw_saved_paths=tuple(),
        raw_failed_boxes=tuple(),
        segmentation_saved_paths=tuple(),
        segmentation_skipped_zero_box_ids=tuple(),
        segmentation_failed_boxes=tuple(),
        tensor_entry_count=int(batch.size),
        learning_train_box_ids=(
            tuple(learning_runtime.train_box_ids)
            if learning_runtime is not None
            else tuple()
        ),
        learning_batch_size=(
            int(learning_runtime.batch_size)
            if learning_runtime is not None and learning_runtime.batch_size is not None
            else None
        ),
        learning_num_workers=(
            int(learning_runtime.num_workers)
            if learning_runtime is not None and learning_runtime.num_workers is not None
            else None
        ),
        eval_validation_box_ids=tuple(eval_runtimes_by_box_id.keys()),
        eval_batch_size=int(eval_batch_size) if bool(build_eval_dataloaders) else None,
        eval_num_workers=int(eval_num_workers) if bool(build_eval_dataloaders) else None,
    )


def extract_bboxes_for_learning(
    raw_array: np.ndarray,
    segmentation_array: np.ndarray,
    *,
    boxes_by_id: Mapping[str, BoundingBox],
    ordered_box_ids: Sequence[str],
    output_dir: str = "",
    context_fill_value: int = -100,
    overwrite: bool = True,
    store_in_session: bool = True,
    build_learning_dataloader: bool = True,
    build_eval_dataloaders: bool = False,
    learning_minivol_size: Optional[int] = None,
    learning_minivol_per_epoch: Optional[int] = None,
    learning_batch_size: int = 4,
    learning_num_workers: int = 8,
    learning_pin_memory: bool = True,
    learning_drop_last: bool = True,
    eval_minivol_size: Optional[int] = None,
    eval_batch_size: int = 4,
    eval_num_workers: int = 8,
    eval_pin_memory: bool = True,
    eval_drop_last: bool = False,
    learning_dataset_factory: Optional[Callable[..., object]] = None,
    learning_sampler_factory: Optional[Callable[..., object]] = None,
    learning_dataloader_factory: Optional[Callable[..., object]] = None,
    eval_dataset_factory: Optional[Callable[..., object]] = None,
    eval_dataloader_factory: Optional[Callable[..., object]] = None,
    eval_buffer_factory: Optional[Callable[..., object]] = None,
    raw_context_extractor: Optional[Callable[..., np.ndarray]] = None,
    segmentation_context_builder: Optional[Callable[..., np.ndarray]] = None,
    raw_saver: Optional[Callable[..., str]] = None,
    segmentation_saver: Optional[Callable[..., str]] = None,
) -> LearningBBoxExtractionOutcome:
    # Legacy compatibility wrapper. Filesystem export args are ignored because
    # learning extraction is now purely in-memory.
    del output_dir, overwrite, raw_saver, segmentation_saver
    return extract_learning_bboxes_in_memory(
        raw_array,
        segmentation_array,
        boxes_by_id=boxes_by_id,
        ordered_box_ids=ordered_box_ids,
        context_fill_value=context_fill_value,
        store_in_session=store_in_session,
        build_learning_dataloader=build_learning_dataloader,
        build_eval_dataloaders=build_eval_dataloaders,
        learning_minivol_size=learning_minivol_size,
        learning_minivol_per_epoch=learning_minivol_per_epoch,
        learning_batch_size=learning_batch_size,
        learning_num_workers=learning_num_workers,
        learning_pin_memory=learning_pin_memory,
        learning_drop_last=learning_drop_last,
        eval_minivol_size=eval_minivol_size,
        eval_batch_size=eval_batch_size,
        eval_num_workers=eval_num_workers,
        eval_pin_memory=eval_pin_memory,
        eval_drop_last=eval_drop_last,
        learning_dataset_factory=learning_dataset_factory,
        learning_sampler_factory=learning_sampler_factory,
        learning_dataloader_factory=learning_dataloader_factory,
        eval_dataset_factory=eval_dataset_factory,
        eval_dataloader_factory=eval_dataloader_factory,
        eval_buffer_factory=eval_buffer_factory,
        raw_context_extractor=raw_context_extractor,
        segmentation_context_builder=segmentation_context_builder,
    )
