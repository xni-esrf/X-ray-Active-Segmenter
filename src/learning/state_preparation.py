from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..bbox import BoundingBox, load_bounding_boxes
from ..data import VolumeData
from ..io import extract_learning_bboxes_in_memory
from ..loading import load_prepared_volume
from .class_weights import compute_and_store_current_learning_class_weights
from .label_coverage import (
    compute_learning_label_coverage,
    format_learning_label_coverage_warning,
)
from .label_space import (
    LearningLabelSpace,
    derive_label_space_from_semantic_segmentation,
)
from .model_instantiation import instantiate_foundation_model_runtime
from .model_training import validate_learning_model_training_preconditions
from .session_store import LearningSession
from .session_store import (
    clear_current_learning_bbox_batch,
    get_current_learning_bbox_batch,
    set_current_learning_label_space,
)
from .training_parameters import (
    DEFAULT_TRAINING_PARAMETERS,
    TrainingParameters,
    validate_training_parameters,
)


@dataclass(frozen=True)
class LearningSourceBundle:
    raw_volume: VolumeData
    segmentation_volume: VolumeData
    segmentation_kind: str
    boxes_by_id: Mapping[str, BoundingBox]
    ordered_box_ids: Tuple[str, ...]

    def close(self) -> None:
        for volume in (self.segmentation_volume, self.raw_volume):
            try:
                volume.close()
            except Exception:
                pass


@dataclass(frozen=True)
class LearningStatePreparationResult:
    outcome: object
    label_space: LearningLabelSpace
    class_weights: Optional[object]
    label_coverage_warning: Optional[str]
    residual_entry_count: int
    learning_box_ids: Tuple[str, ...]
    train_box_ids: Tuple[str, ...]
    validation_box_ids: Tuple[str, ...]


def load_learning_sources_from_paths(
    *,
    raw_volume_path: str,
    segmentation_path: str,
    segmentation_kind: str,
    bbox_path: str,
    load_mode: str,
    cache_max_bytes: int,
) -> LearningSourceBundle:
    normalized_kind = _normalize_segmentation_kind(segmentation_kind)
    raw = load_prepared_volume(
        raw_volume_path,
        kind="raw",
        load_mode=load_mode,
        cache_max_bytes=cache_max_bytes,
        pyramid_levels=1,
    )
    segmentation = load_prepared_volume(
        segmentation_path,
        kind=normalized_kind,
        load_mode=load_mode,
        cache_max_bytes=cache_max_bytes,
        pyramid_levels=1,
    )
    if raw.volume.shape != segmentation.volume.shape:
        raise ValueError(
            "Segmentation shape does not match raw volume shape: "
            f"raw={raw.volume.shape} segmentation={segmentation.volume.shape}"
        )
    bbox_payload = load_bounding_boxes(
        bbox_path,
        expected_shape=raw.volume.shape,
    )
    boxes = tuple(bbox_payload.boxes)
    return LearningSourceBundle(
        raw_volume=raw.volume,
        segmentation_volume=segmentation.volume,
        segmentation_kind=normalized_kind,
        boxes_by_id={box.id: box for box in boxes},
        ordered_box_ids=tuple(box.id for box in boxes),
    )


def semantic_label_space_source_signature(
    *,
    semantic_kind: str,
    semantic_volume: object,
    semantic_state_id: Optional[int] = None,
) -> Tuple[object, ...]:
    semantic_source_path: Optional[str] = None
    loader = getattr(semantic_volume, "loader", None)
    path_obj = getattr(loader, "path", None)
    if isinstance(path_obj, str) and path_obj.strip():
        semantic_source_path = str(path_obj)
    return (
        str(semantic_kind),
        semantic_source_path,
        None if semantic_state_id is None else int(semantic_state_id),
    )


def prepare_learning_state_from_sources(
    sources: LearningSourceBundle,
    *,
    training_parameters: TrainingParameters = DEFAULT_TRAINING_PARAMETERS,
    require_class_weights: bool,
    learning_session: Optional[LearningSession] = None,
    label_space_source_signature: Optional[Tuple[object, ...]] = None,
    class_weights_device: str = "cuda:0",
    extract_learning_bboxes_in_memory_fn: Callable[..., object] = (
        extract_learning_bboxes_in_memory
    ),
    compute_class_weights_fn: Callable[..., object] = (
        compute_and_store_current_learning_class_weights
    ),
    compute_label_coverage_fn: Callable[..., object] = compute_learning_label_coverage,
    format_label_coverage_warning_fn: Callable[[object], Optional[str]] = (
        format_learning_label_coverage_warning
    ),
    derive_label_space_fn: Callable[..., LearningLabelSpace] = (
        derive_label_space_from_semantic_segmentation
    ),
    clear_learning_bbox_batch_fn: Optional[Callable[[], None]] = None,
    get_learning_bbox_batch_fn: Optional[Callable[[], object]] = None,
    learning_num_workers: int = 8,
    learning_pin_memory: bool = True,
    learning_drop_last: bool = True,
    eval_num_workers: int = 8,
    eval_pin_memory: bool = True,
    eval_drop_last: bool = False,
) -> LearningStatePreparationResult:
    return prepare_learning_state_from_volumes(
        raw_volume=sources.raw_volume,
        segmentation_volume=sources.segmentation_volume,
        segmentation_kind=sources.segmentation_kind,
        boxes_by_id=sources.boxes_by_id,
        ordered_box_ids=sources.ordered_box_ids,
        training_parameters=training_parameters,
        require_class_weights=require_class_weights,
        learning_session=learning_session,
        label_space_source_signature=label_space_source_signature,
        class_weights_device=class_weights_device,
        extract_learning_bboxes_in_memory_fn=extract_learning_bboxes_in_memory_fn,
        compute_class_weights_fn=compute_class_weights_fn,
        compute_label_coverage_fn=compute_label_coverage_fn,
        format_label_coverage_warning_fn=format_label_coverage_warning_fn,
        derive_label_space_fn=derive_label_space_fn,
        clear_learning_bbox_batch_fn=clear_learning_bbox_batch_fn,
        get_learning_bbox_batch_fn=get_learning_bbox_batch_fn,
        learning_num_workers=learning_num_workers,
        learning_pin_memory=learning_pin_memory,
        learning_drop_last=learning_drop_last,
        eval_num_workers=eval_num_workers,
        eval_pin_memory=eval_pin_memory,
        eval_drop_last=eval_drop_last,
    )


def prepare_learning_state_from_volumes(
    *,
    raw_volume: VolumeData,
    segmentation_volume: VolumeData,
    segmentation_kind: str,
    boxes_by_id: Mapping[str, BoundingBox],
    ordered_box_ids: Sequence[str],
    training_parameters: TrainingParameters = DEFAULT_TRAINING_PARAMETERS,
    require_class_weights: bool,
    learning_session: Optional[LearningSession] = None,
    label_space_source_signature: Optional[Tuple[object, ...]] = None,
    class_weights_device: str = "cuda:0",
    extract_learning_bboxes_in_memory_fn: Callable[..., object] = (
        extract_learning_bboxes_in_memory
    ),
    compute_class_weights_fn: Callable[..., object] = (
        compute_and_store_current_learning_class_weights
    ),
    compute_label_coverage_fn: Callable[..., object] = compute_learning_label_coverage,
    format_label_coverage_warning_fn: Callable[[object], Optional[str]] = (
        format_learning_label_coverage_warning
    ),
    derive_label_space_fn: Callable[..., LearningLabelSpace] = (
        derive_label_space_from_semantic_segmentation
    ),
    clear_learning_bbox_batch_fn: Optional[Callable[[], None]] = None,
    get_learning_bbox_batch_fn: Optional[Callable[[], object]] = None,
    learning_num_workers: int = 8,
    learning_pin_memory: bool = True,
    learning_drop_last: bool = True,
    eval_num_workers: int = 8,
    eval_pin_memory: bool = True,
    eval_drop_last: bool = False,
) -> LearningStatePreparationResult:
    normalized_segmentation_kind = _normalize_segmentation_kind(segmentation_kind)
    if normalized_segmentation_kind != "semantic":
        raise ValueError(
            "Only semantic segmentation is supported for learning-state preparation."
        )
    normalized_training_parameters = validate_training_parameters(training_parameters)
    normalized_ordered_box_ids = _normalize_ordered_box_ids(ordered_box_ids)
    normalized_boxes_by_id = _normalize_boxes_by_id(boxes_by_id)
    learning_box_ids, train_box_ids, validation_box_ids = _resolve_learning_box_ids(
        boxes_by_id=normalized_boxes_by_id,
        ordered_box_ids=normalized_ordered_box_ids,
    )
    raw_shape = _volume_shape(raw_volume)
    segmentation_shape = _volume_shape(segmentation_volume)
    if (
        raw_shape is not None
        and segmentation_shape is not None
        and raw_shape != segmentation_shape
    ):
        raise ValueError(
            "Segmentation shape does not match raw volume shape: "
            f"raw={raw_shape} segmentation={segmentation_shape}"
        )

    try:
        raw_array = np.asarray(
            raw_volume.get_chunk((slice(None), slice(None), slice(None)))
        )
        segmentation_array = np.asarray(
            segmentation_volume.get_chunk((slice(None), slice(None), slice(None)))
        )
        if label_space_source_signature is None:
            label_space_source_signature = semantic_label_space_source_signature(
                semantic_kind=normalized_segmentation_kind,
                semantic_volume=segmentation_volume,
            )
        label_space = derive_label_space_fn(
            segmentation_array,
            source_signature=label_space_source_signature,
        )
        if learning_session is not None:
            learning_session.set_label_space(label_space)
        else:
            set_current_learning_label_space(label_space)

        extract_kwargs = {
            "boxes_by_id": normalized_boxes_by_id,
            "ordered_box_ids": learning_box_ids,
            "learning_minivol_per_epoch": (
                normalized_training_parameters.patches_per_epoch
            ),
            "learning_batch_size": (
                normalized_training_parameters.training_batch_size
            ),
            "learning_num_workers": learning_num_workers,
            "learning_pin_memory": learning_pin_memory,
            "learning_drop_last": learning_drop_last,
            "build_eval_dataloaders": True,
            "eval_batch_size": (
                normalized_training_parameters.validation_batch_size
            ),
            "eval_num_workers": eval_num_workers,
            "eval_pin_memory": eval_pin_memory,
            "eval_drop_last": eval_drop_last,
        }
        if learning_session is not None:
            extract_kwargs["learning_session"] = learning_session
        outcome = extract_learning_bboxes_in_memory_fn(
            raw_array,
            segmentation_array,
            **extract_kwargs,
        )
        class_weights = None
        if bool(require_class_weights):
            weights_kwargs = {
                "max_weight": 100.0,
                "device": class_weights_device,
            }
            if learning_session is not None:
                weights_kwargs["learning_session"] = learning_session
            class_weights = compute_class_weights_fn(**weights_kwargs)
        label_coverage_warning = _compute_label_coverage_warning(
            learning_session=learning_session,
            compute_label_coverage_fn=compute_label_coverage_fn,
            format_label_coverage_warning_fn=format_label_coverage_warning_fn,
        )
        _clear_learning_bbox_batch(
            learning_session,
            clear_learning_bbox_batch_fn=clear_learning_bbox_batch_fn,
        )
        residual_batch = _get_learning_bbox_batch(
            learning_session,
            get_learning_bbox_batch_fn=get_learning_bbox_batch_fn,
        )
        residual_entry_count = (
            int(residual_batch.size) if residual_batch is not None else 0
        )
    except Exception:
        _clear_learning_bbox_batch(
            learning_session,
            clear_learning_bbox_batch_fn=clear_learning_bbox_batch_fn,
        )
        raise

    if residual_entry_count > 0:
        raise RuntimeError(
            "Dataset build completed, but temporary learning tensors were "
            f"not fully released ({residual_entry_count} entries remain in session)."
        )

    return LearningStatePreparationResult(
        outcome=outcome,
        label_space=label_space,
        class_weights=class_weights,
        label_coverage_warning=label_coverage_warning,
        residual_entry_count=residual_entry_count,
        learning_box_ids=learning_box_ids,
        train_box_ids=train_box_ids,
        validation_box_ids=validation_box_ids,
    )


def instantiate_model_runtime_from_checkpoint(
    *,
    checkpoint_path: str,
    num_classes: int,
    device_ids: Optional[Sequence[object]] = None,
    learning_session: Optional[LearningSession] = None,
) -> object:
    return instantiate_foundation_model_runtime(
        num_classes=num_classes,
        checkpoint_path=checkpoint_path,
        device_ids=device_ids,
        learning_session=learning_session,
    )


def validate_training_preconditions_for_session(
    *,
    require_class_weights: bool = True,
    learning_session: Optional[LearningSession] = None,
) -> object:
    return validate_learning_model_training_preconditions(
        require_class_weights=require_class_weights,
        learning_session=learning_session,
    )


def _normalize_segmentation_kind(value: object) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"semantic", "instance"}:
        raise ValueError("segmentation_kind must be 'semantic' or 'instance'")
    return normalized


def _volume_shape(volume: object) -> Optional[Tuple[int, int, int]]:
    shape = getattr(volume, "shape", None)
    if shape is None:
        info = getattr(volume, "info", None)
        shape = getattr(info, "shape", None)
    if shape is None:
        return None
    try:
        normalized = tuple(int(dim) for dim in tuple(shape))
    except Exception:
        return None
    if len(normalized) != 3:
        return None
    return normalized


def _normalize_ordered_box_ids(values: Sequence[str]) -> Tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("ordered_box_ids must be a sequence of box ids")
    normalized = []
    for value in tuple(values):
        box_id = str(value).strip()
        if not box_id:
            raise ValueError("ordered_box_ids must not contain empty ids")
        normalized.append(box_id)
    if not normalized:
        raise ValueError("There are no bounding boxes to build datasets from.")
    return tuple(normalized)


def _normalize_boxes_by_id(
    boxes_by_id: Mapping[str, BoundingBox],
) -> Mapping[str, BoundingBox]:
    if not isinstance(boxes_by_id, Mapping):
        raise TypeError("boxes_by_id must be a mapping of id to BoundingBox")
    normalized = {}
    for raw_box_id, box in tuple(boxes_by_id.items()):
        box_id = str(raw_box_id).strip()
        if not box_id:
            raise ValueError("boxes_by_id contains an empty id")
        if not isinstance(box, BoundingBox):
            raise TypeError(
                "boxes_by_id values must be BoundingBox instances, "
                f"got {type(box).__name__} for id={box_id!r}"
            )
        normalized[box_id] = box
    return normalized


def _resolve_learning_box_ids(
    *,
    boxes_by_id: Mapping[str, BoundingBox],
    ordered_box_ids: Sequence[str],
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    learning_box_ids = tuple(
        box_id
        for box_id in ordered_box_ids
        if box_id in boxes_by_id and str(boxes_by_id[box_id].label) != "inference"
    )
    train_box_ids = tuple(
        box_id
        for box_id in learning_box_ids
        if box_id in boxes_by_id and str(boxes_by_id[box_id].label) == "train"
    )
    if not train_box_ids:
        raise ValueError(
            "At least one bounding box labeled 'train' is required to build "
            "datasets from bboxes."
        )
    validation_box_ids = tuple(
        box_id
        for box_id in learning_box_ids
        if box_id in boxes_by_id and str(boxes_by_id[box_id].label) == "validation"
    )
    if not validation_box_ids:
        raise ValueError(
            "At least one bounding box labeled 'validation' is required to "
            "build datasets from bboxes."
        )
    return learning_box_ids, train_box_ids, validation_box_ids


def _compute_label_coverage_warning(
    *,
    learning_session: Optional[LearningSession],
    compute_label_coverage_fn: Callable[..., object],
    format_label_coverage_warning_fn: Callable[[object], Optional[str]],
) -> Optional[str]:
    try:
        coverage_kwargs = {}
        if learning_session is not None:
            coverage_kwargs["learning_session"] = learning_session
        label_coverage = compute_label_coverage_fn(**coverage_kwargs)
        return format_label_coverage_warning_fn(label_coverage)
    except ValueError as exc:
        if not str(exc).startswith("No "):
            raise
        return None


def _clear_learning_bbox_batch(
    learning_session: Optional[LearningSession],
    *,
    clear_learning_bbox_batch_fn: Optional[Callable[[], None]] = None,
) -> None:
    if clear_learning_bbox_batch_fn is not None:
        clear_learning_bbox_batch_fn()
        return
    if learning_session is not None:
        learning_session.clear_bbox_batch()
        return
    clear_current_learning_bbox_batch()


def _get_learning_bbox_batch(
    learning_session: Optional[LearningSession],
    *,
    get_learning_bbox_batch_fn: Optional[Callable[[], object]] = None,
) -> object:
    if get_learning_bbox_batch_fn is not None:
        return get_learning_bbox_batch_fn()
    if learning_session is not None:
        return learning_session.get_bbox_batch()
    return get_current_learning_bbox_batch()
