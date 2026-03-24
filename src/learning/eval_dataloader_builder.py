from __future__ import annotations

from numbers import Integral
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

try:
    import torch
except Exception:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]

from .eval_bbox_dataset import DestVolBuffer, EvalBBoxDataset, InferenceDestVolBuffer
from .session_store import (
    LearningBBoxEvalRuntime,
    LearningBBoxTensorBatch,
    LearningBBoxTensorEntry,
    get_current_learning_bbox_batch,
    set_current_learning_eval_runtimes_by_box_id,
)


_MASK_LABEL = -100


def _best_effort_invoke(callable_obj) -> None:
    try:
        callable_obj()
    except Exception:
        return


def _best_effort_call_methods(resource: object, *, method_names: Tuple[str, ...]) -> None:
    if resource is None:
        return
    for method_name in method_names:
        method = getattr(resource, method_name, None)
        if callable(method):
            _best_effort_invoke(method)


def _best_effort_shutdown_dataloader_workers(dataloader: object) -> None:
    if dataloader is None:
        return

    iterator = getattr(dataloader, "_iterator", None)
    if iterator is not None:
        shutdown_workers = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown_workers):
            _best_effort_invoke(shutdown_workers)
        try:
            setattr(dataloader, "_iterator", None)
        except Exception:
            pass

    shutdown_workers = getattr(dataloader, "_shutdown_workers", None)
    if callable(shutdown_workers):
        _best_effort_invoke(shutdown_workers)

    _best_effort_call_methods(
        dataloader,
        method_names=("shutdown", "close", "stop", "terminate"),
    )


def _best_effort_dispose_eval_runtimes(
    runtimes_by_box_id: Dict[str, LearningBBoxEvalRuntime],
) -> None:
    for runtime in tuple(runtimes_by_box_id.values()):
        _best_effort_shutdown_dataloader_workers(runtime.dataloader)
        _best_effort_call_methods(
            runtime.buffer,
            method_names=("shutdown", "close", "stop", "terminate"),
        )


def _coerce_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be >= 1, got {normalized}")
    return normalized


def _coerce_non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be >= 0, got {normalized}")
    return normalized


def _coerce_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _coerce_label_values(label_values: Sequence[object]) -> Tuple[int, ...]:
    if not isinstance(label_values, Sequence):
        raise TypeError(
            "label_values must be a sequence of class IDs, "
            f"got {type(label_values).__name__}"
        )
    normalized = []
    for raw_value in tuple(label_values):
        if isinstance(raw_value, bool) or not isinstance(raw_value, Integral):
            raise TypeError(
                "label_values must contain integers only, "
                f"got {type(raw_value).__name__}"
            )
        value = int(raw_value)
        if value == _MASK_LABEL:
            raise ValueError("label_values must not include -100 (reserved mask label)")
        if value in normalized:
            raise ValueError(f"label_values must not contain duplicates, got {value}")
        normalized.append(value)
    if not normalized:
        raise ValueError("label_values must contain at least one class label")
    return tuple(normalized)


def _resolve_shared_num_classes_from_eval_runtimes(
    runtimes_by_box_id: Dict[str, LearningBBoxEvalRuntime],
) -> int:
    if not runtimes_by_box_id:
        raise ValueError("No evaluation runtimes/buffers were built.")

    resolved_num_classes: Optional[int] = None
    for box_id, runtime in tuple(runtimes_by_box_id.items()):
        buffer_obj = runtime.buffer
        if not hasattr(buffer_obj, "num_classes"):
            raise ValueError(
                f"Evaluation buffer for box_id={box_id!r} does not expose 'num_classes'."
            )
        raw_num_classes = getattr(buffer_obj, "num_classes")
        num_classes = _coerce_positive_int(raw_num_classes, name="num_classes")
        if resolved_num_classes is None:
            resolved_num_classes = num_classes
            continue
        if num_classes != resolved_num_classes:
            raise ValueError(
                "All evaluation buffers must share the same num_classes; "
                f"expected {resolved_num_classes}, got {num_classes} for box_id={box_id!r}."
            )

    if resolved_num_classes is None:
        raise ValueError("No evaluation buffer num_classes could be resolved.")
    return int(resolved_num_classes)


def _default_dataloader_factory():
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("PyTorch is required to build the evaluation DataLoader") from exc
    return DataLoader


def _record_dispose_error(errors: list[str], *, context: str, exc: Exception) -> None:
    errors.append(f"{context}: {type(exc).__name__}: {exc}")


def _call_methods_with_errors(
    resource: object,
    *,
    method_names: Tuple[str, ...],
    resource_name: str,
    errors: list[str],
) -> None:
    if resource is None:
        return
    for method_name in method_names:
        method = getattr(resource, method_name, None)
        if callable(method):
            try:
                method()
            except Exception as exc:
                _record_dispose_error(
                    errors,
                    context=f"{resource_name}.{method_name}()",
                    exc=exc,
                )


def _shutdown_dataloader_workers_with_errors(
    dataloader: object,
    *,
    errors: list[str],
) -> None:
    if dataloader is None:
        return

    iterator = getattr(dataloader, "_iterator", None)
    if iterator is not None:
        shutdown_workers = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown_workers):
            try:
                shutdown_workers()
            except Exception as exc:
                _record_dispose_error(
                    errors,
                    context="dataloader._iterator._shutdown_workers()",
                    exc=exc,
                )
        try:
            setattr(dataloader, "_iterator", None)
        except Exception as exc:
            _record_dispose_error(
                errors,
                context="setattr(dataloader, '_iterator', None)",
                exc=exc,
            )

    shutdown_workers = getattr(dataloader, "_shutdown_workers", None)
    if callable(shutdown_workers):
        try:
            shutdown_workers()
        except Exception as exc:
            _record_dispose_error(
                errors,
                context="dataloader._shutdown_workers()",
                exc=exc,
            )

    _call_methods_with_errors(
        dataloader,
        method_names=("shutdown", "close", "stop", "terminate"),
        resource_name="dataloader",
        errors=errors,
    )


def _batch_entries(batch: LearningBBoxTensorBatch) -> Tuple[LearningBBoxTensorEntry, ...]:
    if not isinstance(batch, LearningBBoxTensorBatch):
        raise TypeError(
            "batch must be a LearningBBoxTensorBatch, "
            f"got {type(batch).__name__}"
        )
    return tuple(batch.entries)


def _inference_entries(batch: LearningBBoxTensorBatch) -> Tuple[LearningBBoxTensorEntry, ...]:
    return tuple(entry for entry in _batch_entries(batch) if str(entry.label) == "inference")


def _resolve_dataset_volume_shape(dataset: object, *, fallback_shape: Sequence[object]) -> Tuple[int, int, int]:
    volume_obj = getattr(dataset, "vol", None)
    shape_obj = getattr(volume_obj, "shape", None)
    if shape_obj is None:
        shape_obj = fallback_shape
    shape_tuple = tuple(shape_obj)
    if len(shape_tuple) != 3:
        raise ValueError(
            "Inference dataset must expose a 3D volume shape via dataset.vol.shape; "
            f"got {shape_tuple}"
        )
    return tuple(int(v) for v in shape_tuple)


def _validation_entries(batch: LearningBBoxTensorBatch) -> Tuple[LearningBBoxTensorEntry, ...]:
    return tuple(entry for entry in _batch_entries(batch) if str(entry.label) == "validation")


def _train_and_validation_entries(
    batch: LearningBBoxTensorBatch,
) -> Tuple[LearningBBoxTensorEntry, ...]:
    return tuple(
        entry for entry in _batch_entries(batch) if str(entry.label) in {"train", "validation"}
    )


def compute_eval_label_values_from_batch(batch: LearningBBoxTensorBatch) -> Tuple[int, ...]:
    source_entries = _train_and_validation_entries(batch)
    if not source_entries:
        raise ValueError(
            "No train/validation segmentation tensors are available to compute eval label values."
        )
    if torch is None:  # pragma: no cover - environment dependent
        raise ImportError("PyTorch is required to compute eval label values")

    unique_values = set()
    for entry in source_entries:
        segmentation = entry.segmentation_tensor
        if not isinstance(segmentation, torch.Tensor):
            raise TypeError(
                "segmentation_tensor must be a torch.Tensor in learning batch entries"
            )
        values = torch.unique(segmentation).tolist()
        for value in values:
            integer = int(value)
            if integer == _MASK_LABEL:
                continue
            unique_values.add(integer)

    if not unique_values:
        raise ValueError(
            "No non-masked labels were found in train/validation segmentation tensors."
        )
    return tuple(sorted(unique_values))


def build_eval_dataloader_runtimes_from_batch(
    batch: LearningBBoxTensorBatch,
    *,
    minivol_size: int = 200,
    batch_size: int = 4,
    num_workers: int = 8,
    pin_memory: bool = True,
    drop_last: bool = False,
    dataset_factory: Optional[Callable[..., object]] = None,
    dataloader_factory: Optional[Callable[..., object]] = None,
    buffer_factory: Optional[Callable[..., object]] = None,
    store_in_session: bool = True,
) -> Dict[str, LearningBBoxEvalRuntime]:
    normalized_minivol_size = _coerce_positive_int(minivol_size, name="minivol_size")
    normalized_batch_size = _coerce_positive_int(batch_size, name="batch_size")
    normalized_num_workers = _coerce_non_negative_int(num_workers, name="num_workers")
    normalized_pin_memory = _coerce_bool(pin_memory, name="pin_memory")
    normalized_drop_last = _coerce_bool(drop_last, name="drop_last")
    normalized_store_in_session = _coerce_bool(store_in_session, name="store_in_session")

    validation_entries = _validation_entries(batch)
    if not validation_entries:
        raise ValueError(
            "No validation bounding boxes labeled 'validation' were found in the current learning batch."
        )

    label_values = compute_eval_label_values_from_batch(batch)

    if dataset_factory is None:
        dataset_factory = EvalBBoxDataset
    if dataloader_factory is None:
        dataloader_factory = _default_dataloader_factory()
    if buffer_factory is None:
        buffer_factory = DestVolBuffer

    runtimes: Dict[str, LearningBBoxEvalRuntime] = {}
    try:
        for entry in validation_entries:
            dataset = dataset_factory(
                entry.raw_tensor,
                minivol_size=normalized_minivol_size,
            )
            dataloader = dataloader_factory(
                dataset,
                batch_size=normalized_batch_size,
                num_workers=normalized_num_workers,
                pin_memory=normalized_pin_memory,
                drop_last=normalized_drop_last,
            )
            buffer = buffer_factory(
                entry.segmentation_tensor,
                dataset.vol.shape,
                label_values,
                minivol_size=normalized_minivol_size,
            )
            runtime = LearningBBoxEvalRuntime(
                box_id=entry.box_id,
                dataloader=dataloader,
                buffer=buffer,
            )
            runtimes[entry.box_id] = runtime

        _resolve_shared_num_classes_from_eval_runtimes(runtimes)

        if normalized_store_in_session:
            return set_current_learning_eval_runtimes_by_box_id(runtimes)
        return dict(runtimes)
    except Exception:
        _best_effort_dispose_eval_runtimes(runtimes)
        raise


def build_eval_dataloader_runtimes_from_current_batch(
    *,
    minivol_size: int = 200,
    batch_size: int = 4,
    num_workers: int = 8,
    pin_memory: bool = True,
    drop_last: bool = False,
    dataset_factory: Optional[Callable[..., object]] = None,
    dataloader_factory: Optional[Callable[..., object]] = None,
    buffer_factory: Optional[Callable[..., object]] = None,
    store_in_session: bool = True,
) -> Dict[str, LearningBBoxEvalRuntime]:
    batch = get_current_learning_bbox_batch()
    if batch is None:
        raise ValueError("No learning tensor batch is available in session storage.")
    return build_eval_dataloader_runtimes_from_batch(
        batch,
        minivol_size=minivol_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        dataset_factory=dataset_factory,
        dataloader_factory=dataloader_factory,
        buffer_factory=buffer_factory,
        store_in_session=store_in_session,
    )


def build_inference_dataloader_runtime_from_entry(
    entry: LearningBBoxTensorEntry,
    *,
    label_values: Sequence[object],
    minivol_size: int = 200,
    batch_size: int = 4,
    num_workers: int = 8,
    pin_memory: bool = True,
    drop_last: bool = False,
    dataset_factory: Optional[Callable[..., object]] = None,
    dataloader_factory: Optional[Callable[..., object]] = None,
    buffer_factory: Optional[Callable[..., object]] = None,
) -> LearningBBoxEvalRuntime:
    if not isinstance(entry, LearningBBoxTensorEntry):
        raise TypeError(
            "entry must be a LearningBBoxTensorEntry, "
            f"got {type(entry).__name__}"
        )
    normalized_label_values = _coerce_label_values(label_values)
    normalized_minivol_size = _coerce_positive_int(minivol_size, name="minivol_size")
    normalized_batch_size = _coerce_positive_int(batch_size, name="batch_size")
    normalized_num_workers = _coerce_non_negative_int(num_workers, name="num_workers")
    normalized_pin_memory = _coerce_bool(pin_memory, name="pin_memory")
    normalized_drop_last = _coerce_bool(drop_last, name="drop_last")

    if dataset_factory is None:
        dataset_factory = EvalBBoxDataset
    if dataloader_factory is None:
        dataloader_factory = _default_dataloader_factory()
    if buffer_factory is None:
        buffer_factory = InferenceDestVolBuffer

    dataset = dataset_factory(
        entry.raw_tensor,
        minivol_size=normalized_minivol_size,
    )
    dataloader = dataloader_factory(
        dataset,
        batch_size=normalized_batch_size,
        num_workers=normalized_num_workers,
        pin_memory=normalized_pin_memory,
        drop_last=normalized_drop_last,
    )
    volume_shape = _resolve_dataset_volume_shape(dataset, fallback_shape=entry.raw_tensor.shape)
    buffer = buffer_factory(
        volume_shape,
        normalized_label_values,
        minivol_size=normalized_minivol_size,
    )
    return LearningBBoxEvalRuntime(
        box_id=entry.box_id,
        dataloader=dataloader,
        buffer=buffer,
    )


def build_inference_dataloader_runtimes_from_batch(
    batch: LearningBBoxTensorBatch,
    *,
    label_values: Sequence[object],
    minivol_size: int = 200,
    batch_size: int = 4,
    num_workers: int = 8,
    pin_memory: bool = True,
    drop_last: bool = False,
    dataset_factory: Optional[Callable[..., object]] = None,
    dataloader_factory: Optional[Callable[..., object]] = None,
    buffer_factory: Optional[Callable[..., object]] = None,
) -> Dict[str, LearningBBoxEvalRuntime]:
    inference_entries = _inference_entries(batch)
    if not inference_entries:
        raise ValueError(
            "No inference bounding boxes labeled 'inference' were found in the current learning batch."
        )

    runtimes: Dict[str, LearningBBoxEvalRuntime] = {}
    try:
        for entry in inference_entries:
            runtime = build_inference_dataloader_runtime_from_entry(
                entry,
                label_values=label_values,
                minivol_size=minivol_size,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
                dataset_factory=dataset_factory,
                dataloader_factory=dataloader_factory,
                buffer_factory=buffer_factory,
            )
            runtimes[entry.box_id] = runtime
        return dict(runtimes)
    except Exception:
        dispose_inference_runtimes(runtimes)
        raise


def dispose_inference_runtime(runtime: LearningBBoxEvalRuntime) -> Tuple[str, ...]:
    if not isinstance(runtime, LearningBBoxEvalRuntime):
        raise TypeError(
            "runtime must be a LearningBBoxEvalRuntime, "
            f"got {type(runtime).__name__}"
        )
    errors: list[str] = []
    _shutdown_dataloader_workers_with_errors(runtime.dataloader, errors=errors)
    _call_methods_with_errors(
        runtime.buffer,
        method_names=("shutdown", "close", "stop", "terminate"),
        resource_name="buffer",
        errors=errors,
    )
    return tuple(errors)


def dispose_inference_runtimes(
    runtimes_by_box_id: Mapping[str, LearningBBoxEvalRuntime],
) -> Dict[str, Tuple[str, ...]]:
    if not isinstance(runtimes_by_box_id, Mapping):
        raise TypeError(
            "runtimes_by_box_id must be a mapping of box_id -> LearningBBoxEvalRuntime, "
            f"got {type(runtimes_by_box_id).__name__}"
        )
    cleanup_errors: Dict[str, Tuple[str, ...]] = {}
    for raw_box_id, runtime in tuple(runtimes_by_box_id.items()):
        box_id = str(raw_box_id)
        if not isinstance(runtime, LearningBBoxEvalRuntime):
            raise TypeError(
                "runtimes_by_box_id values must be LearningBBoxEvalRuntime, "
                f"got {type(runtime).__name__} for box_id={box_id!r}"
            )
        errors = dispose_inference_runtime(runtime)
        if errors:
            cleanup_errors[box_id] = tuple(errors)
    return cleanup_errors
