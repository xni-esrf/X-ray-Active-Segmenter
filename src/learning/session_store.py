from __future__ import annotations

from dataclasses import dataclass, field, replace
from numbers import Integral
from threading import Lock
from typing import Dict, Mapping, Optional, Sequence, Tuple

from ..bbox import BoundingBoxLabel

_ALLOWED_LABELS = ("train", "validation", "inference")
_CURRENT_BATCH: Optional["LearningBBoxTensorBatch"] = None
_CURRENT_DATALOADER_RUNTIME: Optional["LearningBBoxDataLoaderRuntime"] = None
_CURRENT_EVAL_RUNTIMES_BY_BOX_ID: Dict[str, "LearningBBoxEvalRuntime"] = {}
_CURRENT_MODEL_RUNTIME: Optional["LearningModelRuntime"] = None
_STORE_LOCK = Lock()


def _coerce_box_id(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"box_id must be a string, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError("box_id must be a non-empty string")
    return normalized


def _coerce_index(value: object) -> int:
    return _coerce_positive_int(value, name="index")


def _coerce_label(value: object) -> BoundingBoxLabel:
    if not isinstance(value, str):
        raise TypeError(f"label must be a string, got {type(value).__name__}")
    normalized = value.strip().lower()
    if normalized not in _ALLOWED_LABELS:
        raise ValueError(f"label must be one of {_ALLOWED_LABELS}, got {normalized!r}")
    return normalized  # type: ignore[return-value]


def _coerce_torch_tensor(value: object, *, name: str):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required for learning tensor session storage") from exc
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be stored on CPU, got device={value.device}")
    return value


def _coerce_present(value: object, *, name: str) -> object:
    if value is None:
        raise ValueError(f"{name} must not be None")
    return value


def _coerce_optional_positive_int(value: object, *, name: str) -> Optional[int]:
    if value is None:
        return None
    return _coerce_positive_int(value, name=name)


def _coerce_optional_non_negative_int(value: object, *, name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer when provided, got {type(value).__name__}")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be >= 0, got {normalized}")
    return normalized


def _coerce_optional_bool(value: object, *, name: str) -> Optional[bool]:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool when provided, got {type(value).__name__}")
    return value


def _coerce_optional_class_weights(value: object, *, name: str):
    if value is None:
        return None
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required for learning class-weight session storage") from exc
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor when provided, got {type(value).__name__}")
    if int(value.ndim) != 1:
        raise ValueError(f"{name} must be a 1D tensor, got ndim={int(value.ndim)}")
    if int(value.numel()) <= 0:
        raise ValueError(f"{name} must contain at least one value")
    if value.dtype != torch.float32:
        raise ValueError(f"{name} must have dtype torch.float32, got {value.dtype}")
    return value


def _coerce_non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be >= 0, got {normalized}")
    return normalized


def _coerce_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be >= 1, got {normalized}")
    return normalized


def _coerce_non_empty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty string")
    return normalized


def _coerce_device_ids(value: object) -> Tuple[int, ...]:
    if not isinstance(value, Sequence):
        raise TypeError(f"device_ids must be a sequence of integers, got {type(value).__name__}")

    normalized = tuple(
        _coerce_non_negative_int(raw_device_id, name="device_id")
        for raw_device_id in tuple(value)
    )
    if not normalized:
        raise ValueError("device_ids must contain at least one device id")
    if len(set(normalized)) != len(normalized):
        raise ValueError("device_ids must not contain duplicates")
    return normalized


def _coerce_hyperparameters(value: object) -> Dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(
            "hyperparameters must be a mapping of non-empty string keys to values, "
            f"got {type(value).__name__}"
        )
    normalized: Dict[str, object] = {}
    for raw_key, raw_value in tuple(value.items()):
        key = _coerce_non_empty_string(raw_key, name="hyperparameters key")
        normalized[key] = raw_value
    return normalized


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


def _best_effort_dispose_runtime(runtime: Optional["LearningBBoxDataLoaderRuntime"]) -> None:
    if runtime is None:
        return
    _best_effort_shutdown_dataloader_workers(runtime.dataloader)
    _best_effort_call_methods(
        runtime.sampler,
        method_names=("shutdown", "close", "stop", "terminate"),
    )
    _best_effort_call_methods(
        runtime.dataset,
        method_names=("shutdown", "close", "stop", "terminate"),
    )


def _best_effort_dispose_eval_runtime(runtime: Optional["LearningBBoxEvalRuntime"]) -> None:
    if runtime is None:
        return
    _best_effort_shutdown_dataloader_workers(runtime.dataloader)
    _best_effort_call_methods(
        runtime.buffer,
        method_names=("shutdown", "close", "stop", "terminate"),
    )


def _best_effort_dispose_eval_runtime_map(
    runtimes_by_box_id: Mapping[str, "LearningBBoxEvalRuntime"],
) -> None:
    for runtime in tuple(runtimes_by_box_id.values()):
        _best_effort_dispose_eval_runtime(runtime)


def _best_effort_dispose_model_runtime(runtime: Optional["LearningModelRuntime"]) -> None:
    if runtime is None:
        return
    _best_effort_call_methods(
        runtime.optimizer,
        method_names=("shutdown", "close", "stop", "terminate"),
    )
    _best_effort_call_methods(
        runtime.model,
        method_names=("shutdown", "close", "stop", "terminate"),
    )


@dataclass(frozen=True)
class LearningBBoxTensorEntry:
    box_id: str
    index: int
    label: BoundingBoxLabel
    raw_tensor: object
    segmentation_tensor: object

    def __post_init__(self) -> None:
        object.__setattr__(self, "box_id", _coerce_box_id(self.box_id))
        object.__setattr__(self, "index", _coerce_index(self.index))
        object.__setattr__(self, "label", _coerce_label(self.label))
        object.__setattr__(
            self,
            "raw_tensor",
            _coerce_torch_tensor(self.raw_tensor, name="raw_tensor"),
        )
        object.__setattr__(
            self,
            "segmentation_tensor",
            _coerce_torch_tensor(self.segmentation_tensor, name="segmentation_tensor"),
        )


@dataclass(frozen=True)
class LearningBBoxTensorBatch:
    entries: Tuple[LearningBBoxTensorEntry, ...]

    def __post_init__(self) -> None:
        normalized_entries = tuple(self.entries)
        for entry in normalized_entries:
            if not isinstance(entry, LearningBBoxTensorEntry):
                raise TypeError(
                    "entries must contain only LearningBBoxTensorEntry instances, "
                    f"got {type(entry).__name__}"
                )
        object.__setattr__(self, "entries", normalized_entries)

    @property
    def size(self) -> int:
        return int(len(self.entries))

    @property
    def box_ids(self) -> Tuple[str, ...]:
        return tuple(entry.box_id for entry in self.entries)


@dataclass(frozen=True)
class LearningBBoxDataLoaderRuntime:
    dataset: object
    sampler: object
    dataloader: object
    train_box_ids: Tuple[str, ...]
    minivol_size: Optional[int] = None
    minivol_per_epoch: Optional[int] = None
    batch_size: Optional[int] = None
    num_workers: Optional[int] = None
    pin_memory: Optional[bool] = None
    drop_last: Optional[bool] = None
    class_weights: Optional[object] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset", _coerce_present(self.dataset, name="dataset"))
        object.__setattr__(self, "sampler", _coerce_present(self.sampler, name="sampler"))
        object.__setattr__(self, "dataloader", _coerce_present(self.dataloader, name="dataloader"))

        normalized_box_ids = tuple(_coerce_box_id(box_id) for box_id in tuple(self.train_box_ids))
        if len(set(normalized_box_ids)) != len(normalized_box_ids):
            raise ValueError("train_box_ids must not contain duplicates")
        object.__setattr__(self, "train_box_ids", normalized_box_ids)

        object.__setattr__(
            self,
            "minivol_size",
            _coerce_optional_positive_int(self.minivol_size, name="minivol_size"),
        )
        object.__setattr__(
            self,
            "minivol_per_epoch",
            _coerce_optional_positive_int(self.minivol_per_epoch, name="minivol_per_epoch"),
        )
        object.__setattr__(
            self,
            "batch_size",
            _coerce_optional_positive_int(self.batch_size, name="batch_size"),
        )
        object.__setattr__(
            self,
            "num_workers",
            _coerce_optional_non_negative_int(self.num_workers, name="num_workers"),
        )
        object.__setattr__(
            self,
            "pin_memory",
            _coerce_optional_bool(self.pin_memory, name="pin_memory"),
        )
        object.__setattr__(
            self,
            "drop_last",
            _coerce_optional_bool(self.drop_last, name="drop_last"),
        )
        object.__setattr__(
            self,
            "class_weights",
            _coerce_optional_class_weights(self.class_weights, name="class_weights"),
        )

    @property
    def train_count(self) -> int:
        return int(len(self.train_box_ids))


@dataclass(frozen=True)
class LearningBBoxEvalRuntime:
    box_id: str
    dataloader: object
    buffer: object

    def __post_init__(self) -> None:
        object.__setattr__(self, "box_id", _coerce_box_id(self.box_id))
        object.__setattr__(
            self,
            "dataloader",
            _coerce_present(self.dataloader, name="dataloader"),
        )
        object.__setattr__(
            self,
            "buffer",
            _coerce_present(self.buffer, name="buffer"),
        )


@dataclass(frozen=True)
class LearningModelRuntime:
    model: object
    optimizer: object
    checkpoint_path: str
    device_ids: Tuple[int, ...]
    num_classes: int
    hyperparameters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", _coerce_present(self.model, name="model"))
        object.__setattr__(self, "optimizer", _coerce_present(self.optimizer, name="optimizer"))
        object.__setattr__(
            self,
            "checkpoint_path",
            _coerce_non_empty_string(self.checkpoint_path, name="checkpoint_path"),
        )
        object.__setattr__(self, "device_ids", _coerce_device_ids(self.device_ids))
        object.__setattr__(
            self,
            "num_classes",
            _coerce_positive_int(self.num_classes, name="num_classes"),
        )
        object.__setattr__(self, "hyperparameters", _coerce_hyperparameters(self.hyperparameters))


def set_current_learning_bbox_batch(batch: LearningBBoxTensorBatch) -> LearningBBoxTensorBatch:
    if not isinstance(batch, LearningBBoxTensorBatch):
        raise TypeError(
            "batch must be a LearningBBoxTensorBatch, "
            f"got {type(batch).__name__}"
        )
    global _CURRENT_BATCH
    with _STORE_LOCK:
        _CURRENT_BATCH = batch
    return batch


def set_current_learning_bbox_entries(
    entries: Sequence[LearningBBoxTensorEntry],
) -> LearningBBoxTensorBatch:
    batch = LearningBBoxTensorBatch(entries=tuple(entries))
    return set_current_learning_bbox_batch(batch)


def get_current_learning_bbox_batch() -> Optional[LearningBBoxTensorBatch]:
    with _STORE_LOCK:
        return _CURRENT_BATCH


def clear_current_learning_bbox_batch() -> None:
    global _CURRENT_BATCH
    with _STORE_LOCK:
        _CURRENT_BATCH = None


def set_current_learning_dataloader_runtime(
    runtime: LearningBBoxDataLoaderRuntime,
) -> LearningBBoxDataLoaderRuntime:
    if not isinstance(runtime, LearningBBoxDataLoaderRuntime):
        raise TypeError(
            "runtime must be a LearningBBoxDataLoaderRuntime, "
            f"got {type(runtime).__name__}"
        )
    global _CURRENT_DATALOADER_RUNTIME
    previous_runtime: Optional[LearningBBoxDataLoaderRuntime] = None
    with _STORE_LOCK:
        previous_runtime = _CURRENT_DATALOADER_RUNTIME
        _CURRENT_DATALOADER_RUNTIME = runtime
    if previous_runtime is not runtime:
        _best_effort_dispose_runtime(previous_runtime)
    return runtime


def set_current_learning_dataloader_components(
    *,
    dataset: object,
    sampler: object,
    dataloader: object,
    train_box_ids: Sequence[str],
    minivol_size: Optional[int] = None,
    minivol_per_epoch: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    drop_last: Optional[bool] = None,
    class_weights: Optional[object] = None,
) -> LearningBBoxDataLoaderRuntime:
    runtime = LearningBBoxDataLoaderRuntime(
        dataset=dataset,
        sampler=sampler,
        dataloader=dataloader,
        train_box_ids=tuple(train_box_ids),
        minivol_size=minivol_size,
        minivol_per_epoch=minivol_per_epoch,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        class_weights=class_weights,
    )
    return set_current_learning_dataloader_runtime(runtime)


def get_current_learning_dataloader_runtime() -> Optional[LearningBBoxDataLoaderRuntime]:
    with _STORE_LOCK:
        return _CURRENT_DATALOADER_RUNTIME


def set_current_learning_dataloader_class_weights(
    class_weights: object,
) -> LearningBBoxDataLoaderRuntime:
    global _CURRENT_DATALOADER_RUNTIME
    with _STORE_LOCK:
        current_runtime = _CURRENT_DATALOADER_RUNTIME
        if current_runtime is None:
            raise ValueError("No learning dataloader runtime is available in session storage.")
        updated_runtime = replace(
            current_runtime,
            class_weights=class_weights,
        )
        _CURRENT_DATALOADER_RUNTIME = updated_runtime
    return updated_runtime


def clear_current_learning_dataloader_runtime() -> None:
    global _CURRENT_DATALOADER_RUNTIME
    previous_runtime: Optional[LearningBBoxDataLoaderRuntime] = None
    with _STORE_LOCK:
        previous_runtime = _CURRENT_DATALOADER_RUNTIME
        _CURRENT_DATALOADER_RUNTIME = None
    _best_effort_dispose_runtime(previous_runtime)


def _normalize_eval_runtimes_by_box_id(
    runtimes_by_box_id: Mapping[str, object],
) -> Dict[str, LearningBBoxEvalRuntime]:
    if not isinstance(runtimes_by_box_id, Mapping):
        raise TypeError(
            "runtimes_by_box_id must be a mapping of box_id -> LearningBBoxEvalRuntime, "
            f"got {type(runtimes_by_box_id).__name__}"
        )

    normalized: Dict[str, LearningBBoxEvalRuntime] = {}
    for raw_box_id, raw_runtime in tuple(runtimes_by_box_id.items()):
        normalized_box_id = _coerce_box_id(raw_box_id)
        if isinstance(raw_runtime, LearningBBoxEvalRuntime):
            runtime = raw_runtime
        else:
            raise TypeError(
                "runtimes_by_box_id values must be LearningBBoxEvalRuntime, "
                f"got {type(raw_runtime).__name__} for id={normalized_box_id}"
            )
        if runtime.box_id != normalized_box_id:
            raise ValueError(
                "Mapping key and runtime.box_id must match: "
                f"key={normalized_box_id!r}, runtime.box_id={runtime.box_id!r}"
            )
        normalized[normalized_box_id] = runtime
    return normalized


def set_current_learning_eval_runtimes_by_box_id(
    runtimes_by_box_id: Mapping[str, LearningBBoxEvalRuntime],
) -> Dict[str, LearningBBoxEvalRuntime]:
    normalized = _normalize_eval_runtimes_by_box_id(runtimes_by_box_id)
    global _CURRENT_EVAL_RUNTIMES_BY_BOX_ID
    previous: Dict[str, LearningBBoxEvalRuntime] = {}
    with _STORE_LOCK:
        previous = _CURRENT_EVAL_RUNTIMES_BY_BOX_ID
        _CURRENT_EVAL_RUNTIMES_BY_BOX_ID = dict(normalized)
    _best_effort_dispose_eval_runtime_map(previous)
    return dict(normalized)


def set_current_learning_eval_runtime_components_by_box_id(
    components_by_box_id: Mapping[str, Tuple[object, object]],
) -> Dict[str, LearningBBoxEvalRuntime]:
    if not isinstance(components_by_box_id, Mapping):
        raise TypeError(
            "components_by_box_id must be a mapping of box_id -> (dataloader, buffer), "
            f"got {type(components_by_box_id).__name__}"
        )

    runtimes: Dict[str, LearningBBoxEvalRuntime] = {}
    for raw_box_id, component_pair in tuple(components_by_box_id.items()):
        normalized_box_id = _coerce_box_id(raw_box_id)
        if not isinstance(component_pair, tuple) or len(component_pair) != 2:
            raise TypeError(
                "components_by_box_id values must be tuple(dataloader, buffer), "
                f"got {type(component_pair).__name__} for id={normalized_box_id}"
            )
        dataloader, buffer = component_pair
        runtimes[normalized_box_id] = LearningBBoxEvalRuntime(
            box_id=normalized_box_id,
            dataloader=dataloader,
            buffer=buffer,
        )
    return set_current_learning_eval_runtimes_by_box_id(runtimes)


def get_current_learning_eval_runtimes_by_box_id() -> Dict[str, LearningBBoxEvalRuntime]:
    with _STORE_LOCK:
        return dict(_CURRENT_EVAL_RUNTIMES_BY_BOX_ID)


def clear_current_learning_eval_runtimes_by_box_id() -> None:
    global _CURRENT_EVAL_RUNTIMES_BY_BOX_ID
    previous: Dict[str, LearningBBoxEvalRuntime] = {}
    with _STORE_LOCK:
        previous = _CURRENT_EVAL_RUNTIMES_BY_BOX_ID
        _CURRENT_EVAL_RUNTIMES_BY_BOX_ID = {}
    _best_effort_dispose_eval_runtime_map(previous)


def set_current_learning_model_runtime(
    runtime: LearningModelRuntime,
) -> LearningModelRuntime:
    if not isinstance(runtime, LearningModelRuntime):
        raise TypeError(
            "runtime must be a LearningModelRuntime, "
            f"got {type(runtime).__name__}"
        )

    global _CURRENT_MODEL_RUNTIME
    previous_runtime: Optional[LearningModelRuntime] = None
    with _STORE_LOCK:
        previous_runtime = _CURRENT_MODEL_RUNTIME
        _CURRENT_MODEL_RUNTIME = runtime
    if previous_runtime is not runtime:
        _best_effort_dispose_model_runtime(previous_runtime)
    return runtime


def set_current_learning_model_components(
    *,
    model: object,
    optimizer: object,
    checkpoint_path: str,
    device_ids: Sequence[object],
    num_classes: int,
    hyperparameters: Optional[Mapping[str, object]] = None,
) -> LearningModelRuntime:
    runtime = LearningModelRuntime(
        model=model,
        optimizer=optimizer,
        checkpoint_path=checkpoint_path,
        device_ids=tuple(device_ids),
        num_classes=num_classes,
        hyperparameters=dict(hyperparameters or {}),
    )
    return set_current_learning_model_runtime(runtime)


def get_current_learning_model_runtime() -> Optional[LearningModelRuntime]:
    with _STORE_LOCK:
        return _CURRENT_MODEL_RUNTIME


def clear_current_learning_model_runtime() -> None:
    global _CURRENT_MODEL_RUNTIME
    previous_runtime: Optional[LearningModelRuntime] = None
    with _STORE_LOCK:
        previous_runtime = _CURRENT_MODEL_RUNTIME
        _CURRENT_MODEL_RUNTIME = None
    _best_effort_dispose_model_runtime(previous_runtime)
