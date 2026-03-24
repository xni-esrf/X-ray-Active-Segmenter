from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from numbers import Integral
from typing import Callable, Optional, Sequence, Tuple

from .train_bbox_dataset import TrainBBoxDataset
from .session_store import (
    LearningBBoxDataLoaderRuntime,
    LearningBBoxTensorBatch,
    LearningBBoxTensorEntry,
    get_current_learning_bbox_batch,
    set_current_learning_dataloader_components,
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


def _default_sampler_factory():
    try:
        from torch.utils.data import WeightedRandomSampler
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "PyTorch is required to build the learning WeightedRandomSampler"
        ) from exc
    return WeightedRandomSampler


def _default_dataloader_factory():
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("PyTorch is required to build the learning DataLoader") from exc
    return DataLoader


def _train_entries(batch: LearningBBoxTensorBatch) -> Tuple[LearningBBoxTensorEntry, ...]:
    if not isinstance(batch, LearningBBoxTensorBatch):
        raise TypeError(
            "batch must be a LearningBBoxTensorBatch, "
            f"got {type(batch).__name__}"
        )
    return tuple(entry for entry in batch.entries if str(entry.label) == "train")


def extract_train_tensor_pairs(
    batch: LearningBBoxTensorBatch,
) -> Tuple[Tuple[object, object], ...]:
    train_entries = _train_entries(batch)
    return tuple(
        (entry.raw_tensor, entry.segmentation_tensor)
        for entry in train_entries
    )


def build_learning_dataloader_from_batch(
    batch: LearningBBoxTensorBatch,
    *,
    minivol_size: int,
    minivol_per_epoch: int,
    batch_size: int = 4,
    num_workers: int = 8,
    pin_memory: bool = True,
    drop_last: bool = True,
    dataset_factory: Optional[Callable[..., object]] = None,
    sampler_factory: Optional[Callable[..., object]] = None,
    dataloader_factory: Optional[Callable[..., object]] = None,
    store_in_session: bool = True,
) -> LearningBBoxDataLoaderRuntime:
    normalized_minivol_size = _coerce_positive_int(minivol_size, name="minivol_size")
    normalized_minivol_per_epoch = _coerce_positive_int(
        minivol_per_epoch, name="minivol_per_epoch"
    )
    normalized_batch_size = _coerce_positive_int(batch_size, name="batch_size")
    normalized_num_workers = _coerce_non_negative_int(num_workers, name="num_workers")
    normalized_pin_memory = _coerce_bool(pin_memory, name="pin_memory")
    normalized_drop_last = _coerce_bool(drop_last, name="drop_last")
    normalized_store_in_session = _coerce_bool(store_in_session, name="store_in_session")

    train_entries = _train_entries(batch)
    if not train_entries:
        raise ValueError(
            "No training bounding boxes labeled 'train' were found in the current learning batch."
        )
    tensor_pairs = tuple(
        (entry.raw_tensor, entry.segmentation_tensor) for entry in train_entries
    )

    if dataset_factory is None:
        dataset_factory = TrainBBoxDataset
    dataset = dataset_factory(
        tensor_pairs,
        minivol_size=normalized_minivol_size,
        minivol_per_epoch=normalized_minivol_per_epoch,
    )

    dataset_weights = getattr(dataset, "weights", None)
    if not isinstance(dataset_weights, SequenceABC):
        raise TypeError(
            "Dataset must provide a sequence 'weights' attribute for WeightedRandomSampler"
        )

    dataset_len = len(dataset)
    if not isinstance(dataset_len, int) or dataset_len <= 0:
        raise ValueError(f"Dataset length must be >= 1, got {dataset_len}")

    if sampler_factory is None:
        sampler_factory = _default_sampler_factory()
    sampler = sampler_factory(dataset_weights, dataset_len)

    if dataloader_factory is None:
        dataloader_factory = _default_dataloader_factory()
    dataloader = dataloader_factory(
        dataset,
        batch_size=normalized_batch_size,
        sampler=sampler,
        num_workers=normalized_num_workers,
        pin_memory=normalized_pin_memory,
        drop_last=normalized_drop_last,
    )

    train_box_ids = tuple(entry.box_id for entry in train_entries)
    if normalized_store_in_session:
        return set_current_learning_dataloader_components(
            dataset=dataset,
            sampler=sampler,
            dataloader=dataloader,
            train_box_ids=train_box_ids,
            minivol_size=normalized_minivol_size,
            minivol_per_epoch=normalized_minivol_per_epoch,
            batch_size=normalized_batch_size,
            num_workers=normalized_num_workers,
            pin_memory=normalized_pin_memory,
            drop_last=normalized_drop_last,
        )

    return LearningBBoxDataLoaderRuntime(
        dataset=dataset,
        sampler=sampler,
        dataloader=dataloader,
        train_box_ids=train_box_ids,
        minivol_size=normalized_minivol_size,
        minivol_per_epoch=normalized_minivol_per_epoch,
        batch_size=normalized_batch_size,
        num_workers=normalized_num_workers,
        pin_memory=normalized_pin_memory,
        drop_last=normalized_drop_last,
    )


def build_learning_dataloader_from_current_batch(
    *,
    minivol_size: int,
    minivol_per_epoch: int,
    batch_size: int = 4,
    num_workers: int = 8,
    pin_memory: bool = True,
    drop_last: bool = True,
    dataset_factory: Optional[Callable[..., object]] = None,
    sampler_factory: Optional[Callable[..., object]] = None,
    dataloader_factory: Optional[Callable[..., object]] = None,
    store_in_session: bool = True,
) -> LearningBBoxDataLoaderRuntime:
    batch = get_current_learning_bbox_batch()
    if batch is None:
        raise ValueError("No learning tensor batch is available in session storage.")
    return build_learning_dataloader_from_batch(
        batch,
        minivol_size=minivol_size,
        minivol_per_epoch=minivol_per_epoch,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        dataset_factory=dataset_factory,
        sampler_factory=sampler_factory,
        dataloader_factory=dataloader_factory,
        store_in_session=store_in_session,
    )
