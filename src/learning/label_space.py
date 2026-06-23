from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Optional, Tuple

import numpy as np


def _coerce_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    return int(value)


def _coerce_source_signature(value: object) -> Optional[Tuple[object, ...]]:
    if value is None:
        return None
    if not isinstance(value, tuple):
        raise TypeError(
            "source_signature must be a tuple when provided, "
            f"got {type(value).__name__}"
        )
    return tuple(value)


def _normalize_label_values(
    values: object,
    *,
    background_label: int,
    mask_label: int,
) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        raise TypeError(
            "label_values must be an iterable of integers, "
            f"got {type(values).__name__}"
        )

    normalized = set()
    for raw_value in values:  # type: ignore[union-attr]
        value = _coerce_int(raw_value, name="label value")
        if value == int(mask_label):
            raise ValueError(
                f"label_values must not include mask label {int(mask_label)}"
            )
        normalized.add(int(value))

    normalized.add(int(background_label))
    ordered = tuple(sorted(normalized))
    foreground_labels = tuple(label for label in ordered if label != int(background_label))
    if not foreground_labels:
        raise ValueError(
            "label_values must include at least one foreground label in addition "
            f"to background label {int(background_label)}"
        )
    return ordered


@dataclass(frozen=True)
class LearningLabelSpace:
    label_values: Tuple[int, ...]
    background_label: int = 0
    mask_label: int = -100
    source_signature: Optional[Tuple[object, ...]] = None

    def __post_init__(self) -> None:
        background_label = _coerce_int(
            self.background_label,
            name="background_label",
        )
        mask_label = _coerce_int(self.mask_label, name="mask_label")
        if background_label == mask_label:
            raise ValueError("background_label must differ from mask_label")

        normalized_label_values = _normalize_label_values(
            self.label_values,
            background_label=background_label,
            mask_label=mask_label,
        )
        object.__setattr__(self, "background_label", int(background_label))
        object.__setattr__(self, "mask_label", int(mask_label))
        object.__setattr__(self, "label_values", normalized_label_values)
        object.__setattr__(
            self,
            "source_signature",
            _coerce_source_signature(self.source_signature),
        )

    @property
    def num_classes(self) -> int:
        return int(len(self.label_values))


def derive_label_space_from_semantic_segmentation(
    segmentation: object,
    *,
    background_label: int = 0,
    mask_label: int = -100,
    source_signature: Optional[Tuple[object, ...]] = None,
) -> LearningLabelSpace:
    array = np.asarray(segmentation)
    if array.ndim <= 0:
        raise ValueError(
            "semantic segmentation must be an array with at least one dimension"
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(
            f"semantic segmentation must have an integer dtype, got {array.dtype}"
        )
    label_values = tuple(
        int(value)
        for value in np.unique(array).tolist()
        if int(value) != int(mask_label)
    )
    return LearningLabelSpace(
        label_values=label_values,
        background_label=background_label,
        mask_label=mask_label,
        source_signature=source_signature,
    )
