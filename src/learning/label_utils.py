from __future__ import annotations

from numbers import Integral
from typing import Optional, Sequence, Tuple

import numpy as np


MASK_LABEL = -100


def resolve_optional_torch(torch_module: Optional[object] = None) -> Optional[object]:
    if torch_module is not None:
        return torch_module
    try:  # pragma: no cover - import availability is environment dependent
        import torch
    except Exception:  # pragma: no cover - import availability is environment dependent
        return None
    return torch


def coerce_label_values(
    values: object,
    *,
    name: str = "label_values",
    mask_label: int = MASK_LABEL,
    allow_duplicates: bool = False,
) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of class ids, got {type(values).__name__}")

    normalized = []
    for raw_value in tuple(values):
        if isinstance(raw_value, bool) or not isinstance(raw_value, Integral):
            raise TypeError(
                f"{name} must contain integers only, got {type(raw_value).__name__}"
            )
        value = int(raw_value)
        if value == int(mask_label):
            raise ValueError(f"{name} must not include {int(mask_label)} (reserved mask label)")
        if value in normalized:
            if allow_duplicates:
                continue
            raise ValueError(f"{name} must not contain duplicates, got {value}")
        normalized.append(value)

    if not normalized:
        raise ValueError(f"{name} must contain at least one class label")
    return tuple(normalized)


def unique_non_mask_values(
    values: object,
    *,
    mask_label: int = MASK_LABEL,
    torch_module: Optional[object] = None,
) -> Tuple[int, ...]:
    torch_mod = resolve_optional_torch(torch_module)
    if torch_mod is not None and isinstance(values, getattr(torch_mod, "Tensor")):
        unique_values = getattr(torch_mod, "unique")(values.to(dtype=getattr(torch_mod, "long")))
        return tuple(
            sorted(
                int(value)
                for value in unique_values.tolist()
                if int(value) != int(mask_label)
            )
        )

    array = np.asarray(values)
    return tuple(
        sorted(
            int(value)
            for value in np.unique(array).tolist()
            if int(value) != int(mask_label)
        )
    )


def _format_unexpected_labels(values: Sequence[object]) -> str:
    ordered = tuple(sorted(int(value) for value in tuple(values)))
    suffix = "," if len(ordered) == 1 else ""
    return "(" + ", ".join(str(value) for value in ordered) + suffix + ")"


def encode_target_labels(
    target: object,
    *,
    label_values: Sequence[object],
    mask_label: int = MASK_LABEL,
    torch_module: Optional[object] = None,
):
    """Map semantic label values to compact class indices for loss functions.

    Segmentation tensors store user-facing semantic label values, which may be
    sparse, while CrossEntropyLoss expects class-index targets in [0, C - 1].
    """
    normalized_mask_label = int(mask_label)
    normalized_label_values = coerce_label_values(
        label_values,
        mask_label=normalized_mask_label,
    )

    torch_mod = resolve_optional_torch(torch_module)
    if torch_mod is not None and isinstance(target, getattr(torch_mod, "Tensor")):
        if getattr(target, "dtype", None) == getattr(torch_mod, "bool"):
            raise ValueError("target labels must have an integer dtype, got torch.bool")
        is_floating_point = getattr(torch_mod, "is_floating_point", None)
        if callable(is_floating_point) and bool(is_floating_point(target)):
            raise ValueError(
                f"target labels must have an integer dtype, got {target.dtype}"
            )

        target_long = target.to(dtype=getattr(torch_mod, "long"))
        encoded = getattr(torch_mod, "full_like")(
            target_long,
            fill_value=normalized_mask_label,
            dtype=getattr(torch_mod, "long"),
        )
        matched = target_long == normalized_mask_label
        for class_index, label in enumerate(normalized_label_values):
            label_mask = target_long == int(label)
            encoded[label_mask] = int(class_index)
            matched = matched | label_mask

        if not bool(getattr(torch_mod, "all")(matched).item()):
            unexpected = getattr(torch_mod, "unique")(target_long[~matched])
            unexpected_values = tuple(
                int(value) for value in unexpected.detach().cpu().tolist()
            )
            raise ValueError(
                "target labels contain values not present in label_values: "
                f"{_format_unexpected_labels(unexpected_values)}; "
                f"label_values={tuple(int(value) for value in normalized_label_values)}"
            )
        return encoded

    array = np.asarray(target)
    if array.dtype == np.dtype(bool) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"target labels must have an integer dtype, got {array.dtype}")

    target_array = array.astype(np.int64, copy=False)
    encoded_array = np.full(target_array.shape, normalized_mask_label, dtype=np.int64)
    matched_array = target_array == normalized_mask_label
    for class_index, label in enumerate(normalized_label_values):
        label_mask = target_array == int(label)
        encoded_array[label_mask] = int(class_index)
        matched_array = np.logical_or(matched_array, label_mask)

    if not np.all(matched_array):
        unexpected_values = tuple(
            int(value)
            for value in np.unique(
                target_array[np.logical_not(matched_array)]
            ).tolist()
        )
        raise ValueError(
            "target labels contain values not present in label_values: "
            f"{_format_unexpected_labels(unexpected_values)}; "
            f"label_values={tuple(int(value) for value in normalized_label_values)}"
        )
    return encoded_array
