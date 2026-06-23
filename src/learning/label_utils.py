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
