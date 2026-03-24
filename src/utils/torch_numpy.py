from __future__ import annotations

from typing import Any, Optional

import numpy as np


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("PyTorch is required for NumPy-to-torch tensor conversion") from exc
    return torch


def ensure_native_endian_array(array: np.ndarray) -> np.ndarray:
    """Return a NumPy array that uses the platform-native byte order."""
    normalized = np.asarray(array)
    if normalized.dtype.byteorder == "|" or normalized.dtype.isnative:
        return normalized
    swapped = normalized.byteswap(inplace=False)
    return swapped.view(swapped.dtype.newbyteorder("="))


def torch_from_numpy_safe(array: np.ndarray, *, torch_module: Optional[Any] = None):
    """Create a torch tensor from NumPy after normalizing non-native byte order."""
    normalized = ensure_native_endian_array(array)
    torch_mod = _require_torch() if torch_module is None else torch_module
    from_numpy = getattr(torch_mod, "from_numpy", None)
    if not callable(from_numpy):
        raise TypeError("torch_module must define from_numpy(array)")
    return from_numpy(normalized)
