from __future__ import annotations

from numbers import Integral


DEFAULT_INFERENCE_MINIVOL_SIZE = 200


def coerce_inference_minivol_size(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(
            "minivol_size must be an integer, "
            f"got {type(value).__name__}"
        )
    normalized = int(value)
    if normalized < 2:
        raise ValueError("minivol_size must be >= 2 for overlap extraction")
    return normalized


def inference_stride_for_minivol_size(minivol_size: object) -> int:
    normalized = coerce_inference_minivol_size(minivol_size)
    return int(normalized // 2)
