from __future__ import annotations

from numbers import Integral


DEFAULT_INFERENCE_MINIVOL_SIZE = 200
DEFAULT_INFERENCE_STRIDE = DEFAULT_INFERENCE_MINIVOL_SIZE // 2
DEFAULT_LARGE_CROP_EDGE_VOXELS = 2200
DEFAULT_LARGE_CROP_VOXEL_BUDGET = DEFAULT_LARGE_CROP_EDGE_VOXELS**3

INFERENCE_INTERNAL_CROP_DISCARD_MARGIN = DEFAULT_INFERENCE_STRIDE
INFERENCE_CROP_EXTENT_OVERLAP = 2 * INFERENCE_INTERNAL_CROP_DISCARD_MARGIN


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


def inference_internal_crop_discard_margin_for_minivol_size(
    minivol_size: object,
) -> int:
    return inference_stride_for_minivol_size(minivol_size)


def inference_crop_extent_overlap_for_minivol_size(minivol_size: object) -> int:
    return 2 * inference_internal_crop_discard_margin_for_minivol_size(minivol_size)


def inference_valid_step_for_crop_size(
    crop_size: object,
    *,
    minivol_size: object = DEFAULT_INFERENCE_MINIVOL_SIZE,
) -> int:
    if isinstance(crop_size, bool) or not isinstance(crop_size, Integral):
        raise TypeError(
            "crop_size must be an integer, "
            f"got {type(crop_size).__name__}"
        )
    normalized_crop_size = int(crop_size)
    overlap = inference_crop_extent_overlap_for_minivol_size(minivol_size)
    valid_step = normalized_crop_size - int(overlap)
    if valid_step < 1:
        raise ValueError(
            "crop_size must be larger than the crop extent overlap; "
            f"got crop_size={normalized_crop_size}, overlap={overlap}"
        )
    return int(valid_step)

