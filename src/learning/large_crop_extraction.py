from __future__ import annotations

from typing import Tuple

import numpy as np

from .large_crop_inference_plan import LargeCropWindow, ShapeZYX


def extract_large_crop_from_array(
    volume_array: np.ndarray,
    *,
    window: LargeCropWindow,
) -> np.ndarray:
    array = _coerce_volume_array(volume_array)
    clipped = np.asarray(array[window.extraction.raw_slices])
    return _pad_extracted_crop(clipped, window=window)


def extract_large_crop_from_volume(
    volume: object,
    *,
    window: LargeCropWindow,
) -> np.ndarray:
    get_chunk = getattr(volume, "get_chunk", None)
    if not callable(get_chunk):
        raise TypeError("volume must define get_chunk(zyx_slices)")
    clipped = np.asarray(get_chunk(window.extraction.raw_slices))
    return _pad_extracted_crop(clipped, window=window)


def _coerce_volume_array(volume_array: np.ndarray) -> np.ndarray:
    array = np.asarray(volume_array)
    if array.ndim != 3:
        raise ValueError(
            f"volume_array must be a 3D array (z, y, x), got ndim={array.ndim}"
        )
    return array


def _pad_extracted_crop(
    clipped: np.ndarray,
    *,
    window: LargeCropWindow,
) -> np.ndarray:
    array = np.asarray(clipped)
    if array.ndim != 3:
        raise ValueError(f"extracted crop must be 3D (z, y, x), got ndim={array.ndim}")

    pad_width = _pad_width(window)
    if any(before > 0 or after > 0 for before, after in pad_width):
        _validate_reflect_padding_supported(array.shape, pad_width)
        array = np.pad(array, pad_width, mode="reflect")

    expected_shape = tuple(int(axis) for axis in window.crop_shape)
    if tuple(int(axis) for axis in array.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected large-crop extraction shape: "
            f"got={tuple(array.shape)} expected={expected_shape}"
        )
    return array


def _pad_width(window: LargeCropWindow) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    extraction = window.extraction
    return (
        (int(extraction.pad_before[0]), int(extraction.pad_after[0])),
        (int(extraction.pad_before[1]), int(extraction.pad_after[1])),
        (int(extraction.pad_before[2]), int(extraction.pad_after[2])),
    )


def _validate_reflect_padding_supported(
    shape: ShapeZYX,
    pad_width: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
) -> None:
    axis_names = ("z", "y", "x")
    for axis, (before, after) in enumerate(pad_width):
        if int(before) <= 0 and int(after) <= 0:
            continue
        if int(shape[axis]) <= 1:
            raise ValueError(
                "Cannot apply reflect padding on "
                f"{axis_names[axis]} axis with length <= 1."
            )

