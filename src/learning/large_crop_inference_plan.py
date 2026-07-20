from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from numbers import Integral
from typing import List, Optional, Sequence, Tuple

from .inference_geometry import (
    DEFAULT_INFERENCE_MINIVOL_SIZE,
    DEFAULT_INFERENCE_STRIDE,
    DEFAULT_LARGE_CROP_EDGE_VOXELS,
    DEFAULT_LARGE_CROP_VOXEL_BUDGET,
    inference_crop_extent_overlap_for_minivol_size,
    inference_internal_crop_discard_margin_for_minivol_size,
    inference_stride_for_minivol_size,
)
from .zero_occupancy import ZeroOccupancyGrid, region_is_definitely_empty


AxisBounds = Tuple[int, int]
BoundsZYX = Tuple[AxisBounds, AxisBounds, AxisBounds]
ShapeZYX = Tuple[int, int, int]
SliceZYX = Tuple[slice, slice, slice]


@dataclass(frozen=True)
class CropExtractionPlan:
    raw_slices: SliceZYX
    pad_before: ShapeZYX
    pad_after: ShapeZYX


@dataclass(frozen=True)
class LargeCropWindow:
    index: int
    grid_index: ShapeZYX
    crop_slices: SliceZYX
    valid_slices: SliceZYX
    valid_slices_in_crop: SliceZYX
    requested_output_slices: SliceZYX
    requested_output_slices_in_crop: SliceZYX
    requested_output_slices_in_raw: SliceZYX
    extraction: CropExtractionPlan

    @property
    def crop_shape(self) -> ShapeZYX:
        return _shape_from_slices(self.crop_slices)

    @property
    def valid_shape(self) -> ShapeZYX:
        return _shape_from_slices(self.valid_slices)

    @property
    def writes_requested_output(self) -> bool:
        return all(_slice_size(axis_slice) > 0 for axis_slice in self.requested_output_slices)


@dataclass(frozen=True)
class LargeCropInferencePlan:
    requested_bounds: BoundsZYX
    raw_volume_shape: ShapeZYX
    requested_shape: ShapeZYX
    normalized_origin_in_raw: ShapeZYX
    normalized_shape: ShapeZYX
    requested_slices_in_normalized: SliceZYX
    context_margin: int
    minivol_size: int
    stride: int
    internal_discard_margin: int
    crop_extent_overlap: int
    crop_grid_shape: ShapeZYX
    valid_step_shape: ShapeZYX
    windows: Tuple[LargeCropWindow, ...]
    predicted_skip_crop_count: Optional[int] = None
    predicted_skip_voxel_count: Optional[int] = None

    @property
    def requires_cropping(self) -> bool:
        return len(self.windows) > 1

    @property
    def total_crop_count(self) -> int:
        return len(self.windows)


@dataclass(frozen=True)
class _AxisCandidate:
    axis: int
    minimum_size: int
    split_count: int
    normalized_size: int
    valid_step: int
    max_crop_size: int


def build_large_crop_inference_plan(
    *,
    requested_bounds: BoundsZYX,
    raw_volume_shape: Sequence[object],
    context_margin: int = DEFAULT_INFERENCE_STRIDE,
    minivol_size: int = DEFAULT_INFERENCE_MINIVOL_SIZE,
    voxel_budget: int = DEFAULT_LARGE_CROP_VOXEL_BUDGET,
    preferred_max_crop_edge: int = DEFAULT_LARGE_CROP_EDGE_VOXELS,
    occupancy: Optional[ZeroOccupancyGrid] = None,
) -> LargeCropInferencePlan:
    normalized_raw_shape = _coerce_shape(raw_volume_shape, name="raw_volume_shape")
    normalized_requested_bounds = _coerce_bounds(
        requested_bounds,
        raw_volume_shape=normalized_raw_shape,
    )
    normalized_context_margin = _coerce_non_negative_int(
        context_margin,
        name="context_margin",
    )
    normalized_minivol_size = _coerce_positive_int(
        minivol_size,
        name="minivol_size",
    )
    normalized_voxel_budget = _coerce_positive_int(
        voxel_budget,
        name="voxel_budget",
    )
    normalized_preferred_edge = _coerce_positive_int(
        preferred_max_crop_edge,
        name="preferred_max_crop_edge",
    )

    stride = inference_stride_for_minivol_size(normalized_minivol_size)
    internal_margin = inference_internal_crop_discard_margin_for_minivol_size(
        normalized_minivol_size
    )
    crop_overlap = inference_crop_extent_overlap_for_minivol_size(
        normalized_minivol_size
    )
    requested_shape = _shape_from_bounds(normalized_requested_bounds)
    minimum_shape = tuple(
        _ceil_to_multiple(
            int(axis_size) + 2 * int(normalized_context_margin),
            int(stride),
        )
        for axis_size in requested_shape
    )

    axis_candidates = tuple(
        _axis_candidates(
            axis=axis,
            minimum_size=int(minimum_shape[axis]),
            stride=int(stride),
            internal_margin=int(internal_margin),
            preferred_max_crop_edge=int(normalized_preferred_edge),
        )
        for axis in range(3)
    )
    best_candidates, feasible_candidates = _select_crop_grid(
        axis_candidates=axis_candidates,
        voxel_budget=int(normalized_voxel_budget),
    )

    normalized_origin_in_raw = tuple(
        int(normalized_requested_bounds[axis][0]) - int(normalized_context_margin)
        for axis in range(3)
    )
    requested_slices_in_normalized = tuple(
        slice(
            int(normalized_context_margin),
            int(normalized_context_margin) + int(requested_shape[axis]),
        )
        for axis in range(3)
    )
    requested_bounds_origin = tuple(
        int(normalized_requested_bounds[axis][0]) for axis in range(3)
    )

    selected_candidates = best_candidates
    predicted_skip_crop_count: Optional[int] = None
    predicted_skip_voxel_count: Optional[int] = None
    if occupancy is not None:
        (
            selected_candidates,
            predicted_skip_crop_count,
            predicted_skip_voxel_count,
        ) = _select_zero_skip_aware_crop_grid(
            feasible_candidates,
            occupancy=occupancy,
            requested_shape=requested_shape,
            raw_volume_shape=normalized_raw_shape,
            normalized_origin_in_raw=normalized_origin_in_raw,
            requested_slices_in_normalized=requested_slices_in_normalized,
            internal_margin=int(internal_margin),
            requested_bounds_origin=requested_bounds_origin,
        )

    normalized_shape = tuple(
        int(candidate.normalized_size) for candidate in selected_candidates
    )
    crop_grid_shape = tuple(
        int(candidate.split_count) for candidate in selected_candidates
    )
    valid_step_shape = tuple(
        int(candidate.valid_step) for candidate in selected_candidates
    )

    windows = _build_crop_windows(
        requested_shape=requested_shape,
        raw_volume_shape=normalized_raw_shape,
        normalized_origin_in_raw=normalized_origin_in_raw,
        normalized_shape=normalized_shape,
        requested_slices_in_normalized=requested_slices_in_normalized,
        crop_grid_shape=crop_grid_shape,
        valid_step_shape=valid_step_shape,
        internal_margin=int(internal_margin),
        requested_bounds_origin=requested_bounds_origin,
    )

    return LargeCropInferencePlan(
        requested_bounds=normalized_requested_bounds,
        raw_volume_shape=normalized_raw_shape,
        requested_shape=requested_shape,
        normalized_origin_in_raw=normalized_origin_in_raw,
        normalized_shape=normalized_shape,
        requested_slices_in_normalized=requested_slices_in_normalized,
        context_margin=int(normalized_context_margin),
        minivol_size=int(normalized_minivol_size),
        stride=int(stride),
        internal_discard_margin=int(internal_margin),
        crop_extent_overlap=int(crop_overlap),
        crop_grid_shape=crop_grid_shape,
        valid_step_shape=valid_step_shape,
        windows=windows,
        predicted_skip_crop_count=predicted_skip_crop_count,
        predicted_skip_voxel_count=predicted_skip_voxel_count,
    )


def _axis_candidates(
    *,
    axis: int,
    minimum_size: int,
    stride: int,
    internal_margin: int,
    preferred_max_crop_edge: int,
) -> Tuple[_AxisCandidate, ...]:
    max_splits = max(1, _ceil_div(int(minimum_size), int(stride)))
    candidates: list[_AxisCandidate] = []
    for split_count in range(1, max_splits + 1):
        valid_step = _ceil_to_multiple(
            _ceil_div(int(minimum_size), int(split_count)),
            int(stride),
        )
        normalized_size = int(valid_step) * int(split_count)
        max_crop_size = (
            int(normalized_size)
            if int(split_count) == 1
            else int(valid_step) + 2 * int(internal_margin)
        )
        candidates.append(
            _AxisCandidate(
                axis=int(axis),
                minimum_size=int(minimum_size),
                split_count=int(split_count),
                normalized_size=int(normalized_size),
                valid_step=int(valid_step),
                max_crop_size=int(max_crop_size),
            )
        )
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                int(candidate.split_count),
                max(0, int(candidate.max_crop_size) - int(preferred_max_crop_edge)),
                int(candidate.normalized_size) - int(candidate.minimum_size),
            ),
        )
    )


_AxisCandidateCombo = Tuple[_AxisCandidate, _AxisCandidate, _AxisCandidate]


def _candidate_grid_secondary_score(
    candidates: _AxisCandidateCombo,
) -> Tuple[int, int, int]:
    split_small_axis_penalty = sum(
        1
        for candidate in candidates
        if int(candidate.minimum_size) <= DEFAULT_LARGE_CROP_EDGE_VOXELS
        and int(candidate.split_count) > 1
    )
    extra_padding = sum(
        int(candidate.normalized_size) - int(candidate.minimum_size)
        for candidate in candidates
    )
    preferred_edge_excess = sum(
        max(0, int(candidate.max_crop_size) - DEFAULT_LARGE_CROP_EDGE_VOXELS)
        for candidate in candidates
    )
    return split_small_axis_penalty, extra_padding, preferred_edge_excess


def _select_crop_grid(
    *,
    axis_candidates: Tuple[
        Tuple[_AxisCandidate, ...],
        Tuple[_AxisCandidate, ...],
        Tuple[_AxisCandidate, ...],
    ],
    voxel_budget: int,
) -> Tuple[_AxisCandidateCombo, Tuple[Tuple[int, _AxisCandidateCombo], ...]]:
    best: _AxisCandidateCombo | None = None
    best_score: Tuple[int, int, int, int, int] | None = None
    feasible: List[Tuple[int, _AxisCandidateCombo]] = []
    for candidates in product(*axis_candidates):
        crop_voxels = _product(
            int(candidate.max_crop_size) for candidate in candidates
        )
        if crop_voxels > int(voxel_budget):
            continue
        split_count_product = _product(
            int(candidate.split_count) for candidate in candidates
        )
        feasible.append((int(split_count_product), candidates))
        split_small_axis_penalty, extra_padding, preferred_edge_excess = (
            _candidate_grid_secondary_score(candidates)
        )
        score = (
            int(split_count_product),
            int(split_small_axis_penalty),
            int(extra_padding),
            int(preferred_edge_excess),
            -int(crop_voxels),
        )
        if best_score is None or score < best_score:
            best = candidates
            best_score = score
    if best is None:
        raise ValueError(
            "Could not find a crop grid within the configured voxel budget."
        )
    return best, tuple(feasible)


def _select_zero_skip_aware_crop_grid(
    feasible: Tuple[Tuple[int, _AxisCandidateCombo], ...],
    *,
    occupancy: ZeroOccupancyGrid,
    requested_shape: ShapeZYX,
    raw_volume_shape: ShapeZYX,
    normalized_origin_in_raw: ShapeZYX,
    requested_slices_in_normalized: SliceZYX,
    internal_margin: int,
    requested_bounds_origin: ShapeZYX,
) -> Tuple[_AxisCandidateCombo, int, int]:
    best_count = min(split_count_product for split_count_product, _combo in feasible)
    shortlist = [
        (split_count_product, combo)
        for split_count_product, combo in feasible
        if split_count_product < 2 * best_count
    ]

    best_combo: _AxisCandidateCombo | None = None
    best_score: Tuple[int, int, int, int, int, int] | None = None
    best_skipped_crop_count = 0
    best_skipped_voxels = 0
    for split_count_product, combo in shortlist:
        normalized_shape = tuple(int(candidate.normalized_size) for candidate in combo)
        crop_grid_shape = tuple(int(candidate.split_count) for candidate in combo)
        valid_step_shape = tuple(int(candidate.valid_step) for candidate in combo)
        windows = _build_crop_windows(
            requested_shape=requested_shape,
            raw_volume_shape=raw_volume_shape,
            normalized_origin_in_raw=normalized_origin_in_raw,
            normalized_shape=normalized_shape,
            requested_slices_in_normalized=requested_slices_in_normalized,
            crop_grid_shape=crop_grid_shape,
            valid_step_shape=valid_step_shape,
            internal_margin=int(internal_margin),
            requested_bounds_origin=requested_bounds_origin,
        )
        if any(
            axis_slice.stop <= axis_slice.start
            for window in windows
            for axis_slice in window.extraction.raw_slices
        ):
            # An over-split candidate can produce crops that fall entirely
            # beyond the real volume (degenerate/reversed raw_slices), which
            # would fail at extraction time. The content-blind selection
            # never picks such candidates; skip them here too.
            continue
        empty_flags = tuple(
            region_is_definitely_empty(occupancy, raw_slices=window.extraction.raw_slices)
            for window in windows
        )
        skipped_voxels = sum(
            _product(int(axis) for axis in window.crop_shape)
            for window, is_empty in zip(windows, empty_flags)
            if is_empty
        )
        skipped_crop_count = sum(1 for is_empty in empty_flags if is_empty)
        crop_voxels = _product(int(candidate.max_crop_size) for candidate in combo)
        split_small_axis_penalty, extra_padding, preferred_edge_excess = (
            _candidate_grid_secondary_score(combo)
        )
        score = (
            -int(skipped_voxels),
            int(split_count_product),
            int(split_small_axis_penalty),
            int(extra_padding),
            int(preferred_edge_excess),
            -int(crop_voxels),
        )
        if best_score is None or score < best_score:
            best_combo = combo
            best_score = score
            best_skipped_crop_count = skipped_crop_count
            best_skipped_voxels = skipped_voxels
    if best_combo is None:
        raise ValueError("Could not select a zero-skip-aware crop grid.")
    return best_combo, int(best_skipped_crop_count), int(best_skipped_voxels)


def _build_crop_windows(
    *,
    requested_shape: ShapeZYX,
    raw_volume_shape: ShapeZYX,
    normalized_origin_in_raw: ShapeZYX,
    normalized_shape: ShapeZYX,
    requested_slices_in_normalized: SliceZYX,
    crop_grid_shape: ShapeZYX,
    valid_step_shape: ShapeZYX,
    internal_margin: int,
    requested_bounds_origin: ShapeZYX,
) -> Tuple[LargeCropWindow, ...]:
    windows: list[LargeCropWindow] = []
    requested_region = tuple(requested_slices_in_normalized)
    for grid_index in product(
        range(int(crop_grid_shape[0])),
        range(int(crop_grid_shape[1])),
        range(int(crop_grid_shape[2])),
    ):
        valid_slices = tuple(
            slice(
                int(grid_index[axis]) * int(valid_step_shape[axis]),
                (
                    int(grid_index[axis]) + 1
                )
                * int(valid_step_shape[axis]),
            )
            for axis in range(3)
        )
        crop_slices = tuple(
            slice(
                0
                if int(grid_index[axis]) == 0
                else int(valid_slices[axis].start) - int(internal_margin),
                int(normalized_shape[axis])
                if int(grid_index[axis]) == int(crop_grid_shape[axis]) - 1
                else int(valid_slices[axis].stop) + int(internal_margin),
            )
            for axis in range(3)
        )
        valid_slices_in_crop = _relative_slices(valid_slices, origin=crop_slices)
        requested_intersection = _intersect_slices(valid_slices, requested_region)
        requested_output_slices = _relative_slices(
            requested_intersection,
            origin=requested_region,
        )
        requested_output_slices_in_crop = _relative_slices(
            requested_intersection,
            origin=crop_slices,
        )
        requested_output_slices_in_raw = _shift_slices(
            requested_output_slices,
            offset=requested_bounds_origin,
        )
        extraction = _crop_extraction_plan(
            crop_slices=crop_slices,
            raw_volume_shape=raw_volume_shape,
            normalized_origin_in_raw=normalized_origin_in_raw,
        )
        windows.append(
            LargeCropWindow(
                index=len(windows),
                grid_index=(
                    int(grid_index[0]),
                    int(grid_index[1]),
                    int(grid_index[2]),
                ),
                crop_slices=crop_slices,
                valid_slices=valid_slices,
                valid_slices_in_crop=valid_slices_in_crop,
                requested_output_slices=requested_output_slices,
                requested_output_slices_in_crop=requested_output_slices_in_crop,
                requested_output_slices_in_raw=requested_output_slices_in_raw,
                extraction=extraction,
            )
        )
    return tuple(windows)


def _crop_extraction_plan(
    *,
    crop_slices: SliceZYX,
    raw_volume_shape: ShapeZYX,
    normalized_origin_in_raw: ShapeZYX,
) -> CropExtractionPlan:
    raw_slices: list[slice] = []
    pad_before: list[int] = []
    pad_after: list[int] = []
    for axis in range(3):
        raw_start = int(normalized_origin_in_raw[axis]) + int(crop_slices[axis].start)
        raw_stop = int(normalized_origin_in_raw[axis]) + int(crop_slices[axis].stop)
        clipped_start = max(0, int(raw_start))
        clipped_stop = min(int(raw_volume_shape[axis]), int(raw_stop))
        raw_slices.append(slice(clipped_start, clipped_stop))
        pad_before.append(max(0, -int(raw_start)))
        pad_after.append(max(0, int(raw_stop) - int(raw_volume_shape[axis])))
    return CropExtractionPlan(
        raw_slices=(raw_slices[0], raw_slices[1], raw_slices[2]),
        pad_before=(int(pad_before[0]), int(pad_before[1]), int(pad_before[2])),
        pad_after=(int(pad_after[0]), int(pad_after[1]), int(pad_after[2])),
    )


def _coerce_bounds(
    bounds: BoundsZYX,
    *,
    raw_volume_shape: ShapeZYX,
) -> BoundsZYX:
    if len(bounds) != 3:
        raise ValueError("requested_bounds must contain exactly 3 axes (z, y, x)")
    normalized: list[AxisBounds] = []
    for axis, axis_bounds in enumerate(bounds):
        if len(axis_bounds) != 2:
            raise ValueError(
                f"requested_bounds[{axis}] must contain exactly 2 values"
            )
        start = _coerce_non_negative_int(
            axis_bounds[0],
            name=f"requested_bounds[{axis}][0]",
        )
        stop = _coerce_positive_int(
            axis_bounds[1],
            name=f"requested_bounds[{axis}][1]",
        )
        if int(stop) <= int(start):
            raise ValueError(
                f"requested_bounds[{axis}] must satisfy start < stop"
            )
        if int(stop) > int(raw_volume_shape[axis]):
            raise ValueError(
                f"requested_bounds[{axis}] stop exceeds raw volume shape"
            )
        normalized.append((int(start), int(stop)))
    return (normalized[0], normalized[1], normalized[2])


def _coerce_shape(values: Sequence[object], *, name: str) -> ShapeZYX:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 values (z, y, x)")
    return (
        _coerce_positive_int(values[0], name=f"{name}[0]"),
        _coerce_positive_int(values[1], name=f"{name}[1]"),
        _coerce_positive_int(values[2], name=f"{name}[2]"),
    )


def _coerce_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    normalized = int(value)
    if normalized < 1:
        raise ValueError(f"{name} must be >= 1")
    return normalized


def _coerce_non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be >= 0")
    return normalized


def _shape_from_bounds(bounds: BoundsZYX) -> ShapeZYX:
    return tuple(int(stop) - int(start) for start, stop in bounds)  # type: ignore[return-value]


def _shape_from_slices(slices: SliceZYX) -> ShapeZYX:
    return tuple(_slice_size(axis_slice) for axis_slice in slices)  # type: ignore[return-value]


def _slice_size(axis_slice: slice) -> int:
    return max(0, int(axis_slice.stop) - int(axis_slice.start))


def _relative_slices(slices: SliceZYX, *, origin: SliceZYX) -> SliceZYX:
    return tuple(
        slice(
            int(slices[axis].start) - int(origin[axis].start),
            int(slices[axis].stop) - int(origin[axis].start),
        )
        for axis in range(3)
    )  # type: ignore[return-value]


def _shift_slices(slices: SliceZYX, *, offset: ShapeZYX) -> SliceZYX:
    return tuple(
        slice(
            int(slices[axis].start) + int(offset[axis]),
            int(slices[axis].stop) + int(offset[axis]),
        )
        for axis in range(3)
    )  # type: ignore[return-value]


def _intersect_slices(first: SliceZYX, second: SliceZYX) -> SliceZYX:
    return tuple(
        slice(
            max(int(first[axis].start), int(second[axis].start)),
            min(int(first[axis].stop), int(second[axis].stop)),
        )
        for axis in range(3)
    )  # type: ignore[return-value]


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return _ceil_div(int(value), int(multiple)) * int(multiple)


def _product(values) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return int(result)

