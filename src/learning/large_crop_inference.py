from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from ..bbox import BoundingBox
from .inference import (
    LearningInferenceBackgroundResult,
    LearningInferencePrediction,
    run_learning_inference,
)
from .inference_geometry import (
    DEFAULT_INFERENCE_MINIVOL_SIZE,
    DEFAULT_INFERENCE_STRIDE,
    DEFAULT_LARGE_CROP_VOXEL_BUDGET,
)
from .large_crop_extraction import extract_large_crop_from_volume
from .large_crop_inference_plan import (
    BoundsZYX,
    LargeCropInferencePlan,
    LargeCropWindow,
    ShapeZYX,
    build_large_crop_inference_plan,
)
from .large_crop_zarr_writer import create_large_crop_zarr_output_writer


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OneCropLargeCropInferenceResult:
    output_path: str
    plan: LargeCropInferencePlan
    written_crop_count: int


@dataclass(frozen=True)
class LargeCropInferenceResult:
    output_path: str
    plan: LargeCropInferencePlan
    written_crop_count: int


def run_large_crop_inference_to_zarr(
    *,
    model_runtime: object,
    raw_volume: object,
    requested_bounds: BoundsZYX,
    label_values: Sequence[int],
    output_path: str,
    output_dtype: object = np.uint8,
    output_chunks: Optional[Sequence[object]] = None,
    overwrite: bool = False,
    batch_size: int = 16,
    context_margin: int = DEFAULT_INFERENCE_STRIDE,
    minivol_size: int = DEFAULT_INFERENCE_MINIVOL_SIZE,
    voxel_budget: int = DEFAULT_LARGE_CROP_VOXEL_BUDGET,
    writer_factory: Callable[..., object] = create_large_crop_zarr_output_writer,
    run_learning_inference_func: Callable[..., LearningInferenceBackgroundResult] = (
        run_learning_inference
    ),
    extract_large_crop_from_volume_func: Callable[..., np.ndarray] = (
        extract_large_crop_from_volume
    ),
) -> LargeCropInferenceResult:
    plan = _build_plan_for_volume(
        raw_volume=raw_volume,
        requested_bounds=requested_bounds,
        context_margin=context_margin,
        minivol_size=minivol_size,
        voxel_budget=voxel_budget,
    )
    return _run_large_crop_plan_to_zarr(
        plan=plan,
        model_runtime=model_runtime,
        raw_volume=raw_volume,
        label_values=label_values,
        output_path=output_path,
        output_dtype=output_dtype,
        output_chunks=output_chunks,
        overwrite=overwrite,
        batch_size=batch_size,
        writer_factory=writer_factory,
        run_learning_inference_func=run_learning_inference_func,
        extract_large_crop_from_volume_func=extract_large_crop_from_volume_func,
    )


def run_one_crop_large_crop_inference_to_zarr(
    *,
    model_runtime: object,
    raw_volume: object,
    requested_bounds: BoundsZYX,
    label_values: Sequence[int],
    output_path: str,
    output_dtype: object = np.uint8,
    output_chunks: Optional[Sequence[object]] = None,
    overwrite: bool = False,
    batch_size: int = 16,
    context_margin: int = DEFAULT_INFERENCE_STRIDE,
    minivol_size: int = DEFAULT_INFERENCE_MINIVOL_SIZE,
    voxel_budget: int = DEFAULT_LARGE_CROP_VOXEL_BUDGET,
    writer_factory: Callable[..., object] = create_large_crop_zarr_output_writer,
    run_learning_inference_func: Callable[..., LearningInferenceBackgroundResult] = (
        run_learning_inference
    ),
    extract_large_crop_from_volume_func: Callable[..., np.ndarray] = (
        extract_large_crop_from_volume
    ),
) -> OneCropLargeCropInferenceResult:
    plan = _build_plan_for_volume(
        raw_volume=raw_volume,
        requested_bounds=requested_bounds,
        context_margin=context_margin,
        minivol_size=minivol_size,
        voxel_budget=voxel_budget,
    )
    if plan.requires_cropping:
        raise ValueError(
            "run_one_crop_large_crop_inference_to_zarr requires a one-crop plan; "
            f"got crop_grid_shape={plan.crop_grid_shape}"
        )

    result = _run_large_crop_plan_to_zarr(
        plan=plan,
        model_runtime=model_runtime,
        raw_volume=raw_volume,
        label_values=label_values,
        output_path=output_path,
        output_dtype=output_dtype,
        output_chunks=output_chunks,
        overwrite=overwrite,
        batch_size=batch_size,
        writer_factory=writer_factory,
        run_learning_inference_func=run_learning_inference_func,
        extract_large_crop_from_volume_func=extract_large_crop_from_volume_func,
    )
    return OneCropLargeCropInferenceResult(
        output_path=result.output_path,
        plan=result.plan,
        written_crop_count=result.written_crop_count,
    )


def _build_plan_for_volume(
    *,
    raw_volume: object,
    requested_bounds: BoundsZYX,
    context_margin: int,
    minivol_size: int,
    voxel_budget: int,
) -> LargeCropInferencePlan:
    raw_volume_shape = _volume_shape(raw_volume)
    return build_large_crop_inference_plan(
        requested_bounds=requested_bounds,
        raw_volume_shape=raw_volume_shape,
        context_margin=context_margin,
        minivol_size=minivol_size,
        voxel_budget=voxel_budget,
    )


def _run_large_crop_plan_to_zarr(
    *,
    plan: LargeCropInferencePlan,
    model_runtime: object,
    raw_volume: object,
    label_values: Sequence[int],
    output_path: str,
    output_dtype: object,
    output_chunks: Optional[Sequence[object]],
    overwrite: bool,
    batch_size: int,
    writer_factory: Callable[..., object],
    run_learning_inference_func: Callable[..., LearningInferenceBackgroundResult],
    extract_large_crop_from_volume_func: Callable[..., np.ndarray],
) -> LargeCropInferenceResult:
    total_crops = int(plan.total_crop_count)
    normalized_labels = tuple(int(value) for value in tuple(label_values))
    output_dtype_np = np.dtype(output_dtype)
    _log_large_crop_plan(
        plan=plan,
        output_path=str(output_path),
        output_dtype=output_dtype_np,
        output_chunks=output_chunks,
        label_values=normalized_labels,
        batch_size=int(batch_size),
    )
    writer = writer_factory(
        output_path,
        shape=plan.requested_shape,
        dtype=output_dtype_np,
        chunks=output_chunks,
        overwrite=overwrite,
    )
    written_crop_count = 0
    for crop_number, window in enumerate(plan.windows, start=1):
        LOGGER.info(
            "Processing large crop %d/%d (%.1f%%): grid_index=%s crop_shape=%s valid_shape=%s dest=%s",
            int(crop_number),
            total_crops,
            _percent(int(crop_number) - 1, total_crops),
            window.grid_index,
            window.crop_shape,
            window.valid_shape,
            _format_slices(window.requested_output_slices),
        )
        LOGGER.info(
            "Extracting large crop %d/%d: crop=%s valid=%s raw=%s pad_before=%s pad_after=%s",
            int(crop_number),
            total_crops,
            _format_slices(window.crop_slices),
            _format_slices(window.valid_slices),
            _format_slices(window.extraction.raw_slices),
            window.extraction.pad_before,
            window.extraction.pad_after,
        )
        crop_array = np.asarray(
            extract_large_crop_from_volume_func(raw_volume, window=window)
        )
        if tuple(int(axis) for axis in crop_array.shape) != tuple(window.crop_shape):
            raise RuntimeError(
                "Extracted crop shape does not match plan: "
                f"got={tuple(crop_array.shape)} expected={window.crop_shape}"
            )
        LOGGER.info(
            "Extracted large crop %d/%d: shape=%s dtype=%s bytes=%d",
            int(crop_number),
            total_crops,
            tuple(int(axis) for axis in crop_array.shape),
            str(crop_array.dtype),
            int(crop_array.nbytes),
        )
        inference_result = _run_dense_inference_for_window(
            model_runtime=model_runtime,
            window=window,
            crop_array=crop_array,
            label_values=normalized_labels,
            batch_size=batch_size,
            crop_number=int(crop_number),
            total_crops=total_crops,
            run_learning_inference_func=run_learning_inference_func,
        )
        prediction = _single_successful_prediction(inference_result)
        predicted_bbox = np.asarray(prediction.predicted_bbox)
        predicted_bbox = _cast_prediction_labels_to_output_dtype(
            predicted_bbox,
            output_dtype=output_dtype_np,
            crop_number=int(crop_number),
            total_crops=total_crops,
        )
        LOGGER.info(
            "Dense inference prediction ready for large crop %d/%d: shape=%s dtype=%s bytes=%d",
            int(crop_number),
            total_crops,
            tuple(int(axis) for axis in predicted_bbox.shape),
            str(predicted_bbox.dtype),
            int(predicted_bbox.nbytes),
        )
        if bool(
            writer.write_window_prediction(
                predicted_bbox,
                window=window,
                crop_number=int(crop_number),
                total_crops=total_crops,
            )
        ):
            written_crop_count += 1
        LOGGER.info(
            "Finished large crop %d/%d (%.1f%%): written_crops=%d",
            int(crop_number),
            total_crops,
            _percent(int(crop_number), total_crops),
            int(written_crop_count),
        )

    LOGGER.info(
        "Large-crop inference completed: total_crops=%d written_crops=%d output_path=%s",
        total_crops,
        int(written_crop_count),
        str(output_path),
    )
    return LargeCropInferenceResult(
        output_path=str(output_path),
        plan=plan,
        written_crop_count=int(written_crop_count),
    )


def _run_dense_inference_for_window(
    *,
    model_runtime: object,
    window: LargeCropWindow,
    crop_array: np.ndarray,
    label_values: Tuple[int, ...],
    batch_size: int,
    crop_number: int,
    total_crops: int,
    run_learning_inference_func: Callable[..., LearningInferenceBackgroundResult],
) -> LearningInferenceBackgroundResult:
    crop_shape = tuple(int(axis) for axis in window.crop_shape)
    box = BoundingBox.from_bounds(
        box_id=f"large-crop-{int(window.index)}",
        z0=0,
        z1=int(crop_shape[0]),
        y0=0,
        y1=int(crop_shape[1]),
        x0=0,
        x1=int(crop_shape[2]),
        label="inference",
        volume_shape=crop_shape,
    )
    LOGGER.info(
        "Running dense inference for large crop %d/%d: crop_shape=%s batch_size=%d labels=%s",
        int(crop_number),
        int(total_crops),
        crop_shape,
        int(batch_size),
        label_values,
    )
    result = run_learning_inference_func(
        model_runtime=model_runtime,
        inference_boxes=(box,),
        raw_array=crop_array,
        label_values=label_values,
        volume_shape=crop_shape,
        progress_callback=lambda progress: _log_dense_inference_progress(
            progress,
            crop_number=int(crop_number),
            total_crops=int(total_crops),
        ),
        extract_bbox_context_from_array_func=(
            lambda array, **_kwargs: np.asarray(array)
        ),
        plan_bbox_context_func=lambda **_kwargs: _identity_context_plan(crop_shape),
        use_tiled_score_buffer=False,
        batch_size=int(batch_size),
        async_accumulation=True,
        async_accumulation_queue_size=2,
    )
    LOGGER.info(
        "Dense inference returned for large crop %d/%d: "
        "predictions=%d failures=%d cleanup_warning_boxes=%d",
        int(crop_number),
        int(total_crops),
        len(tuple(getattr(result, "predictions", ()))),
        len(dict(getattr(result, "failure_by_box_id", {}))),
        len(dict(getattr(result, "cleanup_errors_by_box_id", {}))),
    )
    return result


def _single_successful_prediction(
    inference_result: LearningInferenceBackgroundResult,
) -> LearningInferencePrediction:
    failures = dict(getattr(inference_result, "failure_by_box_id", {}))
    if failures:
        formatted = "; ".join(
            f"{box_id}: {message}" for box_id, message in failures.items()
        )
        raise RuntimeError(f"One-crop dense inference failed: {formatted}")
    predictions = tuple(getattr(inference_result, "predictions", ()))
    if len(predictions) != 1:
        raise RuntimeError(
            "One-crop dense inference expected exactly one prediction, "
            f"got {len(predictions)}"
        )
    return predictions[0]


def _cast_prediction_labels_to_output_dtype(
    predicted_bbox: np.ndarray,
    *,
    output_dtype: np.dtype,
    crop_number: int,
    total_crops: int,
) -> np.ndarray:
    array = np.asarray(predicted_bbox)
    dtype = np.dtype(output_dtype)
    if not np.issubdtype(dtype, np.integer):
        raise TypeError(f"output_dtype must be an integer dtype, got {dtype}")
    if int(array.size) > 0:
        min_value = int(np.min(array))
        max_value = int(np.max(array))
        info = np.iinfo(dtype)
        if min_value < int(info.min) or max_value > int(info.max):
            raise ValueError(
                "Dense inference prediction labels do not fit output dtype for "
                f"large crop {int(crop_number)}/{int(total_crops)}: "
                f"range=({min_value}, {max_value}) dtype={dtype}"
            )
    if array.dtype == dtype:
        return array
    return array.astype(dtype, copy=False)


def _log_large_crop_plan(
    *,
    plan: LargeCropInferencePlan,
    output_path: str,
    output_dtype: np.dtype,
    output_chunks: Optional[Sequence[object]],
    label_values: Tuple[int, ...],
    batch_size: int,
) -> None:
    LOGGER.info(
        "Large-crop inference started: total_crops=%d crop_grid_shape=%s "
        "requires_cropping=%s requested_bounds=%s requested_shape=%s output_path=%s",
        int(plan.total_crop_count),
        plan.crop_grid_shape,
        bool(plan.requires_cropping),
        plan.requested_bounds,
        plan.requested_shape,
        output_path,
    )
    LOGGER.info(
        "Large-crop inference plan: raw_shape=%s normalized_origin=%s "
        "normalized_shape=%s requested_in_normalized=%s",
        plan.raw_volume_shape,
        plan.normalized_origin_in_raw,
        plan.normalized_shape,
        _format_slices(plan.requested_slices_in_normalized),
    )
    LOGGER.info(
        "Large-crop inference parameters: context_margin=%d minivol_size=%d "
        "stride=%d internal_discard_margin=%d crop_extent_overlap=%d "
        "valid_step_shape=%s batch_size=%d labels=%s output_dtype=%s chunks=%s",
        int(plan.context_margin),
        int(plan.minivol_size),
        int(plan.stride),
        int(plan.internal_discard_margin),
        int(plan.crop_extent_overlap),
        plan.valid_step_shape,
        int(batch_size),
        label_values,
        str(output_dtype),
        "auto" if output_chunks is None else tuple(output_chunks),
    )


def _log_dense_inference_progress(
    progress: object,
    *,
    crop_number: int,
    total_crops: int,
) -> None:
    completed = int(getattr(progress, "completed_count"))
    total = int(getattr(progress, "total_count"))
    LOGGER.info(
        "Dense inference progress for large crop %d/%d: inner=%d/%d "
        "box_id=%s status=%s failed=%d",
        int(crop_number),
        int(total_crops),
        completed,
        total,
        str(getattr(progress, "box_id")),
        "ok" if bool(getattr(progress, "succeeded")) else "failed",
        int(getattr(progress, "failed_count")),
    )


def _identity_context_plan(shape: ShapeZYX):
    class _Axis:
        def __init__(self, size: int) -> None:
            self.extend_before = 0
            self.original_size = int(size)

    class _Plan:
        def __init__(self, shape_zyx: ShapeZYX) -> None:
            self.z = _Axis(int(shape_zyx[0]))
            self.y = _Axis(int(shape_zyx[1]))
            self.x = _Axis(int(shape_zyx[2]))

    return _Plan(shape)


def _volume_shape(volume: object) -> ShapeZYX:
    shape = getattr(volume, "shape", None)
    if shape is None:
        array = getattr(volume, "array", None)
        shape = getattr(array, "shape", None)
    if shape is None:
        raise TypeError("raw_volume must expose a shape")
    if len(shape) != 3:
        raise ValueError(f"raw_volume shape must be length 3, got {shape}")
    return (int(shape[0]), int(shape[1]), int(shape[2]))


def _format_slices(slices) -> str:
    return (
        "("
        + ", ".join(
            f"{int(axis_slice.start)}:{int(axis_slice.stop)}"
            for axis_slice in slices
        )
        + ")"
    )


def _percent(completed_count: int, total_count: int) -> float:
    if int(total_count) <= 0:
        return 100.0
    return 100.0 * float(completed_count) / float(total_count)
