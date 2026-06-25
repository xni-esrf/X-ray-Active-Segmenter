from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Optional, Sequence, Tuple

import numpy as np

from ..data import open_volume
from ..io.loader import InMemoryVolumeLoader
from ..io.saver import save_segmentation_volume
from ..learning import (
    LearningSession,
    apply_inference_predictions_to_array,
    instantiate_model_runtime_from_checkpoint,
    load_learning_sources_from_paths,
    prepare_learning_state_from_sources,
    run_learning_inference,
    save_foundation_model_checkpoint,
    train_learning_model_with_validation_loop,
    validate_foundation_checkpoint_load_preconditions,
    validate_foundation_model_instantiation_preconditions,
    validate_training_preconditions_for_session,
)
from ..utils import setup_logging
from .job_spec import HeadlessJobSpec, load_headless_job_spec


LOGGER = logging.getLogger(__name__)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        spec = load_headless_job_spec(args.job)
        _setup_headless_logging(spec=spec, log_level=args.log_level)
        run_headless_job(spec, validate_only=bool(args.validate_only))
    except NotImplementedError as exc:
        LOGGER.error("%s", exc)
        return 2
    except Exception as exc:
        LOGGER.exception("Headless job failed: %s", exc)
        return 1
    return 0


def run_headless_job(spec: HeadlessJobSpec, *, validate_only: bool = False) -> None:
    LOGGER.info("Starting headless %s job", spec.kind)
    LOGGER.info("Job directory: %s", spec.job_dir)
    context = _open_headless_inputs(spec)
    try:
        LOGGER.info(
            "Opened raw volume: path=%s shape=%s dtype=%s",
            spec.raw_volume_path,
            context.raw_volume.shape,
            context.raw_volume.dtype,
        )
        LOGGER.info(
            "Opened %s segmentation: path=%s shape=%s dtype=%s",
            spec.segmentation_kind,
            spec.segmentation_path,
            context.segmentation_volume.shape,
            context.segmentation_volume.dtype,
        )
        LOGGER.info("Loaded %d bounding box(es): %s", context.bbox_count, spec.bbox_path)
        if validate_only:
            LOGGER.info("Validation-only mode completed successfully.")
            return
        if spec.kind == "train":
            _run_training_job(spec, context)
            return
        if spec.kind == "inference":
            _run_inference_job(spec, context)
            return
        raise ValueError(f"Unsupported headless job kind: {spec.kind}")
    finally:
        context.close()


class _HeadlessInputContext:
    def __init__(self, *, sources) -> None:
        self.sources = sources
        self.raw_volume = sources.raw_volume
        self.segmentation_volume = sources.segmentation_volume
        self.bbox_count = int(len(sources.boxes_by_id))

    def close(self) -> None:
        close = getattr(self.sources, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                LOGGER.debug("Failed to close headless sources cleanly", exc_info=True)


def _open_headless_inputs(spec: HeadlessJobSpec) -> _HeadlessInputContext:
    _require_existing_input_path(spec.raw_volume_path, name="raw_volume_path")
    _require_existing_input_path(spec.segmentation_path, name="segmentation_path")
    _require_existing_input_path(spec.bbox_path, name="bbox_path")
    if spec.input_checkpoint_path is not None:
        _require_existing_input_path(
            spec.input_checkpoint_path,
            name="input_checkpoint_path",
        )
    if spec.kind == "train":
        _require_writable_output_path(
            spec.output_checkpoint_path,
            name="output_checkpoint_path",
        )
    if spec.kind == "inference":
        _require_writable_output_path(
            spec.output_segmentation_path,
            name="output_segmentation_path",
        )

    sources = load_learning_sources_from_paths(
        raw_volume_path=spec.raw_volume_path,
        segmentation_path=spec.segmentation_path,
        segmentation_kind=spec.segmentation_kind,
        bbox_path=spec.bbox_path,
        load_mode=spec.load_mode,
        cache_max_bytes=spec.cache_max_bytes,
    )
    return _HeadlessInputContext(sources=sources)


def _run_training_job(spec: HeadlessJobSpec, context: _HeadlessInputContext) -> None:
    if spec.input_checkpoint_path is None:
        raise ValueError("Training jobs require input_checkpoint_path.")
    if spec.output_checkpoint_path is None:
        raise ValueError("Training jobs require output_checkpoint_path.")

    session = LearningSession()
    LOGGER.info("Preparing training and validation datasets from saved inputs.")
    prepared_state = prepare_learning_state_from_sources(
        context.sources,
        training_parameters=spec.training_parameters,
        require_class_weights=True,
        learning_session=session,
    )
    if prepared_state.label_coverage_warning:
        LOGGER.warning("%s", prepared_state.label_coverage_warning)
    LOGGER.info(
        "Prepared learning state: train_boxes=%d validation_boxes=%d labels=%s",
        len(prepared_state.train_box_ids),
        len(prepared_state.validation_box_ids),
        tuple(prepared_state.label_space.label_values),
    )

    LOGGER.info("Instantiating model from checkpoint: %s", spec.input_checkpoint_path)
    instantiation_preconditions = validate_foundation_model_instantiation_preconditions(
        require_min_gpu_count=2,
        learning_session=session,
    )
    runtime = instantiate_model_runtime_from_checkpoint(
        checkpoint_path=spec.input_checkpoint_path,
        num_classes=int(instantiation_preconditions.num_classes),
        device_ids=tuple(instantiation_preconditions.device_ids),
        learning_session=session,
    )
    _persist_runtime_label_space_metadata(
        runtime,
        prepared_state.label_space,
        mark_trained=False,
    )

    preconditions = validate_training_preconditions_for_session(
        require_class_weights=True,
        learning_session=session,
    )
    LOGGER.info(
        "Training started: total epochs=%d, early_stop_patience=%d",
        int(2 * preconditions.train_runtime.train_count),
        int(spec.training_parameters.early_stopping_patience),
    )
    result = train_learning_model_with_validation_loop(
        preconditions=preconditions,
        mixed_precision=True,
        early_stop_patience=int(spec.training_parameters.early_stopping_patience),
        progress_callback=_log_training_epoch_progress,
        learning_session=session,
    )
    _persist_runtime_label_space_metadata(
        preconditions.model_runtime,
        prepared_state.label_space,
        mark_trained=True,
    )
    save_path = save_foundation_model_checkpoint(
        runtime=preconditions.model_runtime,
        checkpoint_path=spec.output_checkpoint_path,
    )
    LOGGER.info(
        "Training completed: reason=%s, completed_epochs=%d/%d, best_epoch=%s, "
        "best_weighted_dice=%s, checkpoint=%s",
        result.stop_reason,
        int(result.completed_epoch_count),
        int(result.total_epoch_count),
        "none" if result.best_epoch is None else str(result.best_epoch),
        (
            "none"
            if result.best_weighted_mean_dice is None
            else f"{float(result.best_weighted_mean_dice):.6g}"
        ),
        save_path,
    )


def _run_inference_job(spec: HeadlessJobSpec, context: _HeadlessInputContext) -> None:
    if spec.input_checkpoint_path is None:
        raise ValueError("Inference jobs require input_checkpoint_path.")
    if spec.output_segmentation_path is None:
        raise ValueError("Inference jobs require output_segmentation_path.")
    if spec.output_segmentation_format is None:
        raise ValueError("Inference jobs require output_segmentation_format.")

    inference_boxes = _inference_boxes_from_sources(context.sources)
    LOGGER.info(
        "Preparing headless inference: inference_boxes=%d checkpoint=%s",
        len(inference_boxes),
        spec.input_checkpoint_path,
    )
    checkpoint_preconditions = validate_foundation_checkpoint_load_preconditions(
        spec.input_checkpoint_path,
        require_min_gpu_count=2,
    )
    runtime = instantiate_model_runtime_from_checkpoint(
        checkpoint_path=spec.input_checkpoint_path,
        num_classes=int(checkpoint_preconditions.num_classes),
        device_ids=tuple(checkpoint_preconditions.device_ids),
    )

    raw_array = _read_full_volume_array(context.raw_volume)
    segmentation_array = _read_full_volume_array(context.segmentation_volume)
    LOGGER.info(
        "Inference started: boxes=%d label_values=%s",
        len(inference_boxes),
        tuple(int(value) for value in tuple(checkpoint_preconditions.label_values)),
    )
    inference_result = run_learning_inference(
        model_runtime=runtime,
        inference_boxes=inference_boxes,
        raw_array=raw_array,
        label_values=tuple(checkpoint_preconditions.label_values),
        volume_shape=tuple(int(value) for value in tuple(context.raw_volume.shape)),
        progress_callback=_log_inference_progress,
    )
    (
        output_array,
        changed_voxel_count,
        applied_box_ids,
        apply_failures_by_box_id,
    ) = apply_inference_predictions_to_array(
        segmentation_array,
        inference_result.predictions,
    )
    failures_by_box_id = dict(inference_result.failure_by_box_id)
    failures_by_box_id.update(apply_failures_by_box_id)
    if not applied_box_ids and failures_by_box_id:
        raise RuntimeError(
            "Headless inference failed for all boxes: "
            + "; ".join(
                f"{box_id}: {message}"
                for box_id, message in sorted(failures_by_box_id.items())
            )
        )

    output_volume = _open_output_segmentation_volume(
        output_array,
        path=spec.output_segmentation_path,
        source_volume=context.segmentation_volume,
    )
    save_path = save_segmentation_volume(
        output_volume,
        spec.output_segmentation_path,
        save_format=spec.output_segmentation_format,
        overwrite=True,
    )
    LOGGER.info(
        "Inference completed: boxes=%d applied=%d failed=%d changed_voxels=%d output=%s",
        int(inference_result.total_count),
        len(applied_box_ids),
        len(failures_by_box_id),
        int(changed_voxel_count),
        save_path,
    )
    if failures_by_box_id:
        for box_id, message in sorted(failures_by_box_id.items()):
            LOGGER.warning("Inference box failed: box_id=%s error=%s", box_id, message)
    for box_id, messages in sorted(inference_result.cleanup_errors_by_box_id.items()):
        for message in tuple(messages):
            LOGGER.warning("Inference cleanup warning: box_id=%s error=%s", box_id, message)


def _log_training_epoch_progress(progress: object) -> None:
    LOGGER.info(
        "Training epoch completed: %d/%d, loss=%.6g, weighted_dice=%.6g, "
        "best_epoch=%s, best_weighted_dice=%s, epochs_without_improvement=%d",
        int(getattr(progress, "completed_epoch_count")),
        int(getattr(progress, "total_epoch_count")),
        float(getattr(progress, "mean_loss")),
        float(getattr(progress, "weighted_mean_dice")),
        "none"
        if getattr(progress, "best_epoch") is None
        else str(getattr(progress, "best_epoch")),
        (
            "none"
            if getattr(progress, "best_weighted_mean_dice") is None
            else f"{float(getattr(progress, 'best_weighted_mean_dice')):.6g}"
        ),
        int(getattr(progress, "epochs_without_improvement")),
    )


def _log_inference_progress(progress: object) -> None:
    LOGGER.info(
        "Inference progress: %d/%d boxes (%.1f%%), box_id=%s, status=%s, failed=%d",
        int(getattr(progress, "completed_count")),
        int(getattr(progress, "total_count")),
        float(getattr(progress, "percent_complete")),
        str(getattr(progress, "box_id")),
        "ok" if bool(getattr(progress, "succeeded")) else "failed",
        int(getattr(progress, "failed_count")),
    )


def _inference_boxes_from_sources(sources: object) -> Tuple[object, ...]:
    boxes_by_id = getattr(sources, "boxes_by_id", {})
    ordered_box_ids = tuple(getattr(sources, "ordered_box_ids", tuple(boxes_by_id)))
    boxes = []
    for box_id in ordered_box_ids:
        box = boxes_by_id.get(box_id)
        if box is not None and str(getattr(box, "label", "")).strip().lower() == "inference":
            boxes.append(box)
    if not boxes:
        raise ValueError("Headless inference requires at least one bbox labeled 'inference'.")
    return tuple(boxes)


def _read_full_volume_array(volume: object) -> np.ndarray:
    return np.asarray(volume.get_chunk((slice(None), slice(None), slice(None))))


def _open_output_segmentation_volume(
    array: np.ndarray,
    *,
    path: str,
    source_volume: object,
) -> object:
    source_info = getattr(source_volume, "info", None)
    return open_volume(
        InMemoryVolumeLoader(
            path=path,
            array=array,
            voxel_spacing=getattr(source_info, "voxel_spacing", (1.0, 1.0, 1.0)),
            axes=getattr(source_info, "axes", "zyx"),
        )
    )


def _persist_runtime_label_space_metadata(
    runtime: object,
    label_space: object,
    *,
    mark_trained: bool,
) -> None:
    hyperparameters = getattr(runtime, "hyperparameters", None)
    if not isinstance(hyperparameters, dict):
        return
    label_values = tuple(int(value) for value in tuple(getattr(label_space, "label_values")))
    hyperparameters["label_values"] = label_values
    hyperparameters["label_space"] = {
        "label_values": label_values,
        "background_label": int(getattr(label_space, "background_label", 0)),
        "mask_label": int(getattr(label_space, "mask_label", -100)),
        "source_signature": getattr(label_space, "source_signature", None),
    }
    if bool(mark_trained):
        previous_count = hyperparameters.get("training_run_count", 0)
        try:
            normalized_count = int(previous_count)
        except Exception:
            normalized_count = 0
        if normalized_count < 0:
            normalized_count = 0
        hyperparameters["trained_in_app"] = True
        hyperparameters["training_run_count"] = normalized_count + 1
    if "source_checkpoint_path" not in hyperparameters:
        checkpoint_path = getattr(runtime, "checkpoint_path", None)
        if isinstance(checkpoint_path, str) and checkpoint_path.strip():
            hyperparameters["source_checkpoint_path"] = checkpoint_path


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a saved headless learning job")
    parser.add_argument("job", help="Path to .headless-job job.json")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and reopen job inputs without running training/inference",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level for terminal and job log output",
    )
    return parser.parse_args(argv)


def _setup_headless_logging(*, spec: HeadlessJobSpec, log_level: str) -> None:
    setup_logging(log_level)
    job_dir = Path(spec.job_dir).expanduser()
    job_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / "headless.log"
    root_logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, str(log_level).upper(), logging.INFO))
    root_logger.addHandler(file_handler)
    LOGGER.info("Writing headless log to %s", log_path)


def _require_existing_input_path(path: Optional[str], *, name: str) -> None:
    if path is None or not str(path).strip():
        raise ValueError(f"{name} must be a non-empty path")
    base = str(path).split("::", 1)[0]
    if not Path(base).expanduser().exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")


def _require_writable_output_path(path: Optional[str], *, name: str) -> None:
    if path is None or not str(path).strip():
        raise ValueError(f"{name} must be a non-empty path")
    target = Path(path).expanduser()
    parent = target.parent
    if parent and not parent.exists():
        raise FileNotFoundError(f"{name} parent directory does not exist: {parent}")


if __name__ == "__main__":
    sys.exit(main())
