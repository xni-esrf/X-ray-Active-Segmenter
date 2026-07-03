from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Optional, Sequence, Tuple

import numpy as np

from ..bbox import load_bounding_boxes
from ..data import open_volume
from ..io.loader import InMemoryVolumeLoader
from ..learning import (
    LearningSourceBundle,
    LearningSession,
    instantiate_model_runtime_from_checkpoint,
    load_learning_sources_from_paths,
    prepare_learning_state_from_sources,
    run_large_crop_inference_to_zarr,
    save_foundation_model_checkpoint,
    train_learning_model_with_validation_loop,
    validate_foundation_checkpoint_load_preconditions,
    validate_foundation_model_instantiation_preconditions,
    validate_training_preconditions_for_session,
)
from ..loading import load_prepared_volume
from ..utils import setup_logging
from .job_spec import HeadlessJobSpec, load_headless_job_spec


LOGGER = logging.getLogger(__name__)
_PERIODIC_CHECKPOINT_INTERVAL_EPOCHS = 5


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


def _periodic_checkpoint_path(final_checkpoint_path: str, *, epoch: int) -> str:
    normalized_epoch = int(epoch)
    if normalized_epoch <= 0:
        raise ValueError("epoch must be a positive 1-based integer")
    final_path = Path(str(final_checkpoint_path).strip())
    if final_path.suffix.lower() != ".cp":
        raise ValueError("Periodic training checkpoints require a final .cp checkpoint path.")
    return str(final_path.with_name(f"{final_path.stem}_epoch{normalized_epoch}{final_path.suffix}"))


class _HeadlessPeriodicCheckpointManager:
    def __init__(
        self,
        *,
        runtime: object,
        final_checkpoint_path: str,
        interval_epochs: int = _PERIODIC_CHECKPOINT_INTERVAL_EPOCHS,
    ) -> None:
        normalized_interval = int(interval_epochs)
        if normalized_interval <= 0:
            raise ValueError("interval_epochs must be positive")
        self._runtime = runtime
        self._final_checkpoint_path = str(final_checkpoint_path)
        self._interval_epochs = normalized_interval
        self._latest_checkpoint_path: Optional[Path] = None

    @property
    def latest_checkpoint_path(self) -> Optional[str]:
        if self._latest_checkpoint_path is None:
            return None
        return str(self._latest_checkpoint_path)

    def on_epoch_completed(self, progress: object) -> None:
        completed_epoch_count = int(getattr(progress, "completed_epoch_count"))
        if completed_epoch_count <= 0:
            return
        if completed_epoch_count % self._interval_epochs != 0:
            return

        checkpoint_path = _periodic_checkpoint_path(
            self._final_checkpoint_path,
            epoch=completed_epoch_count,
        )
        saved_path = save_foundation_model_checkpoint(
            runtime=self._runtime,
            checkpoint_path=checkpoint_path,
        )
        previous_path = self._latest_checkpoint_path
        self._latest_checkpoint_path = Path(saved_path)
        if previous_path is None or previous_path == self._latest_checkpoint_path:
            return
        try:
            previous_path.unlink()
        except FileNotFoundError:
            return
        except Exception:
            LOGGER.warning(
                "Failed to remove previous periodic checkpoint: %s",
                previous_path,
                exc_info=True,
            )

    def cleanup_after_successful_final_save(self) -> None:
        latest_path = self._latest_checkpoint_path
        if latest_path is None:
            return
        try:
            latest_path.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            LOGGER.warning(
                "Failed to remove periodic checkpoint after final checkpoint save: %s",
                latest_path,
                exc_info=True,
            )
            return
        self._latest_checkpoint_path = None


def _open_headless_inputs(spec: HeadlessJobSpec) -> _HeadlessInputContext:
    _require_existing_input_path(spec.raw_volume_path, name="raw_volume_path")
    _require_existing_input_path(spec.bbox_path, name="bbox_path")
    if spec.segmentation_path is not None:
        _require_existing_input_path(spec.segmentation_path, name="segmentation_path")
    elif spec.kind == "train":
        raise ValueError("Training jobs require segmentation_path.")
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

    if spec.segmentation_path is None:
        sources = _load_inference_sources_without_segmentation(spec)
    else:
        sources = load_learning_sources_from_paths(
            raw_volume_path=spec.raw_volume_path,
            segmentation_path=spec.segmentation_path,
            segmentation_kind=spec.segmentation_kind,
            bbox_path=spec.bbox_path,
            load_mode=spec.load_mode,
            cache_max_bytes=spec.cache_max_bytes,
        )
    return _HeadlessInputContext(sources=sources)


def _load_inference_sources_without_segmentation(spec: HeadlessJobSpec) -> LearningSourceBundle:
    raw = load_prepared_volume(
        spec.raw_volume_path,
        kind="raw",
        load_mode=spec.load_mode,
        cache_max_bytes=spec.cache_max_bytes,
        pyramid_levels=1,
    )
    bbox_payload = load_bounding_boxes(
        spec.bbox_path,
        expected_shape=raw.volume.shape,
    )
    empty_segmentation = np.zeros(raw.volume.shape, dtype=np.int32)
    segmentation_volume = open_volume(
        InMemoryVolumeLoader(
            path="<generated-empty-semantic-segmentation>",
            array=empty_segmentation,
            voxel_spacing=raw.volume.info.voxel_spacing,
            axes=raw.volume.info.axes,
        )
    )
    boxes = tuple(bbox_payload.boxes)
    return LearningSourceBundle(
        raw_volume=raw.volume,
        segmentation_volume=segmentation_volume,
        segmentation_kind="semantic",
        boxes_by_id={box.id: box for box in boxes},
        ordered_box_ids=tuple(box.id for box in boxes),
    )


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
    periodic_checkpoints = _HeadlessPeriodicCheckpointManager(
        runtime=preconditions.model_runtime,
        final_checkpoint_path=spec.output_checkpoint_path,
    )
    result = train_learning_model_with_validation_loop(
        preconditions=preconditions,
        mixed_precision=True,
        early_stop_patience=int(spec.training_parameters.early_stopping_patience),
        progress_callback=_log_training_epoch_progress,
        epoch_completion_callback=periodic_checkpoints.on_epoch_completed,
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
    if str(result.stop_reason).strip().lower() in {"early_stop", "max_epoch"}:
        periodic_checkpoints.cleanup_after_successful_final_save()
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
    if str(spec.output_segmentation_format).strip().lower() != "zarr":
        raise ValueError("Headless large-crop inference requires Zarr output.")

    inference_boxes = _inference_boxes_from_sources(context.sources)
    if len(inference_boxes) != 1:
        raise ValueError(
            "Headless large-crop inference currently requires exactly one inference bbox; "
            f"got {len(inference_boxes)}."
        )
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

    inference_box = inference_boxes[0]
    requested_bounds = (
        (int(inference_box.z0), int(inference_box.z1)),
        (int(inference_box.y0), int(inference_box.y1)),
        (int(inference_box.x0), int(inference_box.x1)),
    )
    LOGGER.info(
        "Large-crop inference started: boxes=%d label_values=%s requested_bounds=%s",
        len(inference_boxes),
        tuple(int(value) for value in tuple(checkpoint_preconditions.label_values)),
        requested_bounds,
    )

    result = run_large_crop_inference_to_zarr(
        model_runtime=runtime,
        raw_volume=context.raw_volume,
        requested_bounds=requested_bounds,
        label_values=tuple(checkpoint_preconditions.label_values),
        output_path=spec.output_segmentation_path,
        output_dtype=_headless_inference_output_dtype(
            tuple(checkpoint_preconditions.label_values),
            context=context,
        ),
        overwrite=True,
        batch_size=int(spec.training_parameters.inference_batch_size),
    )
    LOGGER.info(
        "Large-crop inference completed: crops=%d written=%d output=%s",
        int(result.plan.total_crop_count),
        int(result.written_crop_count),
        result.output_path,
    )


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


def _headless_inference_output_dtype(
    label_values: Sequence[object],
    *,
    context: _HeadlessInputContext,
) -> np.dtype:
    segmentation_volume = getattr(context, "segmentation_volume", None)
    segmentation_dtype = getattr(segmentation_volume, "dtype", None)
    if segmentation_dtype is not None:
        return np.dtype(segmentation_dtype)

    normalized_labels = tuple(int(value) for value in tuple(label_values))
    if not normalized_labels:
        return np.dtype(np.uint8)
    min_label = min(normalized_labels)
    max_label = max(normalized_labels)
    if min_label >= 0 and max_label <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if min_label >= 0 and max_label <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    return np.dtype(np.int64)


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
