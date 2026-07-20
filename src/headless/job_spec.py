from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional

from ..learning.training_parameters import (
    DEFAULT_TRAINING_PARAMETERS,
    TrainingParameters,
    validate_training_parameters,
)


HeadlessJobKind = Literal["train", "inference"]
SegmentationKind = Literal["semantic", "instance"]

DEFAULT_HEADLESS_JOB_DIR_NAME = "headless-job"


@dataclass(frozen=True)
class HeadlessJobSpec:
    kind: HeadlessJobKind
    raw_volume_path: str
    bbox_path: str
    segmentation_path: Optional[str] = None
    segmentation_kind: SegmentationKind = "semantic"
    load_mode: str = "lazy"
    cache_max_bytes: int = 512 * 1024 * 1024
    training_parameters: TrainingParameters = DEFAULT_TRAINING_PARAMETERS
    input_checkpoint_path: Optional[str] = None
    output_checkpoint_path: Optional[str] = None
    output_segmentation_path: Optional[str] = None
    output_segmentation_format: Optional[str] = None
    job_dir: str = DEFAULT_HEADLESS_JOB_DIR_NAME
    created_at: str = ""
    source_pid: Optional[int] = None

    def __post_init__(self) -> None:
        kind = _coerce_kind(self.kind)
        segmentation_kind = _coerce_segmentation_kind(self.segmentation_kind)
        load_mode = str(self.load_mode).strip().lower()
        if load_mode not in {"ram", "lazy"}:
            raise ValueError("load_mode must be 'ram' or 'lazy'")
        cache_max_bytes = _coerce_positive_int(
            self.cache_max_bytes,
            name="cache_max_bytes",
        )
        training_parameters = validate_training_parameters(self.training_parameters)
        created_at = str(self.created_at).strip() or _utc_timestamp()

        _require_non_empty_path(self.raw_volume_path, name="raw_volume_path")
        _require_non_empty_path(self.bbox_path, name="bbox_path")
        _require_non_empty_path(self.job_dir, name="job_dir")
        if self.segmentation_path is not None:
            _require_non_empty_path(self.segmentation_path, name="segmentation_path")
        if self.input_checkpoint_path is not None:
            _require_non_empty_path(
                self.input_checkpoint_path,
                name="input_checkpoint_path",
            )

        if kind == "train":
            _require_non_empty_path(self.segmentation_path, name="segmentation_path")
            _require_non_empty_path(
                self.input_checkpoint_path,
                name="input_checkpoint_path",
            )
            _require_non_empty_path(
                self.output_checkpoint_path,
                name="output_checkpoint_path",
            )
        if kind == "inference":
            _require_non_empty_path(
                self.input_checkpoint_path,
                name="input_checkpoint_path",
            )
            _require_non_empty_path(
                self.output_segmentation_path,
                name="output_segmentation_path",
            )
            output_format = str(self.output_segmentation_format or "").strip().lower()
            if output_format != "zarr":
                raise ValueError(
                    "output_segmentation_format must be 'zarr' for headless inference"
                )

        source_pid = self.source_pid
        if source_pid is not None:
            source_pid = _coerce_positive_int(source_pid, name="source_pid")

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "segmentation_kind", segmentation_kind)
        object.__setattr__(self, "load_mode", load_mode)
        object.__setattr__(self, "cache_max_bytes", cache_max_bytes)
        object.__setattr__(self, "training_parameters", training_parameters)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "source_pid", source_pid)

    def to_json_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["training_parameters"] = asdict(self.training_parameters)
        return data

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> "HeadlessJobSpec":
        if not isinstance(data, Mapping):
            raise TypeError("headless job spec must be a JSON object")
        values = dict(data)
        raw_training_parameters = values.get("training_parameters", None)
        if raw_training_parameters is None:
            training_parameters = DEFAULT_TRAINING_PARAMETERS
        elif isinstance(raw_training_parameters, Mapping):
            training_parameters = TrainingParameters(
                learning_rate=raw_training_parameters.get("learning_rate", 5e-5),
                training_batch_size=raw_training_parameters.get(
                    "training_batch_size",
                    4,
                ),
                validation_batch_size=raw_training_parameters.get(
                    "validation_batch_size",
                    4,
                ),
                inference_batch_size=raw_training_parameters.get(
                    "inference_batch_size",
                    4,
                ),
                patches_per_epoch=raw_training_parameters.get(
                    "patches_per_epoch",
                    DEFAULT_TRAINING_PARAMETERS.patches_per_epoch,
                ),
                early_stopping_patience=raw_training_parameters.get(
                    "early_stopping_patience",
                    DEFAULT_TRAINING_PARAMETERS.early_stopping_patience,
                ),
                skip_empty_regions=raw_training_parameters.get(
                    "skip_empty_regions",
                    DEFAULT_TRAINING_PARAMETERS.skip_empty_regions,
                ),
            )
        else:
            raise TypeError("training_parameters must be a JSON object")
        values["training_parameters"] = training_parameters
        return cls(**values)


def save_headless_job_spec(spec: HeadlessJobSpec, path: str) -> str:
    normalized_path = str(Path(path).expanduser())
    target = Path(normalized_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(spec.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return normalized_path


def load_headless_job_spec(path: str) -> HeadlessJobSpec:
    normalized_path = str(Path(path).expanduser())
    data = json.loads(Path(normalized_path).read_text(encoding="utf-8"))
    return HeadlessJobSpec.from_json_dict(data)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _coerce_kind(value: object) -> HeadlessJobKind:
    normalized = str(value).strip().lower()
    if normalized not in {"train", "inference"}:
        raise ValueError("kind must be 'train' or 'inference'")
    return normalized  # type: ignore[return-value]


def _coerce_segmentation_kind(value: object) -> SegmentationKind:
    normalized = str(value).strip().lower()
    if normalized not in {"semantic", "instance"}:
        raise ValueError("segmentation_kind must be 'semantic' or 'instance'")
    return normalized  # type: ignore[return-value]


def _coerce_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if normalized <= 0:
        raise ValueError(f"{name} must be > 0")
    return normalized


def _require_non_empty_path(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path")
    return value
