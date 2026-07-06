from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


@dataclass(frozen=True)
class TrainingParameters:
    learning_rate: float = 5e-5
    training_batch_size: int = 4
    validation_batch_size: int = 4
    inference_batch_size: int = 4
    patches_per_epoch: int = 2000
    early_stopping_patience: int = 7


DEFAULT_TRAINING_PARAMETERS = TrainingParameters()


def _coerce_positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if numeric <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return numeric


def _coerce_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    integer = int(value)
    if integer < 1:
        raise ValueError(f"{name} must be >= 1")
    return integer


def validate_training_parameters(parameters: TrainingParameters) -> TrainingParameters:
    if not isinstance(parameters, TrainingParameters):
        raise TypeError(
            "parameters must be a TrainingParameters instance, "
            f"got {type(parameters).__name__}"
        )
    return TrainingParameters(
        learning_rate=_coerce_positive_float(
            parameters.learning_rate,
            name="learning_rate",
        ),
        training_batch_size=_coerce_positive_int(
            parameters.training_batch_size,
            name="training_batch_size",
        ),
        validation_batch_size=_coerce_positive_int(
            parameters.validation_batch_size,
            name="validation_batch_size",
        ),
        inference_batch_size=_coerce_positive_int(
            parameters.inference_batch_size,
            name="inference_batch_size",
        ),
        patches_per_epoch=_coerce_positive_int(
            parameters.patches_per_epoch,
            name="patches_per_epoch",
        ),
        early_stopping_patience=_coerce_positive_int(
            parameters.early_stopping_patience,
            name="early_stopping_patience",
        ),
    )
