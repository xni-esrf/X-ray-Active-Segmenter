"""Controller building blocks for gradual MainWindow extraction."""

from .context import (
    ControllerContext,
    DialogCallback,
    DialogPort,
    MainWindowController,
)
from .learning_state_controller import (
    LearningStateController,
    LearningStateControllerOperations,
)
from .inference_controller import InferenceController, InferenceControllerOperations
from .model_controller import ModelController, ModelControllerOperations
from .training_controller import TrainingController, TrainingControllerOperations

__all__ = [
    "ControllerContext",
    "DialogCallback",
    "DialogPort",
    "InferenceController",
    "InferenceControllerOperations",
    "LearningStateController",
    "LearningStateControllerOperations",
    "MainWindowController",
    "ModelController",
    "ModelControllerOperations",
    "TrainingController",
    "TrainingControllerOperations",
]
