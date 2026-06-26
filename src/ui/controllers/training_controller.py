from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from ...learning import (
    LearningTrainingLoopResult,
    validate_training_parameters,
    validate_learning_model_training_preconditions,
)
from ...learning.qt_workers import LearningTrainingWorker
from ...utils import get_logger
from ..dialogs import show_info, show_warning


_LOGGER = get_logger(__name__)


class TrainingControllerContext(Protocol):
    _training_running: bool
    _training_worker: Optional[object]
    _training_thread: Optional[object]
    _deferred_close_after_training: bool
    _deferred_close_training_mode: str
    _training_parameters: object
    bottom_panel: object

    def _abort_if_learning_training_running(self) -> bool: ...

    def _ensure_learning_state_for_action(self, action: str) -> bool: ...

    def _train_model_on_dataset_with_dialog(self) -> bool: ...

    def _request_learning_training_stop(self) -> None: ...

    def _training_is_running(self) -> bool: ...

    def _ensure_training_runtime_for_new_training(self) -> bool: ...

    def _start_learning_training_background(self, *, preconditions: object) -> None: ...

    def _enter_learning_training_running_state(
        self,
        *,
        worker: object,
        thread: object,
    ) -> None: ...

    def _exit_learning_training_running_state(self) -> None: ...

    def _mark_current_model_runtime_as_trained(
        self,
        *,
        completed_epoch_count: int,
    ) -> None: ...

    def _clear_deferred_close_training_state(self) -> None: ...

    def _refresh_learning_training_ui_state(self) -> None: ...

    def _refresh_learning_inference_ui_state(self) -> None: ...


@dataclass(frozen=True)
class TrainingControllerOperations:
    show_warning: Callable[..., object] = show_warning
    show_info: Callable[..., object] = show_info
    validate_learning_model_training_preconditions: Callable[..., object] = (
        validate_learning_model_training_preconditions
    )
    qthread_factory: Callable[[object], object] = QThread
    training_worker_factory: Callable[[], object] = LearningTrainingWorker
    qapplication_instance: Callable[[], object] = QApplication.instance
    learning_session_kwargs: Callable[[object], dict[str, object]] = lambda _context: {}
    inference_navigation_lock_active: Callable[[object], bool] = lambda _context: False
    logger: object = _LOGGER


@dataclass
class TrainingController:
    context: TrainingControllerContext
    operations: TrainingControllerOperations

    def training_is_running(self) -> bool:
        return bool(self.context._training_running)

    def training_parameters(self):
        return validate_training_parameters(self.context._training_parameters)

    def handle_train_model_request(self) -> None:
        if self.operations.inference_navigation_lock_active(self.context):
            return
        if self.context._abort_if_learning_training_running():
            return
        if not bool(self.context._ensure_learning_state_for_action("train")):
            return
        self.context._train_model_on_dataset_with_dialog()

    def handle_stop_training_request(self) -> None:
        self.context._request_learning_training_stop()

    def request_learning_training_stop(self) -> None:
        if not self.context._training_is_running():
            return
        worker = self.context._training_worker
        request_stop = getattr(worker, "request_stop", None)
        if callable(request_stop):
            request_stop()

    def train_model_on_dataset_with_dialog(self) -> bool:
        if not bool(self.context._ensure_training_runtime_for_new_training()):
            return False

        try:
            preconditions = self.operations.validate_learning_model_training_preconditions(
                require_class_weights=True,
                **self.operations.learning_session_kwargs(self.context),
            )
        except Exception as exc:
            self.operations.show_warning(str(exc), parent=self.context)
            return False

        try:
            self.context._start_learning_training_background(
                preconditions=preconditions
            )
        except Exception as exc:
            self.context._exit_learning_training_running_state()
            self.operations.show_warning(str(exc), parent=self.context)
            return False
        return True

    def start_learning_training_background(self, *, preconditions: object) -> None:
        thread = self.operations.qthread_factory(self.context)
        worker = self.operations.training_worker_factory()
        training_parameters = self.training_parameters()
        worker.configure(
            preconditions=preconditions,
            early_stop_patience=training_parameters.early_stopping_patience,
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.completed.connect(self.context._on_learning_training_completed)
        worker.failed.connect(self.context._on_learning_training_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self.context._on_learning_training_thread_finished)
        thread.finished.connect(thread.deleteLater)

        try:
            self.context._enter_learning_training_running_state(
                worker=worker,
                thread=thread,
            )
            thread.start()
        except Exception:
            try:
                thread.quit()
            except Exception:
                pass
            try:
                worker.deleteLater()
            except Exception:
                pass
            try:
                thread.deleteLater()
            except Exception:
                pass
            raise

    def on_learning_training_completed(self, result: object) -> None:
        if not isinstance(result, LearningTrainingLoopResult):
            self.operations.show_warning(
                "Training finished with an invalid result payload.",
                parent=self.context,
            )
            return
        normalized_reason = str(result.stop_reason).strip().lower()
        if normalized_reason == "early_stop":
            stop_reason_text = "early stop"
        elif normalized_reason == "max_epoch":
            stop_reason_text = "max epoch"
        elif normalized_reason == "user_stop":
            stop_reason_text = "stopped by user"
        else:
            self.operations.show_warning(
                f"Training finished with an invalid stop reason: {result.stop_reason!r}.",
                parent=self.context,
            )
            return
        best_epoch_text = (
            "N/A"
            if result.best_epoch is None
            else str(int(result.best_epoch))
        )
        best_dice_text = (
            "N/A"
            if result.best_weighted_mean_dice is None
            else f"{float(result.best_weighted_mean_dice):.6g}"
        )
        self.context._mark_current_model_runtime_as_trained(
            completed_epoch_count=int(result.completed_epoch_count)
        )
        self.operations.show_info(
            (
                "Training is over.\n"
                f"- reason: {stop_reason_text}\n"
                f"- best epoch: {best_epoch_text}\n"
                f"- best weighted dice: {best_dice_text}"
            ),
            parent=self.context,
        )

    def on_learning_training_failed(self, message: str) -> None:
        normalized_message = str(message).strip()
        if not normalized_message:
            normalized_message = "Unknown training error."
        self.operations.show_warning(
            f"Training aborted: {normalized_message}",
            parent=self.context,
        )

    def on_learning_training_thread_finished(self) -> None:
        self.context._exit_learning_training_running_state()
        if not self.context._deferred_close_after_training:
            return
        self.context._clear_deferred_close_training_state()

        app_instance = self.operations.qapplication_instance()
        if app_instance is None:
            return
        set_quit_on_last = getattr(app_instance, "setQuitOnLastWindowClosed", None)
        if callable(set_quit_on_last):
            try:
                set_quit_on_last(True)
            except Exception:
                pass
        quit_method = getattr(app_instance, "quit", None)
        if callable(quit_method):
            quit_method()

    def clear_deferred_close_training_state(self) -> None:
        self.context._deferred_close_after_training = False
        self.context._deferred_close_training_mode = "none"

    def set_deferred_close_after_stop_training(self) -> None:
        self.context._deferred_close_after_training = True
        self.context._deferred_close_training_mode = "stop_and_close"

    def refresh_learning_training_ui_state(self) -> None:
        training_running = bool(self.context._training_is_running())
        self.context.bottom_panel.set_learning_training_running(training_running)
        self.context._refresh_learning_inference_ui_state()
        self.context.bottom_panel.set_stop_training_enabled(training_running)

    def enter_learning_training_running_state(self, *, worker: object, thread: object) -> None:
        self.context._training_running = True
        self.context._training_worker = worker
        self.context._training_thread = thread
        self.context._refresh_learning_training_ui_state()

    def exit_learning_training_running_state(self) -> None:
        self.context._training_running = False
        self.context._training_worker = None
        self.context._training_thread = None
        self.context._refresh_learning_training_ui_state()
