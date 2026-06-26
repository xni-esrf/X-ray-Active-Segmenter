from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from ...learning import (
    DEFAULT_TRAINING_PARAMETERS,
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
    bottom_panel: object


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
    mark_current_model_runtime_as_trained: Callable[..., object] = (
        lambda _context, *, completed_epoch_count: None
    )
    refresh_learning_inference_ui_state: Callable[[object], object] = (
        lambda _context: None
    )
    logger: object = _LOGGER


@dataclass
class TrainingController:
    context: TrainingControllerContext
    operations: TrainingControllerOperations

    def training_is_running(self) -> bool:
        return bool(self.context._training_running)

    def training_parameters(self):
        parameters = getattr(
            self.context,
            "_training_parameters",
            DEFAULT_TRAINING_PARAMETERS,
        )
        return validate_training_parameters(parameters)

    def handle_train_model_request(self) -> None:
        if self.operations.inference_navigation_lock_active(self.context):
            return
        abort_if_training_running = getattr(
            self.context,
            "_abort_if_learning_training_running",
            None,
        )
        if callable(abort_if_training_running) and abort_if_training_running():
            return
        ensure_learning_state = getattr(self.context, "_ensure_learning_state_for_action", None)
        if callable(ensure_learning_state):
            if not bool(ensure_learning_state("train")):
                return
        train_with_dialog = getattr(self.context, "_train_model_on_dataset_with_dialog", None)
        if callable(train_with_dialog):
            train_with_dialog()
        else:
            self.train_model_on_dataset_with_dialog()

    def handle_stop_training_request(self) -> None:
        request_stop = getattr(self.context, "_request_learning_training_stop", None)
        if callable(request_stop):
            request_stop()
            return
        self.request_learning_training_stop()

    def request_learning_training_stop(self) -> None:
        training_is_running = getattr(self.context, "_training_is_running", None)
        is_running = (
            bool(training_is_running())
            if callable(training_is_running)
            else self.training_is_running()
        )
        if not is_running:
            return
        worker = self.context._training_worker
        request_stop = getattr(worker, "request_stop", None)
        if callable(request_stop):
            request_stop()

    def train_model_on_dataset_with_dialog(self) -> bool:
        ensure_training_runtime = getattr(
            self.context,
            "_ensure_training_runtime_for_new_training",
            None,
        )
        if callable(ensure_training_runtime):
            if not bool(ensure_training_runtime()):
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
            start_background = getattr(
                self.context,
                "_start_learning_training_background",
                None,
            )
            if callable(start_background):
                start_background(preconditions=preconditions)
            else:
                self.start_learning_training_background(preconditions=preconditions)
        except Exception as exc:
            exit_running = getattr(
                self.context,
                "_exit_learning_training_running_state",
                None,
            )
            if callable(exit_running):
                exit_running()
            else:
                self.exit_learning_training_running_state()
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
            enter_running = getattr(
                self.context,
                "_enter_learning_training_running_state",
                None,
            )
            if callable(enter_running):
                enter_running(worker=worker, thread=thread)
            else:
                self.enter_learning_training_running_state(worker=worker, thread=thread)
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
        marker = getattr(self.context, "_mark_current_model_runtime_as_trained", None)
        if callable(marker):
            marker(completed_epoch_count=int(result.completed_epoch_count))
        else:
            self.operations.mark_current_model_runtime_as_trained(
                self.context,
                completed_epoch_count=int(result.completed_epoch_count),
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
        exit_running = getattr(self.context, "_exit_learning_training_running_state", None)
        if callable(exit_running):
            exit_running()
        else:
            self.exit_learning_training_running_state()
        if not bool(getattr(self.context, "_deferred_close_after_training", False)):
            return
        clear_state = getattr(self.context, "_clear_deferred_close_training_state", None)
        if callable(clear_state):
            clear_state()
        else:
            self.clear_deferred_close_training_state()

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
        training_is_running = getattr(self.context, "_training_is_running", None)
        training_running = (
            bool(training_is_running())
            if callable(training_is_running)
            else self.training_is_running()
        )
        self.context.bottom_panel.set_learning_training_running(training_running)
        refresh_inference = getattr(
            self.context,
            "_refresh_learning_inference_ui_state",
            None,
        )
        if callable(refresh_inference):
            refresh_inference()
        else:
            self.operations.refresh_learning_inference_ui_state(self.context)
        self.context.bottom_panel.set_stop_training_enabled(training_running)

    def enter_learning_training_running_state(self, *, worker: object, thread: object) -> None:
        self.context._training_running = True
        self.context._training_worker = worker
        self.context._training_thread = thread
        refresh = getattr(self.context, "_refresh_learning_training_ui_state", None)
        if callable(refresh):
            refresh()
        else:
            self.refresh_learning_training_ui_state()

    def exit_learning_training_running_state(self) -> None:
        self.context._training_running = False
        self.context._training_worker = None
        self.context._training_thread = None
        refresh = getattr(self.context, "_refresh_learning_training_ui_state", None)
        if callable(refresh):
            refresh()
        else:
            self.refresh_learning_training_ui_state()
