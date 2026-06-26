from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from src.learning import (
        DEFAULT_FOUNDATION_CHECKPOINT_PATH,
        LearningLabelSpace,
        LearningSession,
        LearningTrainingLoopResult,
        TrainingParameters,
    )
    from src.ui.main_window import MainWindow
except Exception:  # pragma: no cover - environment dependent
    DEFAULT_FOUNDATION_CHECKPOINT_PATH = "foundation_model/MAE_XNT.cp"
    MainWindow = None  # type: ignore[assignment]
    LearningLabelSpace = None  # type: ignore[assignment]
    LearningSession = None  # type: ignore[assignment]
    LearningTrainingLoopResult = None  # type: ignore[assignment]
    TrainingParameters = None  # type: ignore[assignment]


class _FakeSignal:
    def __init__(self) -> None:
        self.callbacks: list[object] = []

    def connect(self, callback: object) -> None:
        self.callbacks.append(callback)


class _FakeTrainingThread:
    def __init__(self, parent: object) -> None:
        self.parent = parent
        self.started = _FakeSignal()
        self.finished = _FakeSignal()
        self.started_called = False

    def start(self) -> None:
        self.started_called = True

    def quit(self) -> None:
        pass

    def deleteLater(self) -> None:
        pass


class _FakeTrainingWorker:
    def __init__(self) -> None:
        self.completed = _FakeSignal()
        self.failed = _FakeSignal()
        self.finished = _FakeSignal()
        self.configure_kwargs: dict[str, object] = {}
        self.thread: object | None = None

    def configure(self, **kwargs: object) -> None:
        self.configure_kwargs = dict(kwargs)

    def moveToThread(self, thread: object) -> None:
        self.thread = thread

    def run(self) -> None:
        pass

    def deleteLater(self) -> None:
        pass


@unittest.skipUnless(MainWindow is not None, "MainWindow is not available")
class MainWindowTrainModelFlowTests(unittest.TestCase):
    def test_handle_train_model_request_calls_train_dialog_when_not_running(self) -> None:
        called = []
        window_like = SimpleNamespace(
            _abort_if_learning_training_running=lambda: False,
            _ensure_learning_state_for_action=lambda action: called.append(
                ("ensure", str(action))
            )
            or True,
            _train_model_on_dataset_with_dialog=lambda: called.append("train"),
        )

        MainWindow._handle_train_model_request(window_like)

        self.assertEqual(called, [("ensure", "train"), "train"])

    def test_handle_train_model_request_aborts_when_learning_state_prepare_fails(self) -> None:
        called = []
        window_like = SimpleNamespace(
            _abort_if_learning_training_running=lambda: False,
            _ensure_learning_state_for_action=lambda action: called.append(
                ("ensure", str(action))
            )
            or False,
            _train_model_on_dataset_with_dialog=lambda: called.append("train"),
        )

        MainWindow._handle_train_model_request(window_like)

        self.assertEqual(called, [("ensure", "train")])

    def test_train_model_on_dataset_warns_and_aborts_on_precondition_error(self) -> None:
        window_like = SimpleNamespace(
            _ensure_training_runtime_for_new_training=lambda: True,
            _start_learning_training_background=lambda **_: None,
            _exit_learning_training_running_state=lambda: None,
        )

        with patch(
            "src.ui.main_window.validate_learning_model_training_preconditions",
            side_effect=ValueError("Cannot train model because required learning state is missing:\n- foo"),
        ) as validate_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock:
            result = MainWindow._train_model_on_dataset_with_dialog(window_like)

        self.assertFalse(result)
        validate_mock.assert_called_once_with(require_class_weights=True)
        warning_mock.assert_called_once()
        self.assertIn("required learning state is missing", warning_mock.call_args.args[0].lower())
        self.assertIs(warning_mock.call_args.kwargs["parent"], window_like)

    def test_train_model_on_dataset_starts_background_on_valid_preconditions(self) -> None:
        started = []
        window_like = SimpleNamespace(
            _ensure_training_runtime_for_new_training=lambda: True,
            _start_learning_training_background=lambda **kwargs: started.append(kwargs),
            _exit_learning_training_running_state=lambda: None,
        )
        preconditions = object()

        with patch(
            "src.ui.main_window.validate_learning_model_training_preconditions",
            return_value=preconditions,
        ) as validate_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock:
            result = MainWindow._train_model_on_dataset_with_dialog(window_like)

        self.assertTrue(result)
        validate_mock.assert_called_once_with(require_class_weights=True)
        warning_mock.assert_not_called()
        self.assertEqual(started, [{"preconditions": preconditions}])

    @unittest.skipUnless(
        TrainingParameters is not None,
        "Training parameters type unavailable",
    )
    def test_start_learning_training_background_uses_current_early_stop_patience(self) -> None:
        preconditions = object()
        fake_thread = _FakeTrainingThread(parent=None)
        fake_worker = _FakeTrainingWorker()
        entered = []
        window_like = SimpleNamespace(
            _training_parameters=TrainingParameters(early_stopping_patience=6),
            _training_running=False,
            _training_worker=None,
            _training_thread=None,
            _deferred_close_after_training=False,
            _deferred_close_training_mode="none",
            _deferred_close_checkpoint_path=None,
            _on_learning_training_completed=lambda _result: None,
            _on_learning_training_failed=lambda _message: None,
            _on_learning_training_thread_finished=lambda: None,
            _enter_learning_training_running_state=lambda **kwargs: entered.append(kwargs),
        )

        with patch("src.ui.main_window.QThread", return_value=fake_thread), patch(
            "src.ui.main_window.LearningTrainingWorker",
            return_value=fake_worker,
        ):
            MainWindow._start_learning_training_background(
                window_like,
                preconditions=preconditions,
            )

        self.assertEqual(
            fake_worker.configure_kwargs,
            {
                "preconditions": preconditions,
                "early_stop_patience": 6,
            },
        )
        self.assertIs(fake_worker.thread, fake_thread)
        self.assertEqual(entered, [{"worker": fake_worker, "thread": fake_thread}])
        self.assertTrue(fake_thread.started_called)

    def test_train_model_on_dataset_warns_and_exits_when_background_start_fails(self) -> None:
        exits = []
        window_like = SimpleNamespace(
            _ensure_training_runtime_for_new_training=lambda: True,
            _start_learning_training_background=lambda **_: (_ for _ in ()).throw(
                RuntimeError("background thread failed to start")
            ),
            _exit_learning_training_running_state=lambda: exits.append("exit"),
        )
        preconditions = object()

        with patch(
            "src.ui.main_window.validate_learning_model_training_preconditions",
            return_value=preconditions,
        ) as validate_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock:
            result = MainWindow._train_model_on_dataset_with_dialog(window_like)

        self.assertFalse(result)
        validate_mock.assert_called_once_with(require_class_weights=True)
        self.assertEqual(exits, ["exit"])
        warning_mock.assert_called_once()
        self.assertIn("failed to start", warning_mock.call_args.args[0].lower())
        self.assertIs(warning_mock.call_args.kwargs["parent"], window_like)

    def test_train_model_on_dataset_aborts_when_training_runtime_policy_rejects(self) -> None:
        window_like = SimpleNamespace(
            _ensure_training_runtime_for_new_training=lambda: False,
            _start_learning_training_background=lambda **_: None,
            _exit_learning_training_running_state=lambda: None,
        )

        with patch(
            "src.ui.main_window.validate_learning_model_training_preconditions"
        ) as validate_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock:
            result = MainWindow._train_model_on_dataset_with_dialog(window_like)

        self.assertFalse(result)
        validate_mock.assert_not_called()
        warning_mock.assert_not_called()

    def test_ensure_training_runtime_reinitializes_when_runtime_is_missing(self) -> None:
        calls = []
        window_like = SimpleNamespace(
            _reinitialize_training_runtime_from_default_checkpoint=lambda: calls.append(
                "reinitialize"
            )
            or True,
        )

        with patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=None,
        ), patch(
            "src.ui.main_window.confirm_replace_training_model_with_default_checkpoint"
        ) as confirm_mock:
            result = MainWindow._ensure_training_runtime_for_new_training(window_like)

        self.assertTrue(result)
        self.assertEqual(calls, ["reinitialize"])
        confirm_mock.assert_not_called()

    def test_ensure_training_runtime_aborts_when_confirmation_declined(self) -> None:
        calls = []
        runtime = object()
        window_like = SimpleNamespace(
            _runtime_requires_training_reinitialization=lambda _runtime: True,
            _reinitialize_training_runtime_from_default_checkpoint=lambda: calls.append(
                "reinitialize"
            )
            or True,
        )

        with patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=runtime,
        ), patch(
            "src.ui.main_window.confirm_replace_training_model_with_default_checkpoint",
            return_value=False,
        ) as confirm_mock:
            result = MainWindow._ensure_training_runtime_for_new_training(window_like)

        self.assertFalse(result)
        confirm_mock.assert_called_once()
        self.assertEqual(calls, [])

    def test_ensure_training_runtime_reinitializes_when_confirmation_accepted(self) -> None:
        calls = []
        runtime = object()
        window_like = SimpleNamespace(
            _runtime_requires_training_reinitialization=lambda _runtime: True,
            _reinitialize_training_runtime_from_default_checkpoint=lambda: calls.append(
                "reinitialize"
            )
            or True,
        )

        with patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=runtime,
        ), patch(
            "src.ui.main_window.confirm_replace_training_model_with_default_checkpoint",
            return_value=True,
        ) as confirm_mock:
            result = MainWindow._ensure_training_runtime_for_new_training(window_like)

        self.assertTrue(result)
        confirm_mock.assert_called_once()
        self.assertEqual(calls, ["reinitialize"])

    def test_ensure_training_runtime_keeps_runtime_when_policy_allows_it(self) -> None:
        runtime = object()
        calls = []
        window_like = SimpleNamespace(
            _runtime_requires_training_reinitialization=lambda _runtime: False,
            _reinitialize_training_runtime_from_default_checkpoint=lambda: calls.append(
                "reinitialize"
            )
            or True,
        )

        with patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=runtime,
        ), patch(
            "src.ui.main_window.confirm_replace_training_model_with_default_checkpoint"
        ) as confirm_mock:
            result = MainWindow._ensure_training_runtime_for_new_training(window_like)

        self.assertTrue(result)
        self.assertEqual(calls, [])
        confirm_mock.assert_not_called()

    def test_runtime_reuse_accepts_runtime_matching_current_label_space(self) -> None:
        session = LearningSession()
        session.set_label_space(LearningLabelSpace(label_values=(0, 2, 5)))
        runtime = SimpleNamespace(
            checkpoint_path=DEFAULT_FOUNDATION_CHECKPOINT_PATH,
            num_classes=3,
            hyperparameters={
                "source_checkpoint_path": DEFAULT_FOUNDATION_CHECKPOINT_PATH,
                "trained_in_app": False,
                "training_run_count": 0,
                "label_values": (0, 2, 5),
            },
        )
        window_like = SimpleNamespace(_learning_session=session)

        result = MainWindow._runtime_requires_training_reinitialization(
            window_like,
            runtime,
        )

        self.assertFalse(result)

    def test_runtime_reuse_rejects_runtime_label_values_outside_current_label_space(
        self,
    ) -> None:
        session = LearningSession()
        session.set_label_space(LearningLabelSpace(label_values=(0, 2, 5)))
        runtime = SimpleNamespace(
            checkpoint_path=DEFAULT_FOUNDATION_CHECKPOINT_PATH,
            num_classes=3,
            hyperparameters={
                "source_checkpoint_path": DEFAULT_FOUNDATION_CHECKPOINT_PATH,
                "trained_in_app": False,
                "training_run_count": 0,
                "label_values": (0, 1, 2),
            },
        )
        window_like = SimpleNamespace(_learning_session=session)

        result = MainWindow._runtime_requires_training_reinitialization(
            window_like,
            runtime,
        )

        self.assertTrue(result)

    def test_runtime_reuse_rejects_runtime_label_space_metadata_mismatch(
        self,
    ) -> None:
        session = LearningSession()
        session.set_label_space(LearningLabelSpace(label_values=(0, 2, 5)))
        runtime = SimpleNamespace(
            checkpoint_path=DEFAULT_FOUNDATION_CHECKPOINT_PATH,
            num_classes=3,
            hyperparameters={
                "source_checkpoint_path": DEFAULT_FOUNDATION_CHECKPOINT_PATH,
                "trained_in_app": False,
                "training_run_count": 0,
                "label_values": (0, 2, 5),
                "label_space": {
                    "label_values": (0, 2, 5),
                    "background_label": 0,
                    "mask_label": -1,
                },
            },
        )
        window_like = SimpleNamespace(_learning_session=session)

        result = MainWindow._runtime_requires_training_reinitialization(
            window_like,
            runtime,
        )

        self.assertTrue(result)

    @unittest.skipUnless(
        LearningTrainingLoopResult is not None,
        "Learning training loop result type unavailable",
    )
    def test_on_learning_training_completed_shows_summary_message(self) -> None:
        result = LearningTrainingLoopResult(
            completed_epoch_count=4,
            total_epoch_count=6,
            stop_reason="early_stop",
            best_epoch=1,
            best_weighted_mean_dice=0.8125,
            early_stop_patience=2,
            mixed_precision_enabled=True,
        )
        window_like = SimpleNamespace()

        with patch("src.ui.main_window.show_info") as info_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock:
            MainWindow._on_learning_training_completed(window_like, result)

        warning_mock.assert_not_called()
        info_mock.assert_called_once()
        info_text = info_mock.call_args.args[0]
        self.assertIn("Training is over.", info_text)
        self.assertIn("reason: early stop", info_text)
        self.assertIn("best epoch: 1", info_text)
        self.assertIn("best weighted dice: 0.8125", info_text)
        self.assertIs(info_mock.call_args.kwargs["parent"], window_like)

    @unittest.skipUnless(
        LearningTrainingLoopResult is not None,
        "Learning training loop result type unavailable",
    )
    def test_on_learning_training_completed_marks_runtime_as_trained_when_epochs_completed(self) -> None:
        result = LearningTrainingLoopResult(
            completed_epoch_count=2,
            total_epoch_count=6,
            stop_reason="early_stop",
            best_epoch=1,
            best_weighted_mean_dice=0.8125,
            early_stop_patience=2,
            mixed_precision_enabled=True,
        )
        runtime = SimpleNamespace(
            hyperparameters={
                "trained_in_app": False,
                "training_run_count": 0,
                "source_checkpoint_path": "foundation_model/weights_epoch_190.cp",
            },
            checkpoint_path="foundation_model/weights_epoch_190.cp",
        )
        window_like = SimpleNamespace()

        with patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=runtime,
        ), patch("src.ui.main_window.show_info") as info_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock:
            MainWindow._on_learning_training_completed(window_like, result)

        warning_mock.assert_not_called()
        info_mock.assert_called_once()
        self.assertTrue(runtime.hyperparameters["trained_in_app"])
        self.assertEqual(runtime.hyperparameters["training_run_count"], 1)

    @unittest.skipUnless(
        LearningTrainingLoopResult is not None,
        "Learning training loop result type unavailable",
    )
    def test_on_learning_training_completed_user_stop_without_best_shows_na_fields(self) -> None:
        result = LearningTrainingLoopResult(
            completed_epoch_count=0,
            total_epoch_count=6,
            stop_reason="user_stop",
            best_epoch=None,
            best_weighted_mean_dice=None,
            early_stop_patience=2,
            mixed_precision_enabled=True,
        )
        window_like = SimpleNamespace()

        with patch("src.ui.main_window.show_info") as info_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock:
            MainWindow._on_learning_training_completed(window_like, result)

        warning_mock.assert_not_called()
        info_mock.assert_called_once()
        info_text = info_mock.call_args.args[0]
        self.assertIn("reason: stopped by user", info_text)
        self.assertIn("best epoch: N/A", info_text)
        self.assertIn("best weighted dice: N/A", info_text)

    @unittest.skipUnless(
        LearningTrainingLoopResult is not None,
        "Learning training loop result type unavailable",
    )
    def test_on_learning_training_completed_rejects_invalid_payload(self) -> None:
        window_like = SimpleNamespace()

        with patch("src.ui.main_window.show_info") as info_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock:
            MainWindow._on_learning_training_completed(window_like, object())

        info_mock.assert_not_called()
        warning_mock.assert_called_once()
        self.assertIn("invalid result payload", warning_mock.call_args.args[0].lower())

    @unittest.skipUnless(
        LearningTrainingLoopResult is not None,
        "Learning training loop result type unavailable",
    )
    def test_on_learning_training_completed_rejects_invalid_stop_reason(self) -> None:
        result = LearningTrainingLoopResult(
            completed_epoch_count=2,
            total_epoch_count=2,
            stop_reason="unexpected_reason",
            best_epoch=1,
            best_weighted_mean_dice=0.5,
            early_stop_patience=2,
            mixed_precision_enabled=True,
        )
        window_like = SimpleNamespace()

        with patch("src.ui.main_window.show_info") as info_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock:
            MainWindow._on_learning_training_completed(window_like, result)

        info_mock.assert_not_called()
        warning_mock.assert_called_once()
        self.assertIn("invalid stop reason", warning_mock.call_args.args[0].lower())

    def test_on_learning_training_failed_shows_warning(self) -> None:
        window_like = SimpleNamespace()

        with patch("src.ui.main_window.show_warning") as warning_mock:
            MainWindow._on_learning_training_failed(window_like, " boom ")

        warning_mock.assert_called_once()
        warning_text = warning_mock.call_args.args[0]
        self.assertIn("Training aborted:", warning_text)
        self.assertIn("boom", warning_text)
        self.assertIs(warning_mock.call_args.kwargs["parent"], window_like)

    def test_on_learning_training_thread_finished_exits_running_state(self) -> None:
        exits = []
        window_like = SimpleNamespace(
            _exit_learning_training_running_state=lambda: exits.append("exit"),
        )

        MainWindow._on_learning_training_thread_finished(window_like)

        self.assertEqual(exits, ["exit"])

    def test_on_learning_training_thread_finished_with_deferred_close_quits_application(self) -> None:
        exits: list[str] = []
        clears: list[str] = []
        quit_on_last_values: list[bool] = []
        quit_calls: list[str] = []
        app_like = SimpleNamespace(
            setQuitOnLastWindowClosed=lambda value: quit_on_last_values.append(bool(value)),
            quit=lambda: quit_calls.append("quit"),
        )
        window_like = SimpleNamespace(
            _deferred_close_after_training=True,
            _exit_learning_training_running_state=lambda: exits.append("exit"),
            _clear_deferred_close_training_state=lambda: clears.append("clear"),
        )

        with patch("src.ui.main_window.QApplication.instance", return_value=app_like):
            MainWindow._on_learning_training_thread_finished(window_like)

        self.assertEqual(exits, ["exit"])
        self.assertEqual(clears, ["clear"])
        self.assertEqual(quit_on_last_values, [True])
        self.assertEqual(quit_calls, ["quit"])

    def test_on_learning_training_thread_finished_with_deferred_close_without_app_instance(self) -> None:
        exits: list[str] = []
        clears: list[str] = []
        window_like = SimpleNamespace(
            _deferred_close_after_training=True,
            _exit_learning_training_running_state=lambda: exits.append("exit"),
            _clear_deferred_close_training_state=lambda: clears.append("clear"),
        )

        with patch("src.ui.main_window.QApplication.instance", return_value=None):
            MainWindow._on_learning_training_thread_finished(window_like)

        self.assertEqual(exits, ["exit"])
        self.assertEqual(clears, ["clear"])


if __name__ == "__main__":
    unittest.main()
