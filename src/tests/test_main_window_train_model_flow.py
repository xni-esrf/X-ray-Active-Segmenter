from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from src.ui.main_window import _LearningTrainingWorker
    from src.learning import LearningTrainingLoopResult
    from src.ui.main_window import MainWindow
except Exception:  # pragma: no cover - environment dependent
    _LearningTrainingWorker = None  # type: ignore[assignment]
    MainWindow = None  # type: ignore[assignment]
    LearningTrainingLoopResult = None  # type: ignore[assignment]


@unittest.skipUnless(MainWindow is not None, "MainWindow is not available")
class MainWindowTrainModelFlowTests(unittest.TestCase):
    def test_handle_train_model_request_calls_train_dialog_when_not_running(self) -> None:
        called = []
        window_like = SimpleNamespace(
            _abort_if_learning_training_running=lambda: False,
            _train_model_on_dataset_with_dialog=lambda: called.append("train"),
        )

        MainWindow._handle_train_model_request(window_like)

        self.assertEqual(called, ["train"])

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

    @unittest.skipUnless(
        LearningTrainingLoopResult is not None,
        "Learning training loop result type unavailable",
    )
    def test_on_learning_training_completed_shows_summary_message(self) -> None:
        result = LearningTrainingLoopResult(
            completed_epoch_count=4,
            total_epoch_count=6,
            stop_reason="early_stop",
            best_epoch_index=1,
            best_weighted_mean_accuracy=0.8125,
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
        self.assertIn("best epoch (0-based): 1", info_text)
        self.assertIn("best weighted accuracy: 0.8125", info_text)
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
            best_epoch_index=1,
            best_weighted_mean_accuracy=0.8125,
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
            best_epoch_index=None,
            best_weighted_mean_accuracy=None,
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
        self.assertIn("best epoch (0-based): N/A", info_text)
        self.assertIn("best weighted accuracy: N/A", info_text)

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
    def test_on_learning_training_completed_logs_without_dialog_in_background_close_mode(self) -> None:
        result = LearningTrainingLoopResult(
            completed_epoch_count=3,
            total_epoch_count=6,
            stop_reason="max_epoch",
            best_epoch_index=2,
            best_weighted_mean_accuracy=0.91,
            early_stop_patience=2,
            mixed_precision_enabled=True,
        )
        marker_calls: list[int] = []
        window_like = SimpleNamespace(
            _deferred_close_after_training=True,
            _deferred_close_training_mode="continue_in_background",
            _deferred_close_checkpoint_path="/tmp/background_best.cp",
            _mark_current_model_runtime_as_trained=lambda completed_epoch_count: marker_calls.append(
                int(completed_epoch_count)
            ),
        )

        with patch("src.ui.main_window._LOGGER") as logger_mock, patch(
            "src.ui.main_window.show_info"
        ) as info_mock, patch("src.ui.main_window.show_warning") as warning_mock:
            MainWindow._on_learning_training_completed(window_like, result)

        self.assertEqual(marker_calls, [3])
        info_mock.assert_not_called()
        warning_mock.assert_not_called()
        logger_mock.info.assert_called_once()
        self.assertIn("Background training completed", logger_mock.info.call_args.args[0])

    @unittest.skipUnless(
        LearningTrainingLoopResult is not None,
        "Learning training loop result type unavailable",
    )
    def test_on_learning_training_completed_logs_invalid_payload_without_dialog_in_background_close_mode(self) -> None:
        window_like = SimpleNamespace(
            _deferred_close_after_training=True,
            _deferred_close_training_mode="continue_in_background",
        )

        with patch("src.ui.main_window._LOGGER") as logger_mock, patch(
            "src.ui.main_window.show_info"
        ) as info_mock, patch("src.ui.main_window.show_warning") as warning_mock:
            MainWindow._on_learning_training_completed(window_like, object())

        info_mock.assert_not_called()
        warning_mock.assert_not_called()
        logger_mock.error.assert_called_once()
        self.assertIn(
            "invalid result payload",
            str(logger_mock.error.call_args.args[0]).lower(),
        )

    @unittest.skipUnless(
        LearningTrainingLoopResult is not None,
        "Learning training loop result type unavailable",
    )
    def test_on_learning_training_completed_rejects_invalid_stop_reason(self) -> None:
        result = LearningTrainingLoopResult(
            completed_epoch_count=2,
            total_epoch_count=2,
            stop_reason="unexpected_reason",
            best_epoch_index=0,
            best_weighted_mean_accuracy=0.5,
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

    def test_on_learning_training_failed_logs_without_dialog_in_background_close_mode(self) -> None:
        window_like = SimpleNamespace(
            _deferred_close_after_training=True,
            _deferred_close_training_mode="continue_in_background",
        )

        with patch("src.ui.main_window._LOGGER") as logger_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock:
            MainWindow._on_learning_training_failed(
                window_like,
                "training crashed",
            )

        warning_mock.assert_not_called()
        logger_mock.error.assert_called_once()
        self.assertIn(
            "Background training aborted",
            logger_mock.error.call_args.args[0],
        )

    def test_on_learning_training_failed_logs_save_error_without_dialog_in_background_close_mode(self) -> None:
        window_like = SimpleNamespace(
            _deferred_close_after_training=True,
            _deferred_close_training_mode="continue_in_background",
        )

        with patch("src.ui.main_window._LOGGER") as logger_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock:
            MainWindow._on_learning_training_failed(
                window_like,
                "Failed to save training completion checkpoint to /tmp/best.cp: disk full",
            )

        warning_mock.assert_not_called()
        logger_mock.error.assert_called_once()
        self.assertIn(
            "completion checkpoint save failed",
            str(logger_mock.error.call_args.args[0]).lower(),
        )

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


@unittest.skipUnless(
    _LearningTrainingWorker is not None and LearningTrainingLoopResult is not None,
    "Training worker is not available",
)
class LearningTrainingWorkerTests(unittest.TestCase):
    def test_worker_completion_checkpoint_request_can_be_set_and_cleared(self) -> None:
        worker = _LearningTrainingWorker()

        worker.request_completion_checkpoint_save("  /tmp/background_best.cp  ")
        self.assertEqual(
            worker._completion_checkpoint_save_path(),
            "/tmp/background_best.cp",
        )

        worker.clear_completion_checkpoint_save_request()
        self.assertIsNone(worker._completion_checkpoint_save_path())

    def test_worker_run_saves_completion_checkpoint_on_max_epoch(self) -> None:
        worker = _LearningTrainingWorker()
        preconditions = object()
        worker.configure(
            preconditions=preconditions,
            completion_checkpoint_path="/tmp/background_best.cp",
        )
        result = LearningTrainingLoopResult(
            completed_epoch_count=4,
            total_epoch_count=8,
            stop_reason="max_epoch",
            best_epoch_index=3,
            best_weighted_mean_accuracy=0.87,
            early_stop_patience=2,
            mixed_precision_enabled=True,
        )
        completed_payloads: list[object] = []
        failed_messages: list[str] = []
        finished_calls: list[str] = []
        worker.completed.connect(lambda payload: completed_payloads.append(payload))
        worker.failed.connect(lambda message: failed_messages.append(str(message)))
        worker.finished.connect(lambda: finished_calls.append("finished"))

        runtime = object()
        with patch(
            "src.ui.main_window.train_learning_model_with_validation_loop",
            return_value=result,
        ) as train_mock, patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=runtime,
        ) as runtime_mock, patch(
            "src.ui.main_window.save_foundation_model_checkpoint"
        ) as save_mock:
            worker.run()

        train_mock.assert_called_once_with(
            preconditions=preconditions,
            mixed_precision=True,
            early_stop_patience=2,
            stop_event=worker._stop_event,
        )
        runtime_mock.assert_called_once_with()
        save_mock.assert_called_once_with(
            runtime=runtime,
            checkpoint_path="/tmp/background_best.cp",
        )
        self.assertEqual(completed_payloads, [result])
        self.assertEqual(failed_messages, [])
        self.assertEqual(finished_calls, ["finished"])

    def test_worker_run_skips_completion_checkpoint_save_on_user_stop(self) -> None:
        worker = _LearningTrainingWorker()
        preconditions = object()
        worker.configure(
            preconditions=preconditions,
            completion_checkpoint_path="/tmp/background_best.cp",
        )
        result = LearningTrainingLoopResult(
            completed_epoch_count=0,
            total_epoch_count=8,
            stop_reason="user_stop",
            best_epoch_index=None,
            best_weighted_mean_accuracy=None,
            early_stop_patience=2,
            mixed_precision_enabled=True,
        )

        with patch(
            "src.ui.main_window.train_learning_model_with_validation_loop",
            return_value=result,
        ) as train_mock, patch(
            "src.ui.main_window.get_current_learning_model_runtime"
        ) as runtime_mock, patch(
            "src.ui.main_window.save_foundation_model_checkpoint"
        ) as save_mock:
            worker.run()

        train_mock.assert_called_once()
        runtime_mock.assert_not_called()
        save_mock.assert_not_called()

    def test_worker_run_reports_failure_when_completion_checkpoint_save_fails(self) -> None:
        worker = _LearningTrainingWorker()
        preconditions = object()
        worker.configure(
            preconditions=preconditions,
            completion_checkpoint_path="/tmp/background_best.cp",
        )
        result = LearningTrainingLoopResult(
            completed_epoch_count=4,
            total_epoch_count=8,
            stop_reason="early_stop",
            best_epoch_index=3,
            best_weighted_mean_accuracy=0.87,
            early_stop_patience=2,
            mixed_precision_enabled=True,
        )
        completed_payloads: list[object] = []
        failed_messages: list[str] = []
        finished_calls: list[str] = []
        worker.completed.connect(lambda payload: completed_payloads.append(payload))
        worker.failed.connect(lambda message: failed_messages.append(str(message)))
        worker.finished.connect(lambda: finished_calls.append("finished"))

        with patch(
            "src.ui.main_window.train_learning_model_with_validation_loop",
            return_value=result,
        ), patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=None,
        ):
            worker.run()

        self.assertEqual(completed_payloads, [])
        self.assertEqual(finished_calls, ["finished"])
        self.assertEqual(len(failed_messages), 1)
        self.assertIn("completion checkpoint", failed_messages[0].lower())


if __name__ == "__main__":
    unittest.main()
