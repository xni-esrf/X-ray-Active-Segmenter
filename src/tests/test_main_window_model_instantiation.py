from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from src.ui.main_window import MainWindow
except Exception:  # pragma: no cover - environment dependent
    MainWindow = None  # type: ignore[assignment]


@unittest.skipUnless(MainWindow is not None, "MainWindow is not available")
class MainWindowModelInstantiationFlowTests(unittest.TestCase):
    def test_handle_load_model_request_does_not_prepare_learning_state(self) -> None:
        calls = []
        window_like = SimpleNamespace(
            _abort_if_learning_training_running=lambda: False,
            _ensure_learning_state_for_action=lambda _action: calls.append("prepare"),
            _instantiate_foundation_model_with_dialog=lambda: calls.append("load"),
        )

        with patch.object(
            MainWindow,
            "_inference_navigation_lock_active",
            return_value=False,
        ):
            MainWindow._handle_load_model_request(window_like)

        self.assertEqual(calls, ["load"])

    def test_instantiate_model_aborts_when_reinitialize_is_declined(self) -> None:
        existing_runtime = object()
        window_like = SimpleNamespace()
        preconditions = SimpleNamespace(
            checkpoint_path="foundation_model/weights_epoch_190.cp",
            num_classes=6,
            label_values=(0, 1, 2, 3, 4, 5),
            device_ids=(0, 1),
        )
        checkpoint_dialog_result = SimpleNamespace(
            accepted=True,
            path="foundation_model/weights_epoch_190.cp",
        )

        with patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=existing_runtime,
        ), patch(
            "src.ui.main_window.confirm_reinitialize_model",
            return_value=False,
        ) as confirm_mock, patch(
            "src.ui.main_window.validate_foundation_checkpoint_load_preconditions",
            return_value=preconditions,
        ) as validate_mock, patch(
            "src.ui.main_window.open_model_checkpoint_dialog",
            return_value=checkpoint_dialog_result,
        ) as checkpoint_dialog_mock, patch(
            "src.ui.main_window.instantiate_foundation_model_runtime"
        ) as instantiate_mock, patch(
            "src.ui.main_window.show_warning"
        ) as show_warning_mock, patch(
            "src.ui.main_window.show_info"
        ) as show_info_mock:
            result = MainWindow._instantiate_foundation_model_with_dialog(window_like)

        self.assertFalse(result)
        checkpoint_dialog_mock.assert_called_once_with(window_like)
        validate_mock.assert_called_once_with(
            "foundation_model/weights_epoch_190.cp",
            require_min_gpu_count=2,
        )
        confirm_mock.assert_called_once_with(parent=window_like)
        instantiate_mock.assert_not_called()
        show_warning_mock.assert_not_called()
        show_info_mock.assert_not_called()

    def test_instantiate_model_confirms_then_reinitializes_existing_runtime(self) -> None:
        existing_runtime = object()
        window_like = SimpleNamespace()
        preconditions = SimpleNamespace(
            checkpoint_path="foundation_model/weights_epoch_190.cp",
            num_classes=6,
            label_values=(0, 1, 2, 3, 4, 5),
            device_ids=(0, 1),
        )
        instantiated_runtime = SimpleNamespace(
            checkpoint_path="foundation_model/weights_epoch_190.cp",
            num_classes=6,
            device_ids=(0, 1),
            hyperparameters={},
        )
        checkpoint_dialog_result = SimpleNamespace(
            accepted=True,
            path="foundation_model/weights_epoch_190.cp",
        )

        with patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=existing_runtime,
        ), patch(
            "src.ui.main_window.confirm_reinitialize_model",
            return_value=True,
        ) as confirm_mock, patch(
            "src.ui.main_window.validate_foundation_checkpoint_load_preconditions",
            return_value=preconditions,
        ) as validate_mock, patch(
            "src.ui.main_window.open_model_checkpoint_dialog",
            return_value=checkpoint_dialog_result,
        ) as checkpoint_dialog_mock, patch(
            "src.ui.main_window.instantiate_foundation_model_runtime",
            return_value=instantiated_runtime,
        ) as instantiate_mock, patch(
            "src.ui.main_window.show_warning"
        ) as show_warning_mock, patch(
            "src.ui.main_window.show_info"
        ) as show_info_mock:
            result = MainWindow._instantiate_foundation_model_with_dialog(window_like)

        self.assertTrue(result)
        confirm_mock.assert_called_once_with(parent=window_like)
        checkpoint_dialog_mock.assert_called_once_with(window_like)
        validate_mock.assert_called_once_with(
            "foundation_model/weights_epoch_190.cp",
            require_min_gpu_count=2,
        )
        instantiate_mock.assert_called_once_with(
            num_classes=6,
            device_ids=(0, 1),
            checkpoint_path="foundation_model/weights_epoch_190.cp",
        )
        self.assertEqual(instantiated_runtime.hyperparameters.get("label_values"), (0, 1, 2, 3, 4, 5))
        show_warning_mock.assert_not_called()
        show_info_mock.assert_called_once()
        info_text = show_info_mock.call_args.args[0]
        self.assertIn("Foundation model loaded from checkpoint.", info_text)
        self.assertIn("weights_epoch_190.cp", info_text)

    def test_instantiate_model_warns_and_aborts_when_gpu_count_is_too_low(self) -> None:
        window_like = SimpleNamespace()
        checkpoint_dialog_result = SimpleNamespace(
            accepted=True,
            path="foundation_model/weights_epoch_190.cp",
        )

        with patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=None,
        ), patch(
            "src.ui.main_window.confirm_reinitialize_model"
        ) as confirm_mock, patch(
            "src.ui.main_window.validate_foundation_checkpoint_load_preconditions",
            side_effect=RuntimeError(
                "At least 2 CUDA devices are required to instantiate the model; found 1."
            ),
        ) as validate_mock, patch(
            "src.ui.main_window.open_model_checkpoint_dialog",
            return_value=checkpoint_dialog_result,
        ) as checkpoint_dialog_mock, patch(
            "src.ui.main_window.instantiate_foundation_model_runtime"
        ) as instantiate_mock, patch(
            "src.ui.main_window.show_warning"
        ) as show_warning_mock, patch(
            "src.ui.main_window.show_info"
        ) as show_info_mock:
            result = MainWindow._instantiate_foundation_model_with_dialog(window_like)

        self.assertFalse(result)
        confirm_mock.assert_not_called()
        checkpoint_dialog_mock.assert_called_once_with(window_like)
        validate_mock.assert_called_once_with(
            "foundation_model/weights_epoch_190.cp",
            require_min_gpu_count=2,
        )
        instantiate_mock.assert_not_called()
        show_info_mock.assert_not_called()
        show_warning_mock.assert_called_once()
        self.assertIn("At least 2 CUDA devices", show_warning_mock.call_args.args[0])

    def test_instantiate_model_warns_when_checkpoint_metadata_is_missing(self) -> None:
        window_like = SimpleNamespace()
        checkpoint_dialog_result = SimpleNamespace(
            accepted=True,
            path="foundation_model/weights_epoch_190.cp",
        )

        with patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=None,
        ), patch(
            "src.ui.main_window.confirm_reinitialize_model"
        ) as confirm_mock, patch(
            "src.ui.main_window.validate_foundation_checkpoint_load_preconditions",
            side_effect=ValueError(
                "Checkpoint metadata.hyperparameters.label_values is missing."
            ),
        ) as validate_mock, patch(
            "src.ui.main_window.open_model_checkpoint_dialog",
            return_value=checkpoint_dialog_result,
        ) as checkpoint_dialog_mock, patch(
            "src.ui.main_window.instantiate_foundation_model_runtime"
        ) as instantiate_mock, patch(
            "src.ui.main_window.show_warning"
        ) as show_warning_mock, patch(
            "src.ui.main_window.show_info"
        ) as show_info_mock:
            result = MainWindow._instantiate_foundation_model_with_dialog(window_like)

        self.assertFalse(result)
        confirm_mock.assert_not_called()
        checkpoint_dialog_mock.assert_called_once_with(window_like)
        validate_mock.assert_called_once_with(
            "foundation_model/weights_epoch_190.cp",
            require_min_gpu_count=2,
        )
        instantiate_mock.assert_not_called()
        show_info_mock.assert_not_called()
        show_warning_mock.assert_called_once()
        warning_text = show_warning_mock.call_args.args[0]
        self.assertIn("label_values is missing", warning_text)

    def test_instantiate_model_persists_label_values_from_checkpoint_metadata(self) -> None:
        window_like = SimpleNamespace()
        preconditions = SimpleNamespace(
            checkpoint_path="normalized/weights_epoch_190.cp",
            num_classes=3,
            label_values=(0, 1, 2),
            device_ids=(0, 1),
        )
        instantiated_runtime = SimpleNamespace(
            checkpoint_path="normalized/weights_epoch_190.cp",
            num_classes=3,
            device_ids=(0, 1),
            hyperparameters={},
        )
        checkpoint_dialog_result = SimpleNamespace(
            accepted=True,
            path="foundation_model/weights_epoch_190.cp",
        )

        with patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=None,
        ), patch(
            "src.ui.main_window.confirm_reinitialize_model"
        ) as confirm_mock, patch(
            "src.ui.main_window.validate_foundation_checkpoint_load_preconditions",
            return_value=preconditions,
        ) as validate_mock, patch(
            "src.ui.main_window.open_model_checkpoint_dialog",
            return_value=checkpoint_dialog_result,
        ) as checkpoint_dialog_mock, patch(
            "src.ui.main_window.instantiate_foundation_model_runtime",
            return_value=instantiated_runtime,
        ) as instantiate_mock, patch(
            "src.ui.main_window.show_warning"
        ) as show_warning_mock, patch(
            "src.ui.main_window.show_info"
        ) as show_info_mock:
            result = MainWindow._instantiate_foundation_model_with_dialog(window_like)

        self.assertTrue(result)
        confirm_mock.assert_not_called()
        checkpoint_dialog_mock.assert_called_once_with(window_like)
        validate_mock.assert_called_once_with(
            "foundation_model/weights_epoch_190.cp",
            require_min_gpu_count=2,
        )
        instantiate_mock.assert_called_once_with(
            num_classes=3,
            device_ids=(0, 1),
            checkpoint_path="normalized/weights_epoch_190.cp",
        )
        show_warning_mock.assert_not_called()
        show_info_mock.assert_called_once()
        self.assertEqual(instantiated_runtime.hyperparameters.get("label_values"), (0, 1, 2))

    def test_save_model_warns_and_aborts_when_model_runtime_is_missing(self) -> None:
        window_like = SimpleNamespace()

        with patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=None,
        ) as runtime_mock, patch(
            "src.ui.main_window.open_save_model_checkpoint_dialog"
        ) as dialog_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock, patch(
            "src.ui.main_window.show_info"
        ) as info_mock:
            result = MainWindow._save_model_with_dialog(window_like)

        self.assertFalse(result)
        runtime_mock.assert_called_once_with()
        dialog_mock.assert_not_called()
        warning_mock.assert_called_once()
        self.assertIn("Load a model before saving", warning_mock.call_args.args[0])
        self.assertIs(warning_mock.call_args.kwargs["parent"], window_like)
        info_mock.assert_not_called()

    def test_save_model_aborts_silently_when_dialog_is_canceled(self) -> None:
        runtime = SimpleNamespace(num_classes=6, device_ids=(0, 1))
        window_like = SimpleNamespace(
            _save_model_runtime_checkpoint=lambda *_args, **_kwargs: None,
        )

        with patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=runtime,
        ), patch(
            "src.ui.main_window.open_save_model_checkpoint_dialog",
            return_value=SimpleNamespace(accepted=False, path=None),
        ) as dialog_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock, patch(
            "src.ui.main_window.show_info"
        ) as info_mock:
            result = MainWindow._save_model_with_dialog(window_like)

        self.assertFalse(result)
        dialog_mock.assert_called_once_with(window_like)
        warning_mock.assert_not_called()
        info_mock.assert_not_called()

    def test_save_model_saves_checkpoint_and_reports_success(self) -> None:
        runtime = SimpleNamespace(num_classes=6, device_ids=(0, 1))
        save_calls = []
        window_like = SimpleNamespace(
            _save_model_runtime_checkpoint=lambda runtime_obj, *, checkpoint_path: save_calls.append(
                (runtime_obj, checkpoint_path)
            ),
        )

        with patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=runtime,
        ), patch(
            "src.ui.main_window.open_save_model_checkpoint_dialog",
            return_value=SimpleNamespace(accepted=True, path="tmp/checkpoint.cp"),
        ) as dialog_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock, patch(
            "src.ui.main_window.show_info"
        ) as info_mock:
            result = MainWindow._save_model_with_dialog(window_like)

        self.assertTrue(result)
        dialog_mock.assert_called_once_with(window_like)
        self.assertEqual(save_calls, [(runtime, "tmp/checkpoint.cp")])
        warning_mock.assert_not_called()
        info_mock.assert_called_once()
        info_text = info_mock.call_args.args[0]
        self.assertIn("Model checkpoint saved.", info_text)
        self.assertIn("tmp/checkpoint.cp", info_text)


if __name__ == "__main__":
    unittest.main()
