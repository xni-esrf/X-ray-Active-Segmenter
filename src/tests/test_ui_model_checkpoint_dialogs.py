from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Optional
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    import src.ui.dialogs as dialogs_module
    from src.ui.dialogs import (
        InferenceCloseDecision,
        TrainingCloseDecision,
        ask_inference_running_close_decision,
        ask_training_running_close_decision,
        confirm_replace_training_model_with_default_checkpoint,
        open_save_model_checkpoint_dialog,
    )
except Exception:  # pragma: no cover - environment dependent
    dialogs_module = None  # type: ignore[assignment]
    InferenceCloseDecision = None  # type: ignore[assignment]
    TrainingCloseDecision = None  # type: ignore[assignment]
    ask_inference_running_close_decision = None  # type: ignore[assignment]
    ask_training_running_close_decision = None  # type: ignore[assignment]
    open_save_model_checkpoint_dialog = None  # type: ignore[assignment]
    confirm_replace_training_model_with_default_checkpoint = None  # type: ignore[assignment]


@unittest.skipUnless(open_save_model_checkpoint_dialog is not None, "Dialogs are not available")
class ModelCheckpointSaveDialogTests(unittest.TestCase):
    def test_open_save_model_checkpoint_dialog_appends_cp_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            selected_path = str(Path(temp_dir) / "my_model")
            with patch(
                "src.ui.dialogs.QFileDialog.getSaveFileName",
                return_value=(selected_path, "Model checkpoint (*.cp)"),
            ) as save_dialog_mock, patch(
                "src.ui.dialogs.confirm_overwrite"
            ) as overwrite_mock:
                result = open_save_model_checkpoint_dialog()

        self.assertTrue(result.accepted)
        self.assertEqual(result.path, selected_path + ".cp")
        save_dialog_mock.assert_called_once()
        overwrite_mock.assert_not_called()

    def test_open_save_model_checkpoint_dialog_aborts_when_overwrite_declined(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_path = Path(temp_dir) / "existing_model.cp"
            existing_path.write_text("already_there")
            with patch(
                "src.ui.dialogs.QFileDialog.getSaveFileName",
                return_value=(str(existing_path), "Model checkpoint (*.cp)"),
            ), patch(
                "src.ui.dialogs.confirm_overwrite",
                return_value=False,
            ) as overwrite_mock:
                result = open_save_model_checkpoint_dialog()

        self.assertFalse(result.accepted)
        self.assertIsNone(result.path)
        overwrite_mock.assert_called_once_with(str(existing_path), parent=None)

    def test_open_save_model_checkpoint_dialog_accepts_when_overwrite_confirmed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_path = Path(temp_dir) / "existing_model.cp"
            existing_path.write_text("already_there")
            with patch(
                "src.ui.dialogs.QFileDialog.getSaveFileName",
                return_value=(str(existing_path), "Model checkpoint (*.cp)"),
            ), patch(
                "src.ui.dialogs.confirm_overwrite",
                return_value=True,
            ) as overwrite_mock:
                result = open_save_model_checkpoint_dialog()

        self.assertTrue(result.accepted)
        self.assertEqual(result.path, str(existing_path))
        overwrite_mock.assert_called_once_with(str(existing_path), parent=None)

    def test_open_save_model_checkpoint_dialog_reprompts_when_overwrite_declined_with_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_path = Path(temp_dir) / "existing_model.cp"
            existing_path.write_text("already_there")
            alternative_path = Path(temp_dir) / "new_model.cp"
            with patch(
                "src.ui.dialogs.QFileDialog.getSaveFileName",
                side_effect=[
                    (str(existing_path), "Model checkpoint (*.cp)"),
                    (str(alternative_path), "Model checkpoint (*.cp)"),
                ],
            ) as save_dialog_mock, patch(
                "src.ui.dialogs.confirm_overwrite",
                return_value=False,
            ) as overwrite_mock:
                result = open_save_model_checkpoint_dialog(
                    retry_on_overwrite_decline=True
                )

        self.assertTrue(result.accepted)
        self.assertEqual(result.path, str(alternative_path))
        self.assertEqual(save_dialog_mock.call_count, 2)
        overwrite_mock.assert_called_once_with(str(existing_path), parent=None)

    def test_open_save_model_checkpoint_dialog_retry_returns_false_when_user_cancels_after_decline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_path = Path(temp_dir) / "existing_model.cp"
            existing_path.write_text("already_there")
            with patch(
                "src.ui.dialogs.QFileDialog.getSaveFileName",
                side_effect=[
                    (str(existing_path), "Model checkpoint (*.cp)"),
                    ("", "Model checkpoint (*.cp)"),
                ],
            ) as save_dialog_mock, patch(
                "src.ui.dialogs.confirm_overwrite",
                return_value=False,
            ) as overwrite_mock:
                result = open_save_model_checkpoint_dialog(
                    retry_on_overwrite_decline=True
                )

        self.assertFalse(result.accepted)
        self.assertIsNone(result.path)
        self.assertEqual(save_dialog_mock.call_count, 2)
        overwrite_mock.assert_called_once_with(str(existing_path), parent=None)


@unittest.skipUnless(
    confirm_replace_training_model_with_default_checkpoint is not None,
    "Dialogs are not available",
)
class TrainingModelReplacementDialogTests(unittest.TestCase):
    def test_confirm_replace_training_model_returns_true_when_yes(self) -> None:
        self.assertIsNotNone(dialogs_module)
        with patch(
            "src.ui.dialogs.QMessageBox.question",
            return_value=dialogs_module.QMessageBox.StandardButton.Yes,
        ):
            result = confirm_replace_training_model_with_default_checkpoint(
                checkpoint_path="foundation_model/weights_epoch_190.cp",
                parent=None,
            )

        self.assertTrue(result)

    def test_confirm_replace_training_model_returns_false_when_no(self) -> None:
        self.assertIsNotNone(dialogs_module)
        with patch(
            "src.ui.dialogs.QMessageBox.question",
            return_value=dialogs_module.QMessageBox.StandardButton.No,
        ):
            result = confirm_replace_training_model_with_default_checkpoint(
                checkpoint_path="foundation_model/weights_epoch_190.cp",
                parent=None,
            )

        self.assertFalse(result)


@unittest.skipUnless(
    ask_training_running_close_decision is not None and TrainingCloseDecision is not None,
    "Dialogs are not available",
)
class TrainingCloseDecisionDialogTests(unittest.TestCase):
    @staticmethod
    def _fake_message_box_class(*, clicked_label: Optional[str]):
        class _FakeMessageBox:
            class Icon:
                Warning = object()

            class ButtonRole:
                AcceptRole = object()
                DestructiveRole = object()
                RejectRole = object()

            def __init__(self, parent=None) -> None:
                self._clicked = None

            def setIcon(self, _icon) -> None:
                pass

            def setWindowTitle(self, _title: str) -> None:
                pass

            def setText(self, _text: str) -> None:
                pass

            def setInformativeText(self, _text: str) -> None:
                pass

            def addButton(self, text: str, _role):
                button = object()
                if clicked_label is not None and text == clicked_label:
                    self._clicked = button
                return button

            def setDefaultButton(self, _button) -> None:
                pass

            def exec(self) -> None:
                pass

            def clickedButton(self):
                return self._clicked

        return _FakeMessageBox

    def test_returns_stop_and_close_when_stop_button_clicked(self) -> None:
        fake_box = self._fake_message_box_class(
            clicked_label="Stop training and close"
        )
        with patch("src.ui.dialogs.QMessageBox", fake_box):
            result = ask_training_running_close_decision(parent=None)
        self.assertEqual(result, TrainingCloseDecision.STOP_AND_CLOSE)

    def test_returns_continue_in_background_when_continue_button_clicked(self) -> None:
        fake_box = self._fake_message_box_class(clicked_label="Continue in background")
        with patch("src.ui.dialogs.QMessageBox", fake_box):
            result = ask_training_running_close_decision(parent=None)
        self.assertEqual(result, TrainingCloseDecision.CONTINUE_IN_BACKGROUND)

    def test_returns_cancel_when_cancel_button_clicked(self) -> None:
        fake_box = self._fake_message_box_class(clicked_label="Cancel")
        with patch("src.ui.dialogs.QMessageBox", fake_box):
            result = ask_training_running_close_decision(parent=None)
        self.assertEqual(result, TrainingCloseDecision.CANCEL)

    def test_returns_cancel_when_no_button_is_clicked(self) -> None:
        fake_box = self._fake_message_box_class(clicked_label=None)
        with patch("src.ui.dialogs.QMessageBox", fake_box):
            result = ask_training_running_close_decision(parent=None)
        self.assertEqual(result, TrainingCloseDecision.CANCEL)


@unittest.skipUnless(
    ask_inference_running_close_decision is not None and InferenceCloseDecision is not None,
    "Dialogs are not available",
)
class InferenceCloseDecisionDialogTests(unittest.TestCase):
    @staticmethod
    def _fake_message_box_class(*, clicked_label: Optional[str]):
        class _FakeMessageBox:
            class Icon:
                Warning = object()

            class ButtonRole:
                AcceptRole = object()
                DestructiveRole = object()
                RejectRole = object()

            def __init__(self, parent=None) -> None:
                self._clicked = None

            def setIcon(self, _icon) -> None:
                pass

            def setWindowTitle(self, _title: str) -> None:
                pass

            def setText(self, _text: str) -> None:
                pass

            def setInformativeText(self, _text: str) -> None:
                pass

            def addButton(self, text: str, _role):
                button = object()
                if clicked_label is not None and text == clicked_label:
                    self._clicked = button
                return button

            def setDefaultButton(self, _button) -> None:
                pass

            def exec(self) -> None:
                pass

            def clickedButton(self):
                return self._clicked

        return _FakeMessageBox

    def test_returns_stop_and_close_when_stop_button_clicked(self) -> None:
        fake_box = self._fake_message_box_class(
            clicked_label="Stop inference and close"
        )
        with patch("src.ui.dialogs.QMessageBox", fake_box):
            result = ask_inference_running_close_decision(parent=None)
        self.assertEqual(result, InferenceCloseDecision.STOP_AND_CLOSE)

    def test_returns_continue_in_background_when_continue_button_clicked(self) -> None:
        fake_box = self._fake_message_box_class(clicked_label="Continue in background")
        with patch("src.ui.dialogs.QMessageBox", fake_box):
            result = ask_inference_running_close_decision(parent=None)
        self.assertEqual(result, InferenceCloseDecision.CONTINUE_IN_BACKGROUND)

    def test_returns_cancel_when_cancel_button_clicked(self) -> None:
        fake_box = self._fake_message_box_class(clicked_label="Cancel")
        with patch("src.ui.dialogs.QMessageBox", fake_box):
            result = ask_inference_running_close_decision(parent=None)
        self.assertEqual(result, InferenceCloseDecision.CANCEL)

    def test_returns_cancel_when_no_button_is_clicked(self) -> None:
        fake_box = self._fake_message_box_class(clicked_label=None)
        with patch("src.ui.dialogs.QMessageBox", fake_box):
            result = ask_inference_running_close_decision(parent=None)
        self.assertEqual(result, InferenceCloseDecision.CANCEL)


if __name__ == "__main__":
    unittest.main()
