from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QMessageBox,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..learning import (
    DEFAULT_TRAINING_PARAMETERS,
    TrainingParameters,
    validate_training_parameters,
)


@dataclass
class DialogResult:
    accepted: bool
    message: Optional[str] = None
    path: Optional[str] = None


@dataclass
class SaveDialogResult:
    accepted: bool
    path: Optional[str] = None
    format: Optional[str] = None


@dataclass
class TrainingParametersDialogResult:
    accepted: bool
    parameters: Optional[TrainingParameters] = None


class UnsavedChangesDecision(str, Enum):
    SAVE = "save"
    DISCARD = "discard"
    CANCEL = "cancel"


class TrainingCloseDecision(str, Enum):
    STOP_AND_CLOSE = "stop_and_close"
    CONTINUE_IN_BACKGROUND = "continue_in_background"
    CANCEL = "cancel"


class InferenceCloseDecision(str, Enum):
    STOP_AND_CLOSE = "stop_and_close"
    CONTINUE_IN_BACKGROUND = "continue_in_background"
    CANCEL = "cancel"


def show_warning(message: str, parent: Optional[QWidget] = None) -> DialogResult:
    QMessageBox.warning(parent, "Warning", message)
    return DialogResult(accepted=False, message=message)


def show_info(message: str, parent: Optional[QWidget] = None) -> DialogResult:
    QMessageBox.information(parent, "Info", message)
    return DialogResult(accepted=True, message=message)


def confirm_overwrite(path: str, parent: Optional[QWidget] = None) -> bool:
    answer = QMessageBox.question(
        parent,
        "Overwrite Existing File",
        f"The selected path already exists:\n{path}\n\nDo you want to overwrite it?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return answer == QMessageBox.StandardButton.Yes


def confirm_reinitialize_model(parent: Optional[QWidget] = None) -> bool:
    answer = QMessageBox.question(
        parent,
        "Re-initialize Model",
        (
            "A model is already instantiated.\n\n"
            "It will be re-initialized if you continue.\n\n"
            "Do you want to continue?"
        ),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return answer == QMessageBox.StandardButton.Yes


def confirm_replace_training_model_with_default_checkpoint(
    *,
    checkpoint_path: str,
    parent: Optional[QWidget] = None,
) -> bool:
    normalized_checkpoint_path = str(checkpoint_path).strip()
    if not normalized_checkpoint_path:
        normalized_checkpoint_path = "foundation_model/weights_epoch_190.cp"
    answer = QMessageBox.question(
        parent,
        "Replace Training Model",
        (
            "Training is only allowed from the default foundation checkpoint.\n\n"
            "The current in-memory model will be replaced before training starts.\n\n"
            f"Checkpoint: {normalized_checkpoint_path}\n\n"
            "Do you want to continue?"
        ),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return answer == QMessageBox.StandardButton.Yes


class TrainingParametersDialog(QDialog):
    def __init__(
        self,
        parameters: TrainingParameters,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Training Parameters")
        self._parameters = validate_training_parameters(parameters)

        layout = QVBoxLayout(self)

        warning = QLabel(
            "These options are intended for advanced users. "
            "The default training parameters should be appropriate most of the time."
        )
        warning.setWordWrap(True)
        warning.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(warning)

        form = QFormLayout()
        self._learning_rate = QDoubleSpinBox()
        self._learning_rate.setRange(1e-12, 1e6)
        self._learning_rate.setDecimals(12)
        self._learning_rate.setSingleStep(1e-5)
        self._learning_rate.setCorrectionMode(
            QDoubleSpinBox.CorrectionMode.CorrectToNearestValue
        )

        self._training_batch_size = QSpinBox()
        self._training_batch_size.setRange(1, 1_000_000)

        self._validation_batch_size = QSpinBox()
        self._validation_batch_size.setRange(1, 1_000_000)

        self._patches_per_epoch = QSpinBox()
        self._patches_per_epoch.setRange(1, 1_000_000_000)

        self._early_stopping_patience = QSpinBox()
        self._early_stopping_patience.setRange(1, 1_000_000)

        form.addRow("Learning rate", self._learning_rate)
        form.addRow("Training batch size", self._training_batch_size)
        form.addRow("Validation batch size", self._validation_batch_size)
        form.addRow("Patches per epoch", self._patches_per_epoch)
        form.addRow("Early stopping patience", self._early_stopping_patience)
        layout.addLayout(form)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Reset
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        reset_button = self._buttons.button(QDialogButtonBox.StandardButton.Reset)
        if reset_button is not None:
            reset_button.clicked.connect(self.reset_to_defaults)
        layout.addWidget(self._buttons)

        self.set_parameters(self._parameters)

    def set_parameters(self, parameters: TrainingParameters) -> None:
        normalized = validate_training_parameters(parameters)
        self._learning_rate.setValue(float(normalized.learning_rate))
        self._training_batch_size.setValue(int(normalized.training_batch_size))
        self._validation_batch_size.setValue(int(normalized.validation_batch_size))
        self._patches_per_epoch.setValue(int(normalized.patches_per_epoch))
        self._early_stopping_patience.setValue(int(normalized.early_stopping_patience))

    def reset_to_defaults(self) -> None:
        self.set_parameters(DEFAULT_TRAINING_PARAMETERS)

    def parameters(self) -> TrainingParameters:
        return validate_training_parameters(
            TrainingParameters(
                learning_rate=float(self._learning_rate.value()),
                training_batch_size=int(self._training_batch_size.value()),
                validation_batch_size=int(self._validation_batch_size.value()),
                patches_per_epoch=int(self._patches_per_epoch.value()),
                early_stopping_patience=int(self._early_stopping_patience.value()),
            )
        )


def open_training_parameters_dialog(
    parameters: TrainingParameters,
    parent: Optional[QWidget] = None,
) -> TrainingParametersDialogResult:
    dialog = TrainingParametersDialog(parameters, parent=parent)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return TrainingParametersDialogResult(accepted=False, parameters=None)
    return TrainingParametersDialogResult(
        accepted=True,
        parameters=dialog.parameters(),
    )


def ask_unsaved_changes(
    parent: Optional[QWidget] = None,
    *,
    context: Optional[str] = None,
    subject: str = "segmentation",
) -> UnsavedChangesDecision:
    normalized_subject = str(subject).strip()
    if not normalized_subject:
        normalized_subject = "segmentation"
    title_subject = normalized_subject.title()
    dialog = QMessageBox(parent)
    dialog.setIcon(QMessageBox.Icon.Warning)
    dialog.setWindowTitle(f"Unsaved {title_subject}")
    dialog.setText(f"The current {normalized_subject} contains unsaved manual changes.")
    if context:
        dialog.setInformativeText(f"Do you want to save before {context}?")
    else:
        dialog.setInformativeText("Do you want to save before continuing?")

    save_button = dialog.addButton("Save...", QMessageBox.ButtonRole.AcceptRole)
    discard_button = dialog.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
    cancel_button = dialog.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
    dialog.setDefaultButton(save_button)
    dialog.exec()

    clicked = dialog.clickedButton()
    if clicked is save_button:
        return UnsavedChangesDecision.SAVE
    if clicked is discard_button:
        return UnsavedChangesDecision.DISCARD
    if clicked is cancel_button:
        return UnsavedChangesDecision.CANCEL
    return UnsavedChangesDecision.CANCEL


def ask_training_running_close_decision(
    parent: Optional[QWidget] = None,
) -> TrainingCloseDecision:
    dialog = QMessageBox(parent)
    dialog.setIcon(QMessageBox.Icon.Warning)
    dialog.setWindowTitle("Training In Progress")
    dialog.setText("A model training is currently running.")
    dialog.setInformativeText("Choose what to do before closing this window.")

    stop_button = dialog.addButton(
        "Stop training and close",
        QMessageBox.ButtonRole.DestructiveRole,
    )
    continue_button = dialog.addButton(
        "Continue in background",
        QMessageBox.ButtonRole.AcceptRole,
    )
    cancel_button = dialog.addButton(
        "Cancel",
        QMessageBox.ButtonRole.RejectRole,
    )
    dialog.setDefaultButton(cancel_button)
    dialog.exec()

    clicked = dialog.clickedButton()
    if clicked is stop_button:
        return TrainingCloseDecision.STOP_AND_CLOSE
    if clicked is continue_button:
        return TrainingCloseDecision.CONTINUE_IN_BACKGROUND
    if clicked is cancel_button:
        return TrainingCloseDecision.CANCEL
    return TrainingCloseDecision.CANCEL


def ask_inference_running_close_decision(
    parent: Optional[QWidget] = None,
) -> InferenceCloseDecision:
    dialog = QMessageBox(parent)
    dialog.setIcon(QMessageBox.Icon.Warning)
    dialog.setWindowTitle("Inference In Progress")
    dialog.setText("A segmentation inference is currently running.")
    dialog.setInformativeText("Choose what to do before closing this window.")

    stop_button = dialog.addButton(
        "Stop inference and close",
        QMessageBox.ButtonRole.DestructiveRole,
    )
    continue_button = dialog.addButton(
        "Continue in background",
        QMessageBox.ButtonRole.AcceptRole,
    )
    cancel_button = dialog.addButton(
        "Cancel",
        QMessageBox.ButtonRole.RejectRole,
    )
    dialog.setDefaultButton(cancel_button)
    dialog.exec()

    clicked = dialog.clickedButton()
    if clicked is stop_button:
        return InferenceCloseDecision.STOP_AND_CLOSE
    if clicked is continue_button:
        return InferenceCloseDecision.CONTINUE_IN_BACKGROUND
    if clicked is cancel_button:
        return InferenceCloseDecision.CANCEL
    return InferenceCloseDecision.CANCEL


def _normalize_volume_path(path: str) -> str:
    resolved = Path(path).expanduser()
    if resolved.suffix.lower() == ".zarr":
        return str(resolved)
    for parent in (resolved,) + tuple(resolved.parents):
        if parent.suffix.lower() == ".zarr":
            return str(parent)
    return str(resolved)


def open_file_dialog(parent: Optional[QWidget] = None) -> DialogResult:
    filters = "Volume files (*.npy *.npz *.tif *.tiff *.h5 *.hdf5 *.hdf *.zarr);;All files (*)"
    dialog = QFileDialog(parent, "Open Volume")
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
    dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
    dialog.setNameFilter(filters)

    def _accept_zarr_directory(path: str) -> None:
        normalized = _normalize_volume_path(path)
        candidate = Path(normalized)
        if candidate.suffix.lower() == ".zarr" and candidate.exists():
            dialog.selectFile(str(candidate))
            dialog.accept()

    dialog.directoryEntered.connect(_accept_zarr_directory)

    if dialog.exec() != QFileDialog.DialogCode.Accepted:
        return DialogResult(accepted=False, path=None)
    selected = dialog.selectedFiles()
    candidate = selected[0] if selected else dialog.directory().absolutePath()
    normalized = _normalize_volume_path(candidate)
    if not Path(normalized).exists():
        return DialogResult(accepted=False, path=None)
    return DialogResult(accepted=True, path=normalized)


def open_bounding_boxes_dialog(parent: Optional[QWidget] = None) -> DialogResult:
    filters = "Bounding Box files (*.bbox.txt *.txt);;All files (*)"
    path, _selected_filter = QFileDialog.getOpenFileName(
        parent,
        "Open Bounding Boxes",
        "",
        filters,
    )
    if not path:
        return DialogResult(accepted=False, path=None)
    normalized = str(Path(path).expanduser())
    if not Path(normalized).exists():
        return DialogResult(accepted=False, path=None)
    return DialogResult(accepted=True, path=normalized)


def open_model_checkpoint_dialog(parent: Optional[QWidget] = None) -> DialogResult:
    filters = "Model checkpoint (*.cp)"
    path, _selected_filter = QFileDialog.getOpenFileName(
        parent,
        "Load Model Checkpoint",
        "",
        filters,
    )
    if not path:
        return DialogResult(accepted=False, path=None)
    normalized = str(Path(path).expanduser())
    resolved = Path(normalized)
    if not resolved.exists() or not resolved.is_file():
        return DialogResult(accepted=False, path=None)
    if resolved.suffix.lower() != ".cp":
        return DialogResult(accepted=False, path=None)
    return DialogResult(accepted=True, path=normalized)


def open_save_model_checkpoint_dialog(
    parent: Optional[QWidget] = None,
    *,
    retry_on_overwrite_decline: bool = False,
) -> DialogResult:
    filters = "Model checkpoint (*.cp)"
    while True:
        path, _selected_filter = QFileDialog.getSaveFileName(
            parent,
            "Save Model Checkpoint",
            "",
            filters,
        )
        if not path:
            return DialogResult(accepted=False, path=None)

        normalized = str(Path(path).expanduser())
        resolved = Path(normalized)
        if resolved.suffix.lower() != ".cp":
            normalized = f"{normalized}.cp"
            resolved = Path(normalized)

        parent_directory = resolved.parent
        if not parent_directory.exists() or not parent_directory.is_dir():
            return DialogResult(accepted=False, path=None)
        if resolved.exists():
            if not resolved.is_file():
                return DialogResult(accepted=False, path=None)
            if not confirm_overwrite(str(resolved), parent=parent):
                if retry_on_overwrite_decline:
                    continue
                return DialogResult(accepted=False, path=None)

        return DialogResult(accepted=True, path=str(resolved))


def open_save_bounding_boxes_dialog(parent: Optional[QWidget] = None) -> DialogResult:
    filters = "Bounding Box files (*.bbox.txt *.txt);;All files (*)"
    path, _selected_filter = QFileDialog.getSaveFileName(
        parent,
        "Save Bounding Boxes",
        "",
        filters,
    )
    if not path:
        return DialogResult(accepted=False, path=None)
    normalized = _normalize_bounding_boxes_save_path(path)
    return DialogResult(accepted=True, path=normalized)


def confirm_replace_bounding_boxes(parent: Optional[QWidget] = None) -> bool:
    answer = QMessageBox.question(
        parent,
        "Replace Bounding Boxes",
        (
            "Current bounding boxes will be replaced by the file contents.\n\n"
            "Do you want to continue?"
        ),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return answer == QMessageBox.StandardButton.Yes


def confirm_replace_inference_bboxes(parent: Optional[QWidget] = None) -> bool:
    answer = QMessageBox.question(
        parent,
        "Replace Inference BBoxes",
        (
            "At least one inference bounding box is not empty.\n\n"
            "Running inference will replace segmentation content inside all inference bboxes.\n\n"
            "Do you want to continue?"
        ),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return answer == QMessageBox.StandardButton.Yes


def open_save_segmentation_dialog(parent: Optional[QWidget] = None) -> SaveDialogResult:
    filters = (
        "TIFF (*.tif *.tiff);;"
        "NumPy (*.npy);;"
        "NumPy Compressed (*.npz);;"
        "HDF5 (*.h5 *.hdf5 *.hdf);;"
        "Zarr (*.zarr)"
    )
    path, selected_filter = QFileDialog.getSaveFileName(
        parent,
        "Save Segmentation",
        "",
        filters,
    )
    if not path:
        return SaveDialogResult(accepted=False)
    normalized_path, save_format = _normalize_save_path(path, selected_filter)
    return SaveDialogResult(accepted=True, path=normalized_path, format=save_format)


def _normalize_save_path(path: str, selected_filter: str) -> Tuple[str, str]:
    resolved = str(Path(path).expanduser())
    suffix = Path(resolved).suffix.lower()
    save_format = _format_from_suffix(suffix)
    if save_format is None:
        save_format = _format_from_filter(selected_filter)
        extension = _default_extension_for_format(save_format)
        if extension:
            resolved = resolved + extension
    return resolved, save_format


def _normalize_bounding_boxes_save_path(path: str) -> str:
    resolved = str(Path(path).expanduser())
    suffix = Path(resolved).suffix.lower()
    if suffix:
        return resolved
    return resolved + ".bbox.txt"


def _format_from_suffix(suffix: str) -> Optional[str]:
    if suffix in {".tif", ".tiff"}:
        return "tiff"
    if suffix == ".npy":
        return "npy"
    if suffix == ".npz":
        return "npz"
    if suffix in {".h5", ".hdf5", ".hdf"}:
        return "hdf5"
    if suffix == ".zarr":
        return "zarr"
    return None


def _format_from_filter(selected_filter: str) -> str:
    if selected_filter.startswith("TIFF"):
        return "tiff"
    if selected_filter.startswith("NumPy Compressed"):
        return "npz"
    if selected_filter.startswith("NumPy"):
        return "npy"
    if selected_filter.startswith("HDF5"):
        return "hdf5"
    if selected_filter.startswith("Zarr"):
        return "zarr"
    return "npy"


def _default_extension_for_format(save_format: str) -> str:
    if save_format == "tiff":
        return ".tif"
    if save_format == "npy":
        return ".npy"
    if save_format == "npz":
        return ".npz"
    if save_format == "hdf5":
        return ".h5"
    if save_format == "zarr":
        return ".zarr"
    return ""
