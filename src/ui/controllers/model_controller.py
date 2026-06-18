from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Callable, Mapping, Optional, Protocol, Tuple

from ...learning import (
    get_current_learning_model_runtime,
    instantiate_foundation_model_runtime,
    save_foundation_model_checkpoint,
    validate_foundation_checkpoint_load_preconditions,
    validate_foundation_model_instantiation_preconditions,
)
from ...utils import exception_message
from ..dialogs import (
    confirm_reinitialize_model,
    confirm_replace_training_model_with_default_checkpoint,
    open_model_checkpoint_dialog,
    open_save_model_checkpoint_dialog,
    show_info,
    show_warning,
)


class ModelControllerContext(Protocol):
    def _persist_model_runtime_label_values_from_eval_runtimes(
        self,
        runtime: object,
        *,
        eval_runtimes_by_box_id: Mapping[str, object],
    ) -> None:
        ...

    def _reinitialize_training_runtime_from_default_checkpoint(self) -> bool:
        ...

    def _runtime_requires_training_reinitialization(self, runtime: object) -> bool:
        ...

    def _save_model_runtime_checkpoint(
        self,
        runtime: object,
        *,
        checkpoint_path: str,
    ) -> None:
        ...


@dataclass(frozen=True)
class ModelControllerOperations:
    show_warning: Callable[..., object] = show_warning
    show_info: Callable[..., object] = show_info
    open_model_checkpoint_dialog: Callable[..., object] = open_model_checkpoint_dialog
    open_save_model_checkpoint_dialog: Callable[..., object] = (
        open_save_model_checkpoint_dialog
    )
    confirm_reinitialize_model: Callable[..., bool] = confirm_reinitialize_model
    confirm_replace_training_model_with_default_checkpoint: Callable[..., bool] = (
        confirm_replace_training_model_with_default_checkpoint
    )
    validate_foundation_checkpoint_load_preconditions: Callable[..., object] = (
        validate_foundation_checkpoint_load_preconditions
    )
    validate_foundation_model_instantiation_preconditions: Callable[..., object] = (
        validate_foundation_model_instantiation_preconditions
    )
    instantiate_foundation_model_runtime: Callable[..., object] = (
        instantiate_foundation_model_runtime
    )
    save_foundation_model_checkpoint: Callable[..., object] = (
        save_foundation_model_checkpoint
    )
    get_learning_model_runtime: Callable[[object], object] = (
        lambda _context: get_current_learning_model_runtime()
    )
    learning_session_kwargs: Callable[[object], dict[str, object]] = lambda _context: {}
    exception_message: Callable[[Exception], str] = exception_message
    normalize_checkpoint_identity: Callable[[object], Optional[str]] = lambda _path: None
    resolve_shared_eval_label_values: Callable[
        [Mapping[str, object]],
        Tuple[int, ...],
    ] = lambda _eval_runtimes_by_box_id: tuple()
    resolve_inference_label_values_for_runtime: Callable[[object], Tuple[int, ...]] = (
        lambda _runtime: tuple()
    )
    default_training_checkpoint_path: str = "foundation_model/MAE_XNT.cp"


@dataclass
class ModelController:
    context: ModelControllerContext
    operations: ModelControllerOperations

    def save_model_with_dialog(self) -> bool:
        runtime = self.operations.get_learning_model_runtime(self.context)
        if runtime is None:
            self.operations.show_warning(
                "Load a model before saving.",
                parent=self.context,
            )
            return False

        dialog_result = self.operations.open_save_model_checkpoint_dialog(self.context)
        if not dialog_result.accepted or not dialog_result.path:
            return False

        checkpoint_path = str(Path(dialog_result.path).expanduser())
        if Path(checkpoint_path).suffix.lower() != ".cp":
            self.operations.show_warning(
                "Model checkpoints must use the .cp extension.",
                parent=self.context,
            )
            return False

        try:
            save_model_runtime_checkpoint = getattr(
                self.context,
                "_save_model_runtime_checkpoint",
                None,
            )
            if callable(save_model_runtime_checkpoint):
                save_model_runtime_checkpoint(runtime, checkpoint_path=checkpoint_path)
            else:
                self.save_model_runtime_checkpoint(
                    runtime,
                    checkpoint_path=checkpoint_path,
                )
        except Exception as exc:
            self.operations.show_warning(
                self.operations.exception_message(exc),
                parent=self.context,
            )
            return False

        self.operations.show_info(
            (
                "Model checkpoint saved.\n"
                f"- checkpoint: {checkpoint_path}\n"
                f"- num_classes: {runtime.num_classes}\n"
                f"- device_ids: {runtime.device_ids}"
            ),
            parent=self.context,
        )
        return True

    def save_model_runtime_checkpoint(
        self,
        runtime: object,
        *,
        checkpoint_path: str,
    ) -> None:
        self.operations.save_foundation_model_checkpoint(
            runtime=runtime,
            checkpoint_path=checkpoint_path,
        )

    def runtime_training_provenance(
        self,
        runtime: object,
    ) -> Tuple[Optional[str], bool, int]:
        source_checkpoint_path = getattr(runtime, "checkpoint_path", None)
        trained_in_app = False
        training_run_count = 0

        hyperparameters_obj = getattr(runtime, "hyperparameters", None)
        if isinstance(hyperparameters_obj, Mapping):
            raw_source = hyperparameters_obj.get("source_checkpoint_path")
            if isinstance(raw_source, str) and raw_source.strip():
                source_checkpoint_path = raw_source.strip()

            raw_trained = hyperparameters_obj.get("trained_in_app")
            if isinstance(raw_trained, bool):
                trained_in_app = bool(raw_trained)

            raw_run_count = hyperparameters_obj.get("training_run_count")
            if isinstance(raw_run_count, Integral) and not isinstance(raw_run_count, bool):
                if int(raw_run_count) >= 0:
                    training_run_count = int(raw_run_count)

        if training_run_count > 0 and not trained_in_app:
            trained_in_app = True
        if not isinstance(source_checkpoint_path, str) or not source_checkpoint_path.strip():
            source_checkpoint_path = None

        return source_checkpoint_path, bool(trained_in_app), int(training_run_count)

    def runtime_requires_training_reinitialization(self, runtime: object) -> bool:
        source_checkpoint_path, trained_in_app, training_run_count = (
            self.runtime_training_provenance(runtime)
        )
        if trained_in_app or training_run_count > 0:
            return True
        default_identity = self.operations.normalize_checkpoint_identity(
            self.operations.default_training_checkpoint_path
        )
        source_identity = self.operations.normalize_checkpoint_identity(source_checkpoint_path)
        return bool(default_identity is None or source_identity != default_identity)

    def reinitialize_training_runtime_from_default_checkpoint(self) -> bool:
        checkpoint_path = self.operations.default_training_checkpoint_path
        try:
            preconditions = (
                self.operations.validate_foundation_model_instantiation_preconditions(
                    require_min_gpu_count=2,
                    **self.operations.learning_session_kwargs(self.context),
                )
            )
            runtime = self.operations.instantiate_foundation_model_runtime(
                num_classes=preconditions.num_classes,
                device_ids=preconditions.device_ids,
                checkpoint_path=checkpoint_path,
                **self.operations.learning_session_kwargs(self.context),
            )
            eval_runtimes_by_box_id = getattr(preconditions, "eval_runtimes_by_box_id", None)
            if isinstance(eval_runtimes_by_box_id, Mapping):
                persist_label_values = getattr(
                    self.context,
                    "_persist_model_runtime_label_values_from_eval_runtimes",
                    None,
                )
                if callable(persist_label_values):
                    persist_label_values(
                        runtime,
                        eval_runtimes_by_box_id=eval_runtimes_by_box_id,
                    )
                else:
                    self.persist_model_runtime_label_values_from_eval_runtimes(
                        runtime,
                        eval_runtimes_by_box_id=eval_runtimes_by_box_id,
                    )
        except Exception as exc:
            message = self.operations.exception_message(exc)
            self.operations.show_warning(
                (
                    f"{message}\n\n"
                    "Training requires the default foundation checkpoint:\n"
                    f"{checkpoint_path}"
                ),
                parent=self.context,
            )
            return False
        return True

    def ensure_training_runtime_for_new_training(self) -> bool:
        runtime = self.operations.get_learning_model_runtime(self.context)
        if runtime is None:
            reinitialize = getattr(
                self.context,
                "_reinitialize_training_runtime_from_default_checkpoint",
                None,
            )
            if callable(reinitialize):
                return bool(reinitialize())
            return self.reinitialize_training_runtime_from_default_checkpoint()

        requires_reinitialization = getattr(
            self.context,
            "_runtime_requires_training_reinitialization",
            None,
        )
        if callable(requires_reinitialization):
            should_reinitialize = bool(requires_reinitialization(runtime))
        else:
            should_reinitialize = self.runtime_requires_training_reinitialization(runtime)
        if not should_reinitialize:
            return True

        if not self.operations.confirm_replace_training_model_with_default_checkpoint(
            checkpoint_path=self.operations.default_training_checkpoint_path,
            parent=self.context,
        ):
            return False

        reinitialize = getattr(
            self.context,
            "_reinitialize_training_runtime_from_default_checkpoint",
            None,
        )
        if callable(reinitialize):
            return bool(reinitialize())
        return self.reinitialize_training_runtime_from_default_checkpoint()

    def mark_current_model_runtime_as_trained(self, *, completed_epoch_count: int) -> None:
        if int(completed_epoch_count) <= 0:
            return
        runtime = self.operations.get_learning_model_runtime(self.context)
        if runtime is None:
            return
        hyperparameters_obj = getattr(runtime, "hyperparameters", None)
        if not isinstance(hyperparameters_obj, dict):
            return
        previous_count = hyperparameters_obj.get("training_run_count", 0)
        try:
            normalized_count = int(previous_count)
        except Exception:
            normalized_count = 0
        if normalized_count < 0:
            normalized_count = 0
        hyperparameters_obj["trained_in_app"] = True
        hyperparameters_obj["training_run_count"] = normalized_count + 1
        if "source_checkpoint_path" not in hyperparameters_obj:
            checkpoint_path_obj = getattr(runtime, "checkpoint_path", None)
            if isinstance(checkpoint_path_obj, str) and checkpoint_path_obj.strip():
                hyperparameters_obj["source_checkpoint_path"] = checkpoint_path_obj

    def persist_model_runtime_label_values_from_eval_runtimes(
        self,
        runtime: object,
        *,
        eval_runtimes_by_box_id: Mapping[str, object],
    ) -> None:
        if runtime is None:
            return
        hyperparameters_obj = getattr(runtime, "hyperparameters", None)
        if not isinstance(hyperparameters_obj, dict):
            return
        resolved_label_values = self.operations.resolve_shared_eval_label_values(
            eval_runtimes_by_box_id
        )
        hyperparameters_obj["label_values"] = tuple(
            int(value) for value in tuple(resolved_label_values)
        )

    def instantiate_foundation_model_with_dialog(self) -> bool:
        try:
            dialog_result = self.operations.open_model_checkpoint_dialog(self.context)
            if not dialog_result.accepted or not dialog_result.path:
                return False
            checkpoint_path = str(Path(dialog_result.path).expanduser())
            if Path(checkpoint_path).suffix.lower() != ".cp":
                self.operations.show_warning(
                    "Model checkpoints must use the .cp extension.",
                    parent=self.context,
                )
                return False

            preconditions = (
                self.operations.validate_foundation_checkpoint_load_preconditions(
                    checkpoint_path,
                    require_min_gpu_count=2,
                )
            )
            checkpoint_path = str(preconditions.checkpoint_path)
            existing_runtime = self.operations.get_learning_model_runtime(self.context)
            if existing_runtime is not None:
                if not self.operations.confirm_reinitialize_model(parent=self.context):
                    return False

            runtime = self.operations.instantiate_foundation_model_runtime(
                num_classes=preconditions.num_classes,
                device_ids=preconditions.device_ids,
                checkpoint_path=checkpoint_path,
                **self.operations.learning_session_kwargs(self.context),
            )
            hyperparameters_obj = getattr(runtime, "hyperparameters", None)
            if not isinstance(hyperparameters_obj, dict):
                raise ValueError("Loaded model runtime does not expose mutable hyperparameters.")
            hyperparameters_obj["label_values"] = tuple(
                int(value) for value in tuple(preconditions.label_values)
            )
            resolved_label_values = self.operations.resolve_inference_label_values_for_runtime(
                runtime
            )
            if tuple(resolved_label_values) != tuple(preconditions.label_values):
                raise ValueError(
                    "Loaded model runtime label_values do not match checkpoint metadata."
                )
        except Exception as exc:
            self.operations.show_warning(
                self.operations.exception_message(exc),
                parent=self.context,
            )
            return False

        self.operations.show_info(
            (
                "Foundation model loaded from checkpoint.\n"
                f"- checkpoint: {runtime.checkpoint_path}\n"
                f"- num_classes: {runtime.num_classes}\n"
                f"- device_ids: {runtime.device_ids}\n"
                f"- label_values: {tuple(preconditions.label_values)}"
            ),
            parent=self.context,
        )
        return True

