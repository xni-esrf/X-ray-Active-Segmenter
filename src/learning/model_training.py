from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from .session_store import (
    LearningBBoxDataLoaderRuntime,
    LearningBBoxEvalRuntime,
    LearningModelRuntime,
    get_current_learning_dataloader_runtime,
    get_current_learning_eval_runtimes_by_box_id,
    get_current_learning_model_runtime,
)


_MASK_LABEL = -100


@dataclass(frozen=True)
class LearningTrainingPreconditions:
    model_runtime: LearningModelRuntime
    train_runtime: LearningBBoxDataLoaderRuntime
    eval_runtimes_by_box_id: Dict[str, LearningBBoxEvalRuntime]
    class_weights: object
    validation_valid_voxel_counts_by_box_id: Dict[str, int]
    total_validation_valid_voxel_count: int


@dataclass(frozen=True)
class LearningTrainEpochResult:
    epoch_index: int
    total_epoch_count: int
    base_learning_rate: float
    num_batches: int
    mean_loss: float
    mixed_precision_used: bool


@dataclass(frozen=True)
class LearningValidationEvalResult:
    weighted_mean_accuracy: float
    per_box_accuracy_by_box_id: Dict[str, float]
    valid_voxel_counts_by_box_id: Dict[str, int]
    total_valid_voxel_count: int
    mixed_precision_used: bool


@dataclass(frozen=True)
class LearningTrainingLoopResult:
    completed_epoch_count: int
    total_epoch_count: int
    stop_reason: str
    best_epoch_index: Optional[int]
    best_weighted_mean_accuracy: Optional[float]
    early_stop_patience: int
    mixed_precision_enabled: bool


class _LearningTrainingStopRequested(RuntimeError):
    """Raised internally to stop training/eval at the next batch boundary."""


def _require_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _require_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    return int(value)


def _require_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    return float(value)


def _coerce_accuracy_value(value: object) -> float:
    torch_mod = _resolve_torch(None)
    if torch_mod is not None and isinstance(value, getattr(torch_mod, "Tensor")):
        if int(value.numel()) != 1:
            raise ValueError("accuracy tensor must contain a single scalar value")
        return float(value.detach().item())
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"accuracy must be a scalar real value, got {type(value).__name__}")
    return float(value)


def _require_callable(value: object, *, name: str):
    if not callable(value):
        raise TypeError(f"{name} must be callable")
    return value


def _is_stop_requested(stop_event: Optional[object]) -> bool:
    if stop_event is None:
        return False
    is_set = getattr(stop_event, "is_set", None)
    if not callable(is_set):
        raise TypeError("stop_event must expose a callable is_set() when provided")
    return bool(is_set())


def _resolve_torch(torch_module: Optional[object] = None):
    if torch_module is not None:
        return torch_module
    try:
        import torch
    except Exception:  # pragma: no cover - import availability is environment dependent
        return None
    return torch


def _count_valid_voxels(
    ground_truth: object,
    *,
    mask_label: int,
    torch_module: Optional[object] = None,
) -> int:
    torch_mod = _resolve_torch(torch_module)
    if torch_mod is not None and isinstance(ground_truth, getattr(torch_mod, "Tensor")):
        valid_mask = ground_truth != int(mask_label)
        return int(valid_mask.sum().item())

    ground_truth_array = np.asarray(ground_truth)
    if int(ground_truth_array.size) <= 0:
        return 0
    return int(np.count_nonzero(ground_truth_array != int(mask_label)))


def _format_missing_preconditions_message(missing_items) -> str:
    lines = [
        "Cannot train model because required learning state is missing:",
    ]
    for item in tuple(missing_items):
        lines.append(f"- {item}")
    return "\n".join(lines)


def _resolve_model_runtime(
    model_runtime: Optional[LearningModelRuntime],
) -> LearningModelRuntime:
    resolved = get_current_learning_model_runtime() if model_runtime is None else model_runtime
    if resolved is None:
        raise ValueError("No learning model runtime is available in session storage.")
    if not isinstance(resolved, LearningModelRuntime):
        raise TypeError(
            "model_runtime must be a LearningModelRuntime, "
            f"got {type(resolved).__name__}"
        )
    return resolved


def _resolve_train_runtime(
    train_runtime: Optional[LearningBBoxDataLoaderRuntime],
) -> LearningBBoxDataLoaderRuntime:
    resolved = get_current_learning_dataloader_runtime() if train_runtime is None else train_runtime
    if resolved is None:
        raise ValueError("No training dataloader runtime is available in session storage.")
    if not isinstance(resolved, LearningBBoxDataLoaderRuntime):
        raise TypeError(
            "train_runtime must be a LearningBBoxDataLoaderRuntime, "
            f"got {type(resolved).__name__}"
        )
    return resolved


def _resolve_training_device(
    torch_mod,
    *,
    model_runtime: LearningModelRuntime,
    device: Optional[str],
):
    if device is not None:
        normalized_device = str(device).strip()
        if not normalized_device:
            raise ValueError("device must be a non-empty string when provided")
        return getattr(torch_mod, "device")(normalized_device)

    if not model_runtime.device_ids:
        raise ValueError("model_runtime.device_ids must contain at least one CUDA id")
    first_device_id = int(model_runtime.device_ids[0])
    cuda = getattr(torch_mod, "cuda", None)
    if cuda is None:
        raise RuntimeError("CUDA runtime is required to resolve model base device.")
    is_available = bool(getattr(cuda, "is_available", lambda: False)())
    device_count = int(getattr(cuda, "device_count", lambda: 0)())
    if not is_available or device_count <= first_device_id:
        raise RuntimeError(
            f"CUDA device cuda:{first_device_id} is required for training, but unavailable."
        )
    return getattr(torch_mod, "device")(f"cuda:{first_device_id}")


def _resolve_training_hyperparameters(
    model_runtime: LearningModelRuntime,
    optimizer: object,
    *,
    initial_lr: Optional[float],
    lwise_lr_decay: Optional[float],
) -> Tuple[float, float]:
    hyperparameters = dict(model_runtime.hyperparameters)

    if initial_lr is None:
        if "lr" in hyperparameters:
            resolved_initial_lr = _require_real(hyperparameters["lr"], name="initial_lr")
        else:
            param_groups = getattr(optimizer, "param_groups", None)
            if not isinstance(param_groups, list) and not isinstance(param_groups, tuple):
                raise ValueError(
                    "Cannot resolve initial_lr from optimizer.param_groups; provide initial_lr explicitly."
                )
            if not param_groups:
                raise ValueError(
                    "Cannot resolve initial_lr from optimizer.param_groups; no param groups are defined."
                )
            first_group = param_groups[0]
            if not isinstance(first_group, Mapping) or "lr" not in first_group:
                raise ValueError(
                    "Cannot resolve initial_lr from optimizer.param_groups[0]['lr']; provide initial_lr explicitly."
                )
            resolved_initial_lr = _require_real(first_group["lr"], name="initial_lr")
    else:
        resolved_initial_lr = _require_real(initial_lr, name="initial_lr")

    if resolved_initial_lr <= 0.0:
        raise ValueError(f"initial_lr must be > 0, got {resolved_initial_lr}")

    if lwise_lr_decay is None:
        if "lwise_lr_decay" in hyperparameters:
            resolved_lwise = _require_real(
                hyperparameters["lwise_lr_decay"],
                name="lwise_lr_decay",
            )
        else:
            resolved_lwise = 1.0
    else:
        resolved_lwise = _require_real(lwise_lr_decay, name="lwise_lr_decay")
    if resolved_lwise <= 0.0:
        raise ValueError(f"lwise_lr_decay must be > 0, got {resolved_lwise}")

    return float(resolved_initial_lr), float(resolved_lwise)


def _compute_epoch_base_learning_rate(
    *,
    initial_lr: float,
    epoch_index: int,
    total_epoch_count: int,
) -> float:
    return float(initial_lr) / 10.0 + (
        float(initial_lr) - float(initial_lr) / 10.0
    ) * 0.5 * (1.0 + math.cos(math.pi * float(epoch_index) / float(total_epoch_count + 1)))


def _apply_optimizer_learning_rates(
    optimizer: object,
    *,
    base_lr: float,
    lwise_lr_decay: float,
) -> None:
    param_groups = getattr(optimizer, "param_groups", None)
    if not isinstance(param_groups, list) and not isinstance(param_groups, tuple):
        raise TypeError("optimizer must expose a sequence param_groups attribute")
    for group in tuple(param_groups):
        if not isinstance(group, dict):
            raise TypeError("optimizer.param_groups must contain dictionaries")
        if float(lwise_lr_decay) != 1.0:
            decay_rate = _require_real(group.get("lwise_lr_decay_rate", 1.0), name="lwise_lr_decay_rate")
            group["lr"] = float(base_lr) * float(decay_rate)
        else:
            group["lr"] = float(base_lr)


def _clear_validation_buffer(buffer: object) -> None:
    for method_name in ("clear", "reset", "reset_buffer", "clear_buffer"):
        method = getattr(buffer, method_name, None)
        if callable(method):
            method()
            return

    if not hasattr(buffer, "buffer_vol"):
        raise ValueError(
            "Evaluation buffer cannot be cleared: no supported clear/reset method or buffer_vol attribute."
        )
    buffer_vol = getattr(buffer, "buffer_vol")
    zero_ = getattr(buffer_vol, "zero_", None)
    if callable(zero_):
        zero_()
        return
    fill = getattr(buffer_vol, "fill", None)
    if callable(fill):
        fill(0)
        return
    raise ValueError(
        "Evaluation buffer cannot be cleared: buffer_vol does not support zero_() or fill(0)."
    )


def _clone_model_state_dict_to_cpu(
    model: object,
    *,
    torch_module: Optional[object] = None,
) -> Dict[str, object]:
    state_dict_method = _require_callable(getattr(model, "state_dict", None), name="model.state_dict")
    state_dict = state_dict_method()
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"model.state_dict() must return a mapping, got {type(state_dict).__name__}")

    torch_mod = _resolve_torch(torch_module)
    normalized_state: Dict[str, object] = {}
    for raw_key, raw_value in tuple(state_dict.items()):
        key = str(raw_key)
        if torch_mod is not None and isinstance(raw_value, getattr(torch_mod, "Tensor")):
            normalized_state[key] = raw_value.detach().cpu().clone()
            continue
        normalized_state[key] = copy.deepcopy(raw_value)
    return normalized_state


def _restore_model_state_dict(
    model: object,
    state_dict_cpu: Mapping[str, object],
) -> None:
    load_state_dict = _require_callable(getattr(model, "load_state_dict", None), name="model.load_state_dict")
    load_state_dict(dict(state_dict_cpu))


def _resolve_validation_valid_voxel_counts(
    eval_runtimes_by_box_id: Mapping[str, LearningBBoxEvalRuntime],
    *,
    mask_label: int,
    torch_module: Optional[object] = None,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for box_id, runtime in tuple(eval_runtimes_by_box_id.items()):
        if not isinstance(runtime, LearningBBoxEvalRuntime):
            raise TypeError(
                "eval_runtimes_by_box_id values must be LearningBBoxEvalRuntime, "
                f"got {type(runtime).__name__} for box_id={box_id!r}"
            )
        buffer_obj = runtime.buffer
        if not hasattr(buffer_obj, "ground_truth"):
            raise ValueError(
                f"Evaluation buffer for box_id={box_id!r} does not expose 'ground_truth'."
            )
        ground_truth = getattr(buffer_obj, "ground_truth")
        valid_voxel_count = _count_valid_voxels(
            ground_truth,
            mask_label=mask_label,
            torch_module=torch_module,
        )
        if int(valid_voxel_count) <= 0:
            raise ValueError(
                "Validation buffer has no valid voxels "
                f"(ground_truth != {mask_label}) for box_id={box_id!r}."
            )
        counts[str(box_id)] = int(valid_voxel_count)
    return counts


def validate_learning_model_training_preconditions(
    *,
    model_runtime: Optional[LearningModelRuntime] = None,
    train_runtime: Optional[LearningBBoxDataLoaderRuntime] = None,
    eval_runtimes_by_box_id: Optional[Mapping[str, LearningBBoxEvalRuntime]] = None,
    require_class_weights: bool = True,
    mask_label: int = _MASK_LABEL,
    torch_module: Optional[object] = None,
) -> LearningTrainingPreconditions:
    normalized_require_class_weights = _require_bool(
        require_class_weights,
        name="require_class_weights",
    )
    normalized_mask_label = _require_int(mask_label, name="mask_label")

    resolved_model_runtime = (
        get_current_learning_model_runtime() if model_runtime is None else model_runtime
    )
    resolved_train_runtime = (
        get_current_learning_dataloader_runtime() if train_runtime is None else train_runtime
    )
    resolved_eval_runtimes = (
        get_current_learning_eval_runtimes_by_box_id()
        if eval_runtimes_by_box_id is None
        else dict(eval_runtimes_by_box_id)
    )

    if (
        resolved_model_runtime is not None
        and not isinstance(resolved_model_runtime, LearningModelRuntime)
    ):
        raise TypeError(
            "model_runtime must be a LearningModelRuntime, "
            f"got {type(resolved_model_runtime).__name__}"
        )
    if (
        resolved_train_runtime is not None
        and not isinstance(resolved_train_runtime, LearningBBoxDataLoaderRuntime)
    ):
        raise TypeError(
            "train_runtime must be a LearningBBoxDataLoaderRuntime, "
            f"got {type(resolved_train_runtime).__name__}"
        )

    missing_items = []
    if resolved_model_runtime is None:
        missing_items.append("model runtime (Load Model).")
    if resolved_train_runtime is None:
        missing_items.append("training dataloader runtime (Build Dataset from Bbox).")
    if not resolved_eval_runtimes:
        missing_items.append("evaluation runtimes/buffers (Build Dataset from Bbox).")
    if (
        resolved_train_runtime is not None
        and normalized_require_class_weights
        and getattr(resolved_train_runtime, "class_weights", None) is None
    ):
        missing_items.append("class weights on training dataloader runtime (Build Dataset from Bbox).")
    if missing_items:
        raise ValueError(_format_missing_preconditions_message(missing_items))

    class_weights = getattr(resolved_train_runtime, "class_weights", None)
    if normalized_require_class_weights and class_weights is None:
        raise ValueError(
            "Cannot train model because required learning state is missing:\n"
            "- class weights on training dataloader runtime (Build Dataset from Bbox)."
        )

    valid_voxel_counts_by_box_id = _resolve_validation_valid_voxel_counts(
        resolved_eval_runtimes,
        mask_label=normalized_mask_label,
        torch_module=torch_module,
    )
    total_valid_voxel_count = int(sum(valid_voxel_counts_by_box_id.values()))
    if total_valid_voxel_count <= 0:
        raise ValueError(
            "Validation buffers contain no valid voxels (ground_truth != mask label)."
        )

    return LearningTrainingPreconditions(
        model_runtime=resolved_model_runtime,
        train_runtime=resolved_train_runtime,
        eval_runtimes_by_box_id=dict(resolved_eval_runtimes),
        class_weights=class_weights,
        validation_valid_voxel_counts_by_box_id=dict(valid_voxel_counts_by_box_id),
        total_validation_valid_voxel_count=total_valid_voxel_count,
    )


def train_learning_model_for_one_epoch(
    *,
    epoch_index: int,
    total_epoch_count: int,
    model_runtime: Optional[LearningModelRuntime] = None,
    train_runtime: Optional[LearningBBoxDataLoaderRuntime] = None,
    initial_lr: Optional[float] = None,
    lwise_lr_decay: Optional[float] = None,
    mixed_precision: bool = True,
    label_smoothing: float = 0.025,
    ignore_index: int = _MASK_LABEL,
    scaler: Optional[object] = None,
    device: Optional[str] = None,
    stop_event: Optional[object] = None,
    torch_module: Optional[object] = None,
) -> Tuple[LearningTrainEpochResult, object]:
    torch_mod = _resolve_torch(torch_module)
    if torch_mod is None:  # pragma: no cover - environment dependent
        raise ImportError("PyTorch is required to train the learning model.")

    normalized_epoch_index = _require_int(epoch_index, name="epoch_index")
    normalized_total_epoch_count = _require_int(total_epoch_count, name="total_epoch_count")
    if normalized_epoch_index < 0:
        raise ValueError(f"epoch_index must be >= 0, got {normalized_epoch_index}")
    if normalized_total_epoch_count <= 0:
        raise ValueError(f"total_epoch_count must be >= 1, got {normalized_total_epoch_count}")
    if normalized_epoch_index >= normalized_total_epoch_count:
        raise ValueError(
            "epoch_index must be < total_epoch_count, "
            f"got epoch_index={normalized_epoch_index} total_epoch_count={normalized_total_epoch_count}"
        )
    normalized_mixed_precision = _require_bool(mixed_precision, name="mixed_precision")
    normalized_label_smoothing = _require_real(label_smoothing, name="label_smoothing")
    if normalized_label_smoothing < 0.0 or normalized_label_smoothing > 1.0:
        raise ValueError(
            f"label_smoothing must be in [0, 1], got {normalized_label_smoothing}"
        )
    normalized_ignore_index = _require_int(ignore_index, name="ignore_index")

    resolved_model_runtime = _resolve_model_runtime(model_runtime)
    resolved_train_runtime = _resolve_train_runtime(train_runtime)
    class_weights = getattr(resolved_train_runtime, "class_weights", None)
    if class_weights is None:
        raise ValueError(
            "Training dataloader runtime is missing class_weights. "
            "Build Dataset from Bbox first."
        )
    if not isinstance(class_weights, getattr(torch_mod, "Tensor")):
        raise TypeError(
            "train_runtime.class_weights must be a torch.Tensor, "
            f"got {type(class_weights).__name__}"
        )
    if int(class_weights.ndim) != 1:
        raise ValueError(
            f"train_runtime.class_weights must be a 1D tensor, got ndim={int(class_weights.ndim)}"
        )
    if int(class_weights.numel()) <= 0:
        raise ValueError("train_runtime.class_weights must contain at least one value")

    model = resolved_model_runtime.model
    optimizer = resolved_model_runtime.optimizer
    dataloader = resolved_train_runtime.dataloader

    resolved_initial_lr, resolved_lwise_lr_decay = _resolve_training_hyperparameters(
        resolved_model_runtime,
        optimizer,
        initial_lr=initial_lr,
        lwise_lr_decay=lwise_lr_decay,
    )
    base_lr = _compute_epoch_base_learning_rate(
        initial_lr=resolved_initial_lr,
        epoch_index=normalized_epoch_index,
        total_epoch_count=normalized_total_epoch_count,
    )
    _apply_optimizer_learning_rates(
        optimizer,
        base_lr=base_lr,
        lwise_lr_decay=resolved_lwise_lr_decay,
    )

    resolved_device = _resolve_training_device(
        torch_mod,
        model_runtime=resolved_model_runtime,
        device=device,
    )
    class_weights_on_device = class_weights.to(
        device=resolved_device,
        dtype=getattr(torch_mod, "float32"),
    )

    loss_func = getattr(getattr(torch_mod, "nn"), "CrossEntropyLoss")(
        weight=class_weights_on_device,
        ignore_index=normalized_ignore_index,
        label_smoothing=normalized_label_smoothing,
    )

    train_method = getattr(model, "train", None)
    if callable(train_method):
        train_method()

    if scaler is None:
        scaler = getattr(getattr(torch_mod, "amp"), "GradScaler")(enabled=normalized_mixed_precision)

    resolved_device_type = str(getattr(resolved_device, "type", "cpu"))
    autocast_enabled = bool(normalized_mixed_precision and resolved_device_type == "cuda")

    num_batches = 0
    cumulative_loss = 0.0
    for minivols, annotations in dataloader:
        if _is_stop_requested(stop_event):
            raise _LearningTrainingStopRequested("Training stop requested by user.")

        minivols = minivols.to(resolved_device)
        annotations = annotations.to(
            device=resolved_device,
            dtype=getattr(torch_mod, "long"),
        )

        zero_grad = getattr(optimizer, "zero_grad", None)
        if not callable(zero_grad):
            raise TypeError("optimizer must define a callable zero_grad()")
        zero_grad()

        with getattr(torch_mod, "autocast")(
            device_type=resolved_device_type,
            enabled=autocast_enabled,
            dtype=getattr(torch_mod, "bfloat16"),
        ):
            dec_output = model(minivols)
            cur_loss = loss_func(input=dec_output, target=annotations)

        scale_method = getattr(scaler, "scale", None)
        if not callable(scale_method):
            raise TypeError("scaler must define a callable scale(loss)")
        scaled = scale_method(cur_loss)
        backward = getattr(scaled, "backward", None)
        if not callable(backward):
            raise TypeError("scaler.scale(loss) result must define backward()")
        backward()

        step_method = getattr(scaler, "step", None)
        if not callable(step_method):
            raise TypeError("scaler must define a callable step(optimizer)")
        step_method(optimizer)

        update_method = getattr(scaler, "update", None)
        if not callable(update_method):
            raise TypeError("scaler must define a callable update()")
        update_method()

        cumulative_loss += float(cur_loss.detach().item())
        num_batches += 1

    if num_batches <= 0:
        if _is_stop_requested(stop_event):
            raise _LearningTrainingStopRequested("Training stop requested by user.")
        raise ValueError("Training dataloader produced zero batches for the epoch.")

    result = LearningTrainEpochResult(
        epoch_index=normalized_epoch_index,
        total_epoch_count=normalized_total_epoch_count,
        base_learning_rate=float(base_lr),
        num_batches=int(num_batches),
        mean_loss=float(cumulative_loss / float(num_batches)),
        mixed_precision_used=bool(autocast_enabled),
    )
    return result, scaler


def evaluate_learning_model_on_validation_dataloaders(
    *,
    preconditions: Optional[LearningTrainingPreconditions] = None,
    model_runtime: Optional[LearningModelRuntime] = None,
    eval_runtimes_by_box_id: Optional[Mapping[str, LearningBBoxEvalRuntime]] = None,
    valid_voxel_counts_by_box_id: Optional[Mapping[str, object]] = None,
    mixed_precision: bool = True,
    device: Optional[str] = None,
    stop_event: Optional[object] = None,
    torch_module: Optional[object] = None,
) -> LearningValidationEvalResult:
    torch_mod = _resolve_torch(torch_module)
    if torch_mod is None:  # pragma: no cover - environment dependent
        raise ImportError("PyTorch is required to evaluate the learning model.")
    normalized_mixed_precision = _require_bool(mixed_precision, name="mixed_precision")

    if preconditions is not None:
        if not isinstance(preconditions, LearningTrainingPreconditions):
            raise TypeError(
                "preconditions must be a LearningTrainingPreconditions, "
                f"got {type(preconditions).__name__}"
            )
        resolved_model_runtime = preconditions.model_runtime
        resolved_eval_runtimes = dict(preconditions.eval_runtimes_by_box_id)
        resolved_valid_counts = dict(preconditions.validation_valid_voxel_counts_by_box_id)
    else:
        resolved_model_runtime = _resolve_model_runtime(model_runtime)
        resolved_eval_runtimes = (
            get_current_learning_eval_runtimes_by_box_id()
            if eval_runtimes_by_box_id is None
            else dict(eval_runtimes_by_box_id)
        )
        if valid_voxel_counts_by_box_id is None:
            resolved_valid_counts = _resolve_validation_valid_voxel_counts(
                resolved_eval_runtimes,
                mask_label=_MASK_LABEL,
                torch_module=torch_mod,
            )
        else:
            resolved_valid_counts = {}
            for raw_box_id, raw_count in tuple(valid_voxel_counts_by_box_id.items()):
                box_id = str(raw_box_id)
                resolved_count = _require_int(raw_count, name=f"valid_voxel_counts_by_box_id[{box_id!r}]")
                if resolved_count <= 0:
                    raise ValueError(
                        f"valid voxel count must be > 0 for box_id={box_id!r}, got {resolved_count}"
                    )
                resolved_valid_counts[box_id] = resolved_count

    if not resolved_eval_runtimes:
        raise ValueError("No evaluation runtimes/buffers are available in session storage.")
    for box_id, runtime in tuple(resolved_eval_runtimes.items()):
        if not isinstance(runtime, LearningBBoxEvalRuntime):
            raise TypeError(
                "eval_runtimes_by_box_id values must be LearningBBoxEvalRuntime, "
                f"got {type(runtime).__name__} for box_id={box_id!r}"
            )
        if box_id not in resolved_valid_counts:
            raise ValueError(
                f"Missing valid voxel count for evaluation runtime box_id={box_id!r}."
            )

    total_valid_voxel_count = int(sum(int(v) for v in tuple(resolved_valid_counts.values())))
    if total_valid_voxel_count <= 0:
        raise ValueError("Validation weighted accuracy requires a positive total valid voxel count.")

    resolved_device = _resolve_training_device(
        torch_mod,
        model_runtime=resolved_model_runtime,
        device=device,
    )
    resolved_device_type = str(getattr(resolved_device, "type", "cpu"))
    autocast_enabled = bool(normalized_mixed_precision and resolved_device_type == "cuda")

    model = resolved_model_runtime.model
    was_training = bool(getattr(model, "training", False))
    eval_method = getattr(model, "eval", None)
    if callable(eval_method):
        eval_method()

    weighted_accuracy_sum = 0.0
    per_box_accuracy: Dict[str, float] = {}
    with getattr(torch_mod, "no_grad")():
        for box_id, runtime in tuple(resolved_eval_runtimes.items()):
            if _is_stop_requested(stop_event):
                raise _LearningTrainingStopRequested("Training stop requested by user.")
            _clear_validation_buffer(runtime.buffer)
            dataloader = runtime.dataloader
            add_batch = getattr(runtime.buffer, "add_batch", None)
            if not callable(add_batch):
                raise TypeError(
                    f"Evaluation buffer for box_id={box_id!r} must define add_batch(batch, coordinates)."
                )
            get_acc_pred = getattr(runtime.buffer, "get_acc_pred", None)
            if not callable(get_acc_pred):
                raise TypeError(
                    f"Evaluation buffer for box_id={box_id!r} must define get_acc_pred()."
                )

            for minivols, coordinates in dataloader:
                if _is_stop_requested(stop_event):
                    raise _LearningTrainingStopRequested("Training stop requested by user.")
                minivols = minivols.to(resolved_device)
                with getattr(torch_mod, "autocast")(
                    device_type=resolved_device_type,
                    enabled=autocast_enabled,
                    dtype=getattr(torch_mod, "bfloat16"),
                ):
                    pred_minivols = model(minivols)
                add_batch(pred_minivols.detach().cpu(), coordinates)

            accuracy_value = _coerce_accuracy_value(get_acc_pred())
            per_box_accuracy[box_id] = float(accuracy_value)
            weighted_accuracy_sum += float(accuracy_value) * float(resolved_valid_counts[box_id])

    if was_training:
        train_method = getattr(model, "train", None)
        if callable(train_method):
            train_method()

    return LearningValidationEvalResult(
        weighted_mean_accuracy=float(weighted_accuracy_sum / float(total_valid_voxel_count)),
        per_box_accuracy_by_box_id=dict(per_box_accuracy),
        valid_voxel_counts_by_box_id={box_id: int(count) for box_id, count in tuple(resolved_valid_counts.items())},
        total_valid_voxel_count=int(total_valid_voxel_count),
        mixed_precision_used=bool(autocast_enabled),
    )


def train_learning_model_with_validation_loop(
    *,
    preconditions: Optional[LearningTrainingPreconditions] = None,
    model_runtime: Optional[LearningModelRuntime] = None,
    train_runtime: Optional[LearningBBoxDataLoaderRuntime] = None,
    eval_runtimes_by_box_id: Optional[Mapping[str, LearningBBoxEvalRuntime]] = None,
    mixed_precision: bool = True,
    early_stop_patience: int = 2,
    total_epoch_count: Optional[int] = None,
    label_smoothing: float = 0.025,
    ignore_index: int = _MASK_LABEL,
    device: Optional[str] = None,
    stop_event: Optional[object] = None,
    torch_module: Optional[object] = None,
) -> LearningTrainingLoopResult:
    torch_mod = _resolve_torch(torch_module)
    if torch_mod is None:  # pragma: no cover - environment dependent
        raise ImportError("PyTorch is required to run the learning training loop.")

    normalized_mixed_precision = _require_bool(mixed_precision, name="mixed_precision")
    normalized_patience = _require_int(early_stop_patience, name="early_stop_patience")
    if normalized_patience <= 0:
        raise ValueError(f"early_stop_patience must be >= 1, got {normalized_patience}")

    if preconditions is None:
        preconditions = validate_learning_model_training_preconditions(
            model_runtime=model_runtime,
            train_runtime=train_runtime,
            eval_runtimes_by_box_id=eval_runtimes_by_box_id,
            require_class_weights=True,
            mask_label=_MASK_LABEL,
            torch_module=torch_mod,
        )
    elif not isinstance(preconditions, LearningTrainingPreconditions):
        raise TypeError(
            "preconditions must be a LearningTrainingPreconditions, "
            f"got {type(preconditions).__name__}"
        )

    resolved_total_epoch_count: int
    if total_epoch_count is None:
        resolved_total_epoch_count = int(2 * preconditions.train_runtime.train_count)
    else:
        resolved_total_epoch_count = _require_int(total_epoch_count, name="total_epoch_count")
    if resolved_total_epoch_count <= 0:
        raise ValueError(
            "total_epoch_count must be >= 1. "
            f"Resolved value: {resolved_total_epoch_count}."
        )

    model = preconditions.model_runtime.model
    scaler = None
    best_epoch_index = -1
    best_weighted_mean_accuracy = float("-inf")
    epochs_without_improvement = 0
    completed_epoch_count = 0
    stop_reason = "max_epoch"
    best_model_state_dict_cpu: Optional[Dict[str, object]] = None

    try:
        for epoch_index in range(resolved_total_epoch_count):
            if _is_stop_requested(stop_event):
                stop_reason = "user_stop"
                break

            train_result, scaler = train_learning_model_for_one_epoch(
                epoch_index=epoch_index,
                total_epoch_count=resolved_total_epoch_count,
                model_runtime=preconditions.model_runtime,
                train_runtime=preconditions.train_runtime,
                mixed_precision=normalized_mixed_precision,
                label_smoothing=label_smoothing,
                ignore_index=ignore_index,
                scaler=scaler,
                device=device,
                stop_event=stop_event,
                torch_module=torch_mod,
            )
            if not math.isfinite(float(train_result.mean_loss)):
                raise ValueError(
                    "Training loss must remain finite. "
                    f"Epoch {epoch_index} produced mean_loss={train_result.mean_loss!r}."
                )

            if _is_stop_requested(stop_event):
                stop_reason = "user_stop"
                break

            eval_result = evaluate_learning_model_on_validation_dataloaders(
                preconditions=preconditions,
                mixed_precision=normalized_mixed_precision,
                device=device,
                stop_event=stop_event,
                torch_module=torch_mod,
            )
            weighted_mean_accuracy = float(eval_result.weighted_mean_accuracy)
            if not math.isfinite(weighted_mean_accuracy):
                raise ValueError(
                    "Validation weighted mean accuracy must be finite. "
                    f"Epoch {epoch_index} produced weighted_mean_accuracy={weighted_mean_accuracy!r}."
                )

            if weighted_mean_accuracy > best_weighted_mean_accuracy:
                best_weighted_mean_accuracy = weighted_mean_accuracy
                best_epoch_index = int(epoch_index)
                best_model_state_dict_cpu = _clone_model_state_dict_to_cpu(
                    model,
                    torch_module=torch_mod,
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            completed_epoch_count += 1
            if epochs_without_improvement >= normalized_patience:
                stop_reason = "early_stop"
                break
    except _LearningTrainingStopRequested:
        stop_reason = "user_stop"
    except Exception:
        if best_model_state_dict_cpu is not None:
            try:
                _restore_model_state_dict(model, best_model_state_dict_cpu)
            except Exception:
                pass
        raise

    if best_model_state_dict_cpu is None or best_epoch_index < 0:
        if stop_reason == "user_stop":
            return LearningTrainingLoopResult(
                completed_epoch_count=int(completed_epoch_count),
                total_epoch_count=int(resolved_total_epoch_count),
                stop_reason=str(stop_reason),
                best_epoch_index=None,
                best_weighted_mean_accuracy=None,
                early_stop_patience=int(normalized_patience),
                mixed_precision_enabled=bool(normalized_mixed_precision),
            )
        raise RuntimeError(
            "Training loop did not produce any valid checkpointable epoch state."
        )
    _restore_model_state_dict(model, best_model_state_dict_cpu)

    return LearningTrainingLoopResult(
        completed_epoch_count=int(completed_epoch_count),
        total_epoch_count=int(resolved_total_epoch_count),
        stop_reason=str(stop_reason),
        best_epoch_index=int(best_epoch_index),
        best_weighted_mean_accuracy=float(best_weighted_mean_accuracy),
        early_stop_patience=int(normalized_patience),
        mixed_precision_enabled=bool(normalized_mixed_precision),
    )
