from __future__ import annotations

from numbers import Integral, Real
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .session_store import (
    LearningBBoxDataLoaderRuntime,
    LearningBBoxEvalRuntime,
    get_current_learning_dataloader_runtime,
    get_current_learning_eval_runtimes_by_box_id,
    set_current_learning_dataloader_class_weights,
)


_MASK_LABEL = -100


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("PyTorch is required to compute learning class weights") from exc
    return torch


def _coerce_positive_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    normalized = float(value)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be > 0, got {normalized}")
    return normalized


def _coerce_label_values(label_values: Sequence[object]) -> Tuple[int, ...]:
    if not isinstance(label_values, Sequence):
        raise TypeError(f"label_values must be a sequence, got {type(label_values).__name__}")
    normalized = []
    for raw_value in tuple(label_values):
        if isinstance(raw_value, bool) or not isinstance(raw_value, Integral):
            raise TypeError(
                "label_values must contain integers only, "
                f"got {type(raw_value).__name__}"
            )
        integer = int(raw_value)
        if integer == _MASK_LABEL:
            raise ValueError("label_values must not include -100 (reserved mask label)")
        if integer in normalized:
            raise ValueError(f"label_values must not contain duplicates, got {integer}")
        normalized.append(integer)
    if not normalized:
        raise ValueError("label_values must contain at least one class label")
    return tuple(normalized)


def _resolve_target_device(torch_mod, *, device: str):
    normalized = str(device).strip()
    if not normalized:
        raise ValueError("device must be a non-empty string")
    if normalized.startswith("cuda"):
        cuda = getattr(torch_mod, "cuda", None)
        if cuda is None:
            raise RuntimeError(
                f"CUDA device {normalized} is required to compute class weights, but CUDA is unavailable."
            )
        is_available_fn = getattr(cuda, "is_available", None)
        device_count_fn = getattr(cuda, "device_count", None)
        available = bool(is_available_fn()) if callable(is_available_fn) else False
        device_count = int(device_count_fn()) if callable(device_count_fn) else 0
        if not available or device_count <= 0:
            raise RuntimeError(
                f"CUDA device {normalized} is required to compute class weights, but CUDA is unavailable."
            )
        if normalized == "cuda:0" and device_count < 1:
            raise RuntimeError("CUDA device cuda:0 is required to compute class weights.")
    return getattr(torch_mod, "device")(normalized)


def _resolve_shared_eval_label_values(
    eval_runtimes_by_box_id: Mapping[str, LearningBBoxEvalRuntime],
) -> Tuple[int, ...]:
    if not isinstance(eval_runtimes_by_box_id, Mapping):
        raise TypeError(
            "eval_runtimes_by_box_id must be a mapping of box_id -> LearningBBoxEvalRuntime, "
            f"got {type(eval_runtimes_by_box_id).__name__}"
        )
    if not eval_runtimes_by_box_id:
        raise ValueError("No evaluation runtimes/buffers are available in session storage.")

    resolved_label_values: Optional[Tuple[int, ...]] = None
    for box_id, runtime in tuple(eval_runtimes_by_box_id.items()):
        if not isinstance(runtime, LearningBBoxEvalRuntime):
            raise TypeError(
                "eval_runtimes_by_box_id values must be LearningBBoxEvalRuntime, "
                f"got {type(runtime).__name__} for id={box_id!r}"
            )
        buffer_obj = runtime.buffer
        if not hasattr(buffer_obj, "label_values"):
            raise ValueError(
                f"Evaluation buffer for box_id={box_id!r} does not expose 'label_values'."
            )
        label_values = _coerce_label_values(getattr(buffer_obj, "label_values"))
        if resolved_label_values is None:
            resolved_label_values = label_values
            continue
        if label_values != resolved_label_values:
            raise ValueError(
                "All evaluation buffers must share the same label_values ordering; "
                f"expected {resolved_label_values}, got {label_values} for box_id={box_id!r}."
            )
    if resolved_label_values is None:
        raise ValueError("No evaluation buffer label_values could be resolved.")
    return resolved_label_values


def _extract_train_segmentation_tensors(
    train_runtime: LearningBBoxDataLoaderRuntime,
) -> Tuple[object, ...]:
    if not isinstance(train_runtime, LearningBBoxDataLoaderRuntime):
        raise TypeError(
            "train_runtime must be a LearningBBoxDataLoaderRuntime, "
            f"got {type(train_runtime).__name__}"
        )
    dataset = train_runtime.dataset
    if not hasattr(dataset, "annot_tensors"):
        raise ValueError(
            "Training dataset does not expose 'annot_tensors'; cannot compute class weights."
        )
    raw_tensors = getattr(dataset, "annot_tensors")
    if not isinstance(raw_tensors, Sequence):
        raise TypeError(
            "Training dataset annot_tensors must be a sequence of torch tensors."
        )
    tensors = tuple(raw_tensors)
    if not tensors:
        raise ValueError("Training dataset annot_tensors is empty; cannot compute class weights.")
    return tensors


def compute_class_weights_from_segmentation_tensors(
    segmentation_tensors: Sequence[object],
    *,
    label_values: Sequence[object],
    max_weight: float = 100.0,
    device: str = "cuda:0",
    torch_module: Optional[object] = None,
):
    torch_mod = _require_torch() if torch_module is None else torch_module
    resolved_max_weight = _coerce_positive_real(max_weight, name="max_weight")
    resolved_label_values = _coerce_label_values(label_values)
    target_device = _resolve_target_device(torch_mod, device=device)

    if not isinstance(segmentation_tensors, Sequence):
        raise TypeError(
            "segmentation_tensors must be a sequence of torch tensors."
        )
    normalized_tensors = tuple(segmentation_tensors)
    if not normalized_tensors:
        raise ValueError("segmentation_tensors must contain at least one tensor.")

    label_counts: Dict[int, int] = {label: 0 for label in resolved_label_values}
    for tensor in normalized_tensors:
        if not isinstance(tensor, torch_mod.Tensor):
            raise TypeError(
                "segmentation_tensors must contain torch.Tensor instances only, "
                f"got {type(tensor).__name__}"
            )
        unique_values, unique_counts = torch_mod.unique(
            tensor.to(dtype=torch_mod.long),
            return_counts=True,
        )
        for raw_value, raw_count in zip(
            unique_values.tolist(),
            unique_counts.tolist(),
        ):
            value = int(raw_value)
            if value == _MASK_LABEL:
                continue
            if value not in label_counts:
                raise ValueError(
                    "Training segmentation contains label value not present in eval label_values: "
                    f"{value}"
                )
            label_counts[value] += int(raw_count)

    positive_counts = [count for count in tuple(label_counts.values()) if count > 0]
    if not positive_counts:
        raise ValueError("No non-masked train voxels were found to compute class weights.")
    max_count = int(max(positive_counts))

    weights = []
    for label in resolved_label_values:
        count = int(label_counts[label])
        if count <= 0:
            weight = float(resolved_max_weight)
        else:
            weight = float(max_count) / float(count)
            if weight > resolved_max_weight:
                weight = float(resolved_max_weight)
        weights.append(float(weight))

    return torch_mod.tensor(weights, dtype=torch_mod.float32, device=target_device)


def compute_and_store_current_learning_class_weights(
    *,
    max_weight: float = 100.0,
    device: str = "cuda:0",
    torch_module: Optional[object] = None,
):
    train_runtime = get_current_learning_dataloader_runtime()
    if train_runtime is None:
        raise ValueError("No training dataloader runtime is available in session storage.")
    eval_runtimes_by_box_id = get_current_learning_eval_runtimes_by_box_id()
    label_values = _resolve_shared_eval_label_values(eval_runtimes_by_box_id)
    train_segmentation_tensors = _extract_train_segmentation_tensors(train_runtime)
    class_weights = compute_class_weights_from_segmentation_tensors(
        train_segmentation_tensors,
        label_values=label_values,
        max_weight=max_weight,
        device=device,
        torch_module=torch_module,
    )
    updated_runtime = set_current_learning_dataloader_class_weights(class_weights)
    return updated_runtime.class_weights

