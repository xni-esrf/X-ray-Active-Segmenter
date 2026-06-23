from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

from .label_utils import coerce_label_values, unique_non_mask_values
from .label_space import LearningLabelSpace
from .session_store import (
    LearningBBoxDataLoaderRuntime,
    LearningBBoxEvalRuntime,
    LearningSession,
    get_current_learning_dataloader_runtime,
    get_current_learning_eval_runtimes_by_box_id,
    get_current_learning_label_space,
)


@dataclass(frozen=True)
class LearningLabelCoverage:
    label_values: Tuple[int, ...]
    train_present_label_values: Tuple[int, ...]
    validation_present_label_values: Tuple[int, ...]
    missing_train_label_values: Tuple[int, ...]
    missing_validation_label_values: Tuple[int, ...]

    @property
    def has_missing_labels(self) -> bool:
        return bool(self.missing_train_label_values or self.missing_validation_label_values)


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
            "Training dataset does not expose 'annot_tensors'; cannot inspect label coverage."
        )
    raw_tensors = getattr(dataset, "annot_tensors")
    if not isinstance(raw_tensors, Sequence):
        raise TypeError("Training dataset annot_tensors must be a sequence.")
    tensors = tuple(raw_tensors)
    if not tensors:
        raise ValueError("Training dataset annot_tensors is empty.")
    return tensors


def _extract_validation_ground_truths(
    eval_runtimes_by_box_id: Mapping[str, LearningBBoxEvalRuntime],
) -> Tuple[object, ...]:
    if not isinstance(eval_runtimes_by_box_id, Mapping):
        raise TypeError(
            "eval_runtimes_by_box_id must be a mapping of box_id -> LearningBBoxEvalRuntime, "
            f"got {type(eval_runtimes_by_box_id).__name__}"
        )
    if not eval_runtimes_by_box_id:
        raise ValueError("No evaluation runtimes/buffers are available in session storage.")

    ground_truths = []
    for box_id, runtime in tuple(eval_runtimes_by_box_id.items()):
        if not isinstance(runtime, LearningBBoxEvalRuntime):
            raise TypeError(
                "eval_runtimes_by_box_id values must be LearningBBoxEvalRuntime, "
                f"got {type(runtime).__name__} for id={box_id!r}"
            )
        buffer_obj = runtime.buffer
        if not hasattr(buffer_obj, "ground_truth"):
            raise ValueError(
                f"Evaluation buffer for box_id={box_id!r} does not expose ground_truth."
            )
        ground_truths.append(getattr(buffer_obj, "ground_truth"))
    return tuple(ground_truths)


def _resolve_label_space(
    *,
    learning_session: Optional[LearningSession],
    label_space: Optional[LearningLabelSpace],
) -> LearningLabelSpace:
    resolved = label_space
    if resolved is None:
        resolved = (
            learning_session.get_label_space()
            if learning_session is not None
            else get_current_learning_label_space()
        )
    if not isinstance(resolved, LearningLabelSpace):
        raise ValueError("No current learning label space is available.")
    return resolved


def _resolve_train_runtime(
    *,
    learning_session: Optional[LearningSession],
    train_runtime: Optional[LearningBBoxDataLoaderRuntime],
) -> LearningBBoxDataLoaderRuntime:
    resolved = train_runtime
    if resolved is None:
        resolved = (
            learning_session.get_dataloader_runtime()
            if learning_session is not None
            else get_current_learning_dataloader_runtime()
        )
    if not isinstance(resolved, LearningBBoxDataLoaderRuntime):
        raise ValueError("No training dataloader runtime is available in session storage.")
    return resolved


def _resolve_eval_runtimes(
    *,
    learning_session: Optional[LearningSession],
    eval_runtimes_by_box_id: Optional[Mapping[str, LearningBBoxEvalRuntime]],
) -> Mapping[str, LearningBBoxEvalRuntime]:
    if eval_runtimes_by_box_id is not None:
        return dict(eval_runtimes_by_box_id)
    return (
        learning_session.get_eval_runtimes_by_box_id()
        if learning_session is not None
        else get_current_learning_eval_runtimes_by_box_id()
    )


def _present_label_values(
    tensors: Sequence[object],
    *,
    mask_label: int,
    torch_module: Optional[object],
) -> Tuple[int, ...]:
    present = set()
    for tensor in tuple(tensors):
        present.update(
            unique_non_mask_values(
                tensor,
                mask_label=mask_label,
                torch_module=torch_module,
            )
        )
    return tuple(sorted(int(value) for value in present))


def compute_learning_label_coverage(
    *,
    train_runtime: Optional[LearningBBoxDataLoaderRuntime] = None,
    eval_runtimes_by_box_id: Optional[Mapping[str, LearningBBoxEvalRuntime]] = None,
    label_space: Optional[LearningLabelSpace] = None,
    learning_session: Optional[LearningSession] = None,
    torch_module: Optional[object] = None,
) -> LearningLabelCoverage:
    resolved_label_space = _resolve_label_space(
        learning_session=learning_session,
        label_space=label_space,
    )
    label_values = coerce_label_values(
        resolved_label_space.label_values,
        name="label_space.label_values",
    )
    allowed_values = set(label_values)
    mask_label = int(resolved_label_space.mask_label)

    resolved_train_runtime = _resolve_train_runtime(
        learning_session=learning_session,
        train_runtime=train_runtime,
    )
    resolved_eval_runtimes = _resolve_eval_runtimes(
        learning_session=learning_session,
        eval_runtimes_by_box_id=eval_runtimes_by_box_id,
    )

    train_present = _present_label_values(
        _extract_train_segmentation_tensors(resolved_train_runtime),
        mask_label=mask_label,
        torch_module=torch_module,
    )
    validation_present = _present_label_values(
        _extract_validation_ground_truths(resolved_eval_runtimes),
        mask_label=mask_label,
        torch_module=torch_module,
    )

    unexpected_train = tuple(value for value in train_present if value not in allowed_values)
    unexpected_validation = tuple(
        value for value in validation_present if value not in allowed_values
    )
    if unexpected_train:
        raise ValueError(
            "Training segmentation contains label values outside the current "
            f"label space: {unexpected_train}"
        )
    if unexpected_validation:
        raise ValueError(
            "Validation segmentation contains label values outside the current "
            f"label space: {unexpected_validation}"
        )

    return LearningLabelCoverage(
        label_values=tuple(label_values),
        train_present_label_values=tuple(train_present),
        validation_present_label_values=tuple(validation_present),
        missing_train_label_values=tuple(
            value for value in label_values if value not in set(train_present)
        ),
        missing_validation_label_values=tuple(
            value for value in label_values if value not in set(validation_present)
        ),
    )


def format_learning_label_coverage_warning(
    coverage: LearningLabelCoverage,
) -> Optional[str]:
    if not isinstance(coverage, LearningLabelCoverage):
        raise TypeError(
            "coverage must be a LearningLabelCoverage, "
            f"got {type(coverage).__name__}"
        )
    if not coverage.has_missing_labels:
        return None

    lines = [
        "Some labels from the current label space are missing from the prepared learning data."
    ]
    if coverage.missing_train_label_values:
        lines.append(
            "- Missing from train boxes: "
            + ", ".join(str(value) for value in coverage.missing_train_label_values)
        )
    if coverage.missing_validation_label_values:
        lines.append(
            "- Missing from validation boxes: "
            + ", ".join(str(value) for value in coverage.missing_validation_label_values)
        )
    lines.append(
        "These labels remain model output classes. Missing train labels use the maximum loss weight."
    )
    return "\n".join(lines)
