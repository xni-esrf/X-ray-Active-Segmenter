from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Protocol, Tuple

from ...io import extract_learning_bboxes_in_memory
from ...learning import (
    DEFAULT_TRAINING_PARAMETERS,
    clear_current_learning_bbox_batch,
    compute_and_store_current_learning_class_weights,
    compute_learning_label_coverage,
    derive_label_space_from_semantic_segmentation,
    format_learning_label_coverage_warning,
    get_current_learning_bbox_batch,
    get_current_learning_dataloader_runtime,
    get_current_learning_eval_runtimes_by_box_id,
    prepare_learning_state_from_volumes,
    semantic_label_space_source_signature,
    set_current_learning_label_space,
    validate_training_parameters,
)
from ..dialogs import show_info, show_warning


LearningStateAction = str


class LearningStateContext(Protocol):
    """Host contract for learning-state preparation.

    This intentionally captures only the attributes currently needed by the
    learning-state workflow, so the controller does not import MainWindow.
    """

    state: object
    bottom_panel: object
    _raw_volume: object
    _bbox_manager: object
    _segmentation_editor: object
    _learning_state_signature: Optional[Tuple[object, ...]]
    _learning_state_stale: bool

    def _active_segmentation_volume(self) -> Optional[Tuple[str, object]]:
        ...


@dataclass(frozen=True)
class LearningStateControllerOperations:
    show_warning: Callable[..., object] = show_warning
    show_info: Callable[..., object] = show_info
    prepare_learning_state_from_volumes: Callable[..., object] = (
        prepare_learning_state_from_volumes
    )
    extract_learning_bboxes_in_memory: Callable[..., object] = (
        extract_learning_bboxes_in_memory
    )
    compute_and_store_current_learning_class_weights: Callable[..., object] = (
        compute_and_store_current_learning_class_weights
    )
    compute_learning_label_coverage: Callable[..., object] = (
        compute_learning_label_coverage
    )
    format_learning_label_coverage_warning: Callable[[object], Optional[str]] = (
        format_learning_label_coverage_warning
    )
    get_learning_dataloader_runtime: Callable[[object], object] = (
        lambda _context: get_current_learning_dataloader_runtime()
    )
    get_learning_eval_runtimes_by_box_id: Callable[[object], Mapping[str, object]] = (
        lambda _context: get_current_learning_eval_runtimes_by_box_id()
    )
    clear_learning_bbox_batch: Callable[[object], None] = (
        lambda _context: clear_current_learning_bbox_batch()
    )
    get_learning_bbox_batch: Callable[[object], object] = (
        lambda _context: get_current_learning_bbox_batch()
    )
    set_learning_label_space: Callable[[object, object], object] = (
        lambda _context, label_space: set_current_learning_label_space(label_space)
    )
    derive_label_space_from_semantic_segmentation: Callable[..., object] = (
        derive_label_space_from_semantic_segmentation
    )
    learning_session_kwargs: Callable[[object], Dict[str, object]] = lambda _context: {}
    format_class_weights_for_summary: Callable[[object], Optional[str]] = (
        lambda _class_weights: None
    )


@dataclass
class LearningStateController:
    context: LearningStateContext
    operations: LearningStateControllerOperations

    def training_parameters(self):
        parameters = getattr(
            self.context,
            "_training_parameters",
            DEFAULT_TRAINING_PARAMETERS,
        )
        return validate_training_parameters(parameters)

    def semantic_label_space_source_signature(
        self,
        *,
        semantic_kind: str,
        semantic_volume: object,
    ) -> Tuple[object, ...]:
        semantic_state_id: Optional[int] = None
        editor = getattr(self.context, "_segmentation_editor", None)
        if editor is not None:
            semantic_state_id = int(editor.state_id)
        return semantic_label_space_source_signature(
            semantic_kind=semantic_kind,
            semantic_volume=semantic_volume,
            semantic_state_id=semantic_state_id,
        )

    def ensure_for_action(self, action: LearningStateAction) -> bool:
        """Ensure learning datasets/runtimes are available for a learning action."""

        normalized_action = str(action).strip().lower()
        if normalized_action not in {"load_model", "train", "inference"}:
            raise ValueError(
                "action must be one of: 'load_model', 'train', or 'inference', "
                f"got {action!r}"
            )
        if normalized_action == "inference":
            # Inference builds per-inference-box runtimes directly and should not
            # trigger train/validation learning-state rebuilds.
            return True

        require_class_weights = normalized_action == "train"
        train_runtime = self.operations.get_learning_dataloader_runtime(self.context)
        eval_runtimes_by_box_id = self.operations.get_learning_eval_runtimes_by_box_id(
            self.context
        )
        missing_learning_state = bool(train_runtime is None or not eval_runtimes_by_box_id)
        if (
            not missing_learning_state
            and require_class_weights
            and getattr(train_runtime, "class_weights", None) is None
        ):
            missing_learning_state = True

        current_signature_getter = getattr(
            self.context,
            "_current_learning_state_signature",
            None,
        )
        current_signature = (
            current_signature_getter()
            if callable(current_signature_getter)
            else self.current_signature()
        )
        signature_changed = bool(
            self.context._learning_state_signature is None
            or self.context._learning_state_signature != current_signature
        )
        should_rebuild = bool(
            missing_learning_state
            or self.context._learning_state_stale
            or signature_changed
        )
        if not should_rebuild:
            return True

        prepare_learning_state = getattr(self.context, "_prepare_learning_state", None)
        if callable(prepare_learning_state):
            return bool(
                prepare_learning_state(
                    require_class_weights=require_class_weights,
                    show_success_dialog=False,
                )
            )
        return self.prepare_learning_state(
            require_class_weights=require_class_weights,
            show_success_dialog=False,
        )

    def prepare_learning_state(
        self,
        *,
        require_class_weights: bool,
        show_success_dialog: bool,
    ) -> bool:
        context = self.context
        parent = context
        state = getattr(context, "state", None)
        if not bool(getattr(state, "volume_loaded", False)) or context._raw_volume is None:
            self.operations.show_warning(
                "Load a raw volume before building datasets from bounding boxes.",
                parent=parent,
            )
            return False

        ordered_box_ids = tuple(
            row.box_id for row in getattr(context.bottom_panel, "state").bbox_rows
        )
        if not ordered_box_ids:
            self.operations.show_warning(
                "There are no bounding boxes to build datasets from.",
                parent=parent,
            )
            return False
        boxes_by_id = {box.id: box for box in context._bbox_manager.boxes()}
        learning_box_ids = tuple(
            box_id
            for box_id in ordered_box_ids
            if box_id in boxes_by_id and str(boxes_by_id[box_id].label) != "inference"
        )
        train_box_ids = tuple(
            box_id
            for box_id in learning_box_ids
            if box_id in boxes_by_id and str(boxes_by_id[box_id].label) == "train"
        )
        if not train_box_ids:
            self.operations.show_warning(
                (
                    "At least one bounding box labeled 'train' is required to build "
                    "datasets from bboxes."
                ),
                parent=parent,
            )
            return False

        active_segmentation = context._active_segmentation_volume()
        if active_segmentation is None:
            self.operations.show_warning(
                (
                    "Load a semantic segmentation map before building datasets "
                    "from bounding boxes."
                ),
                parent=parent,
            )
            return False
        seg_kind, seg_volume = active_segmentation
        if seg_kind != "semantic":
            self.operations.show_warning(
                (
                    "Only semantic segmentation is supported for learning-state "
                    "preparation."
                ),
                parent=parent,
            )
            return False
        validation_box_ids = tuple(
            box_id
            for box_id in learning_box_ids
            if box_id in boxes_by_id and str(boxes_by_id[box_id].label) == "validation"
        )
        if not validation_box_ids:
            self.operations.show_warning(
                (
                    "At least one bounding box labeled 'validation' is required to "
                    "build datasets from bboxes."
                ),
                parent=parent,
            )
            return False

        try:
            result = self.operations.prepare_learning_state_from_volumes(
                raw_volume=context._raw_volume,
                segmentation_volume=seg_volume,
                segmentation_kind=seg_kind,
                boxes_by_id=boxes_by_id,
                ordered_box_ids=learning_box_ids,
                training_parameters=self.training_parameters(),
                require_class_weights=bool(require_class_weights),
                label_space_source_signature=self.semantic_label_space_source_signature(
                    semantic_kind=seg_kind,
                    semantic_volume=seg_volume,
                ),
                extract_learning_bboxes_in_memory_fn=(
                    self.operations.extract_learning_bboxes_in_memory
                ),
                compute_class_weights_fn=(
                    self.operations.compute_and_store_current_learning_class_weights
                ),
                compute_label_coverage_fn=self.operations.compute_learning_label_coverage,
                format_label_coverage_warning_fn=(
                    self.operations.format_learning_label_coverage_warning
                ),
                derive_label_space_fn=(
                    self.operations.derive_label_space_from_semantic_segmentation
                ),
                clear_learning_bbox_batch_fn=lambda: self.operations.clear_learning_bbox_batch(
                    context
                ),
                get_learning_bbox_batch_fn=lambda: self.operations.get_learning_bbox_batch(
                    context
                ),
                **self.operations.learning_session_kwargs(context),
            )
        except Exception as exc:
            self.operations.show_warning(str(exc), parent=parent)
            return False

        current_signature_getter = getattr(context, "_current_learning_state_signature", None)
        if callable(current_signature_getter):
            context._learning_state_signature = current_signature_getter()
        else:
            try:
                context._learning_state_signature = self.current_signature()
            except Exception:
                context._learning_state_signature = None
        context._learning_state_stale = False

        if result.label_coverage_warning:
            self.operations.show_warning(result.label_coverage_warning, parent=parent)

        if not bool(show_success_dialog):
            return True

        summary_lines = [
            "Built bounding box learning datasets and buffers in memory.",
            (
                "- Temporary tensor entries built then released: "
                f"{result.outcome.tensor_entry_count}"
            ),
        ]
        if result.outcome.learning_train_box_ids:
            summary_lines.append(
                (
                    "- Learning DataLoader: "
                    f"{len(result.outcome.learning_train_box_ids)} train bboxes, "
                    f"batch_size={result.outcome.learning_batch_size}, "
                    f"num_workers={result.outcome.learning_num_workers}"
                )
            )
        if result.outcome.eval_validation_box_ids:
            summary_lines.append(
                (
                    "- Evaluation DataLoaders: "
                    f"{len(result.outcome.eval_validation_box_ids)} validation bboxes, "
                    f"batch_size={result.outcome.eval_batch_size}, "
                    f"num_workers={result.outcome.eval_num_workers}"
                )
            )
        if result.class_weights is not None:
            formatted_weights = self.operations.format_class_weights_for_summary(
                result.class_weights
            )
            if formatted_weights is None:
                summary_lines.append("- Loss class weights initialized on cuda:0.")
            else:
                summary_lines.append(
                    "- Loss class weights initialized on cuda:0: "
                    f"{formatted_weights}"
                )

        self.operations.show_info(
            "\n".join(summary_lines),
            parent=parent,
        )
        return True

    def current_signature(self) -> Tuple[object, ...]:
        context = self.context
        bbox_revision = int(getattr(context._bbox_manager, "revision", 0))
        ordered_box_ids = tuple(
            str(row.box_id).strip()
            for row in tuple(context.bottom_panel.state.bbox_rows)
            if str(row.box_id).strip()
        )
        boxes_by_id = {box.id: box for box in context._bbox_manager.boxes()}
        ordered_box_signature = []
        for box_id in ordered_box_ids:
            box = boxes_by_id.get(box_id)
            if box is None:
                ordered_box_signature.append((box_id, "<missing>"))
                continue
            ordered_box_signature.append(
                (
                    str(box.id),
                    str(box.label),
                    int(box.z0),
                    int(box.z1),
                    int(box.y0),
                    int(box.y1),
                    int(box.x0),
                    int(box.x1),
                )
            )

        active_segmentation = context._active_segmentation_volume()
        semantic_kind: Optional[str] = None
        semantic_source_path: Optional[str] = None
        if active_segmentation is not None:
            semantic_kind, volume = active_segmentation
            loader = getattr(volume, "loader", None)
            path_obj = getattr(loader, "path", None)
            if isinstance(path_obj, str) and path_obj.strip():
                semantic_source_path = str(path_obj)

        semantic_state_id: Optional[int] = None
        editor = context._segmentation_editor
        if editor is not None:
            semantic_state_id = int(editor.state_id)

        return (
            bbox_revision,
            tuple(ordered_box_signature),
            semantic_kind,
            semantic_source_path,
            semantic_state_id,
        )
