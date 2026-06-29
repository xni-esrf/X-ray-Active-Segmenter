from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication, QWidget

from ...learning import get_current_learning_model_runtime
from ...learning.qt_workers import (
    LearningInferenceBackgroundResult,
    LearningInferenceStopRequested,
    LearningInferenceWorker,
)
from ...utils import exception_message, get_logger
from ..dialogs import confirm_replace_inference_bboxes, show_info, show_warning


_LOGGER = get_logger(__name__)


class InferenceControllerContext(Protocol):
    _inference_running: bool
    _inference_stop_requested: bool
    _inference_worker: Optional[object]
    _inference_thread: Optional[object]
    _deferred_close_after_inference: bool
    _deferred_close_inference_mode: str
    _segmentation_editor: Optional[object]
    _semantic_volume: Optional[object]
    _raw_volume: Optional[object]
    _annotation_labels_dirty: bool
    _annotation_kind: str
    _bbox_manager: object
    bottom_panel: object

    def _abort_if_learning_training_running(self) -> bool: ...

    def _ensure_learning_state_for_action(self, action: str) -> bool: ...

    def _segment_inference_bboxes_with_dialog(self) -> bool: ...

    def _request_learning_inference_stop(self) -> None: ...

    def _active_segmentation_volume(self) -> Optional[Tuple[str, object]]: ...

    def _ensure_editable_segmentation_for_annotation(self) -> bool: ...

    def _show_inference_navigation_only_notice(self) -> None: ...

    def _start_learning_inference_background(
        self,
        *,
        model_runtime: object,
        inference_boxes: Sequence[object],
        raw_array: np.ndarray,
        label_values: Sequence[int],
        volume_shape: Sequence[int],
    ) -> None: ...

    def _enter_learning_inference_running_state(
        self,
        *,
        worker: object,
        thread: object,
    ) -> None: ...

    def _exit_learning_inference_running_state(self) -> None: ...

    def _inference_is_running(self) -> bool: ...

    def _training_is_running(self) -> bool: ...

    def _refresh_learning_inference_ui_state(self) -> None: ...

    def _refresh_undo_ui_state(self) -> None: ...

    def _clear_deferred_close_inference_state(self) -> None: ...

    def _apply_inference_predictions_in_single_commit(
        self,
        *,
        editor: object,
        predictions: Sequence[object],
        initial_failure_by_box_id: Optional[Mapping[str, str]] = None,
    ) -> Tuple[int, Tuple[str, ...], Dict[str, str]]: ...

    def _on_learning_inference_canceled(self, message: str) -> None: ...

    def _end_annotation_modification(self) -> None: ...

    def _sync_renderer_segmentation_labels(self) -> None: ...

    def _request_hover_readout(self) -> None: ...

    def _request_picked_readout(self) -> None: ...

    def _record_global_history_for_segmentation_operation(self, operation: object) -> None: ...

    def _refresh_annotation_ui_state(self) -> None: ...

    def render_all(self) -> None: ...


@dataclass(frozen=True)
class InferenceControllerOperations:
    show_warning: Callable[..., object] = show_warning
    show_info: Callable[..., object] = show_info
    confirm_replace_inference_bboxes: Callable[..., bool] = (
        confirm_replace_inference_bboxes
    )
    get_learning_model_runtime: Callable[[object], object] = (
        lambda _context: get_current_learning_model_runtime()
    )
    resolve_inference_label_values_for_runtime: Callable[[object], Tuple[int, ...]] = (
        lambda _runtime: tuple()
    )
    ordered_inference_boxes: Callable[..., Tuple[object, ...]] = lambda **_kwargs: tuple()
    find_overlapping_box_id_pairs: Callable[[Sequence[object]], Tuple[Tuple[str, str], ...]] = (
        lambda _boxes: tuple()
    )
    apply_predicted_bbox_to_editor: Callable[..., int] = lambda **_kwargs: 0
    exception_message: Callable[[Exception], str] = exception_message
    qthread_factory: Callable[[object], object] = QThread
    qthread_current_thread: Callable[[], object] = QThread.currentThread
    inference_worker_factory: Callable[[], object] = LearningInferenceWorker
    qapplication_instance: Callable[[], object] = QApplication.instance
    inference_navigation_lock_active: Callable[[object], bool] = lambda _context: False
    inference_stop_already_requested: Callable[[object], bool] = lambda _context: False
    logger: object = _LOGGER


@dataclass
class InferenceController:
    context: InferenceControllerContext
    operations: InferenceControllerOperations

    def inference_is_running(self) -> bool:
        return bool(self.context._inference_running)

    def handle_segment_inference_request(self) -> None:
        if self.operations.inference_navigation_lock_active(self.context):
            return
        if self.context._abort_if_learning_training_running():
            return
        if not bool(self.context._ensure_learning_state_for_action("inference")):
            return
        self.context._segment_inference_bboxes_with_dialog()

    def handle_stop_inference_request(self) -> None:
        self.context._request_learning_inference_stop()

    def segment_inference_bboxes_with_dialog(self) -> bool:
        model_runtime = self.operations.get_learning_model_runtime(self.context)
        if model_runtime is None:
            self.operations.show_warning(
                "Load a model before running Segment Inference BBox.",
                parent=self.context,
            )
            return False

        ordered_box_ids = tuple(row.box_id for row in self.context.bottom_panel.state.bbox_rows)
        boxes_by_id = {box.id: box for box in self.context._bbox_manager.boxes()}
        inference_boxes = self.operations.ordered_inference_boxes(
            ordered_box_ids=ordered_box_ids,
            boxes_by_id=boxes_by_id,
        )
        if not inference_boxes:
            self.operations.show_warning(
                (
                    "At least one bounding box labeled 'inference' is required to run "
                    "Segment Inference BBox."
                ),
                parent=self.context,
            )
            return False

        overlapping_pairs = self.operations.find_overlapping_box_id_pairs(inference_boxes)
        if overlapping_pairs:
            pair_text = ", ".join(
                f"{first_box_id} <-> {second_box_id}"
                for first_box_id, second_box_id in overlapping_pairs
            )
            self.operations.show_warning(
                (
                    "Inference bounding boxes overlap. Overlap is not supported for "
                    "Segment Inference BBox.\n\n"
                    f"Overlapping pairs: {pair_text}"
                ),
                parent=self.context,
            )
            return False

        try:
            label_values = self.operations.resolve_inference_label_values_for_runtime(
                model_runtime
            )
        except Exception as exc:
            self.operations.show_warning(str(exc), parent=self.context)
            return False

        active_segmentation = self.context._active_segmentation_volume()
        if active_segmentation is not None:
            active_kind, _active_volume = active_segmentation
            if active_kind == "instance" and self.context._semantic_volume is None:
                self.operations.show_warning(
                    (
                        "Segment Inference BBox requires a semantic map, but the active "
                        "map is instance and no semantic map is loaded."
                    ),
                    parent=self.context,
                )
                return False
        elif self.context._semantic_volume is None:
            if self.context._raw_volume is None:
                self.operations.show_warning(
                    "Load a raw volume before running Segment Inference BBox.",
                    parent=self.context,
                )
                return False
            self.context._annotation_kind = "semantic"
            if not self.context._ensure_editable_segmentation_for_annotation():
                self.operations.show_warning(
                    "Could not auto-create an empty semantic map for Segment Inference BBox.",
                    parent=self.context,
                )
                return False

        semantic_volume = self.context._semantic_volume
        if semantic_volume is None:
            self.operations.show_warning(
                "A semantic map is required to run Segment Inference BBox.",
                parent=self.context,
            )
            return False

        if not label_values:
            self.operations.show_warning(
                "Validation buffers did not provide any class label values.",
                parent=self.context,
            )
            return False

        has_non_empty_inference_bbox = False
        try:
            for box in inference_boxes:
                bbox_values = np.asarray(
                    semantic_volume.get_chunk(
                        (
                            slice(int(box.z0), int(box.z1)),
                            slice(int(box.y0), int(box.y1)),
                            slice(int(box.x0), int(box.x1)),
                        )
                    )
                )
                if np.any((bbox_values != 0) & (bbox_values != -100)):
                    has_non_empty_inference_bbox = True
                    break
        except Exception as exc:
            self.operations.show_warning(str(exc), parent=self.context)
            return False

        if has_non_empty_inference_bbox:
            if not self.operations.confirm_replace_inference_bboxes(parent=self.context):
                return False

        raw_volume = self.context._raw_volume
        if raw_volume is None:
            self.operations.show_warning(
                "Load a raw volume before running Segment Inference BBox.",
                parent=self.context,
            )
            return False

        editor = self.context._segmentation_editor
        if editor is None or editor.kind != "semantic":
            self.operations.show_warning(
                "A semantic map is required to run Segment Inference BBox.",
                parent=self.context,
            )
            return False

        try:
            raw_array = np.asarray(
                raw_volume.get_chunk((slice(None), slice(None), slice(None)))
            )
        except Exception as exc:
            self.operations.show_warning(
                self.operations.exception_message(exc),
                parent=self.context,
            )
            return False

        self.context._show_inference_navigation_only_notice()
        try:
            start_result = self.context._start_learning_inference_background(
                model_runtime=model_runtime,
                inference_boxes=inference_boxes,
                raw_array=raw_array,
                label_values=label_values,
                volume_shape=self.context._bbox_manager.volume_shape,
            )
        except Exception as exc:
            self.context._exit_learning_inference_running_state()
            self.operations.show_warning(
                self.operations.exception_message(exc),
                parent=self.context,
            )
            return False
        if isinstance(start_result, bool):
            return bool(start_result)
        return True

    def run_learning_inference_inline_compat(
        self,
        *,
        model_runtime: object,
        inference_boxes: Sequence[object],
        raw_array: np.ndarray,
        label_values: Sequence[int],
        volume_shape: Sequence[int],
    ) -> bool:
        worker = self.operations.inference_worker_factory()
        worker.configure(
            model_runtime=model_runtime,
            inference_boxes=inference_boxes,
            raw_array=raw_array,
            label_values=label_values,
            volume_shape=volume_shape,
            use_tiled_score_buffer=False,
        )
        try:
            result = worker._run_inference()
        except LearningInferenceStopRequested as exc:
            self.operations.show_info(
                f"Segment Inference BBox canceled: {self.operations.exception_message(exc)}",
                parent=self.context,
            )
            return False
        except Exception as exc:
            self.operations.show_warning(
                self.operations.exception_message(exc),
                parent=self.context,
            )
            return False

        editor = self.context._segmentation_editor
        if editor is None or getattr(editor, "kind", None) != "semantic":
            self.operations.show_warning(
                "A semantic map is required to run Segment Inference BBox.",
                parent=self.context,
            )
            return False

        failure_by_box_id: Dict[str, str] = dict(result.failure_by_box_id)
        cleanup_errors_by_box_id: Dict[str, Tuple[str, ...]] = {
            str(box_id): tuple(errors)
            for box_id, errors in tuple(result.cleanup_errors_by_box_id.items())
        }
        succeeded_box_ids: list[str] = []
        changed_voxel_count_total = 0

        begin_modification = getattr(editor, "begin_modification", None)
        commit_modification = getattr(editor, "commit_modification", None)
        cancel_modification = getattr(editor, "cancel_modification", None)
        in_modification = False
        try:
            if callable(begin_modification):
                begin_modification("segment_inference_bboxes")
                in_modification = True
            for prediction in tuple(result.predictions):
                box = prediction.box
                try:
                    changed_count = self.operations.apply_predicted_bbox_to_editor(
                        editor=editor,
                        box=box,
                        predicted_bbox=prediction.predicted_bbox,
                    )
                    changed_voxel_count_total += int(changed_count)
                    succeeded_box_ids.append(str(box.id))
                except Exception as exc:
                    failure_by_box_id[str(box.id)] = self.operations.exception_message(exc)
            if callable(commit_modification):
                committed = commit_modification()
                in_modification = False
                self.context._record_global_history_for_segmentation_operation(
                    committed
                )
        except Exception as exc:
            if in_modification and callable(cancel_modification):
                try:
                    cancel_modification()
                except Exception:
                    pass
            self.operations.show_warning(
                self.operations.exception_message(exc),
                parent=self.context,
            )
            return False

        if changed_voxel_count_total > 0:
            self.context._annotation_labels_dirty = True
            self.context._sync_renderer_segmentation_labels()
            self.context._request_hover_readout()
            self.context._request_picked_readout()
            self.context.render_all()
        self.context._refresh_annotation_ui_state()

        return self._show_inference_summary(
            total_count=int(result.total_count),
            succeeded_box_ids=tuple(succeeded_box_ids),
            failure_by_box_id=failure_by_box_id,
            cleanup_errors_by_box_id=cleanup_errors_by_box_id,
            changed_voxel_count_total=int(changed_voxel_count_total),
        )

    def show_inference_navigation_only_notice(self) -> None:
        parent = self.context if isinstance(self.context, QWidget) else None
        self.operations.show_info(
            (
                "Segment Inference BBox is starting.\n\n"
                "During inference, only navigation remains enabled:\n"
                "- Slice navigation, zoom/pan, contrast, and level controls\n"
                "- Bounding-box selection and double-click cursor jump\n\n"
                "Annotation/edit actions, undo/redo, and file/model operations are "
                "temporarily disabled.\n"
                "Use 'Stop Inference' to cancel the running inference."
            ),
            parent=parent,
        )

    def start_learning_inference_background(
        self,
        *,
        model_runtime: object,
        inference_boxes: Sequence[object],
        raw_array: np.ndarray,
        label_values: Sequence[int],
        volume_shape: Sequence[int],
    ) -> None:
        thread = self.operations.qthread_factory(self.context)
        worker = self.operations.inference_worker_factory()
        worker.configure(
            model_runtime=model_runtime,
            inference_boxes=inference_boxes,
            raw_array=raw_array,
            label_values=label_values,
            volume_shape=volume_shape,
            use_tiled_score_buffer=False,
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.completed.connect(self.context._on_learning_inference_completed)
        worker.canceled.connect(self.context._on_learning_inference_canceled)
        worker.failed.connect(self.context._on_learning_inference_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self.context._on_learning_inference_thread_finished)
        thread.finished.connect(thread.deleteLater)

        try:
            self.context._enter_learning_inference_running_state(
                worker=worker,
                thread=thread,
            )
            thread.start()
        except Exception:
            try:
                thread.quit()
            except Exception:
                pass
            try:
                worker.deleteLater()
            except Exception:
                pass
            try:
                thread.deleteLater()
            except Exception:
                pass
            raise

    def apply_inference_predictions_in_single_commit(
        self,
        *,
        editor: object,
        predictions: Sequence[object],
        initial_failure_by_box_id: Optional[Mapping[str, str]] = None,
    ) -> Tuple[int, Tuple[str, ...], Dict[str, str]]:
        current_thread = self.operations.qthread_current_thread()
        context_thread_getter = getattr(self.context, "thread", None)
        context_thread = context_thread_getter() if callable(context_thread_getter) else None
        if current_thread is not context_thread:
            raise RuntimeError("Inference predictions must be applied on the main UI thread.")

        failure_by_box_id: Dict[str, str] = {}
        if isinstance(initial_failure_by_box_id, Mapping):
            for box_id, reason in tuple(initial_failure_by_box_id.items()):
                failure_by_box_id[str(box_id)] = str(reason)

        succeeded_box_ids: list[str] = []
        changed_voxel_count_total = 0

        self.context._end_annotation_modification()
        if self.context._annotation_labels_dirty:
            self.context._sync_renderer_segmentation_labels()
        if self.operations.inference_stop_already_requested(self.context):
            raise LearningInferenceStopRequested(
                "Inference canceled by user before applying predictions."
            )
        editor.begin_modification("segment_inference_bboxes")
        try:
            for prediction in tuple(predictions):
                if self.operations.inference_stop_already_requested(self.context):
                    raise LearningInferenceStopRequested(
                        "Inference canceled by user before commit."
                    )
                box = prediction.box
                try:
                    changed_count = self.operations.apply_predicted_bbox_to_editor(
                        editor=editor,
                        box=box,
                        predicted_bbox=prediction.predicted_bbox,
                    )
                    changed_voxel_count_total += int(changed_count)
                    succeeded_box_ids.append(str(box.id))
                except Exception as exc:
                    failure_by_box_id[str(box.id)] = self.operations.exception_message(exc)
        except LearningInferenceStopRequested:
            cancel_modification = getattr(editor, "cancel_modification", None)
            if callable(cancel_modification):
                cancel_modification()
            raise
        else:
            committed_operation = editor.commit_modification()
            self.context._record_global_history_for_segmentation_operation(
                committed_operation
            )

        return (
            int(changed_voxel_count_total),
            tuple(succeeded_box_ids),
            dict(failure_by_box_id),
        )

    def on_learning_inference_completed(self, result: object) -> None:
        try:
            if not isinstance(result, LearningInferenceBackgroundResult):
                self.operations.show_warning(
                    "Segment Inference BBox completed with an invalid result payload.",
                    parent=self.context,
                )
                return
            if self.operations.inference_stop_already_requested(self.context):
                self.context._on_learning_inference_canceled(
                    "Inference canceled by user before applying predictions."
                )
                return

            editor = self.context._segmentation_editor
            if editor is None or editor.kind != "semantic":
                self.operations.show_warning(
                    (
                        "Segment Inference BBox completed, but the semantic map is no longer "
                        "available. Predictions were discarded."
                    ),
                    parent=self.context,
                )
                return

            failure_by_box_id = dict(result.failure_by_box_id)
            cleanup_errors_by_box_id = {
                str(box_id): tuple(errors)
                for box_id, errors in tuple(result.cleanup_errors_by_box_id.items())
            }
            (
                changed_voxel_count_total,
                succeeded_box_ids,
                failure_by_box_id,
            ) = self.context._apply_inference_predictions_in_single_commit(
                editor=editor,
                predictions=result.predictions,
                initial_failure_by_box_id=failure_by_box_id,
            )
        except LearningInferenceStopRequested as exc:
            self.context._on_learning_inference_canceled(str(exc))
            return
        finally:
            self.clear_learning_inference_stop_request_state()

        if changed_voxel_count_total > 0:
            self.context._annotation_labels_dirty = True
            self.context._sync_renderer_segmentation_labels()
            self.context._request_hover_readout()
            self.context._request_picked_readout()
            self.context.render_all()
        self.context._refresh_annotation_ui_state()

        self._show_inference_summary(
            total_count=int(result.total_count),
            succeeded_box_ids=tuple(succeeded_box_ids),
            failure_by_box_id=failure_by_box_id,
            cleanup_errors_by_box_id=cleanup_errors_by_box_id,
            changed_voxel_count_total=int(changed_voxel_count_total),
        )

    def on_learning_inference_canceled(self, message: str) -> None:
        try:
            normalized_message = str(message).strip()
            if not normalized_message:
                normalized_message = "Inference canceled by user."
            self.operations.show_info(
                f"Segment Inference BBox canceled: {normalized_message}",
                parent=self.context,
            )
        finally:
            self.clear_learning_inference_stop_request_state()

    def on_learning_inference_failed(self, message: str) -> None:
        try:
            normalized_message = str(message).strip()
            if not normalized_message:
                normalized_message = "Unknown inference error."
            self.operations.show_warning(
                f"Segment Inference BBox aborted: {normalized_message}",
                parent=self.context,
            )
        finally:
            self.clear_learning_inference_stop_request_state()

    def on_learning_inference_thread_finished(self) -> None:
        self.context._exit_learning_inference_running_state()
        if not self.context._deferred_close_after_inference:
            return
        self.context._clear_deferred_close_inference_state()

        app_instance = self.operations.qapplication_instance()
        if app_instance is None:
            return
        set_quit_on_last = getattr(app_instance, "setQuitOnLastWindowClosed", None)
        if callable(set_quit_on_last):
            try:
                set_quit_on_last(True)
            except Exception:
                pass
        quit_method = getattr(app_instance, "quit", None)
        if callable(quit_method):
            quit_method()

    def request_learning_inference_stop(self) -> None:
        if not self.context._inference_is_running():
            return
        if self.operations.inference_stop_already_requested(self.context):
            return
        worker = self.context._inference_worker
        request_stop = getattr(worker, "request_stop", None)
        if callable(request_stop):
            self.context._inference_stop_requested = True
            self.context._refresh_learning_inference_ui_state()
            request_stop()

    def clear_learning_inference_stop_request_state(self) -> None:
        self.context._inference_stop_requested = False

    def clear_deferred_close_inference_state(self) -> None:
        self.context._deferred_close_after_inference = False
        self.context._deferred_close_inference_mode = "none"

    def set_deferred_close_after_stop_inference(self) -> None:
        self.context._deferred_close_after_inference = True
        self.context._deferred_close_inference_mode = "stop_and_close"

    def finalize_deferred_close_inference_and_quit(self) -> None:
        self.context._clear_deferred_close_inference_state()

        app_instance = self.operations.qapplication_instance()
        if app_instance is None:
            return
        set_quit_on_last = getattr(app_instance, "setQuitOnLastWindowClosed", None)
        if callable(set_quit_on_last):
            try:
                set_quit_on_last(True)
            except Exception:
                pass
        quit_method = getattr(app_instance, "quit", None)
        if callable(quit_method):
            quit_method()

    def refresh_learning_inference_ui_state(self) -> None:
        inference_running = self.operations.inference_navigation_lock_active(self.context)
        training_running = bool(self.context._training_is_running())
        learning_actions_enabled = not training_running and not inference_running
        self.context.bottom_panel.set_segment_inference_enabled(learning_actions_enabled)
        self.context.bottom_panel.set_train_model_enabled(learning_actions_enabled)
        bottom_panel_state = getattr(self.context.bottom_panel, "state", None)
        inference_navigation_only_state = bool(
            getattr(bottom_panel_state, "learning_inference_navigation_only", False)
            or getattr(self.context.bottom_panel, "_inference_navigation_only_mode", False)
            or getattr(self.context.bottom_panel, "_stop_inference_enabled_requested", False)
        )
        should_update_inference_controls = bool(
            inference_running or inference_navigation_only_state
        )
        set_stop_inference_enabled = getattr(
            self.context.bottom_panel,
            "set_stop_inference_enabled",
            None,
        )
        if should_update_inference_controls and callable(set_stop_inference_enabled):
            stop_enabled = bool(
                inference_running
                and not self.operations.inference_stop_already_requested(self.context)
            )
            set_stop_inference_enabled(stop_enabled)
        set_navigation_only_mode = getattr(
            self.context.bottom_panel,
            "set_inference_navigation_only_mode",
            None,
        )
        if should_update_inference_controls and callable(set_navigation_only_mode):
            set_navigation_only_mode(inference_running)
        refresh_undo_ui_state = getattr(self.context, "_refresh_undo_ui_state", None)
        if callable(refresh_undo_ui_state):
            refresh_undo_ui_state()

    def enter_learning_inference_running_state(
        self,
        *,
        worker: object,
        thread: object,
    ) -> None:
        self.context._inference_running = True
        self.context._inference_stop_requested = False
        self.context._inference_worker = worker
        self.context._inference_thread = thread
        self.context._refresh_learning_inference_ui_state()

    def exit_learning_inference_running_state(self) -> None:
        self.context._inference_running = False
        self.context._inference_worker = None
        self.context._inference_thread = None
        self.context._refresh_learning_inference_ui_state()

    def _show_inference_summary(
        self,
        *,
        total_count: int,
        succeeded_box_ids: Sequence[str],
        failure_by_box_id: Mapping[str, str],
        cleanup_errors_by_box_id: Mapping[str, Sequence[str]],
        changed_voxel_count_total: int,
    ) -> bool:
        success_count = int(len(tuple(succeeded_box_ids)))
        failure_count = int(len(failure_by_box_id))
        cleanup_warning_count = int(
            sum(len(errors) for errors in cleanup_errors_by_box_id.values())
        )

        if failure_count <= 0 and cleanup_warning_count <= 0:
            title_line = "Segment Inference BBox completed: all inference bboxes succeeded."
        elif failure_count > 0 and success_count <= 0:
            title_line = "Segment Inference BBox failed: no inference bbox was successfully processed."
        elif failure_count > 0:
            title_line = "Segment Inference BBox completed with partial success."
        else:
            title_line = "Segment Inference BBox completed with cleanup warnings."

        summary_lines = [
            title_line,
            f"- processed inference bboxes: {int(total_count)}",
            f"- succeeded: {success_count}",
            f"- failed: {failure_count}",
            f"- changed voxels: {int(changed_voxel_count_total)}",
        ]
        if succeeded_box_ids:
            summary_lines.append("- succeeded bbox ids: " + ", ".join(succeeded_box_ids))
        if failure_by_box_id:
            summary_lines.append("- failed bbox reasons:")
            for box_id, reason in tuple(failure_by_box_id.items()):
                summary_lines.append(f"  - {box_id}: {reason}")
        if cleanup_errors_by_box_id:
            summary_lines.append("- cleanup warnings:")
            for box_id, errors in tuple(cleanup_errors_by_box_id.items()):
                for error in tuple(errors):
                    summary_lines.append(f"  - {box_id}: {error}")

        summary = "\n".join(summary_lines)
        if failure_by_box_id or cleanup_errors_by_box_id:
            self.operations.show_warning(summary, parent=self.context)
        else:
            self.operations.show_info(summary, parent=self.context)
        return bool(failure_count <= 0)
