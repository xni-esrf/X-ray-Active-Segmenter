from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.bbox import BoundingBox

try:
    from src.learning import (
        LearningBBoxEvalRuntime,
        LearningLabelSpace,
        LearningSession,
        TrainingParameters,
    )
    from src.learning.qt_workers import (
        LearningInferenceBackgroundResult,
        LearningInferenceStopRequested,
    )
    from src.ui.main_window import MainWindow, QThread
except Exception:  # pragma: no cover - environment dependent
    MainWindow = None  # type: ignore[assignment]
    LearningBBoxEvalRuntime = None  # type: ignore[assignment]
    LearningLabelSpace = None  # type: ignore[assignment]
    LearningSession = None  # type: ignore[assignment]
    TrainingParameters = None  # type: ignore[assignment]
    LearningInferenceBackgroundResult = None  # type: ignore[assignment]
    QThread = None  # type: ignore[assignment]
    LearningInferenceStopRequested = None  # type: ignore[assignment]


@unittest.skipUnless(MainWindow is not None, "MainWindow is not available")
class MainWindowLearningTrainingStateTests(unittest.TestCase):
    def test_handle_change_training_parameters_request_applies_accepted_dialog_values(self) -> None:
        parameters = TrainingParameters(training_batch_size=8)
        window_like = SimpleNamespace(
            _training_parameters=TrainingParameters(),
            _learning_state_stale=False,
            _learning_session=LearningSession(),
        )

        with patch(
            "src.ui.main_window.open_training_parameters_dialog",
            return_value=SimpleNamespace(accepted=True, parameters=parameters),
        ) as dialog_mock:
            MainWindow._handle_change_training_parameters_request(window_like)

        dialog_mock.assert_called_once_with(TrainingParameters(), parent=window_like)
        self.assertEqual(window_like._training_parameters, parameters)
        self.assertTrue(window_like._learning_state_stale)

    def test_apply_training_parameters_marks_learning_state_stale_for_dataloader_changes(self) -> None:
        window_like = SimpleNamespace(
            _training_parameters=TrainingParameters(),
            _learning_state_stale=False,
            _learning_session=LearningSession(),
        )

        MainWindow._apply_training_parameters(
            window_like,
            TrainingParameters(
                training_batch_size=8,
                validation_batch_size=6,
                patches_per_epoch=2000,
            ),
        )

        self.assertEqual(window_like._training_parameters.training_batch_size, 8)
        self.assertEqual(window_like._training_parameters.validation_batch_size, 6)
        self.assertEqual(window_like._training_parameters.patches_per_epoch, 2000)
        self.assertTrue(window_like._learning_state_stale)

    def test_apply_training_parameters_keeps_learning_state_current_for_early_stop_only(self) -> None:
        window_like = SimpleNamespace(
            _training_parameters=TrainingParameters(),
            _learning_state_stale=False,
            _learning_session=LearningSession(),
        )

        MainWindow._apply_training_parameters(
            window_like,
            TrainingParameters(early_stopping_patience=6),
        )

        self.assertEqual(window_like._training_parameters.early_stopping_patience, 6)
        self.assertFalse(window_like._learning_state_stale)

    def test_train_request_rebuilds_learning_state_after_dataloader_parameter_change(self) -> None:
        prepare_calls = []
        train_calls = []

        def _prepare_learning_state(**kwargs: object) -> bool:
            prepare_calls.append(dict(kwargs))
            window_like._learning_state_stale = False
            return True

        window_like = SimpleNamespace(
            _training_parameters=TrainingParameters(),
            _learning_state_signature=("sig",),
            _learning_state_stale=False,
            _current_learning_state_signature=lambda: ("sig",),
            _inference_running=False,
            _training_running=False,
            _abort_if_learning_training_running=lambda: False,
            _ensure_learning_state_for_action=lambda action: MainWindow._ensure_learning_state_for_action(
                window_like,
                action,
            ),
            _prepare_learning_state=_prepare_learning_state,
            _train_model_on_dataset_with_dialog=lambda: train_calls.append("train"),
        )

        MainWindow._apply_training_parameters(
            window_like,
            TrainingParameters(training_batch_size=8),
        )
        with patch(
            "src.ui.main_window.get_current_learning_dataloader_runtime",
            return_value=SimpleNamespace(class_weights=object()),
        ), patch(
            "src.ui.main_window.get_current_learning_eval_runtimes_by_box_id",
            return_value={"bbox_0001": object()},
        ):
            MainWindow._handle_train_model_request(window_like)

        self.assertFalse(window_like._learning_state_stale)
        self.assertEqual(
            prepare_calls,
            [
                {
                    "require_class_weights": True,
                    "show_success_dialog": False,
                }
            ],
        )
        self.assertEqual(train_calls, ["train"])

    def test_apply_training_parameters_updates_existing_optimizer_learning_rate(self) -> None:
        optimizer = SimpleNamespace(
            param_groups=[
                {"lr": 5e-5, "lwise_lr_decay_rate": 1.0},
                {"lr": 2.5e-5, "lwise_lr_decay_rate": 0.5},
                {"lr": 5e-5},
            ]
        )
        session = LearningSession()
        runtime = session.set_model_components(
            model=object(),
            optimizer=optimizer,
            checkpoint_path="model.cp",
            device_ids=(0,),
            num_classes=2,
            hyperparameters={"lr": 5e-5},
        )
        window_like = SimpleNamespace(
            _training_parameters=TrainingParameters(),
            _learning_state_stale=False,
            _learning_session=session,
        )

        MainWindow._apply_training_parameters(
            window_like,
            TrainingParameters(learning_rate=0.002),
        )

        self.assertEqual(window_like._training_parameters.learning_rate, 0.002)
        self.assertFalse(window_like._learning_state_stale)
        self.assertEqual(runtime.hyperparameters["lr"], 0.002)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.002)
        self.assertEqual(optimizer.param_groups[1]["lr"], 0.001)
        self.assertEqual(optimizer.param_groups[2]["lr"], 0.002)

    @unittest.skipUnless(LearningSession is not None, "LearningSession is not available")
    def test_clear_learning_label_space_for_uses_window_session(self) -> None:
        session = LearningSession()
        session.set_label_space(LearningLabelSpace(label_values=(0, 1)))
        window_like = SimpleNamespace(_learning_session=session)

        MainWindow._clear_learning_label_space_for(window_like)

        self.assertIsNone(session.get_label_space())

    @unittest.skipUnless(LearningSession is not None, "LearningSession is not available")
    def test_semantic_edit_marks_learning_state_stale_and_clears_label_space(self) -> None:
        session = LearningSession()
        session.set_label_space(LearningLabelSpace(label_values=(0, 1)))
        operation = SimpleNamespace(changed_voxels=3, operation_id="op-1")
        editor = SimpleNamespace(
            kind="semantic",
            latest_undo_operation_id=lambda: "another-op",
        )
        window_like = SimpleNamespace(
            _learning_session=session,
            _learning_state_stale=False,
            _segmentation_editor=editor,
        )

        MainWindow._record_global_history_for_segmentation_operation(
            window_like,
            operation,
        )

        self.assertTrue(window_like._learning_state_stale)
        self.assertIsNone(session.get_label_space())

    def _box(
        self,
        *,
        box_id: str = "bbox_0001",
        z0: int = 1,
        z1: int = 4,
        y0: int = 2,
        y1: int = 6,
        x0: int = 3,
        x1: int = 8,
        label: str = "train",
    ) -> BoundingBox:
        return BoundingBox.from_bounds(
            box_id=box_id,
            z0=z0,
            z1=z1,
            y0=y0,
            y1=y1,
            x0=x0,
            x1=x1,
            label=label,
            volume_shape=(20, 30, 40),
        )

    def _make_operation_editor(
        self,
        *,
        array: np.ndarray,
        active_label: int = 1,
        commit_result: object = None,
        raise_on_erase: bool = False,
        raise_on_assign: bool = False,
    ) -> object:
        class _Editor:
            def __init__(self) -> None:
                self._array = np.asarray(array)
                self.active_label = int(active_label)
                self.begin_calls: list[str] = []
                self.erase_calls: list[tuple[np.ndarray, str, bool, object]] = []
                self.erase_masked_region_calls: list[
                    tuple[tuple[int, int], tuple[int, int], tuple[int, int], np.ndarray, object, str]
                ] = []
                self.assign_calls: list[tuple[np.ndarray, int, str, bool]] = []
                self.commit_calls = 0
                self.cancel_calls = 0

            def array_view(self) -> np.ndarray:
                return self._array

            def begin_modification(self, name: str) -> None:
                self.begin_calls.append(str(name))

            def erase(
                self,
                coordinates: object,
                *,
                target_label: object = None,
                operation_name: str = "erase",
                ignore_out_of_bounds: bool = True,
            ) -> object:
                if raise_on_erase:
                    raise ValueError("erase boom")
                self.erase_calls.append(
                    (
                        np.asarray(coordinates, dtype=np.int64).copy(),
                        str(operation_name),
                        bool(ignore_out_of_bounds),
                        target_label,
                    )
                )
                return None

            def erase_masked_region(
                self,
                *,
                z_bounds: tuple[int, int],
                y_bounds: tuple[int, int],
                x_bounds: tuple[int, int],
                region_mask: object,
                target_label: object = None,
                operation_name: str = "erase_masked_region",
            ) -> object:
                if raise_on_erase:
                    raise ValueError("erase boom")
                self.erase_masked_region_calls.append(
                    (
                        tuple(int(v) for v in z_bounds),
                        tuple(int(v) for v in y_bounds),
                        tuple(int(v) for v in x_bounds),
                        np.asarray(region_mask, dtype=bool).copy(),
                        target_label,
                        str(operation_name),
                    )
                )
                return None

            def assign(
                self,
                coordinates: object,
                *,
                label: int,
                operation_name: str = "assign",
                ignore_out_of_bounds: bool = True,
            ) -> object:
                if raise_on_assign:
                    raise ValueError("assign boom")
                self.assign_calls.append(
                    (
                        np.asarray(coordinates, dtype=np.int64).copy(),
                        int(label),
                        str(operation_name),
                        bool(ignore_out_of_bounds),
                    )
                )
                return None

            def commit_modification(self) -> object:
                self.commit_calls += 1
                return commit_result

            def cancel_modification(self) -> None:
                self.cancel_calls += 1

        return _Editor()

    def _make_learning_actions_bottom_panel(
        self,
        calls: list,
        *,
        train_enabled_event: str = "train_enabled",
    ) -> SimpleNamespace:
        return SimpleNamespace(
            set_learning_training_running=lambda running: calls.append(
                ("running", bool(running))
            ),
            set_segment_inference_enabled=lambda enabled: calls.append(
                ("segment_enabled", bool(enabled))
            ),
            set_train_model_enabled=lambda enabled: calls.append(
                (train_enabled_event, bool(enabled))
            ),
            set_stop_training_enabled=lambda enabled: calls.append(
                ("stop_enabled", bool(enabled))
            ),
            set_stop_inference_enabled=lambda enabled: calls.append(
                ("stop_inference_enabled", bool(enabled))
            ),
            set_inference_navigation_only_mode=lambda enabled: calls.append(
                ("navigation_only", bool(enabled))
            ),
        )

    def _make_training_state_window(
        self,
        calls: list,
        *,
        training_running: bool = False,
    ) -> SimpleNamespace:
        window_like = SimpleNamespace(
            _training_running=bool(training_running),
            _training_worker=None,
            _training_thread=None,
            bottom_panel=self._make_learning_actions_bottom_panel(
                calls,
                train_enabled_event="enabled",
            ),
        )
        window_like._training_is_running = lambda: bool(window_like._training_running)
        window_like._refresh_learning_inference_ui_state = (
            lambda: MainWindow._refresh_learning_inference_ui_state(window_like)
        )
        window_like._refresh_learning_training_ui_state = (
            lambda: MainWindow._refresh_learning_training_ui_state(window_like)
        )
        return window_like

    def _make_inference_state_window(
        self,
        calls: list,
        *,
        training_running: bool = False,
        inference_running: bool = True,
        stop_requested: bool = False,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            bottom_panel=self._make_learning_actions_bottom_panel(calls),
            _training_is_running=lambda: bool(training_running),
            _inference_is_running=lambda: bool(inference_running),
            _inference_stop_requested=bool(stop_requested),
            _refresh_undo_ui_state=lambda: calls.append(("refresh_undo", None)),
        )

    def _make_bbox_selection_window(
        self,
        *,
        selected_ids: tuple[str, ...] = tuple(),
        manager_selected_id: str | None = None,
        boxes: tuple[BoundingBox, ...] = tuple(),
        editor: object = None,
        ensure_result: bool = True,
        ensure_calls: list[str] | None = None,
        history_calls: list[object] | None = None,
        refresh_calls: list[str] | None = None,
        end_calls: list[str] | None = None,
    ) -> SimpleNamespace:
        def _ensure_editor() -> bool:
            if ensure_calls is not None:
                ensure_calls.append("ensure")
            return bool(ensure_result)

        window_like = SimpleNamespace(
            bottom_panel=SimpleNamespace(
                selected_bounding_boxes=lambda: tuple(selected_ids),
                state=SimpleNamespace(bbox_selected_ids=tuple(selected_ids)),
            ),
            _bbox_manager=SimpleNamespace(
                selected_id=manager_selected_id,
                boxes=lambda: tuple(boxes),
            ),
            _segmentation_editor=editor,
            _ensure_editable_segmentation_for_annotation=_ensure_editor,
            _end_annotation_modification=(
                (lambda: end_calls.append("end"))
                if end_calls is not None
                else (lambda: None)
            ),
            _record_global_history_for_segmentation_operation=(
                (lambda operation: history_calls.append(operation))
                if history_calls is not None
                else (lambda _operation: None)
            ),
            _sync_renderer_segmentation_labels=(
                (lambda: refresh_calls.append("sync_labels"))
                if refresh_calls is not None
                else (lambda: None)
            ),
            _request_hover_readout=(
                (lambda: refresh_calls.append("hover"))
                if refresh_calls is not None
                else (lambda: None)
            ),
            _request_picked_readout=(
                (lambda: refresh_calls.append("picked"))
                if refresh_calls is not None
                else (lambda: None)
            ),
            render_all=(
                (lambda: refresh_calls.append("render"))
                if refresh_calls is not None
                else (lambda: None)
            ),
            _refresh_annotation_ui_state=(
                (lambda: refresh_calls.append("ui"))
                if refresh_calls is not None
                else (lambda: None)
            ),
        )
        return window_like

    def test_refresh_learning_training_ui_state_sets_bottom_panel_flags(self) -> None:
        calls: list[tuple[str, object]] = []
        window_like = self._make_training_state_window(calls)

        MainWindow._refresh_learning_training_ui_state(window_like)

        self.assertEqual(
            calls,
            [
                ("running", False),
                ("segment_enabled", True),
                ("enabled", True),
                ("stop_enabled", False),
            ],
        )

    def test_enter_and_exit_learning_training_running_state_updates_runtime_and_ui(self) -> None:
        calls: list[tuple[str, object]] = []
        window_like = self._make_training_state_window(calls)

        worker = object()
        thread = object()

        MainWindow._enter_learning_training_running_state(
            window_like,
            worker=worker,
            thread=thread,
        )

        self.assertTrue(window_like._training_running)
        self.assertIs(window_like._training_worker, worker)
        self.assertIs(window_like._training_thread, thread)
        self.assertEqual(
            calls,
            [
                ("running", True),
                ("segment_enabled", False),
                ("enabled", False),
                ("stop_enabled", True),
            ],
        )

        calls.clear()
        MainWindow._exit_learning_training_running_state(window_like)

        self.assertFalse(window_like._training_running)
        self.assertIsNone(window_like._training_worker)
        self.assertIsNone(window_like._training_thread)
        self.assertEqual(
            calls,
            [
                ("running", False),
                ("segment_enabled", True),
                ("enabled", True),
                ("stop_enabled", False),
            ],
        )

    def test_clear_deferred_close_training_state_resets_flags(self) -> None:
        window_like = SimpleNamespace(
            _deferred_close_after_training=True,
            _deferred_close_training_mode="stop_and_close",
        )

        MainWindow._clear_deferred_close_training_state(window_like)

        self.assertFalse(window_like._deferred_close_after_training)
        self.assertEqual(window_like._deferred_close_training_mode, "none")

    def test_set_deferred_close_after_stop_training_sets_stop_mode(self) -> None:
        window_like = SimpleNamespace(
            _deferred_close_after_training=False,
            _deferred_close_training_mode="none",
        )

        MainWindow._set_deferred_close_after_stop_training(window_like)

        self.assertTrue(window_like._deferred_close_after_training)
        self.assertEqual(window_like._deferred_close_training_mode, "stop_and_close")

    def test_handle_load_model_request_warns_and_aborts_when_training_running(self) -> None:
        called = []
        window_like = SimpleNamespace(
            _training_is_running=lambda: True,
            _instantiate_foundation_model_with_dialog=lambda: called.append("instantiate"),
            _abort_if_learning_training_running=lambda: MainWindow._abort_if_learning_training_running(
                window_like
            ),
        )

        with patch("src.ui.main_window.show_warning") as warning_mock:
            MainWindow._handle_load_model_request(window_like)

        self.assertEqual(called, [])
        warning_mock.assert_called_once()
        self.assertIn("training is running", warning_mock.call_args.args[0].lower())
        self.assertIs(warning_mock.call_args.kwargs["parent"], window_like)

    def test_handle_save_model_request_warns_and_aborts_when_training_running(self) -> None:
        called = []
        window_like = SimpleNamespace(
            _training_is_running=lambda: True,
            _save_model_with_dialog=lambda: called.append("save"),
            _abort_if_learning_training_running=lambda: MainWindow._abort_if_learning_training_running(
                window_like
            ),
        )

        with patch("src.ui.main_window.show_warning") as warning_mock:
            MainWindow._handle_save_model_request(window_like)

        self.assertEqual(called, [])
        warning_mock.assert_called_once()
        self.assertIn("training is running", warning_mock.call_args.args[0].lower())
        self.assertIs(warning_mock.call_args.kwargs["parent"], window_like)

    def test_handle_train_model_request_warns_and_aborts_when_training_running(self) -> None:
        window_like = SimpleNamespace(
            _training_is_running=lambda: True,
            _abort_if_learning_training_running=lambda: MainWindow._abort_if_learning_training_running(
                window_like
            ),
        )

        with patch("src.ui.main_window.show_warning") as warning_mock:
            MainWindow._handle_train_model_request(window_like)

        warning_mock.assert_called_once()
        self.assertIn("training is running", warning_mock.call_args.args[0].lower())
        self.assertIs(warning_mock.call_args.kwargs["parent"], window_like)

    def test_handle_segment_inference_request_warns_and_aborts_when_training_running(self) -> None:
        called = []
        window_like = SimpleNamespace(
            _training_is_running=lambda: True,
            _inference_is_running=lambda: False,
            _segment_inference_bboxes_with_dialog=lambda: called.append("segment"),
            _abort_if_learning_training_running=lambda: MainWindow._abort_if_learning_training_running(
                window_like
            ),
        )

        with patch("src.ui.main_window.show_warning") as warning_mock:
            MainWindow._handle_segment_inference_request(window_like)

        self.assertEqual(called, [])
        warning_mock.assert_called_once()
        self.assertIn("training is running", warning_mock.call_args.args[0].lower())
        self.assertIs(warning_mock.call_args.kwargs["parent"], window_like)

    def test_handle_segment_inference_request_calls_dialog_when_not_training(self) -> None:
        called = []
        window_like = SimpleNamespace(
            _training_is_running=lambda: False,
            _inference_is_running=lambda: False,
            _ensure_learning_state_for_action=lambda action: called.append(
                ("ensure", str(action))
            )
            or True,
            _segment_inference_bboxes_with_dialog=lambda: called.append("segment"),
            _abort_if_learning_training_running=lambda: MainWindow._abort_if_learning_training_running(
                window_like
            ),
        )

        with patch("src.ui.main_window.show_warning") as warning_mock:
            MainWindow._handle_segment_inference_request(window_like)

        self.assertEqual(called, [("ensure", "inference"), "segment"])
        warning_mock.assert_not_called()

    def test_handle_segment_inference_request_aborts_when_learning_state_prepare_fails(self) -> None:
        called = []
        window_like = SimpleNamespace(
            _training_is_running=lambda: False,
            _inference_is_running=lambda: False,
            _ensure_learning_state_for_action=lambda action: called.append(
                ("ensure", str(action))
            )
            or False,
            _segment_inference_bboxes_with_dialog=lambda: called.append("segment"),
            _abort_if_learning_training_running=lambda: MainWindow._abort_if_learning_training_running(
                window_like
            ),
        )

        with patch("src.ui.main_window.show_warning") as warning_mock:
            MainWindow._handle_segment_inference_request(window_like)

        self.assertEqual(called, [("ensure", "inference")])
        warning_mock.assert_not_called()

    def test_ensure_learning_state_for_inference_skips_prepare_and_returns_true(self) -> None:
        prepare_calls = []
        window_like = SimpleNamespace(
            _prepare_learning_state=lambda **kwargs: prepare_calls.append(dict(kwargs)) or False,
        )

        result = MainWindow._ensure_learning_state_for_action(window_like, "inference")

        self.assertTrue(result)
        self.assertEqual(prepare_calls, [])

    def test_ensure_learning_state_for_train_rebuilds_when_learning_state_is_missing(self) -> None:
        prepare_calls = []
        window_like = SimpleNamespace(
            _learning_state_signature=("sig",),
            _learning_state_stale=False,
            _current_learning_state_signature=lambda: ("sig",),
            _prepare_learning_state=lambda **kwargs: prepare_calls.append(dict(kwargs)) or True,
        )

        with patch(
            "src.ui.main_window.get_current_learning_dataloader_runtime",
            return_value=None,
        ), patch(
            "src.ui.main_window.get_current_learning_eval_runtimes_by_box_id",
            return_value={},
        ):
            result = MainWindow._ensure_learning_state_for_action(window_like, "train")

        self.assertTrue(result)
        self.assertEqual(
            prepare_calls,
            [
                {
                    "require_class_weights": True,
                    "show_success_dialog": False,
                }
            ],
        )

    def test_ensure_learning_state_for_load_model_rebuilds_when_learning_state_is_missing(self) -> None:
        prepare_calls = []
        window_like = SimpleNamespace(
            _learning_state_signature=("sig",),
            _learning_state_stale=False,
            _current_learning_state_signature=lambda: ("sig",),
            _prepare_learning_state=lambda **kwargs: prepare_calls.append(dict(kwargs)) or True,
        )

        with patch(
            "src.ui.main_window.get_current_learning_dataloader_runtime",
            return_value=None,
        ), patch(
            "src.ui.main_window.get_current_learning_eval_runtimes_by_box_id",
            return_value={},
        ):
            result = MainWindow._ensure_learning_state_for_action(window_like, "load_model")

        self.assertTrue(result)
        self.assertEqual(
            prepare_calls,
            [
                {
                    "require_class_weights": False,
                    "show_success_dialog": False,
                }
            ],
        )

    @unittest.skipUnless(LearningSession is not None, "LearningSession is not available")
    def test_ensure_learning_state_reads_window_learning_session(self) -> None:
        session = LearningSession()
        session.set_dataloader_components(
            dataset=object(),
            sampler=object(),
            dataloader=object(),
            train_box_ids=("bbox_0001",),
        )
        session.set_eval_runtimes_by_box_id(
            {
                "bbox_0002": LearningBBoxEvalRuntime(
                    box_id="bbox_0002",
                    dataloader=object(),
                    buffer=object(),
                )
            }
        )
        prepare_calls = []
        window_like = SimpleNamespace(
            _learning_session=session,
            _learning_state_signature=("sig",),
            _learning_state_stale=False,
            _current_learning_state_signature=lambda: ("sig",),
            _prepare_learning_state=lambda **kwargs: prepare_calls.append(dict(kwargs)) or False,
        )

        with patch(
            "src.ui.main_window.get_current_learning_dataloader_runtime",
            side_effect=AssertionError("global dataloader getter should not be used"),
        ), patch(
            "src.ui.main_window.get_current_learning_eval_runtimes_by_box_id",
            side_effect=AssertionError("global eval getter should not be used"),
        ):
            result = MainWindow._ensure_learning_state_for_action(window_like, "load_model")

        self.assertTrue(result)
        self.assertEqual(prepare_calls, [])

    def test_handle_open_request_is_silent_when_inference_running(self) -> None:
        calls = []
        window_like = SimpleNamespace(
            _inference_is_running=lambda: True,
            _maybe_resolve_unsaved_segmentation=lambda **_kwargs: calls.append("seg") or True,
            _maybe_resolve_unsaved_bounding_boxes=lambda **_kwargs: calls.append("bbox") or True,
        )

        with patch("src.ui.main_window.open_file_dialog") as dialog_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock:
            MainWindow._handle_open_request(window_like)

        self.assertEqual(calls, [])
        dialog_mock.assert_not_called()
        warning_mock.assert_not_called()

    def test_handle_undo_requested_is_silent_when_inference_running(self) -> None:
        calls = []
        window_like = SimpleNamespace(
            _inference_is_running=lambda: True,
            _finalize_bbox_history_transaction=lambda: calls.append("finalize"),
            _end_annotation_modification=lambda: calls.append("end"),
        )

        with patch("src.ui.main_window.show_warning") as warning_mock:
            MainWindow._handle_undo_requested(window_like)

        self.assertEqual(calls, [])
        warning_mock.assert_not_called()

    def test_handle_bounding_box_face_moved_is_noop_when_inference_running(self) -> None:
        calls = []
        window_like = SimpleNamespace(
            _inference_is_running=lambda: True,
            _bbox_manager=SimpleNamespace(
                get=lambda box_id: calls.append(("get", str(box_id))),
                selected_id="bbox_0001",
            ),
            _sync_bounding_boxes_ui=lambda: calls.append(("sync",)),
        )

        MainWindow._handle_bounding_box_face_moved(window_like, "bbox_0001", "x_min", 4)

        self.assertEqual(calls, [])

    def test_refresh_learning_inference_ui_state_sets_navigation_only_mode(self) -> None:
        calls: list[tuple[str, object]] = []
        window_like = self._make_inference_state_window(calls)

        MainWindow._refresh_learning_inference_ui_state(window_like)

        self.assertEqual(
            calls,
            [
                ("segment_enabled", False),
                ("train_enabled", False),
                ("stop_inference_enabled", True),
                ("navigation_only", True),
                ("refresh_undo", None),
            ],
        )

    def test_refresh_learning_inference_ui_state_disables_stop_button_after_cancel_request(self) -> None:
        calls: list[tuple[str, object]] = []
        window_like = self._make_inference_state_window(
            calls,
            stop_requested=True,
        )

        MainWindow._refresh_learning_inference_ui_state(window_like)

        self.assertEqual(
            calls,
            [
                ("segment_enabled", False),
                ("train_enabled", False),
                ("stop_inference_enabled", False),
                ("navigation_only", True),
                ("refresh_undo", None),
            ],
        )

    def test_request_learning_inference_stop_requests_worker_once_and_disables_button(self) -> None:
        calls: list[str] = []
        window_like = SimpleNamespace(
            _inference_is_running=lambda: True,
            _inference_stop_requested=False,
            _inference_worker=SimpleNamespace(request_stop=lambda: calls.append("request_stop")),
            _refresh_learning_inference_ui_state=lambda: calls.append("refresh_ui"),
        )

        MainWindow._request_learning_inference_stop(window_like)
        MainWindow._request_learning_inference_stop(window_like)

        self.assertTrue(window_like._inference_stop_requested)
        self.assertEqual(calls, ["refresh_ui", "request_stop"])

    def test_enter_and_exit_learning_inference_running_state_resets_stop_request(self) -> None:
        calls: list[str] = []
        worker = object()
        thread = object()
        window_like = SimpleNamespace(
            _inference_running=False,
            _inference_stop_requested=True,
            _inference_worker=None,
            _inference_thread=None,
            _refresh_learning_inference_ui_state=lambda: calls.append("refresh"),
        )

        MainWindow._enter_learning_inference_running_state(
            window_like,
            worker=worker,
            thread=thread,
        )
        self.assertTrue(window_like._inference_running)
        self.assertFalse(window_like._inference_stop_requested)
        self.assertIs(window_like._inference_worker, worker)
        self.assertIs(window_like._inference_thread, thread)

        MainWindow._exit_learning_inference_running_state(window_like)
        self.assertFalse(window_like._inference_running)
        self.assertFalse(window_like._inference_stop_requested)
        self.assertIsNone(window_like._inference_worker)
        self.assertIsNone(window_like._inference_thread)
        self.assertEqual(calls, ["refresh", "refresh"])

    def test_refresh_undo_ui_state_disables_undo_redo_while_inference_running(self) -> None:
        calls: list[tuple[str, int, bool]] = []
        window_like = SimpleNamespace(
            state=SimpleNamespace(volume_loaded=True),
            _inference_is_running=lambda: True,
            _global_history=SimpleNamespace(
                undo_depth=lambda: 3,
                redo_depth=lambda: 2,
            ),
            bottom_panel=SimpleNamespace(
                set_undo_state=lambda *, depth, enabled: calls.append(
                    ("undo", int(depth), bool(enabled))
                ),
                set_redo_state=lambda *, depth, enabled: calls.append(
                    ("redo", int(depth), bool(enabled))
                ),
            ),
        )

        MainWindow._refresh_undo_ui_state(window_like)

        self.assertEqual(
            calls,
            [
                ("undo", 3, False),
                ("redo", 2, False),
            ],
        )

    def test_show_inference_navigation_only_notice_displays_info_popup(self) -> None:
        window_like = SimpleNamespace()

        with patch("src.ui.main_window.show_info") as info_mock:
            MainWindow._show_inference_navigation_only_notice(window_like)

        info_mock.assert_called_once()
        info_text = str(info_mock.call_args.args[0]).lower()
        self.assertIn("segment inference bbox is starting", info_text)
        self.assertIn("only navigation remains enabled", info_text)
        self.assertIn("stop inference", info_text)
        self.assertIsNone(info_mock.call_args.kwargs.get("parent"))

    def test_on_learning_inference_completed_skips_apply_when_stop_already_requested(self) -> None:
        self.assertIsNotNone(LearningInferenceBackgroundResult)
        canceled_messages: list[str] = []
        calls: list[str] = []
        result = LearningInferenceBackgroundResult(
            total_count=1,
            predictions=tuple(),
            failure_by_box_id={},
            cleanup_errors_by_box_id={},
        )
        window_like = SimpleNamespace(
            _inference_stop_requested=True,
            _segmentation_editor=SimpleNamespace(kind="semantic"),
            _apply_inference_predictions_in_single_commit=lambda **_kwargs: calls.append("apply"),
            _on_learning_inference_canceled=lambda message: canceled_messages.append(str(message)),
        )

        with patch("src.ui.main_window.show_warning") as warning_mock:
            MainWindow._on_learning_inference_completed(window_like, result)

        self.assertEqual(calls, [])
        self.assertEqual(
            canceled_messages,
            ["Inference canceled by user before applying predictions."],
        )
        self.assertFalse(window_like._inference_stop_requested)
        warning_mock.assert_not_called()

    def test_on_learning_inference_completed_handles_stop_during_apply_as_canceled(self) -> None:
        self.assertIsNotNone(LearningInferenceBackgroundResult)
        self.assertIsNotNone(LearningInferenceStopRequested)
        canceled_messages: list[str] = []
        stop_error = LearningInferenceStopRequested(
            "Inference canceled by user before commit."
        )
        result = LearningInferenceBackgroundResult(
            total_count=1,
            predictions=tuple(),
            failure_by_box_id={},
            cleanup_errors_by_box_id={},
        )

        def _raise_stop(**_kwargs: object) -> object:
            raise stop_error

        window_like = SimpleNamespace(
            _inference_stop_requested=False,
            _segmentation_editor=SimpleNamespace(kind="semantic"),
            _apply_inference_predictions_in_single_commit=_raise_stop,
            _on_learning_inference_canceled=lambda message: canceled_messages.append(str(message)),
        )

        with patch("src.ui.main_window.show_warning") as warning_mock:
            MainWindow._on_learning_inference_completed(window_like, result)

        self.assertEqual(
            canceled_messages,
            ["Inference canceled by user before commit."],
        )
        self.assertFalse(window_like._inference_stop_requested)
        warning_mock.assert_not_called()

    def test_on_learning_inference_completed_clears_stop_request_for_invalid_payload(self) -> None:
        window_like = SimpleNamespace(_inference_stop_requested=True)

        with patch("src.ui.main_window.show_warning") as warning_mock:
            MainWindow._on_learning_inference_completed(window_like, object())

        warning_mock.assert_called_once()
        self.assertIn("invalid result payload", warning_mock.call_args.args[0].lower())
        self.assertFalse(window_like._inference_stop_requested)

    def test_on_learning_inference_canceled_clears_stop_request_flag(self) -> None:
        window_like = SimpleNamespace(_inference_stop_requested=True)

        with patch("src.ui.main_window.show_info") as info_mock:
            MainWindow._on_learning_inference_canceled(window_like, "")

        info_mock.assert_called_once()
        self.assertIn("canceled", info_mock.call_args.args[0].lower())
        self.assertFalse(window_like._inference_stop_requested)

    def test_on_learning_inference_failed_clears_stop_request_flag(self) -> None:
        window_like = SimpleNamespace(_inference_stop_requested=True)

        with patch("src.ui.main_window.show_warning") as warning_mock:
            MainWindow._on_learning_inference_failed(window_like, "")

        warning_mock.assert_called_once()
        self.assertIn("aborted", warning_mock.call_args.args[0].lower())
        self.assertFalse(window_like._inference_stop_requested)

    def test_on_learning_inference_thread_finished_deferred_close_quits_app(self) -> None:
        actions: list[str] = []
        quit_on_last_closed_values: list[bool] = []
        app_like = SimpleNamespace(
            setQuitOnLastWindowClosed=lambda value: quit_on_last_closed_values.append(bool(value)),
            quit=lambda: actions.append("quit"),
        )
        window_like = SimpleNamespace(
            _deferred_close_after_inference=True,
            _exit_learning_inference_running_state=lambda: actions.append("exit_state"),
            _clear_deferred_close_inference_state=lambda: actions.append("clear_state"),
        )

        with patch("src.ui.main_window.QApplication.instance", return_value=app_like):
            MainWindow._on_learning_inference_thread_finished(window_like)

        self.assertEqual(actions, ["exit_state", "clear_state", "quit"])
        self.assertEqual(quit_on_last_closed_values, [True])

    def test_apply_inference_predictions_single_commit_aborts_before_begin_when_stop_requested(
        self,
    ) -> None:
        self.assertIsNotNone(LearningInferenceStopRequested)
        self.assertIsNotNone(QThread)
        box = self._box(box_id="bbox_0001", label="inference")
        editor = self._make_operation_editor(
            array=np.zeros((20, 30, 40), dtype=np.uint16),
            active_label=1,
            commit_result=SimpleNamespace(changed_voxels=0, operation_id=1),
        )
        history_calls: list[object] = []
        window_like = SimpleNamespace(
            _inference_stop_requested=True,
            thread=lambda: QThread.currentThread(),
            _end_annotation_modification=lambda: None,
            _annotation_labels_dirty=False,
            _sync_renderer_segmentation_labels=lambda: None,
            _record_global_history_for_segmentation_operation=lambda operation: history_calls.append(
                operation
            ),
        )
        prediction = SimpleNamespace(
            box=box,
            predicted_bbox=np.zeros((3, 4, 5), dtype=np.uint16),
        )

        with patch("src.ui.main_window._apply_predicted_bbox_to_editor") as apply_mock:
            with self.assertRaises(LearningInferenceStopRequested):
                MainWindow._apply_inference_predictions_in_single_commit(
                    window_like,
                    editor=editor,
                    predictions=(prediction,),
                    initial_failure_by_box_id={},
                )

        apply_mock.assert_not_called()
        self.assertEqual(editor.begin_calls, [])
        self.assertEqual(editor.cancel_calls, 0)
        self.assertEqual(editor.commit_calls, 0)
        self.assertEqual(history_calls, [])

    def test_apply_inference_predictions_single_commit_cancels_when_stop_requested_mid_loop(
        self,
    ) -> None:
        self.assertIsNotNone(LearningInferenceStopRequested)
        self.assertIsNotNone(QThread)
        box1 = self._box(box_id="bbox_0001", label="inference")
        box2 = self._box(
            box_id="bbox_0002",
            label="inference",
            z0=4,
            z1=7,
            y0=4,
            y1=7,
            x0=4,
            x1=7,
        )
        editor = self._make_operation_editor(
            array=np.zeros((20, 30, 40), dtype=np.uint16),
            active_label=1,
            commit_result=SimpleNamespace(changed_voxels=6, operation_id=2),
        )
        history_calls: list[object] = []
        window_like = SimpleNamespace(
            _inference_stop_requested=False,
            thread=lambda: QThread.currentThread(),
            _end_annotation_modification=lambda: None,
            _annotation_labels_dirty=False,
            _sync_renderer_segmentation_labels=lambda: None,
            _record_global_history_for_segmentation_operation=lambda operation: history_calls.append(
                operation
            ),
        )
        predictions = (
            SimpleNamespace(
                box=box1,
                predicted_bbox=np.zeros((3, 4, 5), dtype=np.uint16),
            ),
            SimpleNamespace(
                box=box2,
                predicted_bbox=np.zeros((3, 3, 3), dtype=np.uint16),
            ),
        )

        def _apply_side_effect(*_args: object, **_kwargs: object) -> int:
            window_like._inference_stop_requested = True
            return 6

        with patch(
            "src.ui.main_window._apply_predicted_bbox_to_editor",
            side_effect=_apply_side_effect,
        ) as apply_mock:
            with self.assertRaises(LearningInferenceStopRequested):
                MainWindow._apply_inference_predictions_in_single_commit(
                    window_like,
                    editor=editor,
                    predictions=predictions,
                    initial_failure_by_box_id={},
                )

        self.assertEqual(apply_mock.call_count, 1)
        self.assertEqual(editor.begin_calls, ["segment_inference_bboxes"])
        self.assertEqual(editor.cancel_calls, 1)
        self.assertEqual(editor.commit_calls, 0)
        self.assertEqual(history_calls, [])

    def test_handle_stop_training_request_is_silent_when_not_running(self) -> None:
        called = []
        window_like = SimpleNamespace(
            _training_is_running=lambda: False,
            _training_worker=None,
            _request_learning_training_stop=lambda: MainWindow._request_learning_training_stop(
                window_like
            ),
        )

        with patch("src.ui.main_window.show_warning") as warning_mock:
            MainWindow._handle_stop_training_request(window_like)

        self.assertEqual(called, [])
        warning_mock.assert_not_called()

    def test_handle_stop_training_request_calls_worker_request_stop_when_running(self) -> None:
        calls = []
        worker = SimpleNamespace(
            request_stop=lambda: calls.append("stop"),
        )
        window_like = SimpleNamespace(
            _training_is_running=lambda: True,
            _training_worker=worker,
            _request_learning_training_stop=lambda: MainWindow._request_learning_training_stop(
                window_like
            ),
        )

        with patch("src.ui.main_window.show_warning") as warning_mock:
            MainWindow._handle_stop_training_request(window_like)

        self.assertEqual(calls, ["stop"])
        warning_mock.assert_not_called()

    def test_handle_load_model_request_calls_dialog_without_preparing_learning_state(self) -> None:
        called = []
        window_like = SimpleNamespace(
            _training_is_running=lambda: False,
            _ensure_learning_state_for_action=lambda action: called.append(
                ("ensure", str(action))
            )
            or True,
            _instantiate_foundation_model_with_dialog=lambda: called.append("instantiate"),
            _abort_if_learning_training_running=lambda: MainWindow._abort_if_learning_training_running(
                window_like
            ),
        )

        with patch("src.ui.main_window.show_warning") as warning_mock:
            MainWindow._handle_load_model_request(window_like)

        self.assertEqual(called, ["instantiate"])
        warning_mock.assert_not_called()

    def test_handle_load_model_request_ignores_learning_state_prepare_result(self) -> None:
        called = []
        window_like = SimpleNamespace(
            _training_is_running=lambda: False,
            _ensure_learning_state_for_action=lambda action: called.append(
                ("ensure", str(action))
            )
            or False,
            _instantiate_foundation_model_with_dialog=lambda: called.append("instantiate"),
            _abort_if_learning_training_running=lambda: MainWindow._abort_if_learning_training_running(
                window_like
            ),
        )

        with patch("src.ui.main_window.show_warning") as warning_mock:
            MainWindow._handle_load_model_request(window_like)

        self.assertEqual(called, ["instantiate"])
        warning_mock.assert_not_called()

    def test_handle_save_model_request_calls_dialog_when_not_training(self) -> None:
        called = []
        window_like = SimpleNamespace(
            _training_is_running=lambda: False,
            _save_model_with_dialog=lambda: called.append("save"),
            _abort_if_learning_training_running=lambda: MainWindow._abort_if_learning_training_running(
                window_like
            ),
        )

        with patch("src.ui.main_window.show_warning") as warning_mock:
            MainWindow._handle_save_model_request(window_like)

        self.assertEqual(called, ["save"])
        warning_mock.assert_not_called()

    def test_handle_median_filter_selected_request_delegates_to_shared_handler(self) -> None:
        calls = []
        window_like = SimpleNamespace(
            _handle_selected_bbox_segmentation_processing_request=lambda operation: calls.append(
                str(operation)
            )
        )

        MainWindow._handle_median_filter_selected_request(window_like)

        self.assertEqual(calls, ["median_filter"])

    def test_handle_erosion_selected_request_delegates_to_shared_handler(self) -> None:
        calls = []
        window_like = SimpleNamespace(
            _handle_selected_bbox_segmentation_processing_request=lambda operation: calls.append(
                str(operation)
            )
        )

        MainWindow._handle_erosion_selected_request(window_like)

        self.assertEqual(calls, ["erosion"])

    def test_handle_dilation_selected_request_delegates_to_shared_handler(self) -> None:
        calls = []
        window_like = SimpleNamespace(
            _handle_selected_bbox_segmentation_processing_request=lambda operation: calls.append(
                str(operation)
            )
        )

        MainWindow._handle_dilation_selected_request(window_like)

        self.assertEqual(calls, ["dilation"])

    def test_handle_erase_bbox_segmentation_request_delegates_to_erase_method(self) -> None:
        calls = []
        window_like = SimpleNamespace(
            _erase_selected_bbox_segmentation=lambda: calls.append("erase")
        )

        MainWindow._handle_erase_bbox_segmentation_request(window_like)

        self.assertEqual(calls, ["erase"])

    def test_handle_bounding_box_double_clicked_silently_ignores_missing_box(self) -> None:
        get_calls = []
        cursor_calls = []
        window_like = SimpleNamespace(
            _bbox_manager=SimpleNamespace(
                get=lambda box_id: get_calls.append(str(box_id)) or None,
            ),
            sync_manager=SimpleNamespace(
                set_cursor_indices=lambda indices: cursor_calls.append(tuple(indices))
            ),
        )

        with patch("src.ui.main_window.show_warning") as warning_mock, patch(
            "src.ui.main_window.show_info"
        ) as info_mock:
            MainWindow._handle_bounding_box_double_clicked(window_like, "bbox_9999")

        self.assertEqual(get_calls, ["bbox_9999"])
        self.assertEqual(cursor_calls, [])
        warning_mock.assert_not_called()
        info_mock.assert_not_called()

    def test_handle_bounding_box_double_clicked_normalizes_id_before_lookup(self) -> None:
        get_calls = []
        cursor_calls = []
        box = self._box(box_id="bbox_0001")
        window_like = SimpleNamespace(
            _bbox_manager=SimpleNamespace(
                get=lambda box_id: get_calls.append(str(box_id)) or box,
                volume_shape=(20, 30, 40),
            )
            ,
            sync_manager=SimpleNamespace(
                set_cursor_indices=lambda indices: cursor_calls.append(tuple(indices))
            ),
        )

        with patch("src.ui.main_window.show_warning") as warning_mock, patch(
            "src.ui.main_window.show_info"
        ) as info_mock:
            MainWindow._handle_bounding_box_double_clicked(window_like, "  bbox_0001  ")

        self.assertEqual(get_calls, ["bbox_0001"])
        self.assertEqual(cursor_calls, [(2, 4, 5)])
        warning_mock.assert_not_called()
        info_mock.assert_not_called()

    def test_handle_bounding_box_double_clicked_rounds_half_up_and_clamps_to_volume(self) -> None:
        cursor_calls = []
        window_like = SimpleNamespace(
            _bbox_manager=SimpleNamespace(
                get=lambda _box_id: SimpleNamespace(center_index_space=(2.5, 9.6, 99.2)),
                volume_shape=(4, 10, 11),
            ),
            sync_manager=SimpleNamespace(
                set_cursor_indices=lambda indices: cursor_calls.append(tuple(indices))
            ),
        )

        MainWindow._handle_bounding_box_double_clicked(window_like, "bbox_0002")

        self.assertEqual(cursor_calls, [(3, 9, 10)])

    def test_handle_bounding_box_double_clicked_selects_box_syncs_ui_and_refreshes_readouts(
        self,
    ) -> None:
        calls = []
        box = self._box(box_id="bbox_0002")
        window_like = SimpleNamespace(
            _bbox_manager=SimpleNamespace(
                get=lambda _box_id: box,
                select=lambda box_id: calls.append(("select", str(box_id))),
                volume_shape=(20, 30, 40),
            ),
            sync_manager=SimpleNamespace(
                set_cursor_indices=lambda indices: calls.append(("cursor", tuple(indices)))
            ),
            _sync_bounding_boxes_ui=lambda: calls.append(("sync_ui",)),
            _request_hover_readout=lambda: calls.append(("hover",)),
            _request_picked_readout=lambda: calls.append(("picked",)),
        )

        MainWindow._handle_bounding_box_double_clicked(window_like, "bbox_0002")

        self.assertEqual(
            calls,
            [
                ("cursor", (2, 4, 5)),
                ("select", "bbox_0002"),
                ("sync_ui",),
                ("hover",),
                ("picked",),
            ],
        )

    def test_handle_bounding_box_double_clicked_uses_readout_refresh_fallbacks(self) -> None:
        calls = []
        box = self._box(box_id="bbox_0003")
        window_like = SimpleNamespace(
            _bbox_manager=SimpleNamespace(
                get=lambda _box_id: box,
                select=lambda _box_id: None,
                volume_shape=(20, 30, 40),
            ),
            sync_manager=SimpleNamespace(set_cursor_indices=lambda _indices: None),
            _sync_bounding_boxes_ui=lambda: None,
            _refresh_hover_readout=lambda: calls.append("hover_refresh"),
            _refresh_picked_readout=lambda: calls.append("picked_refresh"),
        )

        MainWindow._handle_bounding_box_double_clicked(window_like, "bbox_0003")

        self.assertEqual(calls, ["hover_refresh", "picked_refresh"])

    def test_handle_bounding_box_double_clicked_returns_silently_when_select_fails(self) -> None:
        calls = []
        box = self._box(box_id="bbox_0004")

        def _raise_key_error(_box_id: str) -> None:
            raise KeyError("missing")

        window_like = SimpleNamespace(
            _bbox_manager=SimpleNamespace(
                get=lambda _box_id: box,
                select=_raise_key_error,
                volume_shape=(20, 30, 40),
            ),
            sync_manager=SimpleNamespace(
                set_cursor_indices=lambda indices: calls.append(("cursor", tuple(indices)))
            ),
            _sync_bounding_boxes_ui=lambda: calls.append(("sync_ui",)),
            _request_hover_readout=lambda: calls.append(("hover",)),
            _request_picked_readout=lambda: calls.append(("picked",)),
        )

        with patch("src.ui.main_window.show_warning") as warning_mock:
            MainWindow._handle_bounding_box_double_clicked(window_like, "bbox_0004")

        self.assertEqual(calls, [("cursor", (2, 4, 5))])
        warning_mock.assert_not_called()

    def test_erase_selected_bbox_segmentation_warns_when_no_bbox_selected(self) -> None:
        ensure_calls = []
        window_like = self._make_bbox_selection_window(
            editor=object(),
            ensure_calls=ensure_calls,
        )

        with patch("src.ui.main_window.show_info") as info_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock:
            MainWindow._erase_selected_bbox_segmentation(window_like)

        self.assertEqual(ensure_calls, [])
        warning_mock.assert_called_once()
        self.assertIn("select one or more bounding boxes", warning_mock.call_args.args[0].lower())
        self.assertIs(warning_mock.call_args.kwargs["parent"], window_like)
        info_mock.assert_not_called()

    def test_erase_selected_bbox_segmentation_uses_view_selected_bbox_when_table_selection_is_empty(
        self,
    ) -> None:
        ensure_calls = []
        window_like = self._make_bbox_selection_window(
            manager_selected_id="bbox_0007",
            editor=None,
            ensure_result=False,
            ensure_calls=ensure_calls,
        )

        with patch("src.ui.main_window.show_info") as info_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock:
            MainWindow._erase_selected_bbox_segmentation(window_like)

        self.assertEqual(ensure_calls, ["ensure"])
        warning_mock.assert_called_once()
        self.assertIn("no semantic or instance segmentation map", warning_mock.call_args.args[0].lower())
        self.assertIs(warning_mock.call_args.kwargs["parent"], window_like)
        info_mock.assert_not_called()

    def test_erase_selected_bbox_segmentation_warns_when_editor_still_missing_after_ensure(self) -> None:
        ensure_calls = []
        window_like = self._make_bbox_selection_window(
            selected_ids=("bbox_0001",),
            editor=None,
            ensure_calls=ensure_calls,
        )

        with patch("src.ui.main_window.show_info") as info_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock:
            MainWindow._erase_selected_bbox_segmentation(window_like)

        self.assertEqual(ensure_calls, ["ensure"])
        warning_mock.assert_called_once()
        self.assertIn("no semantic or instance segmentation map", warning_mock.call_args.args[0].lower())
        self.assertIs(warning_mock.call_args.kwargs["parent"], window_like)
        info_mock.assert_not_called()

    def test_erase_selected_bbox_segmentation_warns_when_selected_boxes_are_missing(self) -> None:
        window_like = self._make_bbox_selection_window(
            selected_ids=("bbox_0007",),
            editor=object(),
        )

        with patch("src.ui.main_window.show_info") as info_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock:
            MainWindow._erase_selected_bbox_segmentation(window_like)

        warning_mock.assert_called_once()
        self.assertIn("no longer available", warning_mock.call_args.args[0].lower())
        self.assertIs(warning_mock.call_args.kwargs["parent"], window_like)
        info_mock.assert_not_called()

    def test_erase_selected_bbox_segmentation_erases_union_in_single_modification(self) -> None:
        box1 = self._box(box_id="bbox_0001", z0=1, z1=4, y0=2, y1=5, x0=3, x1=6)
        box2 = self._box(box_id="bbox_0002", z0=3, z1=6, y0=4, y1=7, x0=5, x1=8)
        committed_operation = SimpleNamespace(changed_voxels=53, operation_id=55)
        editor = self._make_operation_editor(
            array=np.ones((20, 30, 40), dtype=np.uint16),
            active_label=4,
            commit_result=committed_operation,
        )
        end_calls = []
        history_calls = []
        refresh_calls = []
        window_like = self._make_bbox_selection_window(
            selected_ids=("bbox_0001",),
            manager_selected_id="bbox_0002",
            boxes=(box1, box2),
            editor=editor,
            history_calls=history_calls,
            refresh_calls=refresh_calls,
            end_calls=end_calls,
        )

        with patch("src.ui.main_window.show_info") as info_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock:
            MainWindow._erase_selected_bbox_segmentation(window_like)

        self.assertEqual(end_calls, ["end"])
        self.assertEqual(editor.begin_calls, ["erase_bbox_segmentation_selected"])
        self.assertEqual(editor.commit_calls, 1)
        self.assertEqual(editor.cancel_calls, 0)
        self.assertEqual(history_calls, [committed_operation])
        self.assertEqual(
            refresh_calls,
            ["sync_labels", "hover", "picked", "render", "ui"],
        )
        self.assertEqual(len(editor.erase_calls), 0)
        self.assertEqual(len(editor.erase_masked_region_calls), 1)
        (
            erase_z_bounds,
            erase_y_bounds,
            erase_x_bounds,
            erase_region_mask,
            erase_target_label,
            erase_operation_name,
        ) = editor.erase_masked_region_calls[0]
        self.assertEqual(erase_operation_name, "erase_bbox_segmentation_selected")
        self.assertIsNone(erase_target_label)
        self.assertEqual(erase_z_bounds, (1, 6))
        self.assertEqual(erase_y_bounds, (2, 7))
        self.assertEqual(erase_x_bounds, (3, 8))
        self.assertEqual(erase_region_mask.shape, (5, 5, 5))
        self.assertEqual(int(np.count_nonzero(erase_region_mask)), 53)
        self.assertTrue(bool(erase_region_mask[0, 0, 0]))
        self.assertTrue(bool(erase_region_mask[4, 4, 4]))
        warning_mock.assert_not_called()
        info_mock.assert_not_called()

    def test_erase_selected_bbox_segmentation_only_refreshes_annotation_ui_when_unchanged(self) -> None:
        box = self._box(box_id="bbox_0001", z0=1, z1=4, y0=2, y1=5, x0=3, x1=6)
        committed_operation = SimpleNamespace(changed_voxels=0, operation_id=56)
        editor = self._make_operation_editor(
            array=np.zeros((20, 30, 40), dtype=np.uint16),
            active_label=4,
            commit_result=committed_operation,
        )
        history_calls = []
        refresh_calls = []
        window_like = self._make_bbox_selection_window(
            selected_ids=("bbox_0001",),
            boxes=(box,),
            editor=editor,
            history_calls=history_calls,
            refresh_calls=refresh_calls,
        )

        with patch("src.ui.main_window.show_info") as info_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock:
            MainWindow._erase_selected_bbox_segmentation(window_like)

        self.assertEqual(editor.begin_calls, ["erase_bbox_segmentation_selected"])
        self.assertEqual(editor.commit_calls, 1)
        self.assertEqual(editor.cancel_calls, 0)
        self.assertEqual(history_calls, [committed_operation])
        self.assertEqual(refresh_calls, ["ui"])
        warning_mock.assert_not_called()
        info_mock.assert_not_called()

    def test_erase_selected_bbox_segmentation_does_not_materialize_absolute_coordinates(self) -> None:
        box = self._box(box_id="bbox_0001", z0=1, z1=4, y0=2, y1=5, x0=3, x1=6)
        committed_operation = SimpleNamespace(changed_voxels=27, operation_id=57)
        editor = self._make_operation_editor(
            array=np.ones((20, 30, 40), dtype=np.uint16),
            active_label=4,
            commit_result=committed_operation,
        )
        window_like = self._make_bbox_selection_window(
            selected_ids=("bbox_0001",),
            boxes=(box,),
            editor=editor,
        )

        with patch.object(
            MainWindow,
            "_mask_to_absolute_coordinates",
            side_effect=AssertionError("absolute coordinate materialization should not be used"),
        ), patch("src.ui.main_window.show_warning") as warning_mock:
            MainWindow._erase_selected_bbox_segmentation(window_like)

        self.assertEqual(len(editor.erase_masked_region_calls), 1)
        warning_mock.assert_not_called()

    def test_erase_selected_bbox_segmentation_cancels_and_warns_on_editor_error(self) -> None:
        box = self._box(box_id="bbox_0001", z0=1, z1=4, y0=2, y1=5, x0=3, x1=6)
        editor = self._make_operation_editor(
            array=np.ones((20, 30, 40), dtype=np.uint16),
            active_label=4,
            commit_result=SimpleNamespace(changed_voxels=27, operation_id=56),
            raise_on_erase=True,
        )
        history_calls = []
        window_like = self._make_bbox_selection_window(
            selected_ids=("bbox_0001",),
            boxes=(box,),
            editor=editor,
            history_calls=history_calls,
        )

        with patch("src.ui.main_window.show_info") as info_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock:
            MainWindow._erase_selected_bbox_segmentation(window_like)

        self.assertEqual(editor.begin_calls, ["erase_bbox_segmentation_selected"])
        self.assertEqual(editor.cancel_calls, 1)
        self.assertEqual(editor.commit_calls, 0)
        self.assertEqual(history_calls, [])
        warning_mock.assert_called_once()
        self.assertIn("erase boom", warning_mock.call_args.args[0].lower())
        self.assertIs(warning_mock.call_args.kwargs["parent"], window_like)
        info_mock.assert_not_called()

    def test_shared_selected_bbox_processing_request_delegates_to_processing_method(self) -> None:
        calls = []
        window_like = SimpleNamespace(
            _process_selected_bbox_segmentation_operation=lambda operation: calls.append(
                str(operation)
            )
        )

        MainWindow._handle_selected_bbox_segmentation_processing_request(
            window_like,
            "median_filter",
        )

        self.assertEqual(calls, ["median_filter"])

    def test_process_selected_bbox_segmentation_operation_warns_when_no_bbox_selected(self) -> None:
        ensure_calls = []
        window_like = self._make_bbox_selection_window(
            editor=object(),
            ensure_calls=ensure_calls,
        )

        with patch("src.ui.main_window.show_info") as info_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock:
            MainWindow._process_selected_bbox_segmentation_operation(window_like, "dilation")

        self.assertEqual(ensure_calls, [])
        warning_mock.assert_called_once()
        self.assertIn("select one or more bounding boxes", warning_mock.call_args.args[0].lower())
        self.assertIs(warning_mock.call_args.kwargs["parent"], window_like)
        info_mock.assert_not_called()

    def test_process_selected_bbox_segmentation_operation_auto_ensures_editor_without_enabling_annotation_mode(
        self,
    ) -> None:
        ensure_calls = []
        box = self._box(box_id="bbox_0001")
        editor = self._make_operation_editor(
            array=np.zeros((20, 30, 40), dtype=np.uint16),
            active_label=3,
            commit_result=None,
        )
        history_calls = []
        end_calls = []
        window_like = SimpleNamespace(
            bottom_panel=SimpleNamespace(
                selected_bounding_boxes=lambda: ("bbox_0001",),
                state=SimpleNamespace(bbox_selected_ids=("bbox_0001",)),
            ),
            _bbox_manager=SimpleNamespace(boxes=lambda: (box,)),
            _segmentation_editor=None,
            state=SimpleNamespace(annotation_mode_enabled=False),
            _record_global_history_for_segmentation_operation=lambda operation: history_calls.append(
                operation
            ),
            _end_annotation_modification=lambda: end_calls.append("end"),
        )

        def _ensure() -> bool:
            ensure_calls.append("ensure")
            window_like._segmentation_editor = editor
            return True

        window_like._ensure_editable_segmentation_for_annotation = _ensure

        with patch("src.ui.main_window.show_info") as info_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock:
            MainWindow._process_selected_bbox_segmentation_operation(window_like, "median_filter")

        self.assertEqual(ensure_calls, ["ensure"])
        self.assertFalse(window_like.state.annotation_mode_enabled)
        self.assertEqual(end_calls, ["end"])
        self.assertEqual(editor.begin_calls, ["median_filter_selected"])
        self.assertEqual(editor.commit_calls, 1)
        self.assertEqual(history_calls, [None])
        warning_mock.assert_not_called()
        info_mock.assert_called_once()
        info_text = info_mock.call_args.args[0]
        self.assertIn("Median Filter Selected processing is over.", info_text)
        self.assertIn("- selected bounding boxes: 1", info_text)
        self.assertIn("- changed voxels: 0", info_text)
        self.assertIs(info_mock.call_args.kwargs["parent"], window_like)

    def test_process_selected_bbox_segmentation_operation_warns_when_editor_cannot_be_ensured(self) -> None:
        ensure_calls = []
        box = self._box(box_id="bbox_0001")
        window_like = self._make_bbox_selection_window(
            selected_ids=("bbox_0001",),
            boxes=(box,),
            editor=None,
            ensure_result=False,
            ensure_calls=ensure_calls,
        )

        with patch("src.ui.main_window.show_info") as info_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock:
            MainWindow._process_selected_bbox_segmentation_operation(window_like, "erosion")

        self.assertEqual(ensure_calls, ["ensure"])
        warning_mock.assert_called_once()
        self.assertIn("no semantic or instance segmentation map", warning_mock.call_args.args[0].lower())
        self.assertIs(warning_mock.call_args.kwargs["parent"], window_like)
        info_mock.assert_not_called()

    def test_process_selected_bbox_segmentation_operation_applies_binary_delta_in_single_modification(
        self,
    ) -> None:
        box = self._box(box_id="bbox_0001", z0=1, z1=4, y0=2, y1=5, x0=3, x1=6)
        array = np.zeros((20, 30, 40), dtype=np.uint16)
        array[1, 2, 3] = 7
        committed_operation = SimpleNamespace(changed_voxels=2, operation_id=77)
        editor = self._make_operation_editor(
            array=array,
            active_label=11,
            commit_result=committed_operation,
        )
        history_calls = []
        end_calls = []
        refresh_calls = []
        window_like = SimpleNamespace(
            bottom_panel=SimpleNamespace(
                selected_bounding_boxes=lambda: ("bbox_0001",),
                state=SimpleNamespace(bbox_selected_ids=("bbox_0001",)),
            ),
            _bbox_manager=SimpleNamespace(boxes=lambda: (box,)),
            _segmentation_editor=editor,
            _ensure_editable_segmentation_for_annotation=lambda: True,
            _record_global_history_for_segmentation_operation=lambda operation: history_calls.append(
                operation
            ),
            _end_annotation_modification=lambda: end_calls.append("end"),
            _sync_renderer_segmentation_labels=lambda: refresh_calls.append("sync_labels"),
            _request_hover_readout=lambda: refresh_calls.append("hover"),
            _request_picked_readout=lambda: refresh_calls.append("picked"),
            render_all=lambda: refresh_calls.append("render"),
            _refresh_annotation_ui_state=lambda: refresh_calls.append("ui"),
        )
        after_mask = np.zeros((3, 3, 3), dtype=bool)
        after_mask[2, 2, 2] = True

        with patch.object(MainWindow, "_compute_selected_bbox_binary_operation", return_value=after_mask):
            with patch("src.ui.main_window.show_info") as info_mock, patch(
                "src.ui.main_window.show_warning"
            ) as warning_mock:
                MainWindow._process_selected_bbox_segmentation_operation(
                    window_like,
                    "dilation",
                )

        self.assertEqual(end_calls, ["end"])
        self.assertEqual(editor.begin_calls, ["dilation_selected"])
        self.assertEqual(editor.commit_calls, 1)
        self.assertEqual(editor.cancel_calls, 0)
        self.assertEqual(len(editor.erase_calls), 1)
        self.assertEqual(len(editor.assign_calls), 1)
        np.testing.assert_array_equal(
            editor.erase_calls[0][0],
            np.asarray([[1, 2, 3]], dtype=np.int64),
        )
        self.assertEqual(editor.erase_calls[0][1], "dilation_selected")
        self.assertFalse(editor.erase_calls[0][2])
        np.testing.assert_array_equal(
            editor.assign_calls[0][0],
            np.asarray([[3, 4, 5]], dtype=np.int64),
        )
        self.assertEqual(editor.assign_calls[0][1], 11)
        self.assertEqual(editor.assign_calls[0][2], "dilation_selected")
        self.assertFalse(editor.assign_calls[0][3])
        self.assertEqual(history_calls, [committed_operation])
        self.assertEqual(
            refresh_calls,
            ["sync_labels", "hover", "picked", "render", "ui"],
        )
        info_mock.assert_called_once()
        info_text = info_mock.call_args.args[0]
        self.assertIn("Dilation Selected processing is over.", info_text)
        self.assertIn("- selected bounding boxes: 1", info_text)
        self.assertIn("- changed voxels: 2", info_text)
        self.assertIs(info_mock.call_args.kwargs["parent"], window_like)
        warning_mock.assert_not_called()

    def test_process_selected_bbox_segmentation_operation_uses_fallback_changed_voxel_count_when_missing(
        self,
    ) -> None:
        box = self._box(box_id="bbox_0001", z0=1, z1=4, y0=2, y1=5, x0=3, x1=6)
        array = np.zeros((20, 30, 40), dtype=np.uint16)
        array[1, 2, 3] = 5
        committed_operation = SimpleNamespace(operation_id=99)
        editor = self._make_operation_editor(
            array=array,
            active_label=6,
            commit_result=committed_operation,
        )
        refresh_calls = []
        window_like = SimpleNamespace(
            bottom_panel=SimpleNamespace(
                selected_bounding_boxes=lambda: ("bbox_0001",),
                state=SimpleNamespace(bbox_selected_ids=("bbox_0001",)),
            ),
            _bbox_manager=SimpleNamespace(boxes=lambda: (box,)),
            _segmentation_editor=editor,
            _ensure_editable_segmentation_for_annotation=lambda: True,
            _record_global_history_for_segmentation_operation=lambda _operation: None,
            _end_annotation_modification=lambda: None,
            _sync_renderer_segmentation_labels=lambda: refresh_calls.append("sync_labels"),
            _request_hover_readout=lambda: refresh_calls.append("hover"),
            _request_picked_readout=lambda: refresh_calls.append("picked"),
            render_all=lambda: refresh_calls.append("render"),
            _refresh_annotation_ui_state=lambda: refresh_calls.append("ui"),
        )
        after_mask = np.zeros((3, 3, 3), dtype=bool)
        after_mask[2, 2, 2] = True

        with patch.object(MainWindow, "_compute_selected_bbox_binary_operation", return_value=after_mask):
            with patch("src.ui.main_window.show_info") as info_mock, patch(
                "src.ui.main_window.show_warning"
            ) as warning_mock:
                MainWindow._process_selected_bbox_segmentation_operation(
                    window_like,
                    "dilation",
                )

        self.assertEqual(
            refresh_calls,
            ["sync_labels", "hover", "picked", "render", "ui"],
        )
        warning_mock.assert_not_called()
        info_mock.assert_called_once()
        info_text = info_mock.call_args.args[0]
        self.assertIn("- changed voxels: 2", info_text)

    def test_process_selected_bbox_segmentation_operation_assigns_new_voxels_from_neighbor_labels(
        self,
    ) -> None:
        box = self._box(box_id="bbox_0001", z0=1, z1=4, y0=2, y1=5, x0=3, x1=6)
        array = np.zeros((20, 30, 40), dtype=np.uint16)
        array[2, 3, 3] = 1
        editor = self._make_operation_editor(
            array=array,
            active_label=2,
            commit_result=SimpleNamespace(changed_voxels=2, operation_id=101),
        )
        window_like = SimpleNamespace(
            bottom_panel=SimpleNamespace(
                selected_bounding_boxes=lambda: ("bbox_0001",),
                state=SimpleNamespace(bbox_selected_ids=("bbox_0001",)),
            ),
            _bbox_manager=SimpleNamespace(boxes=lambda: (box,)),
            _segmentation_editor=editor,
            _ensure_editable_segmentation_for_annotation=lambda: True,
            _record_global_history_for_segmentation_operation=lambda _operation: None,
            _end_annotation_modification=lambda: None,
            _refresh_annotation_ui_state=lambda: None,
        )
        after_mask = np.zeros((3, 3, 3), dtype=bool)
        after_mask[1, 1, 1] = True

        with patch.object(MainWindow, "_compute_selected_bbox_binary_operation", return_value=after_mask):
            with patch("src.ui.main_window.show_info") as info_mock, patch(
                "src.ui.main_window.show_warning"
            ) as warning_mock:
                MainWindow._process_selected_bbox_segmentation_operation(
                    window_like,
                    "dilation",
                )

        self.assertEqual(len(editor.assign_calls), 1)
        np.testing.assert_array_equal(
            editor.assign_calls[0][0],
            np.asarray([[2, 3, 4]], dtype=np.int64),
        )
        self.assertEqual(editor.assign_calls[0][1], 1)
        warning_mock.assert_not_called()
        info_mock.assert_called_once()

    def test_process_selected_bbox_segmentation_operation_never_erases_outside_union_domain(
        self,
    ) -> None:
        box1 = self._box(box_id="bbox_0001", z0=1, z1=3, y0=2, y1=4, x0=3, x1=5)
        box2 = self._box(box_id="bbox_0002", z0=1, z1=3, y0=2, y1=4, x0=7, x1=9)
        array = np.zeros((20, 30, 40), dtype=np.uint16)
        array[1, 2, 3] = 8  # inside selected-union domain
        array[1, 2, 5] = 8  # in the gap between boxes (outside selected union)
        editor = self._make_operation_editor(
            array=array,
            active_label=4,
            commit_result=SimpleNamespace(changed_voxels=1, operation_id=104),
        )
        window_like = SimpleNamespace(
            bottom_panel=SimpleNamespace(
                selected_bounding_boxes=lambda: ("bbox_0001", "bbox_0002"),
                state=SimpleNamespace(bbox_selected_ids=("bbox_0001", "bbox_0002")),
            ),
            _bbox_manager=SimpleNamespace(boxes=lambda: (box1, box2)),
            _segmentation_editor=editor,
            _ensure_editable_segmentation_for_annotation=lambda: True,
            _record_global_history_for_segmentation_operation=lambda _operation: None,
            _end_annotation_modification=lambda: None,
            _refresh_annotation_ui_state=lambda: None,
        )
        after_mask = np.zeros((2, 2, 6), dtype=bool)

        with patch.object(
            MainWindow,
            "_compute_selected_bbox_binary_operation_with_halo_context",
            return_value=after_mask,
        ):
            with patch("src.ui.main_window.show_info") as info_mock, patch(
                "src.ui.main_window.show_warning"
            ) as warning_mock:
                MainWindow._process_selected_bbox_segmentation_operation(
                    window_like,
                    "erosion",
                )

        self.assertEqual(len(editor.erase_calls), 1)
        erased_coordinates = editor.erase_calls[0][0]
        np.testing.assert_array_equal(
            erased_coordinates,
            np.asarray([[1, 2, 3]], dtype=np.int64),
        )
        self.assertEqual(len(editor.assign_calls), 0)
        warning_mock.assert_not_called()
        info_mock.assert_called_once()

    def test_process_selected_bbox_segmentation_operation_never_assigns_outside_union_domain(
        self,
    ) -> None:
        box1 = self._box(box_id="bbox_0001", z0=1, z1=3, y0=2, y1=4, x0=3, x1=5)
        box2 = self._box(box_id="bbox_0002", z0=1, z1=3, y0=2, y1=4, x0=7, x1=9)
        editor = self._make_operation_editor(
            array=np.zeros((20, 30, 40), dtype=np.uint16),
            active_label=4,
            commit_result=SimpleNamespace(changed_voxels=16, operation_id=105),
        )
        window_like = SimpleNamespace(
            bottom_panel=SimpleNamespace(
                selected_bounding_boxes=lambda: ("bbox_0001", "bbox_0002"),
                state=SimpleNamespace(bbox_selected_ids=("bbox_0001", "bbox_0002")),
            ),
            _bbox_manager=SimpleNamespace(boxes=lambda: (box1, box2)),
            _segmentation_editor=editor,
            _ensure_editable_segmentation_for_annotation=lambda: True,
            _record_global_history_for_segmentation_operation=lambda _operation: None,
            _end_annotation_modification=lambda: None,
            _refresh_annotation_ui_state=lambda: None,
        )
        after_mask = np.ones((2, 2, 6), dtype=bool)

        with patch.object(
            MainWindow,
            "_compute_selected_bbox_binary_operation_with_halo_context",
            return_value=after_mask,
        ):
            with patch("src.ui.main_window.show_info") as info_mock, patch(
                "src.ui.main_window.show_warning"
            ) as warning_mock:
                MainWindow._process_selected_bbox_segmentation_operation(
                    window_like,
                    "dilation",
                )

        self.assertEqual(len(editor.erase_calls), 0)
        self.assertEqual(len(editor.assign_calls), 1)
        assigned_coordinates = editor.assign_calls[0][0]
        self.assertEqual(int(assigned_coordinates.shape[0]), 16)
        self.assertEqual(editor.assign_calls[0][1], 4)
        for coordinate in assigned_coordinates:
            z = int(coordinate[0])
            y = int(coordinate[1])
            x = int(coordinate[2])
            in_box1 = 1 <= z < 3 and 2 <= y < 4 and 3 <= x < 5
            in_box2 = 1 <= z < 3 and 2 <= y < 4 and 7 <= x < 9
            self.assertTrue(
                in_box1 or in_box2,
                msg=f"Assigned coordinate outside selected union: {(z, y, x)}",
            )
        warning_mock.assert_not_called()
        info_mock.assert_called_once()

    def test_process_selected_bbox_segmentation_operation_only_refreshes_annotation_ui_when_unchanged(
        self,
    ) -> None:
        box = self._box(box_id="bbox_0001", z0=1, z1=4, y0=2, y1=5, x0=3, x1=6)
        editor = self._make_operation_editor(
            array=np.zeros((20, 30, 40), dtype=np.uint16),
            active_label=9,
            commit_result=SimpleNamespace(changed_voxels=0, operation_id=100),
        )
        refresh_calls = []
        window_like = SimpleNamespace(
            bottom_panel=SimpleNamespace(
                selected_bounding_boxes=lambda: ("bbox_0001",),
                state=SimpleNamespace(bbox_selected_ids=("bbox_0001",)),
            ),
            _bbox_manager=SimpleNamespace(boxes=lambda: (box,)),
            _segmentation_editor=editor,
            _ensure_editable_segmentation_for_annotation=lambda: True,
            _record_global_history_for_segmentation_operation=lambda _operation: None,
            _end_annotation_modification=lambda: None,
            _sync_renderer_segmentation_labels=lambda: refresh_calls.append("sync_labels"),
            _request_hover_readout=lambda: refresh_calls.append("hover"),
            _request_picked_readout=lambda: refresh_calls.append("picked"),
            render_all=lambda: refresh_calls.append("render"),
            _refresh_annotation_ui_state=lambda: refresh_calls.append("ui"),
        )
        after_mask = np.zeros((3, 3, 3), dtype=bool)

        with patch.object(MainWindow, "_compute_selected_bbox_binary_operation", return_value=after_mask):
            with patch("src.ui.main_window.show_info") as info_mock, patch(
                "src.ui.main_window.show_warning"
            ) as warning_mock:
                MainWindow._process_selected_bbox_segmentation_operation(
                    window_like,
                    "median_filter",
                )

        self.assertEqual(refresh_calls, ["ui"])
        warning_mock.assert_not_called()
        info_mock.assert_called_once()
        info_text = info_mock.call_args.args[0]
        self.assertIn("- changed voxels: 0", info_text)

    def test_process_selected_bbox_segmentation_operation_cancels_and_warns_on_editor_error(
        self,
    ) -> None:
        box = self._box(box_id="bbox_0001", z0=1, z1=4, y0=2, y1=5, x0=3, x1=6)
        array = np.zeros((20, 30, 40), dtype=np.uint16)
        editor = self._make_operation_editor(
            array=array,
            active_label=2,
            commit_result=SimpleNamespace(changed_voxels=1, operation_id=13),
            raise_on_assign=True,
        )
        history_calls = []
        window_like = SimpleNamespace(
            bottom_panel=SimpleNamespace(
                selected_bounding_boxes=lambda: ("bbox_0001",),
                state=SimpleNamespace(bbox_selected_ids=("bbox_0001",)),
            ),
            _bbox_manager=SimpleNamespace(boxes=lambda: (box,)),
            _segmentation_editor=editor,
            _ensure_editable_segmentation_for_annotation=lambda: True,
            _record_global_history_for_segmentation_operation=lambda operation: history_calls.append(
                operation
            ),
            _end_annotation_modification=lambda: None,
        )
        after_mask = np.zeros((3, 3, 3), dtype=bool)
        after_mask[2, 2, 2] = True

        with patch.object(MainWindow, "_compute_selected_bbox_binary_operation", return_value=after_mask):
            with patch("src.ui.main_window.show_info") as info_mock, patch(
                "src.ui.main_window.show_warning"
            ) as warning_mock:
                MainWindow._process_selected_bbox_segmentation_operation(
                    window_like,
                    "median_filter",
                )

        self.assertEqual(editor.begin_calls, ["median_filter_selected"])
        self.assertEqual(editor.cancel_calls, 1)
        self.assertEqual(editor.commit_calls, 0)
        self.assertEqual(history_calls, [])
        warning_mock.assert_called_once()
        self.assertIn("assign boom", warning_mock.call_args.args[0].lower())
        self.assertIs(warning_mock.call_args.kwargs["parent"], window_like)
        info_mock.assert_not_called()

    def test_process_selected_bbox_segmentation_operation_warns_when_editor_remains_missing_after_ensure(
        self,
    ) -> None:
        ensure_calls = []
        box = self._box(box_id="bbox_0001")
        window_like = SimpleNamespace(
            bottom_panel=SimpleNamespace(
                selected_bounding_boxes=lambda: ("bbox_0001",),
                state=SimpleNamespace(bbox_selected_ids=("bbox_0001",)),
            ),
            _bbox_manager=SimpleNamespace(boxes=lambda: (box,)),
            _segmentation_editor=None,
            _ensure_editable_segmentation_for_annotation=lambda: ensure_calls.append("ensure")
            or True,
        )

        with patch("src.ui.main_window.show_info") as info_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock:
            MainWindow._process_selected_bbox_segmentation_operation(window_like, "dilation")

        self.assertEqual(ensure_calls, ["ensure"])
        warning_mock.assert_called_once()
        self.assertIn("no semantic or instance segmentation map", warning_mock.call_args.args[0].lower())
        self.assertIs(warning_mock.call_args.kwargs["parent"], window_like)
        info_mock.assert_not_called()

    def test_process_selected_bbox_segmentation_operation_warns_when_selected_boxes_are_missing(self) -> None:
        window_like = SimpleNamespace(
            bottom_panel=SimpleNamespace(
                selected_bounding_boxes=lambda: ("bbox_0007",),
                state=SimpleNamespace(bbox_selected_ids=("bbox_0007",)),
            ),
            _bbox_manager=SimpleNamespace(boxes=lambda: tuple()),
            _segmentation_editor=object(),
            _ensure_editable_segmentation_for_annotation=lambda: True,
        )

        with patch("src.ui.main_window.show_info") as info_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock:
            MainWindow._process_selected_bbox_segmentation_operation(window_like, "dilation")

        warning_mock.assert_called_once()
        self.assertIn("no longer available", warning_mock.call_args.args[0].lower())
        self.assertIs(warning_mock.call_args.kwargs["parent"], window_like)
        info_mock.assert_not_called()

    def test_resolve_selected_bounding_boxes_filters_missing_and_preserves_order(self) -> None:
        box1 = self._box(box_id="bbox_0001")
        box2 = self._box(box_id="bbox_0002", z0=6, z1=9, y0=7, y1=10, x0=8, x1=12)
        window_like = SimpleNamespace(
            _bbox_manager=SimpleNamespace(boxes=lambda: (box1, box2)),
        )

        resolved = MainWindow._resolve_selected_bounding_boxes(
            window_like,
            ("bbox_0002", "missing", "bbox_0001", "bbox_0002"),
        )

        self.assertEqual(tuple(box.id for box in resolved), ("bbox_0002", "bbox_0001"))

    def test_process_selected_bbox_segmentation_operation_uses_panel_state_fallback_ids(self) -> None:
        box = self._box(box_id="bbox_0001")
        editor = self._make_operation_editor(
            array=np.zeros((20, 30, 40), dtype=np.uint16),
            active_label=4,
            commit_result=None,
        )
        window_like = SimpleNamespace(
            bottom_panel=SimpleNamespace(
                state=SimpleNamespace(bbox_selected_ids=("bbox_0001",)),
            ),
            _bbox_manager=SimpleNamespace(boxes=lambda: (box,)),
            _segmentation_editor=editor,
            _ensure_editable_segmentation_for_annotation=lambda: True,
            _record_global_history_for_segmentation_operation=lambda _operation: None,
            _end_annotation_modification=lambda: None,
            _refresh_annotation_ui_state=lambda: None,
        )

        with patch("src.ui.main_window.show_info") as info_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock:
            MainWindow._process_selected_bbox_segmentation_operation(window_like, "erosion")

        self.assertEqual(editor.begin_calls, ["erosion_selected"])
        self.assertEqual(editor.commit_calls, 1)
        warning_mock.assert_not_called()
        info_mock.assert_called_once()
        self.assertIn("Erosion Selected processing is over.", info_mock.call_args.args[0])

    def test_process_selected_bbox_segmentation_operation_keeps_label_vote_inside_core_union(
        self,
    ) -> None:
        box = self._box(box_id="bbox_0001", z0=1, z1=4, y0=2, y1=5, x0=3, x1=6)
        array = np.zeros((20, 30, 40), dtype=np.uint16)
        # Non-zero label adjacent to the core region but outside the selected union.
        array[2, 3, 6] = 9
        editor = self._make_operation_editor(
            array=array,
            active_label=4,
            commit_result=SimpleNamespace(changed_voxels=1, operation_id=102),
        )
        window_like = SimpleNamespace(
            bottom_panel=SimpleNamespace(
                selected_bounding_boxes=lambda: ("bbox_0001",),
                state=SimpleNamespace(bbox_selected_ids=("bbox_0001",)),
            ),
            _bbox_manager=SimpleNamespace(boxes=lambda: (box,)),
            _segmentation_editor=editor,
            _ensure_editable_segmentation_for_annotation=lambda: True,
            _record_global_history_for_segmentation_operation=lambda _operation: None,
            _end_annotation_modification=lambda: None,
            _refresh_annotation_ui_state=lambda: None,
        )
        after_mask = np.zeros((3, 3, 3), dtype=bool)
        after_mask[1, 1, 2] = True

        with patch.object(
            MainWindow,
            "_compute_selected_bbox_binary_operation_with_halo_context",
            return_value=after_mask,
        ):
            with patch("src.ui.main_window.show_info") as info_mock, patch(
                "src.ui.main_window.show_warning"
            ) as warning_mock:
                MainWindow._process_selected_bbox_segmentation_operation(
                    window_like,
                    "dilation",
                )

        self.assertEqual(len(editor.assign_calls), 1)
        np.testing.assert_array_equal(
            editor.assign_calls[0][0],
            np.asarray([[2, 3, 5]], dtype=np.int64),
        )
        self.assertEqual(editor.assign_calls[0][1], 4)
        warning_mock.assert_not_called()
        info_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
