from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from src.ui.main_window import MainWindow
except Exception:  # pragma: no cover - environment dependent
    MainWindow = None  # type: ignore[assignment]

from src.bbox import BoundingBox


class _FakeBBoxManager:
    def __init__(
        self,
        boxes: tuple[BoundingBox, ...],
        *,
        volume_shape: tuple[int, int, int] = (32, 32, 32),
    ) -> None:
        self._boxes = tuple(boxes)
        self.volume_shape = tuple(int(axis) for axis in volume_shape)

    def boxes(self) -> tuple[BoundingBox, ...]:
        return self._boxes


class _FakeVolume:
    def __init__(self, array: np.ndarray) -> None:
        self._array = np.asarray(array)

    def get_chunk(self, zyx_slices: tuple[slice, slice, slice]) -> np.ndarray:
        return np.asarray(self._array[zyx_slices])


class _FakeSparseVolume:
    def __init__(self, dtype: np.dtype = np.dtype(np.float32)) -> None:
        self.dtype = np.dtype(dtype)

    def get_chunk(self, _zyx_slices: tuple[slice, slice, slice]) -> np.ndarray:
        return np.zeros((1, 1, 1), dtype=self.dtype)


class _FakeSignal:
    def __init__(self) -> None:
        self.connected: list[object] = []

    def connect(self, callback: object) -> None:
        self.connected.append(callback)


class _FakeEditor:
    def __init__(self, shape: tuple[int, int, int]) -> None:
        self.kind = "semantic"
        self.dtype = np.dtype(np.uint16)
        self._array = np.zeros(shape, dtype=np.uint16)

    def begin_modification(self, _name: str) -> None:
        return None

    def commit_modification(self):
        return None

    def array_view(self) -> np.ndarray:
        return self._array

    def assign(
        self,
        coordinates: object,
        *,
        label: int,
        operation_name: str,
        ignore_out_of_bounds: bool,
    ):
        del operation_name, ignore_out_of_bounds
        coords = np.asarray(coordinates, dtype=np.int64)
        if coords.size <= 0:
            return None
        self._array[coords[:, 0], coords[:, 1], coords[:, 2]] = int(label)
        return None


class _FakeTorchTensor:
    def __init__(self, array: np.ndarray) -> None:
        self._array = np.asarray(array)
        self.device = SimpleNamespace(type="cpu")

    def to(self, *_args, **_kwargs):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def __array__(self, dtype=None):
        if dtype is None:
            return np.asarray(self._array)
        return np.asarray(self._array, dtype=dtype)


class _FakeTorchNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False


class _FakeTorchAutocast(_FakeTorchNoGrad):
    pass


class _FakeTorchModule:
    Tensor = _FakeTorchTensor
    float16 = object()
    int16 = object()
    bfloat16 = object()
    cuda = SimpleNamespace(
        is_available=lambda: False,
        device_count=lambda: 0,
    )

    @staticmethod
    def device(_spec: str):
        return SimpleNamespace(type="cpu")

    @staticmethod
    def from_numpy(array: np.ndarray) -> _FakeTorchTensor:
        return _FakeTorchTensor(np.asarray(array))

    @staticmethod
    def zeros(shape: tuple[int, ...], dtype=None):
        del dtype
        return _FakeTorchTensor(np.zeros(shape, dtype=np.int16))

    @staticmethod
    def no_grad() -> _FakeTorchNoGrad:
        return _FakeTorchNoGrad()

    @staticmethod
    def autocast(*, device_type: str, enabled: bool, dtype):
        del device_type, enabled, dtype
        return _FakeTorchAutocast()


class _FakeModel:
    def __init__(self) -> None:
        self.training = False

    def eval(self) -> None:
        self.training = False

    def train(self) -> None:
        self.training = True

    def parameters(self):
        return iter(())

    def __call__(self, minivols):
        del minivols
        return _FakeTorchTensor(np.zeros((1, 1, 1, 1, 1), dtype=np.float32))


@unittest.skipUnless(MainWindow is not None, "MainWindow is not available")
class MainWindowSegmentInferenceFlowTests(unittest.TestCase):
    def _make_box(
        self,
        *,
        box_id: str,
        label: str,
        z0: int,
        z1: int,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        volume_shape: tuple[int, int, int] = (32, 32, 32),
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
            volume_shape=volume_shape,
        )

    def _make_window_like(
        self,
        *,
        boxes: tuple[BoundingBox, ...],
        active_segmentation: tuple[str, object] | None,
        semantic_volume: object | None,
        raw_volume: object | None,
        volume_shape: tuple[int, int, int] = (32, 32, 32),
        ensure_semantic_result: bool = True,
    ) -> SimpleNamespace:
        ordered_rows = tuple(SimpleNamespace(box_id=box.id) for box in boxes)
        ensure_calls: list[str] = []
        editor = _FakeEditor((32, 32, 32))

        def _ensure_semantic() -> bool:
            ensure_calls.append("called")
            if ensure_semantic_result and window_like._semantic_volume is None:
                window_like._semantic_volume = _FakeVolume(
                    np.zeros((32, 32, 32), dtype=np.int16)
                )
            if ensure_semantic_result and window_like._segmentation_editor is None:
                window_like._segmentation_editor = editor
            return bool(ensure_semantic_result)

        def _start_background_inline_compat(**kwargs: object) -> bool:
            return MainWindow._run_learning_inference_inline_compat(
                window_like,
                model_runtime=kwargs["model_runtime"],
                inference_boxes=kwargs["inference_boxes"],
                raw_array=kwargs["raw_array"],
                label_values=kwargs["label_values"],
                volume_shape=kwargs["volume_shape"],
            )

        window_like = SimpleNamespace(
            bottom_panel=SimpleNamespace(state=SimpleNamespace(bbox_rows=ordered_rows)),
            _bbox_manager=_FakeBBoxManager(boxes, volume_shape=volume_shape),
            _active_segmentation_volume=lambda: active_segmentation,
            _semantic_volume=semantic_volume,
            _raw_volume=raw_volume,
            _annotation_kind="instance",
            _segmentation_editor=editor if semantic_volume is not None else None,
            _annotation_labels_dirty=False,
            _ensure_editable_segmentation_for_annotation=_ensure_semantic,
            _end_annotation_modification=lambda: None,
            _record_global_history_for_segmentation_operation=lambda _operation: None,
            _sync_renderer_segmentation_labels=lambda: None,
            _request_hover_readout=lambda: None,
            _request_picked_readout=lambda: None,
            render_all=lambda: None,
            _refresh_annotation_ui_state=lambda: None,
            _show_inference_navigation_only_notice=lambda: None,
            _start_learning_inference_background=_start_background_inline_compat,
            _exit_learning_inference_running_state=lambda: None,
            _ensure_calls=ensure_calls,
        )
        return window_like

    def _make_model_runtime(
        self,
        *,
        label_values: tuple[int, ...] = (0, 1),
        model: object | None = None,
        device_ids: tuple[int, ...] = tuple(),
    ) -> SimpleNamespace:
        return SimpleNamespace(
            model=_FakeModel() if model is None else model,
            device_ids=tuple(device_ids),
            num_classes=len(tuple(label_values)),
            hyperparameters={"label_values": tuple(label_values)},
        )

    def test_segment_inference_aborts_when_model_is_missing(self) -> None:
        inference_box = self._make_box(
            box_id="bbox_0001",
            label="inference",
            z0=1,
            z1=6,
            y0=1,
            y1=6,
            x0=1,
            x1=6,
        )
        window_like = self._make_window_like(
            boxes=(inference_box,),
            active_segmentation=("semantic", object()),
            semantic_volume=object(),
            raw_volume=object(),
        )

        with patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=None,
        ), patch("src.ui.main_window.show_warning") as warning_mock:
            result = MainWindow._segment_inference_bboxes_with_dialog(window_like)

        self.assertFalse(result)
        warning_mock.assert_called_once()
        self.assertIn("load a model", warning_mock.call_args.args[0].lower())

    def test_segment_inference_aborts_when_no_inference_bbox_exists(self) -> None:
        train_box = self._make_box(
            box_id="bbox_0001",
            label="train",
            z0=1,
            z1=6,
            y0=1,
            y1=6,
            x0=1,
            x1=6,
        )
        window_like = self._make_window_like(
            boxes=(train_box,),
            active_segmentation=("semantic", object()),
            semantic_volume=object(),
            raw_volume=object(),
        )

        with patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=object(),
        ), patch("src.ui.main_window.show_warning") as warning_mock:
            result = MainWindow._segment_inference_bboxes_with_dialog(window_like)

        self.assertFalse(result)
        warning_mock.assert_called_once()
        self.assertIn("labeled 'inference'", warning_mock.call_args.args[0])

    def test_segment_inference_aborts_when_model_metadata_label_values_are_missing(self) -> None:
        inference_box = self._make_box(
            box_id="bbox_0001",
            label="inference",
            z0=1,
            z1=6,
            y0=1,
            y1=6,
            x0=1,
            x1=6,
        )
        window_like = self._make_window_like(
            boxes=(inference_box,),
            active_segmentation=("semantic", object()),
            semantic_volume=object(),
            raw_volume=object(),
        )

        with patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=SimpleNamespace(hyperparameters={}, num_classes=2),
        ), patch("src.ui.main_window.show_warning") as warning_mock:
            result = MainWindow._segment_inference_bboxes_with_dialog(window_like)

        self.assertFalse(result)
        warning_mock.assert_called_once()
        warning_text = warning_mock.call_args.args[0].lower()
        self.assertIn("inference requires class label_values", warning_text)
        self.assertIn("loaded model metadata", warning_text)

    def test_segment_inference_aborts_when_model_label_values_do_not_match_num_classes(self) -> None:
        inference_box = self._make_box(
            box_id="bbox_0001",
            label="inference",
            z0=1,
            z1=6,
            y0=1,
            y1=6,
            x0=1,
            x1=6,
        )
        window_like = self._make_window_like(
            boxes=(inference_box,),
            active_segmentation=("semantic", object()),
            semantic_volume=object(),
            raw_volume=object(),
        )

        with patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=SimpleNamespace(
                hyperparameters={"label_values": (0, 1)},
                num_classes=3,
            ),
        ), patch("src.ui.main_window.show_warning") as warning_mock:
            result = MainWindow._segment_inference_bboxes_with_dialog(window_like)

        self.assertFalse(result)
        warning_mock.assert_called_once()
        self.assertIn("label_values length does not match", warning_mock.call_args.args[0])

    def test_segment_inference_blocks_instance_map_without_semantic(self) -> None:
        inference_box = self._make_box(
            box_id="bbox_0001",
            label="inference",
            z0=1,
            z1=6,
            y0=1,
            y1=6,
            x0=1,
            x1=6,
        )
        window_like = self._make_window_like(
            boxes=(inference_box,),
            active_segmentation=("instance", object()),
            semantic_volume=None,
            raw_volume=object(),
        )
        eval_runtimes = {
            "bbox_1001": SimpleNamespace(buffer=SimpleNamespace(label_values=(0, 1, 2))),
        }

        with patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=self._make_model_runtime(label_values=(0, 1, 2)),
        ), patch(
            "src.ui.main_window.get_current_learning_eval_runtimes_by_box_id",
            return_value=eval_runtimes,
        ), patch("src.ui.main_window.show_warning") as warning_mock:
            result = MainWindow._segment_inference_bboxes_with_dialog(window_like)

        self.assertFalse(result)
        warning_mock.assert_called_once()
        self.assertIn("active map is instance", warning_mock.call_args.args[0].lower())
        self.assertEqual(window_like._ensure_calls, [])

    def test_segment_inference_auto_creates_empty_semantic_when_none_loaded(self) -> None:
        inference_box = self._make_box(
            box_id="bbox_0001",
            label="inference",
            z0=1,
            z1=6,
            y0=1,
            y1=6,
            x0=1,
            x1=6,
        )
        window_like = self._make_window_like(
            boxes=(inference_box,),
            active_segmentation=None,
            semantic_volume=None,
            raw_volume=object(),
            ensure_semantic_result=True,
        )
        eval_runtimes = {
            "bbox_1001": SimpleNamespace(buffer=SimpleNamespace(label_values=(0, 1, 2))),
        }

        with patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=self._make_model_runtime(label_values=(0, 1, 2)),
        ), patch(
            "src.ui.main_window.get_current_learning_eval_runtimes_by_box_id",
            return_value=eval_runtimes,
        ), patch("src.ui.main_window.show_warning") as warning_mock:
            result = MainWindow._segment_inference_bboxes_with_dialog(window_like)

        self.assertFalse(result)
        warning_mock.assert_called_once()
        self.assertEqual(window_like._ensure_calls, ["called"])
        self.assertEqual(window_like._annotation_kind, "semantic")

    def test_segment_inference_aborts_when_inference_boxes_overlap(self) -> None:
        inference_box_a = self._make_box(
            box_id="bbox_0001",
            label="inference",
            z0=1,
            z1=10,
            y0=1,
            y1=10,
            x0=1,
            x1=10,
        )
        inference_box_b = self._make_box(
            box_id="bbox_0002",
            label="inference",
            z0=5,
            z1=14,
            y0=5,
            y1=14,
            x0=5,
            x1=14,
        )
        window_like = self._make_window_like(
            boxes=(inference_box_a, inference_box_b),
            active_segmentation=("semantic", object()),
            semantic_volume=object(),
            raw_volume=object(),
        )

        with patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=object(),
        ), patch("src.ui.main_window.show_warning") as warning_mock:
            result = MainWindow._segment_inference_bboxes_with_dialog(window_like)

        self.assertFalse(result)
        warning_mock.assert_called_once()
        warning_text = warning_mock.call_args.args[0]
        self.assertIn("overlap", warning_text.lower())
        self.assertIn("bbox_0001", warning_text)
        self.assertIn("bbox_0002", warning_text)

    def test_segment_inference_prompts_overwrite_when_inference_bbox_is_non_empty(self) -> None:
        inference_box = self._make_box(
            box_id="bbox_0001",
            label="inference",
            z0=2,
            z1=5,
            y0=2,
            y1=5,
            x0=2,
            x1=5,
        )
        semantic_array = np.zeros((16, 16, 16), dtype=np.int16)
        semantic_array[3, 3, 3] = 7
        window_like = self._make_window_like(
            boxes=(inference_box,),
            active_segmentation=("semantic", object()),
            semantic_volume=_FakeVolume(semantic_array),
            raw_volume=object(),
        )
        eval_runtimes = {
            "bbox_1001": SimpleNamespace(buffer=SimpleNamespace(label_values=(0, 1, 2))),
        }

        with patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=self._make_model_runtime(label_values=(0, 1, 2)),
        ), patch(
            "src.ui.main_window.get_current_learning_eval_runtimes_by_box_id",
            return_value=eval_runtimes,
        ), patch(
            "src.ui.main_window.confirm_replace_inference_bboxes",
            return_value=False,
        ) as confirm_mock, patch("src.ui.main_window.show_warning") as warning_mock:
            result = MainWindow._segment_inference_bboxes_with_dialog(window_like)

        self.assertFalse(result)
        confirm_mock.assert_called_once_with(parent=window_like)
        warning_mock.assert_not_called()

    def test_segment_inference_skips_overwrite_prompt_when_inference_bbox_is_empty(self) -> None:
        inference_box = self._make_box(
            box_id="bbox_0001",
            label="inference",
            z0=2,
            z1=5,
            y0=2,
            y1=5,
            x0=2,
            x1=5,
        )
        semantic_array = np.zeros((16, 16, 16), dtype=np.int16)
        semantic_array[3, 3, 3] = -100
        window_like = self._make_window_like(
            boxes=(inference_box,),
            active_segmentation=("semantic", object()),
            semantic_volume=_FakeVolume(semantic_array),
            raw_volume=object(),
        )
        eval_runtimes = {
            "bbox_1001": SimpleNamespace(buffer=SimpleNamespace(label_values=(0, 1, 2))),
        }

        with patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=self._make_model_runtime(label_values=(0, 1, 2)),
        ), patch(
            "src.ui.main_window.get_current_learning_eval_runtimes_by_box_id",
            return_value=eval_runtimes,
        ), patch(
            "src.ui.main_window.confirm_replace_inference_bboxes",
            return_value=True,
        ) as confirm_mock, patch("src.ui.main_window.show_warning") as warning_mock:
            result = MainWindow._segment_inference_bboxes_with_dialog(window_like)

        self.assertFalse(result)
        confirm_mock.assert_not_called()
        warning_mock.assert_called_once()

    def test_segment_inference_reports_all_success_with_info(self) -> None:
        inference_box = self._make_box(
            box_id="bbox_0001",
            label="inference",
            z0=1,
            z1=4,
            y0=1,
            y1=4,
            x0=1,
            x1=4,
        )
        window_like = self._make_window_like(
            boxes=(inference_box,),
            active_segmentation=("semantic", object()),
            semantic_volume=_FakeVolume(np.zeros((32, 32, 32), dtype=np.int16)),
            raw_volume=_FakeVolume(np.zeros((32, 32, 32), dtype=np.float32)),
        )
        eval_runtimes = {
            "bbox_1001": SimpleNamespace(buffer=SimpleNamespace(label_values=(0, 1))),
        }
        fake_runtime = SimpleNamespace(
            dataloader=tuple(),
            buffer=SimpleNamespace(
                add_batch=lambda _batch, _coordinates: None,
                get_pred_labels=lambda: np.ones((3, 3, 3), dtype=np.int64),
            ),
        )
        fake_plan = SimpleNamespace(
            z=SimpleNamespace(extend_before=0, original_size=3),
            y=SimpleNamespace(extend_before=0, original_size=3),
            x=SimpleNamespace(extend_before=0, original_size=3),
        )

        with patch.dict(sys.modules, {"torch": _FakeTorchModule}), patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=self._make_model_runtime(model=_FakeModel()),
        ), patch(
            "src.ui.main_window.get_current_learning_eval_runtimes_by_box_id",
            return_value=eval_runtimes,
        ), patch(
            "src.learning.qt_workers._extract_bbox_context_from_array",
            return_value=np.zeros((3, 3, 3), dtype=np.float32),
        ), patch(
            "src.learning.qt_workers._build_inference_dataloader_runtime_from_entry",
            return_value=fake_runtime,
        ), patch(
            "src.learning.qt_workers._dispose_inference_runtime",
            return_value=tuple(),
        ), patch(
            "src.learning.qt_workers._plan_bbox_context",
            return_value=fake_plan,
        ), patch("src.ui.main_window.show_info") as info_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock:
            result = MainWindow._segment_inference_bboxes_with_dialog(window_like)

        self.assertTrue(result)
        warning_mock.assert_not_called()
        info_mock.assert_called_once()
        self.assertIn("all inference bboxes succeeded", info_mock.call_args.args[0].lower())

    def test_segment_inference_does_not_call_selected_bbox_halo_processing_helper(self) -> None:
        inference_box = self._make_box(
            box_id="bbox_0001",
            label="inference",
            z0=1,
            z1=4,
            y0=1,
            y1=4,
            x0=1,
            x1=4,
        )
        window_like = self._make_window_like(
            boxes=(inference_box,),
            active_segmentation=("semantic", object()),
            semantic_volume=_FakeVolume(np.zeros((32, 32, 32), dtype=np.int16)),
            raw_volume=_FakeVolume(np.zeros((32, 32, 32), dtype=np.float32)),
        )
        eval_runtimes = {
            "bbox_1001": SimpleNamespace(buffer=SimpleNamespace(label_values=(0, 1))),
        }
        fake_runtime = SimpleNamespace(
            dataloader=tuple(),
            buffer=SimpleNamespace(
                add_batch=lambda _batch, _coordinates: None,
                get_pred_labels=lambda: np.ones((3, 3, 3), dtype=np.int64),
            ),
        )
        fake_plan = SimpleNamespace(
            z=SimpleNamespace(extend_before=0, original_size=3),
            y=SimpleNamespace(extend_before=0, original_size=3),
            x=SimpleNamespace(extend_before=0, original_size=3),
        )

        with patch.dict(sys.modules, {"torch": _FakeTorchModule}), patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=self._make_model_runtime(model=_FakeModel()),
        ), patch(
            "src.ui.main_window.get_current_learning_eval_runtimes_by_box_id",
            return_value=eval_runtimes,
        ), patch(
            "src.learning.qt_workers._extract_bbox_context_from_array",
            return_value=np.zeros((3, 3, 3), dtype=np.float32),
        ), patch(
            "src.learning.qt_workers._build_inference_dataloader_runtime_from_entry",
            return_value=fake_runtime,
        ), patch(
            "src.learning.qt_workers._dispose_inference_runtime",
            return_value=tuple(),
        ), patch(
            "src.learning.qt_workers._plan_bbox_context",
            return_value=fake_plan,
        ), patch.object(
            MainWindow,
            "_compute_selected_bbox_binary_operation_with_halo_context",
            side_effect=AssertionError(
                "Selected-bbox halo helper must never be called in inference flow."
            ),
        ) as selected_helper_mock, patch(
            "src.ui.main_window.bbox_ops.compute_selected_bbox_binary_operation_with_halo_context",
            side_effect=AssertionError(
                "Extracted selected-bbox halo helper must never be called in inference flow."
            ),
        ) as extracted_selected_helper_mock, patch("src.ui.main_window.show_info") as info_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock:
            result = MainWindow._segment_inference_bboxes_with_dialog(window_like)

        self.assertTrue(result)
        selected_helper_mock.assert_not_called()
        extracted_selected_helper_mock.assert_not_called()
        warning_mock.assert_not_called()
        info_mock.assert_called_once()
        self.assertIn("all inference bboxes succeeded", info_mock.call_args.args[0].lower())

    def test_segment_inference_reports_partial_success_when_one_bbox_fails(self) -> None:
        inference_box_a = self._make_box(
            box_id="bbox_0001",
            label="inference",
            z0=1,
            z1=4,
            y0=1,
            y1=4,
            x0=1,
            x1=4,
        )
        inference_box_b = self._make_box(
            box_id="bbox_0002",
            label="inference",
            z0=10,
            z1=13,
            y0=10,
            y1=13,
            x0=10,
            x1=13,
        )
        window_like = self._make_window_like(
            boxes=(inference_box_a, inference_box_b),
            active_segmentation=("semantic", object()),
            semantic_volume=_FakeVolume(np.zeros((32, 32, 32), dtype=np.int16)),
            raw_volume=_FakeVolume(np.zeros((32, 32, 32), dtype=np.float32)),
        )
        eval_runtimes = {
            "bbox_1001": SimpleNamespace(buffer=SimpleNamespace(label_values=(0, 1))),
        }
        fake_runtime = SimpleNamespace(
            dataloader=tuple(),
            buffer=SimpleNamespace(
                add_batch=lambda _batch, _coordinates: None,
                get_pred_labels=lambda: np.ones((3, 3, 3), dtype=np.int64),
            ),
        )
        fake_plan = SimpleNamespace(
            z=SimpleNamespace(extend_before=0, original_size=3),
            y=SimpleNamespace(extend_before=0, original_size=3),
            x=SimpleNamespace(extend_before=0, original_size=3),
        )

        call_index = {"value": 0}

        def runtime_side_effect(*_args, **_kwargs):
            call_index["value"] += 1
            if call_index["value"] == 1:
                return fake_runtime
            raise RuntimeError("simulated inference runtime failure")

        with patch.dict(sys.modules, {"torch": _FakeTorchModule}), patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=self._make_model_runtime(model=_FakeModel()),
        ), patch(
            "src.ui.main_window.get_current_learning_eval_runtimes_by_box_id",
            return_value=eval_runtimes,
        ), patch(
            "src.learning.qt_workers._extract_bbox_context_from_array",
            return_value=np.zeros((3, 3, 3), dtype=np.float32),
        ), patch(
            "src.learning.qt_workers._build_inference_dataloader_runtime_from_entry",
            side_effect=runtime_side_effect,
        ), patch(
            "src.learning.qt_workers._dispose_inference_runtime",
            return_value=tuple(),
        ), patch(
            "src.learning.qt_workers._plan_bbox_context",
            return_value=fake_plan,
        ), patch("src.ui.main_window.show_info") as info_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock:
            result = MainWindow._segment_inference_bboxes_with_dialog(window_like)

        self.assertFalse(result)
        info_mock.assert_not_called()
        warning_mock.assert_called_once()
        warning_text = warning_mock.call_args.args[0].lower()
        self.assertIn("partial success", warning_text)
        self.assertIn("bbox_0002", warning_text)

    def test_segment_inference_reports_cleanup_warning_when_dispose_fails(self) -> None:
        inference_box = self._make_box(
            box_id="bbox_0001",
            label="inference",
            z0=1,
            z1=4,
            y0=1,
            y1=4,
            x0=1,
            x1=4,
        )
        window_like = self._make_window_like(
            boxes=(inference_box,),
            active_segmentation=("semantic", object()),
            semantic_volume=_FakeVolume(np.zeros((32, 32, 32), dtype=np.int16)),
            raw_volume=_FakeVolume(np.zeros((32, 32, 32), dtype=np.float32)),
        )
        eval_runtimes = {
            "bbox_1001": SimpleNamespace(buffer=SimpleNamespace(label_values=(0, 1))),
        }
        fake_runtime = SimpleNamespace(
            dataloader=tuple(),
            buffer=SimpleNamespace(
                add_batch=lambda _batch, _coordinates: None,
                get_pred_labels=lambda: np.ones((3, 3, 3), dtype=np.int64),
            ),
        )
        fake_plan = SimpleNamespace(
            z=SimpleNamespace(extend_before=0, original_size=3),
            y=SimpleNamespace(extend_before=0, original_size=3),
            x=SimpleNamespace(extend_before=0, original_size=3),
        )

        with patch.dict(sys.modules, {"torch": _FakeTorchModule}), patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=self._make_model_runtime(model=_FakeModel()),
        ), patch(
            "src.ui.main_window.get_current_learning_eval_runtimes_by_box_id",
            return_value=eval_runtimes,
        ), patch(
            "src.learning.qt_workers._extract_bbox_context_from_array",
            return_value=np.zeros((3, 3, 3), dtype=np.float32),
        ), patch(
            "src.learning.qt_workers._build_inference_dataloader_runtime_from_entry",
            return_value=fake_runtime,
        ), patch(
            "src.learning.qt_workers._dispose_inference_runtime",
            return_value=("buffer.close(): RuntimeError: cleanup boom",),
        ), patch(
            "src.learning.qt_workers._plan_bbox_context",
            return_value=fake_plan,
        ), patch("src.ui.main_window.show_info") as info_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock:
            result = MainWindow._segment_inference_bboxes_with_dialog(window_like)

        self.assertTrue(result)
        info_mock.assert_not_called()
        warning_mock.assert_called_once()
        warning_text = warning_mock.call_args.args[0].lower()
        self.assertIn("cleanup warnings", warning_text)
        self.assertIn("bbox_0001", warning_text)

    def test_segment_inference_prefers_model_metadata_label_values_before_background_start(self) -> None:
        inference_box = self._make_box(
            box_id="bbox_0001",
            label="inference",
            z0=1,
            z1=4,
            y0=1,
            y1=4,
            x0=1,
            x1=4,
        )
        window_like = self._make_window_like(
            boxes=(inference_box,),
            active_segmentation=("semantic", object()),
            semantic_volume=_FakeVolume(np.zeros((32, 32, 32), dtype=np.int16)),
            raw_volume=_FakeVolume(np.zeros((32, 32, 32), dtype=np.float32)),
        )
        call_order: list[str] = []
        started_calls: list[dict[str, object]] = []

        def _show_notice() -> None:
            call_order.append("notice")

        def _start_background(**kwargs: object) -> None:
            call_order.append("start")
            started_calls.append(dict(kwargs))

        window_like._show_inference_navigation_only_notice = _show_notice
        window_like._start_learning_inference_background = _start_background

        with patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=SimpleNamespace(
                hyperparameters={"label_values": (0, 1)},
                num_classes=2,
            ),
        ), patch(
            "src.ui.main_window.get_current_learning_eval_runtimes_by_box_id",
            return_value={},
        ) as eval_runtime_mock, patch(
            "src.ui.main_window._resolve_shared_eval_label_values",
        ), patch("src.ui.main_window.show_warning") as warning_mock:
            result = MainWindow._segment_inference_bboxes_with_dialog(window_like)

        self.assertTrue(result)
        warning_mock.assert_not_called()
        eval_runtime_mock.assert_not_called()
        self.assertEqual(call_order, ["notice", "start"])
        self.assertEqual(len(started_calls), 1)
        started_kwargs = started_calls[0]
        self.assertIn("raw_array", started_kwargs)
        self.assertEqual(np.asarray(started_kwargs["raw_array"]).shape, (32, 32, 32))
        self.assertIn("inference_boxes", started_kwargs)
        self.assertEqual(len(tuple(started_kwargs["inference_boxes"])), 1)
        self.assertIn("label_values", started_kwargs)
        self.assertEqual(tuple(started_kwargs["label_values"]), (0, 1))

    def test_segment_inference_starts_for_large_label_output(self) -> None:
        volume_shape = (2000, 2000, 2000)
        inference_box = self._make_box(
            box_id="bbox_huge",
            label="inference",
            z0=100,
            z1=1100,
            y0=100,
            y1=1100,
            x0=100,
            x1=1100,
            volume_shape=volume_shape,
        )
        window_like = self._make_window_like(
            boxes=(inference_box,),
            active_segmentation=("semantic", object()),
            semantic_volume=_FakeSparseVolume(np.dtype(np.int16)),
            raw_volume=_FakeSparseVolume(np.dtype(np.float32)),
            volume_shape=volume_shape,
        )
        start_calls: list[dict[str, object]] = []
        window_like._start_learning_inference_background = (
            lambda **kwargs: start_calls.append(dict(kwargs))
        )

        with patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=SimpleNamespace(
                hyperparameters={"label_values": (0, 1)},
                num_classes=2,
            ),
        ), patch(
            "src.ui.main_window.get_current_learning_eval_runtimes_by_box_id",
            return_value={},
        ), patch(
            "src.ui.main_window._resolve_shared_eval_label_values",
        ), patch("src.ui.main_window.show_warning") as warning_mock:
            result = MainWindow._segment_inference_bboxes_with_dialog(window_like)

        self.assertTrue(result)
        warning_mock.assert_not_called()
        self.assertEqual(len(start_calls), 1)
        self.assertEqual(tuple(start_calls[0]["inference_boxes"]), (inference_box,))

    def test_segment_inference_does_not_report_memory_mode_info(self) -> None:
        volume_shape = (2000, 2000, 2000)
        inference_box = self._make_box(
            box_id="bbox_tiled",
            label="inference",
            z0=100,
            z1=1000,
            y0=100,
            y1=1000,
            x0=100,
            x1=1000,
            volume_shape=volume_shape,
        )
        window_like = self._make_window_like(
            boxes=(inference_box,),
            active_segmentation=("semantic", object()),
            semantic_volume=_FakeSparseVolume(np.dtype(np.int16)),
            raw_volume=_FakeSparseVolume(np.dtype(np.float32)),
            volume_shape=volume_shape,
        )
        start_calls: list[dict[str, object]] = []
        window_like._start_learning_inference_background = (
            lambda **kwargs: start_calls.append(dict(kwargs))
        )

        with patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=SimpleNamespace(
                hyperparameters={"label_values": (0, 1, 2, 3)},
                num_classes=4,
            ),
        ), patch(
            "src.ui.main_window.get_current_learning_eval_runtimes_by_box_id",
            return_value={},
        ), patch(
            "src.ui.main_window._resolve_shared_eval_label_values",
        ), patch("src.ui.main_window.show_warning") as warning_mock, patch(
            "src.ui.main_window.show_info"
        ) as info_mock:
            result = MainWindow._segment_inference_bboxes_with_dialog(window_like)

        self.assertTrue(result)
        warning_mock.assert_not_called()
        info_mock.assert_not_called()
        self.assertEqual(len(start_calls), 1)

    def test_start_learning_inference_background_passes_raw_array_reference_to_worker(
        self,
    ) -> None:
        inference_box = self._make_box(
            box_id="bbox_0001",
            label="inference",
            z0=1,
            z1=4,
            y0=1,
            y1=4,
            x0=1,
            x1=4,
        )
        raw_array = np.zeros((32, 32, 32), dtype=np.float32)
        configure_calls: list[dict[str, object]] = []
        entered_calls: list[tuple[object, object]] = []

        class _FakeThread:
            def __init__(self, parent: object) -> None:
                self.parent = parent
                self.started = _FakeSignal()
                self.finished = _FakeSignal()
                self.start_calls = 0

            def start(self) -> None:
                self.start_calls += 1

            def quit(self) -> None:
                return None

            def deleteLater(self) -> None:
                return None

        class _FakeWorker:
            def __init__(self) -> None:
                self.completed = _FakeSignal()
                self.canceled = _FakeSignal()
                self.failed = _FakeSignal()
                self.finished = _FakeSignal()
                self.thread = None

            def configure(self, **kwargs: object) -> None:
                configure_calls.append(dict(kwargs))

            def moveToThread(self, thread: object) -> None:
                self.thread = thread

            def run(self) -> None:
                return None

            def deleteLater(self) -> None:
                return None

        thread_instances: list[_FakeThread] = []
        worker_instances: list[_FakeWorker] = []

        def _build_thread(parent: object) -> _FakeThread:
            thread = _FakeThread(parent)
            thread_instances.append(thread)
            return thread

        def _build_worker() -> _FakeWorker:
            worker = _FakeWorker()
            worker_instances.append(worker)
            return worker

        def _enter_running_state(*, worker: object, thread: object) -> None:
            entered_calls.append((worker, thread))

        window_like = SimpleNamespace(
            _enter_learning_inference_running_state=_enter_running_state,
            _on_learning_inference_completed=lambda _result: None,
            _on_learning_inference_canceled=lambda _message: None,
            _on_learning_inference_failed=lambda _message: None,
            _on_learning_inference_thread_finished=lambda: None,
        )

        with patch("src.ui.main_window.QThread", side_effect=_build_thread), patch(
            "src.ui.main_window.LearningInferenceWorker",
            side_effect=_build_worker,
        ):
            MainWindow._start_learning_inference_background(
                window_like,
                model_runtime=object(),
                inference_boxes=(inference_box,),
                raw_array=raw_array,
                label_values=(0, 1),
                volume_shape=raw_array.shape,
            )

        self.assertEqual(len(configure_calls), 1)
        self.assertIs(configure_calls[0]["raw_array"], raw_array)
        self.assertEqual(tuple(configure_calls[0]["inference_boxes"]), (inference_box,))
        self.assertEqual(tuple(configure_calls[0]["label_values"]), (0, 1))
        self.assertEqual(tuple(configure_calls[0]["volume_shape"]), raw_array.shape)
        self.assertIs(configure_calls[0]["use_tiled_score_buffer"], False)
        self.assertEqual(entered_calls, [(worker_instances[0], thread_instances[0])])
        self.assertIs(worker_instances[0].thread, thread_instances[0])
        self.assertEqual(thread_instances[0].start_calls, 1)

    def test_segment_inference_rejects_eval_runtime_label_values_fallback_when_metadata_missing(self) -> None:
        inference_box = self._make_box(
            box_id="bbox_0001",
            label="inference",
            z0=1,
            z1=4,
            y0=1,
            y1=4,
            x0=1,
            x1=4,
        )
        window_like = self._make_window_like(
            boxes=(inference_box,),
            active_segmentation=("semantic", object()),
            semantic_volume=_FakeVolume(np.zeros((32, 32, 32), dtype=np.int16)),
            raw_volume=_FakeVolume(np.zeros((32, 32, 32), dtype=np.float32)),
        )
        eval_runtimes = {
            "bbox_1001": SimpleNamespace(buffer=SimpleNamespace(label_values=(0, 1))),
        }

        window_like._show_inference_navigation_only_notice = lambda: None
        window_like._start_learning_inference_background = lambda **_kwargs: None

        with patch(
            "src.ui.main_window.get_current_learning_model_runtime",
            return_value=SimpleNamespace(
                hyperparameters={},
                num_classes=2,
            ),
        ), patch(
            "src.ui.main_window.get_current_learning_eval_runtimes_by_box_id",
            return_value=eval_runtimes,
        ), patch(
            "src.ui.main_window._resolve_shared_eval_label_values",
            return_value=(0, 1),
        ) as shared_label_values_mock, patch(
            "src.ui.main_window.show_warning"
        ) as warning_mock:
            result = MainWindow._segment_inference_bboxes_with_dialog(window_like)

        self.assertFalse(result)
        shared_label_values_mock.assert_not_called()
        warning_mock.assert_called_once()
        warning_text = warning_mock.call_args.args[0].lower()
        self.assertIn("label_values", warning_text)
        self.assertIn("loaded model metadata", warning_text)


if __name__ == "__main__":
    unittest.main()
