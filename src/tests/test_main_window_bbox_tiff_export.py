from __future__ import annotations

import os
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
from src.io.bbox_tiff_export import LearningBBoxExtractionOutcome


class _FakeVolume:
    def __init__(self, array: np.ndarray) -> None:
        self._array = np.asarray(array)

    def get_chunk(self, zyx_slices: tuple[slice, slice, slice]) -> np.ndarray:
        return np.asarray(self._array[zyx_slices])


class _FakeBBoxManager:
    def __init__(self, boxes: tuple[BoundingBox, ...]) -> None:
        self._boxes = tuple(boxes)

    def boxes(self) -> tuple[BoundingBox, ...]:
        return self._boxes


@unittest.skipUnless(MainWindow is not None, "MainWindow is not available")
class MainWindowBBoxDatasetBuildFlowTests(unittest.TestCase):
    def _make_box(self, *, box_id: str = "bbox_0007", label: str = "train") -> BoundingBox:
        return BoundingBox.from_bounds(
            box_id=box_id,
            z0=1,
            z1=4,
            y0=2,
            y1=5,
            x0=3,
            x1=6,
            label=label,
            volume_shape=(16, 16, 16),
        )

    def _make_window_like(
        self,
        *,
        boxes: tuple[BoundingBox, ...],
        raw_array: np.ndarray,
        active_segmentation: tuple[str, _FakeVolume] | None,
    ) -> SimpleNamespace:
        ordered_rows = tuple(SimpleNamespace(box_id=box.id) for box in boxes)
        window_like = SimpleNamespace()
        window_like.state = SimpleNamespace(volume_loaded=True)
        window_like._raw_volume = _FakeVolume(raw_array)
        window_like._bbox_manager = _FakeBBoxManager(boxes)
        window_like.bottom_panel = SimpleNamespace(state=SimpleNamespace(bbox_rows=ordered_rows))
        window_like._active_segmentation_volume = lambda: active_segmentation
        return window_like

    def test_build_dataset_from_bboxes_requires_active_segmentation_map(self) -> None:
        box = self._make_box()
        window_like = self._make_window_like(
            boxes=(box,),
            raw_array=np.zeros((16, 16, 16), dtype=np.uint8),
            active_segmentation=None,
        )

        with patch("src.ui.main_window.extract_learning_bboxes_in_memory") as extract_mock, patch(
            "src.ui.main_window.show_info"
        ) as show_info_mock, patch("src.ui.main_window.show_warning") as show_warning_mock:
            result = MainWindow._build_dataset_from_bboxes_with_dialog(window_like)

        self.assertFalse(result)
        extract_mock.assert_not_called()
        show_info_mock.assert_not_called()
        show_warning_mock.assert_called_once()
        self.assertIn(
            "Load a semantic segmentation map before building datasets from bounding boxes.",
            show_warning_mock.call_args.args[0],
        )

    def test_build_dataset_from_bboxes_rejects_instance_segmentation(self) -> None:
        box = self._make_box()
        instance_volume = _FakeVolume(np.zeros((16, 16, 16), dtype=np.uint16))
        window_like = self._make_window_like(
            boxes=(box,),
            raw_array=np.zeros((16, 16, 16), dtype=np.uint8),
            active_segmentation=("instance", instance_volume),
        )

        with patch("src.ui.main_window.extract_learning_bboxes_in_memory") as extract_mock, patch(
            "src.ui.main_window.show_info"
        ) as show_info_mock, patch("src.ui.main_window.show_warning") as show_warning_mock:
            result = MainWindow._build_dataset_from_bboxes_with_dialog(window_like)

        self.assertFalse(result)
        extract_mock.assert_not_called()
        show_info_mock.assert_not_called()
        show_warning_mock.assert_called_once()
        self.assertIn(
            "Only semantic segmentation is supported",
            show_warning_mock.call_args.args[0],
        )

    def test_build_dataset_from_bboxes_requires_train_labeled_bbox(self) -> None:
        validation_box = self._make_box(box_id="bbox_0008", label="validation")
        inference_box = self._make_box(box_id="bbox_0011", label="inference")
        seg_volume = _FakeVolume(np.zeros((16, 16, 16), dtype=np.uint16))
        window_like = self._make_window_like(
            boxes=(validation_box, inference_box),
            raw_array=np.zeros((16, 16, 16), dtype=np.uint8),
            active_segmentation=("instance", seg_volume),
        )

        with patch("src.ui.main_window.extract_learning_bboxes_in_memory") as extract_mock, patch(
            "src.ui.main_window.show_info"
        ) as show_info_mock, patch("src.ui.main_window.show_warning") as show_warning_mock:
            result = MainWindow._build_dataset_from_bboxes_with_dialog(window_like)

        self.assertFalse(result)
        extract_mock.assert_not_called()
        show_info_mock.assert_not_called()
        show_warning_mock.assert_called_once()
        self.assertIn(
            "At least one bounding box labeled 'train' is required",
            show_warning_mock.call_args.args[0],
        )

    def test_build_dataset_from_bboxes_requires_validation_labeled_bbox(self) -> None:
        train_box = self._make_box(box_id="bbox_0007", label="train")
        seg_volume = _FakeVolume(np.zeros((16, 16, 16), dtype=np.uint16))
        window_like = self._make_window_like(
            boxes=(train_box,),
            raw_array=np.zeros((16, 16, 16), dtype=np.uint8),
            active_segmentation=("semantic", seg_volume),
        )

        with patch("src.ui.main_window.extract_learning_bboxes_in_memory") as extract_mock, patch(
            "src.ui.main_window.show_info"
        ) as show_info_mock, patch("src.ui.main_window.show_warning") as show_warning_mock:
            result = MainWindow._build_dataset_from_bboxes_with_dialog(window_like)

        self.assertFalse(result)
        extract_mock.assert_not_called()
        show_info_mock.assert_not_called()
        show_warning_mock.assert_called_once()
        self.assertIn(
            "At least one bounding box labeled 'validation' is required",
            show_warning_mock.call_args.args[0],
        )

    def test_build_dataset_from_bboxes_uses_in_memory_extraction(self) -> None:
        train_box = self._make_box(box_id="bbox_0007", label="train")
        inference_box = self._make_box(box_id="bbox_0010", label="inference")
        validation_box = self._make_box(box_id="bbox_0008", label="validation")
        seg_volume = _FakeVolume(np.zeros((16, 16, 16), dtype=np.uint16))
        window_like = self._make_window_like(
            boxes=(train_box, inference_box, validation_box),
            raw_array=np.zeros((16, 16, 16), dtype=np.uint8),
            active_segmentation=("semantic", seg_volume),
        )

        with patch(
            "src.ui.main_window.extract_learning_bboxes_in_memory",
            return_value=LearningBBoxExtractionOutcome(
                tensor_entry_count=2,
                learning_train_box_ids=("bbox_0007",),
                learning_batch_size=4,
                learning_num_workers=8,
                eval_validation_box_ids=("bbox_0008",),
                eval_batch_size=4,
                eval_num_workers=8,
            ),
        ) as extract_mock, patch(
            "src.ui.main_window.compute_and_store_current_learning_class_weights",
            return_value=[1.0, 2.5, 100.0],
        ) as compute_weights_mock, patch(
            "src.ui.main_window.clear_current_learning_bbox_batch"
        ) as clear_batch_mock, patch(
            "src.ui.main_window.get_current_learning_bbox_batch",
            return_value=None,
        ), patch("src.ui.main_window.show_info") as show_info_mock, patch(
            "src.ui.main_window.show_warning"
        ) as show_warning_mock:
            result = MainWindow._build_dataset_from_bboxes_with_dialog(window_like)

        self.assertTrue(result)
        extract_mock.assert_called_once()
        compute_weights_mock.assert_called_once_with(
            max_weight=100.0,
            device="cuda:0",
        )
        clear_batch_mock.assert_called_once()
        show_warning_mock.assert_not_called()

        extract_args = extract_mock.call_args
        self.assertEqual(tuple(extract_args.args[0].shape), (16, 16, 16))
        self.assertEqual(tuple(extract_args.args[1].shape), (16, 16, 16))
        self.assertEqual(extract_args.kwargs["ordered_box_ids"], ("bbox_0007", "bbox_0008"))
        self.assertNotIn("bbox_0010", extract_args.kwargs["ordered_box_ids"])
        self.assertEqual(extract_args.kwargs["learning_batch_size"], 4)
        self.assertEqual(extract_args.kwargs["learning_num_workers"], 8)
        self.assertTrue(extract_args.kwargs["learning_pin_memory"])
        self.assertTrue(extract_args.kwargs["learning_drop_last"])
        self.assertTrue(extract_args.kwargs["build_eval_dataloaders"])
        self.assertEqual(extract_args.kwargs["eval_batch_size"], 4)
        self.assertEqual(extract_args.kwargs["eval_num_workers"], 8)
        self.assertTrue(extract_args.kwargs["eval_pin_memory"])
        self.assertFalse(extract_args.kwargs["eval_drop_last"])

        info_text = show_info_mock.call_args.args[0]
        self.assertIn("Built bounding box learning datasets and buffers in memory.", info_text)
        self.assertIn("- Temporary tensor entries built then released: 2", info_text)
        self.assertIn("Learning DataLoader: 1 train bboxes, batch_size=4, num_workers=8", info_text)
        self.assertIn(
            "Evaluation DataLoaders: 1 validation bboxes, batch_size=4, num_workers=8",
            info_text,
        )
        self.assertIn(
            "Loss class weights initialized on cuda:0: [1, 2.5, 100]",
            info_text,
        )

    def test_build_dataset_from_bboxes_shows_warning_when_extraction_raises(self) -> None:
        train_box = self._make_box(box_id="bbox_0007", label="train")
        validation_box = self._make_box(box_id="bbox_0008", label="validation")
        seg_volume = _FakeVolume(np.zeros((16, 16, 16), dtype=np.uint16))
        window_like = self._make_window_like(
            boxes=(train_box, validation_box),
            raw_array=np.zeros((16, 16, 16), dtype=np.uint8),
            active_segmentation=("semantic", seg_volume),
        )

        with patch(
            "src.ui.main_window.extract_learning_bboxes_in_memory",
            side_effect=RuntimeError("extract boom"),
        ), patch(
            "src.ui.main_window.compute_and_store_current_learning_class_weights"
        ) as compute_weights_mock, patch(
            "src.ui.main_window.clear_current_learning_bbox_batch"
        ) as clear_batch_mock, patch(
            "src.ui.main_window.get_current_learning_bbox_batch",
            return_value=None,
        ), patch("src.ui.main_window.show_info") as show_info_mock, patch(
            "src.ui.main_window.show_warning"
        ) as show_warning_mock:
            result = MainWindow._build_dataset_from_bboxes_with_dialog(window_like)

        self.assertFalse(result)
        compute_weights_mock.assert_not_called()
        clear_batch_mock.assert_called_once()
        show_info_mock.assert_not_called()
        show_warning_mock.assert_called_once()
        self.assertIn("extract boom", show_warning_mock.call_args.args[0])

    def test_build_dataset_from_bboxes_warns_when_tensors_remain_in_session(self) -> None:
        train_box = self._make_box(box_id="bbox_0007", label="train")
        validation_box = self._make_box(box_id="bbox_0008", label="validation")
        seg_volume = _FakeVolume(np.zeros((16, 16, 16), dtype=np.uint16))
        window_like = self._make_window_like(
            boxes=(train_box, validation_box),
            raw_array=np.zeros((16, 16, 16), dtype=np.uint8),
            active_segmentation=("semantic", seg_volume),
        )

        with patch(
            "src.ui.main_window.extract_learning_bboxes_in_memory",
            return_value=LearningBBoxExtractionOutcome(
                tensor_entry_count=2,
                learning_train_box_ids=("bbox_0007",),
                learning_batch_size=4,
                learning_num_workers=8,
                eval_validation_box_ids=("bbox_0008",),
                eval_batch_size=4,
                eval_num_workers=8,
            ),
        ), patch(
            "src.ui.main_window.compute_and_store_current_learning_class_weights",
            return_value=object(),
        ) as compute_weights_mock, patch(
            "src.ui.main_window.clear_current_learning_bbox_batch"
        ) as clear_batch_mock, patch(
            "src.ui.main_window.get_current_learning_bbox_batch",
            return_value=SimpleNamespace(size=1),
        ), patch("src.ui.main_window.show_info") as show_info_mock, patch(
            "src.ui.main_window.show_warning"
        ) as show_warning_mock:
            result = MainWindow._build_dataset_from_bboxes_with_dialog(window_like)

        self.assertFalse(result)
        compute_weights_mock.assert_called_once_with(
            max_weight=100.0,
            device="cuda:0",
        )
        clear_batch_mock.assert_called_once()
        show_info_mock.assert_not_called()
        show_warning_mock.assert_called_once()
        self.assertIn("not fully released", show_warning_mock.call_args.args[0])

    def test_build_dataset_from_bboxes_warns_when_class_weight_computation_raises(self) -> None:
        train_box = self._make_box(box_id="bbox_0007", label="train")
        validation_box = self._make_box(box_id="bbox_0008", label="validation")
        seg_volume = _FakeVolume(np.zeros((16, 16, 16), dtype=np.uint16))
        window_like = self._make_window_like(
            boxes=(train_box, validation_box),
            raw_array=np.zeros((16, 16, 16), dtype=np.uint8),
            active_segmentation=("semantic", seg_volume),
        )

        with patch(
            "src.ui.main_window.extract_learning_bboxes_in_memory",
            return_value=LearningBBoxExtractionOutcome(
                tensor_entry_count=2,
                learning_train_box_ids=("bbox_0007",),
                learning_batch_size=4,
                learning_num_workers=8,
                eval_validation_box_ids=("bbox_0008",),
                eval_batch_size=4,
                eval_num_workers=8,
            ),
        ), patch(
            "src.ui.main_window.compute_and_store_current_learning_class_weights",
            side_effect=RuntimeError("CUDA device cuda:0 is required to compute class weights."),
        ) as compute_weights_mock, patch(
            "src.ui.main_window.clear_current_learning_bbox_batch"
        ) as clear_batch_mock, patch(
            "src.ui.main_window.get_current_learning_bbox_batch",
            return_value=None,
        ), patch("src.ui.main_window.show_info") as show_info_mock, patch(
            "src.ui.main_window.show_warning"
        ) as show_warning_mock:
            result = MainWindow._build_dataset_from_bboxes_with_dialog(window_like)

        self.assertFalse(result)
        compute_weights_mock.assert_called_once_with(
            max_weight=100.0,
            device="cuda:0",
        )
        clear_batch_mock.assert_called_once()
        show_info_mock.assert_not_called()
        show_warning_mock.assert_called_once()
        self.assertIn("cuda:0", show_warning_mock.call_args.args[0].lower())


if __name__ == "__main__":
    unittest.main()
