from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]

from src.bbox import BoundingBox
from src.io.bbox_export_utils import plan_bbox_context
from src.io.bbox_tiff_export import (
    FIXED_LEARNING_MINIVOL_SIZE,
    LearningBBoxExtractionOutcome,
    RawBBoxExportOutcome,
    RawBBoxSaveFailure,
    SegmentationBBoxExportOutcome,
    SegmentationBBoxSaveFailure,
    build_bbox_tiff_filename,
    build_segmentation_bbox_context_from_array,
    build_segmentation_bbox_tiff_filename,
    choose_segmentation_export_dtype,
    extract_bboxes_for_learning,
    extract_learning_bboxes_in_memory,
    extract_learning_bbox_tensors,
    extract_segmentation_bbox_region_from_array,
    export_raw_bboxes_as_tiff,
    export_segmentation_bboxes_as_tiff,
    export_bboxes_as_tiff,
    is_segmentation_bbox_zero_only,
    smallest_signed_integer_dtype_for_range,
)
from src.learning import (
    LearningBBoxDataLoaderRuntime,
    LearningBBoxEvalRuntime,
    LearningBBoxTensorEntry,
    clear_current_learning_bbox_batch,
    clear_current_learning_dataloader_runtime,
    clear_current_learning_eval_runtimes_by_box_id,
    get_current_learning_bbox_batch,
    get_current_learning_dataloader_runtime,
    get_current_learning_eval_runtimes_by_box_id,
    set_current_learning_bbox_entries,
    set_current_learning_dataloader_runtime,
    set_current_learning_eval_runtimes_by_box_id,
)

class BBoxTiffExportNamingTests(unittest.TestCase):
    def test_build_bbox_tiff_filename(self) -> None:
        self.assertEqual(build_bbox_tiff_filename(1, "train"), "bbox1_train.tif")
        self.assertEqual(
            build_bbox_tiff_filename(12, " Validation "),
            "bbox12_validation.tif",
        )
        self.assertEqual(
            build_bbox_tiff_filename(3, "INFERENCE"),
            "bbox3_inference.tif",
        )

    def test_build_bbox_tiff_filename_rejects_invalid_index(self) -> None:
        with self.assertRaises(ValueError):
            build_bbox_tiff_filename(0, "train")
        with self.assertRaises(ValueError):
            build_bbox_tiff_filename(-1, "train")
        with self.assertRaises(TypeError):
            build_bbox_tiff_filename(True, "train")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            build_bbox_tiff_filename(2.2, "train")  # type: ignore[arg-type]

    def test_build_bbox_tiff_filename_rejects_invalid_label(self) -> None:
        with self.assertRaises(ValueError):
            build_bbox_tiff_filename(1, "dev")
        with self.assertRaises(TypeError):
            build_bbox_tiff_filename(1, 3)  # type: ignore[arg-type]

    def test_build_segmentation_bbox_tiff_filename(self) -> None:
        self.assertEqual(
            build_segmentation_bbox_tiff_filename(7, "train"),
            "bbox7_train_seg.tif",
        )
        self.assertEqual(
            build_segmentation_bbox_tiff_filename(42, " Validation "),
            "bbox42_validation_seg.tif",
        )

    def test_build_segmentation_bbox_tiff_filename_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            build_segmentation_bbox_tiff_filename(0, "train")
        with self.assertRaises(ValueError):
            build_segmentation_bbox_tiff_filename(1, "unknown")


class SegmentationBBoxOutcomeModelTests(unittest.TestCase):
    def test_default_outcome_is_empty(self) -> None:
        outcome = SegmentationBBoxExportOutcome()
        self.assertEqual(outcome.saved_paths, tuple())
        self.assertEqual(outcome.skipped_zero_box_ids, tuple())
        self.assertEqual(outcome.failed_boxes, tuple())

    def test_outcome_carries_saved_skipped_and_failed_entries(self) -> None:
        failure = SegmentationBBoxSaveFailure(
            box_id="bbox_0007",
            index=7,
            label="train",
            error="disk full",
        )
        outcome = SegmentationBBoxExportOutcome(
            saved_paths=("/tmp/out/bbox7_train_seg.tif",),
            skipped_zero_box_ids=("bbox_0002",),
            failed_boxes=(failure,),
        )
        self.assertEqual(outcome.saved_paths, ("/tmp/out/bbox7_train_seg.tif",))
        self.assertEqual(outcome.skipped_zero_box_ids, ("bbox_0002",))
        self.assertEqual(len(outcome.failed_boxes), 1)
        self.assertEqual(outcome.failed_boxes[0].error, "disk full")


class RawBBoxOutcomeModelTests(unittest.TestCase):
    def test_default_raw_outcome_is_empty(self) -> None:
        outcome = RawBBoxExportOutcome()
        self.assertEqual(outcome.saved_paths, tuple())
        self.assertEqual(outcome.failed_boxes, tuple())

    def test_raw_outcome_carries_saved_and_failed_entries(self) -> None:
        failure = RawBBoxSaveFailure(
            box_id="bbox_0011",
            index=11,
            label="validation",
            error="permission denied",
        )
        outcome = RawBBoxExportOutcome(
            saved_paths=("/tmp/out/bbox7_train.tif",),
            failed_boxes=(failure,),
        )
        self.assertEqual(outcome.saved_paths, ("/tmp/out/bbox7_train.tif",))
        self.assertEqual(len(outcome.failed_boxes), 1)
        self.assertEqual(outcome.failed_boxes[0].box_id, "bbox_0011")
        self.assertEqual(outcome.failed_boxes[0].error, "permission denied")


class LearningBBoxExtractionOutcomeModelTests(unittest.TestCase):
    def test_default_learning_extraction_outcome_is_empty(self) -> None:
        outcome = LearningBBoxExtractionOutcome()
        self.assertEqual(outcome.raw_saved_paths, tuple())
        self.assertEqual(outcome.raw_failed_boxes, tuple())
        self.assertEqual(outcome.segmentation_saved_paths, tuple())
        self.assertEqual(outcome.segmentation_skipped_zero_box_ids, tuple())
        self.assertEqual(outcome.segmentation_failed_boxes, tuple())
        self.assertEqual(outcome.tensor_entry_count, 0)
        self.assertEqual(outcome.eval_validation_box_ids, tuple())
        self.assertIsNone(outcome.eval_batch_size)
        self.assertIsNone(outcome.eval_num_workers)

    def test_learning_extraction_outcome_carries_all_sections(self) -> None:
        raw_failure = RawBBoxSaveFailure(
            box_id="bbox_0002",
            index=2,
            label="train",
            error="raw write failed",
        )
        seg_failure = SegmentationBBoxSaveFailure(
            box_id="bbox_0003",
            index=3,
            label="inference",
            error="seg write failed",
        )
        outcome = LearningBBoxExtractionOutcome(
            raw_saved_paths=("/tmp/out/bbox7_train.tif",),
            raw_failed_boxes=(raw_failure,),
            segmentation_saved_paths=("/tmp/out/bbox7_train_seg.tif",),
            segmentation_skipped_zero_box_ids=("bbox_0004",),
            segmentation_failed_boxes=(seg_failure,),
            tensor_entry_count=5,
            learning_train_box_ids=("bbox_0001",),
            learning_batch_size=4,
            learning_num_workers=8,
            eval_validation_box_ids=("bbox_0002",),
            eval_batch_size=4,
            eval_num_workers=8,
        )
        self.assertEqual(outcome.raw_saved_paths, ("/tmp/out/bbox7_train.tif",))
        self.assertEqual(len(outcome.raw_failed_boxes), 1)
        self.assertEqual(outcome.segmentation_saved_paths, ("/tmp/out/bbox7_train_seg.tif",))
        self.assertEqual(outcome.segmentation_skipped_zero_box_ids, ("bbox_0004",))
        self.assertEqual(len(outcome.segmentation_failed_boxes), 1)
        self.assertEqual(outcome.tensor_entry_count, 5)
        self.assertEqual(outcome.learning_train_box_ids, ("bbox_0001",))
        self.assertEqual(outcome.learning_batch_size, 4)
        self.assertEqual(outcome.learning_num_workers, 8)
        self.assertEqual(outcome.eval_validation_box_ids, ("bbox_0002",))
        self.assertEqual(outcome.eval_batch_size, 4)
        self.assertEqual(outcome.eval_num_workers, 8)


class SegmentationBBoxDTypeTests(unittest.TestCase):
    def test_smallest_signed_integer_dtype_for_range_thresholds(self) -> None:
        self.assertEqual(
            smallest_signed_integer_dtype_for_range(-100, 127),
            np.dtype(np.int8),
        )
        self.assertEqual(
            smallest_signed_integer_dtype_for_range(-100, 128),
            np.dtype(np.int16),
        )
        self.assertEqual(
            smallest_signed_integer_dtype_for_range(-32768, 32767),
            np.dtype(np.int16),
        )
        self.assertEqual(
            smallest_signed_integer_dtype_for_range(-32769, 32767),
            np.dtype(np.int32),
        )
        self.assertEqual(
            smallest_signed_integer_dtype_for_range(-2_147_483_648, 2_147_483_647),
            np.dtype(np.int32),
        )
        self.assertEqual(
            smallest_signed_integer_dtype_for_range(-2_147_483_649, 2_147_483_647),
            np.dtype(np.int64),
        )

    def test_smallest_signed_integer_dtype_for_range_rejects_invalid_range(self) -> None:
        with self.assertRaises(ValueError):
            smallest_signed_integer_dtype_for_range(5, 4)
        with self.assertRaises(TypeError):
            smallest_signed_integer_dtype_for_range(True, 4)  # type: ignore[arg-type]

    def test_choose_segmentation_export_dtype_uses_context_fill_value(self) -> None:
        array = np.array([0, 127], dtype=np.uint8)
        self.assertEqual(choose_segmentation_export_dtype(array), np.dtype(np.int8))

        array = np.array([0, 128], dtype=np.uint16)
        self.assertEqual(choose_segmentation_export_dtype(array), np.dtype(np.int16))

        array = np.array([0, 10], dtype=np.uint16)
        self.assertEqual(
            choose_segmentation_export_dtype(array, context_fill_value=-50_000),
            np.dtype(np.int32),
        )

    def test_choose_segmentation_export_dtype_handles_empty_integer_array(self) -> None:
        empty = np.array([], dtype=np.uint32)
        self.assertEqual(choose_segmentation_export_dtype(empty), np.dtype(np.int8))

    def test_choose_segmentation_export_dtype_rejects_non_integer_array(self) -> None:
        with self.assertRaises(ValueError):
            choose_segmentation_export_dtype(np.array([1.5], dtype=np.float32))


class SegmentationBBoxContextBuilderTests(unittest.TestCase):
    def test_build_segmentation_bbox_context_centered_fills_context_with_minus_100(self) -> None:
        volume = np.zeros((40, 41, 42), dtype=np.uint8)
        z_bounds = (10, 14)
        y_bounds = (12, 17)
        x_bounds = (20, 26)
        bbox_data = np.arange(4 * 5 * 6, dtype=np.uint8).reshape((4, 5, 6))
        volume[
            z_bounds[0] : z_bounds[1],
            y_bounds[0] : y_bounds[1],
            x_bounds[0] : x_bounds[1],
        ] = bbox_data

        context = build_segmentation_bbox_context_from_array(
            volume,
            z_bounds=z_bounds,
            y_bounds=y_bounds,
            x_bounds=x_bounds,
            context_fill_value=-100,
        )

        self.assertEqual(tuple(context.shape), (300, 300, 300))
        self.assertEqual(context.dtype, np.dtype(np.int8))
        plan = plan_bbox_context(
            z_bounds=z_bounds,
            y_bounds=y_bounds,
            x_bounds=x_bounds,
            volume_shape=volume.shape,
        )
        z_start = int(plan.z.extend_before)
        y_start = int(plan.y.extend_before)
        x_start = int(plan.x.extend_before)
        np.testing.assert_array_equal(
            context[
                z_start : z_start + 4,
                y_start : y_start + 5,
                x_start : x_start + 6,
            ],
            bbox_data.astype(np.int8),
        )
        self.assertEqual(int(np.count_nonzero(context != -100)), int(np.prod(bbox_data.shape)))
        self.assertEqual(int(context[0, 0, 0]), -100)
        self.assertEqual(int(context[-1, -1, -1]), -100)

    def test_build_segmentation_bbox_context_near_border_without_reflect(self) -> None:
        volume = np.zeros((32, 33, 34), dtype=np.uint8)
        z_bounds = (0, 3)
        y_bounds = (30, 33)
        x_bounds = (31, 34)
        bbox_data = (np.arange(3 * 3 * 3, dtype=np.uint8).reshape((3, 3, 3)) + 200)
        volume[
            z_bounds[0] : z_bounds[1],
            y_bounds[0] : y_bounds[1],
            x_bounds[0] : x_bounds[1],
        ] = bbox_data

        context = build_segmentation_bbox_context_from_array(
            volume,
            z_bounds=z_bounds,
            y_bounds=y_bounds,
            x_bounds=x_bounds,
            context_fill_value=-100,
        )

        self.assertEqual(tuple(context.shape), (300, 300, 300))
        self.assertEqual(context.dtype, np.dtype(np.int16))
        plan = plan_bbox_context(
            z_bounds=z_bounds,
            y_bounds=y_bounds,
            x_bounds=x_bounds,
            volume_shape=volume.shape,
        )
        z_start = int(plan.z.extend_before)
        y_start = int(plan.y.extend_before)
        x_start = int(plan.x.extend_before)
        np.testing.assert_array_equal(
            context[
                z_start : z_start + 3,
                y_start : y_start + 3,
                x_start : x_start + 3,
            ],
            bbox_data.astype(np.int16),
        )
        self.assertEqual(int(np.count_nonzero(context != -100)), int(np.prod(bbox_data.shape)))
        self.assertEqual(int(context[0, 0, 0]), -100)
        self.assertEqual(int(context[-1, -1, -1]), -100)

    def test_build_segmentation_bbox_context_rejects_non_integer_array(self) -> None:
        volume = np.zeros((5, 6, 7), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            build_segmentation_bbox_context_from_array(
                volume,
                z_bounds=(1, 3),
                y_bounds=(1, 4),
                x_bounds=(2, 5),
            )


class SegmentationBBoxZeroSkipTests(unittest.TestCase):
    def test_extract_segmentation_bbox_region_returns_original_bbox_crop(self) -> None:
        volume = np.arange(7 * 8 * 9, dtype=np.uint16).reshape((7, 8, 9))
        crop = extract_segmentation_bbox_region_from_array(
            volume,
            z_bounds=(2, 5),
            y_bounds=(1, 6),
            x_bounds=(3, 8),
        )
        np.testing.assert_array_equal(crop, volume[2:5, 1:6, 3:8])
        self.assertEqual(crop.dtype, volume.dtype)

    def test_is_segmentation_bbox_zero_only_true_when_bbox_has_only_background(self) -> None:
        volume = np.zeros((12, 13, 14), dtype=np.uint16)
        volume[0, 0, 0] = 9  # outside tested bbox
        self.assertTrue(
            is_segmentation_bbox_zero_only(
                volume,
                z_bounds=(4, 8),
                y_bounds=(5, 9),
                x_bounds=(6, 10),
            )
        )

    def test_is_segmentation_bbox_zero_only_false_when_bbox_contains_foreground(self) -> None:
        volume = np.zeros((12, 13, 14), dtype=np.uint16)
        volume[6, 7, 8] = 3
        self.assertFalse(
            is_segmentation_bbox_zero_only(
                volume,
                z_bounds=(4, 8),
                y_bounds=(5, 9),
                x_bounds=(6, 10),
            )
        )

    def test_is_segmentation_bbox_zero_only_rejects_non_integer_array(self) -> None:
        volume = np.zeros((6, 7, 8), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            is_segmentation_bbox_zero_only(
                volume,
                z_bounds=(1, 3),
                y_bounds=(2, 5),
                x_bounds=(3, 6),
            )


class BBoxTiffExportBatchTests(unittest.TestCase):
    def test_export_bboxes_as_tiff_uses_box_id_numeric_suffix_for_indexing(self) -> None:
        volume = np.zeros((20, 20, 20), dtype=np.uint8)
        first = BoundingBox.from_bounds(
            box_id="bbox_0002",
            z0=1,
            z1=4,
            y0=2,
            y1=5,
            x0=3,
            x1=6,
            label="train",
            volume_shape=(20, 20, 20),
        )
        second = BoundingBox.from_bounds(
            box_id="bbox_0011",
            z0=10,
            z1=13,
            y0=11,
            y1=14,
            x0=12,
            x1=15,
            label="validation",
            volume_shape=(20, 20, 20),
        )
        boxes_by_id = {first.id: first, second.id: second}
        ordered_ids = (second.id, first.id)
        extract_calls = []
        save_calls = []

        def fake_extractor(
            array: np.ndarray,
            *,
            z_bounds: tuple[int, int],
            y_bounds: tuple[int, int],
            x_bounds: tuple[int, int],
        ) -> np.ndarray:
            extract_calls.append((tuple(array.shape), z_bounds, y_bounds, x_bounds))
            fill_value = len(extract_calls)
            return np.full((2, 2, 2), fill_value, dtype=np.uint8)

        def fake_saver(
            array: np.ndarray,
            *,
            output_dir: str,
            index: int,
            label: str,
            overwrite: bool,
        ) -> str:
            save_calls.append((int(index), str(label), int(array[0, 0, 0]), bool(overwrite)))
            return f"{output_dir}/bbox{index}_{label}.tif"

        paths = export_bboxes_as_tiff(
            volume,
            boxes_by_id=boxes_by_id,
            ordered_box_ids=ordered_ids,
            output_dir="/tmp/out",
            overwrite=True,
            extractor=fake_extractor,
            saver=fake_saver,
        )

        self.assertEqual(
            extract_calls,
            [
                ((20, 20, 20), (10, 13), (11, 14), (12, 15)),
                ((20, 20, 20), (1, 4), (2, 5), (3, 6)),
            ],
        )
        self.assertEqual(
            save_calls,
            [
                (11, "validation", 1, True),
                (2, "train", 2, True),
            ],
        )
        self.assertEqual(
            paths,
            (
                "/tmp/out/bbox11_validation.tif",
                "/tmp/out/bbox2_train.tif",
            ),
        )

    def test_export_bboxes_as_tiff_rejects_box_id_without_numeric_suffix(self) -> None:
        volume = np.zeros((10, 10, 10), dtype=np.uint8)
        box = BoundingBox.from_bounds(
            box_id="no_suffix",
            z0=1,
            z1=2,
            y0=1,
            y1=2,
            x0=1,
            x1=2,
            label="train",
            volume_shape=(10, 10, 10),
        )
        with self.assertRaisesRegex(ValueError, "must end with digits"):
            export_bboxes_as_tiff(
                volume,
                boxes_by_id={box.id: box},
                ordered_box_ids=(box.id,),
                output_dir="/tmp/out",
                extractor=lambda *_args, **_kwargs: np.zeros((1, 1, 1), dtype=np.uint8),
                saver=lambda *_args, **_kwargs: "/tmp/out/x.tif",
            )

    def test_export_bboxes_as_tiff_rejects_duplicate_numeric_suffix(self) -> None:
        volume = np.zeros((10, 10, 10), dtype=np.uint8)
        first = BoundingBox.from_bounds(
            box_id="first_007",
            z0=1,
            z1=2,
            y0=1,
            y1=2,
            x0=1,
            x1=2,
            label="train",
            volume_shape=(10, 10, 10),
        )
        second = BoundingBox.from_bounds(
            box_id="second_7",
            z0=3,
            z1=4,
            y0=3,
            y1=4,
            x0=3,
            x1=4,
            label="validation",
            volume_shape=(10, 10, 10),
        )
        with self.assertRaisesRegex(ValueError, "Duplicate numeric suffix"):
            export_bboxes_as_tiff(
                volume,
                boxes_by_id={first.id: first, second.id: second},
                ordered_box_ids=(first.id, second.id),
                output_dir="/tmp/out",
                extractor=lambda *_args, **_kwargs: np.zeros((1, 1, 1), dtype=np.uint8),
                saver=lambda *_args, **_kwargs: "/tmp/out/x.tif",
            )

    def test_export_bboxes_as_tiff_rejects_missing_and_duplicate_ids(self) -> None:
        volume = np.zeros((10, 10, 10), dtype=np.uint8)
        box = BoundingBox.from_bounds(
            box_id="a",
            z0=1,
            z1=2,
            y0=1,
            y1=2,
            x0=1,
            x1=2,
            label="inference",
            volume_shape=(10, 10, 10),
        )
        boxes_by_id = {"a": box}

        with self.assertRaises(KeyError):
            export_bboxes_as_tiff(
                volume,
                boxes_by_id=boxes_by_id,
                ordered_box_ids=("missing",),
                output_dir="/tmp/out",
                extractor=lambda *_args, **_kwargs: np.zeros((1, 1, 1), dtype=np.uint8),
                saver=lambda *_args, **_kwargs: "/tmp/out/x.tif",
            )

        with self.assertRaises(ValueError):
            export_bboxes_as_tiff(
                volume,
                boxes_by_id=boxes_by_id,
                ordered_box_ids=("a", "a"),
                output_dir="/tmp/out",
                extractor=lambda *_args, **_kwargs: np.zeros((1, 1, 1), dtype=np.uint8),
                saver=lambda *_args, **_kwargs: "/tmp/out/x.tif",
            )

    def test_export_bboxes_as_tiff_rejects_singleton_axis_before_any_write(self) -> None:
        volume = np.zeros((1, 10, 10), dtype=np.uint8)
        box = BoundingBox.from_bounds(
            box_id="a",
            z0=0,
            z1=1,
            y0=2,
            y1=5,
            x0=3,
            x1=6,
            label="train",
            volume_shape=(1, 10, 10),
        )
        boxes_by_id = {"a": box}
        call_counts = {"extract": 0, "save": 0}

        def fake_extractor(*_args, **_kwargs) -> np.ndarray:
            call_counts["extract"] += 1
            return np.zeros((1, 1, 1), dtype=np.uint8)

        def fake_saver(*_args, **_kwargs) -> str:
            call_counts["save"] += 1
            return "/tmp/out/x.tif"

        with self.assertRaisesRegex(ValueError, "axis of length 1"):
            export_bboxes_as_tiff(
                volume,
                boxes_by_id=boxes_by_id,
                ordered_box_ids=("a",),
                output_dir="/tmp/out",
                extractor=fake_extractor,
                saver=fake_saver,
            )
        self.assertEqual(call_counts, {"extract": 0, "save": 0})


class RawBBoxTiffExportBatchTests(unittest.TestCase):
    def test_export_raw_bboxes_as_tiff_continues_and_collects_failures(self) -> None:
        volume = np.zeros((24, 24, 24), dtype=np.uint8)
        box_saved = BoundingBox.from_bounds(
            box_id="bbox_0007",
            z0=2,
            z1=5,
            y0=3,
            y1=6,
            x0=4,
            x1=7,
            label="train",
            volume_shape=(24, 24, 24),
        )
        box_failed = BoundingBox.from_bounds(
            box_id="bbox_0008",
            z0=8,
            z1=11,
            y0=8,
            y1=11,
            x0=8,
            x1=11,
            label="validation",
            volume_shape=(24, 24, 24),
        )
        boxes_by_id = {
            box_saved.id: box_saved,
            box_failed.id: box_failed,
        }
        ordered_ids = (box_failed.id, "missing_0010", box_saved.id)
        save_calls = []

        def fake_extractor(
            _array: np.ndarray,
            *,
            z_bounds: tuple[int, int],
            y_bounds: tuple[int, int],
            x_bounds: tuple[int, int],
        ) -> np.ndarray:
            del y_bounds, x_bounds
            return np.full((2, 2, 2), z_bounds[0], dtype=np.uint8)

        def fake_saver(
            array: np.ndarray,
            *,
            output_dir: str,
            index: int,
            label: str,
            overwrite: bool,
        ) -> str:
            del array
            save_calls.append((int(index), str(label), bool(overwrite)))
            if int(index) == 8:
                raise RuntimeError("raw write failed for test")
            return f"{output_dir}/bbox{index}_{label}.tif"

        outcome = export_raw_bboxes_as_tiff(
            volume,
            boxes_by_id=boxes_by_id,
            ordered_box_ids=ordered_ids,
            output_dir="/tmp/out",
            overwrite=True,
            extractor=fake_extractor,
            saver=fake_saver,
        )

        self.assertEqual(
            save_calls,
            [
                (8, "validation", True),
                (7, "train", True),
            ],
        )
        self.assertEqual(outcome.saved_paths, ("/tmp/out/bbox7_train.tif",))
        self.assertEqual(len(outcome.failed_boxes), 2)
        self.assertEqual(outcome.failed_boxes[0].box_id, "bbox_0008")
        self.assertEqual(outcome.failed_boxes[0].index, 8)
        self.assertEqual(outcome.failed_boxes[0].label, "validation")
        self.assertIn("raw write failed for test", outcome.failed_boxes[0].error)
        self.assertEqual(outcome.failed_boxes[1].box_id, "missing_0010")
        self.assertEqual(outcome.failed_boxes[1].index, 10)
        self.assertIn("Unknown bounding box id", outcome.failed_boxes[1].error)


class SegmentationBBoxTiffExportBatchTests(unittest.TestCase):
    def test_export_segmentation_bboxes_as_tiff_saves_skips_and_collects_failures(self) -> None:
        segmentation = np.zeros((24, 24, 24), dtype=np.uint16)
        box_saved = BoundingBox.from_bounds(
            box_id="bbox_0007",
            z0=2,
            z1=5,
            y0=3,
            y1=6,
            x0=4,
            x1=7,
            label="train",
            volume_shape=(24, 24, 24),
        )
        box_skipped = BoundingBox.from_bounds(
            box_id="bbox_0008",
            z0=8,
            z1=11,
            y0=8,
            y1=11,
            x0=8,
            x1=11,
            label="validation",
            volume_shape=(24, 24, 24),
        )
        box_failed = BoundingBox.from_bounds(
            box_id="bbox_0009",
            z0=12,
            z1=15,
            y0=12,
            y1=15,
            x0=12,
            x1=15,
            label="inference",
            volume_shape=(24, 24, 24),
        )
        segmentation[3, 4, 5] = 2   # inside box_saved
        segmentation[13, 13, 13] = 5  # inside box_failed

        boxes_by_id = {
            box_saved.id: box_saved,
            box_skipped.id: box_skipped,
            box_failed.id: box_failed,
        }
        ordered_ids = (box_failed.id, box_skipped.id, box_saved.id)
        save_calls = []

        def fake_context_builder(
            _array: np.ndarray,
            *,
            z_bounds: tuple[int, int],
            y_bounds: tuple[int, int],
            x_bounds: tuple[int, int],
            context_fill_value: int,
        ) -> np.ndarray:
            del y_bounds, x_bounds
            return np.full((2, 2, 2), z_bounds[0] + int(context_fill_value), dtype=np.int16)

        def fake_saver(
            array: np.ndarray,
            *,
            output_dir: str,
            index: int,
            label: str,
            overwrite: bool,
        ) -> str:
            save_calls.append((int(index), str(label), tuple(array.shape), bool(overwrite)))
            if int(index) == 9:
                raise RuntimeError("write failed for test")
            return f"{output_dir}/bbox{index}_{label}_seg.tif"

        outcome = export_segmentation_bboxes_as_tiff(
            segmentation,
            boxes_by_id=boxes_by_id,
            ordered_box_ids=ordered_ids,
            output_dir="/tmp/out",
            context_fill_value=-100,
            overwrite=True,
            context_builder=fake_context_builder,
            saver=fake_saver,
        )

        self.assertEqual(
            save_calls,
            [
                (9, "inference", (2, 2, 2), True),
                (7, "train", (2, 2, 2), True),
            ],
        )
        self.assertEqual(outcome.saved_paths, ("/tmp/out/bbox7_train_seg.tif",))
        self.assertEqual(outcome.skipped_zero_box_ids, ("bbox_0008",))
        self.assertEqual(len(outcome.failed_boxes), 1)
        self.assertEqual(outcome.failed_boxes[0].box_id, "bbox_0009")
        self.assertEqual(outcome.failed_boxes[0].index, 9)
        self.assertEqual(outcome.failed_boxes[0].label, "inference")
        self.assertIn("write failed for test", outcome.failed_boxes[0].error)

    def test_export_segmentation_bboxes_as_tiff_continues_after_unknown_id(self) -> None:
        segmentation = np.zeros((16, 16, 16), dtype=np.uint16)
        box = BoundingBox.from_bounds(
            box_id="bbox_0007",
            z0=1,
            z1=3,
            y0=1,
            y1=3,
            x0=1,
            x1=3,
            label="train",
            volume_shape=(16, 16, 16),
        )
        segmentation[1, 1, 1] = 1

        def fake_context_builder(
            _array: np.ndarray,
            *,
            z_bounds: tuple[int, int],
            y_bounds: tuple[int, int],
            x_bounds: tuple[int, int],
            context_fill_value: int,
        ) -> np.ndarray:
            del z_bounds, y_bounds, x_bounds, context_fill_value
            return np.ones((1, 1, 1), dtype=np.int16)

        outcome = export_segmentation_bboxes_as_tiff(
            segmentation,
            boxes_by_id={box.id: box},
            ordered_box_ids=("missing_0010", box.id),
            output_dir="/tmp/out",
            context_builder=fake_context_builder,
            saver=lambda _arr, *, output_dir, index, label, overwrite: (
                f"{output_dir}/bbox{index}_{label}_seg.tif"
            ),
        )

        self.assertEqual(outcome.saved_paths, ("/tmp/out/bbox7_train_seg.tif",))
        self.assertEqual(outcome.skipped_zero_box_ids, tuple())
        self.assertEqual(len(outcome.failed_boxes), 1)
        self.assertEqual(outcome.failed_boxes[0].box_id, "missing_0010")
        self.assertEqual(outcome.failed_boxes[0].index, 10)
        self.assertIn("Unknown bounding box id", outcome.failed_boxes[0].error)

    def test_export_segmentation_bboxes_as_tiff_rejects_non_integer_segmentation(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            export_segmentation_bboxes_as_tiff(
                np.zeros((4, 4, 4), dtype=np.float32),
                boxes_by_id={},
                ordered_box_ids=tuple(),
                output_dir="/tmp/out",
            )


@unittest.skipUnless(torch is not None, "PyTorch is not available")
class LearningBBoxTensorExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_current_learning_bbox_batch()
        clear_current_learning_dataloader_runtime()
        clear_current_learning_eval_runtimes_by_box_id()

    def tearDown(self) -> None:
        clear_current_learning_bbox_batch()
        clear_current_learning_dataloader_runtime()
        clear_current_learning_eval_runtimes_by_box_id()

    def _make_box(
        self,
        *,
        box_id: str,
        z0: int,
        z1: int,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        label: str,
        volume_shape: tuple[int, int, int] = (30, 30, 30),
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

    def _learning_runtime_factories(self):
        class _FakeDataset:
            def __init__(self, tensor_pairs, *, minivol_size, minivol_per_epoch):
                self.tensor_pairs = tuple(tensor_pairs)
                self.minivol_size = int(minivol_size)
                self.minivol_per_epoch = int(minivol_per_epoch)
                self.weights = [1 for _ in self.tensor_pairs]

            def __len__(self):
                return int(self.minivol_per_epoch)

        class _FakeSampler:
            def __init__(self, weights, num_samples):
                self.weights = tuple(weights)
                self.num_samples = int(num_samples)

        class _FakeDataLoader:
            def __init__(self, dataset) -> None:
                self.dataset = dataset

        datasets = []
        samplers = []
        loaders = []

        def dataset_factory(tensor_pairs, *, minivol_size, minivol_per_epoch):
            dataset = _FakeDataset(
                tensor_pairs,
                minivol_size=minivol_size,
                minivol_per_epoch=minivol_per_epoch,
            )
            datasets.append(dataset)
            return dataset

        def sampler_factory(weights, num_samples):
            sampler = _FakeSampler(weights, num_samples)
            samplers.append(sampler)
            return sampler

        def dataloader_factory(dataset, **_kwargs):
            loader = _FakeDataLoader(dataset)
            loaders.append(loader)
            return loader

        return (
            dataset_factory,
            sampler_factory,
            dataloader_factory,
            datasets,
            samplers,
            loaders,
        )

    def test_extract_learning_bbox_tensors_builds_entries_in_ui_order_and_stores(self) -> None:
        raw = np.zeros((30, 30, 30), dtype=np.uint8)
        segmentation = np.zeros((30, 30, 30), dtype=np.uint16)
        first = self._make_box(
            box_id="bbox_0011",
            z0=5,
            z1=8,
            y0=6,
            y1=9,
            x0=7,
            x1=10,
            label="validation",
        )
        ignored = self._make_box(
            box_id="bbox_0020",
            z0=2,
            z1=4,
            y0=2,
            y1=4,
            x0=2,
            x1=4,
            label="inference",
        )
        second = self._make_box(
            box_id="bbox_0002",
            z0=10,
            z1=13,
            y0=11,
            y1=14,
            x0=12,
            x1=15,
            label="train",
        )
        boxes_by_id = {first.id: first, ignored.id: ignored, second.id: second}
        ordered_ids = (first.id, ignored.id, second.id)
        raw_calls = []
        seg_calls = []

        def fake_raw_context_extractor(
            _array: np.ndarray,
            *,
            z_bounds: tuple[int, int],
            y_bounds: tuple[int, int],
            x_bounds: tuple[int, int],
        ) -> np.ndarray:
            raw_calls.append((z_bounds, y_bounds, x_bounds))
            return np.full((2, 2, 2), len(raw_calls), dtype=np.float32)

        def fake_segmentation_context_builder(
            _array: np.ndarray,
            *,
            z_bounds: tuple[int, int],
            y_bounds: tuple[int, int],
            x_bounds: tuple[int, int],
            context_fill_value: int,
        ) -> np.ndarray:
            seg_calls.append((z_bounds, y_bounds, x_bounds, int(context_fill_value)))
            return np.full((2, 2, 2), z_bounds[0] + int(context_fill_value), dtype=np.int16)

        batch = extract_learning_bbox_tensors(
            raw,
            segmentation,
            boxes_by_id=boxes_by_id,
            ordered_box_ids=ordered_ids,
            context_fill_value=-100,
            raw_context_extractor=fake_raw_context_extractor,
            segmentation_context_builder=fake_segmentation_context_builder,
        )
        current = get_current_learning_bbox_batch()

        self.assertIs(current, batch)
        self.assertEqual(batch.size, 2)
        self.assertEqual(batch.box_ids, ("bbox_0011", "bbox_0002"))
        self.assertNotIn("bbox_0020", batch.box_ids)
        self.assertEqual(
            raw_calls,
            [
                ((5, 8), (6, 9), (7, 10)),
                ((10, 13), (11, 14), (12, 15)),
            ],
        )
        self.assertEqual(
            seg_calls,
            [
                ((5, 8), (6, 9), (7, 10), -100),
                ((10, 13), (11, 14), (12, 15), -100),
            ],
        )

        first_entry = batch.entries[0]
        second_entry = batch.entries[1]
        self.assertEqual(first_entry.index, 11)
        self.assertEqual(second_entry.index, 2)
        self.assertEqual(first_entry.label, "validation")
        self.assertEqual(second_entry.label, "train")
        self.assertEqual(first_entry.raw_tensor.dtype, torch.float16)
        self.assertEqual(second_entry.raw_tensor.dtype, torch.float16)
        self.assertEqual(first_entry.segmentation_tensor.dtype, torch.int16)
        self.assertEqual(second_entry.segmentation_tensor.dtype, torch.int16)
        self.assertEqual(first_entry.raw_tensor.device.type, "cpu")
        self.assertEqual(first_entry.segmentation_tensor.device.type, "cpu")
        self.assertTrue(
            bool(
                torch.all(first_entry.raw_tensor == torch.tensor(1, dtype=torch.float16)).item()
            )
        )
        self.assertTrue(
            bool(
                torch.all(
                    first_entry.segmentation_tensor == torch.tensor(-95, dtype=torch.int16)
                ).item()
            )
        )

    def test_extract_learning_bbox_tensors_store_in_session_false_keeps_previous_batch(self) -> None:
        seeded_raw = torch.zeros((1, 1, 1), dtype=torch.float16, device="cpu")
        seeded_seg = torch.zeros((1, 1, 1), dtype=torch.int16, device="cpu")
        previous = LearningBBoxTensorEntry(
            box_id="bbox_0009",
            index=9,
            label="inference",
            raw_tensor=seeded_raw,
            segmentation_tensor=seeded_seg,
        )
        previous_batch = set_current_learning_bbox_entries((previous,))

        raw = np.zeros((30, 30, 30), dtype=np.uint8)
        segmentation = np.zeros((30, 30, 30), dtype=np.uint16)
        box = self._make_box(
            box_id="bbox_0001",
            z0=1,
            z1=3,
            y0=2,
            y1=4,
            x0=3,
            x1=5,
            label="train",
        )

        batch = extract_learning_bbox_tensors(
            raw,
            segmentation,
            boxes_by_id={box.id: box},
            ordered_box_ids=(box.id,),
            store_in_session=False,
            raw_context_extractor=lambda *_args, **_kwargs: np.ones((2, 2, 2), dtype=np.uint8),
            segmentation_context_builder=lambda *_args, **_kwargs: np.ones(
                (2, 2, 2), dtype=np.int8
            ),
        )

        current = get_current_learning_bbox_batch()
        self.assertIs(current, previous_batch)
        self.assertEqual(current.box_ids, ("bbox_0009",))
        self.assertEqual(batch.box_ids, ("bbox_0001",))

    def test_extract_learning_bbox_tensors_rejects_duplicate_or_unknown_ids(self) -> None:
        seeded_raw = torch.zeros((1, 1, 1), dtype=torch.float16, device="cpu")
        seeded_seg = torch.zeros((1, 1, 1), dtype=torch.int16, device="cpu")

        previous = LearningBBoxTensorEntry(
            box_id="bbox_0007",
            index=7,
            label="train",
            raw_tensor=seeded_raw,
            segmentation_tensor=seeded_seg,
        )
        previous_batch = set_current_learning_bbox_entries((previous,))

        raw = np.zeros((30, 30, 30), dtype=np.uint8)
        segmentation = np.zeros((30, 30, 30), dtype=np.uint16)
        box = self._make_box(
            box_id="bbox_0001",
            z0=1,
            z1=3,
            y0=1,
            y1=3,
            x0=1,
            x1=3,
            label="train",
        )

        with self.assertRaisesRegex(ValueError, "Duplicate box id"):
            extract_learning_bbox_tensors(
                raw,
                segmentation,
                boxes_by_id={box.id: box},
                ordered_box_ids=(box.id, box.id),
                raw_context_extractor=lambda *_args, **_kwargs: np.zeros(
                    (1, 1, 1), dtype=np.uint8
                ),
                segmentation_context_builder=lambda *_args, **_kwargs: np.zeros(
                    (1, 1, 1), dtype=np.int8
                ),
            )

        with self.assertRaisesRegex(KeyError, "Unknown bounding box id"):
            extract_learning_bbox_tensors(
                raw,
                segmentation,
                boxes_by_id={box.id: box},
                ordered_box_ids=("bbox_9999",),
                raw_context_extractor=lambda *_args, **_kwargs: np.zeros(
                    (1, 1, 1), dtype=np.uint8
                ),
                segmentation_context_builder=lambda *_args, **_kwargs: np.zeros(
                    (1, 1, 1), dtype=np.int8
                ),
            )

        current = get_current_learning_bbox_batch()
        self.assertIs(current, previous_batch)
        self.assertEqual(current.box_ids, ("bbox_0007",))

    def test_extract_learning_bbox_tensors_rejects_invalid_inputs(self) -> None:
        raw = np.zeros((10, 10, 10), dtype=np.uint8)
        segmentation_float = np.zeros((10, 10, 10), dtype=np.float32)
        segmentation_bad_shape = np.zeros((8, 10, 10), dtype=np.uint16)
        box = self._make_box(
            box_id="bbox_0003",
            z0=1,
            z1=2,
            y0=1,
            y1=2,
            x0=1,
            x1=2,
            label="validation",
            volume_shape=(10, 10, 10),
        )

        with self.assertRaisesRegex(ValueError, "integer dtype"):
            extract_learning_bbox_tensors(
                raw,
                segmentation_float,
                boxes_by_id={box.id: box},
                ordered_box_ids=(box.id,),
            )

        with self.assertRaisesRegex(ValueError, "share the same shape"):
            extract_learning_bbox_tensors(
                raw,
                segmentation_bad_shape,
                boxes_by_id={box.id: box},
                ordered_box_ids=(box.id,),
            )

    def test_extract_bboxes_for_learning_builds_tensors_and_dataloaders_in_memory(self) -> None:
        raw = np.zeros((30, 30, 30), dtype=np.uint8)
        segmentation = np.zeros((30, 30, 30), dtype=np.uint16)
        first = self._make_box(
            box_id="bbox_0011",
            z0=5,
            z1=8,
            y0=6,
            y1=9,
            x0=7,
            x1=10,
            label="validation",
        )
        ignored = self._make_box(
            box_id="bbox_0020",
            z0=2,
            z1=4,
            y0=2,
            y1=4,
            x0=2,
            x1=4,
            label="inference",
        )
        second = self._make_box(
            box_id="bbox_0002",
            z0=10,
            z1=13,
            y0=11,
            y1=14,
            x0=12,
            x1=15,
            label="train",
        )
        segmentation[6, 7, 8] = 3  # first box has foreground, second stays zero-only
        boxes_by_id = {first.id: first, ignored.id: ignored, second.id: second}
        ordered_ids = (first.id, ignored.id, second.id)

        def fake_raw_context_extractor(
            _array: np.ndarray,
            *,
            z_bounds: tuple[int, int],
            y_bounds: tuple[int, int],
            x_bounds: tuple[int, int],
        ) -> np.ndarray:
            del y_bounds, x_bounds
            return np.full((2, 2, 2), z_bounds[0], dtype=np.float32)

        def fake_segmentation_context_builder(
            _array: np.ndarray,
            *,
            z_bounds: tuple[int, int],
            y_bounds: tuple[int, int],
            x_bounds: tuple[int, int],
            context_fill_value: int,
        ) -> np.ndarray:
            del y_bounds, x_bounds
            return np.full((2, 2, 2), z_bounds[0] + int(context_fill_value), dtype=np.int16)

        (
            dataset_factory,
            sampler_factory,
            dataloader_factory,
            _datasets,
            _samplers,
            _loaders,
        ) = self._learning_runtime_factories()

        outcome = extract_bboxes_for_learning(
            raw,
            segmentation,
            boxes_by_id=boxes_by_id,
            ordered_box_ids=ordered_ids,
            context_fill_value=-100,
            raw_context_extractor=fake_raw_context_extractor,
            segmentation_context_builder=fake_segmentation_context_builder,
            learning_dataset_factory=dataset_factory,
            learning_sampler_factory=sampler_factory,
            learning_dataloader_factory=dataloader_factory,
        )

        current = get_current_learning_bbox_batch()
        dataloader_runtime = get_current_learning_dataloader_runtime()
        self.assertIsNotNone(current)
        self.assertIsNotNone(dataloader_runtime)
        self.assertEqual(current.size, 2)
        self.assertEqual(current.box_ids, ("bbox_0011", "bbox_0002"))
        self.assertNotIn("bbox_0020", current.box_ids)
        self.assertEqual(dataloader_runtime.train_box_ids, ("bbox_0002",))
        self.assertEqual(dataloader_runtime.train_count, 1)
        self.assertEqual(dataloader_runtime.minivol_size, FIXED_LEARNING_MINIVOL_SIZE)
        self.assertEqual(dataloader_runtime.minivol_per_epoch, 1000)
        self.assertEqual(current.entries[0].label, "validation")
        self.assertEqual(current.entries[1].label, "train")
        self.assertEqual(current.entries[0].raw_tensor.dtype, torch.float16)
        self.assertEqual(current.entries[0].segmentation_tensor.dtype, torch.int16)
        self.assertEqual(outcome.tensor_entry_count, 2)
        self.assertEqual(outcome.raw_saved_paths, tuple())
        self.assertEqual(outcome.raw_failed_boxes, tuple())
        self.assertEqual(outcome.segmentation_saved_paths, tuple())
        self.assertEqual(outcome.segmentation_skipped_zero_box_ids, tuple())
        self.assertEqual(outcome.segmentation_failed_boxes, tuple())
        self.assertEqual(outcome.learning_train_box_ids, ("bbox_0002",))
        self.assertEqual(outcome.learning_batch_size, 4)
        self.assertEqual(outcome.learning_num_workers, 8)

    def test_extract_learning_bboxes_in_memory_accepts_big_endian_raw_context(self) -> None:
        raw = np.zeros((30, 30, 30), dtype=np.uint8)
        segmentation = np.zeros((30, 30, 30), dtype=np.uint16)
        validation_box = self._make_box(
            box_id="bbox_0011",
            z0=5,
            z1=8,
            y0=6,
            y1=9,
            x0=7,
            x1=10,
            label="validation",
        )
        train_box = self._make_box(
            box_id="bbox_0002",
            z0=10,
            z1=13,
            y0=11,
            y1=14,
            x0=12,
            x1=15,
            label="train",
        )
        boxes_by_id = {validation_box.id: validation_box, train_box.id: train_box}
        ordered_ids = (validation_box.id, train_box.id)
        (
            dataset_factory,
            sampler_factory,
            dataloader_factory,
            _datasets,
            _samplers,
            _loaders,
        ) = self._learning_runtime_factories()

        def fake_raw_context_extractor(
            _array: np.ndarray,
            *,
            z_bounds: tuple[int, int],
            y_bounds: tuple[int, int],
            x_bounds: tuple[int, int],
        ) -> np.ndarray:
            del y_bounds, x_bounds
            return np.full((2, 2, 2), z_bounds[0], dtype=np.dtype(">f4"))

        def fake_segmentation_context_builder(
            _array: np.ndarray,
            *,
            z_bounds: tuple[int, int],
            y_bounds: tuple[int, int],
            x_bounds: tuple[int, int],
            context_fill_value: int,
        ) -> np.ndarray:
            del y_bounds, x_bounds
            return np.full((2, 2, 2), z_bounds[0] + int(context_fill_value), dtype=np.int16)

        outcome = extract_learning_bboxes_in_memory(
            raw,
            segmentation,
            boxes_by_id=boxes_by_id,
            ordered_box_ids=ordered_ids,
            context_fill_value=-100,
            raw_context_extractor=fake_raw_context_extractor,
            segmentation_context_builder=fake_segmentation_context_builder,
            learning_dataset_factory=dataset_factory,
            learning_sampler_factory=sampler_factory,
            learning_dataloader_factory=dataloader_factory,
        )

        current = get_current_learning_bbox_batch()
        self.assertIsNotNone(current)
        self.assertEqual(current.size, 2)
        self.assertEqual(current.box_ids, ("bbox_0011", "bbox_0002"))
        self.assertEqual(current.entries[0].raw_tensor.dtype, torch.float16)
        self.assertEqual(current.entries[1].raw_tensor.dtype, torch.float16)
        self.assertEqual(outcome.tensor_entry_count, 2)
        self.assertEqual(outcome.learning_train_box_ids, ("bbox_0002",))

    def test_extract_bboxes_for_learning_builds_eval_runtimes_when_requested(self) -> None:
        raw = np.zeros((30, 30, 30), dtype=np.uint8)
        segmentation = np.zeros((30, 30, 30), dtype=np.uint16)
        train_box = self._make_box(
            box_id="bbox_0002",
            z0=10,
            z1=13,
            y0=11,
            y1=14,
            x0=12,
            x1=15,
            label="train",
        )
        validation_box = self._make_box(
            box_id="bbox_0011",
            z0=5,
            z1=8,
            y0=6,
            y1=9,
            x0=7,
            x1=10,
            label="validation",
        )
        boxes_by_id = {train_box.id: train_box, validation_box.id: validation_box}
        ordered_ids = (validation_box.id, train_box.id)
        (
            dataset_factory,
            sampler_factory,
            dataloader_factory,
            _datasets,
            _samplers,
            _loaders,
        ) = self._learning_runtime_factories()

        with patch(
            "src.io.bbox_tiff_export.build_eval_dataloader_runtimes_from_batch",
            return_value={validation_box.id: object()},
        ) as eval_builder_mock:
            outcome = extract_bboxes_for_learning(
                raw,
                segmentation,
                boxes_by_id=boxes_by_id,
                ordered_box_ids=ordered_ids,
                build_eval_dataloaders=True,
                learning_dataset_factory=dataset_factory,
                learning_sampler_factory=sampler_factory,
                learning_dataloader_factory=dataloader_factory,
            )

        self.assertEqual(outcome.tensor_entry_count, 2)
        self.assertEqual(eval_builder_mock.call_count, 1)
        eval_kwargs = eval_builder_mock.call_args.kwargs
        self.assertEqual(eval_kwargs["minivol_size"], FIXED_LEARNING_MINIVOL_SIZE)
        self.assertEqual(eval_kwargs["batch_size"], 4)
        self.assertEqual(eval_kwargs["num_workers"], 8)
        self.assertTrue(eval_kwargs["pin_memory"])
        self.assertFalse(eval_kwargs["drop_last"])
        self.assertTrue(eval_kwargs["store_in_session"])
        self.assertEqual(outcome.eval_validation_box_ids, (validation_box.id,))
        self.assertEqual(outcome.eval_batch_size, 4)
        self.assertEqual(outcome.eval_num_workers, 8)
        self.assertEqual(outcome.raw_saved_paths, tuple())
        self.assertEqual(outcome.segmentation_saved_paths, tuple())

    def test_extract_bboxes_for_learning_rejects_non_fixed_learning_minivol_size(self) -> None:
        raw = np.zeros((30, 30, 30), dtype=np.uint8)
        segmentation = np.zeros((30, 30, 30), dtype=np.uint16)
        train_box = self._make_box(
            box_id="bbox_0002",
            z0=10,
            z1=13,
            y0=11,
            y1=14,
            x0=12,
            x1=15,
            label="train",
        )
        boxes_by_id = {train_box.id: train_box}
        (
            dataset_factory,
            sampler_factory,
            dataloader_factory,
            _datasets,
            _samplers,
            _loaders,
        ) = self._learning_runtime_factories()

        with self.assertRaisesRegex(ValueError, "learning_minivol_size is fixed to 200"):
            extract_bboxes_for_learning(
                raw,
                segmentation,
                boxes_by_id=boxes_by_id,
                ordered_box_ids=(train_box.id,),
                learning_minivol_size=128,
                learning_dataset_factory=dataset_factory,
                learning_sampler_factory=sampler_factory,
                learning_dataloader_factory=dataloader_factory,
            )

    def test_extract_bboxes_for_learning_rejects_non_fixed_eval_minivol_size(self) -> None:
        raw = np.zeros((30, 30, 30), dtype=np.uint8)
        segmentation = np.zeros((30, 30, 30), dtype=np.uint16)
        validation_box = self._make_box(
            box_id="bbox_0011",
            z0=5,
            z1=8,
            y0=6,
            y1=9,
            x0=7,
            x1=10,
            label="validation",
        )
        boxes_by_id = {validation_box.id: validation_box}

        with self.assertRaisesRegex(ValueError, "eval_minivol_size is fixed to 200"):
            extract_bboxes_for_learning(
                raw,
                segmentation,
                boxes_by_id=boxes_by_id,
                ordered_box_ids=(validation_box.id,),
                build_learning_dataloader=False,
                build_eval_dataloaders=True,
                eval_minivol_size=128,
            )

    def test_extract_bboxes_for_learning_aborts_when_eval_builder_raises(self) -> None:
        raw = np.zeros((30, 30, 30), dtype=np.uint8)
        segmentation = np.zeros((30, 30, 30), dtype=np.uint16)
        train_box = self._make_box(
            box_id="bbox_0002",
            z0=10,
            z1=13,
            y0=11,
            y1=14,
            x0=12,
            x1=15,
            label="train",
        )
        validation_box = self._make_box(
            box_id="bbox_0011",
            z0=5,
            z1=8,
            y0=6,
            y1=9,
            x0=7,
            x1=10,
            label="validation",
        )
        boxes_by_id = {train_box.id: train_box, validation_box.id: validation_box}
        ordered_ids = (validation_box.id, train_box.id)
        raw_save_calls = []
        seg_save_calls = []
        (
            dataset_factory,
            sampler_factory,
            dataloader_factory,
            _datasets,
            _samplers,
            _loaders,
        ) = self._learning_runtime_factories()

        with patch(
            "src.io.bbox_tiff_export.build_eval_dataloader_runtimes_from_batch",
            side_effect=RuntimeError("eval build boom"),
        ):
            with self.assertRaisesRegex(RuntimeError, "eval build boom"):
                extract_bboxes_for_learning(
                    raw,
                    segmentation,
                    boxes_by_id=boxes_by_id,
                    ordered_box_ids=ordered_ids,
                    output_dir="/tmp/out",
                    overwrite=True,
                    build_eval_dataloaders=True,
                    learning_dataset_factory=dataset_factory,
                    learning_sampler_factory=sampler_factory,
                    learning_dataloader_factory=dataloader_factory,
                    raw_saver=lambda *_args, **_kwargs: raw_save_calls.append(1)
                    or "/tmp/out/raw.tif",
                    segmentation_saver=lambda *_args, **_kwargs: seg_save_calls.append(1)
                    or "/tmp/out/seg.tif",
                )

        self.assertEqual(raw_save_calls, [])
        self.assertEqual(seg_save_calls, [])
        self.assertIsNone(get_current_learning_dataloader_runtime())

    def test_extract_bboxes_for_learning_rollback_disposes_new_learning_runtime_on_eval_failure(
        self,
    ) -> None:
        class _FakeIterator:
            def __init__(self) -> None:
                self.shutdown_calls = 0

            def _shutdown_workers(self) -> None:
                self.shutdown_calls += 1

        class _FakeDataLoader:
            def __init__(self) -> None:
                self._iterator = _FakeIterator()
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        class _FakeDataset:
            def __init__(self, tensor_pairs, *, minivol_size, minivol_per_epoch):
                self.tensor_pairs = tuple(tensor_pairs)
                self.minivol_size = int(minivol_size)
                self.minivol_per_epoch = int(minivol_per_epoch)
                self.weights = [1 for _ in self.tensor_pairs]
                self.close_calls = 0

            def __len__(self):
                return int(self.minivol_per_epoch)

            def close(self) -> None:
                self.close_calls += 1

        class _FakeSampler:
            def __init__(self, weights, num_samples):
                self.weights = tuple(weights)
                self.num_samples = int(num_samples)
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        datasets = []
        samplers = []
        loaders = []

        def dataset_factory(tensor_pairs, *, minivol_size, minivol_per_epoch):
            dataset = _FakeDataset(
                tensor_pairs,
                minivol_size=minivol_size,
                minivol_per_epoch=minivol_per_epoch,
            )
            datasets.append(dataset)
            return dataset

        def sampler_factory(weights, num_samples):
            sampler = _FakeSampler(weights, num_samples)
            samplers.append(sampler)
            return sampler

        def dataloader_factory(dataset, **_kwargs):
            del dataset
            loader = _FakeDataLoader()
            loaders.append(loader)
            return loader

        raw = np.zeros((30, 30, 30), dtype=np.uint8)
        segmentation = np.zeros((30, 30, 30), dtype=np.uint16)
        train_box = self._make_box(
            box_id="bbox_0002",
            z0=10,
            z1=13,
            y0=11,
            y1=14,
            x0=12,
            x1=15,
            label="train",
        )
        validation_box = self._make_box(
            box_id="bbox_0011",
            z0=5,
            z1=8,
            y0=6,
            y1=9,
            x0=7,
            x1=10,
            label="validation",
        )
        boxes_by_id = {train_box.id: train_box, validation_box.id: validation_box}
        ordered_ids = (validation_box.id, train_box.id)

        with patch(
            "src.io.bbox_tiff_export.build_eval_dataloader_runtimes_from_batch",
            side_effect=RuntimeError("eval build boom"),
        ):
            with self.assertRaisesRegex(RuntimeError, "eval build boom"):
                extract_bboxes_for_learning(
                    raw,
                    segmentation,
                    boxes_by_id=boxes_by_id,
                    ordered_box_ids=ordered_ids,
                    output_dir="/tmp/out",
                    overwrite=True,
                    build_eval_dataloaders=True,
                    learning_dataset_factory=dataset_factory,
                    learning_sampler_factory=sampler_factory,
                    learning_dataloader_factory=dataloader_factory,
                    raw_saver=lambda *_args, **_kwargs: "/tmp/out/raw.tif",
                    segmentation_saver=lambda *_args, **_kwargs: "/tmp/out/seg.tif",
                )

        self.assertIsNone(get_current_learning_dataloader_runtime())
        self.assertEqual(len(datasets), 1)
        self.assertEqual(len(samplers), 1)
        self.assertEqual(len(loaders), 1)
        self.assertEqual(loaders[0]._iterator.shutdown_calls, 1)
        self.assertEqual(loaders[0].close_calls, 1)
        self.assertEqual(datasets[0].close_calls, 1)
        self.assertEqual(samplers[0].close_calls, 1)

    def test_extract_bboxes_for_learning_clears_previous_eval_runtimes_on_failure(self) -> None:
        class _FakeIterator:
            def __init__(self) -> None:
                self.shutdown_calls = 0

            def _shutdown_workers(self) -> None:
                self.shutdown_calls += 1

        class _FakeDataLoader:
            def __init__(self) -> None:
                self._iterator = _FakeIterator()
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        class _FakeBuffer:
            def __init__(self) -> None:
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        previous_loader = _FakeDataLoader()
        previous_iterator = previous_loader._iterator
        previous_buffer = _FakeBuffer()
        previous_runtime = LearningBBoxEvalRuntime(
            box_id="bbox_0999",
            dataloader=previous_loader,
            buffer=previous_buffer,
        )
        set_current_learning_eval_runtimes_by_box_id({"bbox_0999": previous_runtime})

        raw = np.zeros((20, 20, 20), dtype=np.uint8)
        segmentation = np.zeros((20, 20, 20), dtype=np.uint16)
        train_box = self._make_box(
            box_id="bbox_0005",
            z0=2,
            z1=5,
            y0=3,
            y1=6,
            x0=4,
            x1=7,
            label="train",
            volume_shape=(20, 20, 20),
        )

        with self.assertRaisesRegex(ValueError, "No validation bounding boxes"):
            extract_bboxes_for_learning(
                raw,
                segmentation,
                boxes_by_id={train_box.id: train_box},
                ordered_box_ids=(train_box.id,),
                output_dir="/tmp/out",
                overwrite=True,
                build_learning_dataloader=False,
                build_eval_dataloaders=True,
                raw_saver=lambda *_args, **_kwargs: "/tmp/out/raw.tif",
                segmentation_saver=lambda *_args, **_kwargs: "/tmp/out/seg.tif",
            )

        self.assertEqual(get_current_learning_eval_runtimes_by_box_id(), {})
        self.assertEqual(previous_iterator.shutdown_calls, 1)
        self.assertEqual(previous_loader.close_calls, 1)
        self.assertEqual(previous_buffer.close_calls, 1)

    def test_extract_bboxes_for_learning_rejects_when_no_train_bboxes(self) -> None:
        raw = np.zeros((30, 30, 30), dtype=np.uint8)
        segmentation = np.zeros((30, 30, 30), dtype=np.uint16)
        validation_box = self._make_box(
            box_id="bbox_0011",
            z0=5,
            z1=8,
            y0=6,
            y1=9,
            x0=7,
            x1=10,
            label="validation",
        )
        inference_box = self._make_box(
            box_id="bbox_0002",
            z0=10,
            z1=13,
            y0=11,
            y1=14,
            x0=12,
            x1=15,
            label="inference",
        )
        boxes_by_id = {validation_box.id: validation_box, inference_box.id: inference_box}
        ordered_ids = (validation_box.id, inference_box.id)
        raw_save_calls = []
        seg_save_calls = []

        def fake_raw_saver(
            array: np.ndarray,
            *,
            output_dir: str,
            index: int,
            label: str,
            overwrite: bool,
        ) -> str:
            del array, output_dir, index, label, overwrite
            raw_save_calls.append(1)
            return "/tmp/out/raw.tif"

        def fake_segmentation_saver(
            array: np.ndarray,
            *,
            output_dir: str,
            index: int,
            label: str,
            overwrite: bool,
        ) -> str:
            del array, output_dir, index, label, overwrite
            seg_save_calls.append(1)
            return "/tmp/out/seg.tif"

        with self.assertRaisesRegex(ValueError, "No bounding boxes labeled 'train'"):
            extract_bboxes_for_learning(
                raw,
                segmentation,
                boxes_by_id=boxes_by_id,
                ordered_box_ids=ordered_ids,
                output_dir="/tmp/out",
                overwrite=True,
                raw_saver=fake_raw_saver,
                segmentation_saver=fake_segmentation_saver,
            )

        self.assertEqual(raw_save_calls, [])
        self.assertEqual(seg_save_calls, [])

    def test_extract_bboxes_for_learning_clears_previous_runtime_on_failure(self) -> None:
        class _FakeIterator:
            def __init__(self) -> None:
                self.shutdown_calls = 0

            def _shutdown_workers(self) -> None:
                self.shutdown_calls += 1

        class _FakeDataLoader:
            def __init__(self) -> None:
                self._iterator = _FakeIterator()
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        class _FakeClosable:
            def __init__(self) -> None:
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        previous_dataset = _FakeClosable()
        previous_sampler = _FakeClosable()
        previous_loader = _FakeDataLoader()
        previous_iterator = previous_loader._iterator
        previous_runtime = LearningBBoxDataLoaderRuntime(
            dataset=previous_dataset,
            sampler=previous_sampler,
            dataloader=previous_loader,
            train_box_ids=("bbox_9999",),
            batch_size=4,
            num_workers=8,
        )
        set_current_learning_dataloader_runtime(previous_runtime)

        raw = np.zeros((20, 20, 20), dtype=np.uint8)
        segmentation = np.zeros((20, 20, 20), dtype=np.uint16)
        validation_box = self._make_box(
            box_id="bbox_0005",
            z0=2,
            z1=5,
            y0=3,
            y1=6,
            x0=4,
            x1=7,
            label="validation",
            volume_shape=(20, 20, 20),
        )
        boxes_by_id = {validation_box.id: validation_box}

        with self.assertRaisesRegex(ValueError, "No bounding boxes labeled 'train'"):
            extract_bboxes_for_learning(
                raw,
                segmentation,
                boxes_by_id=boxes_by_id,
                ordered_box_ids=(validation_box.id,),
                output_dir="/tmp/out",
                overwrite=True,
            )

        self.assertIsNone(get_current_learning_dataloader_runtime())
        self.assertEqual(previous_iterator.shutdown_calls, 1)
        self.assertEqual(previous_loader.close_calls, 1)
        self.assertEqual(previous_dataset.close_calls, 1)
        self.assertEqual(previous_sampler.close_calls, 1)

    def test_extract_bboxes_for_learning_replaces_previous_runtime_on_repeated_calls(self) -> None:
        class _FakeIterator:
            def __init__(self) -> None:
                self.shutdown_calls = 0

            def _shutdown_workers(self) -> None:
                self.shutdown_calls += 1

        class _FakeDataLoader:
            def __init__(self) -> None:
                self._iterator = _FakeIterator()
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        class _FakeDataset:
            def __init__(self, tensor_pairs, *, minivol_size, minivol_per_epoch):
                self.tensor_pairs = tuple(tensor_pairs)
                self.minivol_size = int(minivol_size)
                self.minivol_per_epoch = int(minivol_per_epoch)
                self.weights = [1 for _ in self.tensor_pairs]
                self.close_calls = 0

            def __len__(self):
                return int(self.minivol_per_epoch)

            def close(self) -> None:
                self.close_calls += 1

        class _FakeSampler:
            def __init__(self, weights, num_samples):
                self.weights = tuple(weights)
                self.num_samples = int(num_samples)
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        datasets = []
        samplers = []
        loaders = []

        def dataset_factory(tensor_pairs, *, minivol_size, minivol_per_epoch):
            ds = _FakeDataset(
                tensor_pairs,
                minivol_size=minivol_size,
                minivol_per_epoch=minivol_per_epoch,
            )
            datasets.append(ds)
            return ds

        def sampler_factory(weights, num_samples):
            sampler = _FakeSampler(weights, num_samples)
            samplers.append(sampler)
            return sampler

        def dataloader_factory(dataset, **_kwargs):
            del dataset
            loader = _FakeDataLoader()
            loaders.append(loader)
            return loader

        raw = np.zeros((30, 30, 30), dtype=np.uint8)
        segmentation = np.zeros((30, 30, 30), dtype=np.uint16)

        first_box = self._make_box(
            box_id="bbox_0002",
            z0=5,
            z1=8,
            y0=6,
            y1=9,
            x0=7,
            x1=10,
            label="train",
        )
        second_box = self._make_box(
            box_id="bbox_0011",
            z0=10,
            z1=13,
            y0=11,
            y1=14,
            x0=12,
            x1=15,
            label="train",
        )
        segmentation[6, 7, 8] = 3
        segmentation[11, 12, 13] = 4

        outcome_first = extract_bboxes_for_learning(
            raw,
            segmentation,
            boxes_by_id={first_box.id: first_box},
            ordered_box_ids=(first_box.id,),
            output_dir="/tmp/out",
            overwrite=True,
            learning_dataset_factory=dataset_factory,
            learning_sampler_factory=sampler_factory,
            learning_dataloader_factory=dataloader_factory,
            raw_saver=lambda *_args, **_kwargs: "/tmp/out/bbox2_train.tif",
            segmentation_saver=lambda *_args, **_kwargs: "/tmp/out/bbox2_train_seg.tif",
        )

        first_runtime = get_current_learning_dataloader_runtime()
        self.assertIsNotNone(first_runtime)
        self.assertEqual(first_runtime.train_box_ids, ("bbox_0002",))
        self.assertEqual(outcome_first.learning_train_box_ids, ("bbox_0002",))

        outcome_second = extract_bboxes_for_learning(
            raw,
            segmentation,
            boxes_by_id={second_box.id: second_box},
            ordered_box_ids=(second_box.id,),
            output_dir="/tmp/out",
            overwrite=True,
            learning_dataset_factory=dataset_factory,
            learning_sampler_factory=sampler_factory,
            learning_dataloader_factory=dataloader_factory,
            raw_saver=lambda *_args, **_kwargs: "/tmp/out/bbox11_train.tif",
            segmentation_saver=lambda *_args, **_kwargs: "/tmp/out/bbox11_train_seg.tif",
        )

        current_runtime = get_current_learning_dataloader_runtime()
        self.assertIsNotNone(current_runtime)
        self.assertEqual(current_runtime.train_box_ids, ("bbox_0011",))
        self.assertEqual(outcome_second.learning_train_box_ids, ("bbox_0011",))
        self.assertEqual(len(datasets), 2)
        self.assertEqual(len(samplers), 2)
        self.assertEqual(len(loaders), 2)
        self.assertEqual(datasets[0].minivol_per_epoch, 1000)
        self.assertEqual(datasets[1].minivol_per_epoch, 1000)

        # Previous runtime components are disposed when replaced on the second call.
        self.assertEqual(loaders[0]._iterator.shutdown_calls, 1)
        self.assertEqual(loaders[0].close_calls, 1)
        self.assertEqual(datasets[0].close_calls, 1)
        self.assertEqual(samplers[0].close_calls, 1)

    def test_extract_bboxes_for_learning_sets_fixed_default_minivol_per_epoch(self) -> None:
        raw = np.zeros((30, 30, 30), dtype=np.uint8)
        segmentation = np.zeros((30, 30, 30), dtype=np.uint16)
        first = self._make_box(
            box_id="bbox_0002",
            z0=5,
            z1=8,
            y0=6,
            y1=9,
            x0=7,
            x1=10,
            label="train",
        )
        second = self._make_box(
            box_id="bbox_0011",
            z0=10,
            z1=13,
            y0=11,
            y1=14,
            x0=12,
            x1=15,
            label="train",
        )
        segmentation[6, 7, 8] = 3
        segmentation[11, 12, 13] = 5
        (
            dataset_factory,
            sampler_factory,
            dataloader_factory,
            _datasets,
            _samplers,
            _loaders,
        ) = self._learning_runtime_factories()

        extract_bboxes_for_learning(
            raw,
            segmentation,
            boxes_by_id={first.id: first, second.id: second},
            ordered_box_ids=(first.id, second.id),
            output_dir="/tmp/out",
            overwrite=True,
            learning_dataset_factory=dataset_factory,
            learning_sampler_factory=sampler_factory,
            learning_dataloader_factory=dataloader_factory,
            raw_saver=lambda *_args, **_kwargs: "/tmp/out/raw.tif",
            segmentation_saver=lambda *_args, **_kwargs: "/tmp/out/seg.tif",
        )

        runtime = get_current_learning_dataloader_runtime()
        self.assertIsNotNone(runtime)
        self.assertEqual(runtime.train_count, 2)
        self.assertEqual(runtime.minivol_size, FIXED_LEARNING_MINIVOL_SIZE)
        self.assertEqual(runtime.minivol_per_epoch, 1000)


if __name__ == "__main__":
    unittest.main()
