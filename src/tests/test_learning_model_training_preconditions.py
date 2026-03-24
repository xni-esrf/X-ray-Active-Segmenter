from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]

from src.learning import (
    LearningBBoxEvalRuntime,
    clear_current_learning_dataloader_runtime,
    clear_current_learning_eval_runtimes_by_box_id,
    clear_current_learning_model_runtime,
    set_current_learning_dataloader_components,
    set_current_learning_eval_runtimes_by_box_id,
    set_current_learning_model_components,
    validate_learning_model_training_preconditions,
)


class LearningModelTrainingPreconditionsTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_current_learning_model_runtime()
        clear_current_learning_dataloader_runtime()
        clear_current_learning_eval_runtimes_by_box_id()

    def tearDown(self) -> None:
        clear_current_learning_model_runtime()
        clear_current_learning_dataloader_runtime()
        clear_current_learning_eval_runtimes_by_box_id()

    @staticmethod
    def _set_model_runtime() -> None:
        set_current_learning_model_components(
            model=object(),
            optimizer=object(),
            checkpoint_path="foundation_model/weights_epoch_190.cp",
            device_ids=(0, 1),
            num_classes=2,
        )

    @staticmethod
    def _set_train_runtime(*, class_weights: object = None) -> None:
        set_current_learning_dataloader_components(
            dataset=object(),
            sampler=object(),
            dataloader=object(),
            train_box_ids=("bbox_0001",),
            class_weights=class_weights,
        )

    @staticmethod
    def _set_eval_runtimes(*, first_ground_truth, second_ground_truth=None) -> None:
        runtimes = {
            "bbox_0008": LearningBBoxEvalRuntime(
                box_id="bbox_0008",
                dataloader=object(),
                buffer=SimpleNamespace(ground_truth=first_ground_truth),
            )
        }
        if second_ground_truth is not None:
            runtimes["bbox_0011"] = LearningBBoxEvalRuntime(
                box_id="bbox_0011",
                dataloader=object(),
                buffer=SimpleNamespace(ground_truth=second_ground_truth),
            )
        set_current_learning_eval_runtimes_by_box_id(runtimes)

    def test_validate_preconditions_reports_missing_items_in_one_error(self) -> None:
        self._set_train_runtime()

        with self.assertRaisesRegex(ValueError, "required learning state is missing"):
            validate_learning_model_training_preconditions()

        try:
            validate_learning_model_training_preconditions()
        except ValueError as exc:
            message = str(exc)
        else:  # pragma: no cover - defensive
            self.fail("Expected ValueError")

        self.assertIn("model runtime", message)
        self.assertIn("evaluation runtimes/buffers", message)
        self.assertIn("class weights", message)

    def test_validate_preconditions_reports_missing_class_weights(self) -> None:
        self._set_model_runtime()
        self._set_train_runtime()
        self._set_eval_runtimes(
            first_ground_truth=np.array([[[0, 1], [2, -100]]], dtype=np.int16)
        )

        with self.assertRaisesRegex(ValueError, "class weights"):
            validate_learning_model_training_preconditions()

    @unittest.skipUnless(torch is not None, "PyTorch is not available")
    def test_validate_preconditions_resolves_validation_valid_voxel_counts(self) -> None:
        self._set_model_runtime()
        self._set_train_runtime(
            class_weights=torch.tensor([1.0, 2.0], dtype=torch.float32)
        )
        self._set_eval_runtimes(
            first_ground_truth=np.array([[[0, 1], [2, -100]]], dtype=np.int16),
            second_ground_truth=np.array([[[1, -100], [3, 4]]], dtype=np.int16),
        )

        preconditions = validate_learning_model_training_preconditions()

        self.assertEqual(preconditions.validation_valid_voxel_counts_by_box_id["bbox_0008"], 3)
        self.assertEqual(preconditions.validation_valid_voxel_counts_by_box_id["bbox_0011"], 3)
        self.assertEqual(preconditions.total_validation_valid_voxel_count, 6)
        self.assertEqual(tuple(sorted(preconditions.eval_runtimes_by_box_id.keys())), ("bbox_0008", "bbox_0011"))

    def test_validate_preconditions_rejects_validation_buffer_with_zero_valid_voxels(self) -> None:
        self._set_model_runtime()
        self._set_train_runtime()
        self._set_eval_runtimes(
            first_ground_truth=np.array([[[-100, -100], [-100, -100]]], dtype=np.int16)
        )

        with self.assertRaisesRegex(ValueError, "no valid voxels"):
            validate_learning_model_training_preconditions(require_class_weights=False)

    def test_validate_preconditions_rejects_validation_buffer_without_ground_truth(self) -> None:
        self._set_model_runtime()
        self._set_train_runtime()
        set_current_learning_eval_runtimes_by_box_id(
            {
                "bbox_0008": LearningBBoxEvalRuntime(
                    box_id="bbox_0008",
                    dataloader=object(),
                    buffer=SimpleNamespace(),
                )
            }
        )

        with self.assertRaisesRegex(ValueError, "does not expose 'ground_truth'"):
            validate_learning_model_training_preconditions(require_class_weights=False)


if __name__ == "__main__":
    unittest.main()

