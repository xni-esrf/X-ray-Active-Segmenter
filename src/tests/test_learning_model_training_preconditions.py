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
    LearningLabelSpace,
    LearningSession,
    clear_current_learning_dataloader_runtime,
    clear_current_learning_eval_runtimes_by_box_id,
    clear_current_learning_label_space,
    clear_current_learning_model_runtime,
    set_current_learning_dataloader_components,
    set_current_learning_eval_runtimes_by_box_id,
    set_current_learning_model_components,
    validate_learning_model_training_preconditions,
)


class LearningModelTrainingPreconditionsTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_current_learning_label_space()
        clear_current_learning_model_runtime()
        clear_current_learning_dataloader_runtime()
        clear_current_learning_eval_runtimes_by_box_id()

    def tearDown(self) -> None:
        clear_current_learning_label_space()
        clear_current_learning_model_runtime()
        clear_current_learning_dataloader_runtime()
        clear_current_learning_eval_runtimes_by_box_id()

    @staticmethod
    def _set_model_runtime(*, num_classes: int = 2) -> None:
        set_current_learning_model_components(
            model=object(),
            optimizer=object(),
            checkpoint_path="foundation_model/weights_epoch_190.cp",
            device_ids=(0, 1),
            num_classes=int(num_classes),
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
    def test_validate_preconditions_rejects_class_weights_that_do_not_match_model_classes(
        self,
    ) -> None:
        self._set_model_runtime(num_classes=2)
        self._set_train_runtime(
            class_weights=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        )
        self._set_eval_runtimes(
            first_ground_truth=np.array([[[0, 1], [2, -100]]], dtype=np.int16)
        )

        with self.assertRaisesRegex(ValueError, "model num_classes: 2"):
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

    @unittest.skipUnless(torch is not None, "PyTorch is not available")
    def test_validate_preconditions_can_read_explicit_learning_session(self) -> None:
        session = LearningSession()
        session.set_model_components(
            model=object(),
            optimizer=object(),
            checkpoint_path="foundation_model/weights_epoch_190.cp",
            device_ids=(0, 1),
            num_classes=2,
        )
        session.set_dataloader_components(
            dataset=object(),
            sampler=object(),
            dataloader=object(),
            train_box_ids=("bbox_0001",),
            class_weights=torch.tensor([1.0, 2.0], dtype=torch.float32),
        )
        session.set_eval_runtimes_by_box_id(
            {
                "bbox_0008": LearningBBoxEvalRuntime(
                    box_id="bbox_0008",
                    dataloader=object(),
                    buffer=SimpleNamespace(
                        ground_truth=np.array([[[0, 1], [2, -100]]], dtype=np.int16)
                    ),
                )
            }
        )

        preconditions = validate_learning_model_training_preconditions(
            learning_session=session,
        )

        self.assertEqual(preconditions.validation_valid_voxel_counts_by_box_id["bbox_0008"], 3)
        self.assertEqual(preconditions.total_validation_valid_voxel_count, 3)

    @unittest.skipUnless(torch is not None, "PyTorch is not available")
    def test_validate_preconditions_accepts_consistent_label_space(self) -> None:
        session = LearningSession()
        label_space = LearningLabelSpace(label_values=(0, 1, 2))
        session.set_label_space(label_space)
        session.set_model_components(
            model=object(),
            optimizer=object(),
            checkpoint_path="foundation_model/weights_epoch_190.cp",
            device_ids=(0, 1),
            num_classes=3,
            hyperparameters={"label_values": (0, 1, 2)},
        )
        session.set_dataloader_components(
            dataset=object(),
            sampler=object(),
            dataloader=object(),
            train_box_ids=("bbox_0001",),
            class_weights=torch.tensor([1.0, 2.0, 100.0], dtype=torch.float32),
        )
        session.set_eval_runtimes_by_box_id(
            {
                "bbox_0008": LearningBBoxEvalRuntime(
                    box_id="bbox_0008",
                    dataloader=object(),
                    buffer=SimpleNamespace(
                        label_values=(0, 1, 2),
                        ground_truth=np.array([[[0, 1], [2, -100]]], dtype=np.int16),
                    ),
                )
            }
        )

        preconditions = validate_learning_model_training_preconditions(
            learning_session=session,
        )

        self.assertIs(preconditions.label_space, label_space)

    @unittest.skipUnless(torch is not None, "PyTorch is not available")
    def test_validate_preconditions_rejects_model_num_classes_mismatch_with_label_space(
        self,
    ) -> None:
        session = LearningSession()
        session.set_label_space(LearningLabelSpace(label_values=(0, 1, 2)))
        session.set_model_components(
            model=object(),
            optimizer=object(),
            checkpoint_path="foundation_model/weights_epoch_190.cp",
            device_ids=(0, 1),
            num_classes=2,
        )
        session.set_dataloader_components(
            dataset=object(),
            sampler=object(),
            dataloader=object(),
            train_box_ids=("bbox_0001",),
            class_weights=torch.tensor([1.0, 2.0], dtype=torch.float32),
        )
        session.set_eval_runtimes_by_box_id(
            {
                "bbox_0008": LearningBBoxEvalRuntime(
                    box_id="bbox_0008",
                    dataloader=object(),
                    buffer=SimpleNamespace(
                        label_values=(0, 1, 2),
                        ground_truth=np.array([[[0, 1], [2, -100]]], dtype=np.int16),
                    ),
                )
            }
        )

        with self.assertRaisesRegex(ValueError, "label space num_classes: 3"):
            validate_learning_model_training_preconditions(learning_session=session)

    @unittest.skipUnless(torch is not None, "PyTorch is not available")
    def test_validate_preconditions_rejects_model_missing_label_values_with_label_space(
        self,
    ) -> None:
        session = LearningSession()
        session.set_label_space(LearningLabelSpace(label_values=(0, 1, 2)))
        session.set_model_components(
            model=object(),
            optimizer=object(),
            checkpoint_path="foundation_model/weights_epoch_190.cp",
            device_ids=(0, 1),
            num_classes=3,
            hyperparameters={},
        )
        session.set_dataloader_components(
            dataset=object(),
            sampler=object(),
            dataloader=object(),
            train_box_ids=("bbox_0001",),
            class_weights=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        )
        session.set_eval_runtimes_by_box_id(
            {
                "bbox_0008": LearningBBoxEvalRuntime(
                    box_id="bbox_0008",
                    dataloader=object(),
                    buffer=SimpleNamespace(
                        label_values=(0, 1, 2),
                        ground_truth=np.array([[[0, 1], [2, -100]]], dtype=np.int16),
                    ),
                )
            }
        )

        with self.assertRaisesRegex(ValueError, "label_values metadata is missing"):
            validate_learning_model_training_preconditions(learning_session=session)

    @unittest.skipUnless(torch is not None, "PyTorch is not available")
    def test_validate_preconditions_rejects_model_label_space_metadata_mismatch(
        self,
    ) -> None:
        session = LearningSession()
        session.set_label_space(LearningLabelSpace(label_values=(0, 1, 2)))
        session.set_model_components(
            model=object(),
            optimizer=object(),
            checkpoint_path="foundation_model/weights_epoch_190.cp",
            device_ids=(0, 1),
            num_classes=3,
            hyperparameters={
                "label_values": (0, 1, 2),
                "label_space": {
                    "label_values": (0, 1, 2),
                    "background_label": 0,
                    "mask_label": -1,
                },
            },
        )
        session.set_dataloader_components(
            dataset=object(),
            sampler=object(),
            dataloader=object(),
            train_box_ids=("bbox_0001",),
            class_weights=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        )
        session.set_eval_runtimes_by_box_id(
            {
                "bbox_0008": LearningBBoxEvalRuntime(
                    box_id="bbox_0008",
                    dataloader=object(),
                    buffer=SimpleNamespace(
                        label_values=(0, 1, 2),
                        ground_truth=np.array([[[0, 1], [2, -100]]], dtype=np.int16),
                    ),
                )
            }
        )

        with self.assertRaisesRegex(ValueError, "label_space.mask_label"):
            validate_learning_model_training_preconditions(learning_session=session)

    @unittest.skipUnless(torch is not None, "PyTorch is not available")
    def test_validate_preconditions_rejects_eval_labels_outside_label_space(self) -> None:
        session = LearningSession()
        session.set_label_space(LearningLabelSpace(label_values=(0, 1, 2)))
        session.set_model_components(
            model=object(),
            optimizer=object(),
            checkpoint_path="foundation_model/weights_epoch_190.cp",
            device_ids=(0, 1),
            num_classes=3,
            hyperparameters={"label_values": (0, 1, 2)},
        )
        session.set_dataloader_components(
            dataset=object(),
            sampler=object(),
            dataloader=object(),
            train_box_ids=("bbox_0001",),
            class_weights=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        )
        session.set_eval_runtimes_by_box_id(
            {
                "bbox_0008": LearningBBoxEvalRuntime(
                    box_id="bbox_0008",
                    dataloader=object(),
                    buffer=SimpleNamespace(
                        label_values=(0, 1, 2),
                        ground_truth=np.array([[[0, 9], [2, -100]]], dtype=np.int16),
                    ),
                )
            }
        )

        with self.assertRaisesRegex(ValueError, "outside the current label space"):
            validate_learning_model_training_preconditions(learning_session=session)

    @unittest.skipUnless(torch is not None, "PyTorch is not available")
    def test_validate_preconditions_rejects_eval_buffer_label_values_mismatch(
        self,
    ) -> None:
        session = LearningSession()
        session.set_label_space(LearningLabelSpace(label_values=(0, 1, 2)))
        session.set_model_components(
            model=object(),
            optimizer=object(),
            checkpoint_path="foundation_model/weights_epoch_190.cp",
            device_ids=(0, 1),
            num_classes=3,
            hyperparameters={"label_values": (0, 1, 2)},
        )
        session.set_dataloader_components(
            dataset=object(),
            sampler=object(),
            dataloader=object(),
            train_box_ids=("bbox_0001",),
            class_weights=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        )
        session.set_eval_runtimes_by_box_id(
            {
                "bbox_0008": LearningBBoxEvalRuntime(
                    box_id="bbox_0008",
                    dataloader=object(),
                    buffer=SimpleNamespace(
                        label_values=(0, 2, 1),
                        ground_truth=np.array([[[0, 1], [2, -100]]], dtype=np.int16),
                    ),
                )
            }
        )

        with self.assertRaisesRegex(ValueError, "label_values do not match"):
            validate_learning_model_training_preconditions(learning_session=session)

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
