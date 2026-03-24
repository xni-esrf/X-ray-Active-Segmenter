from __future__ import annotations

import unittest
from threading import Event
from unittest.mock import patch

try:
    import torch
except Exception:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]

from src.learning import (
    LearningBBoxDataLoaderRuntime,
    LearningBBoxEvalRuntime,
    LearningModelRuntime,
    LearningTrainEpochResult,
    LearningTrainingPreconditions,
    LearningValidationEvalResult,
    train_learning_model_with_validation_loop,
)


@unittest.skipUnless(torch is not None, "PyTorch is not available")
class LearningModelTrainingLoopTests(unittest.TestCase):
    @staticmethod
    def _make_preconditions(*, train_box_count: int) -> LearningTrainingPreconditions:
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.0)

        model_runtime = LearningModelRuntime(
            model=model,
            optimizer=object(),
            checkpoint_path="foundation_model/weights_epoch_190.cp",
            device_ids=(0,),
            num_classes=2,
            hyperparameters={"lr": 0.001, "lwise_lr_decay": 0.8},
        )
        train_runtime = LearningBBoxDataLoaderRuntime(
            dataset=object(),
            sampler=object(),
            dataloader=object(),
            train_box_ids=tuple(f"bbox_{i:04d}" for i in range(int(train_box_count))),
            class_weights=torch.tensor([1.0, 2.0], dtype=torch.float32),
        )
        eval_runtime = LearningBBoxEvalRuntime(
            box_id="bbox_eval_0001",
            dataloader=object(),
            buffer=object(),
        )
        return LearningTrainingPreconditions(
            model_runtime=model_runtime,
            train_runtime=train_runtime,
            eval_runtimes_by_box_id={"bbox_eval_0001": eval_runtime},
            class_weights=train_runtime.class_weights,
            validation_valid_voxel_counts_by_box_id={"bbox_eval_0001": 10},
            total_validation_valid_voxel_count=10,
        )

    def test_training_loop_stops_early_and_restores_best_weights(self) -> None:
        preconditions = self._make_preconditions(train_box_count=3)
        model = preconditions.model_runtime.model

        def _train_side_effect(**kwargs):
            epoch_index = int(kwargs["epoch_index"])
            with torch.no_grad():
                model.weight.fill_(float(epoch_index + 1))
            return (
                LearningTrainEpochResult(
                    epoch_index=epoch_index,
                    total_epoch_count=int(kwargs["total_epoch_count"]),
                    base_learning_rate=0.001,
                    num_batches=1,
                    mean_loss=1.0,
                    mixed_precision_used=False,
                ),
                object(),
            )

        eval_values = iter((0.50, 0.60, 0.55, 0.54))

        def _eval_side_effect(**_kwargs):
            value = float(next(eval_values))
            return LearningValidationEvalResult(
                weighted_mean_accuracy=value,
                per_box_accuracy_by_box_id={"bbox_eval_0001": value},
                valid_voxel_counts_by_box_id={"bbox_eval_0001": 10},
                total_valid_voxel_count=10,
                mixed_precision_used=False,
            )

        with patch(
            "src.learning.model_training.train_learning_model_for_one_epoch",
            side_effect=_train_side_effect,
        ) as train_mock, patch(
            "src.learning.model_training.evaluate_learning_model_on_validation_dataloaders",
            side_effect=_eval_side_effect,
        ) as eval_mock:
            result = train_learning_model_with_validation_loop(
                preconditions=preconditions,
                mixed_precision=False,
                early_stop_patience=2,
                device="cpu",
            )

        self.assertEqual(train_mock.call_count, 4)
        self.assertEqual(eval_mock.call_count, 4)
        self.assertEqual(result.completed_epoch_count, 4)
        self.assertEqual(result.total_epoch_count, 6)
        self.assertEqual(result.stop_reason, "early_stop")
        self.assertEqual(result.best_epoch_index, 1)
        self.assertAlmostEqual(result.best_weighted_mean_accuracy, 0.60, places=10)
        self.assertAlmostEqual(float(model.weight.detach().item()), 2.0, places=10)

    def test_training_loop_runs_to_max_epoch_and_reports_best_epoch(self) -> None:
        preconditions = self._make_preconditions(train_box_count=2)
        model = preconditions.model_runtime.model

        def _train_side_effect(**kwargs):
            epoch_index = int(kwargs["epoch_index"])
            with torch.no_grad():
                model.weight.fill_(float(epoch_index + 1))
            return (
                LearningTrainEpochResult(
                    epoch_index=epoch_index,
                    total_epoch_count=int(kwargs["total_epoch_count"]),
                    base_learning_rate=0.001,
                    num_batches=1,
                    mean_loss=1.0,
                    mixed_precision_used=False,
                ),
                object(),
            )

        eval_values = iter((0.10, 0.20, 0.30, 0.40))

        def _eval_side_effect(**_kwargs):
            value = float(next(eval_values))
            return LearningValidationEvalResult(
                weighted_mean_accuracy=value,
                per_box_accuracy_by_box_id={"bbox_eval_0001": value},
                valid_voxel_counts_by_box_id={"bbox_eval_0001": 10},
                total_valid_voxel_count=10,
                mixed_precision_used=False,
            )

        with patch(
            "src.learning.model_training.train_learning_model_for_one_epoch",
            side_effect=_train_side_effect,
        ), patch(
            "src.learning.model_training.evaluate_learning_model_on_validation_dataloaders",
            side_effect=_eval_side_effect,
        ):
            result = train_learning_model_with_validation_loop(
                preconditions=preconditions,
                mixed_precision=False,
                early_stop_patience=2,
                device="cpu",
            )

        self.assertEqual(result.completed_epoch_count, 4)
        self.assertEqual(result.total_epoch_count, 4)
        self.assertEqual(result.stop_reason, "max_epoch")
        self.assertEqual(result.best_epoch_index, 3)
        self.assertAlmostEqual(result.best_weighted_mean_accuracy, 0.40, places=10)
        self.assertAlmostEqual(float(model.weight.detach().item()), 4.0, places=10)

    def test_training_loop_restores_best_state_when_eval_metric_is_not_finite(self) -> None:
        preconditions = self._make_preconditions(train_box_count=3)
        model = preconditions.model_runtime.model

        def _train_side_effect(**kwargs):
            epoch_index = int(kwargs["epoch_index"])
            with torch.no_grad():
                model.weight.fill_(float(epoch_index + 1))
            return (
                LearningTrainEpochResult(
                    epoch_index=epoch_index,
                    total_epoch_count=int(kwargs["total_epoch_count"]),
                    base_learning_rate=0.001,
                    num_batches=1,
                    mean_loss=1.0,
                    mixed_precision_used=False,
                ),
                object(),
            )

        eval_values = iter((0.50, float("nan")))

        def _eval_side_effect(**_kwargs):
            value = float(next(eval_values))
            return LearningValidationEvalResult(
                weighted_mean_accuracy=value,
                per_box_accuracy_by_box_id={"bbox_eval_0001": value},
                valid_voxel_counts_by_box_id={"bbox_eval_0001": 10},
                total_valid_voxel_count=10,
                mixed_precision_used=False,
            )

        with patch(
            "src.learning.model_training.train_learning_model_for_one_epoch",
            side_effect=_train_side_effect,
        ), patch(
            "src.learning.model_training.evaluate_learning_model_on_validation_dataloaders",
            side_effect=_eval_side_effect,
        ):
            with self.assertRaisesRegex(ValueError, "must be finite"):
                train_learning_model_with_validation_loop(
                    preconditions=preconditions,
                    mixed_precision=False,
                    early_stop_patience=2,
                    device="cpu",
                )

        self.assertAlmostEqual(float(model.weight.detach().item()), 1.0, places=10)

    def test_training_loop_user_stop_before_best_keeps_current_state_and_returns_na_best(self) -> None:
        preconditions = self._make_preconditions(train_box_count=3)
        stop_event = Event()
        stop_event.set()

        with patch(
            "src.learning.model_training.train_learning_model_for_one_epoch"
        ) as train_mock, patch(
            "src.learning.model_training.evaluate_learning_model_on_validation_dataloaders"
        ) as eval_mock:
            result = train_learning_model_with_validation_loop(
                preconditions=preconditions,
                mixed_precision=False,
                early_stop_patience=2,
                device="cpu",
                stop_event=stop_event,
            )

        train_mock.assert_not_called()
        eval_mock.assert_not_called()
        self.assertEqual(result.completed_epoch_count, 0)
        self.assertEqual(result.stop_reason, "user_stop")
        self.assertIsNone(result.best_epoch_index)
        self.assertIsNone(result.best_weighted_mean_accuracy)


if __name__ == "__main__":
    unittest.main()
