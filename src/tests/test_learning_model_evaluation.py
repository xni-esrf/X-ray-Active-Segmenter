from __future__ import annotations

import unittest

try:
    import torch
except Exception:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]

from src.learning import (
    LearningBBoxEvalRuntime,
    LearningModelRuntime,
    evaluate_learning_model_on_validation_dataloaders,
)


@unittest.skipUnless(torch is not None, "PyTorch is not available")
class LearningModelEvaluationTests(unittest.TestCase):
    if torch is not None:
        class _FakeModel(torch.nn.Module):
            def __init__(self, *, num_classes: int) -> None:
                super().__init__()
                self.num_classes = int(num_classes)

            def forward(self, x):
                batch_size = int(x.shape[0])
                d_size = int(x.shape[2])
                h_size = int(x.shape[3])
                w_size = int(x.shape[4])
                logits = torch.zeros(
                    (batch_size, self.num_classes, d_size, h_size, w_size),
                    dtype=torch.float32,
                    device=x.device,
                )
                logits[:, 1, :, :, :] = 1.0
                return logits
    else:  # pragma: no cover - guarded by class skip
        class _FakeModel:  # type: ignore[no-redef]
            def __init__(self, *, num_classes: int) -> None:
                del num_classes

    class _GuardedBuffer:
        def __init__(self, *, ground_truth, accuracy: float) -> None:
            self.ground_truth = ground_truth
            self._accuracy = float(accuracy)
            self.buffer_vol = torch.full((1,), 7.0, dtype=torch.float32)
            self.add_calls = 0
            self.cleared_before_first_add = False

        def add_batch(self, batch, coordinates) -> None:
            del batch, coordinates
            self.add_calls += 1
            if self.add_calls == 1:
                self.cleared_before_first_add = bool(torch.all(self.buffer_vol == 0))

        def get_acc_pred(self):
            return torch.tensor(self._accuracy, dtype=torch.float32)

    @staticmethod
    def _make_eval_runtime(*, box_id: str, buffer_obj) -> LearningBBoxEvalRuntime:
        samples = [
            (
                torch.randn((1, 4, 4, 4), dtype=torch.float32),
                (0, 0, 0),
            )
        ]
        loader = torch.utils.data.DataLoader(samples, batch_size=1, shuffle=False)
        return LearningBBoxEvalRuntime(
            box_id=box_id,
            dataloader=loader,
            buffer=buffer_obj,
        )

    @staticmethod
    def _make_model_runtime() -> LearningModelRuntime:
        model = LearningModelEvaluationTests._FakeModel(num_classes=2)
        optimizer = object()
        return LearningModelRuntime(
            model=model,
            optimizer=optimizer,
            checkpoint_path="foundation_model/weights_epoch_190.cp",
            device_ids=(0,),
            num_classes=2,
        )

    def test_evaluate_validation_computes_weighted_mean_and_clears_buffers(self) -> None:
        model_runtime = self._make_model_runtime()
        first_buffer = self._GuardedBuffer(
            ground_truth=torch.tensor([[[0, -100], [1, -100]]], dtype=torch.long),
            accuracy=0.5,
        )
        second_buffer = self._GuardedBuffer(
            ground_truth=torch.tensor(
                [
                    [[1, 1], [1, 1]],
                    [[1, 1], [-100, -100]],
                ],
                dtype=torch.long,
            ),
            accuracy=1.0,
        )
        eval_runtimes = {
            "bbox_0008": self._make_eval_runtime(box_id="bbox_0008", buffer_obj=first_buffer),
            "bbox_0011": self._make_eval_runtime(box_id="bbox_0011", buffer_obj=second_buffer),
        }

        self.assertTrue(model_runtime.model.training)
        result = evaluate_learning_model_on_validation_dataloaders(
            model_runtime=model_runtime,
            eval_runtimes_by_box_id=eval_runtimes,
            device="cpu",
            mixed_precision=True,
        )

        self.assertAlmostEqual(result.weighted_mean_accuracy, 0.875, places=8)
        self.assertEqual(result.valid_voxel_counts_by_box_id["bbox_0008"], 2)
        self.assertEqual(result.valid_voxel_counts_by_box_id["bbox_0011"], 6)
        self.assertEqual(result.total_valid_voxel_count, 8)
        self.assertEqual(result.per_box_accuracy_by_box_id["bbox_0008"], 0.5)
        self.assertEqual(result.per_box_accuracy_by_box_id["bbox_0011"], 1.0)
        self.assertFalse(result.mixed_precision_used)
        self.assertTrue(first_buffer.cleared_before_first_add)
        self.assertTrue(second_buffer.cleared_before_first_add)
        self.assertTrue(model_runtime.model.training)

    def test_evaluate_validation_rejects_zero_valid_voxels(self) -> None:
        model_runtime = self._make_model_runtime()
        empty_buffer = self._GuardedBuffer(
            ground_truth=torch.full((1, 2, 2), -100, dtype=torch.long),
            accuracy=1.0,
        )
        eval_runtimes = {
            "bbox_0008": self._make_eval_runtime(box_id="bbox_0008", buffer_obj=empty_buffer),
        }

        with self.assertRaisesRegex(ValueError, "no valid voxels"):
            evaluate_learning_model_on_validation_dataloaders(
                model_runtime=model_runtime,
                eval_runtimes_by_box_id=eval_runtimes,
                device="cpu",
                mixed_precision=False,
            )


if __name__ == "__main__":
    unittest.main()
