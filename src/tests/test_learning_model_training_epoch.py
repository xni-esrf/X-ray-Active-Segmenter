from __future__ import annotations

import unittest

try:
    import torch
except Exception:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]

from src.learning import (
    LearningBBoxDataLoaderRuntime,
    LearningModelRuntime,
    train_learning_model_for_one_epoch,
)


@unittest.skipUnless(torch is not None, "PyTorch is not available")
class LearningModelTrainingEpochTests(unittest.TestCase):
    def _make_train_runtime(
        self,
        *,
        class_weights,
        batch_count: int = 2,
    ) -> LearningBBoxDataLoaderRuntime:
        samples = []
        for _ in range(int(batch_count)):
            minivol = torch.randn((1, 6, 6, 6), dtype=torch.float32)
            annot = torch.randint(low=0, high=3, size=(6, 6, 6), dtype=torch.long)
            samples.append((minivol, annot))
        dataloader = torch.utils.data.DataLoader(samples, batch_size=1, shuffle=False)
        return LearningBBoxDataLoaderRuntime(
            dataset=samples,
            sampler=object(),
            dataloader=dataloader,
            train_box_ids=("bbox_0001",),
            class_weights=class_weights,
        )

    def _make_model_runtime(
        self,
        *,
        initial_lr: float,
        lwise_lr_decay: float,
    ) -> LearningModelRuntime:
        model = torch.nn.Conv3d(1, 3, kernel_size=1)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [model.weight],
                    "lr": float(initial_lr),
                    "weight_decay": 0.0,
                    "lwise_lr_decay_rate": 1.0,
                },
                {
                    "params": [model.bias],
                    "lr": float(initial_lr),
                    "weight_decay": 0.0,
                    "lwise_lr_decay_rate": 0.5,
                },
            ]
        )
        return LearningModelRuntime(
            model=model,
            optimizer=optimizer,
            checkpoint_path="foundation_model/weights_epoch_190.cp",
            device_ids=(0,),
            num_classes=3,
            hyperparameters={
                "lr": float(initial_lr),
                "lwise_lr_decay": float(lwise_lr_decay),
            },
        )

    def test_train_one_epoch_updates_optimizer_lrs_with_layerwise_rates(self) -> None:
        torch.manual_seed(7)
        runtime = self._make_model_runtime(initial_lr=0.01, lwise_lr_decay=0.8)
        train_runtime = self._make_train_runtime(
            class_weights=torch.tensor([1.0, 1.5, 2.0], dtype=torch.float32),
            batch_count=2,
        )

        weight_before = runtime.model.weight.detach().clone()
        result, scaler = train_learning_model_for_one_epoch(
            epoch_index=0,
            total_epoch_count=4,
            model_runtime=runtime,
            train_runtime=train_runtime,
            device="cpu",
            mixed_precision=True,
        )

        self.assertIsNotNone(scaler)
        self.assertEqual(result.epoch_index, 0)
        self.assertEqual(result.total_epoch_count, 4)
        self.assertEqual(result.num_batches, 2)
        self.assertGreater(result.mean_loss, 0.0)
        self.assertFalse(result.mixed_precision_used)
        self.assertAlmostEqual(result.base_learning_rate, 0.01, places=10)
        self.assertAlmostEqual(runtime.optimizer.param_groups[0]["lr"], 0.01, places=10)
        self.assertAlmostEqual(runtime.optimizer.param_groups[1]["lr"], 0.005, places=10)
        self.assertFalse(torch.allclose(weight_before, runtime.model.weight.detach()))

    def test_train_one_epoch_uses_uniform_lr_when_lwise_lr_decay_is_one(self) -> None:
        torch.manual_seed(9)
        runtime = self._make_model_runtime(initial_lr=0.02, lwise_lr_decay=1.0)
        train_runtime = self._make_train_runtime(
            class_weights=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
            batch_count=1,
        )

        result, _ = train_learning_model_for_one_epoch(
            epoch_index=0,
            total_epoch_count=3,
            model_runtime=runtime,
            train_runtime=train_runtime,
            device="cpu",
            mixed_precision=False,
        )

        self.assertAlmostEqual(result.base_learning_rate, 0.02, places=10)
        self.assertAlmostEqual(runtime.optimizer.param_groups[0]["lr"], 0.02, places=10)
        self.assertAlmostEqual(runtime.optimizer.param_groups[1]["lr"], 0.02, places=10)

    def test_train_one_epoch_rejects_missing_class_weights(self) -> None:
        runtime = self._make_model_runtime(initial_lr=0.01, lwise_lr_decay=0.8)
        train_runtime = self._make_train_runtime(class_weights=None, batch_count=1)

        with self.assertRaisesRegex(ValueError, "class_weights"):
            train_learning_model_for_one_epoch(
                epoch_index=0,
                total_epoch_count=2,
                model_runtime=runtime,
                train_runtime=train_runtime,
                device="cpu",
                mixed_precision=False,
            )

    def test_train_one_epoch_rejects_empty_dataloader(self) -> None:
        runtime = self._make_model_runtime(initial_lr=0.01, lwise_lr_decay=0.8)
        empty_dataloader = torch.utils.data.DataLoader([], batch_size=1, shuffle=False)
        train_runtime = LearningBBoxDataLoaderRuntime(
            dataset=[],
            sampler=object(),
            dataloader=empty_dataloader,
            train_box_ids=("bbox_0001",),
            class_weights=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        )

        with self.assertRaisesRegex(ValueError, "zero batches"):
            train_learning_model_for_one_epoch(
                epoch_index=0,
                total_epoch_count=2,
                model_runtime=runtime,
                train_runtime=train_runtime,
                device="cpu",
                mixed_precision=False,
            )


if __name__ == "__main__":
    unittest.main()

