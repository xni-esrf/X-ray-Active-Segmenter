from __future__ import annotations

from types import SimpleNamespace
import unittest

try:
    import torch
except Exception:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]

from src.learning import (
    LearningBBoxEvalRuntime,
    clear_current_learning_dataloader_runtime,
    clear_current_learning_eval_runtimes_by_box_id,
    compute_and_store_current_learning_class_weights,
    compute_class_weights_from_segmentation_tensors,
    get_current_learning_dataloader_runtime,
    set_current_learning_dataloader_components,
    set_current_learning_eval_runtimes_by_box_id,
)


@unittest.skipUnless(torch is not None, "PyTorch is not available")
class LearningClassWeightsTests(unittest.TestCase):
    class _FakeTorchNoCuda:
        Tensor = torch.Tensor if torch is not None else object
        float32 = torch.float32 if torch is not None else None
        long = torch.long if torch is not None else None
        cuda = SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
        )

        @staticmethod
        def unique(*args, **kwargs):
            if torch is None:  # pragma: no cover - guarded by class skip
                raise RuntimeError("PyTorch is required for this test")
            return torch.unique(*args, **kwargs)

        @staticmethod
        def tensor(*args, **kwargs):
            if torch is None:  # pragma: no cover - guarded by class skip
                raise RuntimeError("PyTorch is required for this test")
            return torch.tensor(*args, **kwargs)

        @staticmethod
        def device(spec: str):
            if torch is None:  # pragma: no cover - guarded by class skip
                raise RuntimeError("PyTorch is required for this test")
            return torch.device(spec)

    def setUp(self) -> None:
        clear_current_learning_dataloader_runtime()
        clear_current_learning_eval_runtimes_by_box_id()

    def tearDown(self) -> None:
        clear_current_learning_dataloader_runtime()
        clear_current_learning_eval_runtimes_by_box_id()

    def test_compute_class_weights_uses_inverse_frequency_in_label_value_order(self) -> None:
        first = torch.tensor(
            [
                [[0, 0], [0, 0]],
                [[1, 1], [1, 1]],
            ],
            dtype=torch.int16,
        )
        second = torch.tensor(
            [
                [[0, 0], [0, 0]],
                [[2, 0], [0, 0]],
            ],
            dtype=torch.int16,
        )

        weights = compute_class_weights_from_segmentation_tensors(
            (first, second),
            label_values=(1, 0, 2),
            max_weight=100.0,
            device="cpu",
        )

        self.assertEqual(weights.dtype, torch.float32)
        self.assertEqual(weights.device.type, "cpu")
        self.assertEqual(weights.tolist(), [2.0, 1.0, 8.0])

    def test_compute_class_weights_caps_missing_train_class_at_100(self) -> None:
        first = torch.tensor(
            [
                [[0, 0], [0, 0]],
                [[1, 1], [1, 1]],
            ],
            dtype=torch.int16,
        )

        weights = compute_class_weights_from_segmentation_tensors(
            (first,),
            label_values=(0, 1, 2),
            max_weight=100.0,
            device="cpu",
        )

        self.assertEqual(weights.tolist(), [1.0, 1.0, 100.0])

    def test_compute_class_weights_rejects_cuda0_when_cuda_is_unavailable(self) -> None:
        first = torch.tensor(
            [[[0]]],
            dtype=torch.int16,
        )

        with self.assertRaisesRegex(RuntimeError, "cuda:0"):
            compute_class_weights_from_segmentation_tensors(
                (first,),
                label_values=(0,),
                max_weight=100.0,
                device="cuda:0",
                torch_module=self._FakeTorchNoCuda(),
            )

    def test_compute_and_store_current_learning_class_weights_uses_eval_label_values_order(self) -> None:
        dataset = SimpleNamespace(
            annot_tensors=(
                torch.tensor(
                    [
                        [[0, 0], [0, 0]],
                        [[1, 1], [1, 1]],
                    ],
                    dtype=torch.int16,
                ),
            )
        )
        set_current_learning_dataloader_components(
            dataset=dataset,
            sampler=object(),
            dataloader=object(),
            train_box_ids=("bbox_0007",),
        )
        runtimes = {
            "bbox_0008": LearningBBoxEvalRuntime(
                box_id="bbox_0008",
                dataloader=object(),
                buffer=SimpleNamespace(label_values=(1, 0)),
            ),
            "bbox_0009": LearningBBoxEvalRuntime(
                box_id="bbox_0009",
                dataloader=object(),
                buffer=SimpleNamespace(label_values=(1, 0)),
            ),
        }
        set_current_learning_eval_runtimes_by_box_id(runtimes)

        weights = compute_and_store_current_learning_class_weights(device="cpu")
        current = get_current_learning_dataloader_runtime()

        self.assertIsNotNone(current)
        self.assertIsNotNone(current.class_weights)
        self.assertEqual(weights.tolist(), [1.0, 1.0])
        self.assertEqual(current.class_weights.tolist(), [1.0, 1.0])

    def test_compute_and_store_current_learning_class_weights_rejects_mismatched_eval_label_values(
        self,
    ) -> None:
        dataset = SimpleNamespace(
            annot_tensors=(
                torch.tensor(
                    [
                        [[0, 0], [0, 0]],
                        [[1, 1], [1, 1]],
                    ],
                    dtype=torch.int16,
                ),
            )
        )
        set_current_learning_dataloader_components(
            dataset=dataset,
            sampler=object(),
            dataloader=object(),
            train_box_ids=("bbox_0007",),
        )
        set_current_learning_eval_runtimes_by_box_id(
            {
                "bbox_0008": LearningBBoxEvalRuntime(
                    box_id="bbox_0008",
                    dataloader=object(),
                    buffer=SimpleNamespace(label_values=(0, 1)),
                ),
                "bbox_0009": LearningBBoxEvalRuntime(
                    box_id="bbox_0009",
                    dataloader=object(),
                    buffer=SimpleNamespace(label_values=(1, 0)),
                ),
            }
        )

        with self.assertRaisesRegex(ValueError, "label_values ordering"):
            compute_and_store_current_learning_class_weights(device="cpu")


if __name__ == "__main__":
    unittest.main()
