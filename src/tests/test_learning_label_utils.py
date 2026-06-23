from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]

from src.learning import encode_target_labels


class LearningLabelUtilsTests(unittest.TestCase):
    def test_encode_target_labels_keeps_contiguous_numpy_labels(self) -> None:
        target = np.array(
            [
                [[0, 1], [2, 0]],
            ],
            dtype=np.int16,
        )

        encoded = encode_target_labels(target, label_values=(0, 1, 2))

        self.assertEqual(encoded.dtype, np.int64)
        np.testing.assert_array_equal(
            encoded,
            np.array(
                [
                    [[0, 1], [2, 0]],
                ],
                dtype=np.int64,
            ),
        )

    def test_encode_target_labels_compacts_non_contiguous_numpy_labels(self) -> None:
        target = np.array(
            [
                [[0, 1], [2, 4]],
            ],
            dtype=np.int16,
        )

        encoded = encode_target_labels(target, label_values=(0, 1, 2, 4))

        np.testing.assert_array_equal(
            encoded,
            np.array(
                [
                    [[0, 1], [2, 3]],
                ],
                dtype=np.int64,
            ),
        )

    def test_encode_target_labels_preserves_numpy_mask_label(self) -> None:
        target = np.array(
            [
                [[0, -100], [4, 2]],
            ],
            dtype=np.int16,
        )

        encoded = encode_target_labels(target, label_values=(0, 2, 4))

        np.testing.assert_array_equal(
            encoded,
            np.array(
                [
                    [[0, -100], [2, 1]],
                ],
                dtype=np.int64,
            ),
        )

    def test_encode_target_labels_rejects_unexpected_numpy_label(self) -> None:
        target = np.array([[[0, 7]]], dtype=np.int16)

        with self.assertRaisesRegex(ValueError, "not present in label_values: \\(7,\\)"):
            encode_target_labels(target, label_values=(0, 1, 2, 4))

    def test_encode_target_labels_rejects_non_integer_numpy_labels(self) -> None:
        target = np.array([[[0.0, 1.0]]], dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "integer dtype"):
            encode_target_labels(target, label_values=(0, 1))

    @unittest.skipUnless(torch is not None, "PyTorch is not available")
    def test_encode_target_labels_compacts_torch_tensor_on_same_device(self) -> None:
        target = torch.tensor(
            [
                [[0, 1], [2, 4]],
            ],
            dtype=torch.int16,
        )

        encoded = encode_target_labels(target, label_values=(0, 1, 2, 4))

        self.assertIsInstance(encoded, torch.Tensor)
        self.assertEqual(encoded.dtype, torch.long)
        self.assertEqual(encoded.device, target.device)
        self.assertEqual(encoded.tolist(), [[[0, 1], [2, 3]]])

    @unittest.skipUnless(torch is not None, "PyTorch is not available")
    def test_encode_target_labels_preserves_torch_mask_label(self) -> None:
        target = torch.tensor(
            [
                [[0, -100], [4, 2]],
            ],
            dtype=torch.int16,
        )

        encoded = encode_target_labels(target, label_values=(0, 2, 4))

        self.assertEqual(encoded.tolist(), [[[0, -100], [2, 1]]])

    @unittest.skipUnless(torch is not None, "PyTorch is not available")
    def test_encoded_non_contiguous_labels_satisfy_cross_entropy_target_contract(
        self,
    ) -> None:
        target = torch.tensor(
            [
                [[[0, 1], [2, 4]]],
            ],
            dtype=torch.long,
        )
        encoded = encode_target_labels(target, label_values=(0, 1, 2, 4))
        logits = torch.randn(
            (1, 4, 1, 2, 2),
            dtype=torch.float32,
            requires_grad=True,
        )

        loss = torch.nn.CrossEntropyLoss()(logits, encoded)
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertIsNotNone(logits.grad)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "CUDA is not available",
    )
    def test_encode_target_labels_compacts_cuda_tensor_on_same_device(self) -> None:
        target = torch.tensor([[[0, 4]]], dtype=torch.int16, device="cuda:0")

        encoded = encode_target_labels(target, label_values=(0, 4))

        self.assertEqual(encoded.device.type, "cuda")
        self.assertEqual(encoded.cpu().tolist(), [[[0, 1]]])


if __name__ == "__main__":
    unittest.main()
