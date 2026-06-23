from __future__ import annotations

import unittest

import numpy as np

from src.learning import LearningLabelSpace, derive_label_space_from_semantic_segmentation


class LearningLabelSpaceTests(unittest.TestCase):
    def test_normalizes_labels_in_numeric_order_and_includes_background(self) -> None:
        label_space = LearningLabelSpace(label_values=(5, np.int64(2), 2, 1))

        self.assertEqual(label_space.label_values, (0, 1, 2, 5))
        self.assertEqual(label_space.num_classes, 4)
        self.assertEqual(label_space.background_label, 0)
        self.assertEqual(label_space.mask_label, -100)

    def test_preserves_source_signature_tuple(self) -> None:
        label_space = LearningLabelSpace(
            label_values=(0, 3),
            source_signature=("semantic", "/tmp/seg.tif", 7),
        )

        self.assertEqual(label_space.source_signature, ("semantic", "/tmp/seg.tif", 7))

    def test_rejects_mask_label_in_label_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "mask label -100"):
            LearningLabelSpace(label_values=(0, 1, -100))

    def test_rejects_without_foreground_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "foreground"):
            LearningLabelSpace(label_values=(0,))

    def test_rejects_background_equal_to_mask(self) -> None:
        with self.assertRaisesRegex(ValueError, "must differ"):
            LearningLabelSpace(
                label_values=(0, 1),
                background_label=-100,
                mask_label=-100,
            )

    def test_rejects_non_integer_label(self) -> None:
        with self.assertRaisesRegex(TypeError, "label value"):
            LearningLabelSpace(label_values=(0, 1.5))

    def test_rejects_non_tuple_source_signature(self) -> None:
        with self.assertRaisesRegex(TypeError, "source_signature"):
            LearningLabelSpace(
                label_values=(0, 1),
                source_signature=["semantic"],  # type: ignore[arg-type]
            )

    def test_derives_from_semantic_segmentation_excluding_mask_and_adding_background(
        self,
    ) -> None:
        segmentation = np.array(
            [
                [[5, 2], [-100, 5]],
                [[2, 7], [7, 5]],
            ],
            dtype=np.int16,
        )

        label_space = derive_label_space_from_semantic_segmentation(
            segmentation,
            source_signature=("semantic", "/tmp/seg.tif", 3),
        )

        self.assertEqual(label_space.label_values, (0, 2, 5, 7))
        self.assertEqual(label_space.num_classes, 4)
        self.assertEqual(label_space.source_signature, ("semantic", "/tmp/seg.tif", 3))

    def test_derivation_rejects_non_integer_segmentation(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            derive_label_space_from_semantic_segmentation(
                np.array([0.0, 1.0], dtype=np.float32)
            )

    def test_derivation_rejects_segmentation_without_foreground(self) -> None:
        with self.assertRaisesRegex(ValueError, "foreground"):
            derive_label_space_from_semantic_segmentation(
                np.array([0, -100], dtype=np.int16)
            )


if __name__ == "__main__":
    unittest.main()
