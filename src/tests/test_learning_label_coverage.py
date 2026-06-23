from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from src.learning import (
    LearningBBoxDataLoaderRuntime,
    LearningBBoxEvalRuntime,
    LearningLabelSpace,
    compute_learning_label_coverage,
    format_learning_label_coverage_warning,
)


class LearningLabelCoverageTests(unittest.TestCase):
    @staticmethod
    def _train_runtime(*, tensors: tuple[object, ...]) -> LearningBBoxDataLoaderRuntime:
        return LearningBBoxDataLoaderRuntime(
            dataset=SimpleNamespace(annot_tensors=tuple(tensors)),
            sampler=object(),
            dataloader=object(),
            train_box_ids=("bbox_0001",),
        )

    @staticmethod
    def _eval_runtime(*, ground_truth: object) -> LearningBBoxEvalRuntime:
        return LearningBBoxEvalRuntime(
            box_id="bbox_0002",
            dataloader=object(),
            buffer=SimpleNamespace(ground_truth=ground_truth),
        )

    def test_compute_learning_label_coverage_reports_missing_split_labels(self) -> None:
        coverage = compute_learning_label_coverage(
            label_space=LearningLabelSpace(label_values=(0, 1, 2)),
            train_runtime=self._train_runtime(
                tensors=(
                    np.asarray([[[0, 1], [-100, 1]]], dtype=np.int16),
                ),
            ),
            eval_runtimes_by_box_id={
                "bbox_0002": self._eval_runtime(
                    ground_truth=np.asarray([[[0, 2], [-100, 2]]], dtype=np.int16),
                )
            },
        )

        self.assertEqual(coverage.label_values, (0, 1, 2))
        self.assertEqual(coverage.train_present_label_values, (0, 1))
        self.assertEqual(coverage.validation_present_label_values, (0, 2))
        self.assertEqual(coverage.missing_train_label_values, (2,))
        self.assertEqual(coverage.missing_validation_label_values, (1,))

        warning = format_learning_label_coverage_warning(coverage)
        self.assertIsNotNone(warning)
        self.assertIn("Missing from train boxes: 2", str(warning))
        self.assertIn("Missing from validation boxes: 1", str(warning))

    def test_compute_learning_label_coverage_rejects_labels_outside_label_space(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "outside the current label space"):
            compute_learning_label_coverage(
                label_space=LearningLabelSpace(label_values=(0, 1)),
                train_runtime=self._train_runtime(
                    tensors=(np.asarray([[[0, 7]]], dtype=np.int16),),
                ),
                eval_runtimes_by_box_id={
                    "bbox_0002": self._eval_runtime(
                        ground_truth=np.asarray([[[0, 1]]], dtype=np.int16),
                    )
                },
            )

    def test_format_learning_label_coverage_warning_returns_none_when_complete(
        self,
    ) -> None:
        coverage = compute_learning_label_coverage(
            label_space=LearningLabelSpace(label_values=(0, 1)),
            train_runtime=self._train_runtime(
                tensors=(np.asarray([[[0, 1]]], dtype=np.int16),),
            ),
            eval_runtimes_by_box_id={
                "bbox_0002": self._eval_runtime(
                    ground_truth=np.asarray([[[0, 1]]], dtype=np.int16),
                )
            },
        )

        self.assertIsNone(format_learning_label_coverage_warning(coverage))


if __name__ == "__main__":
    unittest.main()
