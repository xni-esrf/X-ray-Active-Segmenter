from __future__ import annotations

import unittest

import numpy as np

from src.bbox import BoundingBox
from src.learning.inference import (
    LearningInferencePrediction,
    apply_inference_predictions_to_array,
)


class LearningInferenceApplicationTests(unittest.TestCase):
    def test_apply_inference_predictions_writes_bbox_and_reports_changed_voxels(self) -> None:
        segmentation = np.zeros((3, 3, 3), dtype=np.uint8)
        box = BoundingBox.from_bounds(
            box_id="infer-box",
            z0=1,
            z1=3,
            y0=0,
            y1=2,
            x0=1,
            x1=3,
            label="inference",
            volume_shape=segmentation.shape,
        )
        prediction = LearningInferencePrediction(
            box=box,
            predicted_bbox=np.ones((2, 2, 2), dtype=np.uint8),
        )

        output, changed_count, succeeded_ids, failures = apply_inference_predictions_to_array(
            segmentation,
            (prediction,),
        )

        expected = np.zeros((3, 3, 3), dtype=np.uint8)
        expected[1:3, 0:2, 1:3] = 1
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(segmentation, np.zeros((3, 3, 3), dtype=np.uint8))
        self.assertEqual(changed_count, 8)
        self.assertEqual(succeeded_ids, ("infer-box",))
        self.assertEqual(failures, {})

    def test_apply_inference_predictions_reports_shape_mismatch_per_box(self) -> None:
        segmentation = np.zeros((3, 3, 3), dtype=np.uint8)
        box = BoundingBox.from_bounds(
            box_id="infer-box",
            z0=0,
            z1=2,
            y0=0,
            y1=2,
            x0=0,
            x1=2,
            label="inference",
            volume_shape=segmentation.shape,
        )
        prediction = LearningInferencePrediction(
            box=box,
            predicted_bbox=np.ones((1, 2, 2), dtype=np.uint8),
        )

        output, changed_count, succeeded_ids, failures = apply_inference_predictions_to_array(
            segmentation,
            (prediction,),
        )

        np.testing.assert_array_equal(output, segmentation)
        self.assertEqual(changed_count, 0)
        self.assertEqual(succeeded_ids, ())
        self.assertIn("infer-box", failures)
        self.assertIn("Predicted bbox shape does not match bbox size", failures["infer-box"])


if __name__ == "__main__":
    unittest.main()
