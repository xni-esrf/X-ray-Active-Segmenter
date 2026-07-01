from __future__ import annotations

import unittest

from src.learning import (
    DEFAULT_TRAINING_PARAMETERS,
    TrainingParameters,
    validate_training_parameters,
)


class TrainingParametersTests(unittest.TestCase):
    def test_default_training_parameters_match_current_hardcoded_values(self) -> None:
        self.assertEqual(
            DEFAULT_TRAINING_PARAMETERS,
            TrainingParameters(
                learning_rate=5e-5,
                training_batch_size=4,
                validation_batch_size=4,
                inference_batch_size=4,
                patches_per_epoch=1000,
                early_stopping_patience=2,
            ),
        )

    def test_validate_training_parameters_returns_normalized_copy(self) -> None:
        parameters = validate_training_parameters(
            TrainingParameters(
                learning_rate=0.001,
                training_batch_size=2,
                validation_batch_size=3,
                inference_batch_size=6,
                patches_per_epoch=12,
                early_stopping_patience=4,
            )
        )

        self.assertEqual(parameters.learning_rate, 0.001)
        self.assertEqual(parameters.training_batch_size, 2)
        self.assertEqual(parameters.validation_batch_size, 3)
        self.assertEqual(parameters.inference_batch_size, 6)
        self.assertEqual(parameters.patches_per_epoch, 12)
        self.assertEqual(parameters.early_stopping_patience, 4)

    def test_validate_training_parameters_rejects_invalid_values(self) -> None:
        invalid_cases = (
            TrainingParameters(learning_rate=0.0),
            TrainingParameters(learning_rate=float("inf")),
            TrainingParameters(training_batch_size=0),
            TrainingParameters(validation_batch_size=0),
            TrainingParameters(inference_batch_size=0),
            TrainingParameters(patches_per_epoch=0),
            TrainingParameters(early_stopping_patience=0),
        )

        for parameters in invalid_cases:
            with self.subTest(parameters=parameters):
                with self.assertRaises((TypeError, ValueError)):
                    validate_training_parameters(parameters)


if __name__ == "__main__":
    unittest.main()
