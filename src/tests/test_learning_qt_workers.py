from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject

from src.bbox import BoundingBox
from src.learning import LearningTrainingLoopResult
from src.learning.qt_workers import (
    LearningInferenceWorker,
    LearningTrainingWorker,
)
from src.utils import exception_message


class LearningQtWorkerImportTests(unittest.TestCase):
    def test_import_does_not_import_torch(self) -> None:
        script = textwrap.dedent("""
            import os
            import sys

            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            attempted_torch_imports = []

            class _TorchImportBlocker:
                def find_spec(self, fullname, path=None, target=None):
                    del path, target
                    if fullname == "torch" or fullname.startswith("torch."):
                        attempted_torch_imports.append(fullname)
                        raise RuntimeError(f"unexpected import: {fullname}")
                    return None

            sys.meta_path.insert(0, _TorchImportBlocker())

            import src.learning.qt_workers  # noqa: F401

            assert not attempted_torch_imports, attempted_torch_imports
            assert "torch" not in sys.modules
            """)
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["PYTHONPATH"] = os.getcwd()
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

    def test_workers_remain_qobjects_without_multiprocessing_dependency(self) -> None:
        self.assertIsInstance(LearningTrainingWorker(), QObject)
        self.assertIsInstance(LearningInferenceWorker(), QObject)

        module_path = Path(__file__).resolve().parents[1] / "learning" / "qt_workers.py"
        source = module_path.read_text(encoding="utf-8")
        self.assertNotIn("multiprocessing", source)

    def test_main_window_no_longer_exports_private_worker_aliases(
        self,
    ) -> None:
        from src.ui import main_window

        self.assertFalse(hasattr(main_window, "_LearningTrainingWorker"))
        self.assertFalse(hasattr(main_window, "_LearningInferenceWorker"))
        self.assertFalse(hasattr(main_window, "_LearningInferencePrediction"))
        self.assertFalse(hasattr(main_window, "_LearningInferenceBackgroundResult"))
        self.assertFalse(hasattr(main_window, "_LearningInferenceStopRequested"))


class LearningInferenceWorkerConfigureTests(unittest.TestCase):
    def test_configure_keeps_ndarray_raw_array_reference(self) -> None:
        raw_array = np.zeros((2, 3, 4), dtype=np.uint8)
        worker = LearningInferenceWorker()

        worker.configure(
            model_runtime=object(),
            inference_boxes=(
                BoundingBox(
                    id="box-1",
                    z0=0,
                    z1=1,
                    y0=0,
                    y1=2,
                    x0=0,
                    x1=3,
                ),
            ),
            raw_array=raw_array,
            label_values=(0, 1),
            volume_shape=raw_array.shape,
        )

        self.assertIs(worker._configured_raw_array(), raw_array)
        self.assertFalse(worker._use_tiled_score_buffer)
        self.assertEqual(worker._batch_size, 4)

    def test_configure_accepts_tiled_score_buffer_flag(self) -> None:
        raw_array = np.zeros((2, 3, 4), dtype=np.uint8)
        worker = LearningInferenceWorker()

        worker.configure(
            model_runtime=object(),
            inference_boxes=(
                BoundingBox(
                    id="box-1",
                    z0=0,
                    z1=1,
                    y0=0,
                    y1=2,
                    x0=0,
                    x1=3,
                ),
            ),
            raw_array=raw_array,
            label_values=(0, 1),
            volume_shape=raw_array.shape,
            use_tiled_score_buffer=True,
        )

        self.assertTrue(worker._use_tiled_score_buffer)

    def test_configure_accepts_batch_size(self) -> None:
        raw_array = np.zeros((2, 3, 4), dtype=np.uint8)
        worker = LearningInferenceWorker()

        worker.configure(
            model_runtime=object(),
            inference_boxes=(
                BoundingBox(
                    id="box-1",
                    z0=0,
                    z1=1,
                    y0=0,
                    y1=2,
                    x0=0,
                    x1=3,
                ),
            ),
            raw_array=raw_array,
            label_values=(0, 1),
            volume_shape=raw_array.shape,
            batch_size=9,
        )

        self.assertEqual(worker._batch_size, 9)

    def test_configure_rejects_invalid_batch_size(self) -> None:
        worker = LearningInferenceWorker()
        kwargs = dict(
            model_runtime=object(),
            inference_boxes=(
                BoundingBox(
                    id="box-1",
                    z0=0,
                    z1=1,
                    y0=0,
                    y1=2,
                    x0=0,
                    x1=3,
                ),
            ),
            raw_array=np.zeros((2, 3, 4), dtype=np.uint8),
            label_values=(0, 1),
            volume_shape=(2, 3, 4),
        )

        with self.assertRaisesRegex(ValueError, ">= 1"):
            worker.configure(**kwargs, batch_size=0)
        with self.assertRaisesRegex(TypeError, "integer"):
            worker.configure(**kwargs, batch_size=True)
        with self.assertRaisesRegex(TypeError, "integer"):
            worker.configure(**kwargs, batch_size=1.5)

    def test_configure_rejects_non_bool_tiled_score_buffer_flag(self) -> None:
        worker = LearningInferenceWorker()

        with self.assertRaisesRegex(TypeError, "use_tiled_score_buffer"):
            worker.configure(
                model_runtime=object(),
                inference_boxes=(
                    BoundingBox(
                        id="box-1",
                        z0=0,
                        z1=1,
                        y0=0,
                        y1=2,
                        x0=0,
                        x1=3,
                    ),
                ),
                raw_array=np.zeros((2, 3, 4), dtype=np.uint8),
                label_values=(0, 1),
                volume_shape=(2, 3, 4),
                use_tiled_score_buffer=1,
            )

    def test_configure_rejects_empty_inference_boxes(self) -> None:
        worker = LearningInferenceWorker()

        with self.assertRaisesRegex(ValueError, "at least one bounding box"):
            worker.configure(
                model_runtime=object(),
                inference_boxes=tuple(),
                raw_array=np.zeros((2, 3, 4), dtype=np.uint8),
                label_values=(0, 1),
                volume_shape=(2, 3, 4),
            )

    def test_configure_rejects_empty_label_values(self) -> None:
        worker = LearningInferenceWorker()

        with self.assertRaisesRegex(ValueError, "at least one class label"):
            worker.configure(
                model_runtime=object(),
                inference_boxes=(
                    BoundingBox(
                        id="box-1",
                        z0=0,
                        z1=1,
                        y0=0,
                        y1=2,
                        x0=0,
                        x1=3,
                    ),
                ),
                raw_array=np.zeros((2, 3, 4), dtype=np.uint8),
                label_values=tuple(),
                volume_shape=(2, 3, 4),
            )

    def test_configure_rejects_invalid_volume_shape(self) -> None:
        worker = LearningInferenceWorker()

        with self.assertRaisesRegex(ValueError, "volume_shape must be length 3"):
            worker.configure(
                model_runtime=object(),
                inference_boxes=(
                    BoundingBox(
                        id="box-1",
                        z0=0,
                        z1=1,
                        y0=0,
                        y1=2,
                        x0=0,
                        x1=3,
                    ),
                ),
                raw_array=np.zeros((2, 3, 4), dtype=np.uint8),
                label_values=(0, 1),
                volume_shape=(2, 3),
            )


class LearningTrainingWorkerTests(unittest.TestCase):
    def test_run_uses_configured_early_stop_patience(self) -> None:
        worker = LearningTrainingWorker()
        preconditions = object()
        worker.configure(preconditions=preconditions, early_stop_patience=5)
        result = LearningTrainingLoopResult(
            completed_epoch_count=4,
            total_epoch_count=8,
            stop_reason="early_stop",
            best_epoch=4,
            best_weighted_mean_dice=0.87,
            early_stop_patience=5,
            mixed_precision_enabled=True,
        )

        with patch(
            "src.learning.qt_workers._train_learning_model_with_validation_loop",
            return_value=result,
        ) as train_mock:
            worker.run()

        train_mock.assert_called_once_with(
            preconditions=preconditions,
            mixed_precision=True,
            early_stop_patience=5,
            stop_event=worker._stop_event,
        )

    def test_run_does_not_request_periodic_headless_checkpoints(self) -> None:
        worker = LearningTrainingWorker()
        preconditions = object()
        worker.configure(preconditions=preconditions)
        result = LearningTrainingLoopResult(
            completed_epoch_count=4,
            total_epoch_count=8,
            stop_reason="early_stop",
            best_epoch=4,
            best_weighted_mean_dice=0.87,
            early_stop_patience=2,
            mixed_precision_enabled=True,
        )

        with patch(
            "src.learning.qt_workers._train_learning_model_with_validation_loop",
            return_value=result,
        ) as train_mock:
            worker.run()

        train_mock.assert_called_once()
        self.assertNotIn("epoch_completion_callback", train_mock.call_args.kwargs)

    def test_configure_rejects_invalid_early_stop_patience(self) -> None:
        worker = LearningTrainingWorker()

        with self.assertRaisesRegex(ValueError, ">= 1"):
            worker.configure(preconditions=object(), early_stop_patience=0)
        with self.assertRaisesRegex(TypeError, "integer"):
            worker.configure(preconditions=object(), early_stop_patience=True)
        with self.assertRaisesRegex(TypeError, "integer"):
            worker.configure(preconditions=object(), early_stop_patience=1.5)


class LearningQtWorkerExceptionMessageTests(unittest.TestCase):
    def test_exception_message_uses_message_or_exception_type(self) -> None:
        self.assertEqual(exception_message(ValueError(" bad value ")), "bad value")
        self.assertEqual(exception_message(RuntimeError()), "RuntimeError")


if __name__ == "__main__":
    unittest.main()
