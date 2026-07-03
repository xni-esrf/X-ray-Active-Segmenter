from __future__ import annotations

import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.bbox import BoundingBox
from src.headless.job_spec import load_headless_job_spec
from src.learning import TrainingParameters
from src.ui.bottom_panel import BoundingBoxRow
from src.ui.dialogs import DialogResult, SaveDialogResult
from src.ui.main_window import MainWindow


class MainWindowHeadlessLaunchTests(unittest.TestCase):
    def test_train_headless_close_writes_job_saves_input_checkpoint_and_closes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "raw.npy"
            seg_path = root / "seg.npy"
            bbox_path = root / "boxes.bbox.txt"
            output_checkpoint_path = root / "trained.cp"
            job_dir = root / ".headless-job" / "train-job"
            self._touch_inputs(raw_path, seg_path, bbox_path)
            train_box = self._box("train-box", "train")
            validation_box = self._box("validation-box", "validation", z0=1, z1=2)
            window = self._make_window_like(
                raw_path=raw_path,
                seg_path=seg_path,
                bbox_path=bbox_path,
                boxes=(train_box, validation_box),
            )

            saved_checkpoints = []
            spawned_jobs = []
            window._save_model_runtime_checkpoint = (
                lambda _runtime, *, checkpoint_path: saved_checkpoints.append(checkpoint_path)
            )
            window._spawn_headless_after_ui_exit = lambda job_path: spawned_jobs.append(job_path)
            window._create_headless_job_dir = lambda _kind: job_dir

            with patch(
                "src.ui.main_window.open_save_model_checkpoint_dialog",
                return_value=DialogResult(
                    accepted=True,
                    path=str(output_checkpoint_path),
                ),
            ), patch.object(
                MainWindow,
                "_get_learning_model_runtime_for",
                return_value=SimpleNamespace(num_classes=2),
            ):
                result = window._launch_headless_learning_job_and_close(kind="train")

            self.assertTrue(result)
            self.assertTrue(window.closed)
            self.assertEqual(len(spawned_jobs), 1)
            spec = load_headless_job_spec(spawned_jobs[0])
            self.assertEqual(spec.kind, "train")
            self.assertEqual(spec.raw_volume_path, str(raw_path))
            self.assertEqual(spec.segmentation_path, str(seg_path))
            self.assertEqual(spec.bbox_path, str(bbox_path))
            self.assertEqual(spec.output_checkpoint_path, str(output_checkpoint_path))
            self.assertEqual(saved_checkpoints, [str(job_dir / "input_model.cp")])
            self.assertIsNone(window._raw_volume)
            self.assertIsNone(window._semantic_volume)
            self.assertIsNone(window._segmentation_editor)
            self.assertIn("clear_model_runtime", window.learning_session_calls)

    def test_inference_headless_close_writes_job_with_output_segmentation(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "raw.npy"
            seg_path = root / "seg.npy"
            bbox_path = root / "boxes.bbox.txt"
            output_seg_path = root / "out.npy"
            job_dir = root / ".headless-job" / "inference-job"
            self._touch_inputs(raw_path, seg_path, bbox_path)
            inference_box = self._box("infer-box", "inference")
            window = self._make_window_like(
                raw_path=raw_path,
                seg_path=seg_path,
                bbox_path=bbox_path,
                boxes=(inference_box,),
            )

            spawned_jobs = []
            window._save_model_runtime_checkpoint = lambda _runtime, *, checkpoint_path: None
            window._spawn_headless_after_ui_exit = lambda job_path: spawned_jobs.append(job_path)
            window._create_headless_job_dir = lambda _kind: job_dir

            with patch(
                "src.ui.main_window.open_save_segmentation_dialog",
                return_value=SaveDialogResult(
                    accepted=True,
                    path=str(output_seg_path),
                    format="npy",
                ),
            ), patch.object(
                MainWindow,
                "_get_learning_model_runtime_for",
                return_value=SimpleNamespace(num_classes=2),
            ):
                result = window._launch_headless_learning_job_and_close(kind="inference")

            self.assertTrue(result)
            self.assertTrue(window.closed)
            self.assertEqual(len(spawned_jobs), 1)
            spec = load_headless_job_spec(spawned_jobs[0])
            self.assertEqual(spec.kind, "inference")
            self.assertEqual(spec.raw_volume_path, str(raw_path))
            self.assertEqual(spec.segmentation_path, str(seg_path))
            self.assertEqual(spec.bbox_path, str(bbox_path))
            self.assertEqual(spec.output_segmentation_path, str(output_seg_path.with_suffix(".zarr")))
            self.assertEqual(spec.output_segmentation_format, "zarr")
            self.assertIsNone(window._raw_volume)
            self.assertIsNone(window._semantic_volume)
            self.assertIsNone(window._segmentation_editor)

    def test_inference_headless_close_allows_no_loaded_segmentation(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "raw.npy"
            seg_path = root / "seg.npy"
            bbox_path = root / "boxes.bbox.txt"
            output_seg_path = root / "out.npy"
            job_dir = root / ".headless-job" / "inference-job"
            self._touch_inputs(raw_path, seg_path, bbox_path)
            inference_box = self._box("infer-box", "inference")
            window = self._make_window_like(
                raw_path=raw_path,
                seg_path=seg_path,
                bbox_path=bbox_path,
                boxes=(inference_box,),
            )
            window._semantic_volume = None
            window._last_saved_segmentation_path = None
            window._last_saved_segmentation_kind = None
            window._active_segmentation_volume = lambda: None

            spawned_jobs = []
            window._save_model_runtime_checkpoint = lambda _runtime, *, checkpoint_path: None
            window._spawn_headless_after_ui_exit = lambda job_path: spawned_jobs.append(job_path)
            window._create_headless_job_dir = lambda _kind: job_dir

            with patch(
                "src.ui.main_window.open_save_segmentation_dialog",
                return_value=SaveDialogResult(
                    accepted=True,
                    path=str(output_seg_path),
                    format="npy",
                ),
            ), patch.object(
                MainWindow,
                "_get_learning_model_runtime_for",
                return_value=SimpleNamespace(num_classes=2),
            ):
                result = window._launch_headless_learning_job_and_close(kind="inference")

            self.assertTrue(result)
            self.assertEqual(len(spawned_jobs), 1)
            spec = load_headless_job_spec(spawned_jobs[0])
            self.assertEqual(spec.kind, "inference")
            self.assertIsNone(spec.segmentation_path)
            self.assertEqual(spec.segmentation_kind, "semantic")
            self.assertEqual(spec.output_segmentation_path, str(output_seg_path.with_suffix(".zarr")))
            self.assertEqual(spec.output_segmentation_format, "zarr")

    def test_train_headless_close_refuses_when_dirty_segmentation_save_is_canceled(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "raw.npy"
            seg_path = root / "seg.npy"
            bbox_path = root / "boxes.bbox.txt"
            job_dir = root / ".headless-job" / "train-job"
            self._touch_inputs(raw_path, seg_path, bbox_path)
            window = self._make_window_like(
                raw_path=raw_path,
                seg_path=seg_path,
                bbox_path=bbox_path,
                boxes=(
                    self._box("train-box", "train"),
                    self._box("validation-box", "validation", z0=1, z1=2),
                ),
            )
            window._segmentation_editor = SimpleNamespace(dirty=True)
            window._save_active_segmentation_with_dialog = lambda: False
            window._spawn_headless_after_ui_exit = Mock()
            window._create_headless_job_dir = lambda _kind: job_dir

            with patch("src.ui.main_window.show_info"), patch(
                "src.ui.main_window.show_warning"
            ), patch.object(
                MainWindow,
                "_get_learning_model_runtime_for",
                return_value=SimpleNamespace(num_classes=2),
            ):
                result = window._launch_headless_learning_job_and_close(kind="train")

            self.assertFalse(result)
            self.assertFalse(window.closed)
            window._spawn_headless_after_ui_exit.assert_not_called()
            self.assertFalse((job_dir / "job.json").exists())

    def test_inference_headless_close_refuses_when_unsaved_bboxes_save_is_canceled(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "raw.npy"
            seg_path = root / "seg.npy"
            bbox_path = root / "boxes.bbox.txt"
            output_seg_path = root / "out.npy"
            job_dir = root / ".headless-job" / "inference-job"
            self._touch_inputs(raw_path, seg_path, bbox_path)
            window = self._make_window_like(
                raw_path=raw_path,
                seg_path=seg_path,
                bbox_path=bbox_path,
                boxes=(self._box("infer-box", "inference"),),
            )
            window._last_saved_bounding_boxes_path = None
            window._save_bounding_boxes_with_dialog = lambda: False
            window._spawn_headless_after_ui_exit = Mock()
            window._create_headless_job_dir = lambda _kind: job_dir

            with patch("src.ui.main_window.show_info"), patch(
                "src.ui.main_window.show_warning"
            ), patch(
                "src.ui.main_window.open_save_segmentation_dialog",
                return_value=SaveDialogResult(
                    accepted=True,
                    path=str(output_seg_path),
                    format="npy",
                ),
            ), patch.object(
                MainWindow,
                "_get_learning_model_runtime_for",
                return_value=SimpleNamespace(num_classes=2),
            ):
                result = window._launch_headless_learning_job_and_close(kind="inference")

            self.assertFalse(result)
            self.assertFalse(window.closed)
            window._spawn_headless_after_ui_exit.assert_not_called()
            self.assertFalse((job_dir / "job.json").exists())

    def test_headless_buttons_do_not_call_normal_qt_training_or_inference_paths(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "raw.npy"
            seg_path = root / "seg.npy"
            bbox_path = root / "boxes.bbox.txt"
            train_output_path = root / "trained.cp"
            inference_output_path = root / "out.npy"
            train_job_dir = root / ".headless-job" / "train-job"
            inference_job_dir = root / ".headless-job" / "inference-job"
            self._touch_inputs(raw_path, seg_path, bbox_path)
            window = self._make_window_like(
                raw_path=raw_path,
                seg_path=seg_path,
                bbox_path=bbox_path,
                boxes=(
                    self._box("train-box", "train"),
                    self._box("validation-box", "validation", z0=1, z1=2),
                    self._box("infer-box", "inference", z0=0, z1=1),
                ),
            )
            normal_train = Mock(side_effect=AssertionError("normal train path called"))
            normal_inference = Mock(side_effect=AssertionError("normal inference path called"))
            window._train_model_on_dataset_with_dialog = normal_train
            window._segment_inference_bboxes_with_dialog = normal_inference
            spawned_jobs = []
            window._spawn_headless_after_ui_exit = lambda job_path: spawned_jobs.append(job_path)
            job_dirs = iter((train_job_dir, inference_job_dir))
            window._create_headless_job_dir = lambda _kind: next(job_dirs)
            window._save_model_runtime_checkpoint = lambda _runtime, *, checkpoint_path: None

            with patch(
                "src.ui.main_window.open_save_model_checkpoint_dialog",
                return_value=DialogResult(
                    accepted=True,
                    path=str(train_output_path),
                ),
            ), patch(
                "src.ui.main_window.open_save_segmentation_dialog",
                return_value=SaveDialogResult(
                    accepted=True,
                    path=str(inference_output_path),
                    format="npy",
                ),
            ), patch.object(
                MainWindow,
                "_get_learning_model_runtime_for",
                return_value=SimpleNamespace(num_classes=2),
            ):
                train_result = window._launch_headless_learning_job_and_close(kind="train")
                self.assertTrue(train_result)
                window = self._make_window_like(
                    raw_path=raw_path,
                    seg_path=seg_path,
                    bbox_path=bbox_path,
                    boxes=(self._box("infer-box", "inference"),),
                )
                window._train_model_on_dataset_with_dialog = normal_train
                window._segment_inference_bboxes_with_dialog = normal_inference
                window._spawn_headless_after_ui_exit = lambda job_path: spawned_jobs.append(job_path)
                window._create_headless_job_dir = lambda _kind: next(job_dirs)
                window._save_model_runtime_checkpoint = lambda _runtime, *, checkpoint_path: None
                inference_result = window._launch_headless_learning_job_and_close(kind="inference")

            self.assertTrue(inference_result)
            normal_train.assert_not_called()
            normal_inference.assert_not_called()
            self.assertEqual(len(spawned_jobs), 2)

    def test_spawn_headless_after_ui_exit_launches_lightweight_process(self) -> None:
        with TemporaryDirectory() as tmpdir:
            job_path = str(Path(tmpdir) / ".headless-job" / "job.json")
            popen_calls = []

            def fake_popen(command, **kwargs):
                popen_calls.append((command, kwargs))
                return SimpleNamespace(pid=1234)

            with patch("src.ui.main_window.subprocess.Popen", side_effect=fake_popen):
                result = MainWindow._spawn_headless_after_ui_exit(job_path)

            self.assertEqual(result.pid, 1234)
            self.assertEqual(len(popen_calls), 1)
            command, kwargs = popen_calls[0]
            self.assertEqual(
                command,
                [
                    sys.executable,
                    str(Path(__file__).resolve().parents[2] / "launch_headless_after_ui_exit.py"),
                    "--wait-pid",
                    str(os.getpid()),
                    "--job",
                    job_path,
                    "--python",
                    sys.executable,
                    "--runner",
                    str(Path(__file__).resolve().parents[2] / "run_headless_job.py"),
                    "--log-level",
                    "INFO",
                ],
            )
            self.assertEqual(
                kwargs,
                {
                    "close_fds": True,
                    "start_new_session": True,
                },
            )
            self.assertIn("launch_headless_after_ui_exit.py", command[1])
            self.assertTrue(any("run_headless_job.py" in part for part in command))
            self.assertIn("--wait-pid", command)
            self.assertIn(job_path, command)

    def test_headless_segmentation_path_rejects_virtual_generated_path_and_forces_save(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "raw.npy"
            seg_path = root / "seg.npy"
            bbox_path = root / "boxes.bbox.txt"
            saved_seg_path = root / "saved_seg.npy"
            self._touch_inputs(raw_path, seg_path, bbox_path)
            saved_seg_path.write_bytes(b"saved")
            window = self._make_window_like(
                raw_path=raw_path,
                seg_path=seg_path,
                bbox_path=bbox_path,
                boxes=(self._box("infer-box", "inference"),),
            )
            window._last_saved_segmentation_path = None
            window._active_segmentation_volume = lambda: (
                "semantic",
                _FakeVolume(Path(f"{raw_path}::generated-semantic::editable")),
            )
            save_calls = []

            def save_segmentation() -> bool:
                save_calls.append(True)
                window._last_saved_segmentation_path = str(saved_seg_path)
                window._last_saved_segmentation_kind = "semantic"
                return True

            window._save_active_segmentation_with_dialog = save_segmentation

            with patch("src.ui.main_window.show_info"):
                result = window._ensure_headless_segmentation_path()

            self.assertEqual(result, str(saved_seg_path))
            self.assertEqual(save_calls, [True])

    def test_headless_dirty_segmentation_forces_save_even_with_known_file(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "raw.npy"
            seg_path = root / "seg.npy"
            bbox_path = root / "boxes.bbox.txt"
            saved_seg_path = root / "resaved_seg.npy"
            self._touch_inputs(raw_path, seg_path, bbox_path)
            saved_seg_path.write_bytes(b"saved")
            window = self._make_window_like(
                raw_path=raw_path,
                seg_path=seg_path,
                bbox_path=bbox_path,
                boxes=(self._box("infer-box", "inference"),),
            )
            window._segmentation_editor = SimpleNamespace(dirty=True)
            save_calls = []

            def save_segmentation() -> bool:
                save_calls.append(True)
                window._last_saved_segmentation_path = str(saved_seg_path)
                window._last_saved_segmentation_kind = "semantic"
                window._segmentation_editor = SimpleNamespace(dirty=False)
                return True

            window._save_active_segmentation_with_dialog = save_segmentation

            with patch("src.ui.main_window.show_info"):
                result = window._ensure_headless_segmentation_path()

            self.assertEqual(result, str(saved_seg_path))
            self.assertEqual(save_calls, [True])

    def test_headless_bounding_boxes_without_known_file_force_save(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "raw.npy"
            seg_path = root / "seg.npy"
            bbox_path = root / "boxes.bbox.txt"
            saved_bbox_path = root / "saved_boxes.bbox.txt"
            self._touch_inputs(raw_path, seg_path, bbox_path)
            saved_bbox_path.write_text("saved\n", encoding="utf-8")
            window = self._make_window_like(
                raw_path=raw_path,
                seg_path=seg_path,
                bbox_path=bbox_path,
                boxes=(self._box("infer-box", "inference"),),
            )
            window._last_saved_bounding_boxes_path = None
            save_calls = []

            def save_bboxes() -> bool:
                save_calls.append(True)
                window._last_saved_bounding_boxes_path = str(saved_bbox_path)
                return True

            window._save_bounding_boxes_with_dialog = save_bboxes

            with patch("src.ui.main_window.show_info"):
                result = window._ensure_headless_bounding_boxes_path()

            self.assertEqual(result, str(saved_bbox_path))
            self.assertEqual(save_calls, [True])

    def test_headless_reopenable_path_rejects_app_virtual_qualifiers(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            volume_path = root / "volume.npy"
            volume_path.write_bytes(b"volume")

            self.assertTrue(MainWindow._is_headless_reopenable_path(str(volume_path)))
            self.assertTrue(
                MainWindow._is_headless_reopenable_path(f"{volume_path}::dataset")
            )
            self.assertFalse(
                MainWindow._is_headless_reopenable_path(f"{volume_path}::editable")
            )
            self.assertFalse(
                MainWindow._is_headless_reopenable_path(
                    f"{volume_path}::generated-semantic::editable"
                )
            )

    def test_release_ui_state_for_headless_close_clears_learning_and_closes_volumes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "raw.npy"
            seg_path = root / "seg.npy"
            bbox_path = root / "boxes.bbox.txt"
            self._touch_inputs(raw_path, seg_path, bbox_path)
            window = self._make_window_like(
                raw_path=raw_path,
                seg_path=seg_path,
                bbox_path=bbox_path,
                boxes=(self._box("infer-box", "inference"),),
            )
            raw_volume = window._raw_volume
            semantic_volume = window._semantic_volume
            window._release_ui_state_for_headless_close()

            self.assertTrue(raw_volume.closed)
            self.assertTrue(semantic_volume.closed)
            self.assertIsNone(window._raw_volume)
            self.assertIsNone(window._semantic_volume)
            self.assertIsNone(window._instance_volume)
            self.assertIsNone(window._segmentation_editor)
            self.assertEqual(
                window.learning_session_calls,
                [
                    "clear_dataloader_runtime",
                    "clear_eval_runtimes_by_box_id",
                    "clear_model_runtime",
                    "clear_bbox_batch",
                    "clear_label_space",
                ],
            )

    def _make_window_like(
        self,
        *,
        raw_path: Path,
        seg_path: Path,
        bbox_path: Path,
        boxes: tuple[BoundingBox, ...],
    ):
        fake = SimpleNamespace()
        fake._raw_volume = _FakeVolume(raw_path)
        fake._semantic_volume = _FakeVolume(seg_path)
        fake._instance_volume = None
        fake._segmentation_editor = None
        fake._last_saved_segmentation_path = str(seg_path)
        fake._last_saved_segmentation_kind = "semantic"
        fake._last_saved_bounding_boxes_path = str(bbox_path)
        fake._bbox_manager = _FakeBoxManager(boxes)
        fake.bottom_panel = SimpleNamespace(
            state=SimpleNamespace(
                bbox_rows=tuple(
                    BoundingBoxRow(
                        box_id=box.id,
                        label=box.label,
                        size_text="",
                        center_text="",
                    )
                    for box in boxes
                )
            )
        )
        fake._load_mode = "lazy"
        fake._cache_max_bytes = 1024
        fake._training_parameters = TrainingParameters()
        fake._headless_close_requested = False
        fake._pending_render_view_ids = {"axial"}
        fake._pending_annotation_peer_view_ids = {"coronal"}
        fake._annotation_dirty_views = {"sagittal"}
        fake._bbox_pending_peer_view_ids = {"axial"}
        fake._bbox_drag_staged_history_updates = {"box": object()}
        fake._global_history = SimpleNamespace(clear=lambda: None)
        fake.learning_session_calls = []
        fake._learning_session = _FakeLearningSession(fake.learning_session_calls)
        fake.closed = False
        fake._abort_if_learning_training_running = lambda: False
        fake._active_segmentation_volume = lambda: ("semantic", _FakeVolume(seg_path))
        fake._save_active_segmentation_with_dialog = lambda: False
        fake._save_bounding_boxes_with_dialog = lambda: False
        fake.close = lambda: setattr(fake, "closed", True)
        for name in (
            "_launch_headless_learning_job_and_close",
            "_prepare_headless_common_inputs",
            "_build_headless_training_spec",
            "_build_headless_inference_spec",
            "_prepare_headless_input_checkpoint",
            "_ensure_headless_segmentation_path",
            "_ensure_headless_bounding_boxes_path",
            "_require_headless_training_boxes",
            "_require_headless_inference_boxes",
            "_headless_reopenable_volume_path",
            "_release_ui_state_for_headless_close",
            "_release_learning_state_for_headless_close",
            "_close_loaded_volumes_for_headless_close",
        ):
            setattr(fake, name, MethodType(getattr(MainWindow, name), fake))
        fake._is_headless_reopenable_path = MainWindow._is_headless_reopenable_path
        fake._has_unsaved_segmentation_changes = MethodType(
            MainWindow._has_unsaved_segmentation_changes,
            fake,
        )
        fake._has_unsaved_bounding_box_changes = MethodType(
            MainWindow._has_unsaved_bounding_box_changes,
            fake,
        )
        return fake

    @staticmethod
    def _touch_inputs(raw_path: Path, seg_path: Path, bbox_path: Path) -> None:
        np.save(raw_path, np.zeros((2, 2, 2), dtype=np.float32))
        np.save(seg_path, np.zeros((2, 2, 2), dtype=np.uint8))
        bbox_path.write_text("bbox placeholder\n", encoding="utf-8")

    @staticmethod
    def _box(
        box_id: str,
        label: str,
        *,
        z0: int = 0,
        z1: int = 1,
    ) -> BoundingBox:
        return BoundingBox.from_bounds(
            box_id=box_id,
            z0=z0,
            z1=z1,
            y0=0,
            y1=1,
            x0=0,
            x1=1,
            label=label,
            volume_shape=(2, 2, 2),
        )


class _FakeVolume:
    def __init__(self, path: Path) -> None:
        self.loader = SimpleNamespace(path=str(path))
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeBoxManager:
    volume_shape = (2, 2, 2)
    dirty = False

    def __init__(self, boxes: tuple[BoundingBox, ...]) -> None:
        self._boxes = tuple(boxes)

    def boxes(self) -> tuple[BoundingBox, ...]:
        return self._boxes


class _FakeLearningSession:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def clear_dataloader_runtime(self) -> None:
        self._calls.append("clear_dataloader_runtime")

    def clear_eval_runtimes_by_box_id(self) -> None:
        self._calls.append("clear_eval_runtimes_by_box_id")

    def clear_model_runtime(self) -> None:
        self._calls.append("clear_model_runtime")

    def clear_bbox_batch(self) -> None:
        self._calls.append("clear_bbox_batch")

    def clear_label_space(self) -> None:
        self._calls.append("clear_label_space")


if __name__ == "__main__":
    unittest.main()
