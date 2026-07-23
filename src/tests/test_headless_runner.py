from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from src.bbox import BoundingBox, save_bounding_boxes
from src.headless.job_spec import HeadlessJobSpec, save_headless_job_spec
from src.headless.runner import main as headless_main
from src.learning import TrainingParameters
import src.headless.runner as runner_module


class HeadlessRunnerTests(unittest.TestCase):
    def test_periodic_checkpoint_path_adds_epoch_before_cp_suffix(self) -> None:
        self.assertEqual(
            runner_module._periodic_checkpoint_path("/tmp/model.cp", epoch=5),
            "/tmp/model_epoch5.cp",
        )

    def test_periodic_checkpoint_manager_saves_every_interval_and_replaces_previous(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            final_path = root / "trained_model.cp"
            runtime = SimpleNamespace()
            saved_paths: list[str] = []

            def _save_checkpoint(*, runtime: object, checkpoint_path: str) -> str:
                del runtime
                Path(checkpoint_path).write_text("checkpoint", encoding="utf-8")
                saved_paths.append(checkpoint_path)
                return checkpoint_path

            manager = runner_module._HeadlessPeriodicCheckpointManager(
                runtime=runtime,
                final_checkpoint_path=str(final_path),
            )

            with patch.object(
                runner_module,
                "save_foundation_model_checkpoint",
                side_effect=_save_checkpoint,
            ):
                manager.on_epoch_completed(SimpleNamespace(completed_epoch_count=4))
                manager.on_epoch_completed(SimpleNamespace(completed_epoch_count=5))
                epoch5_path = root / "trained_model_epoch5.cp"
                self.assertTrue(epoch5_path.exists())
                self.assertEqual(manager.latest_checkpoint_path, str(epoch5_path))

                manager.on_epoch_completed(SimpleNamespace(completed_epoch_count=10))

            epoch10_path = root / "trained_model_epoch10.cp"
            self.assertEqual(
                saved_paths,
                [
                    str(epoch5_path),
                    str(epoch10_path),
                ],
            )
            self.assertFalse(epoch5_path.exists())
            self.assertTrue(epoch10_path.exists())
            self.assertEqual(manager.latest_checkpoint_path, str(epoch10_path))

    def test_periodic_checkpoint_manager_cleanup_removes_latest_after_final_save(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            final_path = root / "trained_model.cp"
            runtime = SimpleNamespace()

            def _save_checkpoint(*, runtime: object, checkpoint_path: str) -> str:
                del runtime
                Path(checkpoint_path).write_text("checkpoint", encoding="utf-8")
                return checkpoint_path

            manager = runner_module._HeadlessPeriodicCheckpointManager(
                runtime=runtime,
                final_checkpoint_path=str(final_path),
            )

            with patch.object(
                runner_module,
                "save_foundation_model_checkpoint",
                side_effect=_save_checkpoint,
            ):
                manager.on_epoch_completed(SimpleNamespace(completed_epoch_count=5))

            epoch5_path = root / "trained_model_epoch5.cp"
            self.assertTrue(epoch5_path.exists())

            manager.cleanup_after_successful_final_save()

            self.assertFalse(epoch5_path.exists())
            self.assertIsNone(manager.latest_checkpoint_path)

    def test_training_final_save_failure_keeps_latest_periodic_checkpoint(self) -> None:
        sources = SimpleNamespace(
            raw_volume=SimpleNamespace(shape=(2, 2, 2), dtype="float16"),
            segmentation_volume=SimpleNamespace(shape=(2, 2, 2), dtype="uint8"),
            boxes_by_id={"box-1": object()},
            close=lambda: None,
        )
        context = SimpleNamespace(sources=sources)
        label_space = SimpleNamespace(
            label_values=(0, 1),
            background_label=0,
            mask_label=-100,
            source_signature=("semantic", "seg.npy", None),
        )
        prepared_state = SimpleNamespace(
            label_space=label_space,
            label_coverage_warning=None,
            train_box_ids=("train-box",),
            validation_box_ids=("val-box",),
        )
        runtime = SimpleNamespace(
            hyperparameters={},
            checkpoint_path="input.cp",
        )
        preconditions = SimpleNamespace(
            train_runtime=SimpleNamespace(train_count=3),
            model_runtime=runtime,
        )
        result = SimpleNamespace(
            stop_reason="early_stop",
            completed_epoch_count=5,
            total_epoch_count=6,
            best_epoch=5,
            best_weighted_mean_dice=0.75,
        )
        spec = HeadlessJobSpec(
            kind="train",
            raw_volume_path="raw.npy",
            segmentation_path="seg.npy",
            segmentation_kind="semantic",
            bbox_path="boxes.json",
            input_checkpoint_path="input.cp",
            output_checkpoint_path="output.cp",
        )
        manager = SimpleNamespace(
            on_epoch_completed=lambda _progress: None,
            cleanup_after_successful_final_save=Mock(),
        )

        with patch.object(
            runner_module,
            "prepare_learning_state_from_sources",
            return_value=prepared_state,
        ), patch.object(
            runner_module,
            "validate_foundation_model_instantiation_preconditions",
            return_value=SimpleNamespace(num_classes=2, device_ids=(0, 1)),
        ), patch.object(
            runner_module,
            "instantiate_model_runtime_from_checkpoint",
            return_value=runtime,
        ), patch.object(
            runner_module,
            "validate_training_preconditions_for_session",
            return_value=preconditions,
        ), patch.object(
            runner_module,
            "_HeadlessPeriodicCheckpointManager",
            return_value=manager,
        ), patch.object(
            runner_module,
            "train_learning_model_with_validation_loop",
            return_value=result,
        ), patch.object(
            runner_module,
            "save_foundation_model_checkpoint",
            side_effect=RuntimeError("final save failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "final save failed"):
                runner_module._run_training_job(spec, context)

        manager.cleanup_after_successful_final_save.assert_not_called()

    def test_training_user_stop_keeps_latest_periodic_checkpoint_after_final_save(self) -> None:
        sources = SimpleNamespace(
            raw_volume=SimpleNamespace(shape=(2, 2, 2), dtype="float16"),
            segmentation_volume=SimpleNamespace(shape=(2, 2, 2), dtype="uint8"),
            boxes_by_id={"box-1": object()},
            close=lambda: None,
        )
        context = SimpleNamespace(sources=sources)
        label_space = SimpleNamespace(
            label_values=(0, 1),
            background_label=0,
            mask_label=-100,
            source_signature=("semantic", "seg.npy", None),
        )
        prepared_state = SimpleNamespace(
            label_space=label_space,
            label_coverage_warning=None,
            train_box_ids=("train-box",),
            validation_box_ids=("val-box",),
        )
        runtime = SimpleNamespace(
            hyperparameters={},
            checkpoint_path="input.cp",
        )
        preconditions = SimpleNamespace(
            train_runtime=SimpleNamespace(train_count=3),
            model_runtime=runtime,
        )
        result = SimpleNamespace(
            stop_reason="user_stop",
            completed_epoch_count=5,
            total_epoch_count=6,
            best_epoch=5,
            best_weighted_mean_dice=0.75,
        )
        spec = HeadlessJobSpec(
            kind="train",
            raw_volume_path="raw.npy",
            segmentation_path="seg.npy",
            segmentation_kind="semantic",
            bbox_path="boxes.json",
            input_checkpoint_path="input.cp",
            output_checkpoint_path="output.cp",
        )
        manager = SimpleNamespace(
            on_epoch_completed=lambda _progress: None,
            cleanup_after_successful_final_save=Mock(),
        )

        with patch.object(
            runner_module,
            "prepare_learning_state_from_sources",
            return_value=prepared_state,
        ), patch.object(
            runner_module,
            "validate_foundation_model_instantiation_preconditions",
            return_value=SimpleNamespace(num_classes=2, device_ids=(0, 1)),
        ), patch.object(
            runner_module,
            "instantiate_model_runtime_from_checkpoint",
            return_value=runtime,
        ), patch.object(
            runner_module,
            "validate_training_preconditions_for_session",
            return_value=preconditions,
        ), patch.object(
            runner_module,
            "_HeadlessPeriodicCheckpointManager",
            return_value=manager,
        ), patch.object(
            runner_module,
            "train_learning_model_with_validation_loop",
            return_value=result,
        ), patch.object(
            runner_module,
            "save_foundation_model_checkpoint",
            return_value="output.cp",
        ):
            runner_module._run_training_job(spec, context)

        manager.cleanup_after_successful_final_save.assert_not_called()

    def test_training_exception_keeps_latest_periodic_checkpoint(self) -> None:
        sources = SimpleNamespace(
            raw_volume=SimpleNamespace(shape=(2, 2, 2), dtype="float16"),
            segmentation_volume=SimpleNamespace(shape=(2, 2, 2), dtype="uint8"),
            boxes_by_id={"box-1": object()},
            close=lambda: None,
        )
        context = SimpleNamespace(sources=sources)
        label_space = SimpleNamespace(
            label_values=(0, 1),
            background_label=0,
            mask_label=-100,
            source_signature=("semantic", "seg.npy", None),
        )
        prepared_state = SimpleNamespace(
            label_space=label_space,
            label_coverage_warning=None,
            train_box_ids=("train-box",),
            validation_box_ids=("val-box",),
        )
        runtime = SimpleNamespace(
            hyperparameters={},
            checkpoint_path="input.cp",
        )
        preconditions = SimpleNamespace(
            train_runtime=SimpleNamespace(train_count=3),
            model_runtime=runtime,
        )
        spec = HeadlessJobSpec(
            kind="train",
            raw_volume_path="raw.npy",
            segmentation_path="seg.npy",
            segmentation_kind="semantic",
            bbox_path="boxes.json",
            input_checkpoint_path="input.cp",
            output_checkpoint_path="output.cp",
        )
        manager = SimpleNamespace(
            on_epoch_completed=lambda _progress: None,
            cleanup_after_successful_final_save=Mock(),
        )

        with patch.object(
            runner_module,
            "prepare_learning_state_from_sources",
            return_value=prepared_state,
        ), patch.object(
            runner_module,
            "validate_foundation_model_instantiation_preconditions",
            return_value=SimpleNamespace(num_classes=2, device_ids=(0, 1)),
        ), patch.object(
            runner_module,
            "instantiate_model_runtime_from_checkpoint",
            return_value=runtime,
        ), patch.object(
            runner_module,
            "validate_training_preconditions_for_session",
            return_value=preconditions,
        ), patch.object(
            runner_module,
            "_HeadlessPeriodicCheckpointManager",
            return_value=manager,
        ), patch.object(
            runner_module,
            "train_learning_model_with_validation_loop",
            side_effect=RuntimeError("training failed"),
        ), patch.object(
            runner_module,
            "save_foundation_model_checkpoint",
        ) as save_mock:
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                runner_module._run_training_job(spec, context)

        save_mock.assert_not_called()
        manager.cleanup_after_successful_final_save.assert_not_called()

    def test_validate_only_reopens_inputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "raw.npy"
            seg_path = root / "seg.npy"
            bbox_path = root / "boxes.json"
            checkpoint_path = root / "input.cp"
            job_dir = root / "headless-job"
            job_path = job_dir / "job.json"

            np.save(raw_path, np.arange(27, dtype=np.float32).reshape(3, 3, 3))
            np.save(seg_path, np.ones((3, 3, 3), dtype=np.uint8))
            checkpoint_path.write_bytes(b"checkpoint")
            save_bounding_boxes(
                str(bbox_path),
                volume_shape=(3, 3, 3),
                boxes=(
                    BoundingBox.from_bounds(
                        box_id="box-1",
                        z0=0,
                        z1=2,
                        y0=0,
                        y1=2,
                        x0=0,
                        x1=2,
                        label="train",
                        volume_shape=(3, 3, 3),
                    ),
                ),
            )
            spec = HeadlessJobSpec(
                kind="train",
                raw_volume_path=str(raw_path),
                segmentation_path=str(seg_path),
                segmentation_kind="semantic",
                bbox_path=str(bbox_path),
                load_mode="lazy",
                cache_max_bytes=1024 * 1024,
                input_checkpoint_path=str(checkpoint_path),
                output_checkpoint_path=str(root / "trained.cp"),
                job_dir=str(job_dir),
            )
            save_headless_job_spec(spec, str(job_path))

            exit_code = headless_main([str(job_path), "--validate-only"])

            self.assertEqual(exit_code, 0)
            self.assertTrue((job_dir / "headless.log").exists())

    def test_validate_only_reopens_inference_inputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "raw.npy"
            seg_path = root / "seg.npy"
            bbox_path = root / "boxes.json"
            checkpoint_path = root / "input.cp"
            job_dir = root / "headless-job"
            job_path = job_dir / "job.json"

            np.save(raw_path, np.arange(27, dtype=np.float32).reshape(3, 3, 3))
            np.save(seg_path, np.zeros((3, 3, 3), dtype=np.uint8))
            checkpoint_path.write_bytes(b"checkpoint")
            save_bounding_boxes(
                str(bbox_path),
                volume_shape=(3, 3, 3),
                boxes=(
                    BoundingBox.from_bounds(
                        box_id="infer-box",
                        z0=0,
                        z1=2,
                        y0=0,
                        y1=2,
                        x0=0,
                        x1=2,
                        label="inference",
                        volume_shape=(3, 3, 3),
                    ),
                ),
            )
            spec = HeadlessJobSpec(
                kind="inference",
                raw_volume_path=str(raw_path),
                segmentation_path=str(seg_path),
                segmentation_kind="semantic",
                bbox_path=str(bbox_path),
                load_mode="lazy",
                cache_max_bytes=1024 * 1024,
                input_checkpoint_path=str(checkpoint_path),
                output_segmentation_path=str(root / "out.zarr"),
                output_segmentation_format="zarr",
                job_dir=str(job_dir),
            )
            save_headless_job_spec(spec, str(job_path))

            exit_code = headless_main([str(job_path), "--validate-only"])

            self.assertEqual(exit_code, 0)
            self.assertTrue((job_dir / "headless.log").exists())

    def test_validate_only_reopens_inference_inputs_without_segmentation(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "raw.npy"
            bbox_path = root / "boxes.json"
            checkpoint_path = root / "input.cp"
            job_dir = root / "headless-job"
            job_path = job_dir / "job.json"

            np.save(raw_path, np.arange(27, dtype=np.float32).reshape(3, 3, 3))
            torch.save(
                {
                    "metadata": {
                        "num_classes": 3,
                        "hyperparameters": {"label_values": [0, 1, 2]},
                    },
                },
                checkpoint_path,
            )
            save_bounding_boxes(
                str(bbox_path),
                volume_shape=(3, 3, 3),
                boxes=(
                    BoundingBox.from_bounds(
                        box_id="infer-box",
                        z0=0,
                        z1=2,
                        y0=0,
                        y1=2,
                        x0=0,
                        x1=2,
                        label="inference",
                        volume_shape=(3, 3, 3),
                    ),
                ),
            )
            spec = HeadlessJobSpec(
                kind="inference",
                raw_volume_path=str(raw_path),
                bbox_path=str(bbox_path),
                load_mode="lazy",
                cache_max_bytes=1024 * 1024,
                input_checkpoint_path=str(checkpoint_path),
                output_segmentation_path=str(root / "out.zarr"),
                output_segmentation_format="zarr",
                job_dir=str(job_dir),
            )
            save_headless_job_spec(spec, str(job_path))

            exit_code = headless_main([str(job_path), "--validate-only"])

            self.assertEqual(exit_code, 0)
            self.assertTrue((job_dir / "headless.log").exists())

    def test_non_validate_train_returns_not_implemented_code(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "raw.npy"
            seg_path = root / "seg.npy"
            bbox_path = root / "boxes.json"
            checkpoint_path = root / "input.cp"
            job_dir = root / "headless-job"
            job_path = job_dir / "job.json"

            np.save(raw_path, np.zeros((2, 2, 2), dtype=np.float32))
            np.save(seg_path, np.zeros((2, 2, 2), dtype=np.uint8))
            checkpoint_path.write_bytes(b"checkpoint")
            save_bounding_boxes(
                str(bbox_path),
                volume_shape=(2, 2, 2),
                boxes=(
                    BoundingBox.from_bounds(
                        box_id="box-1",
                        z0=0,
                        z1=1,
                        y0=0,
                        y1=1,
                        x0=0,
                        x1=1,
                        label="train",
                        volume_shape=(2, 2, 2),
                    ),
                ),
            )
            spec = HeadlessJobSpec(
                kind="train",
                raw_volume_path=str(raw_path),
                segmentation_path=str(seg_path),
                segmentation_kind="semantic",
                bbox_path=str(bbox_path),
                input_checkpoint_path=str(checkpoint_path),
                output_checkpoint_path=str(root / "trained.cp"),
                job_dir=str(job_dir),
            )
            save_headless_job_spec(spec, str(job_path))

            exit_code = headless_main([str(job_path)])

            self.assertEqual(exit_code, 1)

    def test_training_job_orchestration_uses_session_checkpoint_and_saves_output(self) -> None:
        sources = SimpleNamespace(
            raw_volume=SimpleNamespace(shape=(2, 2, 2), dtype="float16"),
            segmentation_volume=SimpleNamespace(shape=(2, 2, 2), dtype="uint8"),
            boxes_by_id={"box-1": object()},
            close=lambda: None,
        )
        context = SimpleNamespace(sources=sources)
        label_space = SimpleNamespace(
            label_values=(0, 1),
            background_label=0,
            mask_label=-100,
            source_signature=("semantic", "seg.npy", None),
        )
        prepared_state = SimpleNamespace(
            label_space=label_space,
            label_coverage_warning=None,
            train_box_ids=("train-box",),
            validation_box_ids=("val-box",),
        )
        runtime = SimpleNamespace(
            hyperparameters={},
            checkpoint_path="input.cp",
        )
        train_runtime = SimpleNamespace(train_count=3)
        preconditions = SimpleNamespace(
            train_runtime=train_runtime,
            model_runtime=runtime,
        )
        result = SimpleNamespace(
            stop_reason="early_stop",
            completed_epoch_count=2,
            total_epoch_count=6,
            best_epoch=2,
            best_weighted_mean_dice=0.75,
        )
        spec = HeadlessJobSpec(
            kind="train",
            raw_volume_path="raw.npy",
            segmentation_path="seg.npy",
            segmentation_kind="semantic",
            bbox_path="boxes.json",
            input_checkpoint_path="input.cp",
            output_checkpoint_path="output.cp",
        )

        with patch.object(
            runner_module,
            "prepare_learning_state_from_sources",
            return_value=prepared_state,
        ) as prepare_mock, patch.object(
            runner_module,
            "validate_foundation_model_instantiation_preconditions",
            return_value=SimpleNamespace(num_classes=2, device_ids=(0, 1)),
        ) as instantiate_preconditions_mock, patch.object(
            runner_module,
            "instantiate_model_runtime_from_checkpoint",
            return_value=runtime,
        ) as instantiate_mock, patch.object(
            runner_module,
            "validate_training_preconditions_for_session",
            return_value=preconditions,
        ) as training_preconditions_mock, patch.object(
            runner_module,
            "train_learning_model_with_validation_loop",
            return_value=result,
        ) as train_mock, patch.object(
            runner_module,
            "save_foundation_model_checkpoint",
            return_value="output.cp",
        ) as save_mock:
            runner_module._run_training_job(spec, context)

        prepare_mock.assert_called_once()
        self.assertTrue(prepare_mock.call_args.kwargs["require_class_weights"])
        instantiate_preconditions_mock.assert_called_once()
        instantiate_mock.assert_called_once()
        self.assertEqual(instantiate_mock.call_args.kwargs["num_classes"], 2)
        self.assertEqual(instantiate_mock.call_args.kwargs["device_ids"], (0, 1))
        training_preconditions_mock.assert_called_once()
        train_mock.assert_called_once()
        self.assertEqual(
            train_mock.call_args.kwargs["early_stop_patience"],
            spec.training_parameters.early_stopping_patience,
        )
        self.assertTrue(callable(train_mock.call_args.kwargs["epoch_completion_callback"]))
        save_mock.assert_called_once_with(runtime=runtime, checkpoint_path="output.cp")
        self.assertEqual(runtime.hyperparameters["label_values"], (0, 1))
        self.assertTrue(runtime.hyperparameters["trained_in_app"])
        self.assertEqual(runtime.hyperparameters["training_run_count"], 1)

    def test_inference_job_orchestration_runs_large_crop_zarr_inference(self) -> None:
        inference_box = BoundingBox.from_bounds(
            box_id="infer-box",
            z0=0,
            z1=1,
            y0=0,
            y1=2,
            x0=0,
            x1=2,
            label="inference",
            volume_shape=(2, 2, 2),
        )
        train_box = BoundingBox.from_bounds(
            box_id="train-box",
            z0=1,
            z1=2,
            y0=0,
            y1=1,
            x0=0,
            x1=1,
            label="train",
            volume_shape=(2, 2, 2),
        )
        raw_array = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        segmentation_array = np.zeros((2, 2, 2), dtype=np.uint8)
        raw_volume = _FakeVolume(raw_array, axes="zyx")
        segmentation_volume = _FakeVolume(segmentation_array, axes="zyx")
        sources = SimpleNamespace(
            raw_volume=raw_volume,
            segmentation_volume=segmentation_volume,
            boxes_by_id={
                "train-box": train_box,
                "infer-box": inference_box,
            },
            ordered_box_ids=("train-box", "infer-box"),
        )
        context = SimpleNamespace(
            sources=sources,
            raw_volume=raw_volume,
            segmentation_volume=segmentation_volume,
        )
        runtime = SimpleNamespace(model=object())
        tmpdir_cm = TemporaryDirectory()
        self.addCleanup(tmpdir_cm.cleanup)
        root = Path(tmpdir_cm.name)
        output_path = root / "output.zarr"
        job_dir = root / "headless-job" / "inference-job"
        spec = HeadlessJobSpec(
            kind="inference",
            raw_volume_path="raw.npy",
            segmentation_path="seg.npy",
            segmentation_kind="semantic",
            bbox_path="boxes.json",
            input_checkpoint_path="input.cp",
            output_segmentation_path=str(output_path),
            output_segmentation_format="zarr",
            training_parameters=TrainingParameters(
                inference_batch_size=9,
                skip_empty_regions=True,
            ),
            job_dir=str(job_dir),
        )

        with patch.object(
            runner_module,
            "validate_foundation_checkpoint_load_preconditions",
            return_value=SimpleNamespace(
                num_classes=2,
                label_values=(0, 1),
                device_ids=(0, 1),
            ),
        ) as validate_mock, patch.object(
            runner_module,
            "instantiate_model_runtime_from_checkpoint",
            return_value=runtime,
        ) as instantiate_mock, patch.object(
            runner_module,
            "run_large_crop_inference_to_zarr",
            return_value=SimpleNamespace(
                output_path=str(output_path),
                plan=SimpleNamespace(total_crop_count=1),
                written_crop_count=1,
            ),
        ) as inference_mock:
            runner_module._run_inference_job(spec, context)

        validate_mock.assert_called_once_with("input.cp", require_min_gpu_count=2)
        instantiate_mock.assert_called_once_with(
            checkpoint_path="input.cp",
            num_classes=2,
            device_ids=(0, 1),
        )
        inference_mock.assert_called_once()
        self.assertEqual(inference_mock.call_args.kwargs["model_runtime"], runtime)
        self.assertIs(inference_mock.call_args.kwargs["raw_volume"], raw_volume)
        self.assertEqual(
            inference_mock.call_args.kwargs["requested_bounds"],
            ((0, 1), (0, 2), (0, 2)),
        )
        self.assertEqual(inference_mock.call_args.kwargs["label_values"], (0, 1))
        self.assertEqual(inference_mock.call_args.kwargs["output_path"], str(output_path))
        self.assertEqual(inference_mock.call_args.kwargs["output_dtype"], np.dtype(np.uint8))
        self.assertIs(inference_mock.call_args.kwargs["overwrite"], True)
        self.assertEqual(inference_mock.call_args.kwargs["batch_size"], 9)
        self.assertIs(inference_mock.call_args.kwargs["skip_empty_regions"], True)

    def test_headless_inference_output_dtype_uses_label_range_not_segmentation_dtype(self) -> None:
        context = SimpleNamespace(
            segmentation_volume=SimpleNamespace(dtype=np.dtype(np.int32))
        )

        self.assertEqual(
            runner_module._headless_inference_output_dtype(
                (0, 10),
                context=context,
            ),
            np.dtype(np.uint8),
        )
        self.assertEqual(
            runner_module._headless_inference_output_dtype(
                (0, 256),
                context=context,
            ),
            np.dtype(np.uint16),
        )
        self.assertEqual(
            runner_module._headless_inference_output_dtype(
                (-1, 10),
                context=context,
            ),
            np.dtype(np.int32),
        )
        self.assertEqual(
            runner_module._headless_inference_output_dtype(
                (0, np.iinfo(np.int32).max + 1),
                context=context,
            ),
            np.dtype(np.int64),
        )

    def test_inference_job_rejects_non_zarr_output_before_model_setup(self) -> None:
        inference_box = BoundingBox.from_bounds(
            box_id="infer-box",
            z0=0,
            z1=1,
            y0=0,
            y1=1,
            x0=0,
            x1=1,
            label="inference",
            volume_shape=(1, 1, 1),
        )
        raw_volume = _FakeVolume(np.zeros((1, 1, 1), dtype=np.float32), axes="zyx")
        segmentation_volume = _FakeVolume(np.zeros((1, 1, 1), dtype=np.uint8), axes="zyx")
        context = SimpleNamespace(
            sources=SimpleNamespace(
                raw_volume=raw_volume,
                segmentation_volume=segmentation_volume,
                boxes_by_id={"infer-box": inference_box},
                ordered_box_ids=("infer-box",),
            ),
            raw_volume=raw_volume,
            segmentation_volume=segmentation_volume,
        )
        spec = SimpleNamespace(
            input_checkpoint_path="input.cp",
            output_segmentation_path="output.npy",
            output_segmentation_format="npy",
        )

        with patch.object(
            runner_module,
            "validate_foundation_checkpoint_load_preconditions",
        ) as validate_mock:
            with self.assertRaisesRegex(ValueError, "Zarr output"):
                runner_module._run_inference_job(spec, context)

        validate_mock.assert_not_called()

    def test_inference_job_requires_exactly_one_inference_box_before_model_setup(self) -> None:
        first_box = BoundingBox.from_bounds(
            box_id="infer-a",
            z0=0,
            z1=1,
            y0=0,
            y1=1,
            x0=0,
            x1=1,
            label="inference",
            volume_shape=(2, 2, 2),
        )
        second_box = BoundingBox.from_bounds(
            box_id="infer-b",
            z0=1,
            z1=2,
            y0=1,
            y1=2,
            x0=1,
            x1=2,
            label="inference",
            volume_shape=(2, 2, 2),
        )
        raw_volume = _FakeVolume(np.zeros((2, 2, 2), dtype=np.float32), axes="zyx")
        segmentation_volume = _FakeVolume(np.zeros((2, 2, 2), dtype=np.uint8), axes="zyx")
        context = SimpleNamespace(
            sources=SimpleNamespace(
                raw_volume=raw_volume,
                segmentation_volume=segmentation_volume,
                boxes_by_id={
                    "infer-a": first_box,
                    "infer-b": second_box,
                },
                ordered_box_ids=("infer-a", "infer-b"),
            ),
            raw_volume=raw_volume,
            segmentation_volume=segmentation_volume,
        )
        spec = HeadlessJobSpec(
            kind="inference",
            raw_volume_path="raw.npy",
            segmentation_path="seg.npy",
            segmentation_kind="semantic",
            bbox_path="boxes.json",
            input_checkpoint_path="input.cp",
            output_segmentation_path="output.zarr",
            output_segmentation_format="zarr",
        )

        with patch.object(
            runner_module,
            "validate_foundation_checkpoint_load_preconditions",
        ) as validate_mock:
            with self.assertRaisesRegex(ValueError, "exactly one inference bbox"):
                runner_module._run_inference_job(spec, context)

        validate_mock.assert_not_called()


class _FakeVolume:
    def __init__(self, array: np.ndarray, *, axes: str) -> None:
        self._array = np.asarray(array)
        self.shape = tuple(self._array.shape)
        self.dtype = self._array.dtype
        self.info = SimpleNamespace(
            dtype=str(self._array.dtype),
            voxel_spacing=(1.0, 1.0, 1.0),
            axes=axes,
        )

    def get_chunk(self, zyx_slices):
        return np.asarray(self._array[zyx_slices])


if __name__ == "__main__":
    unittest.main()
