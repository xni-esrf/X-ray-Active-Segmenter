from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.headless.job_spec import (
    DEFAULT_HEADLESS_JOB_DIR_NAME,
    HeadlessJobSpec,
    load_headless_job_spec,
    save_headless_job_spec,
)
from src.learning import TrainingParameters


class HeadlessJobSpecTests(unittest.TestCase):
    def test_default_job_dir_is_visible_headless_job_directory(self) -> None:
        spec = HeadlessJobSpec(
            kind="train",
            raw_volume_path="raw.npy",
            segmentation_path="seg.npy",
            bbox_path="boxes.json",
            input_checkpoint_path="input.cp",
            output_checkpoint_path="output.cp",
        )

        self.assertEqual(DEFAULT_HEADLESS_JOB_DIR_NAME, "headless-job")
        self.assertEqual(spec.job_dir, DEFAULT_HEADLESS_JOB_DIR_NAME)

    def test_roundtrip_training_spec(self) -> None:
        with TemporaryDirectory() as tmpdir:
            job_path = str(Path(tmpdir) / "job.json")
            spec = HeadlessJobSpec(
                kind="train",
                raw_volume_path="raw.npy",
                segmentation_path="seg.npy",
                segmentation_kind="semantic",
                bbox_path="boxes.json",
                load_mode="LAZY",
                cache_max_bytes=1234,
                training_parameters=TrainingParameters(
                    learning_rate=0.001,
                    training_batch_size=2,
                    validation_batch_size=3,
                    inference_batch_size=6,
                    patches_per_epoch=4,
                    early_stopping_patience=5,
                ),
                input_checkpoint_path="input.cp",
                output_checkpoint_path="output.cp",
                job_dir=str(Path(tmpdir) / "headless-job"),
                source_pid=42,
            )

            saved_path = save_headless_job_spec(spec, job_path)
            loaded = load_headless_job_spec(saved_path)

            self.assertEqual(loaded.kind, "train")
            self.assertEqual(loaded.load_mode, "lazy")
            self.assertEqual(loaded.cache_max_bytes, 1234)
            self.assertEqual(loaded.training_parameters.learning_rate, 0.001)
            self.assertEqual(loaded.training_parameters.training_batch_size, 2)
            self.assertEqual(loaded.training_parameters.inference_batch_size, 6)
            self.assertEqual(loaded.output_checkpoint_path, "output.cp")
            self.assertEqual(loaded.source_pid, 42)

    def test_inference_spec_requires_checkpoint_output_path_and_format(self) -> None:
        valid_kwargs = dict(
            kind="inference",
            raw_volume_path="raw.npy",
            segmentation_path="seg.npy",
            segmentation_kind="instance",
            bbox_path="boxes.json",
            input_checkpoint_path="model.cp",
            output_segmentation_path="out.zarr",
            output_segmentation_format="zarr",
        )

        spec = HeadlessJobSpec(**valid_kwargs)
        self.assertEqual(spec.output_segmentation_format, "zarr")

        for removed_key in (
            "input_checkpoint_path",
            "output_segmentation_path",
            "output_segmentation_format",
        ):
            kwargs = dict(valid_kwargs)
            kwargs[removed_key] = None
            with self.subTest(removed_key=removed_key):
                with self.assertRaises((TypeError, ValueError)):
                    HeadlessJobSpec(**kwargs)

        with self.assertRaisesRegex(ValueError, "zarr"):
            HeadlessJobSpec(**dict(valid_kwargs, output_segmentation_format="npy"))

    def test_roundtrip_inference_spec(self) -> None:
        with TemporaryDirectory() as tmpdir:
            job_path = str(Path(tmpdir) / "job.json")
            spec = HeadlessJobSpec(
                kind="inference",
                raw_volume_path="raw.npy",
                segmentation_path="seg.npy",
                segmentation_kind="semantic",
                bbox_path="boxes.bbox.txt",
                load_mode="ram",
                cache_max_bytes=4096,
                input_checkpoint_path="input.cp",
                output_segmentation_path="output.zarr",
                output_segmentation_format="zarr",
                job_dir=str(Path(tmpdir) / "headless-job"),
                source_pid=123,
            )

            saved_path = save_headless_job_spec(spec, job_path)
            loaded = load_headless_job_spec(saved_path)

            self.assertEqual(loaded.kind, "inference")
            self.assertEqual(loaded.segmentation_kind, "semantic")
            self.assertEqual(loaded.load_mode, "ram")
            self.assertEqual(loaded.cache_max_bytes, 4096)
            self.assertEqual(loaded.input_checkpoint_path, "input.cp")
            self.assertEqual(loaded.output_segmentation_path, "output.zarr")
            self.assertEqual(loaded.output_segmentation_format, "zarr")
            self.assertEqual(loaded.source_pid, 123)

    def test_inference_spec_allows_missing_input_segmentation(self) -> None:
        spec = HeadlessJobSpec(
            kind="inference",
            raw_volume_path="raw.npy",
            bbox_path="boxes.bbox.txt",
            input_checkpoint_path="input.cp",
            output_segmentation_path="output.zarr",
            output_segmentation_format="zarr",
        )

        self.assertIsNone(spec.segmentation_path)
        self.assertEqual(spec.segmentation_kind, "semantic")

    def test_rejects_invalid_kind_load_mode_and_segmentation_kind(self) -> None:
        base = dict(
            kind="train",
            raw_volume_path="raw.npy",
            segmentation_path="seg.npy",
            segmentation_kind="semantic",
            bbox_path="boxes.json",
            input_checkpoint_path="input.cp",
            output_checkpoint_path="output.cp",
        )
        invalid_cases = (
            dict(base, kind="other"),
            dict(base, load_mode="eager"),
            dict(base, segmentation_kind="mask"),
        )
        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    HeadlessJobSpec(**kwargs)


if __name__ == "__main__":
    unittest.main()
