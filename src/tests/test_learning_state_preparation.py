from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from src.bbox import BoundingBox, save_bounding_boxes
from src.data import open_volume
from src.io.loader import InMemoryVolumeLoader
from src.learning.state_preparation import (
    load_learning_sources_from_paths,
    prepare_learning_state_from_volumes,
    semantic_label_space_source_signature,
)


class LearningStatePreparationTests(unittest.TestCase):
    def test_load_learning_sources_from_paths_opens_volumes_and_boxes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "raw.npy"
            seg_path = root / "seg.npy"
            bbox_path = root / "boxes.json"
            np.save(raw_path, np.zeros((3, 3, 3), dtype=np.float32))
            np.save(seg_path, np.ones((3, 3, 3), dtype=np.uint8))
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

            sources = load_learning_sources_from_paths(
                raw_volume_path=str(raw_path),
                segmentation_path=str(seg_path),
                segmentation_kind="semantic",
                bbox_path=str(bbox_path),
                load_mode="lazy",
                cache_max_bytes=1024 * 1024,
            )
            try:
                self.assertEqual(sources.raw_volume.shape, (3, 3, 3))
                self.assertEqual(sources.segmentation_volume.shape, (3, 3, 3))
                self.assertEqual(sources.ordered_box_ids, ("box-1",))
                self.assertIn("box-1", sources.boxes_by_id)
            finally:
                sources.close()

    def test_semantic_label_space_source_signature_uses_loader_path_and_state(self) -> None:
        volume = open_volume(
            InMemoryVolumeLoader(
                array=np.zeros((2, 2, 2), dtype=np.uint8),
                path="seg.npy",
                voxel_spacing=(1.0, 1.0, 1.0),
            )
        )

        signature = semantic_label_space_source_signature(
            semantic_kind="semantic",
            semantic_volume=volume,
            semantic_state_id=12,
        )

        self.assertEqual(signature, ("semantic", "seg.npy", 12))

    def test_prepare_learning_state_rejects_instance_segmentation_before_materializing(self) -> None:
        raw_volume = open_volume(
            InMemoryVolumeLoader(
                array=np.zeros((2, 2, 2), dtype=np.float32),
                path="raw.npy",
                voxel_spacing=(1.0, 1.0, 1.0),
            )
        )
        seg_volume = open_volume(
            InMemoryVolumeLoader(
                array=np.zeros((2, 2, 2), dtype=np.uint8),
                path="seg.npy",
                voxel_spacing=(1.0, 1.0, 1.0),
            )
        )

        with self.assertRaisesRegex(ValueError, "Only semantic segmentation"):
            prepare_learning_state_from_volumes(
                raw_volume=raw_volume,
                segmentation_volume=seg_volume,
                segmentation_kind="instance",
                boxes_by_id={},
                ordered_box_ids=(),
                require_class_weights=False,
            )


if __name__ == "__main__":
    unittest.main()
