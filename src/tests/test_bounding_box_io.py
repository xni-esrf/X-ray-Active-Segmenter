from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.bbox import (
    BoundingBox,
    BoundingBoxFileData,
    FORMAT_VERSION,
    load_bounding_boxes,
    parse_bounding_boxes_text,
    save_bounding_boxes,
    serialize_bounding_boxes,
)


class BoundingBoxIOTests(unittest.TestCase):
    def test_serialize_and_parse_roundtrip(self) -> None:
        boxes = (
            BoundingBox.from_bounds(
                box_id="bbox_0001",
                z0=1,
                z1=4,
                y0=2,
                y1=7,
                x0=3,
                x1=8,
                label="train",
                volume_shape=(20, 30, 40),
            ),
            BoundingBox.from_bounds(
                box_id="custom_2",
                z0=5,
                z1=9,
                y0=6,
                y1=12,
                x0=10,
                x1=14,
                label="validation",
                volume_shape=(20, 30, 40),
            ),
        )
        data = BoundingBoxFileData(
            version=FORMAT_VERSION,
            volume_shape=(20, 30, 40),
            boxes=boxes,
        )

        text = serialize_bounding_boxes(data)
        parsed = parse_bounding_boxes_text(text, expected_shape=(20, 30, 40))

        self.assertEqual(parsed.version, FORMAT_VERSION)
        self.assertEqual(parsed.volume_shape, (20, 30, 40))
        self.assertEqual(parsed.boxes, boxes)

    def test_parse_v1_box_without_label_defaults_to_train(self) -> None:
        text = """# segmentation_tool_bboxes v1
shape 10 10 10
box legacy 1 2 3 4 5 6
"""
        parsed = parse_bounding_boxes_text(text)
        self.assertEqual(parsed.version, 1)
        self.assertEqual(len(parsed.boxes), 1)
        self.assertEqual(parsed.boxes[0].label, "train")

    def test_parse_accepts_explicit_label(self) -> None:
        text = """# segmentation_tool_bboxes v2
shape 10 10 10
box sample 1 2 3 4 5 6 inference
"""
        parsed = parse_bounding_boxes_text(text)
        self.assertEqual(parsed.version, FORMAT_VERSION)
        self.assertEqual(len(parsed.boxes), 1)
        self.assertEqual(parsed.boxes[0].label, "inference")

    def test_parse_ignores_comments_and_blank_lines(self) -> None:
        text = """
# preamble comment
# segmentation_tool_bboxes v1

shape 8 9 10
# another comment
box a 1 3 2 4 5 7
"""
        parsed = parse_bounding_boxes_text(text)
        self.assertEqual(parsed.volume_shape, (8, 9, 10))
        self.assertEqual(len(parsed.boxes), 1)
        self.assertEqual(parsed.boxes[0].id, "a")

    def test_parse_rejects_missing_header(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing bounding-box header"):
            parse_bounding_boxes_text("shape 10 10 10\nbox a 0 1 0 1 0 1\n")

    def test_parse_rejects_duplicate_ids(self) -> None:
        text = """# segmentation_tool_bboxes v1
shape 10 10 10
box a 0 2 0 2 0 2
box a 3 5 3 5 3 5
"""
        with self.assertRaisesRegex(ValueError, "Duplicate bounding box id"):
            parse_bounding_boxes_text(text)

    def test_parse_rejects_shape_mismatch(self) -> None:
        text = """# segmentation_tool_bboxes v1
shape 9 9 9
box a 0 2 0 2 0 2
"""
        with self.assertRaisesRegex(ValueError, "shape does not match expected shape"):
            parse_bounding_boxes_text(text, expected_shape=(10, 10, 10))

    def test_save_and_load_refuse_overwrite_without_flag(self) -> None:
        boxes = (
            BoundingBox.from_bounds(
                box_id="box_a",
                z0=1,
                z1=3,
                y0=1,
                y1=3,
                x0=1,
                x1=3,
                volume_shape=(10, 10, 10),
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "boxes.txt"
            saved_path = save_bounding_boxes(
                str(path),
                volume_shape=(10, 10, 10),
                boxes=boxes,
            )
            self.assertEqual(saved_path, str(path))

            loaded = load_bounding_boxes(str(path), expected_shape=(10, 10, 10))
            self.assertEqual(loaded.boxes, boxes)

            with self.assertRaises(FileExistsError):
                save_bounding_boxes(
                    str(path),
                    volume_shape=(10, 10, 10),
                    boxes=boxes,
                    overwrite=False,
                )
            save_bounding_boxes(
                str(path),
                volume_shape=(10, 10, 10),
                boxes=boxes,
                overwrite=True,
            )


if __name__ == "__main__":
    unittest.main()
