from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..annotation import EditOperation, SegmentationEditor
from ..bbox import BoundingBox, BoundingBoxManager


def estimate_segmentation_history_bytes(
    editor: SegmentationEditor,
    operation: EditOperation,
) -> int:
    changed_voxels = max(0, int(operation.changed_voxels))
    index_bytes = 8  # np.int64 flat indices
    value_bytes = int(editor.dtype.itemsize)
    return changed_voxels * (index_bytes + value_bytes)


def estimate_bounding_box_history_bytes(
    *,
    before_box: Optional[BoundingBox] = None,
    after_box: Optional[BoundingBox] = None,
) -> int:
    bytes_used = 0
    for box in (before_box, after_box):
        if box is None:
            continue
        bytes_used += (6 * 8) + len(box.id.encode("utf-8")) + len(box.label.encode("utf-8"))
    return max(0, bytes_used)


def _apply_bbox_selection(
    manager: BoundingBoxManager,
    selected_id: Optional[str],
) -> None:
    if selected_id is None:
        manager.select(None)
        return
    if manager.get(selected_id) is None:
        raise RuntimeError(f"Cannot select missing bounding box id: {selected_id}")
    manager.select(selected_id)


@dataclass(frozen=True)
class SegmentationHistoryCommand:
    editor: SegmentationEditor
    operation_id: int
    bytes_used: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", int(self.operation_id))
        object.__setattr__(self, "bytes_used", max(0, int(self.bytes_used)))

    def undo(self) -> None:
        expected_operation_id = int(self.operation_id)
        available_operation_id = self.editor.latest_undo_operation_id()
        if available_operation_id != expected_operation_id:
            raise RuntimeError(
                "Segmentation undo operation is out of sync with global history."
            )
        operation = self.editor.undo_last_modification()
        if operation is None:
            raise RuntimeError("No segmentation operation available to undo.")
        self._validate_operation(operation.operation_id, expected_operation_id, phase="undo")

    def redo(self) -> None:
        expected_operation_id = int(self.operation_id)
        available_operation_id = self.editor.latest_redo_operation_id()
        if available_operation_id != expected_operation_id:
            raise RuntimeError(
                "Segmentation redo operation is out of sync with global history."
            )
        operation = self.editor.redo_last_modification()
        if operation is None:
            raise RuntimeError("No segmentation operation available to redo.")
        self._validate_operation(operation.operation_id, expected_operation_id, phase="redo")

    @staticmethod
    def _validate_operation(
        observed_operation_id: Optional[int],
        expected_operation_id: int,
        *,
        phase: str,
    ) -> None:
        if observed_operation_id != expected_operation_id:
            raise RuntimeError(
                f"Segmentation {phase} operation id mismatch: "
                f"expected {expected_operation_id}, got {observed_operation_id}."
            )

    def on_discard(self, from_stack: str) -> None:
        operation_id = int(self.operation_id)
        if from_stack == "undo":
            self.editor.discard_undo_operation(operation_id)
            return
        if from_stack == "redo":
            self.editor.discard_redo_operation(operation_id)


@dataclass(frozen=True)
class BoundingBoxAddCommand:
    manager: BoundingBoxManager
    box: BoundingBox
    before_selected_id: Optional[str]
    after_selected_id: Optional[str]
    bytes_used: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "bytes_used", max(0, int(self.bytes_used)))

    def undo(self) -> None:
        removed = self.manager.delete(self.box.id)
        if not removed:
            raise RuntimeError(f"Bounding box {self.box.id} is missing for undo.")
        _apply_bbox_selection(self.manager, self.before_selected_id)

    def redo(self) -> None:
        if self.manager.get(self.box.id) is not None:
            raise RuntimeError(f"Bounding box {self.box.id} already exists for redo.")
        self.manager.add(self.box, select=False)
        _apply_bbox_selection(self.manager, self.after_selected_id)


@dataclass(frozen=True)
class BoundingBoxDeleteCommand:
    manager: BoundingBoxManager
    box: BoundingBox
    before_selected_id: Optional[str]
    after_selected_id: Optional[str]
    bytes_used: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "bytes_used", max(0, int(self.bytes_used)))

    def undo(self) -> None:
        if self.manager.get(self.box.id) is not None:
            raise RuntimeError(f"Bounding box {self.box.id} already exists for undo.")
        self.manager.add(self.box, select=False)
        _apply_bbox_selection(self.manager, self.before_selected_id)

    def redo(self) -> None:
        removed = self.manager.delete(self.box.id)
        if not removed:
            raise RuntimeError(f"Bounding box {self.box.id} is missing for redo.")
        _apply_bbox_selection(self.manager, self.after_selected_id)


@dataclass(frozen=True)
class BoundingBoxUpdateCommand:
    manager: BoundingBoxManager
    before_box: BoundingBox
    after_box: BoundingBox
    before_selected_id: Optional[str]
    after_selected_id: Optional[str]
    bytes_used: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "bytes_used", max(0, int(self.bytes_used)))

    def undo(self) -> None:
        existing = self.manager.get(self.before_box.id)
        if existing is None:
            raise RuntimeError(
                f"Bounding box {self.before_box.id} is missing for undo."
            )
        self.manager.replace(self.before_box.id, self.before_box)
        _apply_bbox_selection(self.manager, self.before_selected_id)

    def redo(self) -> None:
        existing = self.manager.get(self.after_box.id)
        if existing is None:
            raise RuntimeError(
                f"Bounding box {self.after_box.id} is missing for redo."
            )
        self.manager.replace(self.after_box.id, self.after_box)
        _apply_bbox_selection(self.manager, self.after_selected_id)
