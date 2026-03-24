from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.bbox import BoundingBox

try:
    from src.ui.main_window import MainWindow
except Exception:  # pragma: no cover - environment dependent
    MainWindow = None  # type: ignore[assignment]


@unittest.skipUnless(MainWindow is not None, "MainWindow is not available")
class MainWindowBoundingBoxMultiSelectionTests(unittest.TestCase):
    def test_handle_bounding_boxes_selected_selects_single_id_in_manager(self) -> None:
        select_calls = []
        window_like = SimpleNamespace(
            _bbox_manager=SimpleNamespace(select=lambda box_id: select_calls.append(box_id)),
            _sync_bounding_boxes_ui=lambda: None,
        )

        MainWindow._handle_bounding_boxes_selected(window_like, ("bbox_0002",))

        self.assertEqual(select_calls, ["bbox_0002"])

    def test_handle_bounding_boxes_selected_clears_primary_for_multi_selection(self) -> None:
        select_calls = []
        window_like = SimpleNamespace(
            _bbox_manager=SimpleNamespace(select=lambda box_id: select_calls.append(box_id)),
            _sync_bounding_boxes_ui=lambda: None,
        )

        MainWindow._handle_bounding_boxes_selected(window_like, ("bbox_0001", "bbox_0002"))

        self.assertEqual(select_calls, [None])

    def test_handle_bounding_boxes_selected_clears_primary_for_empty_selection(self) -> None:
        select_calls = []
        window_like = SimpleNamespace(
            _bbox_manager=SimpleNamespace(select=lambda box_id: select_calls.append(box_id)),
            _sync_bounding_boxes_ui=lambda: None,
        )

        MainWindow._handle_bounding_boxes_selected(window_like, tuple())

        self.assertEqual(select_calls, [None])

    def test_handle_bounding_boxes_selected_resyncs_on_invalid_id(self) -> None:
        sync_calls = []

        def _select(_box_id: object) -> None:
            raise KeyError("unknown bbox")

        window_like = SimpleNamespace(
            _bbox_manager=SimpleNamespace(select=_select),
            _sync_bounding_boxes_ui=lambda: sync_calls.append("sync"),
        )

        MainWindow._handle_bounding_boxes_selected(window_like, ("missing_id",))

        self.assertEqual(sync_calls, ["sync"])

    def test_sync_bounding_boxes_ui_preserves_table_multi_selection_when_manager_has_no_primary(self) -> None:
        calls = []
        bottom_panel = SimpleNamespace(
            set_bounding_boxes=lambda boxes: calls.append(("boxes", tuple(boxes))),
            selected_bounding_boxes=lambda: ("bbox_0001", "bbox_0002"),
            set_selected_bounding_boxes=lambda box_ids: calls.append(("many", tuple(box_ids))),
            set_selected_bounding_box=lambda box_id: calls.append(("single", box_id)),
        )
        window_like = SimpleNamespace(
            _bbox_manager=SimpleNamespace(
                boxes=lambda: (SimpleNamespace(id="bbox_0001"), SimpleNamespace(id="bbox_0002")),
                selected_id=None,
            ),
            bottom_panel=bottom_panel,
        )

        MainWindow._sync_bounding_boxes_ui(window_like)

        self.assertEqual(calls[0][0], "boxes")
        self.assertEqual(calls[1], ("many", ("bbox_0001", "bbox_0002")))
        self.assertEqual(len(calls), 2)

    def test_sync_bounding_boxes_ui_uses_manager_single_selection_when_available(self) -> None:
        calls = []
        bottom_panel = SimpleNamespace(
            set_bounding_boxes=lambda boxes: calls.append(("boxes", tuple(boxes))),
            selected_bounding_boxes=lambda: ("bbox_0001", "bbox_0002"),
            set_selected_bounding_boxes=lambda box_ids: calls.append(("many", tuple(box_ids))),
            set_selected_bounding_box=lambda box_id: calls.append(("single", box_id)),
        )
        window_like = SimpleNamespace(
            _bbox_manager=SimpleNamespace(
                boxes=lambda: (SimpleNamespace(id="bbox_0001"),),
                selected_id="bbox_0001",
            ),
            bottom_panel=bottom_panel,
        )

        MainWindow._sync_bounding_boxes_ui(window_like)

        self.assertEqual(calls[0][0], "boxes")
        self.assertEqual(calls[1], ("single", "bbox_0001"))
        self.assertEqual(len(calls), 2)

    def test_sync_bounding_boxes_ui_falls_back_to_single_setter_without_plural_getter(self) -> None:
        calls = []
        bottom_panel = SimpleNamespace(
            set_bounding_boxes=lambda boxes: calls.append(("boxes", tuple(boxes))),
            set_selected_bounding_box=lambda box_id: calls.append(("single", box_id)),
        )
        window_like = SimpleNamespace(
            _bbox_manager=SimpleNamespace(
                boxes=lambda: tuple(),
                selected_id=None,
            ),
            bottom_panel=bottom_panel,
        )

        MainWindow._sync_bounding_boxes_ui(window_like)

        self.assertEqual(calls, [("boxes", tuple()), ("single", None)])

    def test_handle_bounding_boxes_delete_requested_deletes_all_in_one_transaction(self) -> None:
        box1 = BoundingBox.from_bounds(
            box_id="bbox_0001",
            z0=1,
            z1=4,
            y0=2,
            y1=6,
            x0=3,
            x1=8,
            volume_shape=(20, 30, 40),
        )
        box2 = BoundingBox.from_bounds(
            box_id="bbox_0002",
            z0=5,
            z1=9,
            y0=6,
            y1=12,
            x0=10,
            x1=15,
            volume_shape=(20, 30, 40),
        )
        boxes_by_id = {box1.id: box1, box2.id: box2}
        deleted_ids = []
        select_calls = []

        def _get(box_id: str):
            return boxes_by_id.get(str(box_id))

        def _delete(box_id: str) -> bool:
            normalized = str(box_id)
            if normalized not in boxes_by_id:
                return False
            deleted_ids.append(normalized)
            boxes_by_id.pop(normalized, None)
            return True

        def _select(box_id: object) -> None:
            select_calls.append(box_id)

        begin_calls = []
        pushed_commands = []
        commit_calls = []
        in_transaction_state = {"value": False}
        finalize_calls = []
        refresh_calls = []
        clear_selection_calls = []

        def _begin(name: str) -> None:
            begin_calls.append(str(name))
            in_transaction_state["value"] = True

        def _push(command: object) -> None:
            pushed_commands.append(command)

        def _commit() -> bool:
            commit_calls.append("commit")
            in_transaction_state["value"] = False
            return bool(pushed_commands)

        window_like = SimpleNamespace(
            _bbox_manager=SimpleNamespace(
                get=_get,
                delete=_delete,
                select=_select,
                selected_id=None,
            ),
            _global_history=SimpleNamespace(
                begin_transaction=_begin,
                push=_push,
                commit_transaction=_commit,
                in_transaction=lambda: bool(in_transaction_state["value"]),
            ),
            _finalize_bbox_history_transaction=lambda: finalize_calls.append("finalize"),
            _refresh_undo_ui_state=lambda: refresh_calls.append("refresh"),
            bottom_panel=SimpleNamespace(
                set_selected_bounding_boxes=lambda box_ids: clear_selection_calls.append(tuple(box_ids)),
            ),
        )

        MainWindow._handle_bounding_boxes_delete_requested(
            window_like,
            ("bbox_0001", "bbox_0002"),
        )

        self.assertEqual(finalize_calls, ["finalize"])
        self.assertEqual(begin_calls, ["bbox_delete_selected"])
        self.assertEqual(commit_calls, ["commit"])
        self.assertEqual(len(pushed_commands), 2)
        self.assertEqual(deleted_ids, ["bbox_0001", "bbox_0002"])
        self.assertEqual(clear_selection_calls, [tuple()])
        self.assertEqual(select_calls, [None])
        self.assertEqual(refresh_calls, ["refresh"])

    def test_handle_bounding_boxes_delete_requested_ignores_duplicates_and_blank_ids(self) -> None:
        box1 = BoundingBox.from_bounds(
            box_id="bbox_0001",
            z0=1,
            z1=4,
            y0=2,
            y1=6,
            x0=3,
            x1=8,
            volume_shape=(20, 30, 40),
        )
        boxes_by_id = {box1.id: box1}
        deleted_ids = []
        begin_calls = []
        push_calls = []
        commit_calls = []
        in_transaction_state = {"value": False}

        def _delete(box_id: str) -> bool:
            normalized = str(box_id)
            if normalized not in boxes_by_id:
                return False
            deleted_ids.append(normalized)
            boxes_by_id.pop(normalized, None)
            return True

        window_like = SimpleNamespace(
            _bbox_manager=SimpleNamespace(
                get=lambda box_id: boxes_by_id.get(str(box_id)),
                delete=_delete,
                select=lambda _box_id: None,
                selected_id=None,
            ),
            _global_history=SimpleNamespace(
                begin_transaction=lambda name: begin_calls.append(str(name)) or in_transaction_state.__setitem__("value", True),
                push=lambda command: push_calls.append(command),
                commit_transaction=lambda: commit_calls.append("commit") or in_transaction_state.__setitem__("value", False) or bool(push_calls),
                in_transaction=lambda: bool(in_transaction_state["value"]),
            ),
            _finalize_bbox_history_transaction=lambda: None,
            _refresh_undo_ui_state=lambda: None,
            bottom_panel=SimpleNamespace(set_selected_bounding_boxes=lambda _ids: None),
        )

        MainWindow._handle_bounding_boxes_delete_requested(
            window_like,
            ("bbox_0001", " ", "bbox_0001"),
        )

        self.assertEqual(begin_calls, ["bbox_delete_selected"])
        self.assertEqual(len(push_calls), 1)
        self.assertEqual(commit_calls, ["commit"])
        self.assertEqual(deleted_ids, ["bbox_0001"])

    def test_handle_bounding_boxes_delete_requested_is_noop_when_ids_are_blank(self) -> None:
        finalize_calls = []
        begin_calls = []
        window_like = SimpleNamespace(
            _bbox_manager=SimpleNamespace(
                get=lambda _box_id: None,
                delete=lambda _box_id: False,
                select=lambda _box_id: None,
                selected_id=None,
            ),
            _global_history=SimpleNamespace(
                begin_transaction=lambda name: begin_calls.append(str(name)),
                push=lambda _command: None,
                commit_transaction=lambda: False,
                in_transaction=lambda: False,
            ),
            _finalize_bbox_history_transaction=lambda: finalize_calls.append("finalize"),
            _refresh_undo_ui_state=lambda: None,
            bottom_panel=SimpleNamespace(set_selected_bounding_boxes=lambda _ids: None),
        )

        MainWindow._handle_bounding_boxes_delete_requested(window_like, (" ", ""))

        self.assertEqual(finalize_calls, [])
        self.assertEqual(begin_calls, [])

    def test_handle_bounding_box_delete_requested_delegates_to_plural_handler(self) -> None:
        delegated = []
        window_like = SimpleNamespace(
            _handle_bounding_boxes_delete_requested=lambda box_ids: delegated.append(tuple(box_ids)),
        )

        MainWindow._handle_bounding_box_delete_requested(window_like, "bbox_0004")
        MainWindow._handle_bounding_box_delete_requested(window_like, " ")

        self.assertEqual(delegated, [("bbox_0004",)])

    def test_handle_bounding_boxes_label_changed_updates_all_in_one_transaction(self) -> None:
        box1 = BoundingBox.from_bounds(
            box_id="bbox_0001",
            z0=1,
            z1=4,
            y0=2,
            y1=6,
            x0=3,
            x1=8,
            label="train",
            volume_shape=(20, 30, 40),
        )
        box2 = BoundingBox.from_bounds(
            box_id="bbox_0002",
            z0=5,
            z1=9,
            y0=6,
            y1=12,
            x0=10,
            x1=15,
            label="train",
            volume_shape=(20, 30, 40),
        )
        boxes_by_id = {box1.id: box1, box2.id: box2}
        replaced = []
        begin_calls = []
        pushed_commands = []
        commit_calls = []
        in_transaction_state = {"value": False}
        finalize_calls = []
        refresh_calls = []

        def _replace(box_id: str, box: BoundingBox) -> None:
            boxes_by_id[str(box_id)] = box
            replaced.append((str(box_id), box.label))

        window_like = SimpleNamespace(
            _bbox_manager=SimpleNamespace(
                get=lambda box_id: boxes_by_id.get(str(box_id)),
                replace=_replace,
                selected_id=None,
            ),
            _global_history=SimpleNamespace(
                begin_transaction=lambda name: begin_calls.append(str(name)) or in_transaction_state.__setitem__("value", True),
                push=lambda command: pushed_commands.append(command),
                commit_transaction=lambda: commit_calls.append("commit") or in_transaction_state.__setitem__("value", False) or bool(pushed_commands),
                in_transaction=lambda: bool(in_transaction_state["value"]),
            ),
            _finalize_bbox_history_transaction=lambda: finalize_calls.append("finalize"),
            _refresh_undo_ui_state=lambda: refresh_calls.append("refresh"),
            _sync_bounding_boxes_ui=lambda: None,
        )

        MainWindow._handle_bounding_boxes_label_changed(
            window_like,
            ("bbox_0001", "bbox_0002"),
            "validation",
        )

        self.assertEqual(finalize_calls, ["finalize"])
        self.assertEqual(begin_calls, ["bbox_label_selected"])
        self.assertEqual(commit_calls, ["commit"])
        self.assertEqual(len(pushed_commands), 2)
        self.assertEqual(
            replaced,
            [("bbox_0001", "validation"), ("bbox_0002", "validation")],
        )
        self.assertEqual(boxes_by_id["bbox_0001"].label, "validation")
        self.assertEqual(boxes_by_id["bbox_0002"].label, "validation")
        self.assertEqual(refresh_calls, ["refresh"])

    def test_handle_bounding_boxes_label_changed_ignores_duplicates_and_noops(self) -> None:
        box1 = BoundingBox.from_bounds(
            box_id="bbox_0001",
            z0=1,
            z1=4,
            y0=2,
            y1=6,
            x0=3,
            x1=8,
            label="validation",
            volume_shape=(20, 30, 40),
        )
        boxes_by_id = {box1.id: box1}
        replace_calls = []
        begin_calls = []
        push_calls = []
        commit_calls = []
        refresh_calls = []
        in_transaction_state = {"value": False}

        window_like = SimpleNamespace(
            _bbox_manager=SimpleNamespace(
                get=lambda box_id: boxes_by_id.get(str(box_id)),
                replace=lambda box_id, box: replace_calls.append((str(box_id), box.label)),
                selected_id=None,
            ),
            _global_history=SimpleNamespace(
                begin_transaction=lambda name: begin_calls.append(str(name)) or in_transaction_state.__setitem__("value", True),
                push=lambda command: push_calls.append(command),
                commit_transaction=lambda: commit_calls.append("commit") or in_transaction_state.__setitem__("value", False) or bool(push_calls),
                in_transaction=lambda: bool(in_transaction_state["value"]),
            ),
            _finalize_bbox_history_transaction=lambda: None,
            _refresh_undo_ui_state=lambda: refresh_calls.append("refresh"),
            _sync_bounding_boxes_ui=lambda: None,
        )

        MainWindow._handle_bounding_boxes_label_changed(
            window_like,
            ("bbox_0001", " ", "bbox_0001"),
            "validation",
        )

        self.assertEqual(begin_calls, ["bbox_label_selected"])
        self.assertEqual(commit_calls, ["commit"])
        self.assertEqual(replace_calls, [])
        self.assertEqual(push_calls, [])
        self.assertEqual(refresh_calls, [])

    def test_handle_bounding_boxes_label_changed_rejects_invalid_label_before_transaction(self) -> None:
        finalize_calls = []
        sync_calls = []
        begin_calls = []
        window_like = SimpleNamespace(
            _bbox_manager=SimpleNamespace(
                get=lambda _box_id: None,
                replace=lambda _box_id, _box: None,
                selected_id=None,
            ),
            _global_history=SimpleNamespace(
                begin_transaction=lambda name: begin_calls.append(str(name)),
                push=lambda _command: None,
                commit_transaction=lambda: False,
                in_transaction=lambda: False,
            ),
            _finalize_bbox_history_transaction=lambda: finalize_calls.append("finalize"),
            _refresh_undo_ui_state=lambda: None,
            _sync_bounding_boxes_ui=lambda: sync_calls.append("sync"),
        )

        MainWindow._handle_bounding_boxes_label_changed(
            window_like,
            ("bbox_0001",),
            "not_a_valid_label",
        )

        self.assertEqual(sync_calls, ["sync"])
        self.assertEqual(finalize_calls, [])
        self.assertEqual(begin_calls, [])

    def test_handle_bounding_box_label_changed_delegates_to_plural_handler(self) -> None:
        delegated = []
        sync_calls = []
        window_like = SimpleNamespace(
            _handle_bounding_boxes_label_changed=lambda box_ids, label: delegated.append((tuple(box_ids), str(label))),
            _sync_bounding_boxes_ui=lambda: sync_calls.append("sync"),
        )

        MainWindow._handle_bounding_box_label_changed(window_like, "bbox_0007", "train")
        MainWindow._handle_bounding_box_label_changed(window_like, " ", "train")

        self.assertEqual(delegated, [(("bbox_0007",), "train")])
        self.assertEqual(sync_calls, ["sync"])


if __name__ == "__main__":
    unittest.main()
