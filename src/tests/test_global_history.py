from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import List

from src.history import GlobalHistoryManager


@dataclass
class _StubCommand:
    command_id: str
    log: List[str] = field(default_factory=list)
    bytes_used: int = 1
    undo_calls: int = 0
    redo_calls: int = 0

    def undo(self) -> None:
        self.undo_calls += 1
        self.log.append(f"undo:{self.command_id}")

    def redo(self) -> None:
        self.redo_calls += 1
        self.log.append(f"redo:{self.command_id}")


class GlobalHistoryManagerTests(unittest.TestCase):
    def test_push_and_undo_redo_depths(self) -> None:
        history = GlobalHistoryManager(max_history_entries=10, max_history_bytes=1024)
        first = _StubCommand("first")
        second = _StubCommand("second")

        self.assertTrue(history.push(first))
        self.assertTrue(history.push(second))
        self.assertEqual(history.undo_depth(), 2)
        self.assertEqual(history.redo_depth(), 0)

        undone = history.undo()
        self.assertIs(undone, second)
        self.assertEqual(history.undo_depth(), 1)
        self.assertEqual(history.redo_depth(), 1)

        redone = history.redo()
        self.assertIs(redone, second)
        self.assertEqual(history.undo_depth(), 2)
        self.assertEqual(history.redo_depth(), 0)

    def test_push_clears_redo_history(self) -> None:
        history = GlobalHistoryManager(max_history_entries=10, max_history_bytes=1024)
        first = _StubCommand("first")
        second = _StubCommand("second")
        replacement = _StubCommand("replacement")

        history.push(first)
        history.push(second)
        history.undo()
        self.assertEqual(history.redo_depth(), 1)

        history.push(replacement)
        self.assertEqual(history.redo_depth(), 0)
        self.assertEqual(history.undo_depth(), 2)

    def test_undo_and_redo_call_commands_in_order(self) -> None:
        log: List[str] = []
        history = GlobalHistoryManager(max_history_entries=10, max_history_bytes=1024)
        first = _StubCommand("first", log=log)
        second = _StubCommand("second", log=log)

        history.push(first)
        history.push(second)

        history.undo()
        history.undo()
        history.redo()
        history.redo()

        self.assertEqual(
            log,
            [
                "undo:second",
                "undo:first",
                "redo:first",
                "redo:second",
            ],
        )

    def test_entry_limit_drops_oldest_commands(self) -> None:
        log: List[str] = []
        history = GlobalHistoryManager(max_history_entries=2, max_history_bytes=1024)
        first = _StubCommand("first", log=log)
        second = _StubCommand("second", log=log)
        third = _StubCommand("third", log=log)

        history.push(first)
        history.push(second)
        history.push(third)
        self.assertEqual(history.undo_depth(), 2)

        history.undo()
        history.undo()
        self.assertEqual(log, ["undo:third", "undo:second"])

    def test_byte_limit_drops_oldest_commands(self) -> None:
        log: List[str] = []
        history = GlobalHistoryManager(max_history_entries=10, max_history_bytes=5)
        first = _StubCommand("first", log=log, bytes_used=3)
        second = _StubCommand("second", log=log, bytes_used=3)

        history.push(first)
        history.push(second)
        self.assertEqual(history.undo_depth(), 1)
        self.assertEqual(history.undo_total_bytes(), 3)

        history.undo()
        self.assertEqual(log, ["undo:second"])

    def test_oversized_command_is_not_stored(self) -> None:
        history = GlobalHistoryManager(max_history_entries=10, max_history_bytes=5)
        first = _StubCommand("first", bytes_used=3)
        oversized = _StubCommand("oversized", bytes_used=8)

        history.push(first)
        self.assertFalse(history.push(oversized))
        self.assertEqual(history.undo_depth(), 1)
        self.assertEqual(history.undo_total_bytes(), 3)

    def test_transaction_commits_as_single_undo_entry(self) -> None:
        log: List[str] = []
        history = GlobalHistoryManager(max_history_entries=10, max_history_bytes=1024)
        first = _StubCommand("first", log=log)
        second = _StubCommand("second", log=log)

        history.begin_transaction("bbox_drag")
        history.push(first)
        history.push(second)
        self.assertEqual(history.undo_depth(), 0)

        stored = history.commit_transaction()
        self.assertTrue(stored)
        self.assertEqual(history.undo_depth(), 1)

        history.undo()
        history.redo()
        self.assertEqual(
            log,
            [
                "undo:second",
                "undo:first",
                "redo:first",
                "redo:second",
            ],
        )

    def test_cancel_transaction_discards_staged_commands(self) -> None:
        history = GlobalHistoryManager(max_history_entries=10, max_history_bytes=1024)
        history.begin_transaction("stroke")
        history.push(_StubCommand("first"))
        history.push(_StubCommand("second"))

        discarded = history.cancel_transaction()
        self.assertEqual(discarded, 2)
        self.assertEqual(history.undo_depth(), 0)
        self.assertFalse(history.in_transaction())

    def test_discard_hook_runs_when_trimming_undo_or_clearing_redo(self) -> None:
        class _DiscardAwareCommand(_StubCommand):
            def __init__(self, command_id: str, log: List[str]) -> None:
                super().__init__(command_id=command_id, log=log, bytes_used=1)
                self.discard_log = log

            def on_discard(self, from_stack: str) -> None:
                self.discard_log.append(f"discard:{from_stack}:{self.command_id}")

        log: List[str] = []
        history = GlobalHistoryManager(max_history_entries=1, max_history_bytes=1024)
        first = _DiscardAwareCommand("first", log=log)
        second = _DiscardAwareCommand("second", log=log)

        history.push(first)
        history.push(second)  # trims first from undo
        self.assertIn("discard:undo:first", log)

        history.undo()  # second now on redo stack
        history.push(_DiscardAwareCommand("third", log=log))  # clears redo
        self.assertIn("discard:redo:second", log)


if __name__ == "__main__":
    unittest.main()
