from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import List, Literal, Optional, Protocol, Sequence, Tuple, runtime_checkable


_MAX_HISTORY_ENTRIES = 10
_DEFAULT_MAX_HISTORY_BYTES = 5 * 1024 * 1024 * 1024
HistoryStackName = Literal["undo", "redo"]


@runtime_checkable
class HistoryCommand(Protocol):
    @property
    def bytes_used(self) -> int:
        """Approximate memory retained by this command for undo/redo."""

    def undo(self) -> None:
        """Revert this command."""

    def redo(self) -> None:
        """Reapply this command."""


class CommandGroup:
    """A composite command executed as a single history entry."""

    def __init__(self, name: str, commands: Sequence[HistoryCommand]) -> None:
        normalized_name = str(name).strip() or "transaction"
        normalized_commands = tuple(commands)
        if not normalized_commands:
            raise ValueError("CommandGroup requires at least one command")
        self.name = normalized_name
        self._commands = normalized_commands
        self._bytes_used = sum(_coerce_command_bytes(command) for command in self._commands)

    @property
    def bytes_used(self) -> int:
        return self._bytes_used

    @property
    def commands(self) -> Tuple[HistoryCommand, ...]:
        return self._commands

    def undo(self) -> None:
        for command in reversed(self._commands):
            command.undo()

    def redo(self) -> None:
        for command in self._commands:
            command.redo()

    def on_discard(self, from_stack: HistoryStackName) -> None:
        for command in self._commands:
            callback = getattr(command, "on_discard", None)
            if callable(callback):
                callback(from_stack)


@dataclass
class _PendingTransaction:
    name: str
    commands: List[HistoryCommand]


def _coerce_command_bytes(command: HistoryCommand) -> int:
    raw = getattr(command, "bytes_used", 0)
    if isinstance(raw, bool) or not isinstance(raw, Integral):
        raise TypeError(
            f"History command bytes_used must be an integer, got {type(raw).__name__}"
        )
    return max(0, int(raw))


class GlobalHistoryManager:
    """Global undo/redo stack independent from specific editing tools."""

    def __init__(
        self,
        *,
        max_history_entries: int = _MAX_HISTORY_ENTRIES,
        max_history_bytes: int = _DEFAULT_MAX_HISTORY_BYTES,
    ) -> None:
        self._max_history_entries = max(
            0,
            min(_MAX_HISTORY_ENTRIES, int(max_history_entries)),
        )
        self._max_history_bytes = max(0, int(max_history_bytes))
        self._undo_stack: List[HistoryCommand] = []
        self._redo_stack: List[HistoryCommand] = []
        self._undo_total_bytes = 0
        self._redo_total_bytes = 0
        self._pending_transaction: Optional[_PendingTransaction] = None

    def undo_depth(self) -> int:
        return len(self._undo_stack)

    def redo_depth(self) -> int:
        return len(self._redo_stack)

    def undo_total_bytes(self) -> int:
        return self._undo_total_bytes

    def redo_total_bytes(self) -> int:
        return self._redo_total_bytes

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def in_transaction(self) -> bool:
        return self._pending_transaction is not None

    def clear(self) -> None:
        self._discard_commands(self._undo_stack, from_stack="undo")
        self._discard_commands(self._redo_stack, from_stack="redo")
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._undo_total_bytes = 0
        self._redo_total_bytes = 0
        self._pending_transaction = None

    def begin_transaction(self, name: str = "transaction") -> None:
        if self._pending_transaction is not None:
            raise RuntimeError("A history transaction is already active")
        normalized_name = str(name).strip() or "transaction"
        self._pending_transaction = _PendingTransaction(
            name=normalized_name,
            commands=[],
        )

    def commit_transaction(self) -> bool:
        pending = self._pending_transaction
        if pending is None:
            return False
        self._pending_transaction = None
        if not pending.commands:
            return False
        if len(pending.commands) == 1:
            return self.push(pending.commands[0])
        return self.push(CommandGroup(pending.name, pending.commands))

    def cancel_transaction(self) -> int:
        pending = self._pending_transaction
        if pending is None:
            return 0
        count = len(pending.commands)
        self._discard_commands(pending.commands, from_stack="undo")
        self._pending_transaction = None
        return count

    def push(self, command: HistoryCommand) -> bool:
        if not isinstance(command, HistoryCommand):
            raise TypeError("command must implement HistoryCommand")
        if self._pending_transaction is not None:
            self._pending_transaction.commands.append(command)
            return True
        return self._push_immediate(command)

    def undo(self) -> Optional[HistoryCommand]:
        if self._pending_transaction is not None:
            raise RuntimeError("Cannot undo while a history transaction is active")
        if not self._undo_stack:
            return None
        command = self._undo_stack.pop()
        command_bytes = _coerce_command_bytes(command)
        self._undo_total_bytes = max(0, self._undo_total_bytes - command_bytes)
        try:
            command.undo()
        except Exception:
            self._undo_stack.append(command)
            self._undo_total_bytes += command_bytes
            raise
        self._redo_stack.append(command)
        self._redo_total_bytes += command_bytes
        return command

    def redo(self) -> Optional[HistoryCommand]:
        if self._pending_transaction is not None:
            raise RuntimeError("Cannot redo while a history transaction is active")
        if not self._redo_stack:
            return None
        command = self._redo_stack.pop()
        command_bytes = _coerce_command_bytes(command)
        self._redo_total_bytes = max(0, self._redo_total_bytes - command_bytes)
        try:
            command.redo()
        except Exception:
            self._redo_stack.append(command)
            self._redo_total_bytes += command_bytes
            raise
        self._undo_stack.append(command)
        self._undo_total_bytes += command_bytes
        self._trim_undo_stack()
        return command

    def _push_immediate(self, command: HistoryCommand) -> bool:
        self._clear_redo_stack()
        if self._max_history_entries <= 0 or self._max_history_bytes <= 0:
            self._discard_commands(self._undo_stack, from_stack="undo")
            self._undo_stack.clear()
            self._undo_total_bytes = 0
            self._notify_command_discarded(command, from_stack="undo")
            return False
        command_bytes = _coerce_command_bytes(command)
        if command_bytes > self._max_history_bytes:
            self._notify_command_discarded(command, from_stack="undo")
            return False
        self._undo_stack.append(command)
        self._undo_total_bytes += command_bytes
        self._trim_undo_stack()
        return True

    def _trim_undo_stack(self) -> None:
        while self._undo_stack and (
            len(self._undo_stack) > self._max_history_entries
            or self._undo_total_bytes > self._max_history_bytes
        ):
            dropped = self._undo_stack.pop(0)
            dropped_bytes = _coerce_command_bytes(dropped)
            self._undo_total_bytes = max(0, self._undo_total_bytes - dropped_bytes)
            self._notify_command_discarded(dropped, from_stack="undo")

    def _clear_redo_stack(self) -> None:
        self._discard_commands(self._redo_stack, from_stack="redo")
        self._redo_stack.clear()
        self._redo_total_bytes = 0

    def _discard_commands(
        self,
        commands: Sequence[HistoryCommand],
        *,
        from_stack: HistoryStackName,
    ) -> None:
        for command in tuple(commands):
            self._notify_command_discarded(command, from_stack=from_stack)

    def _notify_command_discarded(
        self,
        command: HistoryCommand,
        *,
        from_stack: HistoryStackName,
    ) -> None:
        callback = getattr(command, "on_discard", None)
        if callable(callback):
            callback(from_stack)
