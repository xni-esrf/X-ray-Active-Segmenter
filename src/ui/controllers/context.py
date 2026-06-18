from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from PySide6.QtWidgets import QWidget

from ...learning import LearningSession
from ..bottom_panel import BottomPanel


DialogCallback = Callable[[str, Optional[QWidget]], object]


@dataclass(frozen=True)
class DialogPort:
    """UI dialog functions supplied by the composition root."""

    show_warning: DialogCallback
    show_info: DialogCallback


class ControllerContext(Protocol):
    """Minimal host contract shared by extracted MainWindow controllers.

    Controllers should depend on this protocol, or a narrower controller-specific
    protocol, instead of importing MainWindow.
    """

    @property
    def parent_widget(self) -> QWidget:
        """Widget used as the parent for dialogs and Qt-owned objects."""

    @property
    def bottom_panel(self) -> BottomPanel:
        """The existing bottom-panel widget while signal wiring remains in MainWindow."""

    @property
    def learning_session(self) -> LearningSession:
        """Per-window learning runtime state."""

    @property
    def dialogs(self) -> DialogPort:
        """Dialog functions available to controllers."""


@dataclass
class MainWindowController:
    """Base class for controllers extracted from MainWindow."""

    context: ControllerContext

