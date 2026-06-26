from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import (
        QApplication,
        QAbstractItemView,
        QGroupBox,
        QSizePolicy,
    )
except Exception:  # pragma: no cover - environment dependent
    QApplication = None  # type: ignore[assignment]
    QAbstractItemView = None  # type: ignore[assignment]
    QGroupBox = None  # type: ignore[assignment]
    QSizePolicy = None  # type: ignore[assignment]

from src.bbox import BoundingBox

try:
    from src.ui.bottom_panel import BottomPanel
except Exception:  # pragma: no cover - environment dependent
    BottomPanel = None  # type: ignore[assignment]


def make_boxes() -> tuple[BoundingBox, BoundingBox]:
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
    return (box1, box2)


@unittest.skipUnless(
    QApplication is not None and BottomPanel is not None,
    "PySide6 is not available",
)
class BottomPanelTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.panel = BottomPanel()

    def _boxes(self) -> tuple[BoundingBox, BoundingBox]:
        return make_boxes()
