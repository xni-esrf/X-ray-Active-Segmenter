from __future__ import annotations

import unittest

from src.ui.controllers import (
    DialogPort,
    InferenceController,
    MainWindowController,
    ModelController,
    TrainingController,
)


class UIControllerContextTests(unittest.TestCase):
    def test_controller_package_exports_base_context_types(self) -> None:
        warnings: list[tuple[str, object]] = []
        infos: list[tuple[str, object]] = []
        dialogs = DialogPort(
            show_warning=lambda message, parent=None: warnings.append((message, parent)),
            show_info=lambda message, parent=None: infos.append((message, parent)),
        )

        dialogs.show_warning("warn", None)
        dialogs.show_info("info", None)

        self.assertEqual(warnings, [("warn", None)])
        self.assertEqual(infos, [("info", None)])
        self.assertTrue(hasattr(InferenceController, "__dataclass_fields__"))
        self.assertTrue(hasattr(MainWindowController, "__dataclass_fields__"))
        self.assertTrue(hasattr(ModelController, "__dataclass_fields__"))
        self.assertTrue(hasattr(TrainingController, "__dataclass_fields__"))


if __name__ == "__main__":
    unittest.main()
