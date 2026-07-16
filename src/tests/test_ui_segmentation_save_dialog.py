from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication

    from src.ui.dialogs import open_save_segmentation_dialog
except Exception:  # pragma: no cover - environment dependent
    QApplication = None  # type: ignore[assignment]
    open_save_segmentation_dialog = None  # type: ignore[assignment]


@unittest.skipUnless(
    QApplication is not None and open_save_segmentation_dialog is not None,
    "Dialogs are not available",
)
class SegmentationSaveDialogFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_default_filters_offer_all_five_formats(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            selected_path = str(Path(temp_dir) / "output.zarr")
            with patch(
                "src.ui.dialogs.QFileDialog.getSaveFileName",
                return_value=(selected_path, "Zarr (*.zarr)"),
            ) as save_dialog_mock:
                result = open_save_segmentation_dialog()

        self.assertTrue(result.accepted)
        filters = save_dialog_mock.call_args.args[3]
        for expected in (
            "TIFF (*.tif *.tiff)",
            "NumPy (*.npy)",
            "NumPy Compressed (*.npz)",
            "HDF5 (*.h5 *.hdf5 *.hdf)",
            "Zarr (*.zarr)",
        ):
            self.assertIn(expected, filters)

    def test_allowed_formats_restricts_filters_to_zarr_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            selected_path = str(Path(temp_dir) / "output.zarr")
            with patch(
                "src.ui.dialogs.QFileDialog.getSaveFileName",
                return_value=(selected_path, "Zarr (*.zarr)"),
            ) as save_dialog_mock:
                result = open_save_segmentation_dialog(allowed_formats=("zarr",))

        self.assertTrue(result.accepted)
        self.assertEqual(result.format, "zarr")
        filters = save_dialog_mock.call_args.args[3]
        self.assertEqual(filters, "Zarr (*.zarr)")
        for unexpected in ("TIFF", "NumPy", "HDF5"):
            self.assertNotIn(unexpected, filters)

    def test_allowed_formats_with_no_match_raises(self) -> None:
        with self.assertRaises(ValueError):
            open_save_segmentation_dialog(allowed_formats=("not-a-format",))


if __name__ == "__main__":
    unittest.main()
