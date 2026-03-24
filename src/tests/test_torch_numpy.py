from __future__ import annotations

import unittest

import numpy as np

from src.utils.torch_numpy import ensure_native_endian_array, torch_from_numpy_safe


class TorchNumpyInteropTests(unittest.TestCase):
    def test_ensure_native_endian_array_returns_native_array_unchanged(self) -> None:
        native = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        normalized = ensure_native_endian_array(native)
        self.assertIs(normalized, native)
        self.assertTrue(normalized.dtype.isnative)

    def test_ensure_native_endian_array_converts_big_endian_and_preserves_values(self) -> None:
        big_endian = np.array([1, 256, 1025], dtype=np.dtype(">u2"))
        normalized = ensure_native_endian_array(big_endian)

        self.assertFalse(big_endian.dtype.isnative)
        self.assertTrue(normalized.dtype.isnative)
        np.testing.assert_array_equal(
            normalized,
            np.array([1, 256, 1025], dtype=np.uint16),
        )

    def test_torch_from_numpy_safe_normalizes_before_handoff(self) -> None:
        captured: dict[str, object] = {}

        class _FakeTorch:
            @staticmethod
            def from_numpy(array: np.ndarray) -> object:
                captured["dtype"] = array.dtype
                captured["values"] = np.asarray(array).copy()
                return ("fake_tensor", array.shape)

        source = np.array([10, 20, 30], dtype=np.dtype(">i2"))
        result = torch_from_numpy_safe(source, torch_module=_FakeTorch)

        self.assertEqual(result, ("fake_tensor", (3,)))
        self.assertIn("dtype", captured)
        self.assertIn("values", captured)
        self.assertTrue(np.dtype(captured["dtype"]).isnative)
        np.testing.assert_array_equal(
            np.asarray(captured["values"]),
            np.array([10, 20, 30], dtype=np.int16),
        )

    def test_torch_from_numpy_safe_rejects_missing_from_numpy(self) -> None:
        with self.assertRaisesRegex(TypeError, "must define from_numpy"):
            torch_from_numpy_safe(np.array([1], dtype=np.uint8), torch_module=object())


if __name__ == "__main__":
    unittest.main()
