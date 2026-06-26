from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.render.gl_backend import GLBackend, _VispyCompatibilityLayer


class GLBackendPointerMappingTests(unittest.TestCase):
    def test_vispy_compatibility_layer_maps_logical_canvas_coordinates(self) -> None:
        captured_inputs: list[np.ndarray] = []

        class _Transform:
            def map(self, coords: np.ndarray) -> np.ndarray:
                captured_inputs.append(np.asarray(coords))
                return np.asarray(coords)

        class _ImageNode:
            def get_transform(self, *args, **kwargs):
                del args, kwargs
                return _Transform()

        mapped = _VispyCompatibilityLayer.map_canvas_to_image(
            canvas=SimpleNamespace(pixel_scale=2.0),
            image_node=_ImageNode(),
            x=11.0,
            y=17.0,
        )

        self.assertEqual(mapped, (11.0, 17.0))
        self.assertEqual(len(captured_inputs), 1)
        self.assertEqual(captured_inputs[0].shape, (1, 2))
        self.assertAlmostEqual(float(captured_inputs[0][0][0]), 11.0, places=6)
        self.assertAlmostEqual(float(captured_inputs[0][0][1]), 17.0, places=6)

    def test_map_canvas_to_image_uses_logical_canvas_coordinates_without_manual_scale(self) -> None:
        captured_inputs: list[np.ndarray] = []

        class _Transform:
            def map(self, coords: np.ndarray) -> np.ndarray:
                captured_inputs.append(np.asarray(coords))
                return np.asarray(coords)

        class _ImageNode:
            def get_transform(self, *args, **kwargs):
                del args, kwargs
                return _Transform()

        backend = GLBackend()
        backend._ready = True
        backend._canvas = SimpleNamespace(pixel_scale=2.0)
        backend._image_node = _ImageNode()

        mapped = backend.map_canvas_to_image(11.0, 17.0)

        self.assertEqual(mapped, (11.0, 17.0))
        self.assertEqual(len(captured_inputs), 1)
        self.assertEqual(captured_inputs[0].shape, (1, 2))
        self.assertAlmostEqual(float(captured_inputs[0][0][0]), 11.0, places=6)
        self.assertAlmostEqual(float(captured_inputs[0][0][1]), 17.0, places=6)

    def test_pointer_mapping_diagnostic_helpers_handle_missing_and_nonfinite_values(self) -> None:
        class _Obj:
            def finite(self) -> float:
                return 2.5

            def nonfinite(self) -> float:
                return float("inf")

            def boom(self) -> float:
                raise RuntimeError("boom")

        obj = _Obj()

        self.assertAlmostEqual(GLBackend._safe_call_float(obj, "finite") or 0.0, 2.5, places=6)
        self.assertIsNone(GLBackend._safe_call_float(obj, "nonfinite"))
        self.assertIsNone(GLBackend._safe_call_float(obj, "boom"))
        self.assertIsNone(GLBackend._safe_call_float(obj, "missing"))
        self.assertEqual(GLBackend._fmt_optional_float(None), "n/a")
        self.assertEqual(GLBackend._fmt_optional_float(3.125), "3.125")
        self.assertEqual(_VispyCompatibilityLayer.fmt_optional_float(None), "n/a")
        self.assertEqual(_VispyCompatibilityLayer.fmt_optional_float(3.125), "3.125")

if __name__ == "__main__":
    unittest.main()
