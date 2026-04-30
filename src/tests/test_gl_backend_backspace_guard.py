from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.render.gl_backend import GLBackend


class GLBackendBackspaceGuardTests(unittest.TestCase):
    def test_disable_camera_backspace_reset_blocks_backspace_and_marks_handled(self) -> None:
        forwarded_keys: list[object] = []

        class _Camera:
            def viewbox_key_event(self, event) -> None:
                forwarded_keys.append(getattr(event, "key", None))

        camera = _Camera()
        GLBackend._disable_camera_backspace_reset(camera)

        backspace_event = SimpleNamespace(key="Backspace", handled=False)
        camera.viewbox_key_event(backspace_event)
        self.assertEqual(forwarded_keys, [])
        self.assertTrue(backspace_event.handled)

        non_backspace_event = SimpleNamespace(key="A", handled=False)
        camera.viewbox_key_event(non_backspace_event)
        self.assertEqual(forwarded_keys, ["A"])
        self.assertFalse(non_backspace_event.handled)

    def test_disable_camera_backspace_reset_is_idempotent(self) -> None:
        class _Camera:
            def viewbox_key_event(self, event) -> None:
                return None

        camera = _Camera()
        GLBackend._disable_camera_backspace_reset(camera)
        wrapped_once = camera.viewbox_key_event
        GLBackend._disable_camera_backspace_reset(camera)
        wrapped_twice = camera.viewbox_key_event

        self.assertIs(wrapped_once, wrapped_twice)


if __name__ == "__main__":
    unittest.main()
