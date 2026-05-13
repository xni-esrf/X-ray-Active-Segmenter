from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import cast

import numpy as np

from src.render import RenderResult
from src.ui.orthogonal_view import OrthogonalView
from src.ui.orthogonal_view import next_zoom_from_action


class OrthogonalViewZoomActionTests(unittest.TestCase):
    def test_zoom_in_decreases_zoom_by_step(self) -> None:
        self.assertAlmostEqual(next_zoom_from_action(0.8, "zoom_in"), 0.7)

    def test_zoom_out_increases_zoom_by_step(self) -> None:
        self.assertAlmostEqual(next_zoom_from_action(0.5, "zoom_out"), 0.6)

    def test_zoom_action_step_is_not_clamped(self) -> None:
        self.assertAlmostEqual(next_zoom_from_action(0.1, "zoom_in"), 0.0)
        self.assertAlmostEqual(next_zoom_from_action(1.0, "zoom_out"), 1.1)

    def test_zoom_action_rejects_unknown_action(self) -> None:
        with self.assertRaises(ValueError):
            next_zoom_from_action(0.5, cast(object, "bad_action"))

    def test_handle_render_result_recenters_when_level_changes_without_zoom_change(self) -> None:
        zoom_calls: list[bool] = []
        uploaded_levels: list[int] = []
        image = np.zeros((8, 8), dtype=np.float32)
        previous = RenderResult(
            view_id="axial",
            axis=0,
            slice_index=4,
            image=image,
            level=0,
            level_scale=1,
        )
        incoming = RenderResult(
            view_id="axial",
            axis=0,
            slice_index=4,
            image=image,
            level=2,
            level_scale=4,
        )
        view_like = SimpleNamespace(
            view_id="axial",
            _latest=previous,
            _pending_overlay_result=None,
            _overlay_flush_scheduled=False,
            _recenter_on_next_render=False,
            _gl_backend=SimpleNamespace(
                upload_texture=lambda *_args, **_kwargs: uploaded_levels.append(incoming.level),
            ),
            _is_overlay_only_refresh=lambda _previous, _result: False,
            _update_crosshair=lambda _image: None,
            _update_picker_marker=lambda _image: None,
            _update_bounding_boxes_overlay=lambda _result: None,
            _apply_pan=lambda _image: None,
            _apply_zoom=lambda _image, *, recenter: zoom_calls.append(bool(recenter)),
        )

        OrthogonalView._handle_render_result(view_like, incoming)

        self.assertEqual(uploaded_levels, [2])
        self.assertEqual(zoom_calls, [True])
        self.assertIs(view_like._latest, incoming)
        self.assertFalse(view_like._recenter_on_next_render)


if __name__ == "__main__":
    unittest.main()
