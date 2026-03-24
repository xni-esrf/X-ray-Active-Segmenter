from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from src.ui.orthogonal_view import OrthogonalView
except Exception:  # pragma: no cover - environment dependent
    OrthogonalView = None  # type: ignore[assignment]


@unittest.skipUnless(OrthogonalView is not None, "OrthogonalView is not available")
class OrthogonalViewShiftClickTests(unittest.TestCase):
    def test_canvas_press_centers_all_views_once_when_cursor_update_succeeds(self) -> None:
        update_calls: list[object] = []
        center_calls: list[str] = []
        view_like = SimpleNamespace(
            _update_cursor_from_position=lambda position: update_calls.append(position) or True,
            _on_center_all_views_requested=lambda: center_calls.append("center"),
        )

        OrthogonalView._handle_canvas_press(view_like, position=object())

        self.assertEqual(len(update_calls), 1)
        self.assertEqual(center_calls, ["center"])

    def test_canvas_press_does_not_center_when_cursor_update_fails(self) -> None:
        center_calls: list[str] = []
        view_like = SimpleNamespace(
            _update_cursor_from_position=lambda _position: False,
            _on_center_all_views_requested=lambda: center_calls.append("center"),
        )

        OrthogonalView._handle_canvas_press(view_like, position=object())

        self.assertEqual(center_calls, [])

    def test_canvas_move_is_noop_for_shift_right_path(self) -> None:
        update_calls: list[object] = []
        view_like = SimpleNamespace(
            _update_cursor_from_position=lambda position: update_calls.append(position) or True,
        )

        OrthogonalView._handle_canvas_move(view_like, position=object())

        self.assertEqual(update_calls, [])

    def test_right_button_mode_stays_center_when_shift_released_after_press(self) -> None:
        calls: list[str] = []
        view_like = SimpleNamespace(
            _right_button_mode=None,
            _handle_canvas_press=lambda _position: calls.append("canvas_press"),
            _handle_pan_press=lambda _position: calls.append("pan_press"),
            _handle_canvas_move=lambda _position: calls.append("canvas_move"),
            _handle_pan_move=lambda _position: calls.append("pan_move"),
            _handle_canvas_release=lambda: calls.append("canvas_release"),
            _handle_pan_release=lambda: calls.append("pan_release"),
        )

        OrthogonalView._handle_right_button_press(view_like, position=object(), shift_pressed=True)
        OrthogonalView._handle_right_button_move(view_like, position=object(), shift_pressed=False)
        OrthogonalView._handle_right_button_release(view_like, shift_pressed=False)

        self.assertEqual(calls, ["canvas_press", "canvas_move", "canvas_release"])
        self.assertIsNone(view_like._right_button_mode)

    def test_right_button_mode_stays_pan_when_shift_pressed_after_press(self) -> None:
        calls: list[str] = []
        view_like = SimpleNamespace(
            _right_button_mode=None,
            _handle_canvas_press=lambda _position: calls.append("canvas_press"),
            _handle_pan_press=lambda _position: calls.append("pan_press"),
            _handle_canvas_move=lambda _position: calls.append("canvas_move"),
            _handle_pan_move=lambda _position: calls.append("pan_move"),
            _handle_canvas_release=lambda: calls.append("canvas_release"),
            _handle_pan_release=lambda: calls.append("pan_release"),
        )

        OrthogonalView._handle_right_button_press(view_like, position=object(), shift_pressed=False)
        OrthogonalView._handle_right_button_move(view_like, position=object(), shift_pressed=True)
        OrthogonalView._handle_right_button_release(view_like, shift_pressed=True)

        self.assertEqual(calls, ["pan_press", "pan_move", "pan_release"])
        self.assertIsNone(view_like._right_button_mode)

    def test_right_button_move_falls_back_to_current_modifier_without_press_mode(self) -> None:
        calls: list[str] = []
        view_like = SimpleNamespace(
            _right_button_mode=None,
            _handle_canvas_move=lambda _position: calls.append("canvas_move"),
            _handle_pan_move=lambda _position: calls.append("pan_move"),
        )

        OrthogonalView._handle_right_button_move(view_like, position=object(), shift_pressed=True)
        OrthogonalView._handle_right_button_move(view_like, position=object(), shift_pressed=False)

        self.assertEqual(calls, ["canvas_move", "pan_move"])

    def test_right_button_release_falls_back_to_current_modifier_without_press_mode(self) -> None:
        calls: list[str] = []
        view_like = SimpleNamespace(
            _right_button_mode=None,
            _handle_canvas_release=lambda: calls.append("canvas_release"),
            _handle_pan_release=lambda: calls.append("pan_release"),
        )

        OrthogonalView._handle_right_button_release(view_like, shift_pressed=True)
        OrthogonalView._handle_right_button_release(view_like, shift_pressed=False)

        self.assertEqual(calls, ["canvas_release", "pan_release"])
        self.assertIsNone(view_like._right_button_mode)

    def test_update_cursor_from_position_returns_true_and_emits_drag_cursor(self) -> None:
        cursor_calls: list[tuple[int, int, int]] = []
        view_like = SimpleNamespace(
            _indices_from_position=lambda _position: (1, 2, 3),
            input_handlers=SimpleNamespace(on_drag_cursor=lambda indices: cursor_calls.append(tuple(indices))),
        )

        result = OrthogonalView._update_cursor_from_position(view_like, position=object())

        self.assertTrue(result)
        self.assertEqual(cursor_calls, [(1, 2, 3)])

    def test_update_cursor_from_position_returns_false_for_unmapped_position(self) -> None:
        cursor_calls: list[tuple[int, int, int]] = []
        view_like = SimpleNamespace(
            _indices_from_position=lambda _position: None,
            input_handlers=SimpleNamespace(on_drag_cursor=lambda indices: cursor_calls.append(tuple(indices))),
        )

        result = OrthogonalView._update_cursor_from_position(view_like, position=object())

        self.assertFalse(result)
        self.assertEqual(cursor_calls, [])

    def test_target_pan_for_cursor_centering_uses_cursor_to_canvas_center_delta(self) -> None:
        view_like = SimpleNamespace(
            _latest=SimpleNamespace(image=SimpleNamespace(shape=(10, 20))),
            _canvas_widget=SimpleNamespace(
                size=lambda: SimpleNamespace(width=lambda: 200, height=lambda: 100)
            ),
            _map_canvas_to_image_coords=lambda _pos, _w, _h: (7.0, 6.0),
            _cursor_xy=lambda _scale: (2.0, 3.0),
            _current_level_scale=lambda: 1,
            state=SimpleNamespace(pan=(1.0, 1.5)),
        )

        result = OrthogonalView.target_pan_for_cursor_centering(view_like)

        self.assertEqual(result, (6.0, 4.5))

    def test_target_pan_for_cursor_centering_falls_back_to_image_center(self) -> None:
        view_like = SimpleNamespace(
            _latest=SimpleNamespace(image=SimpleNamespace(shape=(10, 20))),
            _canvas_widget=SimpleNamespace(
                size=lambda: SimpleNamespace(width=lambda: 200, height=lambda: 100)
            ),
            _map_canvas_to_image_coords=lambda _pos, _w, _h: None,
            _cursor_xy=lambda _scale: (4.0, 1.0),
            _current_level_scale=lambda: 1,
            state=SimpleNamespace(pan=(0.0, 0.0)),
        )

        result = OrthogonalView.target_pan_for_cursor_centering(view_like)

        self.assertEqual(result, (6.0, 4.0))

    def test_indices_from_position_clips_to_nearest_valid_voxel(self) -> None:
        view_like = SimpleNamespace(
            _latest=SimpleNamespace(image=SimpleNamespace(shape=(5, 8))),
            _map_canvas_to_image_coords=lambda _position, _width, _height: (-100.0, 1000.0),
            _current_level_scale=lambda: 2,
            input_handlers=SimpleNamespace(
                sync_manager=SimpleNamespace(state=SimpleNamespace(slice_indices=(7, 11, 13)))
            ),
            state=SimpleNamespace(axis=0),
        )

        result = OrthogonalView._indices_from_position(view_like, position=object())

        self.assertEqual(result, (7, 8, 0))

    def test_indices_from_position_returns_none_when_mapping_is_missing(self) -> None:
        view_like = SimpleNamespace(
            _latest=SimpleNamespace(image=SimpleNamespace(shape=(5, 8))),
            _map_canvas_to_image_coords=lambda _position, _width, _height: None,
            _current_level_scale=lambda: 1,
            input_handlers=SimpleNamespace(
                sync_manager=SimpleNamespace(state=SimpleNamespace(slice_indices=(1, 2, 3)))
            ),
            state=SimpleNamespace(axis=2),
        )

        result = OrthogonalView._indices_from_position(view_like, position=object())

        self.assertIsNone(result)

    def test_indices_from_position_returns_none_for_non_finite_mapping(self) -> None:
        base_view = SimpleNamespace(
            _latest=SimpleNamespace(image=SimpleNamespace(shape=(5, 8))),
            _current_level_scale=lambda: 1,
            input_handlers=SimpleNamespace(
                sync_manager=SimpleNamespace(state=SimpleNamespace(slice_indices=(1, 2, 3)))
            ),
            state=SimpleNamespace(axis=1),
        )

        for mapped in ((np.nan, 1.0), (1.0, np.inf), (-np.inf, 2.0)):
            with self.subTest(mapped=mapped):
                view_like = SimpleNamespace(**vars(base_view))
                view_like._map_canvas_to_image_coords = (
                    lambda _position, _width, _height, value=mapped: value
                )
                result = OrthogonalView._indices_from_position(view_like, position=object())
                self.assertIsNone(result)

    def test_indices_from_position_maps_axis_specific_coordinates(self) -> None:
        base_view = SimpleNamespace(
            _latest=SimpleNamespace(image=SimpleNamespace(shape=(8, 10))),
            _map_canvas_to_image_coords=lambda _position, _width, _height: (1.9, 3.2),
            _current_level_scale=lambda: 2,
            input_handlers=SimpleNamespace(
                sync_manager=SimpleNamespace(state=SimpleNamespace(slice_indices=(7, 11, 13)))
            ),
        )

        expected_by_axis = {
            1: (6, 11, 2),
            2: (6, 2, 13),
        }
        for axis, expected in expected_by_axis.items():
            with self.subTest(axis=axis):
                view_like = SimpleNamespace(**vars(base_view))
                view_like.state = SimpleNamespace(axis=axis)
                result = OrthogonalView._indices_from_position(view_like, position=object())
                self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
