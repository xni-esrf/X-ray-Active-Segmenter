from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.bbox.interaction import BoundingBoxHandleHit
from src.bbox.projection import ProjectedBoundingBox2D
from src.render.gl_backend import (
    GLBackend,
    _BoundingBoxOverlayLayer,
    _CameraViewLayer,
    _GLSceneContext,
    _ImageTextureLayer,
    _MarkerOverlayLayer,
    _SegmentationOverlayLayer,
)


class _FakeNode:
    def __init__(self) -> None:
        self.visible = False
        self.calls: list[dict[str, object]] = []

    def set_data(self, **kwargs: object) -> None:
        self.calls.append(dict(kwargs))


class _FakeImageNode:
    def __init__(self, *, parent: object = None) -> None:
        self.parent = parent
        self.clim = None
        self.order = None
        self.data_calls: list[object] = []

    def set_data(self, data: object) -> None:
        self.data_calls.append(data)


class _FakeSegmentationNode:
    def __init__(self) -> None:
        self.visible = False
        self.opacity = None
        self.cmap = None
        self.clim = None
        self.data_calls: list[object] = []
        self.update_calls = 0

    def set_data(self, data: object) -> None:
        self.data_calls.append(data)

    def update(self) -> None:
        self.update_calls += 1


class _FakeCamera:
    def __init__(self) -> None:
        self.center = None
        self.aspect = None
        self.range_calls: list[dict[str, object]] = []
        self.zoom_calls: list[dict[str, object]] = []

    def set_range(self, **kwargs: object) -> None:
        self.range_calls.append(dict(kwargs))

    def zoom(self, factor: float, center: object = None) -> None:
        self.zoom_calls.append({"factor": factor, "center": center})


class GLBackendBackspaceGuardTests(unittest.TestCase):
    def test_scene_context_keeps_canvas_view_and_scene_references(self) -> None:
        canvas = object()
        scene = object()
        view = SimpleNamespace(scene=scene)

        context = _GLSceneContext(canvas=canvas, view=view, scene=view.scene)

        self.assertIs(context.canvas, canvas)
        self.assertIs(context.view, view)
        self.assertIs(context.scene, scene)

    def test_image_texture_layer_uploads_image_and_recreates_node_for_float_data(self) -> None:
        scene = object()
        created_nodes: list[_FakeImageNode] = []

        def _make_image(_data: object, *, parent: object, cmap: str) -> _FakeImageNode:
            self.assertEqual(cmap, "grays")
            node = _FakeImageNode(parent=parent)
            created_nodes.append(node)
            return node

        scene_module = SimpleNamespace(Image=_make_image)
        layer = _ImageTextureLayer()
        layer.initialize(
            scene_module,
            _GLSceneContext(canvas=object(), view=object(), scene=scene),
        )
        first_node = layer.image_node
        self.assertIs(first_node.parent, scene)
        self.assertEqual(layer.image_dtype.name, "uint8")

        uint_image = np.zeros((2, 3), dtype=np.uint8)
        layer.upload_image(uint_image, scene_module=scene_module)
        self.assertIs(first_node.data_calls[-1], uint_image)
        self.assertEqual(first_node.clim, (0.0, 1.0))

        float_image = np.zeros((2, 3), dtype=np.float32)
        layer.upload_image(float_image, scene_module=scene_module)
        self.assertIsNot(layer.image_node, first_node)
        self.assertIs(layer.image_node.parent, scene)
        self.assertIs(layer.image_node.data_calls[-1], float_image)
        self.assertEqual(layer.image_dtype.name, "float32")

    def test_segmentation_overlay_roi_update_reuses_rgba_cache_and_records_fallback(self) -> None:
        layer = _SegmentationOverlayLayer()
        layer.seg_node = _FakeSegmentationNode()
        segmentation = np.zeros((4, 5), dtype=np.uint16)
        segmentation[1, 2] = 2
        labels = np.array([0, 2], dtype=np.int64)

        layer.update(segmentation, None, labels)

        self.assertEqual(len(layer.seg_node.data_calls), 1)
        self.assertTrue(layer.seg_node.visible)
        original_cache = layer.seg_rgba_cache
        self.assertIs(layer.seg_node.data_calls[-1], original_cache)

        layer.seg_subupload_supported = False
        patch = np.array([[2]], dtype=np.uint16)
        layer.update(
            segmentation,
            None,
            labels,
            segmentation_roi=(2, 3, 3, 4),
            segmentation_patch=patch,
        )

        self.assertIs(layer.seg_rgba_cache, original_cache)
        self.assertIs(layer.seg_node.data_calls[-1], original_cache)
        self.assertEqual(layer.seg_roi_upload_attempts, 1)
        self.assertEqual(layer.seg_roi_full_fallback, 1)
        self.assertEqual(layer.seg_roi_fallback_reasons, {"subupload_disabled": 1})

    def test_marker_overlay_updates_crosshair_and_selection_marker(self) -> None:
        updates: list[str] = []
        canvas = SimpleNamespace(update=lambda: updates.append("update"))
        layer = _MarkerOverlayLayer()
        layer.crosshair_h = _FakeNode()
        layer.crosshair_v = _FakeNode()
        layer.selection_marker_node = _FakeNode()

        layer.set_crosshair(canvas=canvas, x=12.0, y=-3.0, width=10, height=8)
        self.assertEqual(updates, ["update"])
        h_pos = layer.crosshair_h.calls[-1]["pos"]
        v_pos = layer.crosshair_v.calls[-1]["pos"]
        self.assertEqual(h_pos.tolist(), [[0.0, 0.0], [10.0, 0.0]])
        self.assertEqual(v_pos.tolist(), [[9.0, 0.0], [9.0, 8.0]])

        layer.set_selection_marker(canvas=canvas, x=2.0, y=3.0, visible=True)
        self.assertTrue(layer.selection_marker_node.visible)
        self.assertEqual(layer.selection_marker_xy, (2.0, 3.0))
        self.assertEqual(updates, ["update", "update"])

        layer.set_selection_marker(canvas=canvas, x=2.0, y=3.0, visible=True)
        self.assertEqual(updates, ["update", "update"])

        layer.set_selection_marker(canvas=canvas, x=0.0, y=0.0, visible=False)
        self.assertFalse(layer.selection_marker_node.visible)
        self.assertIsNone(layer.selection_marker_xy)
        self.assertEqual(updates, ["update", "update", "update"])

    def test_camera_view_layer_initializes_panzoom_and_backspace_guard(self) -> None:
        camera = _FakeCamera()

        class _View:
            def __init__(self) -> None:
                self.scene = object()
                self._camera = None

            @property
            def camera(self) -> object:
                return self._camera

            @camera.setter
            def camera(self, value: object) -> None:
                if value == "panzoom":
                    self._camera = camera
                else:
                    self._camera = value

        view = _View()
        disabled_cameras: list[object] = []

        def _disable(camera_obj: object) -> None:
            disabled_cameras.append(camera_obj)

        layer = _CameraViewLayer()
        layer.initialize(
            _GLSceneContext(canvas=object(), view=view, scene=view.scene),
            disable_backspace_reset=_disable,
        )

        self.assertIs(view.camera, camera)
        self.assertEqual(disabled_cameras, [camera])
        self.assertEqual(camera.aspect, 1)

    def test_camera_view_layer_fits_image_only_once(self) -> None:
        camera = _FakeCamera()
        view = SimpleNamespace(camera=camera)
        layer = _CameraViewLayer()

        layer.fit_image_if_needed(view=view, width=20, height=10)
        layer.fit_image_if_needed(view=view, width=40, height=30)

        self.assertTrue(layer.fit_done)
        self.assertEqual(
            camera.range_calls,
            [{"x": (0, 20), "y": (0, 10), "margin": 0}],
        )
        self.assertEqual(camera.center, (10.0, 5.0))

    def test_camera_view_layer_pan_updates_camera_center_from_delta(self) -> None:
        updates: list[str] = []
        camera = _FakeCamera()
        camera.center = (10.0, 8.0)
        layer = _CameraViewLayer()

        layer.set_pan(
            view=SimpleNamespace(camera=camera),
            canvas=SimpleNamespace(update=lambda: updates.append("update")),
            pan_x=3.0,
            pan_y=-2.0,
            width=20,
            height=10,
        )

        self.assertEqual(camera.center, (7.0, 10.0))
        self.assertEqual(layer.pan, (3.0, -2.0))
        self.assertEqual(updates, ["update"])

        layer.set_pan(
            view=SimpleNamespace(camera=camera),
            canvas=SimpleNamespace(update=lambda: updates.append("update")),
            pan_x=3.0,
            pan_y=-2.0,
            width=20,
            height=10,
        )
        self.assertEqual(updates, ["update"])

    def test_camera_view_layer_zoom_maps_optional_image_center(self) -> None:
        updates: list[str] = []
        captured_transform_args: list[tuple[object, object]] = []

        class _Transform:
            def map(self, coords: np.ndarray) -> np.ndarray:
                return np.asarray(coords) + np.asarray([[100.0, 200.0]])

        class _ImageNode:
            def get_transform(self, *args: object, **kwargs: object) -> _Transform:
                captured_transform_args.append((args, kwargs))
                return _Transform()

        camera = _FakeCamera()
        camera.center = (5.0, 5.0)
        layer = _CameraViewLayer()

        layer.set_zoom(
            view=SimpleNamespace(camera=camera),
            canvas=SimpleNamespace(update=lambda: updates.append("update")),
            image_node=_ImageNode(),
            zoom=0.5,
            width=20,
            height=10,
            center=(2.0, 3.0),
        )

        self.assertEqual(len(captured_transform_args), 1)
        self.assertAlmostEqual(camera.zoom_calls[-1]["factor"], 0.5, places=6)
        self.assertEqual(camera.zoom_calls[-1]["center"], (102.0, 203.0))
        self.assertAlmostEqual(layer.zoom, 0.5, places=6)
        self.assertEqual(updates, ["update"])

        layer.set_zoom(
            view=SimpleNamespace(camera=camera),
            canvas=SimpleNamespace(update=lambda: updates.append("update")),
            image_node=_ImageNode(),
            zoom=0.5,
            width=20,
            height=10,
            center=(4.0, 6.0),
        )

        self.assertEqual(camera.center, (104.0, 206.0))
        self.assertEqual(len(camera.zoom_calls), 1)
        self.assertEqual(updates, ["update", "update"])

    def test_bounding_box_overlay_updates_lines_and_handle_markers(self) -> None:
        updates: list[str] = []
        canvas = SimpleNamespace(update=lambda: updates.append("update"))
        layer = _BoundingBoxOverlayLayer()
        layer.line_node = _FakeNode()
        layer.selected_line_node = _FakeNode()
        layer.hover_handle_node = _FakeNode()
        layer.active_handle_node = _FakeNode()
        boxes = (
            ProjectedBoundingBox2D(
                box_id="bbox_0001",
                row0=1,
                row1=4,
                col0=2,
                col1=6,
                label="train",
            ),
            ProjectedBoundingBox2D(
                box_id="bbox_0002",
                row0=10,
                row1=14,
                col0=20,
                col1=25,
                label="validation",
            ),
        )

        layer.update(
            boxes,
            canvas=canvas,
            selected_id="bbox_0002",
            hover_hit=BoundingBoxHandleHit(
                box_id="bbox_0001",
                kind="corner",
                handle="top_left",
            ),
            active_hit=BoundingBoxHandleHit(
                box_id="bbox_0002",
                kind="edge",
                handle="right",
            ),
        )

        self.assertTrue(layer.line_node.visible)
        self.assertTrue(layer.selected_line_node.visible)
        self.assertTrue(layer.hover_handle_node.visible)
        self.assertTrue(layer.active_handle_node.visible)
        self.assertEqual(layer.line_node.calls[-1]["connect"], "segments")
        self.assertEqual(layer.selected_line_node.calls[-1]["connect"], "segments")
        self.assertEqual(
            layer.hover_handle_node.calls[-1]["pos"].tolist(),
            [[2.0, 1.0, 0.0]],
        )
        self.assertEqual(
            layer.active_handle_node.calls[-1]["pos"].tolist(),
            [[25.0, 12.0, 0.0]],
        )
        self.assertEqual(updates, ["update"])

        layer.clear(canvas=canvas)
        self.assertFalse(layer.line_node.visible)
        self.assertFalse(layer.selected_line_node.visible)
        self.assertFalse(layer.hover_handle_node.visible)
        self.assertFalse(layer.active_handle_node.visible)
        self.assertEqual(updates, ["update", "update"])

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

    def test_set_segmentation_opacity_clamps_and_updates_node_and_canvas(self) -> None:
        updates: list[str] = []
        backend = GLBackend()
        backend._segmentation_overlay.seg_node = SimpleNamespace(opacity=None)
        backend._canvas = SimpleNamespace(update=lambda: updates.append("update"))

        backend.set_segmentation_opacity(1.5)
        self.assertEqual(backend.segmentation_opacity(), 1.0)
        self.assertEqual(backend._segmentation_overlay.seg_node.opacity, 1.0)

        backend.set_segmentation_opacity(-0.2)
        self.assertEqual(backend.segmentation_opacity(), 0.0)
        self.assertEqual(backend._segmentation_overlay.seg_node.opacity, 0.0)

        backend.set_segmentation_opacity(0.35)
        self.assertAlmostEqual(backend.segmentation_opacity(), 0.35, places=6)
        self.assertAlmostEqual(
            backend._segmentation_overlay.seg_node.opacity,
            0.35,
            places=6,
        )
        self.assertEqual(updates, ["update", "update", "update"])


if __name__ == "__main__":
    unittest.main()
