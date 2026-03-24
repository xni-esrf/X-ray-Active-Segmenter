from __future__ import annotations

import unittest

import numpy as np

from src.data import open_volume
from src.io.loader import VolumeInfo, VolumeLoader
from src.render import Renderer


class _ArrayLoader(VolumeLoader):
    def __init__(
        self,
        array: np.ndarray,
        *,
        path: str = "<array-loader>",
        chunk_shape: tuple[int, int, int] | None = (2, 2, 2),
    ) -> None:
        super().__init__(path)
        self._array = np.asarray(array)
        self.calls = 0
        self._info = VolumeInfo(
            shape=tuple(int(v) for v in self._array.shape),
            dtype=str(self._array.dtype),
            voxel_spacing=(1.0, 1.0, 1.0),
            chunk_shape=chunk_shape,
            axes="zyx",
        )

    @property
    def info(self) -> VolumeInfo:
        return self._info

    def get_chunk(self, zyx_slices: tuple[slice, slice, slice]) -> np.ndarray:
        self.calls += 1
        return np.asarray(self._array[zyx_slices])


class _CastingArrayLoader(VolumeLoader):
    def __init__(
        self,
        array: np.ndarray,
        *,
        info_dtype: str,
        cast_dtype: np.dtype,
        path: str = "<casting-array-loader>",
    ) -> None:
        super().__init__(path)
        self._array = np.asarray(array)
        self._cast_dtype = np.dtype(cast_dtype)
        self.calls = 0
        self._info = VolumeInfo(
            shape=tuple(int(v) for v in self._array.shape),
            dtype=info_dtype,
            voxel_spacing=(1.0, 1.0, 1.0),
            chunk_shape=(2, 2, 2),
            axes="zyx",
        )

    @property
    def info(self) -> VolumeInfo:
        return self._info

    def get_chunk(self, zyx_slices: tuple[slice, slice, slice]) -> np.ndarray:
        self.calls += 1
        chunk = np.asarray(self._array[zyx_slices])
        return chunk.astype(self._cast_dtype, copy=False)


class RendererContrastWindowTests(unittest.TestCase):
    def test_attach_volume_computes_data_range_and_default_window(self) -> None:
        array = np.arange(4 * 4 * 4, dtype=np.float32).reshape((4, 4, 4))
        loader = _ArrayLoader(array, chunk_shape=(2, 2, 2))
        volume = open_volume(loader, cache=None)
        renderer = Renderer()

        renderer.attach_volume(volume)

        self.assertEqual(renderer.get_data_range(), (0.0, 63.0))
        self.assertEqual(renderer.get_window_range(), (0.0, 63.0))
        self.assertGreater(loader.calls, 1)

    def test_set_window_applies_linear_clipping(self) -> None:
        array = np.asarray(
            [
                [[0.0, 25.0], [50.0, 75.0]],
                [[100.0, 125.0], [150.0, 175.0]],
            ],
            dtype=np.float32,
        )
        loader = _ArrayLoader(array, chunk_shape=(1, 2, 2))
        volume = open_volume(loader, cache=None)
        renderer = Renderer()
        renderer.attach_volume(volume)
        renderer.set_window(25.0, 125.0)

        result = renderer.render_slice("axial", axis=0, slice_index=0)

        expected = np.asarray([[0.0, 0.0], [0.25, 0.5]], dtype=np.float32)
        np.testing.assert_allclose(result.image, expected, rtol=0.0, atol=1e-6)

    def test_set_window_rejects_invalid_bounds(self) -> None:
        array = np.arange(3 * 3 * 3, dtype=np.uint16).reshape((3, 3, 3))
        volume = open_volume(_ArrayLoader(array), cache=None)
        renderer = Renderer()
        renderer.attach_volume(volume)

        with self.assertRaisesRegex(ValueError, "vmin < vmax"):
            renderer.set_window(10.0, 10.0)
        with self.assertRaisesRegex(ValueError, "within the raw data range"):
            renderer.set_window(-1.0, 10.0)
        with self.assertRaisesRegex(ValueError, "finite real numbers"):
            renderer.set_window(float("nan"), 10.0)

    def test_attach_volume_rejects_non_finite_float_values(self) -> None:
        array = np.asarray(
            [[[0.0, 1.0], [2.0, np.nan]], [[4.0, 5.0], [6.0, 7.0]]],
            dtype=np.float32,
        )
        volume = open_volume(_ArrayLoader(array), cache=None)
        renderer = Renderer()

        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            renderer.attach_volume(volume)

    def test_attach_volume_failure_keeps_previous_volume_and_window(self) -> None:
        initial = np.arange(2 * 2 * 2, dtype=np.float32).reshape((2, 2, 2))
        initial_volume = open_volume(_ArrayLoader(initial), cache=None)
        renderer = Renderer()
        renderer.attach_volume(initial_volume)

        invalid = np.asarray(
            [[[0.0, 1.0], [2.0, np.nan]], [[4.0, 5.0], [6.0, 7.0]]],
            dtype=np.float32,
        )
        invalid_volume = open_volume(_ArrayLoader(invalid), cache=None)
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            renderer.attach_volume(invalid_volume)

        self.assertEqual(renderer.get_data_range(), (0.0, 7.0))
        self.assertEqual(renderer.get_window_range(), (0.0, 7.0))
        result = renderer.render_slice("axial", axis=0, slice_index=0)
        expected = np.asarray([[0.0, 1.0 / 7.0], [2.0 / 7.0, 3.0 / 7.0]], dtype=np.float32)
        np.testing.assert_allclose(result.image, expected, rtol=0.0, atol=1e-6)

    def test_constant_volume_defaults_to_zero_image(self) -> None:
        array = np.full((2, 2, 2), fill_value=7, dtype=np.uint16)
        volume = open_volume(_ArrayLoader(array), cache=None)
        renderer = Renderer()
        renderer.attach_volume(volume)

        result = renderer.render_slice("axial", axis=0, slice_index=0)

        self.assertEqual(renderer.get_data_range(), (7.0, 7.0))
        self.assertEqual(renderer.get_window_range(), (7.0, 7.0))
        np.testing.assert_allclose(result.image, np.zeros((2, 2), dtype=np.float32), atol=0.0)

    def test_attach_volume_resets_window_to_default_on_new_raw_load(self) -> None:
        first = np.arange(3 * 3 * 3, dtype=np.float32).reshape((3, 3, 3))
        second = np.arange(8, dtype=np.uint16).reshape((2, 2, 2)) + 100
        first_volume = open_volume(_ArrayLoader(first), cache=None)
        second_volume = open_volume(_ArrayLoader(second), cache=None)
        renderer = Renderer()
        renderer.attach_volume(first_volume)
        renderer.set_window(3.0, 10.0)

        renderer.attach_volume(second_volume)

        self.assertEqual(renderer.get_data_range(), (100.0, 107.0))
        self.assertEqual(renderer.get_window_range(), (100.0, 107.0))

    def test_attach_volume_resets_level_mode_to_defaults_on_new_raw_load(self) -> None:
        first = np.arange(4001, dtype=np.float32).reshape((4001, 1, 1))
        second = np.arange(16, dtype=np.float32).reshape((16, 1, 1))
        first_volume = open_volume(_ArrayLoader(first, chunk_shape=(4001, 1, 1)), cache=None)
        second_volume = open_volume(_ArrayLoader(second, chunk_shape=(16, 1, 1)), cache=None)
        renderer = Renderer()
        renderer.attach_volume(
            first_volume,
            levels=(first_volume, first_volume, first_volume, first_volume),
        )
        renderer.set_manual_level(2)
        renderer.set_auto_level_enabled(False)
        self.assertFalse(renderer.is_auto_level_enabled())
        self.assertEqual(renderer.manual_level(), 2)

        renderer.attach_volume(second_volume, levels=(second_volume, second_volume))

        self.assertTrue(renderer.is_auto_level_enabled())
        self.assertEqual(renderer.manual_level(), 0)

    def test_attach_volume_uses_post_cast_loader_values_for_data_range(self) -> None:
        source = np.asarray(
            [[[0.10006, 0.20007], [0.30008, 0.40009]], [[-0.50001, -0.40002], [0.0, 1.00009]]],
            dtype=np.float32,
        )
        loader = _CastingArrayLoader(
            source,
            info_dtype=str(np.dtype(np.float32)),
            cast_dtype=np.float16,
        )
        volume = open_volume(loader, cache=None)
        renderer = Renderer()

        renderer.attach_volume(volume)

        casted = source.astype(np.float16, copy=False)
        expected = (float(np.min(casted)), float(np.max(casted)))
        self.assertEqual(renderer.get_data_range(), expected)
        self.assertEqual(renderer.get_window_range(), expected)

    def test_target_level_uses_zoom_in_auto_mode_and_manual_override_when_disabled(self) -> None:
        raw_array = np.arange(4001, dtype=np.float32).reshape((4001, 1, 1))
        raw_volume = open_volume(
            _ArrayLoader(raw_array, chunk_shape=(4001, 1, 1)),
            cache=None,
        )
        renderer = Renderer()
        renderer.attach_volume(
            raw_volume,
            levels=(raw_volume, raw_volume, raw_volume, raw_volume),
        )

        self.assertTrue(renderer.is_auto_level_enabled())
        self.assertEqual(renderer.target_level_for_view(0, 1.0), 3)
        self.assertEqual(renderer.target_level_for_view(0, 0.6), 2)
        self.assertEqual(renderer.target_level_for_view(0, 0.3), 1)
        self.assertEqual(renderer.target_level_for_view(0, 0.2), 0)

        renderer.set_manual_level(2)
        renderer.set_auto_level_enabled(False)
        self.assertFalse(renderer.is_auto_level_enabled())
        self.assertEqual(renderer.manual_level(), 2)
        self.assertEqual(renderer.target_level_for_view(0, 1.0), 2)
        self.assertEqual(renderer.target_level_for_view(0, 0.6), 2)
        self.assertEqual(renderer.target_level_for_view(0, 0.3), 2)
        self.assertEqual(renderer.target_level_for_view(0, 0.2), 2)

        renderer.set_auto_level_enabled(True)
        self.assertTrue(renderer.is_auto_level_enabled())
        self.assertEqual(renderer.target_level_for_view(0, 1.0), 3)
        self.assertEqual(renderer.target_level_for_view(0, 0.6), 2)
        self.assertEqual(renderer.target_level_for_view(0, 0.3), 1)
        self.assertEqual(renderer.target_level_for_view(0, 0.2), 0)

    def test_set_manual_level_clamps_to_available_range(self) -> None:
        raw_array = np.arange(2001, dtype=np.float32).reshape((2001, 1, 1))
        raw_volume = open_volume(
            _ArrayLoader(raw_array, chunk_shape=(2001, 1, 1)),
            cache=None,
        )
        renderer = Renderer()
        renderer.attach_volume(
            raw_volume,
            levels=(raw_volume, raw_volume, raw_volume),
        )

        self.assertEqual(renderer.available_level_count(), 3)
        self.assertEqual(renderer.set_manual_level(99), 2)
        self.assertEqual(renderer.manual_level(), 2)
        self.assertEqual(renderer.set_manual_level(-4), 0)
        self.assertEqual(renderer.manual_level(), 0)

        renderer.detach_volume()
        self.assertEqual(renderer.available_level_count(), 0)
        self.assertEqual(renderer.set_manual_level(7), 0)
        self.assertEqual(renderer.manual_level(), 0)

    def test_available_level_count_uses_minimum_with_active_segmentation(self) -> None:
        raw_array = np.arange(4001, dtype=np.float32).reshape((4001, 1, 1))
        seg_array = np.zeros((4001, 1, 1), dtype=np.uint8)
        raw_volume = open_volume(
            _ArrayLoader(raw_array, chunk_shape=(4001, 1, 1)),
            cache=None,
        )
        seg_volume = open_volume(
            _ArrayLoader(seg_array, chunk_shape=(4001, 1, 1)),
            cache=None,
        )
        renderer = Renderer()
        renderer.attach_volume(
            raw_volume,
            levels=(raw_volume, raw_volume, raw_volume, raw_volume),
        )
        self.assertEqual(renderer.available_level_count(), 4)

        renderer.attach_segmentation(
            seg_volume,
            levels=(seg_volume, seg_volume),
        )
        self.assertEqual(renderer.available_level_count(), 2)
        self.assertEqual(renderer.set_manual_level(3), 1)
        renderer.set_auto_level_enabled(False)
        self.assertEqual(renderer.target_level_for_view(0, 1.0), 1)

        renderer.detach_segmentation()
        self.assertEqual(renderer.available_level_count(), 4)

    def test_attach_segmentation_clamps_existing_manual_level_immediately(self) -> None:
        raw_array = np.arange(4001, dtype=np.float32).reshape((4001, 1, 1))
        seg_array = np.zeros((4001, 1, 1), dtype=np.uint8)
        raw_volume = open_volume(
            _ArrayLoader(raw_array, chunk_shape=(4001, 1, 1)),
            cache=None,
        )
        seg_volume = open_volume(
            _ArrayLoader(seg_array, chunk_shape=(4001, 1, 1)),
            cache=None,
        )
        renderer = Renderer()
        renderer.attach_volume(
            raw_volume,
            levels=(raw_volume, raw_volume, raw_volume, raw_volume),
        )
        renderer.set_manual_level(3)
        renderer.set_auto_level_enabled(False)
        self.assertEqual(renderer.manual_level(), 3)

        renderer.attach_segmentation(
            seg_volume,
            levels=(seg_volume, seg_volume),
        )

        self.assertEqual(renderer.available_level_count(), 2)
        self.assertEqual(renderer.manual_level(), 1)
        self.assertEqual(renderer.target_level_for_view(0, 1.0), 1)


if __name__ == "__main__":
    unittest.main()
