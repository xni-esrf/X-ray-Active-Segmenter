from __future__ import annotations

import unittest

import numpy as np

from src.learning.streaming_inference_plan import build_streaming_inference_plan
from src.learning.zero_occupancy import (
    compute_streaming_normalization_from_sums,
    minivol_is_run,
    prepare_streaming_occupancy_and_stats,
    scan_occupancy_and_intensity,
    streaming_block_grid_bounds,
    streaming_occupancy_scan_bounds,
)


MINIVOL = 20
STRIDE = 10


class _FakeVolume:
    """A get_chunk-backed volume over a numpy array (optionally chunked)."""

    def __init__(self, array: np.ndarray, chunk_shape=None) -> None:
        self.array = np.asarray(array)
        self.chunk_shape = chunk_shape
        self.calls: list = []

    def get_chunk(self, zyx_slices):
        self.calls.append(zyx_slices)
        return self.array[zyx_slices]


def _run_domain_mask_from_grid(grid, *, bounds, raw_shape, minivol_size, stride):
    """Independent voxel-level run domain: union of run cells' block extents."""
    _, _, cell_lo, cell_hi = streaming_block_grid_bounds(bounds, stride=stride)
    mask = np.zeros(raw_shape, dtype=bool)
    for gz in range(cell_lo[0], cell_hi[0] + 1):
        for gy in range(cell_lo[1], cell_hi[1] + 1):
            for gx in range(cell_lo[2], cell_hi[2] + 1):
                start = (gz * stride, gy * stride, gx * stride)
                if not minivol_is_run(
                    grid,
                    start=start,
                    minivol_size=minivol_size,
                    stride=stride,
                    raw_shape=raw_shape,
                ):
                    continue
                sl = tuple(
                    slice(
                        max(0, start[a]),
                        min(raw_shape[a], start[a] + minivol_size),
                    )
                    for a in range(3)
                )
                mask[sl] = True
    return mask


def _blobby_volume(shape=(80, 80, 80)):
    rng = np.random.default_rng(1234)
    arr = np.zeros(shape, dtype=np.float32)

    def put(lo_frac, hi_frac, low, high):
        sl = tuple(
            slice(int(shape[a] * lo_frac[a]), int(shape[a] * hi_frac[a]))
            for a in range(3)
        )
        size = tuple(sl[a].stop - sl[a].start for a in range(3))
        if all(s > 0 for s in size):
            arr[sl] = rng.uniform(low, high, size=size).astype(np.float32)

    # a couple of non-background blobs (scaled to the volume) with structured values
    put((0.38, 0.35, 0.41), (0.65, 0.55, 0.62), 5.0, 40.0)
    put((0.12, 0.75, 0.15), (0.30, 0.92, 0.32), 1.0, 9.0)
    return arr


class ScanOccupancyAndIntensityTest(unittest.TestCase):
    def test_block_sums_match_direct_reduction(self):
        arr = _blobby_volume((40, 40, 40))
        vol = _FakeVolume(arr)
        bounds = ((0, 40), (0, 40), (0, 40))
        scan = scan_occupancy_and_intensity(vol, bounds=bounds, block_size=STRIDE)
        gz, gy, gx = scan.grid.occupied.shape
        for cz in range(gz):
            for cy in range(gy):
                for cx in range(gx):
                    block = arr[
                        cz * STRIDE : (cz + 1) * STRIDE,
                        cy * STRIDE : (cy + 1) * STRIDE,
                        cx * STRIDE : (cx + 1) * STRIDE,
                    ].astype(np.float64)
                    self.assertAlmostEqual(
                        scan.block_sum[cz, cy, cx], float(block.sum()), places=4
                    )
                    self.assertAlmostEqual(
                        scan.block_sq[cz, cy, cx], float((block * block).sum()), places=3
                    )
                    self.assertEqual(
                        bool(scan.grid.occupied[cz, cy, cx]),
                        bool(np.any(block != 0)),
                    )

    def test_sums_invariant_to_chunk_shape(self):
        arr = _blobby_volume((60, 40, 50))
        bounds = ((0, 60), (0, 40), (0, 50))
        a = scan_occupancy_and_intensity(_FakeVolume(arr), bounds=bounds, block_size=STRIDE)
        b = scan_occupancy_and_intensity(
            _FakeVolume(arr, chunk_shape=(20, 20, 20)), bounds=bounds, block_size=STRIDE
        )
        np.testing.assert_allclose(a.block_sum, b.block_sum, rtol=0, atol=1e-6)
        np.testing.assert_allclose(a.block_sq, b.block_sq, rtol=0, atol=1e-3)
        np.testing.assert_array_equal(a.grid.occupied, b.grid.occupied)


class StreamingScanBoundsTest(unittest.TestCase):
    def test_scan_bounds_are_stride_aligned_and_cover_margin(self):
        bounds = ((133, 777), (10, 990), (500, 999))
        raw_shape = (1000, 2000, 1000)
        scan_bounds = streaming_occupancy_scan_bounds(
            bounds, raw_volume_shape=raw_shape, stride=STRIDE
        )
        write_lo, write_hi, _, _ = streaming_block_grid_bounds(bounds, stride=STRIDE)
        for a in range(3):
            lo, hi = scan_bounds[a]
            self.assertEqual(lo % STRIDE, 0)
            # lo covers two blocks below the first write block (clamped to 0)
            self.assertEqual(lo, max(0, (write_lo[a] - 2) * STRIDE))
            # hi covers two blocks past the last write block (clamped to volume)
            self.assertEqual(hi, min(raw_shape[a], (write_hi[a] + 3) * STRIDE))
            self.assertGreater(hi, lo)


class StreamingNormalizationStatsTest(unittest.TestCase):
    CONFIGS = [
        ((80, 80, 80), ((0, 80), (0, 80), (0, 80))),
        ((80, 80, 80), ((5, 75), (5, 75), (5, 75))),
        ((80, 96, 72), ((13, 77), (10, 90), (7, 65))),  # non-aligned, anisotropic
    ]

    def test_stats_match_direct_numpy_over_run_domain(self):
        for raw_shape, bounds in self.CONFIGS:
            with self.subTest(bounds=bounds):
                arr = _blobby_volume(raw_shape)
                prepass = prepare_streaming_occupancy_and_stats(
                    _FakeVolume(arr),
                    requested_bounds=bounds,
                    raw_volume_shape=raw_shape,
                    minivol_size=MINIVOL,
                )
                stats = prepass.normalization
                domain = _run_domain_mask_from_grid(
                    prepass.grid,
                    bounds=bounds,
                    raw_shape=raw_shape,
                    minivol_size=MINIVOL,
                    stride=STRIDE,
                )
                voxels = arr[domain].astype(np.float64)
                self.assertGreater(voxels.size, 0)
                self.assertEqual(stats.voxel_count, int(voxels.size))
                self.assertAlmostEqual(stats.mean, float(voxels.mean()), places=4)
                expected_std = float(voxels.std())
                if expected_std == 0.0:
                    expected_std = 1.0
                self.assertAlmostEqual(stats.std, expected_std, places=4)

    def test_run_domain_includes_background_shell_zeros(self):
        # A run minivolume's zeros count toward the mean (Choice 1), so the
        # streaming mean is below the mean of the non-zero voxels alone.
        raw_shape = (80, 80, 80)
        bounds = ((0, 80), (0, 80), (0, 80))
        arr = _blobby_volume(raw_shape)
        prepass = prepare_streaming_occupancy_and_stats(
            _FakeVolume(arr),
            requested_bounds=bounds,
            raw_volume_shape=raw_shape,
            minivol_size=MINIVOL,
        )
        nonzero_mean = float(arr[arr != 0].mean())
        self.assertLess(prepass.normalization.mean, nonzero_mean)
        # but there are fewer domain voxels than the whole bbox (skipping happened)
        self.assertLess(prepass.normalization.voxel_count, arr.size)

    def test_empty_region_gives_neutral_stats(self):
        raw_shape = (80, 80, 80)
        bounds = ((0, 80), (0, 80), (0, 80))
        arr = np.zeros(raw_shape, dtype=np.float32)
        prepass = prepare_streaming_occupancy_and_stats(
            _FakeVolume(arr),
            requested_bounds=bounds,
            raw_volume_shape=raw_shape,
            minivol_size=MINIVOL,
        )
        self.assertEqual(prepass.normalization.voxel_count, 0)
        self.assertEqual(prepass.normalization.mean, 0.0)
        self.assertEqual(prepass.normalization.std, 1.0)

    def test_constant_intensity_gives_unit_std(self):
        stats = compute_streaming_normalization_from_sums(
            sum_x=5.0 * 1000, sum_x2=25.0 * 1000, voxel_count=1000
        )
        self.assertAlmostEqual(stats.mean, 5.0)
        self.assertEqual(stats.std, 1.0)


class StreamingPrePassPlanConsistencyTest(unittest.TestCase):
    def test_prepass_run_domain_matches_plan_run_cells(self):
        # The pre-pass grid drives the planner; every run block in the pre-pass
        # domain must correspond to a run cell in the plan and vice versa.
        raw_shape = (80, 80, 80)
        bounds = ((5, 75), (5, 75), (5, 75))
        arr = _blobby_volume(raw_shape)
        prepass = prepare_streaming_occupancy_and_stats(
            _FakeVolume(arr),
            requested_bounds=bounds,
            raw_volume_shape=raw_shape,
            minivol_size=MINIVOL,
        )
        plan = build_streaming_inference_plan(
            requested_bounds=bounds,
            raw_volume_shape=raw_shape,
            minivol_size=MINIVOL,
            occupancy=prepass.grid,
        )
        # Same grid + same rule => identical run decision per cell.
        for cell in plan.cells:
            expected = minivol_is_run(
                prepass.grid,
                start=cell.start,
                minivol_size=MINIVOL,
                stride=STRIDE,
                raw_shape=raw_shape,
            )
            self.assertEqual(cell.run, expected)
        # There is genuinely a mix of run and skipped cells here.
        self.assertGreater(plan.run_cell_count, 0)
        self.assertLess(plan.run_cell_count, plan.total_cell_count)


class StreamingSkipEmptyRegionsTest(unittest.TestCase):
    def test_skip_off_runs_all_cells_and_normalizes_over_whole_bbox(self):
        raw_shape = (80, 80, 80)
        bounds = ((5, 75), (5, 75), (5, 75))
        arr = _blobby_volume(raw_shape)

        skip_on = prepare_streaming_occupancy_and_stats(
            _FakeVolume(arr), requested_bounds=bounds, raw_volume_shape=raw_shape,
            minivol_size=MINIVOL, skip_empty_regions=True,
        )
        skip_off = prepare_streaming_occupancy_and_stats(
            _FakeVolume(arr), requested_bounds=bounds, raw_volume_shape=raw_shape,
            minivol_size=MINIVOL, skip_empty_regions=False,
        )

        # skip off marks the whole scanned grid occupied.
        self.assertTrue(bool(np.all(skip_off.grid.occupied)))
        self.assertFalse(bool(np.all(skip_on.grid.occupied)))

        plan_on = build_streaming_inference_plan(
            requested_bounds=bounds, raw_volume_shape=raw_shape,
            minivol_size=MINIVOL, occupancy=skip_on.grid,
        )
        plan_off = build_streaming_inference_plan(
            requested_bounds=bounds, raw_volume_shape=raw_shape,
            minivol_size=MINIVOL, occupancy=skip_off.grid,
        )
        # skip on: genuine mix; skip off: every cell runs.
        self.assertGreater(plan_on.run_cell_count, 0)
        self.assertLess(plan_on.run_cell_count, plan_on.total_cell_count)
        self.assertEqual(plan_off.run_cell_count, plan_off.total_cell_count)

        # skip-off normalization spans strictly more voxels than skip-on, and
        # matches a direct reduction over the (now whole-bbox) run domain.
        self.assertGreater(
            skip_off.normalization.voxel_count, skip_on.normalization.voxel_count
        )
        domain_off = _run_domain_mask_from_grid(
            skip_off.grid, bounds=bounds, raw_shape=raw_shape,
            minivol_size=MINIVOL, stride=STRIDE,
        )
        voxels_off = arr[domain_off].astype(np.float64)
        self.assertEqual(skip_off.normalization.voxel_count, int(voxels_off.size))
        self.assertAlmostEqual(
            skip_off.normalization.mean, float(voxels_off.mean()), places=4
        )


class StreamingNormalizationFromSumsTest(unittest.TestCase):
    def test_negative_variance_rounding_is_clamped(self):
        # sum_x2/n - mean^2 slightly negative due to rounding -> std falls back to 1.
        # mean = 3/3 = 1, mean^2 = 1; sum_x2/n = 2.9999999999/3 < 1 -> variance < 0.
        stats = compute_streaming_normalization_from_sums(
            sum_x=3.0, sum_x2=2.9999999999, voxel_count=3
        )
        self.assertEqual(stats.mean, 1.0)
        self.assertEqual(stats.std, 1.0)


if __name__ == "__main__":
    unittest.main()
