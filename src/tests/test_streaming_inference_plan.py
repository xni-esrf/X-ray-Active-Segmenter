from __future__ import annotations

import itertools
import unittest

import numpy as np

from src.learning.streaming_inference_plan import (
    build_streaming_inference_plan,
    DEFAULT_CHUNK_STRIDE_MULTIPLE,
)
from src.learning.zero_occupancy import ZeroOccupancyGrid


STRIDE = 100
MINIVOL = 200


def _covering_cell_grid_indices(block_index):
    """The (up to eight) cells whose extent overlaps a block's own extent."""
    return [
        (block_index[0] + dz, block_index[1] + dy, block_index[2] + dx)
        for dz in (-1, 0)
        for dy in (-1, 0)
        for dx in (-1, 0)
    ]


class StreamingInferencePlanInvariantsTest(unittest.TestCase):
    CONFIGS = [
        # (raw_shape, bbox)
        ((1000, 1000, 1000), ((0, 1000), (0, 1000), (0, 1000))),
        ((1000, 1000, 1000), ((200, 800), (300, 700), (100, 900))),
        ((1000, 1000, 1000), ((50, 150), (50, 150), (50, 150))),  # tiny, single-ish block
        ((1000, 1000, 1000), ((0, 250), (400, 650), (750, 1000))),  # touches both edges
        ((640, 1280, 320), ((0, 640), (0, 1280), (0, 320))),  # anisotropic -> scan axis = y
        ((1000, 1000, 1000), ((133, 777), (10, 990), (500, 999))),  # non-stride-aligned
    ]

    def _each_plan(self):
        for raw_shape, bbox in self.CONFIGS:
            plan = build_streaming_inference_plan(
                requested_bounds=bbox,
                raw_volume_shape=raw_shape,
                minivol_size=MINIVOL,
                occupancy=None,
            )
            yield raw_shape, bbox, plan

    def test_scan_axis_is_longest_bbox_axis(self):
        for raw_shape, bbox, plan in self._each_plan():
            with self.subTest(bbox=bbox):
                sizes = [b[1] - b[0] for b in bbox]
                self.assertEqual(plan.scan_axis, int(np.argmax(sizes)))

    def test_cells_are_raster_ordered(self):
        for raw_shape, bbox, plan in self._each_plan():
            with self.subTest(bbox=bbox):
                scan = plan.scan_axis
                order = [scan] + [a for a in range(3) if a != scan]
                keys = [
                    tuple(cell.grid_index[a] for a in order) for cell in plan.cells
                ]
                self.assertEqual(keys, sorted(keys))
                self.assertEqual([c.index for c in plan.cells], list(range(len(plan.cells))))

    def test_cell_start_and_extraction_reconstruct_minivolume(self):
        for raw_shape, bbox, plan in self._each_plan():
            for cell in plan.cells:
                with self.subTest(bbox=bbox, cell=cell.grid_index):
                    self.assertEqual(
                        cell.start,
                        tuple(cell.grid_index[a] * STRIDE for a in range(3)),
                    )
                    ex = cell.extraction
                    for a in range(3):
                        rs = ex.raw_slices[a]
                        # real read window is inside the volume and non-empty
                        self.assertGreaterEqual(rs.start, 0)
                        self.assertLessEqual(rs.stop, raw_shape[a])
                        self.assertGreater(rs.stop, rs.start)
                        # padding only where the minivolume runs off a true edge
                        raw_start = cell.start[a]
                        raw_stop = cell.start[a] + MINIVOL
                        self.assertEqual(ex.pad_before[a], max(0, -raw_start))
                        self.assertEqual(ex.pad_after[a], max(0, raw_stop - raw_shape[a]))
                        # padded read exactly fills a minivolume
                        self.assertEqual(
                            (rs.stop - rs.start) + ex.pad_before[a] + ex.pad_after[a],
                            MINIVOL,
                        )

    def test_write_blocks_tile_the_bbox_exactly(self):
        for raw_shape, bbox, plan in self._each_plan():
            with self.subTest(bbox=bbox):
                shape = tuple(b[1] - b[0] for b in bbox)
                cover = np.zeros(shape, dtype=np.int32)
                for block in plan.blocks:
                    sl = tuple(
                        slice(
                            block.write_slices[a].start - bbox[a][0],
                            block.write_slices[a].stop - bbox[a][0],
                        )
                        for a in range(3)
                    )
                    for a in range(3):
                        # write region is non-empty and inside the bbox
                        self.assertGreater(
                            block.write_slices[a].stop, block.write_slices[a].start
                        )
                        self.assertGreaterEqual(block.write_slices[a].start, bbox[a][0])
                        self.assertLessEqual(block.write_slices[a].stop, bbox[a][1])
                    cover[sl] += 1
                # every bbox voxel is written by exactly one block
                self.assertTrue(np.all(cover == 1))

    def test_finalizing_cell_is_raster_last_covering_cell(self):
        for raw_shape, bbox, plan in self._each_plan():
            cell_index_by_grid = {c.grid_index: c.index for c in plan.cells}
            for block in plan.blocks:
                with self.subTest(bbox=bbox, block=block.grid_index):
                    covering = [
                        cell_index_by_grid[g]
                        for g in _covering_cell_grid_indices(block.grid_index)
                        if g in cell_index_by_grid
                    ]
                    # all eight covering cells exist in the grid
                    self.assertEqual(len(covering), 8)
                    # the block is finalized exactly when the last of them runs
                    self.assertEqual(block.finalizing_cell_index, max(covering))
                    # and that finalizing cell is the block's own grid index
                    self.assertEqual(
                        plan.cells[block.finalizing_cell_index].grid_index,
                        block.grid_index,
                    )

    def test_cell_finalize_events_match_blocks(self):
        for raw_shape, bbox, plan in self._each_plan():
            with self.subTest(bbox=bbox):
                # bijection: each write block finalized by exactly one cell
                finalized = [
                    c.finalizes_block_id
                    for c in plan.cells
                    if c.finalizes_block_id is not None
                ]
                self.assertEqual(sorted(finalized), list(range(len(plan.blocks))))
                for cell in plan.cells:
                    if cell.finalizes_block_id is None:
                        continue
                    block = plan.blocks[cell.finalizes_block_id]
                    self.assertEqual(block.finalizing_cell_index, cell.index)
                    self.assertEqual(block.grid_index, cell.grid_index)

    def test_no_voxel_finalized_before_all_contributions(self):
        # Dense simulation: replay the schedule, deposit each cell's coverage,
        # and assert that when a block is finalized every covering contribution
        # has already been deposited.
        for raw_shape, bbox, plan in self._each_plan():
            with self.subTest(bbox=bbox):
                cell_index_by_grid = {c.grid_index: c.index for c in plan.cells}
                deposited = set()
                for cell in plan.cells:
                    deposited.add(cell.index)
                    if cell.finalizes_block_id is None:
                        continue
                    block = plan.blocks[cell.finalizes_block_id]
                    covering = {
                        cell_index_by_grid[g]
                        for g in _covering_cell_grid_indices(block.grid_index)
                        if g in cell_index_by_grid
                    }
                    self.assertTrue(covering.issubset(deposited))

    def test_chunk_mapping_and_completion(self):
        k = DEFAULT_CHUNK_STRIDE_MULTIPLE
        chunk_size = k * STRIDE
        for raw_shape, bbox, plan in self._each_plan():
            self.assertEqual(plan.chunk_size, chunk_size)
            seen_blocks = set()
            for chunk in plan.chunks:
                with self.subTest(bbox=bbox, chunk=chunk.chunk_index):
                    self.assertTrue(chunk.block_ids)
                    member_finalizers = []
                    for block_id in chunk.block_ids:
                        block = plan.blocks[block_id]
                        seen_blocks.add(block_id)
                        # block belongs to this chunk
                        self.assertEqual(block.chunk_id, chunk.id)
                        self.assertEqual(
                            tuple(block.grid_index[a] // k for a in range(3)),
                            chunk.chunk_index,
                        )
                        member_finalizers.append(block.finalizing_cell_index)
                        # slices_in_chunk maps write_slices relative to chunk origin
                        for a in range(3):
                            self.assertEqual(
                                chunk.chunk_slices[a].start + block.slices_in_chunk[a].start,
                                block.write_slices[a].start,
                            )
                            self.assertEqual(
                                block.slices_in_chunk[a].stop - block.slices_in_chunk[a].start,
                                block.write_slices[a].stop - block.write_slices[a].start,
                            )
                            # write region sits inside the (clamped) chunk extent
                            self.assertGreaterEqual(
                                block.write_slices[a].start, chunk.chunk_slices[a].start
                            )
                            self.assertLessEqual(
                                block.write_slices[a].stop, chunk.chunk_slices[a].stop
                            )
                    # chunk completes exactly when its last block is finalized
                    self.assertEqual(chunk.completion_cell_index, max(member_finalizers))
                    self.assertIn(chunk.id, plan.cells[chunk.completion_cell_index].completes_chunk_ids)
            # chunks partition all write blocks
            self.assertEqual(seen_blocks, set(range(len(plan.blocks))))

    def test_chunk_completion_events_are_consistent(self):
        for raw_shape, bbox, plan in self._each_plan():
            with self.subTest(bbox=bbox):
                completed = []
                for cell in plan.cells:
                    completed.extend(cell.completes_chunk_ids)
                self.assertEqual(sorted(completed), list(range(len(plan.chunks))))

    def test_no_occupancy_means_every_cell_runs(self):
        for raw_shape, bbox, plan in self._each_plan():
            with self.subTest(bbox=bbox):
                self.assertTrue(all(c.run for c in plan.cells))
                self.assertEqual(plan.run_cell_count, plan.total_cell_count)
                self.assertTrue(all(b.any_covering_run for b in plan.blocks))
                self.assertTrue(all(ch.has_data for ch in plan.chunks))


class StreamingInferencePlanSkipTest(unittest.TestCase):
    def _occupancy(self, occupied_blocks, grid_blocks=(10, 10, 10)):
        occupied = np.zeros(grid_blocks, dtype=bool)
        for b in occupied_blocks:
            occupied[b] = True
        return ZeroOccupancyGrid(origin=(0, 0, 0), block_size=STRIDE, occupied=occupied)

    def _expected_run(self, grid_index, occupied, grid_blocks):
        # A cell runs iff any occupied block lies in the 4x4x4 block window
        # [i-1, i+2] per axis (the cell plus its 3x3x3 cell neighbourhood).
        rz = range(max(0, grid_index[0] - 1), min(grid_blocks[0], grid_index[0] + 3))
        ry = range(max(0, grid_index[1] - 1), min(grid_blocks[1], grid_index[1] + 3))
        rx = range(max(0, grid_index[2] - 1), min(grid_blocks[2], grid_index[2] + 3))
        for b in itertools.product(rz, ry, rx):
            if occupied[b]:
                return True
        return False

    def test_skip_matches_neighbour_rule(self):
        grid_blocks = (10, 10, 10)
        occupied_blocks = [(5, 5, 5), (2, 8, 3)]
        occ = self._occupancy(occupied_blocks, grid_blocks)
        occupied = occ.occupied
        plan = build_streaming_inference_plan(
            requested_bounds=((100, 900), (100, 900), (100, 900)),
            raw_volume_shape=(1000, 1000, 1000),
            minivol_size=MINIVOL,
            occupancy=occ,
        )
        run_cells = 0
        for cell in plan.cells:
            expected = self._expected_run(cell.grid_index, occupied, grid_blocks)
            self.assertEqual(
                cell.run, expected, msg=f"cell {cell.grid_index} run mismatch"
            )
            run_cells += int(cell.run)
        # sanity: far fewer cells run than exist (skipping actually happened)
        self.assertLess(run_cells, plan.total_cell_count)
        self.assertGreater(run_cells, 0)

    def test_no_block_mixes_foreground_and_skipped(self):
        grid_blocks = (10, 10, 10)
        occ = self._occupancy([(5, 5, 5)], grid_blocks)
        occupied = occ.occupied
        plan = build_streaming_inference_plan(
            requested_bounds=((100, 900), (100, 900), (100, 900)),
            raw_volume_shape=(1000, 1000, 1000),
            minivol_size=MINIVOL,
            occupancy=occ,
        )
        run_by_grid = {c.grid_index: c.run for c in plan.cells}

        def cell_nonempty(g):
            for dz in (0, 1):
                for dy in (0, 1):
                    for dx in (0, 1):
                        bz, by, bx = g[0] + dz, g[1] + dy, g[2] + dx
                        if (
                            0 <= bz < grid_blocks[0]
                            and 0 <= by < grid_blocks[1]
                            and 0 <= bx < grid_blocks[2]
                            and occupied[bz, by, bx]
                        ):
                            return True
            return False

        for block in plan.blocks:
            covering = _covering_cell_grid_indices(block.grid_index)
            if any(cell_nonempty(g) for g in covering):
                # if the block sees any foreground, none of its cells are skipped
                self.assertTrue(all(run_by_grid[g] for g in covering))

    def test_all_skipped_block_has_no_data(self):
        grid_blocks = (10, 10, 10)
        occ = self._occupancy([(5, 5, 5)], grid_blocks)
        plan = build_streaming_inference_plan(
            requested_bounds=((100, 900), (100, 900), (100, 900)),
            raw_volume_shape=(1000, 1000, 1000),
            minivol_size=MINIVOL,
            occupancy=occ,
        )
        run_by_grid = {c.grid_index: c.run for c in plan.cells}
        for block in plan.blocks:
            covering = _covering_cell_grid_indices(block.grid_index)
            expected_any_run = any(run_by_grid[g] for g in covering)
            self.assertEqual(block.any_covering_run, expected_any_run)
        for chunk in plan.chunks:
            expected = any(plan.blocks[b].any_covering_run for b in chunk.block_ids)
            self.assertEqual(chunk.has_data, expected)


class StreamingInferencePlanValidationTest(unittest.TestCase):
    def test_requires_50_percent_overlap(self):
        with self.assertRaises(ValueError):
            build_streaming_inference_plan(
                requested_bounds=((0, 200), (0, 200), (0, 200)),
                raw_volume_shape=(200, 200, 200),
                minivol_size=200,
                stride=80,  # 200 != 2 * 80
            )

    def test_rejects_bbox_outside_volume(self):
        with self.assertRaises(ValueError):
            build_streaming_inference_plan(
                requested_bounds=((0, 300), (0, 100), (0, 100)),
                raw_volume_shape=(200, 200, 200),
            )

    def test_edge_bbox_produces_reflect_padding(self):
        plan = build_streaming_inference_plan(
            requested_bounds=((0, 300), (0, 300), (0, 300)),
            raw_volume_shape=(1000, 1000, 1000),
            minivol_size=MINIVOL,
        )
        # the low context cell (-1, -1, -1) must read real data and reflect-pad
        low = next(c for c in plan.cells if c.grid_index == (-1, -1, -1))
        self.assertTrue(low.extraction.has_padding)
        for a in range(3):
            self.assertEqual(low.extraction.pad_before[a], STRIDE)
            self.assertEqual(low.extraction.raw_slices[a], slice(0, STRIDE))


if __name__ == "__main__":
    unittest.main()
