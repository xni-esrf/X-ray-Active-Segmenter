from __future__ import annotations

import unittest

try:
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
except Exception:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]

from src.learning import TrainBBoxDataset


@unittest.skipUnless(torch is not None, "PyTorch is not available")
class TrainBBoxDatasetTests(unittest.TestCase):
    def _make_pair(
        self,
        *,
        raw_value_offset: int,
        seg_value: int,
        shape: tuple[int, int, int] = (8, 8, 8),
    ):
        n = int(shape[0] * shape[1] * shape[2])
        raw = (
            torch.arange(n, dtype=torch.float32).reshape(shape)
            + float(raw_value_offset)
        ).to(dtype=torch.float16)
        seg = torch.full(shape, int(seg_value), dtype=torch.int16)
        return raw, seg

    def test_init_sets_weights_and_len(self) -> None:
        first = self._make_pair(raw_value_offset=0, seg_value=1, shape=(8, 8, 8))
        second = self._make_pair(raw_value_offset=10, seg_value=2, shape=(10, 9, 8))
        dataset = TrainBBoxDataset(
            [first, second],
            minivol_size=8,
            minivol_per_epoch=123,
            contr_bright_factors=(1.0, 0.0),
        )

        self.assertEqual(dataset.volume_count, 2)
        self.assertEqual(len(dataset), 123)
        self.assertEqual(
            dataset.weights,
            [
                int(8 * 8 * 8),
                int(10 * 9 * 8),
            ],
        )

    def test_getitem_uses_idx_as_volume_index(self) -> None:
        first = self._make_pair(raw_value_offset=0, seg_value=7)
        second = self._make_pair(raw_value_offset=100, seg_value=13)
        dataset = TrainBBoxDataset(
            [first, second],
            minivol_size=8,
            minivol_per_epoch=8,
            contr_bright_factors=(1.0, 0.0),
        )
        dataset.geom_transform = lambda raw, annot: (raw, annot)  # type: ignore[method-assign]
        dataset.elastic_transform_3d = lambda raw, annot: (raw, annot)  # type: ignore[method-assign]

        raw_0, seg_0 = dataset[0]
        raw_1, seg_1 = dataset[1]

        self.assertEqual(tuple(raw_0.shape), (1, 8, 8, 8))
        self.assertEqual(tuple(raw_1.shape), (1, 8, 8, 8))
        self.assertEqual(raw_0.dtype, torch.float32)
        self.assertEqual(seg_0.dtype, torch.long)
        self.assertTrue(torch.all(seg_0 == 7))
        self.assertTrue(torch.all(seg_1 == 13))

    def test_getitem_raises_when_idx_out_of_range(self) -> None:
        pair = self._make_pair(raw_value_offset=0, seg_value=1)
        dataset = TrainBBoxDataset(
            [pair],
            minivol_size=8,
            minivol_per_epoch=8,
            contr_bright_factors=(1.0, 0.0),
        )

        with self.assertRaises(IndexError):
            _ = dataset[1]

    def test_rejects_minivol_larger_than_source_volume(self) -> None:
        pair = self._make_pair(raw_value_offset=0, seg_value=1, shape=(7, 8, 8))
        with self.assertRaises(ValueError):
            TrainBBoxDataset(
                [pair],
                minivol_size=8,
                minivol_per_epoch=8,
                contr_bright_factors=(1.0, 0.0),
            )

    def test_weighted_sampler_uses_volume_indices(self) -> None:
        first = self._make_pair(raw_value_offset=0, seg_value=4)
        second = self._make_pair(raw_value_offset=50, seg_value=9)
        dataset = TrainBBoxDataset(
            [first, second],
            minivol_size=8,
            minivol_per_epoch=12,
            contr_bright_factors=(1.0, 0.0),
        )
        dataset.geom_transform = lambda raw, annot: (raw, annot)  # type: ignore[method-assign]
        dataset.elastic_transform_3d = lambda raw, annot: (raw, annot)  # type: ignore[method-assign]

        sampler = WeightedRandomSampler(dataset.weights, len(dataset))
        loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )

        seen_values = set()
        for _, seg in loader:
            value = int(seg.flatten()[0].item())
            self.assertIn(value, {4, 9})
            seen_values.add(value)
        self.assertTrue(seen_values)


if __name__ == "__main__":
    unittest.main()
