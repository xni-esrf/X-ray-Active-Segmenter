from __future__ import annotations

import unittest

try:
    import torch
except Exception:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]

from src.learning import (
    LearningBBoxTensorBatch,
    LearningBBoxTensorEntry,
    build_learning_dataloader_from_batch,
    build_learning_dataloader_from_current_batch,
    clear_current_learning_bbox_batch,
    clear_current_learning_dataloader_runtime,
    get_current_learning_dataloader_runtime,
    set_current_learning_bbox_entries,
)


@unittest.skipUnless(torch is not None, "PyTorch is not available")
class LearningDataLoaderBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_current_learning_bbox_batch()
        clear_current_learning_dataloader_runtime()

    def tearDown(self) -> None:
        clear_current_learning_bbox_batch()
        clear_current_learning_dataloader_runtime()

    def _entry(
        self,
        *,
        box_id: str,
        index: int,
        label: str,
        fill_raw: float,
        fill_seg: int,
    ) -> LearningBBoxTensorEntry:
        raw = torch.full((4, 4, 4), float(fill_raw), dtype=torch.float16, device="cpu")
        seg = torch.full((4, 4, 4), int(fill_seg), dtype=torch.int16, device="cpu")
        return LearningBBoxTensorEntry(
            box_id=box_id,
            index=index,
            label=label,
            raw_tensor=raw,
            segmentation_tensor=seg,
        )

    def test_build_from_batch_keeps_only_train_entries_in_order(self) -> None:
        first = self._entry(
            box_id="bbox_0010",
            index=10,
            label="validation",
            fill_raw=0,
            fill_seg=0,
        )
        second = self._entry(
            box_id="bbox_0007",
            index=7,
            label="train",
            fill_raw=1,
            fill_seg=11,
        )
        third = self._entry(
            box_id="bbox_0042",
            index=42,
            label="train",
            fill_raw=2,
            fill_seg=22,
        )
        batch = LearningBBoxTensorBatch(entries=(first, second, third))

        captured = {}

        class _FakeDataset:
            def __init__(self, tensor_pairs, *, minivol_size, minivol_per_epoch):
                captured["tensor_pairs"] = tuple(tensor_pairs)
                captured["minivol_size"] = int(minivol_size)
                captured["minivol_per_epoch"] = int(minivol_per_epoch)
                self.weights = [5, 9]
                self._length = 17

            def __len__(self):
                return int(self._length)

        class _FakeSampler:
            def __init__(self, weights, num_samples):
                captured["sampler_weights"] = tuple(weights)
                captured["sampler_num_samples"] = int(num_samples)
                self.weights = tuple(weights)
                self.num_samples = int(num_samples)

        class _FakeLoader:
            def __init__(
                self,
                dataset,
                *,
                batch_size,
                sampler,
                num_workers,
                pin_memory,
                drop_last,
            ):
                captured["loader_dataset"] = dataset
                captured["loader_batch_size"] = int(batch_size)
                captured["loader_sampler"] = sampler
                captured["loader_num_workers"] = int(num_workers)
                captured["loader_pin_memory"] = bool(pin_memory)
                captured["loader_drop_last"] = bool(drop_last)

        runtime = build_learning_dataloader_from_batch(
            batch,
            minivol_size=200,
            minivol_per_epoch=512,
            batch_size=4,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            dataset_factory=_FakeDataset,
            sampler_factory=_FakeSampler,
            dataloader_factory=_FakeLoader,
            store_in_session=True,
        )

        self.assertEqual(runtime.train_box_ids, ("bbox_0007", "bbox_0042"))
        self.assertEqual(captured["minivol_size"], 200)
        self.assertEqual(captured["minivol_per_epoch"], 512)
        self.assertEqual(len(captured["tensor_pairs"]), 2)
        self.assertIs(captured["tensor_pairs"][0][0], second.raw_tensor)
        self.assertIs(captured["tensor_pairs"][0][1], second.segmentation_tensor)
        self.assertIs(captured["tensor_pairs"][1][0], third.raw_tensor)
        self.assertIs(captured["tensor_pairs"][1][1], third.segmentation_tensor)
        self.assertEqual(captured["sampler_weights"], (5, 9))
        self.assertEqual(captured["sampler_num_samples"], 17)
        self.assertEqual(captured["loader_batch_size"], 4)
        self.assertEqual(captured["loader_num_workers"], 8)
        self.assertTrue(captured["loader_pin_memory"])
        self.assertTrue(captured["loader_drop_last"])

        stored = get_current_learning_dataloader_runtime()
        self.assertIs(stored, runtime)

    def test_build_from_batch_rejects_missing_train_entries(self) -> None:
        batch = LearningBBoxTensorBatch(
            entries=(
                self._entry(
                    box_id="bbox_0003",
                    index=3,
                    label="validation",
                    fill_raw=1,
                    fill_seg=1,
                ),
                self._entry(
                    box_id="bbox_0004",
                    index=4,
                    label="inference",
                    fill_raw=2,
                    fill_seg=2,
                ),
            )
        )

        with self.assertRaisesRegex(ValueError, "No training bounding boxes labeled 'train'"):
            build_learning_dataloader_from_batch(
                batch,
                minivol_size=200,
                minivol_per_epoch=64,
                dataset_factory=lambda *_args, **_kwargs: object(),
                sampler_factory=lambda *_args, **_kwargs: object(),
                dataloader_factory=lambda *_args, **_kwargs: object(),
            )

    def test_build_from_current_batch_rejects_empty_session(self) -> None:
        with self.assertRaisesRegex(ValueError, "No learning tensor batch is available"):
            build_learning_dataloader_from_current_batch(
                minivol_size=200,
                minivol_per_epoch=64,
                dataset_factory=lambda *_args, **_kwargs: object(),
                sampler_factory=lambda *_args, **_kwargs: object(),
                dataloader_factory=lambda *_args, **_kwargs: object(),
            )

    def test_build_from_current_batch_uses_session_batch(self) -> None:
        train_entry = self._entry(
            box_id="bbox_0008",
            index=8,
            label="train",
            fill_raw=3,
            fill_seg=5,
        )
        set_current_learning_bbox_entries((train_entry,))

        class _Dataset:
            def __init__(self, tensor_pairs, *, minivol_size, minivol_per_epoch):
                del minivol_size, minivol_per_epoch
                self.pairs = tuple(tensor_pairs)
                self.weights = [1]

            def __len__(self):
                return 10

        runtime = build_learning_dataloader_from_current_batch(
            minivol_size=200,
            minivol_per_epoch=64,
            dataset_factory=_Dataset,
            sampler_factory=lambda weights, size: ("sampler", tuple(weights), int(size)),
            dataloader_factory=lambda dataset, **kwargs: ("loader", dataset, kwargs),
            store_in_session=False,
        )

        self.assertEqual(runtime.train_box_ids, ("bbox_0008",))
        self.assertEqual(runtime.train_count, 1)


if __name__ == "__main__":
    unittest.main()
