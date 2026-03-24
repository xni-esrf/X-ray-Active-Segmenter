from __future__ import annotations

import unittest

try:
    import torch
except Exception:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]

from src.learning import (
    LearningBBoxEvalRuntime,
    LearningBBoxTensorBatch,
    LearningBBoxTensorEntry,
    build_eval_dataloader_runtimes_from_batch,
    build_eval_dataloader_runtimes_from_current_batch,
    build_inference_dataloader_runtime_from_entry,
    build_inference_dataloader_runtimes_from_batch,
    clear_current_learning_bbox_batch,
    clear_current_learning_eval_runtimes_by_box_id,
    compute_eval_label_values_from_batch,
    dispose_inference_runtime,
    dispose_inference_runtimes,
    get_current_learning_eval_runtimes_by_box_id,
    set_current_learning_bbox_entries,
)


@unittest.skipUnless(torch is not None, "PyTorch is not available")
class EvalDataLoaderBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_current_learning_bbox_batch()
        clear_current_learning_eval_runtimes_by_box_id()

    def tearDown(self) -> None:
        clear_current_learning_bbox_batch()
        clear_current_learning_eval_runtimes_by_box_id()

    def _entry(
        self,
        *,
        box_id: str,
        index: int,
        label: str,
        raw_value: float,
        seg_values: tuple[int, ...],
    ) -> LearningBBoxTensorEntry:
        raw = torch.full((4, 4, 4), float(raw_value), dtype=torch.float16, device="cpu")
        seg = torch.full((4, 4, 4), int(seg_values[0]), dtype=torch.int16, device="cpu")
        if len(seg_values) > 1:
            seg.view(-1)[0] = int(seg_values[1])
        if len(seg_values) > 2:
            seg.view(-1)[1] = int(seg_values[2])
        return LearningBBoxTensorEntry(
            box_id=box_id,
            index=index,
            label=label,
            raw_tensor=raw,
            segmentation_tensor=seg,
        )

    def test_compute_eval_label_values_uses_train_and_validation_and_excludes_mask(self) -> None:
        train_entry = self._entry(
            box_id="bbox_0001",
            index=1,
            label="train",
            raw_value=0,
            seg_values=(0, 5, -100),
        )
        validation_entry = self._entry(
            box_id="bbox_0002",
            index=2,
            label="validation",
            raw_value=1,
            seg_values=(1, 6),
        )
        inference_entry = self._entry(
            box_id="bbox_0003",
            index=3,
            label="inference",
            raw_value=2,
            seg_values=(9,),
        )
        batch = LearningBBoxTensorBatch(entries=(train_entry, validation_entry, inference_entry))

        labels = compute_eval_label_values_from_batch(batch)
        self.assertEqual(labels, (0, 1, 5, 6))

    def test_build_eval_runtimes_from_batch_builds_only_validation_in_order(self) -> None:
        train_entry = self._entry(
            box_id="bbox_0001",
            index=1,
            label="train",
            raw_value=0,
            seg_values=(0, 5),
        )
        first_validation = self._entry(
            box_id="bbox_0009",
            index=9,
            label="validation",
            raw_value=1,
            seg_values=(1,),
        )
        second_validation = self._entry(
            box_id="bbox_0002",
            index=2,
            label="validation",
            raw_value=2,
            seg_values=(6,),
        )
        batch = LearningBBoxTensorBatch(entries=(train_entry, first_validation, second_validation))

        captured = {"datasets": [], "dataloaders": [], "buffers": []}

        class _FakeDataset:
            def __init__(self, raw_tensor, *, minivol_size):
                self.vol = raw_tensor
                self.minivol_size = int(minivol_size)
                captured["datasets"].append(self)

        class _FakeLoader:
            def __init__(self, dataset, **kwargs):
                self.dataset = dataset
                self.kwargs = kwargs
                captured["dataloaders"].append(self)

        class _FakeBuffer:
            def __init__(self, ground_truth, volume_shape, label_values, *, minivol_size):
                self.ground_truth = ground_truth
                self.volume_shape = tuple(int(v) for v in volume_shape)
                self.label_values = tuple(int(v) for v in label_values)
                self.num_classes = int(len(self.label_values))
                self.minivol_size = int(minivol_size)
                captured["buffers"].append(self)

        runtimes = build_eval_dataloader_runtimes_from_batch(
            batch,
            minivol_size=200,
            batch_size=4,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
            dataset_factory=_FakeDataset,
            dataloader_factory=_FakeLoader,
            buffer_factory=_FakeBuffer,
            store_in_session=True,
        )

        self.assertEqual(tuple(runtimes.keys()), ("bbox_0009", "bbox_0002"))
        self.assertEqual(len(captured["datasets"]), 2)
        self.assertEqual(len(captured["dataloaders"]), 2)
        self.assertEqual(len(captured["buffers"]), 2)
        self.assertEqual(captured["buffers"][0].label_values, (0, 1, 5, 6))
        self.assertEqual(captured["buffers"][1].label_values, (0, 1, 5, 6))
        self.assertEqual(captured["dataloaders"][0].kwargs["batch_size"], 4)
        self.assertEqual(captured["dataloaders"][0].kwargs["num_workers"], 8)
        self.assertTrue(captured["dataloaders"][0].kwargs["pin_memory"])
        self.assertFalse(captured["dataloaders"][0].kwargs["drop_last"])

        stored = get_current_learning_eval_runtimes_by_box_id()
        self.assertEqual(tuple(stored.keys()), ("bbox_0009", "bbox_0002"))

    def test_build_eval_runtimes_from_batch_rejects_missing_validation_entries(self) -> None:
        train_entry = self._entry(
            box_id="bbox_0001",
            index=1,
            label="train",
            raw_value=0,
            seg_values=(0,),
        )
        batch = LearningBBoxTensorBatch(entries=(train_entry,))

        with self.assertRaisesRegex(ValueError, "No validation bounding boxes"):
            build_eval_dataloader_runtimes_from_batch(
                batch,
                dataset_factory=lambda *_args, **_kwargs: object(),
                dataloader_factory=lambda *_args, **_kwargs: object(),
                buffer_factory=lambda *_args, **_kwargs: object(),
            )

    def test_build_eval_runtimes_from_batch_rejects_mismatched_num_classes(self) -> None:
        train_entry = self._entry(
            box_id="bbox_0001",
            index=1,
            label="train",
            raw_value=0,
            seg_values=(0, 5),
        )
        first_validation = self._entry(
            box_id="bbox_0009",
            index=9,
            label="validation",
            raw_value=1,
            seg_values=(1,),
        )
        second_validation = self._entry(
            box_id="bbox_0002",
            index=2,
            label="validation",
            raw_value=2,
            seg_values=(6,),
        )
        batch = LearningBBoxTensorBatch(entries=(train_entry, first_validation, second_validation))

        class _FakeIterator:
            def __init__(self) -> None:
                self.shutdown_calls = 0

            def _shutdown_workers(self) -> None:
                self.shutdown_calls += 1

        class _FakeLoader:
            def __init__(self, dataset, **kwargs):
                del dataset, kwargs
                self._iterator = _FakeIterator()
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        class _FakeBuffer:
            def __init__(self, num_classes: int) -> None:
                self.num_classes = int(num_classes)
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        created_loaders = []
        created_buffers = []
        buffer_calls = {"count": 0}

        def dataset_factory(raw, *, minivol_size):
            return type("_DS", (), {"vol": raw, "minivol_size": int(minivol_size)})()

        def dataloader_factory(dataset, **kwargs):
            loader = _FakeLoader(dataset, **kwargs)
            created_loaders.append(loader)
            return loader

        def buffer_factory(ground_truth, volume_shape, label_values, *, minivol_size):
            del ground_truth, volume_shape, minivol_size
            buffer_calls["count"] += 1
            base_num_classes = int(len(tuple(label_values)))
            if buffer_calls["count"] == 2:
                base_num_classes += 1
            buffer = _FakeBuffer(base_num_classes)
            created_buffers.append(buffer)
            return buffer

        with self.assertRaisesRegex(ValueError, "must share the same num_classes"):
            build_eval_dataloader_runtimes_from_batch(
                batch,
                dataset_factory=dataset_factory,
                dataloader_factory=dataloader_factory,
                buffer_factory=buffer_factory,
                store_in_session=True,
            )

        self.assertEqual(len(created_loaders), 2)
        self.assertEqual(len(created_buffers), 2)
        self.assertEqual(created_loaders[0].close_calls, 1)
        self.assertEqual(created_loaders[1].close_calls, 1)
        self.assertEqual(created_buffers[0].close_calls, 1)
        self.assertEqual(created_buffers[1].close_calls, 1)
        self.assertEqual(created_loaders[0]._iterator.shutdown_calls, 1)
        self.assertEqual(created_loaders[1]._iterator.shutdown_calls, 1)
        self.assertEqual(get_current_learning_eval_runtimes_by_box_id(), {})

    def test_build_eval_runtimes_from_batch_rejects_buffer_without_num_classes(self) -> None:
        train_entry = self._entry(
            box_id="bbox_0001",
            index=1,
            label="train",
            raw_value=0,
            seg_values=(0, 5),
        )
        validation_entry = self._entry(
            box_id="bbox_0009",
            index=9,
            label="validation",
            raw_value=1,
            seg_values=(1,),
        )
        batch = LearningBBoxTensorBatch(entries=(train_entry, validation_entry))

        class _FakeIterator:
            def __init__(self) -> None:
                self.shutdown_calls = 0

            def _shutdown_workers(self) -> None:
                self.shutdown_calls += 1

        class _FakeLoader:
            def __init__(self, dataset, **kwargs):
                del dataset, kwargs
                self._iterator = _FakeIterator()
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        class _FakeBuffer:
            def __init__(self) -> None:
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        created_loaders = []
        created_buffers = []

        def dataset_factory(raw, *, minivol_size):
            return type("_DS", (), {"vol": raw, "minivol_size": int(minivol_size)})()

        def dataloader_factory(dataset, **kwargs):
            loader = _FakeLoader(dataset, **kwargs)
            created_loaders.append(loader)
            return loader

        def buffer_factory(ground_truth, volume_shape, label_values, *, minivol_size):
            del ground_truth, volume_shape, label_values, minivol_size
            buffer = _FakeBuffer()
            created_buffers.append(buffer)
            return buffer

        with self.assertRaisesRegex(ValueError, "does not expose 'num_classes'"):
            build_eval_dataloader_runtimes_from_batch(
                batch,
                dataset_factory=dataset_factory,
                dataloader_factory=dataloader_factory,
                buffer_factory=buffer_factory,
                store_in_session=True,
            )

        self.assertEqual(len(created_loaders), 1)
        self.assertEqual(len(created_buffers), 1)
        self.assertEqual(created_loaders[0].close_calls, 1)
        self.assertEqual(created_buffers[0].close_calls, 1)
        self.assertEqual(created_loaders[0]._iterator.shutdown_calls, 1)
        self.assertEqual(get_current_learning_eval_runtimes_by_box_id(), {})

    def test_build_eval_runtimes_from_batch_disposes_partial_runtimes_on_failure(self) -> None:
        train_entry = self._entry(
            box_id="bbox_0001",
            index=1,
            label="train",
            raw_value=0,
            seg_values=(0, 5),
        )
        first_validation = self._entry(
            box_id="bbox_0009",
            index=9,
            label="validation",
            raw_value=1,
            seg_values=(1,),
        )
        second_validation = self._entry(
            box_id="bbox_0002",
            index=2,
            label="validation",
            raw_value=2,
            seg_values=(6,),
        )
        batch = LearningBBoxTensorBatch(entries=(train_entry, first_validation, second_validation))

        class _FakeIterator:
            def __init__(self) -> None:
                self.shutdown_calls = 0

            def _shutdown_workers(self) -> None:
                self.shutdown_calls += 1

        class _FakeLoader:
            def __init__(self, dataset, **kwargs):
                del dataset, kwargs
                self._iterator = _FakeIterator()
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        class _FakeBuffer:
            def __init__(self, ground_truth, volume_shape, label_values, *, minivol_size):
                self.num_classes = int(len(tuple(label_values)))
                del ground_truth, volume_shape, minivol_size
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        created_datasets = []
        created_loaders = []
        created_buffers = []
        dataset_calls = {"count": 0}

        class _FakeDataset:
            def __init__(self, raw_tensor, *, minivol_size):
                del raw_tensor
                self.minivol_size = int(minivol_size)
                self.vol = torch.zeros((4, 4, 4), dtype=torch.float16)
                created_datasets.append(self)

        def dataset_factory(raw_tensor, *, minivol_size):
            dataset_calls["count"] += 1
            if dataset_calls["count"] == 2:
                raise RuntimeError("dataset boom")
            return _FakeDataset(raw_tensor, minivol_size=minivol_size)

        def dataloader_factory(dataset, **kwargs):
            loader = _FakeLoader(dataset, **kwargs)
            created_loaders.append(loader)
            return loader

        def buffer_factory(ground_truth, volume_shape, label_values, *, minivol_size):
            buffer = _FakeBuffer(
                ground_truth,
                volume_shape,
                label_values,
                minivol_size=minivol_size,
            )
            created_buffers.append(buffer)
            return buffer

        with self.assertRaisesRegex(RuntimeError, "dataset boom"):
            build_eval_dataloader_runtimes_from_batch(
                batch,
                dataset_factory=dataset_factory,
                dataloader_factory=dataloader_factory,
                buffer_factory=buffer_factory,
                store_in_session=True,
            )

        self.assertEqual(dataset_calls["count"], 2)
        self.assertEqual(len(created_datasets), 1)
        self.assertEqual(len(created_loaders), 1)
        self.assertEqual(len(created_buffers), 1)
        self.assertEqual(created_loaders[0].close_calls, 1)
        self.assertEqual(created_buffers[0].close_calls, 1)
        self.assertEqual(created_loaders[0]._iterator.shutdown_calls, 1)
        self.assertEqual(get_current_learning_eval_runtimes_by_box_id(), {})

    def test_build_eval_runtimes_from_current_batch_uses_session_batch(self) -> None:
        validation_entry = self._entry(
            box_id="bbox_0011",
            index=11,
            label="validation",
            raw_value=1,
            seg_values=(0, 1),
        )
        set_current_learning_bbox_entries((validation_entry,))

        class _FakeBuffer:
            def __init__(self, gt, shape, labels, *, minivol_size):
                del gt, shape, minivol_size
                self.num_classes = int(len(tuple(labels)))

        runtimes = build_eval_dataloader_runtimes_from_current_batch(
            dataset_factory=lambda raw, *, minivol_size: type(
                "_DS", (), {"vol": raw, "minivol_size": minivol_size}
            )(),
            dataloader_factory=lambda dataset, **kwargs: (dataset, kwargs),
            buffer_factory=_FakeBuffer,
            store_in_session=False,
        )
        self.assertEqual(tuple(runtimes.keys()), ("bbox_0011",))

    def test_build_inference_runtime_from_entry_builds_loader_and_buffer(self) -> None:
        inference_entry = self._entry(
            box_id="bbox_0013",
            index=13,
            label="inference",
            raw_value=7,
            seg_values=(0,),
        )

        captured = {}

        class _FakeDataset:
            def __init__(self, raw_tensor, *, minivol_size):
                self.vol = raw_tensor
                captured["dataset_minivol_size"] = int(minivol_size)

        class _FakeLoader:
            def __init__(self, dataset, **kwargs):
                self.dataset = dataset
                self.kwargs = dict(kwargs)
                captured["loader_kwargs"] = dict(kwargs)

        class _FakeBuffer:
            def __init__(self, volume_shape, label_values, *, minivol_size):
                self.volume_shape = tuple(int(v) for v in volume_shape)
                self.label_values = tuple(int(v) for v in label_values)
                self.minivol_size = int(minivol_size)
                self.num_classes = int(len(self.label_values))

        runtime = build_inference_dataloader_runtime_from_entry(
            inference_entry,
            label_values=(0, 5, 6),
            minivol_size=200,
            batch_size=4,
            num_workers=3,
            pin_memory=True,
            drop_last=False,
            dataset_factory=_FakeDataset,
            dataloader_factory=_FakeLoader,
            buffer_factory=_FakeBuffer,
        )

        self.assertEqual(runtime.box_id, "bbox_0013")
        self.assertEqual(captured["dataset_minivol_size"], 200)
        self.assertEqual(captured["loader_kwargs"]["batch_size"], 4)
        self.assertEqual(captured["loader_kwargs"]["num_workers"], 3)
        self.assertTrue(captured["loader_kwargs"]["pin_memory"])
        self.assertFalse(captured["loader_kwargs"]["drop_last"])
        self.assertEqual(runtime.buffer.label_values, (0, 5, 6))
        self.assertEqual(runtime.buffer.volume_shape, (4, 4, 4))
        self.assertEqual(runtime.buffer.minivol_size, 200)

    def test_build_inference_runtimes_from_batch_builds_only_inference_entries_in_order(self) -> None:
        train_entry = self._entry(
            box_id="bbox_0001",
            index=1,
            label="train",
            raw_value=0,
            seg_values=(0,),
        )
        first_inference = self._entry(
            box_id="bbox_0003",
            index=3,
            label="inference",
            raw_value=1,
            seg_values=(0,),
        )
        validation_entry = self._entry(
            box_id="bbox_0002",
            index=2,
            label="validation",
            raw_value=2,
            seg_values=(0,),
        )
        second_inference = self._entry(
            box_id="bbox_0010",
            index=10,
            label="inference",
            raw_value=3,
            seg_values=(0,),
        )
        batch = LearningBBoxTensorBatch(
            entries=(train_entry, first_inference, validation_entry, second_inference)
        )

        class _FakeDataset:
            def __init__(self, raw_tensor, *, minivol_size):
                self.vol = raw_tensor
                self.minivol_size = int(minivol_size)

        class _FakeLoader:
            def __init__(self, dataset, **kwargs):
                self.dataset = dataset
                self.kwargs = dict(kwargs)

        class _FakeBuffer:
            def __init__(self, volume_shape, label_values, *, minivol_size):
                self.volume_shape = tuple(int(v) for v in volume_shape)
                self.label_values = tuple(int(v) for v in label_values)
                self.minivol_size = int(minivol_size)
                self.num_classes = int(len(self.label_values))

        runtimes = build_inference_dataloader_runtimes_from_batch(
            batch,
            label_values=(0, 5, 6),
            dataset_factory=_FakeDataset,
            dataloader_factory=_FakeLoader,
            buffer_factory=_FakeBuffer,
        )

        self.assertEqual(tuple(runtimes.keys()), ("bbox_0003", "bbox_0010"))
        self.assertEqual(runtimes["bbox_0003"].buffer.label_values, (0, 5, 6))
        self.assertEqual(runtimes["bbox_0010"].buffer.label_values, (0, 5, 6))
        self.assertEqual(get_current_learning_eval_runtimes_by_box_id(), {})

    def test_build_inference_runtimes_from_batch_rejects_missing_inference_entries(self) -> None:
        train_entry = self._entry(
            box_id="bbox_0001",
            index=1,
            label="train",
            raw_value=0,
            seg_values=(0,),
        )
        validation_entry = self._entry(
            box_id="bbox_0002",
            index=2,
            label="validation",
            raw_value=1,
            seg_values=(0,),
        )
        batch = LearningBBoxTensorBatch(entries=(train_entry, validation_entry))

        with self.assertRaisesRegex(ValueError, "No inference bounding boxes"):
            build_inference_dataloader_runtimes_from_batch(
                batch,
                label_values=(0, 1),
                dataset_factory=lambda *_args, **_kwargs: object(),
                dataloader_factory=lambda *_args, **_kwargs: object(),
                buffer_factory=lambda *_args, **_kwargs: object(),
            )

    def test_dispose_inference_runtime_reports_cleanup_errors(self) -> None:
        class _Iterator:
            def _shutdown_workers(self) -> None:
                raise RuntimeError("iterator shutdown boom")

        class _Loader:
            def __init__(self) -> None:
                self._iterator = _Iterator()

            def close(self) -> None:
                raise RuntimeError("loader close boom")

        class _Buffer:
            def close(self) -> None:
                raise RuntimeError("buffer close boom")

        runtime = LearningBBoxEvalRuntime(
            box_id="bbox_0007",
            dataloader=_Loader(),
            buffer=_Buffer(),
        )

        errors = dispose_inference_runtime(runtime)
        self.assertGreaterEqual(len(errors), 3)
        self.assertTrue(any("iterator._shutdown_workers" in error for error in errors))
        self.assertTrue(any("dataloader.close()" in error for error in errors))
        self.assertTrue(any("buffer.close()" in error for error in errors))

    def test_dispose_inference_runtimes_reports_failures_by_box_id(self) -> None:
        class _Iterator:
            def _shutdown_workers(self) -> None:
                return

        class _GoodLoader:
            def __init__(self) -> None:
                self._iterator = _Iterator()

            def close(self) -> None:
                return

        class _BadLoader:
            def __init__(self) -> None:
                self._iterator = _Iterator()

            def close(self) -> None:
                raise RuntimeError("bad loader close")

        class _GoodBuffer:
            def close(self) -> None:
                return

        class _BadBuffer:
            def close(self) -> None:
                raise RuntimeError("bad buffer close")

        good_runtime = LearningBBoxEvalRuntime(
            box_id="bbox_0001",
            dataloader=_GoodLoader(),
            buffer=_GoodBuffer(),
        )
        bad_runtime = LearningBBoxEvalRuntime(
            box_id="bbox_0002",
            dataloader=_BadLoader(),
            buffer=_BadBuffer(),
        )

        cleanup_errors = dispose_inference_runtimes(
            {"bbox_0001": good_runtime, "bbox_0002": bad_runtime}
        )

        self.assertEqual(tuple(cleanup_errors.keys()), ("bbox_0002",))
        self.assertTrue(any("dataloader.close()" in err for err in cleanup_errors["bbox_0002"]))
        self.assertTrue(any("buffer.close()" in err for err in cleanup_errors["bbox_0002"]))


if __name__ == "__main__":
    unittest.main()
