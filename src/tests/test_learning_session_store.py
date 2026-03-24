from __future__ import annotations

import unittest

try:
    import torch
except Exception:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]

from src.learning import (
    LearningBBoxDataLoaderRuntime,
    LearningBBoxEvalRuntime,
    LearningModelRuntime,
    LearningBBoxTensorBatch,
    LearningBBoxTensorEntry,
    clear_current_learning_bbox_batch,
    clear_current_learning_dataloader_runtime,
    clear_current_learning_eval_runtimes_by_box_id,
    clear_current_learning_model_runtime,
    get_current_learning_bbox_batch,
    get_current_learning_dataloader_runtime,
    get_current_learning_eval_runtimes_by_box_id,
    get_current_learning_model_runtime,
    set_current_learning_bbox_batch,
    set_current_learning_dataloader_components,
    set_current_learning_dataloader_class_weights,
    set_current_learning_dataloader_runtime,
    set_current_learning_eval_runtime_components_by_box_id,
    set_current_learning_eval_runtimes_by_box_id,
    set_current_learning_bbox_entries,
    set_current_learning_model_components,
    set_current_learning_model_runtime,
)


@unittest.skipUnless(torch is not None, "PyTorch is not available")
class LearningSessionStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_current_learning_bbox_batch()
        clear_current_learning_dataloader_runtime()
        clear_current_learning_eval_runtimes_by_box_id()

    def tearDown(self) -> None:
        clear_current_learning_bbox_batch()
        clear_current_learning_dataloader_runtime()
        clear_current_learning_eval_runtimes_by_box_id()

    def _entry(self, *, box_id: str, index: int, label: str) -> LearningBBoxTensorEntry:
        raw_tensor = torch.zeros((2, 3, 4), dtype=torch.float16, device="cpu")
        seg_tensor = torch.full((2, 3, 4), -100, dtype=torch.int16, device="cpu")
        return LearningBBoxTensorEntry(
            box_id=box_id,
            index=index,
            label=label,
            raw_tensor=raw_tensor,
            segmentation_tensor=seg_tensor,
        )

    def test_set_and_get_current_batch(self) -> None:
        entry = self._entry(box_id="bbox_0007", index=7, label="train")
        batch = LearningBBoxTensorBatch(entries=(entry,))

        returned = set_current_learning_bbox_batch(batch)
        current = get_current_learning_bbox_batch()

        self.assertIs(returned, batch)
        self.assertIs(current, batch)
        self.assertEqual(current.size, 1)
        self.assertEqual(current.box_ids, ("bbox_0007",))
        self.assertEqual(current.entries[0].label, "train")

    def test_set_current_learning_bbox_entries_replaces_previous_batch(self) -> None:
        first = self._entry(box_id="bbox_0001", index=1, label="train")
        second = self._entry(box_id="bbox_0002", index=2, label="validation")
        third = self._entry(box_id="bbox_0003", index=3, label="inference")

        set_current_learning_bbox_entries((first, second))
        updated = set_current_learning_bbox_entries((third,))
        current = get_current_learning_bbox_batch()

        self.assertIsNotNone(current)
        self.assertIs(current, updated)
        self.assertEqual(current.size, 1)
        self.assertEqual(current.box_ids, ("bbox_0003",))
        self.assertEqual(current.entries[0].label, "inference")

    def test_set_current_learning_bbox_entries_preserves_order(self) -> None:
        first = self._entry(box_id="bbox_0011", index=11, label="validation")
        second = self._entry(box_id="bbox_0002", index=2, label="train")
        third = self._entry(box_id="bbox_0042", index=42, label="inference")

        set_current_learning_bbox_entries((first, second, third))
        current = get_current_learning_bbox_batch()

        self.assertIsNotNone(current)
        self.assertEqual(
            current.box_ids,
            ("bbox_0011", "bbox_0002", "bbox_0042"),
        )
        self.assertEqual(
            tuple(entry.label for entry in current.entries),
            ("validation", "train", "inference"),
        )

    def test_clear_current_batch(self) -> None:
        entry = self._entry(box_id="bbox_0001", index=1, label="train")
        set_current_learning_bbox_entries((entry,))
        self.assertIsNotNone(get_current_learning_bbox_batch())

        clear_current_learning_bbox_batch()

        self.assertIsNone(get_current_learning_bbox_batch())

    def test_entry_rejects_non_tensor_and_non_cpu_inputs(self) -> None:
        raw = torch.zeros((1, 1, 1), dtype=torch.float16, device="cpu")
        seg = torch.zeros((1, 1, 1), dtype=torch.int16, device="cpu")

        with self.assertRaises(TypeError):
            LearningBBoxTensorEntry(
                box_id="bbox_0001",
                index=1,
                label="train",
                raw_tensor=raw.numpy(),
                segmentation_tensor=seg,
            )

        if torch.cuda.is_available():
            with self.assertRaises(ValueError):
                LearningBBoxTensorEntry(
                    box_id="bbox_0001",
                    index=1,
                    label="train",
                    raw_tensor=raw.to("cuda"),
                    segmentation_tensor=seg,
                )


class LearningDataLoaderRuntimeStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_current_learning_dataloader_runtime()
        clear_current_learning_eval_runtimes_by_box_id()

    def tearDown(self) -> None:
        clear_current_learning_dataloader_runtime()
        clear_current_learning_eval_runtimes_by_box_id()

    def test_set_and_get_current_runtime(self) -> None:
        runtime = LearningBBoxDataLoaderRuntime(
            dataset=object(),
            sampler=object(),
            dataloader=object(),
            train_box_ids=("bbox_0007", "bbox_0011"),
            minivol_size=200,
            minivol_per_epoch=1024,
            batch_size=4,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

        returned = set_current_learning_dataloader_runtime(runtime)
        current = get_current_learning_dataloader_runtime()

        self.assertIs(returned, runtime)
        self.assertIs(current, runtime)
        self.assertEqual(current.train_box_ids, ("bbox_0007", "bbox_0011"))
        self.assertEqual(current.train_count, 2)
        self.assertEqual(current.batch_size, 4)
        self.assertEqual(current.num_workers, 8)
        self.assertTrue(current.pin_memory)
        self.assertTrue(current.drop_last)

    def test_set_current_learning_dataloader_components_replaces_previous_runtime(self) -> None:
        set_current_learning_dataloader_components(
            dataset=object(),
            sampler=object(),
            dataloader=object(),
            train_box_ids=("bbox_0001", "bbox_0002"),
            batch_size=4,
        )
        second = set_current_learning_dataloader_components(
            dataset=object(),
            sampler=object(),
            dataloader=object(),
            train_box_ids=("bbox_0042",),
            batch_size=8,
        )
        current = get_current_learning_dataloader_runtime()

        self.assertIsNotNone(current)
        self.assertIs(current, second)
        self.assertEqual(current.train_box_ids, ("bbox_0042",))
        self.assertEqual(current.batch_size, 8)

    def test_clear_runtime(self) -> None:
        set_current_learning_dataloader_components(
            dataset=object(),
            sampler=object(),
            dataloader=object(),
            train_box_ids=("bbox_0001",),
        )
        self.assertIsNotNone(get_current_learning_dataloader_runtime())

        clear_current_learning_dataloader_runtime()

        self.assertIsNone(get_current_learning_dataloader_runtime())

    def test_replacing_runtime_disposes_previous_components(self) -> None:
        class _FakeIterator:
            def __init__(self) -> None:
                self.shutdown_calls = 0

            def _shutdown_workers(self) -> None:
                self.shutdown_calls += 1

        class _FakeDataLoader:
            def __init__(self) -> None:
                self._iterator = _FakeIterator()
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        class _FakeClosable:
            def __init__(self) -> None:
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        first_dataset = _FakeClosable()
        first_sampler = _FakeClosable()
        first_loader = _FakeDataLoader()
        first_iterator = first_loader._iterator
        first_runtime = LearningBBoxDataLoaderRuntime(
            dataset=first_dataset,
            sampler=first_sampler,
            dataloader=first_loader,
            train_box_ids=("bbox_0001",),
        )
        set_current_learning_dataloader_runtime(first_runtime)

        second_runtime = LearningBBoxDataLoaderRuntime(
            dataset=object(),
            sampler=object(),
            dataloader=object(),
            train_box_ids=("bbox_0002",),
        )
        set_current_learning_dataloader_runtime(second_runtime)

        self.assertEqual(first_iterator.shutdown_calls, 1)
        self.assertEqual(first_loader.close_calls, 1)
        self.assertEqual(first_dataset.close_calls, 1)
        self.assertEqual(first_sampler.close_calls, 1)

    def test_clear_runtime_disposes_current_components(self) -> None:
        class _FakeIterator:
            def __init__(self) -> None:
                self.shutdown_calls = 0

            def _shutdown_workers(self) -> None:
                self.shutdown_calls += 1

        class _FakeDataLoader:
            def __init__(self) -> None:
                self._iterator = _FakeIterator()
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        class _FakeClosable:
            def __init__(self) -> None:
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        dataset = _FakeClosable()
        sampler = _FakeClosable()
        loader = _FakeDataLoader()
        iterator_ref = loader._iterator
        runtime = LearningBBoxDataLoaderRuntime(
            dataset=dataset,
            sampler=sampler,
            dataloader=loader,
            train_box_ids=("bbox_0001",),
        )
        set_current_learning_dataloader_runtime(runtime)

        clear_current_learning_dataloader_runtime()

        self.assertEqual(iterator_ref.shutdown_calls, 1)
        self.assertEqual(loader.close_calls, 1)
        self.assertEqual(dataset.close_calls, 1)
        self.assertEqual(sampler.close_calls, 1)

    def test_runtime_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            LearningBBoxDataLoaderRuntime(
                dataset=None,  # type: ignore[arg-type]
                sampler=object(),
                dataloader=object(),
                train_box_ids=("bbox_0001",),
            )

        with self.assertRaises(ValueError):
            LearningBBoxDataLoaderRuntime(
                dataset=object(),
                sampler=object(),
                dataloader=object(),
                train_box_ids=("bbox_0001", "bbox_0001"),
            )

        with self.assertRaises(ValueError):
            LearningBBoxDataLoaderRuntime(
                dataset=object(),
                sampler=object(),
                dataloader=object(),
                train_box_ids=("bbox_0001",),
                batch_size=0,
            )

        with self.assertRaises(TypeError):
            LearningBBoxDataLoaderRuntime(
                dataset=object(),
                sampler=object(),
                dataloader=object(),
                train_box_ids=("bbox_0001",),
                pin_memory=1,  # type: ignore[arg-type]
            )

        if torch is not None:
            with self.assertRaises(ValueError):
                LearningBBoxDataLoaderRuntime(
                    dataset=object(),
                    sampler=object(),
                    dataloader=object(),
                    train_box_ids=("bbox_0001",),
                    class_weights=torch.tensor([1.0], dtype=torch.float64),
                )

    def test_runtime_accepts_zero_num_workers(self) -> None:
        runtime = LearningBBoxDataLoaderRuntime(
            dataset=object(),
            sampler=object(),
            dataloader=object(),
            train_box_ids=("bbox_0001",),
            num_workers=0,
        )
        self.assertEqual(runtime.num_workers, 0)

    def test_set_current_learning_dataloader_class_weights_updates_runtime_metadata_only(self) -> None:
        if torch is None:
            self.skipTest("PyTorch is not available")

        class _FakeIterator:
            def __init__(self) -> None:
                self.shutdown_calls = 0

            def _shutdown_workers(self) -> None:
                self.shutdown_calls += 1

        class _FakeDataLoader:
            def __init__(self) -> None:
                self._iterator = _FakeIterator()
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        class _FakeClosable:
            def __init__(self) -> None:
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        dataset = _FakeClosable()
        sampler = _FakeClosable()
        dataloader = _FakeDataLoader()
        iterator_ref = dataloader._iterator
        runtime = LearningBBoxDataLoaderRuntime(
            dataset=dataset,
            sampler=sampler,
            dataloader=dataloader,
            train_box_ids=("bbox_0001",),
        )
        set_current_learning_dataloader_runtime(runtime)

        weights = torch.tensor([1.0, 2.0], dtype=torch.float32)
        updated = set_current_learning_dataloader_class_weights(weights)
        current = get_current_learning_dataloader_runtime()

        self.assertIsNotNone(current)
        self.assertIs(current, updated)
        self.assertIs(current.class_weights, weights)
        self.assertEqual(iterator_ref.shutdown_calls, 0)
        self.assertEqual(dataloader.close_calls, 0)
        self.assertEqual(dataset.close_calls, 0)
        self.assertEqual(sampler.close_calls, 0)


class LearningEvalRuntimeStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_current_learning_eval_runtimes_by_box_id()

    def tearDown(self) -> None:
        clear_current_learning_eval_runtimes_by_box_id()

    def test_set_and_get_eval_runtimes_by_box_id(self) -> None:
        first = LearningBBoxEvalRuntime(
            box_id="bbox_0007",
            dataloader=object(),
            buffer=object(),
        )
        second = LearningBBoxEvalRuntime(
            box_id="bbox_0011",
            dataloader=object(),
            buffer=object(),
        )

        stored = set_current_learning_eval_runtimes_by_box_id(
            {
                "bbox_0007": first,
                "bbox_0011": second,
            }
        )
        current = get_current_learning_eval_runtimes_by_box_id()

        self.assertEqual(tuple(sorted(stored.keys())), ("bbox_0007", "bbox_0011"))
        self.assertEqual(tuple(sorted(current.keys())), ("bbox_0007", "bbox_0011"))
        self.assertIs(current["bbox_0007"], first)
        self.assertIs(current["bbox_0011"], second)

    def test_set_eval_runtime_components_by_box_id(self) -> None:
        first_loader = object()
        first_buffer = object()
        second_loader = object()
        second_buffer = object()

        stored = set_current_learning_eval_runtime_components_by_box_id(
            {
                "bbox_0003": (first_loader, first_buffer),
                "bbox_0009": (second_loader, second_buffer),
            }
        )

        self.assertEqual(tuple(sorted(stored.keys())), ("bbox_0003", "bbox_0009"))
        self.assertIs(stored["bbox_0003"].dataloader, first_loader)
        self.assertIs(stored["bbox_0003"].buffer, first_buffer)
        self.assertIs(stored["bbox_0009"].dataloader, second_loader)
        self.assertIs(stored["bbox_0009"].buffer, second_buffer)

    def test_replacing_eval_runtimes_disposes_previous_components(self) -> None:
        class _FakeIterator:
            def __init__(self) -> None:
                self.shutdown_calls = 0

            def _shutdown_workers(self) -> None:
                self.shutdown_calls += 1

        class _FakeDataLoader:
            def __init__(self) -> None:
                self._iterator = _FakeIterator()
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        class _FakeBuffer:
            def __init__(self) -> None:
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        first_loader = _FakeDataLoader()
        first_iterator = first_loader._iterator
        first_buffer = _FakeBuffer()
        first_runtime = LearningBBoxEvalRuntime(
            box_id="bbox_0007",
            dataloader=first_loader,
            buffer=first_buffer,
        )
        set_current_learning_eval_runtimes_by_box_id({"bbox_0007": first_runtime})

        second_runtime = LearningBBoxEvalRuntime(
            box_id="bbox_0011",
            dataloader=object(),
            buffer=object(),
        )
        set_current_learning_eval_runtimes_by_box_id({"bbox_0011": second_runtime})

        self.assertEqual(first_iterator.shutdown_calls, 1)
        self.assertEqual(first_loader.close_calls, 1)
        self.assertEqual(first_buffer.close_calls, 1)

    def test_clear_eval_runtimes_disposes_current_components(self) -> None:
        class _FakeIterator:
            def __init__(self) -> None:
                self.shutdown_calls = 0

            def _shutdown_workers(self) -> None:
                self.shutdown_calls += 1

        class _FakeDataLoader:
            def __init__(self) -> None:
                self._iterator = _FakeIterator()
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        class _FakeBuffer:
            def __init__(self) -> None:
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        loader = _FakeDataLoader()
        iterator_ref = loader._iterator
        buffer = _FakeBuffer()
        runtime = LearningBBoxEvalRuntime(
            box_id="bbox_0007",
            dataloader=loader,
            buffer=buffer,
        )
        set_current_learning_eval_runtimes_by_box_id({"bbox_0007": runtime})

        clear_current_learning_eval_runtimes_by_box_id()

        self.assertEqual(iterator_ref.shutdown_calls, 1)
        self.assertEqual(loader.close_calls, 1)
        self.assertEqual(buffer.close_calls, 1)
        self.assertEqual(get_current_learning_eval_runtimes_by_box_id(), {})

    def test_eval_runtime_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            LearningBBoxEvalRuntime(
                box_id="bbox_0007",
                dataloader=None,  # type: ignore[arg-type]
                buffer=object(),
            )
        with self.assertRaises(ValueError):
            LearningBBoxEvalRuntime(
                box_id="bbox_0007",
                dataloader=object(),
                buffer=None,  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            set_current_learning_eval_runtimes_by_box_id(
                {
                    "bbox_0001": LearningBBoxEvalRuntime(
                        box_id="bbox_0002",
                        dataloader=object(),
                        buffer=object(),
                    )
                }
            )
        with self.assertRaises(TypeError):
            set_current_learning_eval_runtime_components_by_box_id(
                {
                    "bbox_0001": object(),  # type: ignore[dict-item]
                }
            )


class LearningModelRuntimeStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_current_learning_model_runtime()

    def tearDown(self) -> None:
        clear_current_learning_model_runtime()

    def test_set_and_get_current_model_runtime(self) -> None:
        runtime = LearningModelRuntime(
            model=object(),
            optimizer=object(),
            checkpoint_path="foundation_model/weights_epoch_190.cp",
            device_ids=(0, 1),
            num_classes=6,
            hyperparameters={"lr": 0.0001, "weight_decay": 0.001},
        )

        returned = set_current_learning_model_runtime(runtime)
        current = get_current_learning_model_runtime()

        self.assertIs(returned, runtime)
        self.assertIs(current, runtime)
        self.assertEqual(current.checkpoint_path, "foundation_model/weights_epoch_190.cp")
        self.assertEqual(current.device_ids, (0, 1))
        self.assertEqual(current.num_classes, 6)
        self.assertEqual(current.hyperparameters["lr"], 0.0001)
        self.assertEqual(current.hyperparameters["weight_decay"], 0.001)

    def test_set_current_learning_model_components_replaces_previous_runtime(self) -> None:
        set_current_learning_model_components(
            model=object(),
            optimizer=object(),
            checkpoint_path="foundation_model/weights_epoch_180.cp",
            device_ids=(0, 1),
            num_classes=4,
            hyperparameters={"lr": 0.0002},
        )
        second = set_current_learning_model_components(
            model=object(),
            optimizer=object(),
            checkpoint_path="foundation_model/weights_epoch_190.cp",
            device_ids=(0, 1, 2),
            num_classes=6,
            hyperparameters={"lr": 0.0001, "lwise_lr_decay": 0.8},
        )
        current = get_current_learning_model_runtime()

        self.assertIsNotNone(current)
        self.assertIs(current, second)
        self.assertEqual(current.checkpoint_path, "foundation_model/weights_epoch_190.cp")
        self.assertEqual(current.device_ids, (0, 1, 2))
        self.assertEqual(current.num_classes, 6)
        self.assertEqual(current.hyperparameters["lwise_lr_decay"], 0.8)

    def test_clear_current_model_runtime(self) -> None:
        set_current_learning_model_components(
            model=object(),
            optimizer=object(),
            checkpoint_path="foundation_model/weights_epoch_190.cp",
            device_ids=(0, 1),
            num_classes=6,
        )
        self.assertIsNotNone(get_current_learning_model_runtime())

        clear_current_learning_model_runtime()

        self.assertIsNone(get_current_learning_model_runtime())

    def test_model_runtime_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            LearningModelRuntime(
                model=None,  # type: ignore[arg-type]
                optimizer=object(),
                checkpoint_path="foundation_model/weights_epoch_190.cp",
                device_ids=(0, 1),
                num_classes=6,
            )

        with self.assertRaises(ValueError):
            LearningModelRuntime(
                model=object(),
                optimizer=None,  # type: ignore[arg-type]
                checkpoint_path="foundation_model/weights_epoch_190.cp",
                device_ids=(0, 1),
                num_classes=6,
            )

        with self.assertRaises(ValueError):
            LearningModelRuntime(
                model=object(),
                optimizer=object(),
                checkpoint_path="   ",
                device_ids=(0, 1),
                num_classes=6,
            )

        with self.assertRaises(ValueError):
            LearningModelRuntime(
                model=object(),
                optimizer=object(),
                checkpoint_path="foundation_model/weights_epoch_190.cp",
                device_ids=tuple(),
                num_classes=6,
            )

        with self.assertRaises(ValueError):
            LearningModelRuntime(
                model=object(),
                optimizer=object(),
                checkpoint_path="foundation_model/weights_epoch_190.cp",
                device_ids=(0, 0),
                num_classes=6,
            )

        with self.assertRaises(ValueError):
            LearningModelRuntime(
                model=object(),
                optimizer=object(),
                checkpoint_path="foundation_model/weights_epoch_190.cp",
                device_ids=(-1, 1),
                num_classes=6,
            )

        with self.assertRaises(ValueError):
            LearningModelRuntime(
                model=object(),
                optimizer=object(),
                checkpoint_path="foundation_model/weights_epoch_190.cp",
                device_ids=(0, 1),
                num_classes=0,
            )

        with self.assertRaises(TypeError):
            LearningModelRuntime(
                model=object(),
                optimizer=object(),
                checkpoint_path="foundation_model/weights_epoch_190.cp",
                device_ids=(0, 1),
                num_classes=6,
                hyperparameters=[],  # type: ignore[arg-type]
            )

        with self.assertRaises(TypeError):
            LearningModelRuntime(
                model=object(),
                optimizer=object(),
                checkpoint_path="foundation_model/weights_epoch_190.cp",
                device_ids=(0, 1),
                num_classes=6,
                hyperparameters={1: "bad key"},  # type: ignore[dict-item]
            )

if __name__ == "__main__":
    unittest.main()
