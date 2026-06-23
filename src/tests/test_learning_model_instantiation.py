from __future__ import annotations

from types import SimpleNamespace
import unittest

from src.learning import (
    LearningBBoxEvalRuntime,
    LearningLabelSpace,
    LearningSession,
    LearningModelRuntime,
    clear_current_learning_label_space,
    set_current_learning_dataloader_components,
    set_current_learning_eval_runtimes_by_box_id,
    set_current_learning_label_space,
    clear_current_learning_dataloader_runtime,
    clear_current_learning_eval_runtimes_by_box_id,
    clear_current_learning_model_runtime,
    get_current_learning_model_runtime,
)
from src.learning.model_instantiation import (
    FoundationModelConfig,
    FoundationInstantiationPreconditions,
    _extract_encoder_weights,
    inspect_foundation_checkpoint_metadata,
    instantiate_foundation_model_runtime,
    save_foundation_model_checkpoint,
    validate_foundation_checkpoint_load_preconditions,
    validate_foundation_model_instantiation_preconditions,
)


class FoundationModelInstantiationHelperTests(unittest.TestCase):
    def test_extract_encoder_weights_normalizes_module_prefix_and_drops_norm(self) -> None:
        state = {
            "encoder_weights": {
                "module.norm.weight": object(),
                "module.norm.bias": object(),
                "module.patch_embed.weight": object(),
                "module.blocks.0.attn.weight": object(),
            }
        }

        normalized = _extract_encoder_weights(state)

        self.assertEqual(
            tuple(sorted(normalized.keys())),
            ("blocks.0.attn.weight", "patch_embed.weight"),
        )

    def test_extract_encoder_weights_rejects_invalid_state(self) -> None:
        with self.assertRaises(KeyError):
            _extract_encoder_weights({})
        with self.assertRaises(TypeError):
            _extract_encoder_weights(object())  # type: ignore[arg-type]


class FoundationCheckpointMetadataTests(unittest.TestCase):
    class _FakeTorchModule:
        def __init__(self, state, *, gpu_count: int = 2) -> None:
            self.state = state
            self.calls = []
            self.cuda = SimpleNamespace(device_count=lambda: int(gpu_count))

        def load(self, path, *, map_location):
            self.calls.append((str(path), map_location))
            return self.state

    def test_inspect_foundation_checkpoint_metadata_reads_num_classes_and_label_values(self) -> None:
        torch_module = self._FakeTorchModule(
            {
                "metadata": {
                    "num_classes": 3,
                    "hyperparameters": {"label_values": (0, 1, 2)},
                }
            }
        )

        metadata = inspect_foundation_checkpoint_metadata(
            "model.cp",
            torch_module=torch_module,
        )

        self.assertEqual(metadata.checkpoint_path, "model.cp")
        self.assertEqual(metadata.num_classes, 3)
        self.assertEqual(metadata.label_values, (0, 1, 2))
        self.assertEqual(torch_module.calls, [("model.cp", "cpu")])

    def test_inspect_foundation_checkpoint_metadata_reads_label_space_metadata(
        self,
    ) -> None:
        torch_module = self._FakeTorchModule(
            {
                "metadata": {
                    "num_classes": 3,
                    "label_space": {
                        "label_values": (0, 2, 5),
                        "background_label": 0,
                        "mask_label": -100,
                        "source_signature": ("semantic", "/tmp/seg.tif", 7),
                    },
                    "hyperparameters": {"label_values": (0, 2, 5)},
                }
            }
        )

        metadata = inspect_foundation_checkpoint_metadata(
            "model.cp",
            torch_module=torch_module,
        )

        self.assertEqual(metadata.label_values, (0, 2, 5))
        self.assertIsNotNone(metadata.label_space)
        self.assertEqual(metadata.label_space.label_values, (0, 2, 5))
        self.assertEqual(metadata.label_space.source_signature, ("semantic", "/tmp/seg.tif", 7))

    def test_inspect_foundation_checkpoint_metadata_rejects_label_space_mismatch(
        self,
    ) -> None:
        torch_module = self._FakeTorchModule(
            {
                "metadata": {
                    "num_classes": 3,
                    "label_space": {"label_values": (0, 1, 3)},
                    "hyperparameters": {"label_values": (0, 1, 2)},
                }
            }
        )

        with self.assertRaisesRegex(ValueError, "label_space.label_values"):
            inspect_foundation_checkpoint_metadata(
                "model.cp",
                torch_module=torch_module,
            )

    def test_inspect_foundation_checkpoint_metadata_rejects_missing_metadata(self) -> None:
        torch_module = self._FakeTorchModule({})

        with self.assertRaisesRegex(ValueError, "metadata is missing"):
            inspect_foundation_checkpoint_metadata(
                "model.cp",
                torch_module=torch_module,
            )

    def test_inspect_foundation_checkpoint_metadata_rejects_missing_num_classes(self) -> None:
        torch_module = self._FakeTorchModule(
            {
                "metadata": {
                    "hyperparameters": {"label_values": (0, 1)},
                }
            }
        )

        with self.assertRaisesRegex(TypeError, "metadata.num_classes"):
            inspect_foundation_checkpoint_metadata(
                "model.cp",
                torch_module=torch_module,
            )

    def test_inspect_foundation_checkpoint_metadata_rejects_missing_label_values(self) -> None:
        torch_module = self._FakeTorchModule(
            {
                "metadata": {
                    "num_classes": 3,
                    "hyperparameters": {},
                }
            }
        )

        with self.assertRaisesRegex(ValueError, "label_values is missing"):
            inspect_foundation_checkpoint_metadata(
                "model.cp",
                torch_module=torch_module,
            )

    def test_inspect_foundation_checkpoint_metadata_rejects_label_value_count_mismatch(self) -> None:
        torch_module = self._FakeTorchModule(
            {
                "metadata": {
                    "num_classes": 3,
                    "hyperparameters": {"label_values": (0, 1)},
                }
            }
        )

        with self.assertRaisesRegex(ValueError, "length must match num_classes"):
            inspect_foundation_checkpoint_metadata(
                "model.cp",
                torch_module=torch_module,
            )

    def test_validate_foundation_checkpoint_load_preconditions_uses_metadata_and_gpus(self) -> None:
        torch_module = self._FakeTorchModule(
            {
                "metadata": {
                    "num_classes": 2,
                    "hyperparameters": {"label_values": (0, 1)},
                }
            },
            gpu_count=3,
        )

        preconditions = validate_foundation_checkpoint_load_preconditions(
            "model.cp",
            require_min_gpu_count=2,
            torch_module=torch_module,
        )

        self.assertEqual(preconditions.num_classes, 2)
        self.assertEqual(preconditions.label_values, (0, 1))
        self.assertEqual(preconditions.available_gpu_count, 3)
        self.assertEqual(preconditions.device_ids, (0, 1, 2))


class FoundationModelInstantiationFlowTests(unittest.TestCase):
    class _FakeParameter:
        def __init__(self, *, requires_grad: bool = True) -> None:
            self.requires_grad = bool(requires_grad)

    class _FakeModel:
        def __init__(
            self,
            minivol_size: int,
            *,
            in_chans: int,
            patch_size: int,
            embed_dim: int,
            depth: int,
            num_heads: int,
            num_classes: int,
        ) -> None:
            self.minivol_size = int(minivol_size)
            self.in_chans = int(in_chans)
            self.patch_size = int(patch_size)
            self.embed_dim = int(embed_dim)
            self.depth = int(depth)
            self.num_heads = int(num_heads)
            self.num_classes = int(num_classes)
            self.blocks = [object(), object()]
            self._decoder_built = False
            self.loaded_state = None
            self.train_calls = 0
            self.to_calls = []
            self._encoder_params = (
                ("patch_embed.weight", FoundationModelInstantiationFlowTests._FakeParameter()),
                ("blocks.1.weight", FoundationModelInstantiationFlowTests._FakeParameter()),
                ("blocks.0.weight", FoundationModelInstantiationFlowTests._FakeParameter()),
            )
            self._decoder_params = (
                ("top_conv_block1.weight", FoundationModelInstantiationFlowTests._FakeParameter()),
                ("top_conv_block2.weight", FoundationModelInstantiationFlowTests._FakeParameter()),
            )

        def named_parameters(self):
            if not self._decoder_built:
                return tuple(self._encoder_params)
            return tuple(self._encoder_params + self._decoder_params)

        def load_state_dict(self, state_dict):
            self.loaded_state = dict(state_dict)

        def state_dict(self):
            return {
                str(name): f"value::{name}"
                for name, _parameter in tuple(self.named_parameters())
            }

        def build_decoder(self) -> None:
            self._decoder_built = True

        def train(self) -> None:
            self.train_calls += 1

        def to(self, device):
            self.to_calls.append(device)

    class _FakeDataParallel:
        def __init__(self, module, device_ids) -> None:
            self.module = module
            self.device_ids = tuple(int(device_id) for device_id in tuple(device_ids))
            self.load_state_dict_calls = []

        def named_parameters(self):
            return tuple(
                (f"module.{name}", parameter)
                for name, parameter in tuple(self.module.named_parameters())
            )

        def state_dict(self):
            return {
                f"module.{name}": object()
                for name, _parameter in tuple(self.module.named_parameters())
            }

        def load_state_dict(self, state_dict, strict=True):
            self.load_state_dict_calls.append((dict(state_dict), bool(strict)))
            return None

    class _FakeOptimizer:
        def __init__(self, param_groups) -> None:
            self.param_groups = tuple(param_groups)

    class _FakeTorchModule:
        def __init__(self) -> None:
            self.cuda = SimpleNamespace(device_count=lambda: 4)
            self.nn = SimpleNamespace(DataParallel=FoundationModelInstantiationFlowTests._FakeDataParallel)
            self.optim = SimpleNamespace(
                AdamW=lambda groups: FoundationModelInstantiationFlowTests._FakeOptimizer(groups)
            )

        @staticmethod
        def device(spec: str) -> str:
            return f"device:{spec}"

    def setUp(self) -> None:
        clear_current_learning_model_runtime()

    def tearDown(self) -> None:
        clear_current_learning_model_runtime()

    def test_instantiate_foundation_model_runtime_with_injected_dependencies(self) -> None:
        torch_module = self._FakeTorchModule()
        checkpoint_calls = []

        def checkpoint_loader(path, *, map_location):
            checkpoint_calls.append((path, map_location))
            return {
                "encoder_weights": {
                    "module.norm.weight": object(),
                    "module.norm.bias": object(),
                    "module.patch_embed.weight": object(),
                    "module.blocks.1.weight": object(),
                    "module.blocks.0.weight": object(),
                }
            }

        runtime = instantiate_foundation_model_runtime(
            num_classes=6,
            config=FoundationModelConfig(checkpoint_path="foundation_model/weights_epoch_190.cp"),
            device_ids=(0, 1),
            store_in_session=False,
            model_factory=self._FakeModel,
            checkpoint_loader=checkpoint_loader,
            data_parallel_factory=self._FakeDataParallel,
            optimizer_factory=self._FakeOptimizer,
            torch_module=torch_module,
        )

        self.assertEqual(runtime.num_classes, 6)
        self.assertEqual(runtime.device_ids, (0, 1))
        self.assertEqual(runtime.checkpoint_path, "foundation_model/weights_epoch_190.cp")
        self.assertEqual(checkpoint_calls, [("foundation_model/weights_epoch_190.cp", "device:cuda:0")])
        self.assertEqual(
            tuple(sorted(runtime.model.module.loaded_state.keys())),
            ("blocks.0.weight", "blocks.1.weight", "patch_embed.weight"),
        )
        self.assertEqual(len(runtime.optimizer.param_groups), 5)
        self.assertAlmostEqual(runtime.optimizer.param_groups[0]["lwise_lr_decay_rate"], 1.0)
        self.assertAlmostEqual(runtime.optimizer.param_groups[1]["lwise_lr_decay_rate"], 1.0)
        self.assertAlmostEqual(runtime.optimizer.param_groups[2]["lwise_lr_decay_rate"], 1.0)
        self.assertAlmostEqual(runtime.optimizer.param_groups[3]["lwise_lr_decay_rate"], 0.8)
        self.assertAlmostEqual(runtime.optimizer.param_groups[4]["lwise_lr_decay_rate"], 0.64)
        self.assertEqual(runtime.hyperparameters["decoder_parameter_count"], 2)
        self.assertEqual(
            runtime.hyperparameters["source_checkpoint_path"],
            "foundation_model/weights_epoch_190.cp",
        )
        self.assertFalse(runtime.hyperparameters["trained_in_app"])
        self.assertEqual(runtime.hyperparameters["training_run_count"], 0)

    def test_instantiate_foundation_model_runtime_stores_runtime_when_requested(self) -> None:
        runtime = instantiate_foundation_model_runtime(
            num_classes=3,
            config=FoundationModelConfig(checkpoint_path="foundation_model/weights_epoch_190.cp"),
            device_ids=(0, 1),
            store_in_session=True,
            model_factory=self._FakeModel,
            checkpoint_loader=lambda _path, *, map_location: {
                "encoder_weights": {
                    "module.patch_embed.weight": object(),
                    "module.blocks.1.weight": object(),
                    "module.blocks.0.weight": object(),
                }
            },
            data_parallel_factory=self._FakeDataParallel,
            optimizer_factory=self._FakeOptimizer,
            torch_module=self._FakeTorchModule(),
        )

        current = get_current_learning_model_runtime()
        self.assertIs(current, runtime)
        self.assertEqual(current.num_classes, 3)

    def test_instantiate_foundation_model_runtime_can_store_in_explicit_session(self) -> None:
        session = LearningSession()

        runtime = instantiate_foundation_model_runtime(
            num_classes=3,
            config=FoundationModelConfig(checkpoint_path="foundation_model/weights_epoch_190.cp"),
            device_ids=(0, 1),
            store_in_session=True,
            model_factory=self._FakeModel,
            checkpoint_loader=lambda _path, *, map_location: {
                "encoder_weights": {
                    "module.patch_embed.weight": object(),
                    "module.blocks.1.weight": object(),
                    "module.blocks.0.weight": object(),
                }
            },
            data_parallel_factory=self._FakeDataParallel,
            optimizer_factory=self._FakeOptimizer,
            torch_module=self._FakeTorchModule(),
            learning_session=session,
        )

        self.assertIs(session.get_model_runtime(), runtime)
        self.assertIsNone(get_current_learning_model_runtime())

    def test_instantiate_foundation_model_runtime_rejects_encoder_key_mismatch(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "do not match UnetrPs8MulticlassesSmall architecture",
        ):
            instantiate_foundation_model_runtime(
                num_classes=6,
                config=FoundationModelConfig(checkpoint_path="foundation_model/weights_epoch_190.cp"),
                device_ids=(0, 1),
                store_in_session=False,
                model_factory=self._FakeModel,
                checkpoint_loader=lambda _path, *, map_location: {
                    "encoder_weights": {
                        # Missing blocks.0.weight on purpose.
                        "module.patch_embed.weight": object(),
                        "module.blocks.1.weight": object(),
                    }
                },
                data_parallel_factory=self._FakeDataParallel,
                optimizer_factory=self._FakeOptimizer,
                torch_module=self._FakeTorchModule(),
            )

    def test_instantiate_foundation_model_runtime_restores_full_model_weights_when_available(self) -> None:
        runtime = instantiate_foundation_model_runtime(
            num_classes=6,
            config=FoundationModelConfig(checkpoint_path="foundation_model/weights_epoch_190.cp"),
            device_ids=(0, 1),
            store_in_session=False,
            model_factory=self._FakeModel,
            checkpoint_loader=lambda _path, *, map_location: {
                "encoder_weights": {
                    "module.patch_embed.weight": object(),
                    "module.blocks.1.weight": object(),
                    "module.blocks.0.weight": object(),
                },
                "state_dict": {
                    "patch_embed.weight": object(),
                    "blocks.1.weight": object(),
                    "blocks.0.weight": object(),
                    "top_conv_block1.weight": object(),
                    "top_conv_block2.weight": object(),
                },
            },
            data_parallel_factory=self._FakeDataParallel,
            optimizer_factory=self._FakeOptimizer,
            torch_module=self._FakeTorchModule(),
        )

        self.assertTrue(runtime.hyperparameters["full_model_restore_applied"])
        self.assertEqual(len(runtime.model.load_state_dict_calls), 1)
        loaded_state, strict = runtime.model.load_state_dict_calls[0]
        self.assertTrue(strict)
        self.assertEqual(
            tuple(sorted(loaded_state.keys())),
            (
                "module.blocks.0.weight",
                "module.blocks.1.weight",
                "module.patch_embed.weight",
                "module.top_conv_block1.weight",
                "module.top_conv_block2.weight",
            ),
        )

    def test_instantiate_foundation_model_runtime_falls_back_when_full_model_restore_fails(self) -> None:
        class _FailingDataParallel(self._FakeDataParallel):
            def load_state_dict(self, state_dict, strict=True):
                raise RuntimeError("strict load failed")

        runtime = instantiate_foundation_model_runtime(
            num_classes=6,
            config=FoundationModelConfig(checkpoint_path="foundation_model/weights_epoch_190.cp"),
            device_ids=(0, 1),
            store_in_session=False,
            model_factory=self._FakeModel,
            checkpoint_loader=lambda _path, *, map_location: {
                "encoder_weights": {
                    "module.patch_embed.weight": object(),
                    "module.blocks.1.weight": object(),
                    "module.blocks.0.weight": object(),
                },
                "state_dict": {
                    "patch_embed.weight": object(),
                    "blocks.1.weight": object(),
                    "blocks.0.weight": object(),
                    "top_conv_block1.weight": object(),
                    "top_conv_block2.weight": object(),
                },
            },
            data_parallel_factory=_FailingDataParallel,
            optimizer_factory=self._FakeOptimizer,
            torch_module=self._FakeTorchModule(),
        )

        self.assertFalse(runtime.hyperparameters["full_model_restore_applied"])

    def test_instantiate_foundation_model_runtime_preserves_training_provenance_from_checkpoint(self) -> None:
        runtime = instantiate_foundation_model_runtime(
            num_classes=6,
            config=FoundationModelConfig(checkpoint_path="foundation_model/weights_epoch_190.cp"),
            checkpoint_path="loaded/custom_trained_model.cp",
            device_ids=(0, 1),
            store_in_session=False,
            model_factory=self._FakeModel,
            checkpoint_loader=lambda _path, *, map_location: {
                "encoder_weights": {
                    "module.patch_embed.weight": object(),
                    "module.blocks.1.weight": object(),
                    "module.blocks.0.weight": object(),
                },
                "metadata": {
                    "hyperparameters": {
                        "source_checkpoint_path": "loaded/custom_trained_model.cp",
                        "trained_in_app": True,
                        "training_run_count": 3,
                        "label_values": (0, 1, 2, 3, 4, 5),
                    }
                },
            },
            data_parallel_factory=self._FakeDataParallel,
            optimizer_factory=self._FakeOptimizer,
            torch_module=self._FakeTorchModule(),
        )

        self.assertEqual(
            runtime.hyperparameters["source_checkpoint_path"],
            "loaded/custom_trained_model.cp",
        )
        self.assertTrue(runtime.hyperparameters["trained_in_app"])
        self.assertEqual(runtime.hyperparameters["training_run_count"], 3)
        self.assertEqual(runtime.hyperparameters["label_values"], (0, 1, 2, 3, 4, 5))

    def test_instantiate_foundation_model_runtime_preserves_label_space_metadata(
        self,
    ) -> None:
        runtime = instantiate_foundation_model_runtime(
            num_classes=3,
            config=FoundationModelConfig(checkpoint_path="foundation_model/weights_epoch_190.cp"),
            device_ids=(0, 1),
            store_in_session=False,
            model_factory=self._FakeModel,
            checkpoint_loader=lambda _path, *, map_location: {
                "encoder_weights": {
                    "module.patch_embed.weight": object(),
                    "module.blocks.1.weight": object(),
                    "module.blocks.0.weight": object(),
                },
                "metadata": {
                    "label_space": {
                        "label_values": (0, 2, 5),
                        "background_label": 0,
                        "mask_label": -100,
                        "source_signature": ("semantic", "/tmp/seg.tif", 9),
                    },
                    "hyperparameters": {
                        "label_values": (0, 2, 5),
                    },
                },
            },
            data_parallel_factory=self._FakeDataParallel,
            optimizer_factory=self._FakeOptimizer,
            torch_module=self._FakeTorchModule(),
        )

        self.assertEqual(runtime.hyperparameters["label_values"], (0, 2, 5))
        self.assertEqual(
            runtime.hyperparameters["label_space"],
            {
                "label_values": (0, 2, 5),
                "background_label": 0,
                "mask_label": -100,
                "source_signature": ("semantic", "/tmp/seg.tif", 9),
            },
        )

    def test_save_checkpoint_roundtrip_load_is_compatible(self) -> None:
        class _SaveTorchModule:
            def __init__(self) -> None:
                self.saved_payload = None
                self.saved_path = None

            def save(self, payload, path) -> None:
                self.saved_payload = payload
                self.saved_path = str(path)

        save_torch_module = _SaveTorchModule()
        source_model = self._FakeModel(
            200,
            in_chans=1,
            patch_size=8,
            embed_dim=1152,
            depth=24,
            num_heads=16,
            num_classes=6,
        )
        source_model.build_decoder()
        source_parallel = self._FakeDataParallel(source_model, (0, 1))
        source_runtime = LearningModelRuntime(
            model=source_parallel,
            optimizer=object(),
            checkpoint_path="foundation_model/weights_epoch_190.cp",
            device_ids=(0, 1),
            num_classes=6,
            hyperparameters={
                "encoder_parameter_count": 3,
                "label_values": (0, 1, 2, 3, 4, 5),
            },
        )

        saved_path = save_foundation_model_checkpoint(
            runtime=source_runtime,
            checkpoint_path="/tmp/roundtrip_saved_model.cp",
            torch_module=save_torch_module,
            saved_at_utc="2026-03-09T00:00:00+00:00",
        )
        self.assertEqual(saved_path, "/tmp/roundtrip_saved_model.cp")
        self.assertIsNotNone(save_torch_module.saved_payload)

        loaded_runtime = instantiate_foundation_model_runtime(
            num_classes=6,
            config=FoundationModelConfig(checkpoint_path="/tmp/roundtrip_saved_model.cp"),
            checkpoint_path="/tmp/roundtrip_saved_model.cp",
            device_ids=(0, 1),
            store_in_session=False,
            model_factory=self._FakeModel,
            checkpoint_loader=lambda _path, *, map_location: save_torch_module.saved_payload,
            data_parallel_factory=self._FakeDataParallel,
            optimizer_factory=self._FakeOptimizer,
            torch_module=self._FakeTorchModule(),
        )

        self.assertTrue(loaded_runtime.hyperparameters["full_model_restore_applied"])
        self.assertEqual(loaded_runtime.hyperparameters["label_values"], (0, 1, 2, 3, 4, 5))
        self.assertEqual(
            tuple(sorted(loaded_runtime.model.module.loaded_state.keys())),
            ("blocks.0.weight", "blocks.1.weight", "patch_embed.weight"),
        )
        self.assertEqual(len(loaded_runtime.model.load_state_dict_calls), 1)


class FoundationModelCheckpointSaveTests(unittest.TestCase):
    class _FakeInnerModel:
        def __init__(self) -> None:
            self._state = {
                "patch_embed.weight": "enc_patch",
                "blocks.1.weight": "enc_block_1",
                "blocks.0.weight": "enc_block_0",
                "top_conv_block1.weight": "dec_top_1",
                "top_conv_block2.weight": "dec_top_2",
            }

        def state_dict(self):
            return dict(self._state)

        def named_parameters(self):
            ordered_names = (
                "patch_embed.weight",
                "blocks.1.weight",
                "blocks.0.weight",
                "top_conv_block1.weight",
                "top_conv_block2.weight",
            )
            return tuple((name, object()) for name in ordered_names)

    class _FakeParallelModel:
        def __init__(self) -> None:
            self.module = FoundationModelCheckpointSaveTests._FakeInnerModel()

        def state_dict(self):
            inner = self.module.state_dict()
            return {
                f"module.{key}": value for key, value in tuple(inner.items())
            }

    class _FakeTorchModule:
        def __init__(self) -> None:
            self.saved_payload = None
            self.saved_path = None

        def save(self, payload, path) -> None:
            self.saved_payload = payload
            self.saved_path = str(path)

    def _make_runtime(self, *, hyperparameters=None) -> LearningModelRuntime:
        resolved_hyperparameters = {
            "label_values": (0, 1, 2, 3, 4, 5),
        }
        resolved_hyperparameters.update(dict(hyperparameters or {}))
        return LearningModelRuntime(
            model=self._FakeParallelModel(),
            optimizer=object(),
            checkpoint_path="foundation_model/weights_epoch_190.cp",
            device_ids=(0, 1),
            num_classes=6,
            hyperparameters=resolved_hyperparameters,
        )

    def test_save_foundation_model_checkpoint_writes_encoder_full_state_and_metadata(self) -> None:
        runtime = self._make_runtime(hyperparameters={"encoder_parameter_count": 3})
        torch_module = self._FakeTorchModule()

        saved_path = save_foundation_model_checkpoint(
            runtime=runtime,
            checkpoint_path="/tmp/foundation_saved.cp",
            torch_module=torch_module,
            saved_at_utc="2026-03-09T00:00:00+00:00",
        )

        self.assertEqual(saved_path, "/tmp/foundation_saved.cp")
        self.assertEqual(torch_module.saved_path, "/tmp/foundation_saved.cp")
        payload = torch_module.saved_payload
        self.assertIsNotNone(payload)
        self.assertIn("encoder_weights", payload)
        self.assertIn("state_dict", payload)
        self.assertIn("metadata", payload)

        encoder_keys = tuple(sorted(payload["encoder_weights"].keys()))
        self.assertEqual(
            encoder_keys,
            ("blocks.0.weight", "blocks.1.weight", "patch_embed.weight"),
        )

        state_keys = tuple(sorted(payload["state_dict"].keys()))
        self.assertEqual(
            state_keys,
            (
                "module.blocks.0.weight",
                "module.blocks.1.weight",
                "module.patch_embed.weight",
                "module.top_conv_block1.weight",
                "module.top_conv_block2.weight",
            ),
        )

        metadata = payload["metadata"]
        self.assertEqual(metadata["format_version"], 1)
        self.assertEqual(metadata["architecture"], "_FakeInnerModel")
        self.assertEqual(metadata["saved_at_utc"], "2026-03-09T00:00:00+00:00")
        self.assertEqual(metadata["num_classes"], 6)
        self.assertEqual(metadata["device_ids"], (0, 1))
        self.assertEqual(metadata["checkpoint_path"], "foundation_model/weights_epoch_190.cp")
        self.assertEqual(
            metadata["label_space"],
            {
                "label_values": (0, 1, 2, 3, 4, 5),
                "background_label": 0,
                "mask_label": -100,
                "source_signature": None,
            },
        )
        self.assertEqual(
            metadata["hyperparameters"]["label_values"],
            (0, 1, 2, 3, 4, 5),
        )
        self.assertEqual(
            metadata["hyperparameters"]["label_space"],
            metadata["label_space"],
        )

    def test_save_foundation_model_checkpoint_preserves_runtime_label_space_metadata(
        self,
    ) -> None:
        runtime = self._make_runtime(
            hyperparameters={
                "encoder_parameter_count": 3,
                "label_values": (0, 2, 5, 8, 13, 21),
                "label_space": {
                    "label_values": (0, 2, 5, 8, 13, 21),
                    "background_label": 0,
                    "mask_label": -100,
                    "source_signature": ("semantic", "/tmp/seg.tif", 11),
                },
            }
        )
        torch_module = self._FakeTorchModule()

        save_foundation_model_checkpoint(
            runtime=runtime,
            checkpoint_path="/tmp/foundation_saved.cp",
            torch_module=torch_module,
            saved_at_utc="2026-03-09T00:00:00+00:00",
        )

        metadata = torch_module.saved_payload["metadata"]
        self.assertEqual(
            metadata["label_space"],
            {
                "label_values": (0, 2, 5, 8, 13, 21),
                "background_label": 0,
                "mask_label": -100,
                "source_signature": ("semantic", "/tmp/seg.tif", 11),
            },
        )

    def test_save_foundation_model_checkpoint_rejects_missing_label_values(self) -> None:
        runtime = self._make_runtime(hyperparameters={"label_values": None})

        with self.assertRaisesRegex((TypeError, ValueError), "label_values"):
            save_foundation_model_checkpoint(
                runtime=runtime,
                checkpoint_path="/tmp/foundation_saved.cp",
                torch_module=self._FakeTorchModule(),
            )

    def test_save_foundation_model_checkpoint_rejects_non_cp_extension(self) -> None:
        runtime = self._make_runtime(hyperparameters={"encoder_parameter_count": 3})

        with self.assertRaisesRegex(ValueError, r"\.cp extension"):
            save_foundation_model_checkpoint(
                runtime=runtime,
                checkpoint_path="/tmp/foundation_saved.pt",
                torch_module=self._FakeTorchModule(),
            )

    def test_save_foundation_model_checkpoint_can_infer_encoder_names_without_count(self) -> None:
        runtime = self._make_runtime(hyperparameters={})
        torch_module = self._FakeTorchModule()

        save_foundation_model_checkpoint(
            runtime=runtime,
            checkpoint_path="/tmp/foundation_saved.cp",
            torch_module=torch_module,
        )

        payload = torch_module.saved_payload
        self.assertIsNotNone(payload)
        encoder_keys = tuple(sorted(payload["encoder_weights"].keys()))
        self.assertEqual(
            encoder_keys,
            ("blocks.0.weight", "blocks.1.weight", "patch_embed.weight"),
        )


class FoundationModelInstantiationPreconditionsTests(unittest.TestCase):
    class _FakeTorchModule:
        def __init__(self, gpu_count: int) -> None:
            self.cuda = SimpleNamespace(device_count=lambda: int(gpu_count))

    @staticmethod
    def _set_training_runtime() -> None:
        set_current_learning_dataloader_components(
            dataset=object(),
            sampler=object(),
            dataloader=object(),
            train_box_ids=("bbox_0001",),
        )

    @staticmethod
    def _set_eval_runtimes(num_classes_by_box_id, *, label_values_by_box_id=None):
        runtimes = {}
        for box_id, num_classes in tuple(num_classes_by_box_id.items()):
            if label_values_by_box_id is not None:
                label_values = tuple(label_values_by_box_id[box_id])
            else:
                label_values = tuple(range(int(num_classes)))
            runtimes[box_id] = LearningBBoxEvalRuntime(
                box_id=box_id,
                dataloader=object(),
                buffer=SimpleNamespace(
                    num_classes=num_classes,
                    label_values=label_values,
                ),
            )
        set_current_learning_eval_runtimes_by_box_id(runtimes)

    def setUp(self) -> None:
        clear_current_learning_model_runtime()
        clear_current_learning_dataloader_runtime()
        clear_current_learning_eval_runtimes_by_box_id()
        clear_current_learning_label_space()

    def tearDown(self) -> None:
        clear_current_learning_model_runtime()
        clear_current_learning_dataloader_runtime()
        clear_current_learning_eval_runtimes_by_box_id()
        clear_current_learning_label_space()

    def test_validate_preconditions_reads_session_and_resolves_num_classes(self) -> None:
        self._set_training_runtime()
        self._set_eval_runtimes({"bbox_0007": 6, "bbox_0011": 6})

        preconditions = validate_foundation_model_instantiation_preconditions(
            torch_module=self._FakeTorchModule(gpu_count=3),
        )

        self.assertIsInstance(preconditions, FoundationInstantiationPreconditions)
        self.assertEqual(preconditions.num_classes, 6)
        self.assertEqual(preconditions.label_values, (0, 1, 2, 3, 4, 5))
        self.assertEqual(preconditions.available_gpu_count, 3)
        self.assertEqual(preconditions.device_ids, (0, 1, 2))
        self.assertEqual(tuple(sorted(preconditions.eval_runtimes_by_box_id.keys())), ("bbox_0007", "bbox_0011"))

    def test_validate_preconditions_can_read_explicit_learning_session(self) -> None:
        session = LearningSession()
        session.set_dataloader_components(
            dataset=object(),
            sampler=object(),
            dataloader=object(),
            train_box_ids=("bbox_0001",),
        )
        session.set_eval_runtimes_by_box_id(
            {
                "bbox_0007": LearningBBoxEvalRuntime(
                    box_id="bbox_0007",
                    dataloader=object(),
                    buffer=SimpleNamespace(
                        num_classes=6,
                        label_values=(0, 1, 2, 3, 4, 5),
                    ),
                )
            }
        )

        preconditions = validate_foundation_model_instantiation_preconditions(
            learning_session=session,
            torch_module=self._FakeTorchModule(gpu_count=2),
        )

        self.assertEqual(preconditions.num_classes, 6)
        self.assertEqual(preconditions.label_values, (0, 1, 2, 3, 4, 5))
        self.assertEqual(tuple(preconditions.eval_runtimes_by_box_id.keys()), ("bbox_0007",))

    def test_validate_preconditions_uses_current_label_space(self) -> None:
        self._set_training_runtime()
        self._set_eval_runtimes(
            {"bbox_0007": 3},
            label_values_by_box_id={"bbox_0007": (0, 2, 5)},
        )
        set_current_learning_label_space(LearningLabelSpace(label_values=(0, 2, 5)))

        preconditions = validate_foundation_model_instantiation_preconditions(
            torch_module=self._FakeTorchModule(gpu_count=2),
        )

        self.assertEqual(preconditions.num_classes, 3)
        self.assertEqual(preconditions.label_values, (0, 2, 5))

    def test_validate_preconditions_rejects_label_space_eval_label_mismatch(self) -> None:
        self._set_training_runtime()
        self._set_eval_runtimes(
            {"bbox_0007": 3},
            label_values_by_box_id={"bbox_0007": (0, 1, 2)},
        )
        set_current_learning_label_space(LearningLabelSpace(label_values=(0, 2, 5)))

        with self.assertRaisesRegex(ValueError, "current label space"):
            validate_foundation_model_instantiation_preconditions(
                torch_module=self._FakeTorchModule(gpu_count=2),
            )

    def test_validate_preconditions_rejects_when_training_runtime_missing(self) -> None:
        self._set_eval_runtimes({"bbox_0007": 6})

        with self.assertRaisesRegex(ValueError, "No training dataloader runtime"):
            validate_foundation_model_instantiation_preconditions(
                torch_module=self._FakeTorchModule(gpu_count=2),
            )

    def test_validate_preconditions_rejects_when_eval_runtime_missing(self) -> None:
        self._set_training_runtime()

        with self.assertRaisesRegex(ValueError, "No evaluation runtimes/buffers"):
            validate_foundation_model_instantiation_preconditions(
                torch_module=self._FakeTorchModule(gpu_count=2),
            )

    def test_validate_preconditions_rejects_when_eval_num_classes_mismatch(self) -> None:
        self._set_training_runtime()
        self._set_eval_runtimes({"bbox_0007": 6, "bbox_0011": 7})

        with self.assertRaisesRegex(ValueError, "must share the same num_classes"):
            validate_foundation_model_instantiation_preconditions(
                torch_module=self._FakeTorchModule(gpu_count=2),
            )

    def test_validate_preconditions_rejects_when_eval_buffer_has_no_num_classes(self) -> None:
        self._set_training_runtime()
        set_current_learning_eval_runtimes_by_box_id(
            {
                "bbox_0007": LearningBBoxEvalRuntime(
                    box_id="bbox_0007",
                    dataloader=object(),
                    buffer=object(),
                )
            }
        )

        with self.assertRaisesRegex(ValueError, "does not expose 'num_classes'"):
            validate_foundation_model_instantiation_preconditions(
                torch_module=self._FakeTorchModule(gpu_count=2),
            )

    def test_validate_preconditions_rejects_when_gpu_count_too_low(self) -> None:
        self._set_training_runtime()
        self._set_eval_runtimes({"bbox_0007": 6})

        with self.assertRaisesRegex(RuntimeError, "At least 2 CUDA devices"):
            validate_foundation_model_instantiation_preconditions(
                torch_module=self._FakeTorchModule(gpu_count=1),
            )


if __name__ == "__main__":
    unittest.main()
