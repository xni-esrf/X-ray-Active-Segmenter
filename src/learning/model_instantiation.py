from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
import re
import warnings
from numbers import Integral, Real
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - import availability is environment dependent
    import torch
    from torch import nn as nn
except Exception:  # pragma: no cover - import availability is environment dependent
    torch = None  # type: ignore[assignment]

    class _ModuleStub:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    class _NNStub:
        Module = _ModuleStub

    nn = _NNStub()  # type: ignore[assignment]

from .session_store import (
    LearningBBoxDataLoaderRuntime,
    LearningBBoxEvalRuntime,
    LearningModelRuntime,
    get_current_learning_dataloader_runtime,
    get_current_learning_eval_runtimes_by_box_id,
    set_current_learning_model_components,
)


DEFAULT_FOUNDATION_CHECKPOINT_PATH = "foundation_model/weights_epoch_190.cp"


def _require_torch():
    if torch is None:  # pragma: no cover - environment dependent
        raise ImportError("PyTorch is required to instantiate the foundation model.")
    return torch


def _require_timm_block():
    try:
        from timm.models.vision_transformer import Block
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "timm is required to instantiate the foundation ViT encoder blocks."
        ) from exc
    return Block


def _coerce_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be >= 1, got {normalized}")
    return normalized


def _coerce_positive_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    normalized = float(value)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be > 0, got {normalized}")
    return normalized


def _coerce_non_empty_path(path: object, *, name: str) -> str:
    if not isinstance(path, str):
        raise TypeError(f"{name} must be a string, got {type(path).__name__}")
    normalized = str(Path(path).expanduser()).strip()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty path")
    return normalized


def _coerce_device_ids(device_ids: Sequence[object]) -> Tuple[int, ...]:
    normalized = []
    for raw_device_id in tuple(device_ids):
        if isinstance(raw_device_id, bool) or not isinstance(raw_device_id, Integral):
            raise TypeError(
                "device_ids must contain integers only, "
                f"got {type(raw_device_id).__name__}"
            )
        device_id = int(raw_device_id)
        if device_id < 0:
            raise ValueError(f"device_ids must be >= 0, got {device_id}")
        normalized.append(device_id)
    if not normalized:
        raise ValueError("device_ids must contain at least one id")
    if len(set(normalized)) != len(normalized):
        raise ValueError("device_ids must not contain duplicates")
    return tuple(normalized)


def build_3d_sincos_position_embedding(
    grid_shape: Sequence[int],
    embed_dim: int,
    *,
    temperature: float = 10000.0,
):
    torch_mod = _require_torch()
    if len(tuple(grid_shape)) != 3:
        raise ValueError(f"grid_shape must be length 3, got {grid_shape}")
    h, w, d = tuple(_coerce_positive_int(v, name="grid_shape axis") for v in tuple(grid_shape))
    normalized_embed_dim = _coerce_positive_int(embed_dim, name="embed_dim")
    normalized_temperature = _coerce_positive_real(temperature, name="temperature")

    grid_h = torch_mod.arange(h, dtype=torch_mod.float32)
    grid_w = torch_mod.arange(w, dtype=torch_mod.float32)
    grid_d = torch_mod.arange(d, dtype=torch_mod.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid_h, grid_w, grid_d = torch_mod.meshgrid(grid_h, grid_w, grid_d)

    if normalized_embed_dim % 6 != 0:
        raise ValueError(
            "embed_dim must be divisible by 6 for 3D sin-cos position embedding, "
            f"got {normalized_embed_dim}"
        )
    pos_dim = normalized_embed_dim // 6
    omega = torch_mod.arange(pos_dim, dtype=torch_mod.float32) / float(pos_dim)
    omega = 1.0 / (normalized_temperature ** omega)
    out_h = torch_mod.einsum("m,d->md", [grid_h.flatten(), omega])
    out_w = torch_mod.einsum("m,d->md", [grid_w.flatten(), omega])
    out_d = torch_mod.einsum("m,d->md", [grid_d.flatten(), omega])
    pos_emb = torch_mod.cat(
        [
            torch_mod.sin(out_h),
            torch_mod.cos(out_h),
            torch_mod.sin(out_w),
            torch_mod.cos(out_w),
            torch_mod.sin(out_d),
            torch_mod.cos(out_d),
        ],
        dim=1,
    )[None, :, :]

    pos_embed = torch_mod.nn.Parameter(pos_emb)
    pos_embed.requires_grad = False
    return pos_embed


class PatchEmbed3D(nn.Module):
    """Patchify 3D minivolumes and convert to transformer token format."""

    def __init__(
        self,
        *,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        norm_layer=None,
    ) -> None:
        _require_torch()
        super().__init__()
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        transposed: bool = False,
        use_bn: bool = True,
    ) -> None:
        _require_torch()
        super().__init__()

        if transposed:
            conv_layer = nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
            )
        else:
            conv_layer = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            )

        layers = []
        if use_bn:
            layers.append(nn.GroupNorm(1, num_channels=in_channels))
        layers.append(conv_layer)
        layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UnetrPs8MulticlassesSmall(nn.Module):
    def __init__(
        self,
        minivol_size: int,
        *,
        in_chans: int = 1,
        patch_size: int = 8,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        embed_layer=PatchEmbed3D,
        act_layer=None,
        num_classes: int = 6,
        block_factory=None,
    ) -> None:
        torch_mod = _require_torch()
        super().__init__()
        if act_layer is None:
            act_layer = nn.GELU
        if block_factory is None:
            block_factory = _require_timm_block()

        self.patch_size = _coerce_positive_int(patch_size, name="patch_size")
        self.in_chans = _coerce_positive_int(in_chans, name="in_chans")
        self.embed_dim = _coerce_positive_int(embed_dim, name="embed_dim")
        self.minivol_size = _coerce_positive_int(minivol_size, name="minivol_size")
        self.num_classes = _coerce_positive_int(num_classes, name="num_classes")

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = embed_layer(
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
        )

        grid_size = self.minivol_size // self.patch_size
        self.pos_emb = build_3d_sincos_position_embedding(
            (grid_size, grid_size, grid_size),
            self.embed_dim,
        )

        dpr = [x.item() for x in torch_mod.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                block_factory(
                    dim=self.embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )

    def build_decoder(self) -> None:
        self.top_conv_block1 = ConvBlock(
            1,
            self.embed_dim // 64,
            transposed=False,
            use_bn=False,
        )
        self.top_conv_block2 = ConvBlock(
            self.embed_dim // 64,
            self.embed_dim // 64,
            transposed=False,
            use_bn=True,
        )

        self.norm_down = nn.LayerNorm(self.embed_dim)
        self.bottom_conv_block1 = ConvBlock(
            self.embed_dim,
            self.embed_dim // 2,
            transposed=False,
            use_bn=False,
        )
        self.bottom_conv_block2 = ConvBlock(
            self.embed_dim // 2,
            self.embed_dim // 16,
            transposed=True,
            use_bn=True,
        )

        self.norm_middeep = nn.LayerNorm(self.embed_dim)
        self.middeep_conv_block1 = ConvBlock(
            self.embed_dim,
            self.embed_dim // 2,
            transposed=False,
            use_bn=False,
        )
        self.middeep_conv_block2 = ConvBlock(
            self.embed_dim // 2,
            self.embed_dim // 16,
            transposed=True,
            use_bn=True,
        )

        self.concatdeep_conv_block1 = ConvBlock(
            self.embed_dim // 8,
            self.embed_dim // 8,
            transposed=False,
            use_bn=True,
        )
        self.concatdeep_conv_block2 = ConvBlock(
            self.embed_dim // 8,
            self.embed_dim // 8,
            transposed=False,
            use_bn=True,
        )
        self.concatdeep_conv_block3 = ConvBlock(
            self.embed_dim // 8,
            self.embed_dim // 32,
            transposed=True,
            use_bn=True,
        )

        self.norm_midshallow = nn.LayerNorm(self.embed_dim)
        self.midshallow_conv_block1 = ConvBlock(
            self.embed_dim,
            self.embed_dim,
            transposed=False,
            use_bn=False,
        )
        self.midshallow_conv_block2 = ConvBlock(
            self.embed_dim,
            self.embed_dim // 8,
            transposed=True,
            use_bn=True,
        )
        self.midshallow_conv_block3 = ConvBlock(
            self.embed_dim // 8,
            self.embed_dim // 8,
            transposed=False,
            use_bn=True,
        )
        self.midshallow_conv_block4 = ConvBlock(
            self.embed_dim // 8,
            self.embed_dim // 32,
            transposed=True,
            use_bn=True,
        )

        self.concatshallow_conv_block1 = ConvBlock(
            self.embed_dim // 16,
            self.embed_dim // 16,
            transposed=False,
            use_bn=True,
        )
        self.concatshallow_conv_block2 = ConvBlock(
            self.embed_dim // 16,
            self.embed_dim // 16,
            transposed=False,
            use_bn=True,
        )
        self.concatshallow_conv_block3 = ConvBlock(
            self.embed_dim // 16,
            self.embed_dim // 64,
            transposed=True,
            use_bn=True,
        )

        self.out_conv_block1 = ConvBlock(
            self.embed_dim // 32,
            self.embed_dim // 32,
            transposed=False,
            use_bn=True,
        )
        self.out_conv_block2 = ConvBlock(
            self.embed_dim // 32,
            self.embed_dim // 32,
            transposed=False,
            use_bn=True,
        )

        self.norm_last = nn.GroupNorm(1, num_channels=self.embed_dim // 32)
        self.conv_last = nn.Conv3d(
            self.embed_dim // 32,
            self.num_classes,
            kernel_size=1,
            padding=0,
        )

    def get_num_layers(self) -> int:
        return int(len(self.blocks))

    def forward(self, x):
        x_up = self.top_conv_block1(x)
        x_up = self.top_conv_block2(x_up)

        x = self.patch_embed(x)
        x = x + self.pos_emb.repeat([x.shape[0], 1, 1])
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i + 1 == len(self.blocks) // 3:
                x_midshallow = x.clone()
            elif i + 1 == 2 * len(self.blocks) // 3:
                x_middeep = x.clone()

        batch_size, _num_patches, embeddim = x.shape
        spatial = self.minivol_size // self.patch_size

        x = self.norm_down(x)
        x = x.permute((0, 2, 1))
        x = x.reshape(batch_size, embeddim, spatial, spatial, spatial)
        x = self.bottom_conv_block1(x)
        x = self.bottom_conv_block2(x)

        x_middeep = self.norm_middeep(x_middeep)
        x_middeep = x_middeep.permute((0, 2, 1))
        x_middeep = x_middeep.reshape(batch_size, embeddim, spatial, spatial, spatial)
        x_middeep = self.middeep_conv_block1(x_middeep)
        x_middeep = self.middeep_conv_block2(x_middeep)

        x = torch.concat([x, x_middeep], 1)
        x = self.concatdeep_conv_block1(x)
        x = self.concatdeep_conv_block2(x)
        x = self.concatdeep_conv_block3(x)

        x_midshallow = self.norm_midshallow(x_midshallow)
        x_midshallow = x_midshallow.permute((0, 2, 1))
        x_midshallow = x_midshallow.reshape(batch_size, embeddim, spatial, spatial, spatial)
        x_midshallow = self.midshallow_conv_block1(x_midshallow)
        x_midshallow = self.midshallow_conv_block2(x_midshallow)
        x_midshallow = self.midshallow_conv_block3(x_midshallow)
        x_midshallow = self.midshallow_conv_block4(x_midshallow)

        x = torch.concat([x, x_midshallow], 1)
        x = self.concatshallow_conv_block1(x)
        x = self.concatshallow_conv_block2(x)
        x = self.concatshallow_conv_block3(x)

        x = torch.concat([x, x_up], 1)
        x = self.out_conv_block1(x)
        x = self.out_conv_block2(x)
        x = self.norm_last(x)
        x = self.conv_last(x)
        return x


# Compatibility alias matching the original script naming.
Unetr_ps8_multiclasses_small = UnetrPs8MulticlassesSmall


@dataclass(frozen=True)
class FoundationModelConfig:
    checkpoint_path: str = DEFAULT_FOUNDATION_CHECKPOINT_PATH
    enc_depth: int = 24
    enc_emb_dim: int = 1152
    minivol_size: int = 200
    patch_size: int = 8
    num_heads: int = 16
    lr: float = 0.00005
    lwise_lr_decay: float = 0.8
    weight_decay: float = 0.001
    in_chans: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "checkpoint_path",
            _coerce_non_empty_path(self.checkpoint_path, name="checkpoint_path"),
        )
        object.__setattr__(self, "enc_depth", _coerce_positive_int(self.enc_depth, name="enc_depth"))
        object.__setattr__(
            self,
            "enc_emb_dim",
            _coerce_positive_int(self.enc_emb_dim, name="enc_emb_dim"),
        )
        object.__setattr__(
            self,
            "minivol_size",
            _coerce_positive_int(self.minivol_size, name="minivol_size"),
        )
        object.__setattr__(
            self,
            "patch_size",
            _coerce_positive_int(self.patch_size, name="patch_size"),
        )
        object.__setattr__(self, "num_heads", _coerce_positive_int(self.num_heads, name="num_heads"))
        object.__setattr__(self, "lr", _coerce_positive_real(self.lr, name="lr"))
        object.__setattr__(
            self,
            "lwise_lr_decay",
            _coerce_positive_real(self.lwise_lr_decay, name="lwise_lr_decay"),
        )
        object.__setattr__(
            self,
            "weight_decay",
            _coerce_positive_real(self.weight_decay, name="weight_decay"),
        )
        object.__setattr__(self, "in_chans", _coerce_positive_int(self.in_chans, name="in_chans"))


DEFAULT_FOUNDATION_MODEL_CONFIG = FoundationModelConfig()

_BLOCK_INDEX_PATTERN = re.compile(r"(?:^|\.)blocks\.(\d+)(?:\.|$)")
_DECODER_PARAMETER_PREFIXES = (
    "top_conv_block1",
    "top_conv_block2",
    "norm_down",
    "bottom_conv_block1",
    "bottom_conv_block2",
    "norm_middeep",
    "middeep_conv_block1",
    "middeep_conv_block2",
    "concatdeep_conv_block1",
    "concatdeep_conv_block2",
    "concatdeep_conv_block3",
    "norm_midshallow",
    "midshallow_conv_block1",
    "midshallow_conv_block2",
    "midshallow_conv_block3",
    "midshallow_conv_block4",
    "concatshallow_conv_block1",
    "concatshallow_conv_block2",
    "concatshallow_conv_block3",
    "out_conv_block1",
    "out_conv_block2",
    "norm_last",
    "conv_last",
)


@dataclass(frozen=True)
class FoundationInstantiationPreconditions:
    train_runtime: LearningBBoxDataLoaderRuntime
    eval_runtimes_by_box_id: Mapping[str, LearningBBoxEvalRuntime]
    num_classes: int
    available_gpu_count: int
    device_ids: Tuple[int, ...]


def _resolve_shared_num_classes_from_eval_runtimes(
    eval_runtimes_by_box_id: Mapping[str, LearningBBoxEvalRuntime],
) -> int:
    if not isinstance(eval_runtimes_by_box_id, Mapping):
        raise TypeError(
            "eval_runtimes_by_box_id must be a mapping of box_id -> LearningBBoxEvalRuntime, "
            f"got {type(eval_runtimes_by_box_id).__name__}"
        )
    if not eval_runtimes_by_box_id:
        raise ValueError("No evaluation runtimes/buffers are available in session storage.")

    resolved_num_classes: Optional[int] = None
    for box_id, runtime in tuple(eval_runtimes_by_box_id.items()):
        if not isinstance(runtime, LearningBBoxEvalRuntime):
            raise TypeError(
                "eval_runtimes_by_box_id values must be LearningBBoxEvalRuntime, "
                f"got {type(runtime).__name__} for id={box_id!r}"
            )
        buffer_obj = runtime.buffer
        if not hasattr(buffer_obj, "num_classes"):
            raise ValueError(
                f"Evaluation buffer for box_id={box_id!r} does not expose 'num_classes'."
            )
        raw_num_classes = getattr(buffer_obj, "num_classes")
        num_classes = _coerce_positive_int(raw_num_classes, name="num_classes")
        if resolved_num_classes is None:
            resolved_num_classes = num_classes
            continue
        if num_classes != resolved_num_classes:
            raise ValueError(
                "All evaluation buffers must share the same num_classes; "
                f"expected {resolved_num_classes}, got {num_classes} for box_id={box_id!r}."
            )

    if resolved_num_classes is None:
        raise ValueError("No evaluation buffer num_classes could be resolved.")
    return int(resolved_num_classes)


def validate_foundation_model_instantiation_preconditions(
    *,
    train_runtime: Optional[LearningBBoxDataLoaderRuntime] = None,
    eval_runtimes_by_box_id: Optional[Mapping[str, LearningBBoxEvalRuntime]] = None,
    require_min_gpu_count: int = 2,
    torch_module: Optional[object] = None,
) -> FoundationInstantiationPreconditions:
    torch_mod = _require_torch() if torch_module is None else torch_module
    normalized_min_gpu_count = _coerce_positive_int(
        require_min_gpu_count,
        name="require_min_gpu_count",
    )

    resolved_train_runtime = (
        get_current_learning_dataloader_runtime()
        if train_runtime is None
        else train_runtime
    )
    if resolved_train_runtime is None:
        raise ValueError(
            "No training dataloader runtime is available in session storage. "
            "Build datasets from bounding boxes first."
        )
    if not isinstance(resolved_train_runtime, LearningBBoxDataLoaderRuntime):
        raise TypeError(
            "train_runtime must be a LearningBBoxDataLoaderRuntime, "
            f"got {type(resolved_train_runtime).__name__}"
        )

    resolved_eval_runtimes = (
        get_current_learning_eval_runtimes_by_box_id()
        if eval_runtimes_by_box_id is None
        else dict(eval_runtimes_by_box_id)
    )
    resolved_num_classes = _resolve_shared_num_classes_from_eval_runtimes(
        resolved_eval_runtimes
    )

    available_gpu_count = int(getattr(getattr(torch_mod, "cuda"), "device_count")())
    if available_gpu_count < normalized_min_gpu_count:
        raise RuntimeError(
            f"At least {normalized_min_gpu_count} CUDA devices are required to instantiate the model; "
            f"found {available_gpu_count}."
        )
    device_ids = tuple(range(available_gpu_count))

    return FoundationInstantiationPreconditions(
        train_runtime=resolved_train_runtime,
        eval_runtimes_by_box_id=dict(resolved_eval_runtimes),
        num_classes=int(resolved_num_classes),
        available_gpu_count=available_gpu_count,
        device_ids=device_ids,
    )


def _extract_encoder_weights(checkpoint_state: Mapping[str, object]) -> Dict[str, object]:
    if not isinstance(checkpoint_state, Mapping):
        raise TypeError(
            "checkpoint_state must be a mapping containing 'encoder_weights', "
            f"got {type(checkpoint_state).__name__}"
        )

    raw_encoder_weights = checkpoint_state.get("encoder_weights")
    if not isinstance(raw_encoder_weights, Mapping):
        raise KeyError("checkpoint_state['encoder_weights'] must be a mapping")

    normalized: Dict[str, object] = {}
    for raw_key, raw_value in tuple(raw_encoder_weights.items()):
        key = str(raw_key)
        if key.startswith("module."):
            key = key[7:]
        if key in {"norm.weight", "norm.bias"}:
            continue
        normalized[key] = raw_value
    return normalized


def _shape_tuple(value: object) -> Optional[Tuple[int, ...]]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dim) for dim in tuple(shape))
    except Exception:
        return None


def _validate_encoder_weights_match_model(
    model: object,
    encoder_weights: Mapping[str, object],
) -> None:
    named_parameters_method = getattr(model, "named_parameters", None)
    if not callable(named_parameters_method):
        raise TypeError("model must define a callable named_parameters()")

    expected_params = {
        str(name): parameter for name, parameter in tuple(named_parameters_method())
    }
    if not expected_params:
        raise ValueError("model named_parameters() returned no encoder parameters")

    expected_keys = set(expected_params.keys())
    observed_keys = set(str(key) for key in tuple(encoder_weights.keys()))
    missing_keys = sorted(expected_keys - observed_keys)
    extra_keys = sorted(observed_keys - expected_keys)
    if missing_keys or extra_keys:
        message_parts = [
            "Checkpoint encoder weights do not match UnetrPs8MulticlassesSmall architecture."
        ]
        if missing_keys:
            preview = ", ".join(missing_keys[:8])
            if len(missing_keys) > 8:
                preview = f"{preview}, ..."
            message_parts.append(f"Missing keys: {preview}")
        if extra_keys:
            preview = ", ".join(extra_keys[:8])
            if len(extra_keys) > 8:
                preview = f"{preview}, ..."
            message_parts.append(f"Unexpected keys: {preview}")
        raise ValueError(" ".join(message_parts))

    shape_mismatches = []
    for key in sorted(expected_keys):
        expected_shape = _shape_tuple(expected_params[key])
        observed_shape = _shape_tuple(encoder_weights[key])
        if expected_shape != observed_shape:
            shape_mismatches.append((key, expected_shape, observed_shape))
    if shape_mismatches:
        key, expected_shape, observed_shape = shape_mismatches[0]
        raise ValueError(
            "Checkpoint encoder weights do not match UnetrPs8MulticlassesSmall architecture. "
            f"Shape mismatch for key '{key}': expected {expected_shape}, got {observed_shape}."
        )


def _extract_optional_full_model_weights(
    checkpoint_state: Mapping[str, object],
) -> Optional[Dict[str, object]]:
    for key in ("model_state_dict", "state_dict", "model_weights", "weights"):
        raw_value = checkpoint_state.get(key)
        if raw_value is None:
            continue
        if not isinstance(raw_value, Mapping):
            raise TypeError(
                f"checkpoint_state['{key}'] must be a mapping when present"
            )
        return {str(weight_key): weight for weight_key, weight in tuple(raw_value.items())}
    return None


def _extract_training_provenance_from_checkpoint_state(
    checkpoint_state: Mapping[str, object],
    *,
    resolved_checkpoint_path: str,
) -> Dict[str, object]:
    source_checkpoint_path = str(resolved_checkpoint_path)
    trained_in_app = False
    training_run_count = 0

    metadata_obj = checkpoint_state.get("metadata")
    if isinstance(metadata_obj, Mapping):
        metadata_hyperparameters = metadata_obj.get("hyperparameters")
        if isinstance(metadata_hyperparameters, Mapping):
            raw_source = metadata_hyperparameters.get("source_checkpoint_path")
            if isinstance(raw_source, str) and raw_source.strip():
                try:
                    source_checkpoint_path = _coerce_non_empty_path(
                        raw_source,
                        name="source_checkpoint_path",
                    )
                except Exception:
                    source_checkpoint_path = str(resolved_checkpoint_path)

            raw_trained = metadata_hyperparameters.get("trained_in_app")
            if isinstance(raw_trained, bool):
                trained_in_app = bool(raw_trained)

            raw_run_count = metadata_hyperparameters.get("training_run_count")
            if isinstance(raw_run_count, Integral) and not isinstance(raw_run_count, bool):
                if int(raw_run_count) >= 0:
                    training_run_count = int(raw_run_count)

    if training_run_count > 0 and not trained_in_app:
        trained_in_app = True
    if trained_in_app and training_run_count <= 0:
        training_run_count = 1

    return {
        "source_checkpoint_path": str(source_checkpoint_path),
        "trained_in_app": bool(trained_in_app),
        "training_run_count": int(training_run_count),
    }


def _adapt_state_dict_to_target_keys(
    state_dict: Mapping[str, object],
    target_keys: Sequence[str],
) -> Optional[Dict[str, object]]:
    normalized = {str(key): value for key, value in tuple(state_dict.items())}
    source_keys = set(normalized.keys())
    expected_keys = {str(key) for key in tuple(target_keys)}

    if source_keys == expected_keys:
        return dict(normalized)

    if source_keys and all(key.startswith("module.") for key in source_keys):
        stripped = {
            key[7:]: value for key, value in tuple(normalized.items())
        }
        if set(stripped.keys()) == expected_keys:
            return stripped

    if source_keys and all(not key.startswith("module.") for key in source_keys):
        prefixed = {
            f"module.{key}": value for key, value in tuple(normalized.items())
        }
        if set(prefixed.keys()) == expected_keys:
            return prefixed

    return None


def _resolve_encoder_parameter_names_for_checkpoint(
    *,
    named_parameter_names: Sequence[str],
    runtime_hyperparameters: Mapping[str, object],
) -> Tuple[str, ...]:
    ordered_names = tuple(str(name) for name in tuple(named_parameter_names))
    if not ordered_names:
        raise ValueError("Runtime model/module returned no parameters.")

    raw_encoder_count = runtime_hyperparameters.get("encoder_parameter_count")
    try:
        if raw_encoder_count is not None:
            normalized_encoder_count = int(raw_encoder_count)
            if 0 < normalized_encoder_count <= len(ordered_names):
                return tuple(ordered_names[:normalized_encoder_count])
    except Exception:
        pass

    return tuple(
        name
        for name in ordered_names
        if not any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in _DECODER_PARAMETER_PREFIXES
        )
    )


def _extract_encoder_weights_for_checkpoint(
    model_obj: object,
    *,
    runtime_hyperparameters: Mapping[str, object],
) -> Dict[str, object]:
    inner_model = getattr(model_obj, "module", model_obj)
    inner_state_dict_method = getattr(inner_model, "state_dict", None)
    if not callable(inner_state_dict_method):
        raise TypeError("Runtime model/module must define a callable state_dict().")
    inner_state_dict = inner_state_dict_method()
    if not isinstance(inner_state_dict, Mapping):
        raise TypeError(
            "Runtime model/module state_dict() must return a mapping, "
            f"got {type(inner_state_dict).__name__}."
        )

    named_parameters_method = getattr(inner_model, "named_parameters", None)
    if not callable(named_parameters_method):
        raise TypeError("Runtime model/module must define a callable named_parameters().")
    named_parameter_names = tuple(
        str(name) for name, _param in tuple(named_parameters_method())
    )
    encoder_parameter_names = _resolve_encoder_parameter_names_for_checkpoint(
        named_parameter_names=named_parameter_names,
        runtime_hyperparameters=runtime_hyperparameters,
    )
    if not encoder_parameter_names:
        raise ValueError("Could not resolve encoder parameter names from runtime model.")

    missing_encoder_keys = [
        name for name in encoder_parameter_names if name not in inner_state_dict
    ]
    if missing_encoder_keys:
        preview = ", ".join(missing_encoder_keys[:8])
        if len(missing_encoder_keys) > 8:
            preview = f"{preview}, ..."
        raise ValueError(
            "Runtime model state_dict is missing encoder parameters: "
            f"{preview}"
        )

    return {
        str(name): inner_state_dict[name]
        for name in tuple(encoder_parameter_names)
    }


def save_foundation_model_checkpoint(
    *,
    runtime: LearningModelRuntime,
    checkpoint_path: str,
    torch_module: Optional[object] = None,
    saved_at_utc: Optional[str] = None,
) -> str:
    if not isinstance(runtime, LearningModelRuntime):
        raise TypeError(
            "runtime must be a LearningModelRuntime, "
            f"got {type(runtime).__name__}"
        )

    torch_mod = _require_torch() if torch_module is None else torch_module
    save_method = getattr(torch_mod, "save", None)
    if not callable(save_method):
        raise TypeError("torch_module must define a callable save(payload, path)")

    resolved_checkpoint_path = _coerce_non_empty_path(
        checkpoint_path,
        name="checkpoint_path",
    )
    if Path(resolved_checkpoint_path).suffix.lower() != ".cp":
        raise ValueError("Model checkpoints must use the .cp extension.")
    resolved_path_obj = Path(resolved_checkpoint_path)
    parent_directory = resolved_path_obj.parent
    if not parent_directory.exists() or not parent_directory.is_dir():
        raise ValueError(
            f"Checkpoint directory does not exist: {parent_directory}"
        )

    model_obj = runtime.model
    full_state_dict_method = getattr(model_obj, "state_dict", None)
    if not callable(full_state_dict_method):
        raise TypeError("Runtime model must define a callable state_dict().")
    full_state_dict = full_state_dict_method()
    if not isinstance(full_state_dict, Mapping):
        raise TypeError(
            "Runtime model state_dict() must return a mapping, "
            f"got {type(full_state_dict).__name__}."
        )

    runtime_hyperparameters = dict(runtime.hyperparameters)
    runtime_hyperparameters.setdefault(
        "source_checkpoint_path",
        str(runtime.checkpoint_path),
    )
    runtime_hyperparameters.setdefault("trained_in_app", False)
    runtime_hyperparameters.setdefault("training_run_count", 0)
    encoder_weights = _extract_encoder_weights_for_checkpoint(
        model_obj,
        runtime_hyperparameters=runtime_hyperparameters,
    )

    inner_model = getattr(model_obj, "module", model_obj)
    architecture_name = str(
        getattr(getattr(inner_model, "__class__", object), "__name__", "UnknownModel")
    ).strip()
    if not architecture_name:
        architecture_name = "UnknownModel"

    timestamp = datetime.now(timezone.utc).isoformat()
    if saved_at_utc is not None:
        if not isinstance(saved_at_utc, str) or not saved_at_utc.strip():
            raise ValueError("saved_at_utc must be a non-empty string when provided")
        timestamp = saved_at_utc.strip()

    checkpoint_payload = {
        "encoder_weights": encoder_weights,
        "state_dict": dict(full_state_dict),
        "metadata": {
            "format_version": 1,
            "architecture": architecture_name,
            "saved_at_utc": timestamp,
            "num_classes": int(runtime.num_classes),
            "device_ids": tuple(runtime.device_ids),
            "checkpoint_path": str(runtime.checkpoint_path),
            "hyperparameters": runtime_hyperparameters,
        },
    }

    save_method(checkpoint_payload, str(resolved_path_obj))
    return str(resolved_path_obj)


def _build_layerwise_optimizer_groups(
    named_parameters: Sequence[Tuple[str, object]],
    *,
    decoder_parameter_count: int,
    block_count: int,
    base_lr: float,
    lwise_lr_decay: float,
    weight_decay: float,
) -> Tuple[Dict[str, object], ...]:
    if decoder_parameter_count < 0:
        raise ValueError("decoder_parameter_count must be >= 0")
    if block_count <= 0:
        raise ValueError("block_count must be >= 1")

    names_in_order = [name for name, _param in tuple(named_parameters)]
    params_by_name = {name: param for name, param in tuple(named_parameters)}
    names_in_order.reverse()

    cur_decay_rate = 1.0
    cur_block_num = int(block_count)
    groups = []

    for idx, name in enumerate(names_in_order):
        param = params_by_name[name]
        if not bool(getattr(param, "requires_grad", False)):
            continue

        if idx < int(decoder_parameter_count):
            lr_decay_rate = 1.0
        else:
            lr_decay_rate = float(cur_decay_rate)
            block_match = _BLOCK_INDEX_PATTERN.search(name)
            if block_match is not None:
                block_num = int(block_match.group(1))
                if block_num != cur_block_num:
                    cur_block_num = block_num
                    cur_decay_rate = float(cur_decay_rate) * float(lwise_lr_decay)
            else:
                if "emb" not in name:
                    raise AssertionError(
                        "Expected encoder parameter name to include 'emb' or a blocks.<idx> component, "
                        f"got {name!r}"
                    )

        groups.append(
            {
                "params": [param],
                "lr": float(base_lr) * float(lr_decay_rate),
                "weight_decay": float(weight_decay),
                "lwise_lr_decay_rate": float(lr_decay_rate),
            }
        )
    return tuple(groups)


def instantiate_foundation_model_runtime(
    *,
    num_classes: int,
    config: FoundationModelConfig = DEFAULT_FOUNDATION_MODEL_CONFIG,
    checkpoint_path: Optional[str] = None,
    device_ids: Optional[Sequence[object]] = None,
    store_in_session: bool = True,
    model_factory: Optional[Callable[..., object]] = None,
    checkpoint_loader: Optional[Callable[..., Mapping[str, object]]] = None,
    data_parallel_factory: Optional[Callable[..., object]] = None,
    optimizer_factory: Optional[Callable[..., object]] = None,
    torch_module: Optional[object] = None,
) -> LearningModelRuntime:
    torch_mod = _require_torch() if torch_module is None else torch_module
    if not isinstance(store_in_session, bool):
        raise TypeError(f"store_in_session must be a bool, got {type(store_in_session).__name__}")

    if not isinstance(config, FoundationModelConfig):
        raise TypeError(
            "config must be a FoundationModelConfig, "
            f"got {type(config).__name__}"
        )
    normalized_num_classes = _coerce_positive_int(num_classes, name="num_classes")
    resolved_checkpoint_path = _coerce_non_empty_path(
        config.checkpoint_path if checkpoint_path is None else checkpoint_path,
        name="checkpoint_path",
    )
    if Path(resolved_checkpoint_path).suffix.lower() != ".cp":
        raise ValueError("Model checkpoints must use the .cp extension.")

    if device_ids is None:
        available_devices = int(getattr(getattr(torch_mod, "cuda"), "device_count")())
        if available_devices <= 0:
            raise RuntimeError("No CUDA devices are available for foundation model instantiation.")
        resolved_device_ids = tuple(range(available_devices))
    else:
        resolved_device_ids = _coerce_device_ids(tuple(device_ids))

    base_device = getattr(torch_mod, "device")(f"cuda:{resolved_device_ids[0]}")

    if model_factory is None:
        model_factory = UnetrPs8MulticlassesSmall
    model = model_factory(
        config.minivol_size,
        in_chans=config.in_chans,
        patch_size=config.patch_size,
        embed_dim=config.enc_emb_dim,
        depth=config.enc_depth,
        num_heads=config.num_heads,
        num_classes=normalized_num_classes,
    )

    if checkpoint_loader is None:
        checkpoint_loader = lambda path, *, map_location: torch_mod.load(  # noqa: E731
            path,
            map_location=map_location,
        )
    checkpoint_state = checkpoint_loader(
        resolved_checkpoint_path,
        map_location=base_device,
    )
    training_provenance = _extract_training_provenance_from_checkpoint_state(
        checkpoint_state,
        resolved_checkpoint_path=resolved_checkpoint_path,
    )
    encoder_weights = _extract_encoder_weights(checkpoint_state)
    _validate_encoder_weights_match_model(model, encoder_weights)
    load_state_dict = getattr(model, "load_state_dict", None)
    if not callable(load_state_dict):
        raise TypeError("model_factory result must define a callable load_state_dict")
    load_state_dict(encoder_weights)

    encoder_parameter_count = sum(1 for _name, _param in tuple(model.named_parameters()))

    build_decoder = getattr(model, "build_decoder", None)
    if not callable(build_decoder):
        raise TypeError("model_factory result must define a callable build_decoder")
    build_decoder()

    train_method = getattr(model, "train", None)
    if callable(train_method):
        train_method()
    to_method = getattr(model, "to", None)
    if not callable(to_method):
        raise TypeError("model_factory result must define a callable to(device)")
    to_method(base_device)

    if data_parallel_factory is None:
        data_parallel_factory = getattr(getattr(torch_mod, "nn"), "DataParallel")
    parallel_model = data_parallel_factory(model, list(resolved_device_ids))

    full_model_restore_applied = False
    full_model_weights = _extract_optional_full_model_weights(checkpoint_state)
    if full_model_weights is not None:
        parallel_state_dict_method = getattr(parallel_model, "state_dict", None)
        parallel_load_state_dict = getattr(parallel_model, "load_state_dict", None)
        if callable(parallel_state_dict_method) and callable(parallel_load_state_dict):
            target_state_dict = parallel_state_dict_method()
            if isinstance(target_state_dict, Mapping):
                adapted_full_weights = _adapt_state_dict_to_target_keys(
                    full_model_weights,
                    tuple(str(key) for key in tuple(target_state_dict.keys())),
                )
                if adapted_full_weights is not None:
                    try:
                        parallel_load_state_dict(adapted_full_weights, strict=True)
                        full_model_restore_applied = True
                    except Exception:
                        warnings.warn(
                            (
                                "Checkpoint contains full-model weights, but strict full restore failed. "
                                "Falling back to encoder-only restore."
                            ),
                            RuntimeWarning,
                        )
            else:
                warnings.warn(
                    (
                        "Parallel model state_dict() did not return a mapping. "
                        "Falling back to encoder-only restore."
                    ),
                    RuntimeWarning,
                )

    named_parameters = tuple(parallel_model.named_parameters())
    decoder_parameter_count = int(len(named_parameters)) - int(encoder_parameter_count)
    if decoder_parameter_count < 0:
        raise RuntimeError(
            "Decoder parameter count cannot be negative after build_decoder: "
            f"encoder={encoder_parameter_count}, total={len(named_parameters)}"
        )
    blocks = getattr(getattr(parallel_model, "module", None), "blocks", None)
    if blocks is None:
        raise TypeError("Parallel model must expose module.blocks for layer-wise LR decay.")
    block_count = int(len(blocks))
    optimizer_groups = _build_layerwise_optimizer_groups(
        named_parameters,
        decoder_parameter_count=decoder_parameter_count,
        block_count=block_count,
        base_lr=config.lr,
        lwise_lr_decay=config.lwise_lr_decay,
        weight_decay=config.weight_decay,
    )

    if optimizer_factory is None:
        optimizer_factory = getattr(getattr(torch_mod, "optim"), "AdamW")
    optimizer = optimizer_factory(optimizer_groups)

    hyperparameters = {
        "enc_depth": int(config.enc_depth),
        "enc_emb_dim": int(config.enc_emb_dim),
        "minivol_size": int(config.minivol_size),
        "patch_size": int(config.patch_size),
        "num_heads": int(config.num_heads),
        "lr": float(config.lr),
        "lwise_lr_decay": float(config.lwise_lr_decay),
        "weight_decay": float(config.weight_decay),
        "encoder_parameter_count": int(encoder_parameter_count),
        "decoder_parameter_count": int(decoder_parameter_count),
        "full_model_restore_applied": bool(full_model_restore_applied),
        "source_checkpoint_path": str(training_provenance["source_checkpoint_path"]),
        "trained_in_app": bool(training_provenance["trained_in_app"]),
        "training_run_count": int(training_provenance["training_run_count"]),
    }

    if bool(store_in_session):
        return set_current_learning_model_components(
            model=parallel_model,
            optimizer=optimizer,
            checkpoint_path=resolved_checkpoint_path,
            device_ids=resolved_device_ids,
            num_classes=normalized_num_classes,
            hyperparameters=hyperparameters,
        )

    return LearningModelRuntime(
        model=parallel_model,
        optimizer=optimizer,
        checkpoint_path=resolved_checkpoint_path,
        device_ids=resolved_device_ids,
        num_classes=normalized_num_classes,
        hyperparameters=hyperparameters,
    )
