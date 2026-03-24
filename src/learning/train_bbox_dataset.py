from __future__ import annotations

import itertools
from numbers import Integral, Real
from random import randrange, uniform
from typing import Sequence, Tuple


try:  # pragma: no cover - import availability is environment dependent
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - import availability is environment dependent
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[override]
        pass


def _require_torch():
    if torch is None or F is None:  # pragma: no cover - environment dependent
        raise ImportError("PyTorch is required for TrainBBoxDataset")
    return torch


def _coerce_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be >= 1, got {normalized}")
    return normalized


def _coerce_factor(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    return float(value)


class TrainBBoxDataset(Dataset):
    """Random 3D patch dataset backed by in-memory (raw, segmentation) tensor pairs.

    Notes:
    - `__getitem__(idx)` uses `idx` directly as the selected source volume index.
      This is intended to be used with `WeightedRandomSampler(weights, len(dataset))`,
      where `weights` length equals the number of source volumes.
    - `__len__` is controlled by `minivol_per_epoch` and defines how many patches
      are sampled per epoch.
    """

    def __init__(
        self,
        tensor_pairs: Sequence[Tuple[object, object]],
        minivol_size: int = 200,
        minivol_per_epoch: int = 1024,
        contr_bright_factors: Sequence[float] = (0.5, 0.5),
    ) -> None:
        torch_mod = _require_torch()

        if not isinstance(tensor_pairs, Sequence):
            raise TypeError(
                "tensor_pairs must be a sequence of (raw_tensor, segmentation_tensor) tuples"
            )
        if not tensor_pairs:
            raise ValueError("tensor_pairs must contain at least one volume tuple")

        if not isinstance(contr_bright_factors, Sequence) or len(contr_bright_factors) != 2:
            raise ValueError("contr_bright_factors must be a sequence of length 2")

        self.minivol_size = _coerce_positive_int(minivol_size, name="minivol_size")
        self.minivol_per_epoch = _coerce_positive_int(
            minivol_per_epoch, name="minivol_per_epoch"
        )
        self.contr_fact = _coerce_factor(contr_bright_factors[0], name="contr_fact")
        self.bright_fact = _coerce_factor(contr_bright_factors[1], name="bright_fact")

        if self.contr_fact < 0:
            raise ValueError(f"contr_fact must be >= 0, got {self.contr_fact}")
        if self.bright_fact < 0:
            raise ValueError(f"bright_fact must be >= 0, got {self.bright_fact}")

        self.raw_tensors = []
        self.annot_tensors = []
        self.weights = []

        for pair_index, pair in enumerate(tensor_pairs):
            if not isinstance(pair, Sequence) or len(pair) != 2:
                raise TypeError(
                    "Each item in tensor_pairs must be a tuple(raw_tensor, segmentation_tensor), "
                    f"got {type(pair).__name__}"
                )
            raw_tensor, annot_tensor = pair[0], pair[1]

            if not isinstance(raw_tensor, torch_mod.Tensor):
                raise TypeError(
                    f"raw tensor at index {pair_index} must be a torch.Tensor, "
                    f"got {type(raw_tensor).__name__}"
                )
            if not isinstance(annot_tensor, torch_mod.Tensor):
                raise TypeError(
                    f"segmentation tensor at index {pair_index} must be a torch.Tensor, "
                    f"got {type(annot_tensor).__name__}"
                )
            if raw_tensor.device.type != "cpu" or annot_tensor.device.type != "cpu":
                raise ValueError(
                    f"tensor pair at index {pair_index} must be stored on CPU, "
                    f"got raw={raw_tensor.device}, seg={annot_tensor.device}"
                )
            if raw_tensor.ndim != 3 or annot_tensor.ndim != 3:
                raise ValueError(
                    f"tensor pair at index {pair_index} must be 3D, "
                    f"got raw_ndim={raw_tensor.ndim}, seg_ndim={annot_tensor.ndim}"
                )
            if tuple(int(v) for v in raw_tensor.shape) != tuple(int(v) for v in annot_tensor.shape):
                raise ValueError(
                    "raw and segmentation tensor shapes must match at index "
                    f"{pair_index}: raw_shape={tuple(raw_tensor.shape)} "
                    f"seg_shape={tuple(annot_tensor.shape)}"
                )

            shape = tuple(int(v) for v in raw_tensor.shape)
            if any(axis_size < self.minivol_size for axis_size in shape):
                raise ValueError(
                    "minivol_size must be <= each volume axis for every input tensor pair; "
                    f"pair_index={pair_index}, shape={shape}, minivol_size={self.minivol_size}"
                )

            # Match legacy behavior: normalize each source raw volume independently.
            raw_float = raw_tensor.to(dtype=torch_mod.float32)
            raw_mean = raw_float.mean()
            raw_std = raw_float.std()
            if float(raw_std.item()) == 0.0:
                raw_std = torch_mod.tensor(1.0, dtype=torch_mod.float32)
            normalized_raw = (raw_float - raw_mean) / raw_std
            normalized_annot = annot_tensor.to(dtype=torch_mod.long)

            self.raw_tensors.append(normalized_raw)
            self.annot_tensors.append(normalized_annot)
            self.weights.append(int(shape[0] * shape[1] * shape[2]))

        self.SYM_GROUP = []
        for perm in itertools.permutations([0, 1, 2]):
            for flip_pattern in itertools.product([0, 1], repeat=3):
                flips = [ax for ax, flip in enumerate(flip_pattern) if flip == 1]
                self.SYM_GROUP.append((perm, flips))

    @property
    def volume_count(self) -> int:
        return int(len(self.raw_tensors))

    def __len__(self) -> int:
        return int(self.minivol_per_epoch)

    def apply_symmetry(self, x, perm, flips):
        x = x.permute(perm)
        for axis in flips:
            x = x.flip(axis)
        return x

    def geom_transform(self, raw_minivol, annot_minivol):
        torch_mod = _require_torch()
        k = torch_mod.randint(0, len(self.SYM_GROUP), (1,)).item()
        perm, flips = self.SYM_GROUP[k]
        return self.apply_symmetry(raw_minivol, perm, flips), self.apply_symmetry(
            annot_minivol, perm, flips
        )

    def elastic_transform_3d(
        self,
        raw_minivol,
        annot_minivol,
        alpha_max: float = 50.0,
        sigma_min: float = 3.0,
        sigma_max: float = 6.0,
    ):
        torch_mod = _require_torch()

        if tuple(int(v) for v in raw_minivol.shape) != tuple(int(v) for v in annot_minivol.shape):
            raise ValueError("raw_minivol and annot_minivol must have the same shape")
        if raw_minivol.dtype != torch_mod.float32:
            raise ValueError(f"raw_minivol must be torch.float32, got {raw_minivol.dtype}")
        if annot_minivol.dtype != torch_mod.long:
            raise ValueError(f"annot_minivol must be torch.long, got {annot_minivol.dtype}")

        device = raw_minivol.device
        d_size, h_size, w_size = raw_minivol.shape

        alpha = torch_mod.rand(1, device=device).item() * float(alpha_max)
        sigma = float(sigma_min) + torch_mod.rand(1, device=device).item() * float(
            sigma_max - sigma_min
        )
        if alpha == 0.0:
            return raw_minivol.clone(), annot_minivol.clone()

        raw = raw_minivol.unsqueeze(0).unsqueeze(0)
        annot = annot_minivol.unsqueeze(0).unsqueeze(0).float()

        z = torch_mod.linspace(-1, 1, d_size, device=device)
        y = torch_mod.linspace(-1, 1, h_size, device=device)
        x = torch_mod.linspace(-1, 1, w_size, device=device)
        zz, yy, xx = torch_mod.meshgrid(z, y, x, indexing="ij")
        base_grid = torch_mod.stack((xx, yy, zz), dim=-1).unsqueeze(0)

        displacement = torch_mod.randn((1, 3, d_size, h_size, w_size), device=device)

        def gaussian_kernel_1d(size: int, sigma_value: float):
            coords = torch_mod.arange(size, device=device) - size // 2
            kernel = torch_mod.exp(-(coords ** 2) / (2 * sigma_value ** 2))
            kernel /= kernel.sum()
            return kernel

        kernel_size = int(2 * round(2 * sigma) + 1)
        kernel_1d = gaussian_kernel_1d(kernel_size, sigma)

        for axis in range(3):
            kernel = kernel_1d.view(
                1,
                1,
                kernel_size if axis == 0 else 1,
                kernel_size if axis == 1 else 1,
                kernel_size if axis == 2 else 1,
            )
            padding = [0, 0, 0]
            padding[axis] = kernel_size // 2
            displacement = F.conv3d(
                displacement,
                kernel.repeat(3, 1, 1, 1, 1),
                padding=padding,
                groups=3,
            )

        displacement = displacement.permute(0, 2, 3, 4, 1)
        displacement = displacement * alpha / torch_mod.tensor([w_size, h_size, d_size], device=device)
        grid = base_grid + displacement

        raw_out = F.grid_sample(
            raw,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        annot_out = F.grid_sample(
            annot,
            grid,
            mode="nearest",
            padding_mode="border",
            align_corners=True,
        )

        return raw_out.squeeze(0).squeeze(0), annot_out.squeeze(0).squeeze(0).long()

    def __getitem__(self, idx):
        if isinstance(idx, bool) or not isinstance(idx, Integral):
            raise TypeError(f"idx must be an integer, got {type(idx).__name__}")
        selected_index = int(idx)
        if selected_index < 0 or selected_index >= len(self.raw_tensors):
            raise IndexError(
                f"idx out of range for source volume selection: idx={selected_index}, "
                f"volume_count={len(self.raw_tensors)}"
            )

        picked_raw_array = self.raw_tensors[selected_index]
        picked_annot_array = self.annot_tensors[selected_index]

        start_z = randrange(int(picked_raw_array.shape[0]) - self.minivol_size + 1)
        start_x = randrange(int(picked_raw_array.shape[1]) - self.minivol_size + 1)
        start_y = randrange(int(picked_raw_array.shape[2]) - self.minivol_size + 1)

        extracted_minivol = picked_raw_array[
            start_z : start_z + self.minivol_size,
            start_x : start_x + self.minivol_size,
            start_y : start_y + self.minivol_size,
        ]
        extracted_annot = picked_annot_array[
            start_z : start_z + self.minivol_size,
            start_x : start_x + self.minivol_size,
            start_y : start_y + self.minivol_size,
        ]
        extracted_minivol, extracted_annot = self.geom_transform(extracted_minivol, extracted_annot)
        extracted_minivol, extracted_annot = self.elastic_transform_3d(
            extracted_minivol, extracted_annot
        )
        extracted_minivol = extracted_minivol.unsqueeze(0)

        if self.contr_fact != 1 or self.bright_fact != 0:
            cur_contr = uniform(self.contr_fact, 1)
            cur_bright = uniform(-self.bright_fact, self.bright_fact)
            extracted_minivol = extracted_minivol * cur_contr + cur_bright

        return extracted_minivol, extracted_annot
