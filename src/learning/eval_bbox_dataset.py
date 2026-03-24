from __future__ import annotations

from numbers import Integral
from typing import Dict, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - import availability is environment dependent
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - import availability is environment dependent
    torch = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[override]
        pass


_MASK_LABEL = -100


def _require_torch():
    if torch is None:  # pragma: no cover - environment dependent
        raise ImportError("PyTorch is required for EvalBBoxDataset/DestVolBuffer")
    return torch


def _coerce_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be >= 1, got {normalized}")
    return normalized


def _coerce_label_values(values: Sequence[object]) -> Tuple[int, ...]:
    if not isinstance(values, Sequence):
        raise TypeError(f"label_values must be a sequence, got {type(values).__name__}")
    normalized = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(
                "label_values must contain integers only, "
                f"got {type(value).__name__}"
            )
        integer = int(value)
        if integer == _MASK_LABEL:
            raise ValueError("label_values must not include -100 (reserved mask value)")
        if integer not in normalized:
            normalized.append(integer)
    if not normalized:
        raise ValueError("label_values must contain at least one class label")
    return tuple(normalized)


def _build_hann_window(*, minivol_size: int):
    torch_mod = _require_torch()
    hann_z = 0.5 * (
        1
        - np.cos(
            2
            * np.pi
            * np.arange(minivol_size)
            / float(minivol_size - 1)
        )
    )
    hann_x = 0.5 * (
        1
        - np.cos(
            2
            * np.pi
            * np.arange(minivol_size)
            / float(minivol_size - 1)
        )
    )
    hann_y = 0.5 * (
        1
        - np.cos(
            2
            * np.pi
            * np.arange(minivol_size)
            / float(minivol_size - 1)
        )
    )
    return torch_mod.tensor(
        np.outer(hann_x, hann_y)[:, :, np.newaxis] * hann_z[np.newaxis, np.newaxis, :],
        dtype=torch_mod.float32,
    ).unsqueeze(0)


def _add_weighted_batch_to_buffer(
    *,
    batch,
    batch_coordinates,
    buffer_vol,
    hann_window,
    minivol_size: int,
    num_classes: int,
) -> None:
    if batch.ndim != 5:
        raise ValueError(f"batch must be 5D [B, C, D, H, W], got ndim={batch.ndim}")
    if int(batch.shape[1]) != int(num_classes):
        raise ValueError(
            f"batch channel count ({int(batch.shape[1])}) must match num_classes ({num_classes})"
        )
    for i in range(int(batch.shape[0])):
        minivol = batch[i, :, :, :, :]
        minivol_coordinates = [
            int(batch_coordinates[0][i]),
            int(batch_coordinates[1][i]),
            int(batch_coordinates[2][i]),
        ]
        minivol = minivol.to(dtype=buffer_vol.dtype, device=buffer_vol.device)
        minivol = minivol * hann_window

        z0 = minivol_coordinates[0]
        x0 = minivol_coordinates[1]
        y0 = minivol_coordinates[2]
        z1 = z0 + int(minivol_size)
        x1 = x0 + int(minivol_size)
        y1 = y0 + int(minivol_size)
        buffer_vol[:, z0:z1, x0:x1, y0:y1] = (
            minivol + buffer_vol[:, z0:z1, x0:x1, y0:y1]
        )


def _decode_buffer_labels(buffer_vol, channel_index_to_label: Sequence[int], *, dtype):
    torch_mod = _require_torch()
    pred_channel = torch_mod.argmax(buffer_vol, dim=0)
    lookup = torch_mod.tensor(
        tuple(int(v) for v in tuple(channel_index_to_label)),
        dtype=dtype,
        device=pred_channel.device,
    )
    return lookup[pred_channel]


class EvalBBoxDataset(Dataset):
    def __init__(self, vol, minivol_size: int = 200) -> None:
        torch_mod = _require_torch()
        if not isinstance(vol, torch_mod.Tensor):
            raise TypeError(f"vol must be a torch.Tensor, got {type(vol).__name__}")
        if vol.ndim != 3:
            raise ValueError(f"vol must be 3D, got ndim={vol.ndim}")

        self.vol = vol
        self.minivol_size = _coerce_positive_int(minivol_size, name="minivol_size")
        self.volume_shape = tuple(int(v) for v in self.vol.shape)
        stride = int(self.minivol_size // 2)
        if stride <= 0:
            raise ValueError("minivol_size must be >= 2 for overlap extraction")

        self.nb_minivol_z = (self.volume_shape[0] // stride) - 1
        self.nb_minivol_x = (self.volume_shape[1] // stride) - 1
        self.nb_minivol_y = (self.volume_shape[2] // stride) - 1
        if self.nb_minivol_z <= 0 or self.nb_minivol_x <= 0 or self.nb_minivol_y <= 0:
            raise ValueError(
                "volume is too small for the configured minivol_size and overlap stride"
            )

        self.total_nb_minivol = (
            int(self.nb_minivol_z) * int(self.nb_minivol_x) * int(self.nb_minivol_y)
        )

        ds_mean = self.vol.mean()
        ds_std = self.vol.std()
        if float(ds_std.item()) == 0.0:
            ds_std = torch_mod.tensor(1.0, dtype=self.vol.dtype, device=self.vol.device)
        self.vol = (self.vol - ds_mean) / ds_std

    def __len__(self):
        return int(self.total_nb_minivol)

    def __getitem__(self, idx):
        if isinstance(idx, bool) or not isinstance(idx, Integral):
            raise TypeError(f"idx must be an integer, got {type(idx).__name__}")
        normalized_idx = int(idx)
        if normalized_idx < 0 or normalized_idx >= self.total_nb_minivol:
            raise IndexError(
                f"idx out of range for EvalBBoxDataset: idx={normalized_idx}, "
                f"len={self.total_nb_minivol}"
            )

        idx_z = normalized_idx % self.nb_minivol_z
        idx_x = (normalized_idx // self.nb_minivol_z) % self.nb_minivol_x
        idx_y = (normalized_idx // self.nb_minivol_z) // self.nb_minivol_x

        stride = self.minivol_size // 2
        start_z = int(idx_z * stride)
        start_x = int(idx_x * stride)
        start_y = int(idx_y * stride)

        extracted_minivol = self.vol[
            start_z : start_z + self.minivol_size,
            start_x : start_x + self.minivol_size,
            start_y : start_y + self.minivol_size,
        ]
        extracted_minivol = torch.unsqueeze(extracted_minivol, 0)
        return extracted_minivol, (start_z, start_x, start_y)


class DestVolBuffer:
    def __init__(
        self,
        ground_truth,
        volume_shape,
        label_values: Sequence[object],
        minivol_size: int = 200,
    ) -> None:
        torch_mod = _require_torch()
        if not isinstance(ground_truth, torch_mod.Tensor):
            raise TypeError(
                f"ground_truth must be a torch.Tensor, got {type(ground_truth).__name__}"
            )
        if ground_truth.ndim != 3:
            raise ValueError(f"ground_truth must be 3D, got ndim={ground_truth.ndim}")
        if len(volume_shape) != 3:
            raise ValueError(f"volume_shape must be length 3, got {volume_shape}")

        self.minivol_size = _coerce_positive_int(minivol_size, name="minivol_size")
        self.volume_shape = tuple(int(v) for v in volume_shape)
        self.label_values = _coerce_label_values(label_values)
        self.num_classes = int(len(self.label_values))
        self.label_to_channel_index: Dict[int, int] = {
            int(label): int(i) for i, label in enumerate(self.label_values)
        }
        self.channel_index_to_label: Tuple[int, ...] = tuple(self.label_values)

        self.buffer_vol = torch.zeros(
            [self.num_classes, self.volume_shape[0], self.volume_shape[1], self.volume_shape[2]],
            dtype=torch.float32,
        )
        self.hann_window = _build_hann_window(minivol_size=self.minivol_size)

        self.ground_truth = ground_truth

    def add_batch(self, batch, batch_coordinates):
        _add_weighted_batch_to_buffer(
            batch=batch,
            batch_coordinates=batch_coordinates,
            buffer_vol=self.buffer_vol,
            hann_window=self.hann_window,
            minivol_size=self.minivol_size,
            num_classes=self.num_classes,
        )

    def get_acc_pred(self):
        pred_labels = _decode_buffer_labels(
            self.buffer_vol,
            self.channel_index_to_label,
            dtype=self.ground_truth.dtype,
        )

        valid_mask = self.ground_truth != _MASK_LABEL
        assert torch.any(valid_mask), "No valid annotated voxels found."

        correct = (pred_labels == self.ground_truth) & valid_mask
        accuracy = correct.sum().float() / valid_mask.sum().float()
        return accuracy


class InferenceDestVolBuffer:
    def __init__(
        self,
        volume_shape,
        label_values: Sequence[object],
        minivol_size: int = 200,
    ) -> None:
        torch_mod = _require_torch()
        if len(volume_shape) != 3:
            raise ValueError(f"volume_shape must be length 3, got {volume_shape}")

        self.minivol_size = _coerce_positive_int(minivol_size, name="minivol_size")
        self.volume_shape = tuple(int(v) for v in volume_shape)
        self.label_values = _coerce_label_values(label_values)
        self.num_classes = int(len(self.label_values))
        self.label_to_channel_index: Dict[int, int] = {
            int(label): int(i) for i, label in enumerate(self.label_values)
        }
        self.channel_index_to_label: Tuple[int, ...] = tuple(self.label_values)

        self.buffer_vol = torch_mod.zeros(
            [self.num_classes, self.volume_shape[0], self.volume_shape[1], self.volume_shape[2]],
            dtype=torch_mod.float32,
        )
        self.hann_window = _build_hann_window(minivol_size=self.minivol_size)

    def add_batch(self, batch, batch_coordinates):
        _add_weighted_batch_to_buffer(
            batch=batch,
            batch_coordinates=batch_coordinates,
            buffer_vol=self.buffer_vol,
            hann_window=self.hann_window,
            minivol_size=self.minivol_size,
            num_classes=self.num_classes,
        )

    def get_pred_labels(self):
        torch_mod = _require_torch()
        return _decode_buffer_labels(
            self.buffer_vol,
            self.channel_index_to_label,
            dtype=torch_mod.long,
        )
