from __future__ import annotations

import math
from numbers import Integral
from numbers import Real
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - import availability is environment dependent
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - import availability is environment dependent
    torch = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[override]
        pass

from .label_utils import MASK_LABEL, coerce_label_values


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


def _compute_bbox_volume_voxels(volume_shape: Sequence[object]) -> int:
    if len(volume_shape) != 3:
        raise ValueError(f"volume_shape must be length 3, got {volume_shape}")
    volume = 1
    for axis, raw_dim in enumerate(tuple(volume_shape)):
        dim = _coerce_positive_int(raw_dim, name=f"volume_shape[{axis}]")
        volume *= int(dim)
    if volume <= 0:
        raise ValueError("bbox volume must be > 0 voxels")
    return int(volume)


def _coerce_scalar_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a scalar real value, got {type(value).__name__}")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite, got {normalized!r}")
    return normalized


def _compute_filtered_mean_dice_score(
    *,
    pred_labels,
    ground_truth,
    label_values: Sequence[int],
    rare_class_ratio: float = 0.01,
    mask_label: int = MASK_LABEL,
) -> float:
    torch_mod = _require_torch()
    if pred_labels.shape != ground_truth.shape:
        raise ValueError(
            "pred_labels and ground_truth must share shape, "
            f"got {tuple(pred_labels.shape)} vs {tuple(ground_truth.shape)}"
        )
    ratio = _coerce_scalar_real(rare_class_ratio, name="rare_class_ratio")
    if ratio <= 0.0:
        raise ValueError(f"rare_class_ratio must be > 0, got {ratio}")

    valid_mask = ground_truth != int(mask_label)
    if not bool(torch_mod.any(valid_mask).item()):
        raise ValueError("No valid annotated voxels found (all labels are masked).")

    pred_valid = pred_labels[valid_mask]
    ground_truth_valid = ground_truth[valid_mask]

    per_class_ground_truth_count: Dict[int, int] = {}
    max_count = 0
    for label in tuple(int(v) for v in tuple(label_values)):
        count = int((ground_truth_valid == int(label)).sum().item())
        per_class_ground_truth_count[int(label)] = int(count)
        if count > max_count:
            max_count = int(count)

    if max_count <= 0:
        raise ValueError("No non-masked class voxels were found in ground truth.")

    threshold = float(max_count) * float(ratio)
    kept_labels = tuple(
        label
        for label in tuple(int(v) for v in tuple(label_values))
        if float(per_class_ground_truth_count[int(label)]) >= float(threshold)
    )
    if not kept_labels:
        raise ValueError("No classes remained after rare-class filtering.")

    per_class_dice_values = []
    for label in kept_labels:
        pred_is_label = pred_valid == int(label)
        gt_is_label = ground_truth_valid == int(label)
        true_positive = int(torch_mod.logical_and(pred_is_label, gt_is_label).sum().item())
        false_positive = int(
            torch_mod.logical_and(pred_is_label, torch_mod.logical_not(gt_is_label)).sum().item()
        )
        false_negative = int(
            torch_mod.logical_and(torch_mod.logical_not(pred_is_label), gt_is_label).sum().item()
        )

        denominator = float((2 * true_positive) + false_positive + false_negative)
        if denominator <= 0.0:
            raise ValueError(
                "Encountered invalid Dice denominator <= 0 for a kept class; "
                "check label filtering invariants."
            )
        dice_value = float((2.0 * float(true_positive)) / denominator)
        if not math.isfinite(dice_value):
            raise ValueError(f"Computed non-finite Dice value: {dice_value!r}")
        per_class_dice_values.append(float(dice_value))

    bbox_dice = float(sum(per_class_dice_values) / float(len(per_class_dice_values)))
    if not math.isfinite(bbox_dice):
        raise ValueError(f"Computed non-finite bbox Dice score: {bbox_dice!r}")
    return float(bbox_dice)


def compute_volume_weighted_mean_score(
    *,
    score_by_box_id: Mapping[str, object],
    bbox_volume_by_box_id: Mapping[str, object],
) -> float:
    if not score_by_box_id:
        raise ValueError("score_by_box_id must not be empty")

    weighted_sum = 0.0
    total_weight = 0
    for box_id, raw_score in tuple(score_by_box_id.items()):
        normalized_box_id = str(box_id)
        if normalized_box_id not in bbox_volume_by_box_id:
            raise ValueError(
                f"Missing bbox volume for box_id={normalized_box_id!r} in bbox_volume_by_box_id."
            )
        score_value = _coerce_scalar_real(raw_score, name=f"score_by_box_id[{normalized_box_id!r}]")
        weight = _coerce_positive_int(
            bbox_volume_by_box_id[normalized_box_id],
            name=f"bbox_volume_by_box_id[{normalized_box_id!r}]",
        )
        weighted_sum += float(score_value) * float(weight)
        total_weight += int(weight)

    if total_weight <= 0:
        raise ValueError("Total bbox volume weight must be > 0.")
    weighted_mean = float(weighted_sum / float(total_weight))
    if not math.isfinite(weighted_mean):
        raise ValueError(f"Weighted mean score must be finite, got {weighted_mean!r}")
    return float(weighted_mean)


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
        self.label_values = coerce_label_values(label_values, allow_duplicates=True)
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
        self.bbox_voxel_volume = _compute_bbox_volume_voxels(self.volume_shape)

    def add_batch(self, batch, batch_coordinates):
        _add_weighted_batch_to_buffer(
            batch=batch,
            batch_coordinates=batch_coordinates,
            buffer_vol=self.buffer_vol,
            hann_window=self.hann_window,
            minivol_size=self.minivol_size,
            num_classes=self.num_classes,
        )

    def get_dice_pred(self):
        torch_mod = _require_torch()
        pred_labels = _decode_buffer_labels(
            self.buffer_vol,
            self.channel_index_to_label,
            dtype=self.ground_truth.dtype,
        )
        dice_value = _compute_filtered_mean_dice_score(
            pred_labels=pred_labels,
            ground_truth=self.ground_truth,
            label_values=self.channel_index_to_label,
            rare_class_ratio=0.01,
            mask_label=MASK_LABEL,
        )
        return torch_mod.tensor(float(dice_value), dtype=torch_mod.float32)


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
        self.label_values = coerce_label_values(label_values, allow_duplicates=True)
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
