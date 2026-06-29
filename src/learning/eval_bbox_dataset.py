from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from numbers import Real
from pathlib import Path
import tempfile
from typing import Dict, Mapping, Optional, Sequence, Tuple

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


@dataclass(frozen=True)
class _SpatialRegion:
    z0: int
    z1: int
    x0: int
    x1: int
    y0: int
    y1: int

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (
            max(0, int(self.z1) - int(self.z0)),
            max(0, int(self.x1) - int(self.x0)),
            max(0, int(self.y1) - int(self.y0)),
        )


@dataclass(frozen=True)
class _WeightedMinivolIntersection:
    target_slices: Tuple[slice, slice, slice]
    minivol_slices: Tuple[slice, slice, slice]


def _coerce_region_start(values: Sequence[object], *, name: str) -> Tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 values")
    return (
        int(values[0]),
        int(values[1]),
        int(values[2]),
    )


def _coerce_region_shape(values: Sequence[object], *, name: str) -> Tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 values")
    return (
        _coerce_positive_int(values[0], name=f"{name}[0]"),
        _coerce_positive_int(values[1], name=f"{name}[1]"),
        _coerce_positive_int(values[2], name=f"{name}[2]"),
    )


def _region_from_start_and_shape(
    *,
    start: Sequence[object],
    shape: Sequence[object],
    name: str,
) -> _SpatialRegion:
    z0, x0, y0 = _coerce_region_start(start, name=f"{name}.start")
    z_size, x_size, y_size = _coerce_region_shape(shape, name=f"{name}.shape")
    return _SpatialRegion(
        z0=z0,
        z1=z0 + z_size,
        x0=x0,
        x1=x0 + x_size,
        y0=y0,
        y1=y0 + y_size,
    )


def _intersect_regions(
    first: _SpatialRegion,
    second: _SpatialRegion,
) -> Optional[_SpatialRegion]:
    z0 = max(int(first.z0), int(second.z0))
    z1 = min(int(first.z1), int(second.z1))
    x0 = max(int(first.x0), int(second.x0))
    x1 = min(int(first.x1), int(second.x1))
    y0 = max(int(first.y0), int(second.y0))
    y1 = min(int(first.y1), int(second.y1))
    if z0 >= z1 or x0 >= x1 or y0 >= y1:
        return None
    return _SpatialRegion(z0=z0, z1=z1, x0=x0, x1=x1, y0=y0, y1=y1)


def _relative_region_slices(
    region: _SpatialRegion,
    *,
    origin: _SpatialRegion,
) -> Tuple[slice, slice, slice]:
    return (
        slice(int(region.z0) - int(origin.z0), int(region.z1) - int(origin.z0)),
        slice(int(region.x0) - int(origin.x0), int(region.x1) - int(origin.x0)),
        slice(int(region.y0) - int(origin.y0), int(region.y1) - int(origin.y0)),
    )


def _weighted_minivol_intersection(
    *,
    minivol_coordinates: Sequence[object],
    minivol_size: int,
    target_origin: Sequence[object],
    target_shape: Sequence[object],
) -> Optional[_WeightedMinivolIntersection]:
    normalized_minivol_size = _coerce_positive_int(minivol_size, name="minivol_size")
    minivol_region = _region_from_start_and_shape(
        start=minivol_coordinates,
        shape=(normalized_minivol_size, normalized_minivol_size, normalized_minivol_size),
        name="minivol",
    )
    target_region = _region_from_start_and_shape(
        start=target_origin,
        shape=target_shape,
        name="target",
    )
    intersection = _intersect_regions(minivol_region, target_region)
    if intersection is None:
        return None
    return _WeightedMinivolIntersection(
        target_slices=_relative_region_slices(intersection, origin=target_region),
        minivol_slices=_relative_region_slices(intersection, origin=minivol_region),
    )


def _add_weighted_minivol_to_buffer_region(
    *,
    minivol,
    minivol_coordinates: Sequence[object],
    buffer_vol,
    hann_window,
    minivol_size: int,
    target_origin: Sequence[object] = (0, 0, 0),
) -> bool:
    placement = _weighted_minivol_intersection(
        minivol_coordinates=minivol_coordinates,
        minivol_size=minivol_size,
        target_origin=target_origin,
        target_shape=tuple(int(v) for v in buffer_vol.shape[1:4]),
    )
    if placement is None:
        return False

    z_target, x_target, y_target = placement.target_slices
    z_minivol, x_minivol, y_minivol = placement.minivol_slices
    weighted_minivol = (
        minivol[:, z_minivol, x_minivol, y_minivol]
        * hann_window[:, z_minivol, x_minivol, y_minivol]
    )
    buffer_vol[:, z_target, x_target, y_target] = (
        weighted_minivol + buffer_vol[:, z_target, x_target, y_target]
    )
    return True


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
        _add_weighted_minivol_to_buffer_region(
            minivol=minivol,
            minivol_coordinates=minivol_coordinates,
            buffer_vol=buffer_vol,
            hann_window=hann_window,
            minivol_size=minivol_size,
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


class TiledInferenceDestVolBuffer:
    def __init__(
        self,
        volume_shape,
        label_values: Sequence[object],
        minivol_size: int = 200,
        tile_shape: Sequence[object] = (256, 256, 256),
        temp_dir: Optional[str] = None,
    ) -> None:
        torch_mod = _require_torch()
        if len(volume_shape) != 3:
            raise ValueError(f"volume_shape must be length 3, got {volume_shape}")

        self.minivol_size = _coerce_positive_int(minivol_size, name="minivol_size")
        self.volume_shape = _coerce_region_shape(volume_shape, name="volume_shape")
        self.tile_shape = _coerce_region_shape(tile_shape, name="tile_shape")
        self.label_values = coerce_label_values(label_values, allow_duplicates=True)
        self.num_classes = int(len(self.label_values))
        self.label_to_channel_index: Dict[int, int] = {
            int(label): int(i) for i, label in enumerate(self.label_values)
        }
        self.channel_index_to_label: Tuple[int, ...] = tuple(self.label_values)

        self.hann_window = _build_hann_window(minivol_size=self.minivol_size)
        self._hann_window_np = self._tensor_to_numpy_float32(self.hann_window)
        parent_dir = None if temp_dir is None else str(Path(temp_dir).expanduser())
        self._temp_dir = tempfile.TemporaryDirectory(
            prefix="xray_inference_tiles_",
            dir=parent_dir,
        )
        self._tile_buffers: Dict[Tuple[int, int, int], np.memmap] = {}
        self._tile_paths: Dict[Tuple[int, int, int], Path] = {}
        self._closed = False
        del torch_mod

    @property
    def temp_dir_path(self) -> Path:
        return Path(self._temp_dir.name)

    def add_batch(self, batch, batch_coordinates) -> None:
        if self._closed:
            raise RuntimeError("TiledInferenceDestVolBuffer is closed")
        if batch.ndim != 5:
            raise ValueError(f"batch must be 5D [B, C, D, H, W], got ndim={batch.ndim}")
        if int(batch.shape[1]) != int(self.num_classes):
            raise ValueError(
                f"batch channel count ({int(batch.shape[1])}) must match num_classes ({self.num_classes})"
            )

        for i in range(int(batch.shape[0])):
            minivol = self._tensor_to_numpy_float32(batch[i, :, :, :, :])
            minivol_coordinates = (
                int(batch_coordinates[0][i]),
                int(batch_coordinates[1][i]),
                int(batch_coordinates[2][i]),
            )
            for tile_id in self._tile_ids_intersecting_minivol(minivol_coordinates):
                self._add_minivol_to_tile(
                    tile_id=tile_id,
                    minivol=minivol,
                    minivol_coordinates=minivol_coordinates,
                )

    def get_pred_labels(self):
        if self._closed:
            raise RuntimeError("TiledInferenceDestVolBuffer is closed")
        torch_mod = _require_torch()
        output = np.empty(self.volume_shape, dtype=np.int64)
        for tile_id in self._all_tile_ids():
            region = self._tile_region(tile_id)
            tile_shape = region.shape
            tile_buffer = self._tile_buffers.get(tile_id)
            if tile_buffer is None:
                pred_channel = np.zeros(tile_shape, dtype=np.int64)
            else:
                pred_channel = np.argmax(np.asarray(tile_buffer), axis=0).astype(
                    np.int64,
                    copy=False,
                )
            labels = np.asarray(self.channel_index_to_label, dtype=np.int64)[pred_channel]
            output[
                region.z0 : region.z1,
                region.x0 : region.x1,
                region.y0 : region.y1,
            ] = labels
        return torch_mod.as_tensor(output, dtype=torch_mod.long)

    def close(self) -> None:
        if self._closed:
            return
        cleanup_errors: list[str] = []
        for tile_id, tile_buffer in tuple(self._tile_buffers.items()):
            try:
                self._close_tile_buffer(tile_buffer)
            except Exception as exc:
                cleanup_errors.append(
                    f"tile {tile_id}: {type(exc).__name__}: {exc}"
                )
        self._tile_buffers.clear()
        self._tile_paths.clear()
        self._closed = True
        try:
            self._temp_dir.cleanup()
        except Exception as exc:
            cleanup_errors.append(f"temp_dir.cleanup(): {type(exc).__name__}: {exc}")
        if cleanup_errors:
            raise RuntimeError("; ".join(cleanup_errors))

    def shutdown(self) -> None:
        self.close()

    def stop(self) -> None:
        self.close()

    def terminate(self) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> "TiledInferenceDestVolBuffer":
        if self._closed:
            raise RuntimeError("TiledInferenceDestVolBuffer is closed")
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def _tile_region(self, tile_id: Tuple[int, int, int]) -> _SpatialRegion:
        z_index, x_index, y_index = tile_id
        z0 = int(z_index) * int(self.tile_shape[0])
        x0 = int(x_index) * int(self.tile_shape[1])
        y0 = int(y_index) * int(self.tile_shape[2])
        return _SpatialRegion(
            z0=z0,
            z1=min(z0 + int(self.tile_shape[0]), int(self.volume_shape[0])),
            x0=x0,
            x1=min(x0 + int(self.tile_shape[1]), int(self.volume_shape[1])),
            y0=y0,
            y1=min(y0 + int(self.tile_shape[2]), int(self.volume_shape[2])),
        )

    def _all_tile_ids(self) -> Tuple[Tuple[int, int, int], ...]:
        z_tiles = (int(self.volume_shape[0]) + int(self.tile_shape[0]) - 1) // int(self.tile_shape[0])
        x_tiles = (int(self.volume_shape[1]) + int(self.tile_shape[1]) - 1) // int(self.tile_shape[1])
        y_tiles = (int(self.volume_shape[2]) + int(self.tile_shape[2]) - 1) // int(self.tile_shape[2])
        return tuple(
            (z_index, x_index, y_index)
            for z_index in range(z_tiles)
            for x_index in range(x_tiles)
            for y_index in range(y_tiles)
        )

    def _tile_ids_intersecting_minivol(
        self,
        minivol_coordinates: Tuple[int, int, int],
    ) -> Tuple[Tuple[int, int, int], ...]:
        minivol_region = _region_from_start_and_shape(
            start=minivol_coordinates,
            shape=(self.minivol_size, self.minivol_size, self.minivol_size),
            name="minivol",
        )
        volume_region = _region_from_start_and_shape(
            start=(0, 0, 0),
            shape=self.volume_shape,
            name="volume",
        )
        clipped = _intersect_regions(minivol_region, volume_region)
        if clipped is None:
            return tuple()
        z_start = int(clipped.z0) // int(self.tile_shape[0])
        z_stop = (int(clipped.z1) - 1) // int(self.tile_shape[0])
        x_start = int(clipped.x0) // int(self.tile_shape[1])
        x_stop = (int(clipped.x1) - 1) // int(self.tile_shape[1])
        y_start = int(clipped.y0) // int(self.tile_shape[2])
        y_stop = (int(clipped.y1) - 1) // int(self.tile_shape[2])
        return tuple(
            (z_index, x_index, y_index)
            for z_index in range(z_start, z_stop + 1)
            for x_index in range(x_start, x_stop + 1)
            for y_index in range(y_start, y_stop + 1)
        )

    def _open_tile_buffer(self, tile_id: Tuple[int, int, int]) -> np.memmap:
        existing = self._tile_buffers.get(tile_id)
        if existing is not None:
            return existing

        region = self._tile_region(tile_id)
        shape = (
            int(self.num_classes),
            int(region.shape[0]),
            int(region.shape[1]),
            int(region.shape[2]),
        )
        path = self.temp_dir_path / f"tile_{tile_id[0]}_{tile_id[1]}_{tile_id[2]}.dat"
        tile_buffer = np.memmap(path, mode="w+", dtype=np.float32, shape=shape)
        tile_buffer[...] = 0.0
        self._tile_buffers[tile_id] = tile_buffer
        self._tile_paths[tile_id] = path
        return tile_buffer

    @staticmethod
    def _close_tile_buffer(tile_buffer: np.memmap) -> None:
        tile_buffer.flush()
        mmap_obj = getattr(tile_buffer, "_mmap", None)
        if mmap_obj is not None:
            mmap_obj.close()

    def _add_minivol_to_tile(
        self,
        *,
        tile_id: Tuple[int, int, int],
        minivol: np.ndarray,
        minivol_coordinates: Tuple[int, int, int],
    ) -> None:
        region = self._tile_region(tile_id)
        placement = _weighted_minivol_intersection(
            minivol_coordinates=minivol_coordinates,
            minivol_size=self.minivol_size,
            target_origin=(region.z0, region.x0, region.y0),
            target_shape=region.shape,
        )
        if placement is None:
            return

        tile_buffer = self._open_tile_buffer(tile_id)
        z_target, x_target, y_target = placement.target_slices
        z_minivol, x_minivol, y_minivol = placement.minivol_slices
        weighted_minivol = (
            minivol[:, z_minivol, x_minivol, y_minivol]
            * self._hann_window_np[:, z_minivol, x_minivol, y_minivol]
        )
        tile_buffer[:, z_target, x_target, y_target] = (
            tile_buffer[:, z_target, x_target, y_target] + weighted_minivol
        )

    @staticmethod
    def _tensor_to_numpy_float32(value) -> np.ndarray:
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        to_method = getattr(value, "to", None)
        if callable(to_method) and torch is not None:
            try:
                value = to_method(dtype=torch.float32, device="cpu")
            except TypeError:
                value = to_method(dtype=torch.float32)
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        numpy_method = getattr(value, "numpy", None)
        if callable(numpy_method):
            array = numpy_method()
        else:
            array = np.asarray(value)
        return np.asarray(array, dtype=np.float32)
