from __future__ import annotations

import json
import logging
from numbers import Integral
from pathlib import Path
import shutil
from typing import Callable, Optional, Sequence

import numpy as np

from .large_crop_inference_plan import LargeCropWindow, ShapeZYX, SliceZYX


LOGGER = logging.getLogger(__name__)


class LargeCropZarrOutputWriter:
    def __init__(
        self,
        path: str,
        *,
        shape: Sequence[object],
        dtype: object,
        chunks: Optional[Sequence[object]] = None,
        overwrite: bool = False,
        array_factory: Optional[Callable[..., object]] = None,
    ) -> None:
        self.path = str(Path(path).expanduser())
        self.shape = _coerce_shape(shape, name="shape")
        self.dtype = np.dtype(dtype)
        self.chunks = None if chunks is None else _coerce_shape(chunks, name="chunks")
        target = Path(self.path)
        if target.exists() and not bool(overwrite):
            raise FileExistsError(
                f"Refusing to overwrite existing Zarr output without overwrite=True: {self.path}"
            )
        if target.exists() and bool(overwrite):
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        factory = _open_zarr_array if array_factory is None else array_factory
        self._array = factory(
            self.path,
            shape=self.shape,
            dtype=self.dtype,
            chunks=self.chunks,
        )
        LOGGER.info(
            "Created large-crop Zarr output: path=%s shape=%s dtype=%s chunks=%s",
            self.path,
            self.shape,
            str(self.dtype),
            self.chunks,
        )

    @property
    def array(self):
        return self._array

    def write_slices(
        self,
        data: np.ndarray,
        *,
        destination_slices: SliceZYX,
    ) -> None:
        normalized_destination = _coerce_slices(
            destination_slices,
            shape=self.shape,
            name="destination_slices",
        )
        expected_shape = _shape_from_slices(normalized_destination)
        array = np.asarray(data)
        if tuple(int(axis) for axis in array.shape) != expected_shape:
            raise ValueError(
                "data shape does not match destination slices: "
                f"data.shape={tuple(array.shape)} expected={expected_shape}"
            )
        if array.dtype != self.dtype:
            array = array.astype(self.dtype, copy=False)
        self._array[normalized_destination] = array

    def write_window_prediction(
        self,
        prediction_crop: np.ndarray,
        *,
        window: LargeCropWindow,
        crop_number: Optional[int] = None,
        total_crops: Optional[int] = None,
    ) -> bool:
        if not window.writes_requested_output:
            LOGGER.info(
                "Skipping large-crop stitching for crop %s: no requested-output intersection",
                _format_crop_progress(crop_number, total_crops),
            )
            return False

        source_slices = window.requested_output_slices_in_crop
        destination_slices = window.requested_output_slices
        crop_array = np.asarray(prediction_crop)
        if crop_array.ndim != 3:
            raise ValueError(
                f"prediction_crop must be 3D (z, y, x), got ndim={crop_array.ndim}"
            )
        if tuple(int(axis) for axis in crop_array.shape) != tuple(window.crop_shape):
            raise ValueError(
                "prediction_crop shape does not match planned crop shape: "
                f"prediction_crop.shape={tuple(crop_array.shape)} expected={window.crop_shape}"
            )

        LOGGER.info(
            "Writing/stitching large crop %s to Zarr: dest=%s source=%s",
            _format_crop_progress(crop_number, total_crops),
            _format_slices(destination_slices),
            _format_slices(source_slices),
        )
        self.write_slices(
            crop_array[source_slices],
            destination_slices=destination_slices,
        )
        LOGGER.info(
            "Finished writing/stitching large crop %s to Zarr",
            _format_crop_progress(crop_number, total_crops),
        )
        return True


def create_large_crop_zarr_output_writer(
    path: str,
    *,
    shape: Sequence[object],
    dtype: object,
    chunks: Optional[Sequence[object]] = None,
    overwrite: bool = False,
    array_factory: Optional[Callable[..., object]] = None,
) -> LargeCropZarrOutputWriter:
    return LargeCropZarrOutputWriter(
        path,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        overwrite=overwrite,
        array_factory=array_factory,
    )


def _open_zarr_array(
    path: str,
    *,
    shape: ShapeZYX,
    dtype: np.dtype,
    chunks: Optional[ShapeZYX],
):
    resolved_chunks = chunks
    if resolved_chunks is None:
        resolved_chunks = tuple(min(int(axis), 256) for axis in shape)  # type: ignore[assignment]
    _validate_chunks(resolved_chunks, shape=shape)
    return _RawZarrV2Array(
        path,
        shape=shape,
        dtype=dtype,
        chunks=resolved_chunks,
    )


class _RawZarrV2Array:
    def __init__(
        self,
        path: str,
        *,
        shape: ShapeZYX,
        dtype: np.dtype,
        chunks: ShapeZYX,
    ) -> None:
        self.path = str(Path(path).expanduser())
        self.shape = tuple(int(axis) for axis in shape)
        self.dtype = np.dtype(dtype)
        self.chunks = tuple(int(axis) for axis in chunks)
        self._path = Path(self.path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._write_metadata()

    def __getitem__(self, key):
        slices = _coerce_slices(
            _array_key_to_tuple(key),
            shape=self.shape,
            name="zarr_read_slices",
        )
        out = np.zeros(_shape_from_slices(slices), dtype=self.dtype)
        for chunk_index in self._chunk_indices_for_slices(slices):
            chunk_slices = self._chunk_slices(chunk_index)
            overlap = _intersect_slices(slices, chunk_slices)
            if any(int(axis_slice.stop) <= int(axis_slice.start) for axis_slice in overlap):
                continue
            chunk = self._read_chunk(chunk_index)
            out_target = _relative_slices(overlap, origin=slices)
            chunk_source = _relative_slices(overlap, origin=chunk_slices)
            out[out_target] = chunk[chunk_source]
        return out

    def __setitem__(self, key, value) -> None:
        slices = _coerce_slices(
            _array_key_to_tuple(key),
            shape=self.shape,
            name="zarr_write_slices",
        )
        array = np.asarray(value, dtype=self.dtype)
        expected_shape = _shape_from_slices(slices)
        if tuple(int(axis) for axis in array.shape) != expected_shape:
            raise ValueError(
                "value shape does not match zarr write slices: "
                f"value.shape={tuple(array.shape)} expected={expected_shape}"
            )
        for chunk_index in self._chunk_indices_for_slices(slices):
            chunk_slices = self._chunk_slices(chunk_index)
            overlap = _intersect_slices(slices, chunk_slices)
            if any(int(axis_slice.stop) <= int(axis_slice.start) for axis_slice in overlap):
                continue
            chunk = self._read_chunk(chunk_index)
            chunk_target = _relative_slices(overlap, origin=chunk_slices)
            value_source = _relative_slices(overlap, origin=slices)
            chunk[chunk_target] = array[value_source]
            self._write_chunk(chunk_index, chunk)

    def _write_metadata(self) -> None:
        metadata = {
            "zarr_format": 2,
            "shape": list(self.shape),
            "chunks": list(self.chunks),
            "dtype": self.dtype.str,
            "compressor": None,
            "fill_value": 0,
            "order": "C",
            "filters": None,
        }
        (self._path / ".zarray").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (self._path / ".zattrs").write_text("{}\n", encoding="utf-8")

    def _chunk_indices_for_slices(self, slices: SliceZYX):
        ranges = []
        for axis in range(3):
            if int(slices[axis].stop) <= int(slices[axis].start):
                ranges.append(range(0))
                continue
            start = int(slices[axis].start) // int(self.chunks[axis])
            stop = (int(slices[axis].stop) - 1) // int(self.chunks[axis])
            ranges.append(range(start, stop + 1))
        for z_index in ranges[0]:
            for y_index in ranges[1]:
                for x_index in ranges[2]:
                    yield (int(z_index), int(y_index), int(x_index))

    def _chunk_slices(self, chunk_index: ShapeZYX) -> SliceZYX:
        slices = []
        for axis in range(3):
            start = int(chunk_index[axis]) * int(self.chunks[axis])
            stop = min(start + int(self.chunks[axis]), int(self.shape[axis]))
            slices.append(slice(start, stop))
        return (slices[0], slices[1], slices[2])

    def _chunk_shape(self, chunk_index: ShapeZYX) -> ShapeZYX:
        return _shape_from_slices(self._chunk_slices(chunk_index))

    def _chunk_path(self, chunk_index: ShapeZYX) -> Path:
        return self._path / ".".join(str(int(axis)) for axis in chunk_index)

    def _read_chunk(self, chunk_index: ShapeZYX) -> np.ndarray:
        shape = self._chunk_shape(chunk_index)
        path = self._chunk_path(chunk_index)
        if not path.exists():
            return np.zeros(shape, dtype=self.dtype)
        data = np.fromfile(path, dtype=self.dtype)
        expected_size = int(np.prod(shape))
        if int(data.size) != expected_size:
            raise RuntimeError(
                f"Unexpected Zarr chunk size for {path}: got={data.size} expected={expected_size}"
            )
        return data.reshape(shape)

    def _write_chunk(self, chunk_index: ShapeZYX, chunk: np.ndarray) -> None:
        path = self._chunk_path(chunk_index)
        np.asarray(chunk, dtype=self.dtype).tofile(path)


def _coerce_shape(values: Sequence[object], *, name: str) -> ShapeZYX:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 values (z, y, x)")
    return (
        _coerce_positive_int(values[0], name=f"{name}[0]"),
        _coerce_positive_int(values[1], name=f"{name}[1]"),
        _coerce_positive_int(values[2], name=f"{name}[2]"),
    )


def _coerce_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    normalized = int(value)
    if normalized < 1:
        raise ValueError(f"{name} must be >= 1")
    return normalized


def _validate_chunks(chunks: ShapeZYX, *, shape: ShapeZYX) -> None:
    for axis, chunk_size in enumerate(chunks):
        if int(chunk_size) < 1:
            raise ValueError(f"chunks[{axis}] must be >= 1")
        if int(chunk_size) > int(shape[axis]):
            raise ValueError(
                f"chunks[{axis}] must be <= shape[{axis}], got {chunk_size} > {shape[axis]}"
            )


def _array_key_to_tuple(key) -> SliceZYX:
    if not isinstance(key, tuple):
        key = (key,)
    return tuple(key)  # type: ignore[return-value]


def _coerce_slices(
    slices: SliceZYX,
    *,
    shape: ShapeZYX,
    name: str,
) -> SliceZYX:
    if len(slices) != 3:
        raise ValueError(f"{name} must contain exactly 3 slices (z, y, x)")
    normalized = []
    for axis, axis_slice in enumerate(slices):
        if not isinstance(axis_slice, slice):
            raise TypeError(
                f"{name}[{axis}] must be a slice, got {type(axis_slice).__name__}"
            )
        start, stop, step = axis_slice.indices(int(shape[axis]))
        if int(step) != 1:
            raise ValueError(f"{name}[{axis}] must have step 1")
        if int(stop) < int(start):
            raise ValueError(f"{name}[{axis}] stop must be >= start")
        normalized.append(slice(int(start), int(stop)))
    return (normalized[0], normalized[1], normalized[2])


def _intersect_slices(first: SliceZYX, second: SliceZYX) -> SliceZYX:
    return (
        slice(
            max(int(first[0].start), int(second[0].start)),
            min(int(first[0].stop), int(second[0].stop)),
        ),
        slice(
            max(int(first[1].start), int(second[1].start)),
            min(int(first[1].stop), int(second[1].stop)),
        ),
        slice(
            max(int(first[2].start), int(second[2].start)),
            min(int(first[2].stop), int(second[2].stop)),
        ),
    )


def _relative_slices(slices: SliceZYX, *, origin: SliceZYX) -> SliceZYX:
    return (
        slice(
            int(slices[0].start) - int(origin[0].start),
            int(slices[0].stop) - int(origin[0].start),
        ),
        slice(
            int(slices[1].start) - int(origin[1].start),
            int(slices[1].stop) - int(origin[1].start),
        ),
        slice(
            int(slices[2].start) - int(origin[2].start),
            int(slices[2].stop) - int(origin[2].start),
        ),
    )


def _shape_from_slices(slices: SliceZYX) -> ShapeZYX:
    return tuple(
        int(axis_slice.stop) - int(axis_slice.start)
        for axis_slice in slices
    )  # type: ignore[return-value]


def _format_crop_progress(
    crop_number: Optional[int],
    total_crops: Optional[int],
) -> str:
    if crop_number is None:
        return "?"
    if total_crops is None:
        return str(int(crop_number))
    return f"{int(crop_number)}/{int(total_crops)}"


def _format_slices(slices: SliceZYX) -> str:
    return (
        "("
        + ", ".join(
            f"{int(axis_slice.start)}:{int(axis_slice.stop)}"
            for axis_slice in slices
        )
        + ")"
    )
