"""Zarr output writer for streaming inference.

Creates a full-volume Zarr array whose chunk grid matches the plan's output
chunks (``chunk_size == chunk_stride_multiple * stride`` per axis) and whose
``fill_value`` is the background label (``label_values[0]``).  The streaming
executor calls :meth:`StreamingZarrWriter.write_chunk` once per data chunk with a
chunk-aligned region, so every Zarr chunk is written at most once and regions
that are never written (skipped background, or outside the inference bbox) read
back as the background fill value.

The array is created via the real ``zarr`` library; zarr's default codec is a
light Zstd compressor, which is what we want here.
"""

from __future__ import annotations

import logging
from numbers import Integral
from pathlib import Path
import shutil
from typing import Callable, Optional, Sequence, Tuple

import numpy as np


LOGGER = logging.getLogger(__name__)

ShapeZYX = Tuple[int, int, int]
SliceZYX = Tuple[slice, slice, slice]


class StreamingZarrWriter:
    def __init__(
        self,
        path: str,
        *,
        shape: Sequence[object],
        dtype: object,
        chunks: Sequence[object],
        fill_value: int,
        overwrite: bool = False,
        array_factory: Optional[Callable[..., object]] = None,
    ) -> None:
        self.path = str(Path(path).expanduser())
        self.shape = _coerce_shape(shape, name="shape")
        self.dtype = np.dtype(dtype)
        if not np.issubdtype(self.dtype, np.integer):
            raise TypeError(f"streaming output dtype must be integer, got {self.dtype}")
        self.chunks = _coerce_chunks(chunks, shape=self.shape, name="chunks")
        self.fill_value = int(fill_value)

        target = Path(self.path)
        if target.exists():
            if not overwrite:
                raise FileExistsError(
                    "Refusing to overwrite existing Zarr output without overwrite=True: "
                    f"{self.path}"
                )
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        if target.parent:
            target.parent.mkdir(parents=True, exist_ok=True)

        factory = _open_zarr_array if array_factory is None else array_factory
        self._array = factory(
            self.path,
            shape=self.shape,
            dtype=self.dtype,
            chunks=self.chunks,
            fill_value=self.fill_value,
        )
        LOGGER.info(
            "Created streaming Zarr output: path=%s shape=%s dtype=%s chunks=%s fill=%d",
            self.path,
            self.shape,
            str(self.dtype),
            self.chunks,
            self.fill_value,
        )

    @property
    def array(self):
        return self._array

    def write_chunk(self, array: np.ndarray, *, chunk_slices: SliceZYX) -> None:
        normalized = _coerce_slices(chunk_slices, shape=self.shape, name="chunk_slices")
        data = np.asarray(array)
        expected_shape = tuple(
            int(axis_slice.stop) - int(axis_slice.start) for axis_slice in normalized
        )
        if tuple(int(dim) for dim in data.shape) != expected_shape:
            raise ValueError(
                "chunk data shape does not match chunk_slices: "
                f"data.shape={tuple(data.shape)} expected={expected_shape}"
            )
        if data.dtype != self.dtype:
            data = data.astype(self.dtype, copy=False)
        self._array[normalized] = data


def create_streaming_zarr_writer(
    path: str,
    *,
    shape: Sequence[object],
    dtype: object,
    chunks: Sequence[object],
    fill_value: int,
    overwrite: bool = False,
    array_factory: Optional[Callable[..., object]] = None,
) -> StreamingZarrWriter:
    return StreamingZarrWriter(
        path,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        fill_value=fill_value,
        overwrite=overwrite,
        array_factory=array_factory,
    )


def _open_zarr_array(
    path: str,
    *,
    shape: ShapeZYX,
    dtype: np.dtype,
    chunks: ShapeZYX,
    fill_value: int,
):
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("zarr is required to write streaming inference output") from exc
    # zarr's default codec is a light Zstd compressor, which is what we want.
    return zarr.open(
        str(path),
        mode="w",
        shape=tuple(int(dim) for dim in shape),
        chunks=tuple(int(dim) for dim in chunks),
        dtype=np.dtype(dtype),
        fill_value=int(fill_value),
    )


def _coerce_shape(values: Sequence[object], *, name: str) -> ShapeZYX:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 values (z, y, x)")
    return (
        _coerce_positive_int(values[0], name=f"{name}[0]"),
        _coerce_positive_int(values[1], name=f"{name}[1]"),
        _coerce_positive_int(values[2], name=f"{name}[2]"),
    )


def _coerce_chunks(values: Sequence[object], *, shape: ShapeZYX, name: str) -> ShapeZYX:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 values (z, y, x)")
    # Clamp to the volume so a chunk never exceeds the array on an axis (zarr
    # allows only one chunk there anyway; write-once alignment is preserved).
    return (
        min(_coerce_positive_int(values[0], name=f"{name}[0]"), int(shape[0])),
        min(_coerce_positive_int(values[1], name=f"{name}[1]"), int(shape[1])),
        min(_coerce_positive_int(values[2], name=f"{name}[2]"), int(shape[2])),
    )


def _coerce_slices(slices: SliceZYX, *, shape: ShapeZYX, name: str) -> SliceZYX:
    if len(slices) != 3:
        raise ValueError(f"{name} must contain exactly 3 slices (z, y, x)")
    normalized = []
    for axis, axis_slice in enumerate(slices):
        if not isinstance(axis_slice, slice):
            raise TypeError(f"{name}[{axis}] must be a slice, got {type(axis_slice).__name__}")
        start, stop, step = axis_slice.indices(int(shape[axis]))
        if int(step) != 1:
            raise ValueError(f"{name}[{axis}] must have step 1")
        if int(stop) < int(start):
            raise ValueError(f"{name}[{axis}] stop must be >= start")
        normalized.append(slice(int(start), int(stop)))
    return (normalized[0], normalized[1], normalized[2])


def _coerce_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    normalized = int(value)
    if normalized < 1:
        raise ValueError(f"{name} must be >= 1")
    return normalized
