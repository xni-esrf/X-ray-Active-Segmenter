from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data.volume import VolumeData


def save_segmentation_volume(
    volume: VolumeData,
    path: str,
    *,
    save_format: str,
    dataset_name: str = "segmentation",
    overwrite: bool = False,
) -> str:
    normalized_path = str(Path(path).expanduser())
    if Path(normalized_path).exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing path without explicit overwrite=True: {normalized_path}"
        )
    array = np.asarray(volume.get_chunk((slice(None), slice(None), slice(None))))
    target_dtype = np.dtype(volume.info.dtype)
    if array.dtype != target_dtype:
        array = array.astype(target_dtype, copy=False)

    normalized_format = save_format.strip().lower()
    if normalized_format == "tiff":
        _save_tiff(array, normalized_path)
    elif normalized_format == "npy":
        np.save(normalized_path, array)
    elif normalized_format == "npz":
        np.savez_compressed(normalized_path, **{dataset_name: array})
    elif normalized_format == "hdf5":
        _save_hdf5(array, normalized_path, dataset_name=dataset_name)
    elif normalized_format == "zarr":
        _save_zarr(array, normalized_path, dataset_name=dataset_name)
    else:
        raise ValueError(f"Unsupported save format: {save_format}")
    return normalized_path


def _save_tiff(array: np.ndarray, path: str) -> None:
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError("tifffile is required to save TIFF volumes") from exc
    tifffile.imwrite(path, array)


def _save_hdf5(array: np.ndarray, path: str, *, dataset_name: str) -> None:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required to save HDF5 volumes") from exc
    with h5py.File(path, "w") as handle:
        handle.create_dataset(dataset_name, data=array, dtype=array.dtype)


def _save_zarr(array: np.ndarray, path: str, *, dataset_name: str) -> None:
    try:
        import zarr
    except ImportError as exc:
        raise ImportError("zarr is required to save Zarr volumes") from exc
    del dataset_name  # dataset name is not used for root-array zarr saves
    arr = zarr.open(path, mode="w", shape=array.shape, dtype=array.dtype)
    arr[...] = array
