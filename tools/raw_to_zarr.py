#!/usr/bin/env python3
"""Convert a headerless raw / ESRF PyHST .vol volume into a chunked Zarr array.

The output is laid out exactly the way the X-ray Active Segmenter viewer expects
(see src/io/zarr_loader.py):

  * a root Zarr array (``zarr.open(path, mode="r")`` finds ``.shape``)
  * axis order Z, Y, X  (shape == (NUM_Z, NUM_Y, NUM_X))
  * chunked, so the viewer's ``--load-mode lazy`` path can stream sub-volumes

Geometry is resolved in this priority order:

  1. explicit ``--shape Z Y X``
  2. an ESRF ``.info`` / ``.vol.info`` sidecar (NUM_X / NUM_Y / NUM_Z, BYTEORDER)
  3. the ``*<X>x<Y>x<Z>.raw`` filename convention

The voxel dtype is taken from ``--dtype`` if given, otherwise guessed from
``file_size / voxel_count`` (1->uint8, 2->uint16, 4->float32, 8->float64).

The conversion is memory-safe: the source is memory-mapped and streamed to Zarr
one slab of Z-chunks at a time, so arbitrarily large volumes convert in a small,
fixed amount of RAM.

Examples
--------
    # ESRF .vol with a sidecar (dimensions + byteorder auto-detected)
    python3 tools/raw_to_zarr.py sample.vol sample.zarr

    # plain raw, dimensions from the *XxYxZ.raw filename, dtype guessed
    python3 tools/raw_to_zarr.py recon_2048x2048x1500.raw recon.zarr

    # fully explicit
    python3 tools/raw_to_zarr.py vol.raw vol.zarr --shape 1500 2048 2048 \
        --dtype float32 --endian little --chunks 64 64 64
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# itemsize (bytes) -> default dtype when guessing
_ITEMSIZE_TO_DTYPE = {1: "uint8", 2: "uint16", 4: "float32", 8: "float64"}


def _find_info_sidecar(raw_path: Path) -> Optional[Path]:
    """Return an ESRF .info sidecar next to *raw_path*, if one exists."""
    candidates = [
        raw_path.with_suffix(raw_path.suffix + ".info"),  # foo.vol -> foo.vol.info
        raw_path.with_suffix(".info"),                    # foo.vol -> foo.info
        Path(str(raw_path) + ".info"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _parse_info(info_path: Path) -> Tuple[Optional[Tuple[int, int, int]], Optional[str]]:
    """Parse an ESRF PyHST .info file -> ((Z, Y, X), endian) with None when absent."""
    nums: dict[str, int] = {}
    endian: Optional[str] = None
    for line in info_path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("!") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().upper()
        value = value.strip()
        if key in ("NUM_X", "NUM_Y", "NUM_Z"):
            try:
                nums[key] = int(float(value))
            except ValueError:
                pass
        elif key == "BYTEORDER":
            endian = "little" if value.upper().startswith("LOW") else "big"
    shape: Optional[Tuple[int, int, int]] = None
    if {"NUM_X", "NUM_Y", "NUM_Z"} <= nums.keys():
        # File order is X-fastest -> array shape is (Z, Y, X).
        shape = (nums["NUM_Z"], nums["NUM_Y"], nums["NUM_X"])
    return shape, endian


def _parse_filename_dims(raw_path: Path) -> Optional[Tuple[int, int, int]]:
    """Parse a ``*<X>x<Y>x<Z>`` token from the filename -> (Z, Y, X)."""
    match = re.search(r"(\d+)x(\d+)x(\d+)", raw_path.name, flags=re.IGNORECASE)
    if not match:
        return None
    x, y, z = (int(match.group(i)) for i in (1, 2, 3))
    return (z, y, x)


def _guess_dtype(data_bytes: int, voxel_count: int) -> str:
    if voxel_count <= 0:
        raise ValueError("voxel count must be positive to guess dtype")
    if data_bytes % voxel_count != 0:
        raise ValueError(
            f"cannot guess dtype: data size {data_bytes} is not divisible by "
            f"voxel count {voxel_count}. Pass --dtype (and check --offset/--shape)."
        )
    itemsize = data_bytes // voxel_count
    if itemsize not in _ITEMSIZE_TO_DTYPE:
        raise ValueError(
            f"cannot guess dtype: implied itemsize {itemsize} bytes is unsupported. "
            "Pass --dtype explicitly."
        )
    return _ITEMSIZE_TO_DTYPE[itemsize]


def _resolve_geometry(args: argparse.Namespace, raw_path: Path) -> Tuple[Tuple[int, int, int], str, str]:
    """Resolve (shape_zyx, dtype, source-description) from CLI / sidecar / filename."""
    shape = tuple(args.shape) if args.shape else None
    endian = args.endian
    source = "explicit --shape"

    if shape is None:
        sidecar = _find_info_sidecar(raw_path)
        if sidecar is not None:
            info_shape, info_endian = _parse_info(sidecar)
            if info_shape is not None:
                shape = info_shape
                source = f"sidecar {sidecar.name}"
            if endian is None and info_endian is not None:
                endian = info_endian

    if shape is None:
        fname_shape = _parse_filename_dims(raw_path)
        if fname_shape is not None:
            shape = fname_shape
            source = "filename XxYxZ"

    if shape is None:
        raise SystemExit(
            "Could not determine volume dimensions. Provide --shape Z Y X, "
            "or a .info sidecar, or a *XxYxZ.raw filename."
        )

    if any(d <= 0 for d in shape):
        raise SystemExit(f"invalid shape {shape}: all dimensions must be positive")

    endian = endian or "little"  # ESRF/most raw volumes are little-endian
    voxel_count = int(np.prod(shape, dtype=np.int64))
    data_bytes = raw_path.stat().st_size - args.offset

    dtype = args.dtype or _guess_dtype(data_bytes, voxel_count)
    return shape, dtype, f"{source}; dtype={'explicit' if args.dtype else 'guessed'}"


def _open_zarr(out_path: str, shape, chunks, dtype):
    import zarr

    # zarr.open(mode="w", shape/chunks/dtype) creates a root array on v2 and v3.
    try:
        return zarr.open(out_path, mode="w", shape=shape, chunks=chunks, dtype=dtype)
    except TypeError:
        # Fallback for builds where create params must go through create_array.
        return zarr.create_array(
            store=out_path, shape=shape, chunks=chunks, dtype=dtype, overwrite=True
        )


def convert(args: argparse.Namespace) -> int:
    raw_path = Path(args.input).expanduser()
    if not raw_path.is_file():
        raise SystemExit(f"input not found: {raw_path}")

    shape, dtype, how = _resolve_geometry(args, raw_path)
    z, y, x = shape
    src_dtype = np.dtype(dtype).newbyteorder("<" if args.endian_resolved == "little" else ">")
    voxel_count = int(np.prod(shape, dtype=np.int64))
    expected_bytes = voxel_count * src_dtype.itemsize + args.offset
    actual_bytes = raw_path.stat().st_size

    print(f"[raw2zarr] input        : {raw_path}")
    print(f"[raw2zarr] geometry     : shape(z,y,x)={shape}  ({how})")
    print(f"[raw2zarr] dtype/endian : {src_dtype.name}  {args.endian_resolved}-endian  offset={args.offset}")
    print(f"[raw2zarr] size check   : expected {expected_bytes} bytes, file has {actual_bytes} bytes", end="")
    if expected_bytes != actual_bytes:
        print("  <-- MISMATCH")
        if not args.force:
            raise SystemExit(
                "size mismatch: re-check --shape/--dtype/--offset, or pass --force to proceed anyway."
            )
        print("[raw2zarr] proceeding due to --force (volume may be truncated/garbled)")
    else:
        print("  OK")

    out_dtype = np.dtype("float16") if args.cast_float16 and src_dtype.kind == "f" else np.dtype(dtype)
    chunks = tuple(min(c, d) for c, d in zip(args.chunks, shape))

    src = np.memmap(raw_path, dtype=src_dtype, mode="r", offset=args.offset, shape=shape)
    dst = _open_zarr(str(Path(args.output).expanduser()), shape, chunks, out_dtype)

    cz = chunks[0]
    n_slabs = (z + cz - 1) // cz
    for i, z0 in enumerate(range(0, z, cz), start=1):
        z1 = min(z0 + cz, z)
        block = np.ascontiguousarray(src[z0:z1])
        if out_dtype != block.dtype:
            block = block.astype(out_dtype, copy=False)
        dst[z0:z1] = block
        print(f"\r[raw2zarr] writing slab {i}/{n_slabs}  (z={z0}:{z1})", end="", flush=True)
    print()

    print(f"[raw2zarr] output       : {args.output}  shape={shape}  chunks={chunks}  dtype={out_dtype.name}")
    print("[raw2zarr] done. Open it with:")
    print(f"             python3 open_ui_raw_viewer.py {args.output} --load-mode lazy")
    return 0


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert raw/.vol volumes to viewer-ready chunked Zarr.")
    p.add_argument("input", help="path to the raw/.vol volume")
    p.add_argument("output", help="output Zarr path (e.g. volume.zarr)")
    p.add_argument("--shape", type=int, nargs=3, metavar=("Z", "Y", "X"),
                   help="explicit dimensions in Z Y X order (overrides sidecar/filename)")
    p.add_argument("--dtype", type=str, default=None,
                   help="voxel dtype (e.g. float32, uint16, uint8); guessed from file size if omitted")
    p.add_argument("--endian", type=str, choices=("little", "big"), default=None,
                   help="byte order (default: from sidecar, else little)")
    p.add_argument("--offset", type=int, default=0, help="header bytes to skip before voxel data")
    p.add_argument("--chunks", type=int, nargs=3, default=(64, 64, 64), metavar=("CZ", "CY", "CX"),
                   help="Zarr chunk shape in Z Y X (default: 64 64 64)")
    p.add_argument("--cast-float16", action="store_true",
                   help="store float volumes as float16 (matches the viewer's raw read cast; halves size)")
    p.add_argument("--force", action="store_true", help="convert even if the size check fails")
    args = p.parse_args(argv)
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    # Endianness may come from the sidecar inside _resolve_geometry; capture the
    # final value on the namespace so convert() and the dtype byteorder agree.
    raw_path = Path(args.input).expanduser()
    if raw_path.is_file() and args.shape is None and args.endian is None:
        sidecar = _find_info_sidecar(raw_path)
        if sidecar is not None:
            _, info_endian = _parse_info(sidecar)
            args.endian = info_endian
    args.endian_resolved = args.endian or "little"
    return convert(args)


if __name__ == "__main__":
    sys.exit(main())
