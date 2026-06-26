from __future__ import annotations

import argparse
from pathlib import Path

from src.app import run
from src.config import AppConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open 3D volume viewer")
    parser.add_argument("path", nargs="?", help="Path to volume file or dataset")
    parser.add_argument("--semantic", type=str, default=None, help="Path to semantic segmentation map")
    parser.add_argument("--instance", type=str, default=None, help="Path to instance segmentation map")
    parser.add_argument("--bbox", type=str, default=None, help="Path to bounding-box file")
    parser.add_argument(
        "--load-mode",
        type=str,
        choices=("ram", "lazy"),
        default=None,
        help="Volume loading mode: 'ram' for full in-memory load, 'lazy' for on-demand loading",
    )
    parser.add_argument("--max-cache-mb", type=int, default=None, help="Chunk cache size in MB")
    parser.add_argument("--log-level", type=str, default=None, help="Logging level (e.g., INFO, DEBUG)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = AppConfig()
    if args.max_cache_mb is not None:
        config = config.with_cache_megabytes(args.max_cache_mb)
    if args.log_level is not None:
        config = config.with_log_level(args.log_level)
    if args.load_mode is not None:
        config = config.with_load_mode(args.load_mode)

    volume_path = str(Path(args.path).expanduser()) if args.path else None
    semantic_path = str(Path(args.semantic).expanduser()) if args.semantic else None
    instance_path = str(Path(args.instance).expanduser()) if args.instance else None
    bbox_path = str(Path(args.bbox).expanduser()) if args.bbox else None
    run(
        config=config,
        volume_path=volume_path,
        semantic_path=semantic_path,
        instance_path=instance_path,
        bbox_path=bbox_path,
    )


if __name__ == "__main__":
    main()
