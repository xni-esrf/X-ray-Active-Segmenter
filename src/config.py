from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    cache_max_bytes: int = 512 * 1024 * 1024
    log_level: str = "INFO"
    load_mode: str = "ram"

    def with_cache_megabytes(self, size_mb: int) -> "AppConfig":
        return AppConfig(
            cache_max_bytes=int(size_mb) * 1024 * 1024,
            log_level=self.log_level,
            load_mode=self.load_mode,
        )

    def with_log_level(self, level: str) -> "AppConfig":
        return AppConfig(
            cache_max_bytes=self.cache_max_bytes,
            log_level=level.upper(),
            load_mode=self.load_mode,
        )

    def with_load_mode(self, mode: str) -> "AppConfig":
        normalized = mode.strip().lower()
        if normalized not in {"ram", "lazy"}:
            raise ValueError("load_mode must be 'ram' or 'lazy'")
        return AppConfig(
            cache_max_bytes=self.cache_max_bytes,
            log_level=self.log_level,
            load_mode=normalized,
        )
