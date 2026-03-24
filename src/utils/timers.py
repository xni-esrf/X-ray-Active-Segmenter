from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class TimerResult:
    seconds: float


@contextmanager
def timer() -> Iterator[TimerResult]:
    start = time.perf_counter()
    result = TimerResult(seconds=0.0)
    try:
        yield result
    finally:
        result.seconds = time.perf_counter() - start


_PROFILE_ALWAYS_ON = str(os.getenv("SEGMENTATION_PROFILE", "")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def profiling_enabled(logger: Optional[logging.Logger] = None) -> bool:
    if _PROFILE_ALWAYS_ON:
        return True
    if logger is None:
        return False
    return logger.isEnabledFor(logging.DEBUG)


@contextmanager
def maybe_profile(
    name: str,
    *,
    logger: Optional[logging.Logger] = None,
    details: Optional[str] = None,
    min_ms: float = 0.0,
) -> Iterator[None]:
    active_logger = logger or logging.getLogger(__name__)
    if not profiling_enabled(active_logger):
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if elapsed_ms < float(min_ms):
            return
        log_level = logging.INFO if _PROFILE_ALWAYS_ON else logging.DEBUG
        if not active_logger.isEnabledFor(log_level):
            return
        if details:
            active_logger.log(log_level, "PROFILE %s: %.3f ms | %s", name, elapsed_ms, details)
            return
        active_logger.log(log_level, "PROFILE %s: %.3f ms", name, elapsed_ms)
