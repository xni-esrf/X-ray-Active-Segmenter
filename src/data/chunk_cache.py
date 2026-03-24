from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Hashable, Optional

import numpy as np


@dataclass
class CacheStats:
    entries: int
    bytes: int
    hits: int
    misses: int


class ChunkCache:
    def __init__(self, max_bytes: int = 512 * 1024 * 1024) -> None:
        self.max_bytes = max_bytes
        self._entries: "OrderedDict[Hashable, np.ndarray]" = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0

    def get(self, key: Hashable) -> Optional[np.ndarray]:
        if key in self._entries:
            self._hits += 1
            value = self._entries.pop(key)
            self._entries[key] = value
            return value
        self._misses += 1
        return None

    def set(self, key: Hashable, value: np.ndarray) -> None:
        if key in self._entries:
            self._bytes -= self._entries[key].nbytes
            self._entries.pop(key)
        self._entries[key] = value
        self._bytes += value.nbytes
        self._evict_if_needed()

    def clear(self) -> None:
        self._entries.clear()
        self._bytes = 0

    def stats(self) -> CacheStats:
        return CacheStats(
            entries=len(self._entries),
            bytes=self._bytes,
            hits=self._hits,
            misses=self._misses,
        )

    def _evict_if_needed(self) -> None:
        while self._bytes > self.max_bytes and self._entries:
            _, value = self._entries.popitem(last=False)
            self._bytes -= value.nbytes
