from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from ..data import ChunkCache, VolumeData


@dataclass
class ChunkRequest:
    zyx_slices: Tuple[slice, slice, slice]


@dataclass
class ChunkResult:
    zyx_slices: Tuple[slice, slice, slice]
    data: np.ndarray


class IOWorker:
    def __init__(self, volume: VolumeData, cache: Optional[ChunkCache] = None, max_workers: int = 2) -> None:
        self.volume = volume
        self.cache = cache
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def request_chunk(
        self,
        request: ChunkRequest,
        on_complete: Optional[Callable[[ChunkResult], None]] = None,
    ) -> Future[ChunkResult]:
        future = self._executor.submit(self._load_chunk, request)
        if on_complete is not None:
            future.add_done_callback(lambda f: on_complete(f.result()))
        return future

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)

    def _load_chunk(self, request: ChunkRequest) -> ChunkResult:
        data = self.volume.get_chunk(request.zyx_slices)
        if self.cache is not None and self.cache is not self.volume.cache:
            self.cache.set(self.volume.cache_key(request.zyx_slices), data)
        return ChunkResult(zyx_slices=request.zyx_slices, data=data)
