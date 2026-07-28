from __future__ import annotations

import threading
import time
import unittest

from src.learning.streaming_inference import _OutputBufferPool


def _run_with_timeout(fn, timeout=5.0):
    """Run ``fn`` on a worker thread; fail on hang, else return/re-raise."""
    box: dict = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise AssertionError(f"call did not return within {timeout}s (deadlock)")
    if "error" in box:
        raise box["error"]
    return box.get("value")


class OutputBufferPoolTest(unittest.TestCase):
    def test_empty_pool_rejected(self):
        with self.assertRaises(ValueError):
            _OutputBufferPool([])

    def test_acquire_returns_each_buffer_until_exhausted(self):
        bufs = ["A", "B", "C"]
        pool = _OutputBufferPool(bufs)
        self.assertEqual(pool.free_count, 3)
        self.assertEqual(pool.leased_count, 0)
        got = [pool.acquire() for _ in range(3)]
        self.assertEqual(got, ["A", "B", "C"])  # FIFO acquire order
        self.assertEqual(pool.free_count, 0)
        self.assertEqual(pool.leased_count, 3)

    def test_release_is_fifo(self):
        pool = _OutputBufferPool([0, 1, 2])
        self.assertEqual([pool.acquire() for _ in range(3)], [0, 1, 2])
        # release returns the oldest leased (0) to the pool, then (1)
        pool.release()
        pool.release()
        # next acquires hand those back out in the same order they were released
        self.assertEqual(pool.acquire(), 0)
        self.assertEqual(pool.acquire(), 1)

    def test_balanced_counts_after_cycles(self):
        pool = _OutputBufferPool(["x", "y"])
        for _ in range(50):
            pool.acquire()
            pool.release()
        self.assertEqual(pool.free_count, 2)
        self.assertEqual(pool.leased_count, 0)

    def test_acquire_blocks_until_release(self):
        pool = _OutputBufferPool(["only"])
        first = pool.acquire()
        self.assertEqual(first, "only")

        started = threading.Event()
        acquired = threading.Event()

        def waiter():
            started.set()
            pool.acquire()  # must block: no free buffer
            acquired.set()

        t = threading.Thread(target=waiter, daemon=True)
        t.start()
        self.assertTrue(started.wait(1.0))
        # give the waiter a chance to (not) proceed; it must still be blocked
        time.sleep(0.2)
        self.assertFalse(acquired.is_set(), "acquire returned while pool exhausted")
        # releasing the one buffer must unblock the waiter promptly
        pool.release()
        self.assertTrue(acquired.wait(2.0), "release did not unblock a waiting acquire")
        t.join(2.0)
        self.assertFalse(t.is_alive())

    def test_concurrent_producer_consumer_never_reuses_leased(self):
        # A producer acquires and hands the buffer to a consumer via a queue; the
        # consumer marks it in-use, "works", then releases.  The producer asserts
        # each buffer it acquires is NOT currently in use -> proves the pool never
        # hands out a leased buffer.
        from queue import Queue

        pool = _OutputBufferPool([f"buf{i}" for i in range(3)])
        handoff: "Queue" = Queue(maxsize=2)
        in_use: set = set()
        in_use_lock = threading.Lock()
        errors: list = []
        n_items = 300

        def producer():
            try:
                for _ in range(n_items):
                    buf = pool.acquire()
                    with in_use_lock:
                        if buf in in_use:
                            errors.append(f"acquired in-use buffer {buf}")
                    handoff.put(buf)
            except BaseException as exc:  # noqa: BLE001
                errors.append(repr(exc))

        def consumer():
            try:
                for _ in range(n_items):
                    buf = handoff.get()
                    with in_use_lock:
                        in_use.add(buf)
                    time.sleep(0.0005)  # widen the use window
                    with in_use_lock:
                        in_use.discard(buf)
                    pool.release()
            except BaseException as exc:  # noqa: BLE001
                errors.append(repr(exc))

        def run_both():
            p = threading.Thread(target=producer, daemon=True)
            c = threading.Thread(target=consumer, daemon=True)
            p.start()
            c.start()
            p.join()
            c.join()

        _run_with_timeout(run_both, timeout=20.0)
        self.assertEqual(errors, [])
        self.assertEqual(pool.free_count, 3)
        self.assertEqual(pool.leased_count, 0)


if __name__ == "__main__":
    unittest.main()
