from __future__ import annotations

import time
from queue import Empty, Queue
from typing import Any


class SensorQueue:
    def __init__(self) -> None:
        self._queue: Queue = Queue()

    def push(self, data: Any) -> None:
        self._queue.put(data)

    def get_for_frame(self, frame: int, timeout: float) -> Any:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise Empty()
            data = self._queue.get(timeout=remaining)
            if getattr(data, "frame", None) == frame:
                return data
