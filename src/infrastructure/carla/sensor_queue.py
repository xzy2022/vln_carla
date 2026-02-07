from __future__ import annotations

import time
from queue import Empty, Queue
from typing import Generic, Protocol, TypeVar


class HasFrame(Protocol):
    frame: int


FrameT = TypeVar("FrameT", bound=HasFrame)


class SensorQueue(Generic[FrameT]):
    def __init__(self) -> None:
        self._queue: Queue[FrameT] = Queue()

    def push(self, data: FrameT) -> None:
        self._queue.put(data)

    def get_for_frame(self, frame: int, timeout: float) -> FrameT:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise Empty()
            data = self._queue.get(timeout=remaining)
            if data.frame == frame:
                return data
