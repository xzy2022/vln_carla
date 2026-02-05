from __future__ import annotations

from typing import Protocol

from vln_carla.domain.entities import Observation


class LoggerInterface(Protocol):
    def save(self, obs: Observation) -> None:
        ...

    def flush(self) -> None:
        ...
