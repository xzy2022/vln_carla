from __future__ import annotations

from domain.entities import Observation
from usecases.ports.logger_interface import LoggerInterface


class InMemoryLogger(LoggerInterface):
    def __init__(self) -> None:
        self.items: list[Observation] = []

    def save(self, obs: Observation) -> None:
        self.items.append(obs)

    def flush(self) -> None:
        return None

