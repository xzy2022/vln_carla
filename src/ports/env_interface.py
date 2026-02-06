from __future__ import annotations

from typing import Protocol

from domain.entities import Observation, StepResult, VehicleCommand


class EnvInterface(Protocol):
    def reset(self) -> Observation:
        ...

    def step(self, cmd: VehicleCommand) -> StepResult:
        ...

    def close(self) -> None:
        ...

