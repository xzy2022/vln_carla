from __future__ import annotations

from typing import Protocol

from domain.entities import Observation, StepResult, VehicleCommand
from usecases.episode_types import EpisodeSpec, ResetInfo


class EnvInterface(Protocol):
    def reset(self, spec: EpisodeSpec) -> tuple[Observation, ResetInfo]:
        ...

    def step(self, cmd: VehicleCommand) -> StepResult:
        ...

    def close(self) -> None:
        ...

