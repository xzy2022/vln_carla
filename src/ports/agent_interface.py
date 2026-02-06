from __future__ import annotations

from typing import Protocol

from domain.entities import Observation, VehicleCommand


class AgentInterface(Protocol):
    def act(self, obs: Observation) -> VehicleCommand:
        ...

