from __future__ import annotations

from typing import Protocol

from vln_carla.domain.entities import Observation, VehicleCommand


class AgentInterface(Protocol):
    def act(self, obs: Observation) -> VehicleCommand:
        ...
