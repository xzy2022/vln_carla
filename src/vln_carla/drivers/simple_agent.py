from __future__ import annotations

from vln_carla.domain.entities import Observation, VehicleCommand
from vln_carla.ports.agent_interface import AgentInterface


class SimpleAgent(AgentInterface):
    def __init__(self, throttle: float = 0.3) -> None:
        self._throttle = throttle

    def act(self, obs: Observation) -> VehicleCommand:
        return VehicleCommand(throttle=self._throttle, steer=0.0, brake=0.0)
