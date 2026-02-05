from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VehicleState:
    """Right-handed vehicle state in meters/degrees."""

    position: np.ndarray  # shape (3,), meters
    rotation_rpy: np.ndarray  # shape (3,), degrees (roll, pitch, yaw)
    velocity: np.ndarray  # shape (3,), meters/sec


@dataclass(frozen=True)
class VehicleCommand:
    """Low-level vehicle command in normalized ranges."""

    throttle: float
    steer: float
    brake: float

    def clamped(self) -> "VehicleCommand":
        return VehicleCommand(
            throttle=_clamp(self.throttle, 0.0, 1.0),
            steer=_clamp(self.steer, -1.0, 1.0),
            brake=_clamp(self.brake, 0.0, 1.0),
        )


@dataclass(frozen=True)
class Observation:
    rgb: np.ndarray  # (H, W, 3) uint8
    ego: VehicleState
    frame: int
    timestamp: float


@dataclass(frozen=True)
class StepResult:
    obs: Observation
    reward: float
    done: bool
    info: dict


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value
