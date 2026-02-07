from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt

Float32Array: TypeAlias = npt.NDArray[np.float32]
UInt8RgbArray: TypeAlias = npt.NDArray[np.uint8]


class StepInfo(TypedDict, total=False):
    termination_reason: str
    infractions: int
    speed_mps: float


@dataclass(frozen=True)
class VehicleState:
    """Right-handed vehicle state in meters/degrees."""

    position: Float32Array  # shape (3,), meters
    rotation_rpy: Float32Array  # shape (3,), degrees (roll, pitch, yaw)
    velocity: Float32Array  # shape (3,), meters/sec


@dataclass(frozen=True)
class VehicleCommand:
    """Low-level vehicle command in normalized ranges."""

    throttle: float
    steer: float
    brake: float

    def clamped(self) -> VehicleCommand:
        return VehicleCommand(
            throttle=_clamp(self.throttle, 0.0, 1.0),
            steer=_clamp(self.steer, -1.0, 1.0),
            brake=_clamp(self.brake, 0.0, 1.0),
        )


@dataclass(frozen=True)
class Observation:
    """Bundle RGB image, ego state, frame index, and timestamp."""

    rgb: UInt8RgbArray  # shape (H, W, 3), uint8
    ego: VehicleState
    frame: int
    timestamp: float


@dataclass(frozen=True)
class StepResult:
    """Result of one environment step."""

    obs: Observation
    reward: float
    done: bool
    info: StepInfo


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value
