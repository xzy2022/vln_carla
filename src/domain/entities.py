from __future__ import annotations

from dataclasses import dataclass
from typing import Any
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
    # 油门 (0, 1.0)
    # 转向 (-1.0, 1.0)
    # 刹车 (0, 1.0)
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
    """整合了图像数据（RGB 数组）、车辆自身状态（ego）、仿真帧号（frame）和时间戳（timestamp）。"""
    rgb: np.ndarray  # (H, W, 3) uint8
    ego: VehicleState
    frame: int
    timestamp: float


@dataclass(frozen=True)
class StepResult:
    """
    作用：定义环境执行一个动作（Step）后的返回结果。
    功能：包含新的观测值、奖励值（reward）、是否结束（done）以及额外的调试/辅助信息（info）。
    但是我不需要做强化学习，可能这个用不到。
    """
    obs: Observation
    reward: float
    done: bool
    info: dict[str, Any]


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value
