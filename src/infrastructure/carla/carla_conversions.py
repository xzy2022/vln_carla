from __future__ import annotations

import math
from typing import Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

Float32Array: TypeAlias = npt.NDArray[np.float32]
Float64Array: TypeAlias = npt.NDArray[np.float64]


class SupportsXYZ(Protocol):
    x: float
    y: float
    z: float


class SupportsRPY(Protocol):
    roll: float
    pitch: float
    yaw: float


def lh_to_rh_location(location: SupportsXYZ) -> Float32Array:
    return np.array([location.x, -location.y, location.z], dtype=np.float32)


def lh_to_rh_velocity(velocity: SupportsXYZ) -> Float32Array:
    return np.array([velocity.x, -velocity.y, velocity.z], dtype=np.float32)


def lh_to_rh_rotation(rotation: SupportsRPY) -> Float32Array:
    """Convert CARLA left-handed RPY degrees into right-handed RPY degrees."""
    roll = math.radians(rotation.roll)
    pitch = math.radians(rotation.pitch)
    yaw = math.radians(rotation.yaw)

    r_lh = _euler_zyx_to_matrix(yaw, pitch, roll)
    reflect = np.diag([1.0, -1.0, 1.0])
    r_rh = reflect @ r_lh @ reflect

    yaw_rh, pitch_rh, roll_rh = _matrix_to_euler_zyx(r_rh)
    return np.array(
        [math.degrees(roll_rh), math.degrees(pitch_rh), math.degrees(yaw_rh)],
        dtype=np.float32,
    )


def _euler_zyx_to_matrix(yaw: float, pitch: float, roll: float) -> Float64Array:
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cr = math.cos(roll)
    sr = math.sin(roll)

    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _matrix_to_euler_zyx(r: Float64Array) -> tuple[float, float, float]:
    r20 = float(r[2, 0])
    if r20 <= -1.0:
        pitch = math.pi / 2.0
    elif r20 >= 1.0:
        pitch = -math.pi / 2.0
    else:
        pitch = math.asin(-r20)

    if abs(math.cos(pitch)) < 1e-6:
        yaw = 0.0
        roll = math.atan2(-float(r[0, 1]), float(r[1, 1]))
    else:
        yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
        roll = math.atan2(float(r[2, 1]), float(r[2, 2]))

    return yaw, pitch, roll
