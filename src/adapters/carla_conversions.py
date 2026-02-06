from __future__ import annotations

import math
from typing import Any

import numpy as np


def lh_to_rh_location(location: Any) -> np.ndarray:
    return np.array([location.x, -location.y, location.z], dtype=np.float32)


def lh_to_rh_velocity(velocity: Any) -> np.ndarray:
    return np.array([velocity.x, -velocity.y, velocity.z], dtype=np.float32)


def lh_to_rh_rotation(rotation: Any) -> np.ndarray:
    """
    Convert CARLA left-handed rotation (roll, pitch, yaw in degrees)
    to right-handed rotation (roll, pitch, yaw in degrees) by reflection.
    """
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


def _euler_zyx_to_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
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


def _matrix_to_euler_zyx(r: np.ndarray) -> tuple[float, float, float]:
    r20 = float(r[2, 0])
    if r20 <= -1.0:
        pitch = math.pi / 2.0
    elif r20 >= 1.0:
        pitch = -math.pi / 2.0
    else:
        pitch = math.asin(-r20)

    if abs(math.cos(pitch)) < 1e-6:
        yaw = 0.0
        roll = math.atan2(-r[0, 1], r[1, 1])
    else:
        yaw = math.atan2(r[1, 0], r[0, 0])
        roll = math.atan2(r[2, 1], r[2, 2])

    return yaw, pitch, roll
