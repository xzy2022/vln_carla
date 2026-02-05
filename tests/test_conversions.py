import math

import numpy as np

from vln_carla.adapters.carla_conversions import (
    lh_to_rh_location,
    lh_to_rh_rotation,
    lh_to_rh_velocity,
)


class DummyVec:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class DummyRot:
    def __init__(self, roll: float, pitch: float, yaw: float) -> None:
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw


def test_location_velocity_flip_y():
    loc = DummyVec(1.0, 2.0, 3.0)
    vel = DummyVec(-1.0, -2.0, -3.0)

    assert np.allclose(lh_to_rh_location(loc), [1.0, -2.0, 3.0])
    assert np.allclose(lh_to_rh_velocity(vel), [-1.0, 2.0, -3.0])


def test_rotation_yaw_sign_flip():
    rot = DummyRot(roll=0.0, pitch=0.0, yaw=90.0)
    rpy = lh_to_rh_rotation(rot)

    assert math.isclose(rpy[2], -90.0, abs_tol=1e-4)
    assert math.isclose(rpy[0], 0.0, abs_tol=1e-4)
    assert math.isclose(rpy[1], 0.0, abs_tol=1e-4)
