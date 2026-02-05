from vln_carla.domain.entities import VehicleCommand


def test_vehicle_command_clamp():
    cmd = VehicleCommand(throttle=2.0, steer=-2.0, brake=-1.0)
    clamped = cmd.clamped()

    assert clamped.throttle == 1.0
    assert clamped.steer == -1.0
    assert clamped.brake == 0.0
