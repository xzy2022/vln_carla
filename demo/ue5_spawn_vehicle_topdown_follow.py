from __future__ import annotations

import argparse
import ctypes
import time
from pathlib import PurePosixPath


VK_U = 0x55
VK_O = 0x4F
VK_ESC = 0x1B
VK_Q = 0x51
VK_E = 0x45
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28

ALLOWED_UE5_MAPS = ("Mine_01", "Town10HD_Opt")

MAX_FORWARD_SPEED_MPS = 1.0
FIXED_STEER_ABS = 0.5
THROTTLE_CMD = 0.35
BRAKE_CMD = 0.8
SPEED_HOLD_BRAKE_CMD = 0.2
SPECTATOR_YAW_SPEED_DEG_S = 45.0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "UE5 CARLA demo: spawn one small vehicle at a fixed coordinate and keep "
            "spectator vertically top-down above vehicle XY; U/O adjust spectator Z."
        )
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="CARLA server host.")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server RPC port.")
    parser.add_argument("--timeout", type=float, default=10.0, help="Client timeout seconds.")
    parser.add_argument(
        "--map",
        type=str,
        choices=ALLOWED_UE5_MAPS,
        default="Mine_01",
        help="Target UE5 map. Allowed: Mine_01, Town10HD_Opt.",
    )
    parser.add_argument("--tick-hz", type=float, default=30.0, help="Control loop frequency.")
    parser.add_argument("--spawn-x", type=float, default=18, help="Vehicle spawn X coordinate.")
    parser.add_argument("--spawn-y", type=float, default=88, help="Vehicle spawn Y coordinate.")
    parser.add_argument("--spawn-z", type=float, default=-65.5, help="Vehicle spawn Z coordinate.")
    parser.add_argument(
        "--spawn-yaw",
        type=float,
        default=0.0,
        help="Vehicle heading yaw (rotation around Z axis, degrees).",
    )
    parser.add_argument(
        "--spectator-z",
        type=float,
        default=20.0,
        help="Initial spectator Z height in meters.",
    )
    parser.add_argument(
        "--spectator-z-speed",
        type=float,
        default=6.0,
        help="Spectator Z adjustment speed (m/s) for U/O keys.",
    )
    parser.add_argument(
        "--min-spectator-z",
        type=float,
        default=-20.0,
        help="Minimum spectator Z height clamp.",
    )
    parser.add_argument(
        "--max-spectator-z",
        type=float,
        default=120.0,
        help="Maximum spectator Z height clamp.",
    )
    parser.add_argument(
        "--vehicle-blueprint",
        type=str,
        default="vehicle.mini.cooper",
        help="Preferred vehicle blueprint id for the spawned small car.",
    )
    parser.add_argument(
        "--destroy-on-exit",
        action="store_true",
        help="Destroy spawned vehicle on exit.",
    )
    return parser


def is_pressed(user32: ctypes.WinDLL, vk_code: int) -> bool:
    return bool(user32.GetAsyncKeyState(vk_code) & 0x8000)


def short_map_name(map_path: str) -> str:
    return PurePosixPath(map_path).name


def resolve_target_map(requested_short: str, available_maps: list[str]) -> str | None:
    requested_lower = requested_short.lower()
    short_matches = [m for m in available_maps if short_map_name(m).lower() == requested_lower]
    if len(short_matches) == 1:
        return short_matches[0]
    exact_matches = [m for m in available_maps if m.lower().strip("/") == requested_lower]
    if len(exact_matches) == 1:
        return exact_matches[0]
    return None


def clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def pick_vehicle_blueprint(world, preferred_id: str):
    library = world.get_blueprint_library()

    preferred = library.filter(preferred_id)
    if preferred:
        return preferred[0]

    all_vehicles = library.filter("vehicle.*")
    if not all_vehicles:
        raise RuntimeError("No vehicle blueprint found in this CARLA build.")

    return all_vehicles[0]


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.tick_hz <= 0:
        print("[error] --tick-hz must be > 0.")
        return 1
    if args.spectator_z_speed <= 0:
        print("[error] --spectator-z-speed must be > 0.")
        return 1
    if args.min_spectator_z > args.max_spectator_z:
        print("[error] --min-spectator-z must be <= --max-spectator-z.")
        return 1

    try:
        import carla  # type: ignore
    except ModuleNotFoundError:
        print("[error] Python package 'carla' is not installed in this environment.")
        return 1

    try:
        user32 = ctypes.windll.user32
    except AttributeError:
        print("[error] This script requires Windows (ctypes.windll.user32 not available).")
        return 1

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    try:
        world = client.get_world()
    except RuntimeError as exc:
        print(f"[error] failed to connect to CARLA server: {exc}")
        return 1

    current_map = world.get_map().name
    current_map_short = short_map_name(current_map)
    available_maps = sorted(client.get_available_maps())

    target_map = resolve_target_map(args.map, available_maps)
    if target_map is None:
        print(f"[error] target map '{args.map}' is not available on this server.")
        print("[info] available maps:")
        for map_path in available_maps:
            print(f"       - {map_path}")
        return 1

    target_map_short = short_map_name(target_map)
    if current_map_short.lower() != target_map_short.lower():
        print(f"[info] current map: {current_map}")
        print(f"[info] switching map to: {target_map}")
        print("[info] load_world will destroy actors in current world.")
        try:
            world = client.load_world(target_map)
        except RuntimeError as exc:
            print(f"[error] failed to load target map '{target_map}': {exc}")
            return 1
        print(f"[info] switched to map: {world.get_map().name}")
    else:
        print(f"[info] current map already matches target: {current_map}")

    vehicle_bp = pick_vehicle_blueprint(world, args.vehicle_blueprint)
    if vehicle_bp.has_attribute("role_name"):
        vehicle_bp.set_attribute("role_name", "hero")

    # Configurable spawn transform:
    # position = (x, y, z), heading = yaw around Z axis.
    spawn_transform = carla.Transform(
        carla.Location(x=args.spawn_x, y=args.spawn_y, z=args.spawn_z),
        carla.Rotation(pitch=0.0, yaw=args.spawn_yaw, roll=0.0),
    )

    vehicle_actor = world.try_spawn_actor(vehicle_bp, spawn_transform)
    if vehicle_actor is None:
        print(
            f"[error] failed to spawn vehicle at (x={args.spawn_x:.3f}, "
            f"y={args.spawn_y:.3f}, z={args.spawn_z:.3f}, yaw={args.spawn_yaw:.1f}). "
            "Likely a collision at target location."
        )
        return 1

    if not isinstance(vehicle_actor, carla.Vehicle):
        print("[error] spawned actor is not a vehicle.")
        try:
            vehicle_actor.destroy()
        except RuntimeError:
            pass
        return 1

    vehicle = vehicle_actor
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0, hand_brake=False))

    spectator = world.get_spectator()
    spectator_yaw = 0.0
    spectator_rotation = carla.Rotation(pitch=-90.0, yaw=spectator_yaw, roll=0.0)
    spectator_z = clamp(args.spectator_z, args.min_spectator_z, args.max_spectator_z)

    period = 1.0 / args.tick_hz
    settings = world.get_settings()

    print(f"[info] connected to {args.host}:{args.port}")
    print(f"[info] map: {world.get_map().name}")
    print(f"[info] world synchronous_mode={settings.synchronous_mode}")
    print(
        f"[info] spawned '{vehicle.type_id}' at "
        f"x={spawn_transform.location.x:.3f}, y={spawn_transform.location.y:.3f}, "
        f"z={spawn_transform.location.z:.3f}, yaw={spawn_transform.rotation.yaw:.1f}"
    )
    print("[info] spectator follows vehicle XY with fixed vertical-down orientation.")
    print("[info] controls:")
    print("       Up -> throttle forward (speed limited to 1.0 m/s)")
    print("       Down -> brake")
    print("       Left/Right -> steer -0.5/+0.5")
    print("       Q/E -> rotate spectator around Z axis (+/- fixed speed)")
    print("       U/O -> increase/decrease spectator Z")
    print("       ESC -> exit")

    last_time = time.perf_counter()
    last_reported_z = spectator_z

    try:
        while True:
            now = time.perf_counter()
            dt = now - last_time
            last_time = now

            if is_pressed(user32, VK_ESC):
                print("[info] exit by ESC.")
                break

            dz = 0.0
            if is_pressed(user32, VK_U):
                dz += args.spectator_z_speed * dt
            if is_pressed(user32, VK_O):
                dz -= args.spectator_z_speed * dt

            if dz != 0.0:
                spectator_z = clamp(
                    spectator_z + dz,
                    args.min_spectator_z,
                    args.max_spectator_z,
                )

            yaw_delta = 0.0
            q_pressed = is_pressed(user32, VK_Q)
            e_pressed = is_pressed(user32, VK_E)
            if q_pressed and not e_pressed:
                yaw_delta = SPECTATOR_YAW_SPEED_DEG_S * dt
            elif e_pressed and not q_pressed:
                yaw_delta = -SPECTATOR_YAW_SPEED_DEG_S * dt
            if yaw_delta != 0.0:
                spectator_yaw += yaw_delta
                # Keep yaw bounded for readability/stability.
                while spectator_yaw > 180.0:
                    spectator_yaw -= 360.0
                while spectator_yaw < -180.0:
                    spectator_yaw += 360.0

            speed_vec = vehicle.get_velocity()
            speed_mps = (speed_vec.x**2 + speed_vec.y**2 + speed_vec.z**2) ** 0.5

            up_pressed = is_pressed(user32, VK_UP)
            down_pressed = is_pressed(user32, VK_DOWN)
            left_pressed = is_pressed(user32, VK_LEFT)
            right_pressed = is_pressed(user32, VK_RIGHT)

            throttle = 0.0
            brake = 0.0

            if down_pressed:
                brake = BRAKE_CMD
            elif up_pressed:
                if speed_mps < MAX_FORWARD_SPEED_MPS:
                    throttle = THROTTLE_CMD
                else:
                    brake = SPEED_HOLD_BRAKE_CMD

            steer = 0.0
            if left_pressed and not right_pressed:
                steer = -FIXED_STEER_ABS
            elif right_pressed and not left_pressed:
                steer = FIXED_STEER_ABS

            vehicle.apply_control(
                carla.VehicleControl(
                    throttle=throttle,
                    steer=steer,
                    brake=brake,
                    hand_brake=False,
                    reverse=False,
                )
            )

            vehicle_transform = vehicle.get_transform()
            spectator_location = carla.Location(
                x=vehicle_transform.location.x,
                y=vehicle_transform.location.y,
                z=spectator_z,
            )
            spectator_rotation.yaw = spectator_yaw
            spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))

            if abs(spectator_z - last_reported_z) >= 0.01:
                print(f"[spectator] z={spectator_z:.3f}")
                last_reported_z = spectator_z

            if settings.synchronous_mode:
                world.tick()

            elapsed = time.perf_counter() - now
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n[info] exit by Ctrl+C.")
    finally:
        if args.destroy_on_exit:
            try:
                vehicle.destroy()
                print("[info] destroyed spawned vehicle.")
            except RuntimeError:
                print("[warn] failed to destroy spawned vehicle cleanly.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
