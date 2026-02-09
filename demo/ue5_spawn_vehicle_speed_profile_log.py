from __future__ import annotations

import argparse
import math
import time
from datetime import datetime
from pathlib import Path, PurePosixPath


TARGET_MAP_SHORT = "Town10HD_Opt"
CONTROL_MODES = ("constant_velocity", "target_velocity", "vehicle_control", "ackermann")

DEFAULT_SPAWN_X = 0.038
DEFAULT_SPAWN_Y = 15.320
DEFAULT_SPAWN_Z = 0.15
DEFAULT_SPAWN_YAW = 180.0

PHASE1_END_S = 2.0
PHASE2_END_S = 5.0
PHASE1_SPEED_MPS = 1.0
PHASE2_SPEED_MPS = 1.5

MAX_THROTTLE_CMD = 0.85
MAX_BRAKE_CMD = 1.0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Spawn one vehicle at fixed Town10HD_Opt point, run selectable control mode, "
            "and append truth speed logs into tmp/."
        )
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="CARLA server host.")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server RPC port.")
    parser.add_argument("--timeout", type=float, default=10.0, help="Client timeout seconds.")
    parser.add_argument(
        "--map",
        type=str,
        default=TARGET_MAP_SHORT,
        help=f"Target map short name. Default: {TARGET_MAP_SHORT}.",
    )
    parser.add_argument("--tick-hz", type=float, default=20.0, help="Control loop frequency.")
    parser.add_argument("--spawn-x", type=float, default=DEFAULT_SPAWN_X, help="Spawn X.")
    parser.add_argument("--spawn-y", type=float, default=DEFAULT_SPAWN_Y, help="Spawn Y.")
    parser.add_argument("--spawn-z", type=float, default=DEFAULT_SPAWN_Z, help="Spawn Z.")
    parser.add_argument("--spawn-yaw", type=float, default=DEFAULT_SPAWN_YAW, help="Spawn yaw.")
    parser.add_argument(
        "--control-mode",
        type=str,
        choices=CONTROL_MODES,
        default="constant_velocity",
        help=(
            "Vehicle control method. "
            "constant_velocity=enable_constant_velocity, "
            "target_velocity=set_target_velocity, "
            "vehicle_control=apply_control, "
            "ackermann=apply_ackermann_control."
        ),
    )
    parser.add_argument(
        "--vehicle-blueprint",
        type=str,
        default="vehicle.mini.cooper",
        help="Preferred vehicle blueprint id.",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="tmp/town10hd_opt_speed_truth.log",
        help="Append-mode log file path.",
    )
    parser.add_argument(
        "--spectator-z",
        type=float,
        default=20.0,
        help="Spectator top-down height in meters.",
    )
    parser.add_argument(
        "--spectator-yaw",
        type=float,
        default=0.0,
        help="Spectator yaw angle in degrees (pitch is fixed at -90).",
    )
    parser.add_argument(
        "--stop-hold-seconds",
        type=float,
        default=1.0,
        help="How long to keep sampling after entering stop phase.",
    )
    parser.add_argument(
        "--destroy-on-exit",
        action="store_true",
        help="Destroy spawned vehicle on script exit.",
    )
    return parser


def short_map_name(map_path: str) -> str:
    return PurePosixPath(map_path).name


def resolve_target_map(requested_short: str, available_maps: list[str]) -> str | None:
    requested_lower = requested_short.strip().lower().strip("/")
    exact_matches = [m for m in available_maps if m.lower().strip("/") == requested_lower]
    if len(exact_matches) == 1:
        return exact_matches[0]

    short_matches = [m for m in available_maps if short_map_name(m).lower() == requested_lower]
    if len(short_matches) == 1:
        return short_matches[0]
    return None


def pick_vehicle_blueprint(world, preferred_id: str):
    library = world.get_blueprint_library()

    preferred = library.filter(preferred_id)
    if preferred:
        return preferred[0]

    all_vehicles = library.filter("vehicle.*")
    if not all_vehicles:
        raise RuntimeError("No vehicle blueprint found in this CARLA build.")
    return all_vehicles[0]


def command_speed_mps(elapsed_s: float) -> float:
    if elapsed_s < PHASE1_END_S:
        return PHASE1_SPEED_MPS
    if elapsed_s < PHASE2_END_S:
        return PHASE2_SPEED_MPS
    return 0.0


def phase_name(elapsed_s: float) -> str:
    if elapsed_s < PHASE1_END_S:
        return "phase_1_0to2s"
    if elapsed_s < PHASE2_END_S:
        return "phase_2_2to5s"
    return "stop"


def scalar_speed_mps(velocity) -> float:
    return math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z)


def clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def apply_motion_command(vehicle, carla_mod, control_mode: str, cmd_speed_mps: float, truth_speed_mps: float) -> tuple[float, float]:
    throttle_cmd = float("nan")
    brake_cmd = float("nan")

    if control_mode == "constant_velocity":
        if cmd_speed_mps > 0.0:
            vehicle.enable_constant_velocity(carla_mod.Vector3D(cmd_speed_mps, 0.0, 0.0))
            vehicle.apply_control(
                carla_mod.VehicleControl(
                    throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False
                )
            )
            throttle_cmd = 0.0
            brake_cmd = 0.0
        else:
            vehicle.disable_constant_velocity()
            vehicle.apply_control(
                carla_mod.VehicleControl(
                    throttle=0.0, steer=0.0, brake=1.0, hand_brake=False, reverse=False
                )
            )
            throttle_cmd = 0.0
            brake_cmd = 1.0
        return throttle_cmd, brake_cmd

    vehicle.disable_constant_velocity()

    if control_mode == "target_velocity":
        forward = vehicle.get_transform().get_forward_vector()
        vehicle.set_target_velocity(
            carla_mod.Vector3D(
                cmd_speed_mps * forward.x,
                cmd_speed_mps * forward.y,
                cmd_speed_mps * forward.z,
            )
        )
        if cmd_speed_mps > 0.0:
            vehicle.apply_control(
                carla_mod.VehicleControl(
                    throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False
                )
            )
            throttle_cmd = 0.0
            brake_cmd = 0.0
        else:
            vehicle.apply_control(
                carla_mod.VehicleControl(
                    throttle=0.0, steer=0.0, brake=1.0, hand_brake=False, reverse=False
                )
            )
            throttle_cmd = 0.0
            brake_cmd = 1.0
        return throttle_cmd, brake_cmd

    if control_mode == "vehicle_control":
        speed_error = cmd_speed_mps - truth_speed_mps
        if cmd_speed_mps <= 0.0:
            throttle_cmd = 0.0
            brake_cmd = 1.0
        elif speed_error >= 0.0:
            throttle_cmd = clamp(0.75 * speed_error, 0.0, MAX_THROTTLE_CMD)
            brake_cmd = 0.0
        else:
            throttle_cmd = 0.0
            brake_cmd = clamp(-0.90 * speed_error, 0.0, MAX_BRAKE_CMD)

        vehicle.apply_control(
            carla_mod.VehicleControl(
                throttle=throttle_cmd,
                steer=0.0,
                brake=brake_cmd,
                hand_brake=False,
                reverse=False,
            )
        )
        return throttle_cmd, brake_cmd

    if control_mode == "ackermann":
        speed_error = cmd_speed_mps - truth_speed_mps
        target_accel = clamp(2.2 * speed_error, -5.0, 3.0)
        if cmd_speed_mps <= 0.0:
            target_accel = -5.0

        vehicle.apply_ackermann_control(
            carla_mod.VehicleAckermannControl(
                steer=0.0,
                steer_speed=0.0,
                speed=cmd_speed_mps,
                acceleration=target_accel,
                jerk=1.0,
            )
        )
        if cmd_speed_mps <= 0.0:
            vehicle.apply_control(
                carla_mod.VehicleControl(
                    throttle=0.0, steer=0.0, brake=1.0, hand_brake=False, reverse=False
                )
            )
            throttle_cmd = 0.0
            brake_cmd = 1.0
        else:
            throttle_cmd = float("nan")
            brake_cmd = float("nan")
        return throttle_cmd, brake_cmd

    raise ValueError(f"Unsupported control_mode: {control_mode}")


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.tick_hz <= 0:
        print("[error] --tick-hz must be > 0.")
        return 1
    if args.stop_hold_seconds < 0:
        print("[error] --stop-hold-seconds must be >= 0.")
        return 1

    try:
        import carla  # type: ignore
    except ModuleNotFoundError:
        print("[error] Python package 'carla' is not installed in this environment.")
        return 1

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    try:
        world = client.get_world()
    except RuntimeError as exc:
        print(f"[error] failed to connect to CARLA server: {exc}")
        return 1

    current_map = world.get_map().name
    available_maps = sorted(client.get_available_maps())
    target_map = resolve_target_map(args.map, available_maps)
    if target_map is None:
        print(f"[error] target map '{args.map}' is not available on this server.")
        print("[info] available maps:")
        for map_path in available_maps:
            print(f"       - {map_path}")
        return 1

    target_short = short_map_name(target_map).lower()
    if short_map_name(current_map).lower() != target_short:
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

    spawn_transform = carla.Transform(
        carla.Location(x=args.spawn_x, y=args.spawn_y, z=args.spawn_z),
        carla.Rotation(pitch=0.0, yaw=args.spawn_yaw, roll=0.0),
    )
    vehicle_actor = world.try_spawn_actor(vehicle_bp, spawn_transform)
    if vehicle_actor is None:
        print(
            f"[error] failed to spawn vehicle at (x={args.spawn_x:.3f}, y={args.spawn_y:.3f}, "
            f"z={args.spawn_z:.3f}, yaw={args.spawn_yaw:.1f}). Likely collision at target location."
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
    vehicle.set_autopilot(False)
    try:
        vehicle.disable_constant_velocity()
    except RuntimeError:
        pass

    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    period = 1.0 / args.tick_hz
    settings = world.get_settings()
    spectator = world.get_spectator()
    spectator_rotation = carla.Rotation(pitch=-90.0, yaw=args.spectator_yaw, roll=0.0)

    print(f"[info] connected to {args.host}:{args.port}")
    print(f"[info] map: {world.get_map().name}")
    print(
        f"[info] spawned '{vehicle.type_id}' at x={args.spawn_x:.3f}, y={args.spawn_y:.3f}, "
        f"z={args.spawn_z:.3f}, yaw={args.spawn_yaw:.1f}"
    )
    print(f"[info] control mode: {args.control_mode}")
    print(
        "[info] selectable modes: constant_velocity, target_velocity, vehicle_control, ackermann"
    )
    print(
        f"[info] speed profile: 0-{PHASE1_END_S:.0f}s -> {PHASE1_SPEED_MPS:.1f} m/s, "
        f"{PHASE1_END_S:.0f}-{PHASE2_END_S:.0f}s -> {PHASE2_SPEED_MPS:.1f} m/s, then stop"
    )
    print(
        f"[info] spectator follow: top-down at z={args.spectator_z:.2f}, yaw={args.spectator_yaw:.1f}"
    )
    print(f"[info] appending truth speed log to: {log_path}")

    start_wall = datetime.now().isoformat(timespec="seconds")
    start_perf = time.perf_counter()

    try:
        with log_path.open("a", encoding="utf-8") as logf:
            logf.write(
                f"\n=== run_start={start_wall} map={world.get_map().name} "
                f"spawn=({args.spawn_x:.3f},{args.spawn_y:.3f},{args.spawn_z:.3f},yaw={args.spawn_yaw:.1f}) "
                f"control={args.control_mode} ===\n"
            )
            logf.write(
                "elapsed_s,phase,control_mode,cmd_speed_mps,truth_speed_mps,"
                "throttle_cmd,brake_cmd,vx,vy,vz,x,y,z\n"
            )
            logf.flush()

            while True:
                loop_start = time.perf_counter()
                elapsed_s = loop_start - start_perf

                speed_before = scalar_speed_mps(vehicle.get_velocity())
                cmd_speed = command_speed_mps(elapsed_s)
                throttle_cmd, brake_cmd = apply_motion_command(
                    vehicle=vehicle,
                    carla_mod=carla,
                    control_mode=args.control_mode,
                    cmd_speed_mps=cmd_speed,
                    truth_speed_mps=speed_before,
                )

                if settings.synchronous_mode:
                    world.tick()

                transform = vehicle.get_transform()
                velocity = vehicle.get_velocity()
                truth_speed = scalar_speed_mps(velocity)
                spectator.set_transform(
                    carla.Transform(
                        carla.Location(
                            x=transform.location.x,
                            y=transform.location.y,
                            z=args.spectator_z,
                        ),
                        spectator_rotation,
                    )
                )

                logf.write(
                    f"{elapsed_s:.3f},{phase_name(elapsed_s)},{args.control_mode},"
                    f"{cmd_speed:.3f},{truth_speed:.3f},{throttle_cmd:.3f},{brake_cmd:.3f},"
                    f"{velocity.x:.4f},{velocity.y:.4f},{velocity.z:.4f},"
                    f"{transform.location.x:.4f},{transform.location.y:.4f},{transform.location.z:.4f}\n"
                )
                logf.flush()

                if elapsed_s >= PHASE2_END_S + args.stop_hold_seconds:
                    break

                if not settings.synchronous_mode:
                    sleep_time = period - (time.perf_counter() - loop_start)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n[info] interrupted by Ctrl+C.")
    finally:
        try:
            vehicle.disable_constant_velocity()
        except RuntimeError:
            pass
        try:
            vehicle.apply_control(
                carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=False, reverse=False)
            )
        except RuntimeError:
            pass

        if args.destroy_on_exit:
            try:
                vehicle.destroy()
                print("[info] destroyed spawned vehicle.")
            except RuntimeError:
                print("[warn] failed to destroy spawned vehicle cleanly.")

    print("[info] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
