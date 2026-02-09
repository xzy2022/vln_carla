from __future__ import annotations

import argparse
import ctypes
import time
from pathlib import PurePosixPath


# Windows virtual-key codes
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28
VK_U = 0x55
VK_O = 0x4F
VK_ESC = 0x1B

ALLOWED_UE5_MAPS = ("Mine_01", "Town10HD_Opt")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "UE5 CARLA spectator coordinate navigator. "
            "Arrow keys move on XY plane, U/O move on Z axis."
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
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Movement speed in meters per second for each axis key.",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=12.0,
        help="Axis arrow length in meters.",
    )
    parser.add_argument(
        "--axis-z-offset",
        type=float,
        default=0.4,
        help="Lift axis drawing above origin to improve visibility.",
    )
    parser.add_argument(
        "--life-time",
        type=float,
        default=2.0,
        help="Debug draw lifetime in seconds.",
    )
    parser.add_argument(
        "--redraw-interval",
        type=float,
        default=0.5,
        help="Seconds between repeated axis redraws.",
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


def draw_ue_axes(
    world,
    carla_module,
    axis_length: float,
    life_time: float,
    axis_z_offset: float,
) -> None:
    debug = world.debug
    origin = carla_module.Location(x=0.0, y=0.0, z=axis_z_offset)
    x_end = carla_module.Location(x=axis_length, y=0.0, z=axis_z_offset)
    y_end = carla_module.Location(x=0.0, y=axis_length, z=axis_z_offset)
    z_end = carla_module.Location(x=0.0, y=0.0, z=axis_length + axis_z_offset)

    red = carla_module.Color(255, 0, 0)
    green = carla_module.Color(0, 255, 0)
    blue = carla_module.Color(0, 120, 255)
    white = carla_module.Color(255, 255, 255)

    # Keep the coordinate-axis origin marker only, avoid drawing extra origin marker
    # at the spectator initialization position to reduce visual obstruction.
    debug.draw_point(origin, size=0.18, color=white, life_time=life_time)

    debug.draw_arrow(origin, x_end, thickness=0.14, arrow_size=0.45, color=red, life_time=life_time)
    debug.draw_arrow(origin, y_end, thickness=0.14, arrow_size=0.45, color=green, life_time=life_time)
    debug.draw_arrow(origin, z_end, thickness=0.14, arrow_size=0.45, color=blue, life_time=life_time)

    debug.draw_string(x_end, "+X", draw_shadow=True, color=red, life_time=life_time)
    debug.draw_string(y_end, "+Y", draw_shadow=True, color=green, life_time=life_time)
    debug.draw_string(z_end, "+Z", draw_shadow=True, color=blue, life_time=life_time)


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.tick_hz <= 0:
        print("[error] --tick-hz must be > 0.")
        return 1
    if args.speed <= 0:
        print("[error] --speed must be > 0.")
        return 1
    if args.redraw_interval <= 0:
        print("[error] --redraw-interval must be > 0.")
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

    spectator = world.get_spectator()
    location = carla.Location(x=0.0, y=0.0, z=10.0)
    fixed_rotation = carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
    spectator.set_transform(carla.Transform(location, fixed_rotation))
    draw_ue_axes(
        world,
        carla,
        axis_length=args.axis_length,
        life_time=args.life_time,
        axis_z_offset=args.axis_z_offset,
    )

    settings = world.get_settings()
    period = 1.0 / args.tick_hz
    redraw_acc = 0.0

    print(f"[info] connected to {args.host}:{args.port}")
    print(f"[info] map: {world.get_map().name}")
    print(f"[info] sync_mode={settings.synchronous_mode}")
    print("[info] controls:")
    print("       Up/Down -> +X/-X")
    print("       Left/Right -> -Y/+Y")
    print("       U/O -> +Z/-Z")
    print("       ESC -> exit")
    print(f"[info] per-axis speed = {args.speed:.3f} m/s (keys can be combined)")
    print("[info] UE axes are drawn at world origin (X red, Y green, Z blue).")
    print(
        f"[pos] x={location.x:.3f}, y={location.y:.3f}, z={location.z:.3f}"
    )

    last_time = time.perf_counter()
    try:
        while True:
            loop_start = time.perf_counter()
            dt = loop_start - last_time
            last_time = loop_start

            if is_pressed(user32, VK_ESC):
                print("[info] exit by ESC.")
                break

            dx = 0.0
            dy = 0.0
            dz = 0.0

            if is_pressed(user32, VK_UP):
                dx += args.speed * dt
            if is_pressed(user32, VK_DOWN):
                dx -= args.speed * dt
            if is_pressed(user32, VK_RIGHT):
                dy += args.speed * dt
            if is_pressed(user32, VK_LEFT):
                dy -= args.speed * dt
            if is_pressed(user32, VK_U):
                dz += args.speed * dt
            if is_pressed(user32, VK_O):
                dz -= args.speed * dt

            moved = (dx != 0.0) or (dy != 0.0) or (dz != 0.0)
            if moved:
                location = carla.Location(
                    x=location.x + dx,
                    y=location.y + dy,
                    z=location.z + dz,
                )
                spectator.set_transform(carla.Transform(location, fixed_rotation))
                print(
                    f"[pos] x={location.x:.3f}, y={location.y:.3f}, z={location.z:.3f}"
                )
            else:
                # Keep pitch fixed to -90 even if external tools modify spectator.
                spectator.set_transform(carla.Transform(location, fixed_rotation))

            redraw_acc += dt
            if redraw_acc >= args.redraw_interval:
                draw_ue_axes(
                    world,
                    carla,
                    axis_length=args.axis_length,
                    life_time=args.life_time,
                    axis_z_offset=args.axis_z_offset,
                )
                redraw_acc = 0.0

            if settings.synchronous_mode:
                world.tick()

            elapsed = time.perf_counter() - loop_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n[info] exit by Ctrl+C.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
