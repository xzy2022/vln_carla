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
VK_C = 0x43
VK_B = 0x42
VK_ESC = 0x1B

ALLOWED_UE5_MAPS = ("Mine_01", "Town10HD_Opt")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "UE5 CARLA spectator coordinate navigator. "
            "Arrow keys move on XY plane, U/O move on Z axis, C/B spawn actors at current XY."
        )
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="CARLA server host.")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server RPC port.")
    parser.add_argument("--timeout", type=float, default=10.0, help="Client timeout seconds.")
    parser.add_argument(
        "--map",
        type=str,
        choices=ALLOWED_UE5_MAPS,
        default="Town10HD_Opt",
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
    parser.add_argument(
        "--vehicle-blueprint",
        type=str,
        default="vehicle.mini.cooper",
        help="Preferred vehicle blueprint id for C-key spawn.",
    )
    parser.add_argument(
        "--barrel-blueprint",
        type=str,
        default="static.prop.barrel",
        help="Preferred barrel blueprint id for B-key spawn.",
    )
    parser.add_argument(
        "--vehicle-yaw",
        type=float,
        default=0.0,
        help="Yaw of spawned vehicle when pressing C (degrees).",
    )
    parser.add_argument(
        "--barrel-yaw",
        type=float,
        default=0.0,
        help="Yaw of spawned barrel when pressing B (degrees).",
    )
    parser.add_argument(
        "--spawn-min-z",
        type=float,
        default=-200.0,
        help="Minimum Z for spawn search.",
    )
    parser.add_argument(
        "--spawn-max-z",
        type=float,
        default=200.0,
        help="Maximum Z for spawn search.",
    )
    parser.add_argument(
        "--spawn-probe-top-z",
        type=float,
        default=120.0,
        help="Top Z used for ground projection probing.",
    )
    parser.add_argument(
        "--spawn-probe-distance",
        type=float,
        default=300.0,
        help="Ground projection probe distance.",
    )
    parser.add_argument(
        "--destroy-spawned-on-exit",
        action="store_true",
        help="Destroy actors spawned by C/B when exiting.",
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

    # Keep only coordinate-axis origin marker.
    debug.draw_point(origin, size=0.18, color=white, life_time=life_time)

    debug.draw_arrow(origin, x_end, thickness=0.14, arrow_size=0.45, color=red, life_time=life_time)
    debug.draw_arrow(origin, y_end, thickness=0.14, arrow_size=0.45, color=green, life_time=life_time)
    debug.draw_arrow(origin, z_end, thickness=0.14, arrow_size=0.45, color=blue, life_time=life_time)

    debug.draw_string(x_end, "+X", draw_shadow=True, color=red, life_time=life_time)
    debug.draw_string(y_end, "+Y", draw_shadow=True, color=green, life_time=life_time)
    debug.draw_string(z_end, "+Z", draw_shadow=True, color=blue, life_time=life_time)


def pick_vehicle_blueprint(world, preferred_id: str):
    library = world.get_blueprint_library()
    preferred = library.filter(preferred_id)
    if preferred:
        return preferred[0]

    vehicles = library.filter("vehicle.*")
    if not vehicles:
        raise RuntimeError("No vehicle blueprint found in this CARLA build.")
    return vehicles[0]


def pick_barrel_blueprint(world, preferred_id: str):
    library = world.get_blueprint_library()
    preferred = library.filter(preferred_id)
    if preferred:
        return preferred[0]

    all_props = library.filter("static.prop.*")
    if not all_props:
        raise RuntimeError("No static.prop.* blueprint found in this CARLA build.")

    barrel_candidates = [bp for bp in all_props if "barrel" in bp.id.lower()]
    if barrel_candidates:
        return barrel_candidates[0]
    return all_props[0]


def estimate_ground_z(
    world,
    carla_module,
    x: float,
    y: float,
    probe_top_z: float,
    probe_distance: float,
) -> float | None:
    ground_projection = getattr(world, "ground_projection", None)
    if callable(ground_projection):
        labeled = ground_projection(
            carla_module.Location(x=x, y=y, z=probe_top_z),
            probe_distance,
        )
        if labeled is not None and hasattr(labeled, "location"):
            return float(labeled.location.z)
    return None


def generate_candidate_z_values(base_ground_z: float, min_z: float, max_z: float) -> list[float]:
    start = clamp(base_ground_z + 0.6, min_z, max_z)
    offsets = [
        0.0,
        0.2,
        -0.2,
        0.5,
        -0.5,
        1.0,
        -1.0,
        1.5,
        -1.5,
        2.0,
        -2.0,
        3.0,
        -3.0,
        5.0,
        -5.0,
    ]

    seen: set[float] = set()
    candidates: list[float] = []

    def add(z: float) -> None:
        z_rounded = round(clamp(z, min_z, max_z), 3)
        if z_rounded not in seen:
            seen.add(z_rounded)
            candidates.append(z_rounded)

    for delta in offsets:
        add(start + delta)

    z = start - 6.0
    while z <= start + 6.0:
        add(z)
        z += 0.4

    z = min_z
    while z <= max_z:
        add(z)
        z += 2.0

    add(min_z)
    add(max_z)
    return candidates


def try_spawn_actor_at_xy(
    world,
    carla_module,
    blueprint,
    x: float,
    y: float,
    yaw: float,
    min_z: float,
    max_z: float,
    probe_top_z: float,
    probe_distance: float,
    fallback_base_z: float,
):
    base_ground_z = estimate_ground_z(
        world,
        carla_module,
        x=x,
        y=y,
        probe_top_z=probe_top_z,
        probe_distance=probe_distance,
    )
    if base_ground_z is None:
        base_ground_z = fallback_base_z

    for z in generate_candidate_z_values(base_ground_z=base_ground_z, min_z=min_z, max_z=max_z):
        transform = carla_module.Transform(
            carla_module.Location(x=x, y=y, z=z),
            carla_module.Rotation(pitch=0.0, yaw=yaw, roll=0.0),
        )
        actor = world.try_spawn_actor(blueprint, transform)
        if actor is not None:
            return actor, transform, base_ground_z

    return None, None, base_ground_z


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
    if args.spawn_probe_distance <= 0:
        print("[error] --spawn-probe-distance must be > 0.")
        return 1
    if args.spawn_min_z > args.spawn_max_z:
        print("[error] --spawn-min-z must be <= --spawn-max-z.")
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

    try:
        vehicle_bp = pick_vehicle_blueprint(world, args.vehicle_blueprint)
        barrel_bp = pick_barrel_blueprint(world, args.barrel_blueprint)
    except RuntimeError as exc:
        print(f"[error] {exc}")
        return 1

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
    spawned_actors = []

    print(f"[info] connected to {args.host}:{args.port}")
    print(f"[info] map: {world.get_map().name}")
    print(f"[info] sync_mode={settings.synchronous_mode}")
    print("[info] controls:")
    print("       Up/Down -> +X/-X")
    print("       Left/Right -> -Y/+Y")
    print("       U/O -> +Z/-Z")
    print("       C -> spawn one vehicle at current XY with auto Z")
    print("       B -> spawn one barrel at current XY with auto Z")
    print("       ESC -> exit")
    print(f"[info] per-axis speed = {args.speed:.3f} m/s (keys can be combined)")
    print("[info] UE axes are drawn at world origin (X red, Y green, Z blue).")

    last_time = time.perf_counter()
    c_prev = False
    b_prev = False

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
            else:
                # Keep pitch fixed to -90 even if external tools modify spectator.
                spectator.set_transform(carla.Transform(location, fixed_rotation))

            c_now = is_pressed(user32, VK_C)
            if c_now and not c_prev:
                vehicle_actor, vehicle_transform, vehicle_ground_z = try_spawn_actor_at_xy(
                    world=world,
                    carla_module=carla,
                    blueprint=vehicle_bp,
                    x=location.x,
                    y=location.y,
                    yaw=args.vehicle_yaw,
                    min_z=args.spawn_min_z,
                    max_z=args.spawn_max_z,
                    probe_top_z=args.spawn_probe_top_z,
                    probe_distance=args.spawn_probe_distance,
                    fallback_base_z=location.z,
                )
                if vehicle_actor is None or vehicle_transform is None:
                    print(
                        f"[spawn] vehicle FAILED blueprint='{vehicle_bp.id}' "
                        f"x={location.x:.3f}, y={location.y:.3f}, "
                        f"z_search=[{args.spawn_min_z:.3f}, {args.spawn_max_z:.3f}], "
                        f"ground_z_est={vehicle_ground_z:.3f}, yaw={args.vehicle_yaw:.1f}"
                    )
                else:
                    spawned_actors.append(vehicle_actor)
                    print(
                        f"[spawn] vehicle blueprint='{vehicle_actor.type_id}' actor_id={vehicle_actor.id} "
                        f"x={vehicle_transform.location.x:.3f}, "
                        f"y={vehicle_transform.location.y:.3f}, "
                        f"z={vehicle_transform.location.z:.3f}, "
                        f"yaw={vehicle_transform.rotation.yaw:.1f}, "
                        f"ground_z_est={vehicle_ground_z:.3f}"
                    )
            c_prev = c_now

            b_now = is_pressed(user32, VK_B)
            if b_now and not b_prev:
                barrel_actor, barrel_transform, barrel_ground_z = try_spawn_actor_at_xy(
                    world=world,
                    carla_module=carla,
                    blueprint=barrel_bp,
                    x=location.x,
                    y=location.y,
                    yaw=args.barrel_yaw,
                    min_z=args.spawn_min_z,
                    max_z=args.spawn_max_z,
                    probe_top_z=args.spawn_probe_top_z,
                    probe_distance=args.spawn_probe_distance,
                    fallback_base_z=location.z,
                )
                if barrel_actor is None or barrel_transform is None:
                    print(
                        f"[spawn] barrel FAILED blueprint='{barrel_bp.id}' "
                        f"x={location.x:.3f}, y={location.y:.3f}, "
                        f"z_search=[{args.spawn_min_z:.3f}, {args.spawn_max_z:.3f}], "
                        f"ground_z_est={barrel_ground_z:.3f}, yaw={args.barrel_yaw:.1f}"
                    )
                else:
                    spawned_actors.append(barrel_actor)
                    print(
                        f"[spawn] barrel blueprint='{barrel_actor.type_id}' actor_id={barrel_actor.id} "
                        f"x={barrel_transform.location.x:.3f}, "
                        f"y={barrel_transform.location.y:.3f}, "
                        f"z={barrel_transform.location.z:.3f}, "
                        f"yaw={barrel_transform.rotation.yaw:.1f}, "
                        f"ground_z_est={barrel_ground_z:.3f}"
                    )
            b_prev = b_now

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
    finally:
        if args.destroy_spawned_on_exit and spawned_actors:
            destroyed = 0
            failed = 0
            for actor in spawned_actors:
                try:
                    actor.destroy()
                    destroyed += 1
                except RuntimeError:
                    failed += 1
            print(f"[info] destroyed {destroyed} spawned actor(s); failed {failed}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
