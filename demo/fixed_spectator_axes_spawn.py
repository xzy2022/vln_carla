from __future__ import annotations

import argparse
import time


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Connect to a running UE5 CARLA server, lock spectator at a fixed top-down view, "
            "draw UE axes, and spawn one test prop near (5, 0, z<10)."
        )
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="CARLA server host.")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server RPC port.")
    parser.add_argument("--timeout", type=float, default=10.0, help="Client timeout seconds.")
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
    parser.add_argument("--spawn-x", type=float, default=5.0, help="Test prop X coordinate.")
    parser.add_argument("--spawn-y", type=float, default=3.5, help="Test prop Y coordinate.")
    parser.add_argument(
        "--spawn-min-z",
        type=float,
        default=0.2,
        help="Minimum Z (inclusive) for spawn search.",
    )
    parser.add_argument(
        "--spawn-max-z",
        type=float,
        default=9.8,
        help="Maximum Z (must stay < 10 by requirement).",
    )
    parser.add_argument(
        "--prop-blueprint",
        type=str,
        default="static.prop.barrel",
        help="Preferred prop blueprint id. Falls back to first static.prop.* if unavailable.",
    )
    parser.add_argument(
        "--tick-hz",
        type=float,
        default=10.0,
        help="Loop update rate while running.",
    )
    parser.add_argument(
        "--redraw-interval",
        type=float,
        default=0.5,
        help="Seconds between repeated axis redraws.",
    )
    parser.add_argument(
        "--destroy-on-exit",
        action="store_true",
        help="Destroy spawned test actor on Ctrl+C before exit.",
    )
    return parser


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def set_fixed_spectator(world, carla_module):
    spectator = world.get_spectator()
    transform = carla_module.Transform(
        carla_module.Location(x=0.0, y=0.0, z=20.0),
        carla_module.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
    )
    spectator.set_transform(transform)
    return spectator, transform


def draw_ue_axes(
    world,
    carla_module,
    axis_length: float,
    life_time: float,
    axis_z_offset: float,
) -> None:
    debug = world.debug
    origin_raw = carla_module.Location(x=0.0, y=0.0, z=0.0)
    origin = carla_module.Location(x=0.0, y=0.0, z=axis_z_offset)
    x_end = carla_module.Location(x=axis_length, y=0.0, z=axis_z_offset)
    y_end = carla_module.Location(x=0.0, y=axis_length, z=axis_z_offset)
    z_end = carla_module.Location(x=0.0, y=0.0, z=axis_length + axis_z_offset)

    red = carla_module.Color(255, 0, 0)
    green = carla_module.Color(0, 255, 0)
    blue = carla_module.Color(0, 120, 255)
    white = carla_module.Color(255, 255, 255)

    # Mark UE world origin itself.
    debug.draw_point(origin_raw, size=0.25, color=white, life_time=life_time)
    debug.draw_string(
        origin_raw + carla_module.Location(z=0.4),
        "O(0,0,0)",
        draw_shadow=True,
        color=white,
        life_time=life_time,
    )
    # Visual helper from true origin to lifted axis plane.
    debug.draw_arrow(
        origin_raw,
        origin,
        thickness=0.06,
        arrow_size=0.18,
        color=white,
        life_time=life_time,
    )

    debug.draw_arrow(origin, x_end, thickness=0.14, arrow_size=0.45, color=red, life_time=life_time)
    debug.draw_arrow(origin, y_end, thickness=0.14, arrow_size=0.45, color=green, life_time=life_time)
    debug.draw_arrow(origin, z_end, thickness=0.14, arrow_size=0.45, color=blue, life_time=life_time)

    debug.draw_string(x_end, "+X", draw_shadow=True, color=red, life_time=life_time)
    debug.draw_string(y_end, "+Y", draw_shadow=True, color=green, life_time=life_time)
    debug.draw_string(z_end, "+Z", draw_shadow=True, color=blue, life_time=life_time)


def pick_prop_blueprint(world, preferred_blueprint: str):
    library = world.get_blueprint_library()
    preferred = library.filter(preferred_blueprint)
    if preferred:
        return preferred[0], preferred_blueprint

    all_props = library.filter("static.prop.*")
    if not all_props:
        raise RuntimeError("No 'static.prop.*' blueprint found in this CARLA build/map.")

    bp = all_props[0]
    return bp, bp.id


def estimate_ground_z(world, carla_module, x: float, y: float, max_z: float) -> float:
    ground_projection = getattr(world, "ground_projection", None)
    if callable(ground_projection):
        labeled = ground_projection(carla_module.Location(x=x, y=y, z=max_z), 200.0)
        if labeled is not None and hasattr(labeled, "location"):
            return float(labeled.location.z)
    return 0.0


def generate_candidate_z(min_z: float, max_z: float, base_ground_z: float) -> list[float]:
    max_z = min(max_z, 9.9)  # Enforce z < 10 requirement.
    min_z = min(min_z, max_z)

    start = clamp(base_ground_z + 0.6, min_z, max_z)
    offsets = [0.0, 0.25, -0.25, 0.5, -0.5, 0.8, -0.8, 1.2, -1.2, 1.8, -1.8]

    candidates: list[float] = []
    for delta in offsets:
        z = clamp(start + delta, min_z, max_z)
        if z not in candidates:
            candidates.append(z)

    z = min_z
    while z <= max_z:
        rounded = round(z, 3)
        if rounded not in candidates:
            candidates.append(rounded)
        z += 0.3

    return candidates


def spawn_test_prop(world, carla_module, blueprint, x: float, y: float, min_z: float, max_z: float):
    base_ground_z = estimate_ground_z(world, carla_module, x=x, y=y, max_z=max_z)
    for z in generate_candidate_z(min_z=min_z, max_z=max_z, base_ground_z=base_ground_z):
        transform = carla_module.Transform(
            carla_module.Location(x=x, y=y, z=z),
            carla_module.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        )
        actor = world.try_spawn_actor(blueprint, transform)
        if actor is not None:
            return actor, transform
    return None, None


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.spawn_max_z >= 10.0:
        print("[error] --spawn-max-z must be < 10.0")
        return 1
    if args.tick_hz <= 0:
        print("[error] --tick-hz must be > 0")
        return 1
    if args.redraw_interval <= 0:
        print("[error] --redraw-interval must be > 0")
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
        print("[hint] Start UE5 CARLA first (for example with demo/start_ue5_carla.py).")
        return 1

    print(f"[info] connected to {args.host}:{args.port}")
    print(f"[info] map: {world.get_map().name}")
    settings = world.get_settings()
    print(f"[info] world synchronous_mode={settings.synchronous_mode}")

    spectator, fixed_transform = set_fixed_spectator(world, carla)
    print(f"[info] spectator fixed to: {fixed_transform}")

    draw_ue_axes(
        world,
        carla,
        axis_length=args.axis_length,
        life_time=args.life_time,
        axis_z_offset=args.axis_z_offset,
    )
    print("[info] drew UE coordinate axes at origin (X red, Y green, Z blue).")

    blueprint, blueprint_id = pick_prop_blueprint(world, args.prop_blueprint)
    actor, actor_transform = spawn_test_prop(
        world=world,
        carla_module=carla,
        blueprint=blueprint,
        x=args.spawn_x,
        y=args.spawn_y,
        min_z=args.spawn_min_z,
        max_z=args.spawn_max_z,
    )

    if actor is None or actor_transform is None:
        print("[error] failed to spawn test prop at collision-free z under 10m.")
        return 1

    print(
        f"[info] spawned test prop '{blueprint_id}' "
        f"at (x={actor_transform.location.x:.3f}, y={actor_transform.location.y:.3f}, z={actor_transform.location.z:.3f}) "
        f"actor_id={actor.id}"
    )

    period = 1.0 / args.tick_hz
    redraw_acc = 0.0
    print("[info] locking spectator at fixed top-down view. Press Ctrl+C to exit.")

    try:
        while True:
            spectator.set_transform(fixed_transform)
            if settings.synchronous_mode:
                world.tick()
            redraw_acc += period
            if redraw_acc >= args.redraw_interval:
                draw_ue_axes(
                    world,
                    carla,
                    axis_length=args.axis_length,
                    life_time=args.life_time,
                    axis_z_offset=args.axis_z_offset,
                )
                redraw_acc = 0.0
            time.sleep(period)
    except KeyboardInterrupt:
        if args.destroy_on_exit:
            try:
                actor.destroy()
                print("[info] destroyed spawned test prop on exit.")
            except RuntimeError:
                print("[warn] failed to destroy spawned test prop cleanly.")
        print("[info] exit.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
