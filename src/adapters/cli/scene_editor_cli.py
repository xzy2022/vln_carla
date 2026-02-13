from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence

from adapters.cli.scene_editor_keyboard import (
    VK_B,
    VK_C,
    VK_DOWN,
    VK_ESC,
    VK_LEFT,
    VK_O,
    VK_RIGHT,
    VK_U,
    VK_UP,
    KeyboardInterface,
)
from usecases.scene_editor.dtos import (
    ConnectRequest,
    FollowRequest,
    QuickSpawnRequest,
    SpawnResult,
    SpectatorDelta,
)
from usecases.scene_editor.input_port import SceneEditorInputPort


class SceneEditorCliArgs(argparse.Namespace):
    host: str
    port: int
    timeout: float
    map: str | None
    tick_hz: float
    speed: float
    initial_spectator_x: float
    initial_spectator_y: float
    initial_spectator_z: float
    initial_spectator_yaw: float
    vehicle_blueprint: str
    barrel_blueprint: str
    vehicle_yaw: float
    barrel_yaw: float
    spawn_min_z: float
    spawn_max_z: float
    spawn_probe_top_z: float
    spawn_probe_distance: float
    follow_latest: bool
    follow_role_name: str | None
    follow_actor_id: int | None
    destroy_spawned_on_exit: bool


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CARLA scene editor CLI. Arrow keys move spectator on XY, U/O control Z, "
            "C/B spawn vehicle/barrel at current XY with collision-aware Z."
        )
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="CARLA server host.")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server RPC port.")
    parser.add_argument("--timeout", type=float, default=10.0, help="Client timeout seconds.")
    parser.add_argument(
        "--map",
        type=str,
        default=None,
        help="Optional target map name (full path or short name).",
    )
    parser.add_argument("--tick-hz", type=float, default=30.0, help="Control loop frequency.")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Spectator movement speed for each axis key in m/s.",
    )
    parser.add_argument("--initial-spectator-x", type=float, default=0.0)
    parser.add_argument("--initial-spectator-y", type=float, default=0.0)
    parser.add_argument("--initial-spectator-z", type=float, default=20.0)
    parser.add_argument("--initial-spectator-yaw", type=float, default=0.0)
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
        help="Preferred obstacle blueprint id for B-key spawn.",
    )
    parser.add_argument("--vehicle-yaw", type=float, default=0.0, help="Yaw used for C-key spawn.")
    parser.add_argument("--barrel-yaw", type=float, default=0.0, help="Yaw used for B-key spawn.")
    parser.add_argument("--spawn-min-z", type=float, default=-200.0)
    parser.add_argument("--spawn-max-z", type=float, default=200.0)
    parser.add_argument("--spawn-probe-top-z", type=float, default=120.0)
    parser.add_argument("--spawn-probe-distance", type=float, default=300.0)
    parser.add_argument(
        "--follow-latest",
        action="store_true",
        help="Bind top-down follow to latest spawned vehicle.",
    )
    parser.add_argument(
        "--follow-role-name",
        type=str,
        default=None,
        help="Bind top-down follow to vehicle role_name.",
    )
    parser.add_argument(
        "--follow-actor-id",
        type=int,
        default=None,
        help="Bind top-down follow to explicit vehicle actor_id.",
    )
    parser.add_argument(
        "--destroy-spawned-on-exit",
        action="store_true",
        help="Destroy actors spawned by this tool when exiting.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> SceneEditorCliArgs:
    parser = build_arg_parser()
    namespace = SceneEditorCliArgs()
    parsed = parser.parse_args(args=list(argv) if argv is not None else None, namespace=namespace)
    return parsed


def run_scene_editor_cli(
    args: SceneEditorCliArgs,
    usecase: SceneEditorInputPort,
    keyboard: KeyboardInterface,
    *,
    perf_counter: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    _validate_args(args)

    connection = usecase.connect(
        ConnectRequest(
            host=args.host,
            port=args.port,
            timeout_s=args.timeout,
            target_map=args.map,
            initial_spectator_x=args.initial_spectator_x,
            initial_spectator_y=args.initial_spectator_y,
            initial_spectator_z=args.initial_spectator_z,
            initial_spectator_yaw=args.initial_spectator_yaw,
        )
    )
    print(f"[info] connected to {connection.host}:{connection.port}")
    print(f"[info] map: {connection.map_name}")
    print(f"[info] sync_mode={connection.synchronous_mode}")
    print("[info] controls:")
    print("       Up/Down -> +X/-X")
    print("       Left/Right -> -Y/+Y")
    print("       U/O -> +Z/-Z")
    print("       C -> spawn one vehicle at current XY with auto Z")
    print("       B -> spawn one barrel at current XY with auto Z")
    print("       ESC -> exit")
    print(f"[info] per-axis speed = {args.speed:.3f} m/s (keys can be combined)")

    if args.follow_latest:
        follow_state = usecase.bind_topdown_follow()
        print(f"[follow] {follow_state.message}")
    elif args.follow_role_name is not None or args.follow_actor_id is not None:
        follow_state = usecase.bind_topdown_follow(
            FollowRequest(role_name=args.follow_role_name, actor_id=args.follow_actor_id)
        )
        print(f"[follow] {follow_state.message}")

    period = 1.0 / args.tick_hz
    last_time = perf_counter()
    c_prev = False
    b_prev = False

    try:
        while True:
            loop_start = perf_counter()
            dt = loop_start - last_time
            last_time = loop_start

            if keyboard.is_pressed(VK_ESC):
                print("[info] exit by ESC.")
                break

            dx = 0.0
            dy = 0.0
            dz = 0.0
            if keyboard.is_pressed(VK_UP):
                dx += args.speed * dt
            if keyboard.is_pressed(VK_DOWN):
                dx -= args.speed * dt
            if keyboard.is_pressed(VK_RIGHT):
                dy += args.speed * dt
            if keyboard.is_pressed(VK_LEFT):
                dy -= args.speed * dt
            if keyboard.is_pressed(VK_U):
                dz += args.speed * dt
            if keyboard.is_pressed(VK_O):
                dz -= args.speed * dt

            if dx != 0.0 or dy != 0.0 or dz != 0.0:
                usecase.move_spectator(SpectatorDelta(dx=dx, dy=dy, dz=dz))

            c_now = keyboard.is_pressed(VK_C)
            if c_now and not c_prev:
                result = usecase.spawn_vehicle_at_current_xy(
                    QuickSpawnRequest(
                        yaw=args.vehicle_yaw,
                        blueprint_id=args.vehicle_blueprint,
                        spawn_min_z=args.spawn_min_z,
                        spawn_max_z=args.spawn_max_z,
                        spawn_probe_top_z=args.spawn_probe_top_z,
                        spawn_probe_distance=args.spawn_probe_distance,
                    )
                )
                _print_spawn_result("vehicle", result)
            c_prev = c_now

            b_now = keyboard.is_pressed(VK_B)
            if b_now and not b_prev:
                result = usecase.spawn_obstacle_at_current_xy(
                    QuickSpawnRequest(
                        yaw=args.barrel_yaw,
                        blueprint_id=args.barrel_blueprint,
                        spawn_min_z=args.spawn_min_z,
                        spawn_max_z=args.spawn_max_z,
                        spawn_probe_top_z=args.spawn_probe_top_z,
                        spawn_probe_distance=args.spawn_probe_distance,
                    )
                )
                _print_spawn_result("barrel", result)
            b_prev = b_now

            usecase.tick_follow()

            elapsed = perf_counter() - loop_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n[info] exit by Ctrl+C.")
    finally:
        cleanup = usecase.close(destroy_spawned=args.destroy_spawned_on_exit)
        if args.destroy_spawned_on_exit:
            print(
                "[info] destroyed "
                f"{cleanup.destroyed_count} spawned actor(s); failed {cleanup.failed_count}."
            )

    return 0


def _validate_args(args: SceneEditorCliArgs) -> None:
    if args.tick_hz <= 0:
        raise RuntimeError("--tick-hz must be > 0.")
    if args.speed <= 0:
        raise RuntimeError("--speed must be > 0.")
    if args.spawn_probe_distance <= 0:
        raise RuntimeError("--spawn-probe-distance must be > 0.")
    if args.spawn_min_z > args.spawn_max_z:
        raise RuntimeError("--spawn-min-z must be <= --spawn-max-z.")


def _print_spawn_result(label: str, result: SpawnResult) -> None:
    if result.success and result.actor is not None:
        print(
            f"[spawn] {label} blueprint='{result.actor.type_id}' actor_id={result.actor.actor_id} "
            f"x={result.actor.x:.3f}, y={result.actor.y:.3f}, z={result.actor.z:.3f}, "
            f"yaw={result.actor.yaw:.1f}, ground_z_est={result.ground_z_estimate:.3f}"
        )
        return
    print(
        f"[spawn] {label} FAILED requested_blueprint='{result.requested_blueprint_id}' "
        f"used_blueprint='{result.used_blueprint_id}' "
        f"ground_z_est={result.ground_z_estimate:.3f} reason={result.message}"
    )
