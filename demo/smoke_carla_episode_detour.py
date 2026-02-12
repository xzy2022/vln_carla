from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from dataclasses import dataclass

import carla

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adapters.control.simple_agent import SimpleAgent
from domain.entities import Observation, VehicleCommand
from infrastructure.carla.carla_env_adapter import CarlaEnvAdapter
from usecases.episode_info_parser import parse_step_info_payload
from usecases.episode_types import EpisodeSpec, TransformSpec


@dataclass(frozen=True)
class ObstacleSpec:
    blueprint_id: str
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


@dataclass(frozen=True)
class ScenarioSpec:
    map_name: str
    instruction: str
    start_transform: TransformSpec
    goal_transform: TransformSpec
    goal_radius_m: float
    max_steps_default: int
    simple_agent_throttle_default: float
    obstacles: list[ObstacleSpec]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CARLA smoke test for construction detour episode.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument(
        "--scenario",
        type=str,
        default=str(ROOT / "tests" / "fixtures" / "scenarios" / "construction_detour_001.json"),
    )
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--throttle", type=float, default=0.3)
    parser.add_argument(
        "--provoke",
        type=str,
        choices=("collision", "lane", "none"),
        default="collision",
    )
    parser.add_argument("--position-tolerance-m", type=float, default=0.8)
    parser.add_argument("--yaw-tolerance-deg", type=float, default=5.0)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.steps < 50 or args.steps > 200:
        print("[error] --steps must be between 50 and 200.")
        return 2

    scenario = load_scenario(pathlib.Path(args.scenario))

    spawned_obstacles: list[carla.Actor] = []
    env = CarlaEnvAdapter(
        host=args.host,
        port=args.port,
        timeout=5.0,
        fixed_dt=0.1,
        sensor_timeout=3.0,
        map_name=scenario.map_name,
        spectator_follow=True,
    )
    agent = SimpleAgent(throttle=args.throttle if args.throttle > 0 else scenario.simple_agent_throttle_default)
    spec = EpisodeSpec(
        instruction=scenario.instruction,
        start=scenario.start_transform,
        goal=scenario.goal_transform,
        max_steps=args.steps,
        goal_radius_m=scenario.goal_radius_m,
    )

    try:
        if args.provoke == "collision":
            spawned_obstacles = spawn_obstacles(
                host=args.host,
                port=args.port,
                map_name=scenario.map_name,
                obstacles=scenario.obstacles,
            )
            print(f"[info] spawned {len(spawned_obstacles)} obstacle(s) for collision provoke mode.")

        obs, reset_info = env.reset(spec)
        check_ok = check_reset_alignment(
            obs=obs,
            start=scenario.start_transform,
            pos_tolerance_m=args.position_tolerance_m,
            yaw_tolerance_deg=args.yaw_tolerance_deg,
        )
        if not check_ok:
            print("[error] reset pose is not close to expected start transform.")
            return 3

        initial_collision = reset_info.collision_count
        initial_lane = reset_info.lane_invasion_count
        initial_violation = reset_info.violation_count
        collision_changed = False
        lane_changed = False
        last_collision = initial_collision
        last_lane = initial_lane
        last_violation = initial_violation

        print(
            "[info] step telemetry header: "
            "step collision lane violation term speed_mps dist_to_goal_m"
        )
        for idx in range(args.steps):
            cmd = agent.act(obs)
            if args.provoke == "lane" and idx > args.steps // 2:
                steer = 0.55 if idx % 10 < 5 else -0.55
                cmd = VehicleCommand(throttle=cmd.throttle, steer=steer, brake=0.0)

            step_result = env.step(cmd)
            step_info = parse_step_info_payload(step_result.info, step_index=idx + 1)
            obs = step_result.obs

            collision_changed = collision_changed or (step_info.collision_count > initial_collision)
            lane_changed = lane_changed or (step_info.lane_invasion_count > initial_lane)
            last_collision = step_info.collision_count
            last_lane = step_info.lane_invasion_count
            last_violation = step_info.violation_count

            print(
                f"[step] {step_info.step_index:03d} "
                f"{step_info.collision_count} {step_info.lane_invasion_count} "
                f"{step_info.violation_count} {step_info.termination_reason.value} "
                f"{step_info.speed_mps:.3f} {step_info.distance_to_goal_m:.3f}"
            )

            if step_result.done:
                print("[info] episode terminated by env.")
                break

        print(
            "[summary] "
            f"collision_changed={collision_changed} lane_changed={lane_changed} "
            f"final_collision={last_collision} final_lane={last_lane} final_violation={last_violation}"
        )

        if args.provoke == "collision" and not collision_changed:
            print("[error] collision provoke requested but collision_count did not change.")
            return 4
        if args.provoke == "lane" and not lane_changed:
            print("[error] lane provoke requested but lane_invasion_count did not change.")
            return 5

        return 0
    finally:
        env.close()
        destroy_obstacles(spawned_obstacles)


def load_scenario(path: pathlib.Path) -> ScenarioSpec:
    raw = json.loads(path.read_text(encoding="utf-8"))
    obstacles = [
        ObstacleSpec(
            blueprint_id=str(item["blueprint_id"]),
            x=float(item["x"]),
            y=float(item["y"]),
            z=float(item["z"]),
            roll=float(item["roll"]),
            pitch=float(item["pitch"]),
            yaw=float(item["yaw"]),
        )
        for item in raw.get("obstacles", [])
    ]
    return ScenarioSpec(
        map_name=str(raw["map_name"]),
        instruction=str(raw["instruction"]),
        start_transform=_parse_transform(raw["start_transform"]),
        goal_transform=_parse_transform(raw["goal_transform"]),
        goal_radius_m=float(raw.get("goal_radius_m", 2.0)),
        max_steps_default=int(raw.get("max_steps_default", 120)),
        simple_agent_throttle_default=float(raw.get("simple_agent_throttle_default", 0.3)),
        obstacles=obstacles,
    )


def _parse_transform(raw: dict[str, object]) -> TransformSpec:
    return TransformSpec(
        x=float(raw["x"]),
        y=float(raw["y"]),
        z=float(raw["z"]),
        roll=float(raw["roll"]),
        pitch=float(raw["pitch"]),
        yaw=float(raw["yaw"]),
    )


def check_reset_alignment(
    *,
    obs: Observation,
    start: TransformSpec,
    pos_tolerance_m: float,
    yaw_tolerance_deg: float,
) -> bool:
    expected_x = start.x
    expected_y = -start.y
    expected_z = start.z
    actual_x = float(obs.ego.position[0])
    actual_y = float(obs.ego.position[1])
    actual_z = float(obs.ego.position[2])

    dx = actual_x - expected_x
    dy = actual_y - expected_y
    dz = actual_z - expected_z
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)

    expected_yaw = -start.yaw
    actual_yaw = float(obs.ego.rotation_rpy[2])
    yaw_error = abs(normalize_angle_deg(actual_yaw - expected_yaw))

    print(
        "[reset] "
        f"expected_rh=(x={expected_x:.3f}, y={expected_y:.3f}, z={expected_z:.3f}, yaw={expected_yaw:.3f}) "
        f"actual=(x={actual_x:.3f}, y={actual_y:.3f}, z={actual_z:.3f}, yaw={actual_yaw:.3f}) "
        f"distance={distance:.3f} yaw_error={yaw_error:.3f}"
    )
    return distance <= pos_tolerance_m and yaw_error <= yaw_tolerance_deg


def normalize_angle_deg(value: float) -> float:
    angle = value
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def spawn_obstacles(
    *,
    host: str,
    port: int,
    map_name: str,
    obstacles: list[ObstacleSpec],
) -> list[carla.Actor]:
    client = carla.Client(host, port)
    client.set_timeout(5.0)
    world = client.get_world()
    current_map = world.get_map().name.split("/")[-1]
    if current_map != map_name:
        client.load_world(map_name)
        world = client.get_world()

    blueprint_library = world.get_blueprint_library()
    spawned: list[carla.Actor] = []
    for item in obstacles:
        blueprint = blueprint_library.find(item.blueprint_id)
        transform = carla.Transform(
            carla.Location(x=item.x, y=item.y, z=item.z),
            carla.Rotation(roll=item.roll, pitch=item.pitch, yaw=item.yaw),
        )
        actor = world.try_spawn_actor(blueprint, transform)
        if actor is None:
            print(
                "[warn] failed to spawn obstacle "
                f"{item.blueprint_id} at ({item.x:.3f},{item.y:.3f},{item.z:.3f})."
            )
            continue
        spawned.append(actor)
    return spawned


def destroy_obstacles(actors: list[carla.Actor]) -> None:
    for actor in actors:
        try:
            actor.destroy()
        except RuntimeError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
