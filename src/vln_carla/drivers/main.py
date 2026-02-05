from __future__ import annotations

import argparse
import os
import subprocess
import time

from vln_carla.adapters.carla_env_adapter import CarlaEnvAdapter
from vln_carla.domain.errors import EnvConnectionError, EnvStepError
from vln_carla.drivers.in_memory_logger import InMemoryLogger
from vln_carla.drivers.simple_agent import SimpleAgent
from vln_carla.usecases.run_episode import RunEpisodeUseCase


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 0 CARLA runner")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--fixed-dt", type=float, default=0.1)
    parser.add_argument("--sensor-timeout", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--throttle", type=float, default=0.3)
    parser.add_argument("--carla-path", type=str, default=r'D:\Workspace\02_Playground\CARLA_Latest\CarlaUE4.exe', help="Path to CarlaUE4.exe")
    parser.add_argument("--quality-level", type=str, choices=["Epic", "Low"], default=None, help="Server render quality (only if we launch CARLA)")
    parser.add_argument("--map", type=str, default="Town04", help="Map to load")
    parser.add_argument("--spectator-follow", action="store_true", help="Follow ego with spectator")
    parser.add_argument("--no-rendering", action="store_true", help="Disable rendering (GPU sensors return empty data)")
    parser.add_argument("--camera-width", type=int, default=800)
    parser.add_argument("--camera-height", type=int, default=600)
    parser.add_argument("--camera-sensor-tick", type=float, default=None, help="Seconds between camera captures (e.g. 0.2 for 5Hz)")
    parser.add_argument("--unload-map-layers", type=str, default="", help="Comma-separated map layers to unload (e.g. Buildings,Vegetation,ParkedVehicles)")
    return parser


def _ensure_carla_server(carla_exe: str | None, port: int, quality_level: str | None) -> subprocess.Popen | None:
    if not carla_exe:
        return None

    if not os.path.isfile(carla_exe):
        print(f"[warn] CARLA executable not found: {carla_exe}")
        return None

    print(f"Starting CARLA server on port {port}...")
    args = [carla_exe, f"-carla-rpc-port={port}"]
    if quality_level:
        args.append(f"-quality-level={quality_level}")
    return subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def main() -> int:
    args = build_arg_parser().parse_args()

    server_process = _ensure_carla_server(args.carla_path, args.port, args.quality_level)
    if server_process:
        print("Waiting for CARLA server to initialize...")
        time.sleep(10)

    if args.no_rendering:
        print("[warn] --no-rendering will disable GPU sensors (camera frames will be empty).")

    unload_layers = tuple(filter(None, (s.strip() for s in args.unload_map_layers.split(","))))
    env = CarlaEnvAdapter(
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        fixed_dt=args.fixed_dt,
        sensor_timeout=args.sensor_timeout,
        map_name=args.map,
        spectator_follow=args.spectator_follow,
        no_rendering_mode=args.no_rendering,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_sensor_tick=args.camera_sensor_tick,
        unload_map_layers=unload_layers,
    )
    agent = SimpleAgent(throttle=args.throttle)
    logger = InMemoryLogger()
    usecase = RunEpisodeUseCase(env=env, agent=agent, logger=logger, max_steps=args.max_steps)

    try:
        summary = usecase.run()
        print(f"Episode finished: steps={summary['total_steps']} reward={summary['total_reward']:.3f}")
    except (EnvConnectionError, EnvStepError) as exc:
        print(f"[error] {exc}")
        return 1
    finally:
        env.close()
        if server_process:
            server_process.terminate()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
