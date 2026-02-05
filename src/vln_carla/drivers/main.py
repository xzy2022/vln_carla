from __future__ import annotations

import argparse

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
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    env = CarlaEnvAdapter(
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        fixed_dt=args.fixed_dt,
        sensor_timeout=args.sensor_timeout,
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
