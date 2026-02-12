from __future__ import annotations

import json
import os
import pathlib

import pytest

from adapters.control.simple_agent import SimpleAgent
from infrastructure.logging.in_memory_logger import InMemoryLogger
from usecases.episode_types import EpisodeSpec, TerminationReason, TransformSpec
from usecases.run_episode import RunEpisodeUseCase

SCENARIO_PATH = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "scenarios" / "construction_detour_001.json"


@pytest.mark.integration
def test_construction_detour_short_episode_returns_termination_reason() -> None:
    host, port = _get_server_address_or_skip()
    pytest.importorskip("carla")

    # Import here so missing `carla` does not fail collection.
    from infrastructure.carla.carla_env_adapter import CarlaEnvAdapter

    _ensure_server_reachable_or_skip(host, port)
    scenario = _load_scenario()
    max_steps = min(int(scenario.get("max_steps_default", 120)), 100)

    env = CarlaEnvAdapter(
        host=host,
        port=port,
        timeout=5.0,
        fixed_dt=0.1,
        sensor_timeout=3.0,
        map_name=str(scenario["map_name"]),
        spectator_follow=False,
    )
    usecase = RunEpisodeUseCase(
        env=env,
        agent=SimpleAgent(throttle=float(scenario.get("simple_agent_throttle_default", 0.3))),
        logger=InMemoryLogger(),
    )

    spec = EpisodeSpec(
        instruction=str(scenario["instruction"]),
        start=_parse_transform(scenario["start_transform"]),
        goal=_parse_transform(scenario["goal_transform"]),
        goal_radius_m=float(scenario.get("goal_radius_m", 2.0)),
        max_steps=max_steps,
    )

    try:
        result = usecase.run(spec)
    finally:
        env.close()

    assert result.termination_reason != TerminationReason.ONGOING
    assert result.reset_info.termination_reason == TerminationReason.ONGOING
    assert result.metrics.total_steps > 0
    assert len(result.step_log) > 0


def _load_scenario() -> dict[str, object]:
    return json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))


def _parse_transform(raw: object) -> TransformSpec:
    data = raw
    if not isinstance(data, dict):
        raise TypeError("invalid transform payload")
    return TransformSpec(
        x=float(data["x"]),
        y=float(data["y"]),
        z=float(data["z"]),
        roll=float(data["roll"]),
        pitch=float(data["pitch"]),
        yaw=float(data["yaw"]),
    )


def _get_server_address_or_skip() -> tuple[str, int]:
    host = os.getenv("CARLA_SERVER_HOST")
    port_raw = os.getenv("CARLA_SERVER_PORT")
    if not host or not port_raw:
        pytest.skip("integration test skipped: CARLA_SERVER_HOST/CARLA_SERVER_PORT is not set")
    try:
        port = int(port_raw)
    except ValueError:
        pytest.skip("integration test skipped: CARLA_SERVER_PORT is not an integer")
    return host, port


def _ensure_server_reachable_or_skip(host: str, port: int) -> None:
    import carla

    try:
        client = carla.Client(host, port)
        client.set_timeout(3.0)
        client.get_world()
    except Exception as exc:
        pytest.skip(f"integration test skipped: unable to connect CARLA server ({exc})")
