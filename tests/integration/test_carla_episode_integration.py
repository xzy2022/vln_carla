from __future__ import annotations

import dataclasses
import json
import os
import pathlib
from typing import Any

import pytest

from domain.entities import Observation
from infrastructure.agents.simple_agent import SimpleAgent
from infrastructure.logging.in_memory_logger import InMemoryLogger
from usecases.episode_types import (
    EpisodeSpec,
    TerminationReason,
    TransformSpec,
    ViolationThresholdsSpec,
    WorkZoneSeverity,
    WorkZoneSpec,
    WorkZoneThresholdBySeveritySpec,
    WorldXYPointSpec,
)
from usecases.run_episode import RunEpisodeUseCase

SCENARIO_PATH = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "scenarios" / "construction_detour_001.json"


@pytest.mark.integration
def test_construction_detour_short_episode_returns_termination_reason() -> None:
    host, port = _get_server_address_or_skip()
    carla = pytest.importorskip("carla")

    # Import here so missing `carla` does not fail collection.
    from infrastructure.carla.carla_env_adapter import CarlaEnvAdapter

    client = _build_client_or_skip(carla_module=carla, host=host, port=port)
    scenario = _load_scenario()
    max_steps = min(int(scenario.get("max_steps_default", 120)), 100)
    start_transform = _parse_transform(scenario["start_transform"])
    _clear_spawn_blockers(
        carla_module=carla,
        client=client,
        start=start_transform,
        radius_m=3.0,
    )

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
        logger=TopDownFollowLogger(carla_module=carla, client=client, spectator_z=20.0, spectator_yaw=0.0),
    )

    spec = EpisodeSpec(
        instruction=str(scenario["instruction"]),
        start=start_transform,
        goal=_parse_transform(scenario["goal_transform"]),
        goal_radius_m=float(scenario.get("goal_radius_m", 2.0)),
        workzones=_parse_workzones(scenario),
        violation_thresholds=_parse_violation_thresholds(scenario),
        workzone_default_cooldown_steps=int(scenario.get("workzone_default_cooldown_steps", 0)),
        max_steps=max_steps,
    )

    result = _run_with_spawn_retry(
        usecase=usecase,
        env=env,
        spec=spec,
        retries=3,
        carla_module=carla,
        client=client,
    )

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


def _parse_workzones(scenario: dict[str, object]) -> tuple[WorkZoneSpec, ...]:
    raw = scenario.get("workzones", [])
    if not isinstance(raw, list):
        raise TypeError("workzones must be a list")

    workzones: list[WorkZoneSpec] = []
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError("invalid workzone payload")
        polygon_raw = item.get("polygon_world_xy", [])
        if not isinstance(polygon_raw, list):
            raise TypeError("polygon_world_xy must be a list")
        polygon_points: list[WorldXYPointSpec] = []
        for point in polygon_raw:
            if not isinstance(point, dict):
                raise TypeError("invalid polygon point payload")
            polygon_points.append(
                WorldXYPointSpec(x=float(point["x"]), y=float(point["y"])),
            )
        polygon = tuple(polygon_points)
        workzones.append(
            WorkZoneSpec(
                id=str(item["id"]),
                polygon_world_xy=polygon,
                severity=_parse_workzone_severity(item.get("severity", "hard")),
                terminate_on_enter=bool(item.get("terminate_on_enter", False)),
                cooldown_steps=_parse_optional_int(item.get("cooldown_steps")),
            ),
        )
    return tuple(workzones)


def _parse_violation_thresholds(scenario: dict[str, object]) -> ViolationThresholdsSpec:
    raw = scenario.get("violation_thresholds", {})
    if not isinstance(raw, dict):
        raise TypeError("violation_thresholds must be a dict")
    raw_workzone = raw.get("workzone_by_severity", {})
    if not isinstance(raw_workzone, dict):
        raise TypeError("workzone_by_severity must be a dict")
    return ViolationThresholdsSpec(
        lane=_parse_optional_int(raw.get("lane")),
        red_light=_parse_optional_int(raw.get("red_light")),
        workzone_by_severity=WorkZoneThresholdBySeveritySpec(
            hard=int(raw_workzone.get("hard", 1)),
            soft=int(raw_workzone.get("soft", 999)),
        ),
    )


def _parse_optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _parse_workzone_severity(value: object) -> WorkZoneSeverity:
    if isinstance(value, WorkZoneSeverity):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == WorkZoneSeverity.HARD.value:
            return WorkZoneSeverity.HARD
        if normalized == WorkZoneSeverity.SOFT.value:
            return WorkZoneSeverity.SOFT
    raise TypeError(f"invalid workzone severity: {value!r}")


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


def _build_client_or_skip(*, carla_module: Any, host: str, port: int) -> Any:
    try:
        client = carla_module.Client(host, port)
        client.set_timeout(3.0)
        client.get_world()
        return client
    except Exception as exc:
        pytest.skip(f"integration test skipped: unable to connect CARLA server ({exc})")


def _run_with_spawn_retry(
    *,
    usecase: RunEpisodeUseCase,
    env: Any,
    spec: EpisodeSpec,
    retries: int,
    carla_module: Any,
    client: Any,
) -> Any:
    attempts = max(1, retries + 1)
    last_error: Exception | None = None
    for attempt_index in range(attempts):
        attempt_spec = _spec_with_spawn_z_offset(spec, z_offset=0.2 * attempt_index)
        if isinstance(attempt_spec.start, TransformSpec):
            _clear_spawn_blockers(
                carla_module=carla_module,
                client=client,
                start=attempt_spec.start,
                radius_m=3.0,
            )
        try:
            return usecase.run(attempt_spec)
        except RuntimeError as exc:
            last_error = exc
            # CARLA occasionally keeps blocking actor at exact spawn point
            # for a few ticks after a previous run.
            if "Spawn failed because of collision at spawn position" not in str(exc):
                raise
        finally:
            env.close()
    assert last_error is not None
    raise last_error


def _spec_with_spawn_z_offset(spec: EpisodeSpec, *, z_offset: float) -> EpisodeSpec:
    start = spec.start
    if not isinstance(start, TransformSpec) or z_offset <= 0.0:
        return spec
    return dataclasses.replace(
        spec,
        start=TransformSpec(
            x=start.x,
            y=start.y,
            z=start.z + z_offset,
            roll=start.roll,
            pitch=start.pitch,
            yaw=start.yaw,
        ),
    )


def _clear_spawn_blockers(
    *,
    carla_module: Any,
    client: Any,
    start: TransformSpec,
    radius_m: float,
) -> None:
    world = client.get_world()
    start_loc = carla_module.Location(x=start.x, y=start.y, z=start.z)
    actors = world.get_actors()
    for actor in actors:
        type_id = getattr(actor, "type_id", "")
        if not (
            type_id.startswith("vehicle.")
            or type_id.startswith("walker.")
            or type_id.startswith("static.prop.")
        ):
            continue
        try:
            loc = actor.get_transform().location
        except RuntimeError:
            continue
        dx = float(loc.x - start_loc.x)
        dy = float(loc.y - start_loc.y)
        dz = float(loc.z - start_loc.z)
        if (dx * dx + dy * dy + dz * dz) > radius_m * radius_m:
            continue
        try:
            actor.destroy()
        except RuntimeError:
            continue


class TopDownFollowLogger:
    def __init__(
        self,
        *,
        carla_module: Any,
        client: Any,
        spectator_z: float,
        spectator_yaw: float,
    ) -> None:
        self._carla: Any = carla_module
        self._client: Any = client
        self._spectator_z = spectator_z
        self._spectator_yaw = spectator_yaw
        self._delegate = InMemoryLogger()

    def save(self, obs: Observation) -> None:
        self._delegate.save(obs)
        x_lh = float(obs.ego.position[0])
        y_lh = -float(obs.ego.position[1])

        world = self._client.get_world()
        spectator = world.get_spectator()
        location = self._carla.Location(x=x_lh, y=y_lh, z=self._spectator_z)
        rotation = self._carla.Rotation(pitch=-90.0, yaw=self._spectator_yaw, roll=0.0)
        spectator.set_transform(self._carla.Transform(location, rotation))

    def flush(self) -> None:
        self._delegate.flush()
