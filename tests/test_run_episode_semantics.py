from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from domain.entities import Observation, StepResult, VehicleCommand, VehicleState
from usecases.episode_types import EpisodeResult, EpisodeSpec, ResetInfo, TerminationReason
from usecases.run_episode import RunEpisodeUseCase


@dataclass
class ScriptedStep:
    x: float
    info: dict[str, object]
    reward: float = 1.0
    done: bool = False


class ScriptedFakeEnv:
    def __init__(self, *, shortest_path_length_m: float, scripted_steps: list[ScriptedStep]) -> None:
        self._shortest_path_length_m = shortest_path_length_m
        self._steps = scripted_steps
        self._cursor = 0

    def reset(self, spec: EpisodeSpec) -> tuple[Observation, ResetInfo]:
        del spec
        self._cursor = 0
        return _make_obs(0.0), ResetInfo(shortest_path_length_m=self._shortest_path_length_m)

    def step(self, cmd: VehicleCommand) -> StepResult:
        del cmd
        scripted = self._steps[self._cursor]
        self._cursor += 1
        return StepResult(
            obs=_make_obs(scripted.x),
            reward=scripted.reward,
            done=scripted.done,
            info=scripted.info,
        )

    def close(self) -> None:
        return None


class FakeAgent:
    def act(self, obs: Observation) -> VehicleCommand:
        del obs
        return VehicleCommand(throttle=0.2, steer=0.0, brake=0.0)


class FakeLogger:
    def __init__(self) -> None:
        self.items: list[Observation] = []

    def save(self, obs: Observation) -> None:
        self.items.append(obs)

    def flush(self) -> None:
        return None


def test_success_termination_and_sr() -> None:
    env = ScriptedFakeEnv(
        shortest_path_length_m=5.0,
        scripted_steps=[
            _step(1, x=2.0),
            _step(2, x=5.0, reached_goal=True, done=True),
        ],
    )
    result = _run(env, max_steps=5)

    assert result.termination_reason == TerminationReason.SUCCESS
    assert result.metrics.sr == 1.0


def test_timeout_when_max_steps_reached_without_goal() -> None:
    env = ScriptedFakeEnv(
        shortest_path_length_m=5.0,
        scripted_steps=[_step(i + 1, x=float(i + 1)) for i in range(10)],
    )
    result = _run(env, max_steps=3)

    assert result.termination_reason == TerminationReason.TIMEOUT
    assert result.metrics.sr == 0.0
    assert result.metrics.total_steps == 3


def test_collision_termination_increments_counter() -> None:
    env = ScriptedFakeEnv(
        shortest_path_length_m=10.0,
        scripted_steps=[
            _step(1, x=1.0),
            _step(2, x=2.0, collision_count=1, done=True),
        ],
    )
    result = _run(env, max_steps=10)

    assert result.termination_reason == TerminationReason.COLLISION
    assert result.metrics.collision_count == 1


def test_spl_formula_uses_shortest_and_actual_path_length() -> None:
    env = ScriptedFakeEnv(
        shortest_path_length_m=10.0,
        scripted_steps=[
            _step(1, x=4.0),
            _step(2, x=8.0),
            _step(3, x=12.0, reached_goal=True, done=True),
        ],
    )
    result = _run(env, max_steps=10)

    assert result.termination_reason == TerminationReason.SUCCESS
    assert result.metrics.actual_path_length_m == 12.0
    assert result.metrics.spl == 10.0 / 12.0


def test_stepinfo_accumulates_collision_lane_and_red_light() -> None:
    env = ScriptedFakeEnv(
        shortest_path_length_m=10.0,
        scripted_steps=[
            _step(1, x=1.0, lane_invasion_count=1, red_light_violation_count=0),
            _step(2, x=2.0, lane_invasion_count=1, red_light_violation_count=1),
            _step(
                3,
                x=3.0,
                collision_count=1,
                lane_invasion_count=2,
                red_light_violation_count=1,
                done=True,
            ),
        ],
    )
    result = _run(env, max_steps=10)

    assert result.step_log[0].lane_invasion_count == 1
    assert result.step_log[1].red_light_violation_count == 1
    assert result.step_log[2].collision_count == 1
    assert result.step_log[2].violation_count == 3


def test_primary_reason_prefers_collision_over_success() -> None:
    env = ScriptedFakeEnv(
        shortest_path_length_m=1.0,
        scripted_steps=[
            _step(
                1,
                x=1.0,
                reached_goal=True,
                collision_count=1,
                done=True,
                termination_reasons=["SUCCESS"],
            ),
        ],
    )
    result = _run(env, max_steps=10)

    assert result.termination_reason == TerminationReason.COLLISION
    assert TerminationReason.COLLISION in result.termination_reasons
    assert TerminationReason.SUCCESS in result.termination_reasons


def test_violation_formula_excludes_collision_count() -> None:
    env = ScriptedFakeEnv(
        shortest_path_length_m=5.0,
        scripted_steps=[
            _step(
                1,
                x=1.0,
                collision_count=5,
                lane_invasion_count=2,
                red_light_violation_count=1,
                violation_count=999,
                done=True,
            ),
        ],
    )
    result = _run(env, max_steps=5)

    assert result.step_log[-1].violation_count == 3
    assert result.metrics.violation_count == 3
    assert result.metrics.collision_count == 5


def _run(env: ScriptedFakeEnv, *, max_steps: int) -> EpisodeResult:
    usecase = RunEpisodeUseCase(
        env=env,
        agent=FakeAgent(),
        logger=FakeLogger(),
    )
    spec = EpisodeSpec(instruction="test", max_steps=max_steps)
    return usecase.run(spec)


def _step(
    step_index: int,
    *,
    x: float,
    done: bool = False,
    termination_reason: str = "ONGOING",
    termination_reasons: list[str] | None = None,
    collision_count: int = 0,
    lane_invasion_count: int = 0,
    red_light_violation_count: int = 0,
    violation_count: int = 0,
    stuck_count: int = 0,
    reached_goal: bool = False,
    speed_mps: float = 3.0,
    distance_to_goal_m: float = 10.0,
) -> ScriptedStep:
    reasons = termination_reasons if termination_reasons is not None else []
    return ScriptedStep(
        x=x,
        done=done,
        info={
            "step_index": step_index,
            "termination_reason": termination_reason,
            "termination_reasons": reasons,
            "collision_count": collision_count,
            "lane_invasion_count": lane_invasion_count,
            "red_light_violation_count": red_light_violation_count,
            "violation_count": violation_count,
            "stuck_count": stuck_count,
            "reached_goal": reached_goal,
            "speed_mps": speed_mps,
            "distance_to_goal_m": distance_to_goal_m,
        },
    )


def _make_obs(x: float) -> Observation:
    state = VehicleState(
        position=np.array([x, 0.0, 0.0], dtype=np.float32),
        rotation_rpy=_vec(0.0),
        velocity=_vec(0.0),
    )
    return Observation(rgb=_rgb(), ego=state, frame=0, timestamp=0.0)


def _vec(value: float) -> npt.NDArray[np.float32]:
    return np.array([value, value, value], dtype=np.float32)


def _rgb() -> npt.NDArray[np.uint8]:
    return np.zeros((2, 2, 3), dtype=np.uint8)
