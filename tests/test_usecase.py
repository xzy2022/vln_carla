from __future__ import annotations

import numpy as np
import numpy.typing as npt

from domain.entities import Observation, StepResult, VehicleCommand, VehicleState
from usecases.episode_types import EpisodeSpec, ResetInfo
from usecases.run_episode import RunEpisodeUseCase


class FakeEnv:
    def __init__(self, success_at_step: int) -> None:
        self._success_at_step = success_at_step
        self._steps = 0

    def reset(self, spec: EpisodeSpec) -> tuple[Observation, ResetInfo]:
        del spec
        self._steps = 0
        return _make_obs(0.0), ResetInfo(shortest_path_length_m=3.0)

    def step(self, cmd: VehicleCommand) -> StepResult:
        del cmd
        self._steps += 1
        done = self._steps >= self._success_at_step
        return StepResult(
            obs=_make_obs(float(self._steps)),
            reward=1.0,
            done=done,
            info={
                "step_index": self._steps,
                "termination_reason": "SUCCESS" if done else "ONGOING",
                "termination_reasons": ["SUCCESS"] if done else [],
                "collision_count": 0,
                "lane_invasion_count": 0,
                "red_light_violation_count": 0,
                "workzone_violation_count": 0,
                "violation_count": 0,
                "stuck_count": 0,
                "reached_goal": done,
                "speed_mps": 2.0,
                "distance_to_goal_m": 0.0 if done else 10.0,
            },
        )

    def close(self) -> None:
        return None


class FakeAgent:
    def act(self, obs: Observation) -> VehicleCommand:
        del obs
        return VehicleCommand(throttle=0.1, steer=0.0, brake=0.0)


class FakeLogger:
    def __init__(self) -> None:
        self.items: list[Observation] = []

    def save(self, obs: Observation) -> None:
        self.items.append(obs)

    def flush(self) -> None:
        return None


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


def test_run_episode_stops_on_done() -> None:
    env = FakeEnv(success_at_step=3)
    agent = FakeAgent()
    logger = FakeLogger()
    usecase = RunEpisodeUseCase(env=env, agent=agent, logger=logger)
    spec = EpisodeSpec(instruction="go", max_steps=10)

    result = usecase.run(spec)

    assert result.metrics.total_steps == 3
    assert result.total_reward == 3.0
    assert result.termination_reason.value == "SUCCESS"
    assert len(logger.items) == 3


def test_run_episode_stops_when_should_stop() -> None:
    env = FakeEnv(success_at_step=10)
    agent = FakeAgent()
    logger = FakeLogger()
    calls = 0

    def should_stop() -> bool:
        nonlocal calls
        calls += 1
        return calls > 2

    usecase = RunEpisodeUseCase(env=env, agent=agent, logger=logger, should_stop=should_stop)
    spec = EpisodeSpec(instruction="go", max_steps=10)
    result = usecase.run(spec)

    assert result.metrics.total_steps == 2
    assert result.total_reward == 2.0
    assert result.termination_reason.value == "ERROR"
    assert len(logger.items) == 2
