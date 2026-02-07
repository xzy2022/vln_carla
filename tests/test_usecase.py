from __future__ import annotations

import numpy as np
import numpy.typing as npt

from domain.entities import Observation, StepResult, VehicleCommand, VehicleState
from usecases.run_episode import RunEpisodeUseCase


class FakeEnv:
    def __init__(self, max_steps: int) -> None:
        self._max_steps = max_steps
        self._steps = 0

    def reset(self) -> Observation:
        return _make_obs()

    def step(self, cmd: VehicleCommand) -> StepResult:
        del cmd
        self._steps += 1
        done = self._steps >= self._max_steps
        return StepResult(obs=_make_obs(), reward=1.0, done=done, info={})

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


def _make_obs() -> Observation:
    state = VehicleState(
        position=_vec(0.0),
        rotation_rpy=_vec(0.0),
        velocity=_vec(0.0),
    )
    return Observation(rgb=_rgb(), ego=state, frame=0, timestamp=0.0)


def _vec(value: float) -> npt.NDArray[np.float32]:
    return np.array([value, value, value], dtype=np.float32)


def _rgb() -> npt.NDArray[np.uint8]:
    return np.zeros((2, 2, 3), dtype=np.uint8)


def test_run_episode_stops_on_done() -> None:
    env = FakeEnv(max_steps=3)
    agent = FakeAgent()
    logger = FakeLogger()
    usecase = RunEpisodeUseCase(env=env, agent=agent, logger=logger)

    summary = usecase.run()

    assert summary.total_steps == 3
    assert summary.total_reward == 3.0
    assert len(logger.items) == 3


def test_run_episode_stops_when_should_stop() -> None:
    env = FakeEnv(max_steps=10)
    agent = FakeAgent()
    logger = FakeLogger()
    calls = 0

    def should_stop() -> bool:
        nonlocal calls
        calls += 1
        return calls > 2

    usecase = RunEpisodeUseCase(env=env, agent=agent, logger=logger, should_stop=should_stop)
    summary = usecase.run()

    assert summary.total_steps == 2
    assert summary.total_reward == 2.0
    assert len(logger.items) == 2
