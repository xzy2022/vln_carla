from vln_carla.domain.entities import Observation, StepResult, VehicleCommand, VehicleState
from vln_carla.usecases.run_episode import RunEpisodeUseCase


class FakeEnv:
    def __init__(self, max_steps: int) -> None:
        self._max_steps = max_steps
        self._steps = 0

    def reset(self) -> Observation:
        return _make_obs()

    def step(self, cmd: VehicleCommand) -> StepResult:
        self._steps += 1
        done = self._steps >= self._max_steps
        return StepResult(obs=_make_obs(), reward=1.0, done=done, info={})

    def close(self) -> None:
        return None


class FakeAgent:
    def act(self, obs: Observation) -> VehicleCommand:
        return VehicleCommand(throttle=0.1, steer=0.0, brake=0.0)


class FakeLogger:
    def __init__(self) -> None:
        self.items = []

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


def _vec(value: float):
    import numpy as np

    return np.array([value, value, value], dtype=np.float32)


def _rgb():
    import numpy as np

    return np.zeros((2, 2, 3), dtype=np.uint8)


def test_run_episode_stops_on_done():
    env = FakeEnv(max_steps=3)
    agent = FakeAgent()
    logger = FakeLogger()
    usecase = RunEpisodeUseCase(env=env, agent=agent, logger=logger)

    summary = usecase.run()

    assert summary["total_steps"] == 3
    assert summary["total_reward"] == 3.0
    assert len(logger.items) == 3
