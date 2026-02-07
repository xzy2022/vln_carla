from __future__ import annotations

from collections.abc import Callable

from domain.entities import StepResult
from usecases.dtos import EpisodeSummary
from usecases.ports.agent_interface import AgentInterface
from usecases.ports.env_interface import EnvInterface
from usecases.ports.logger_interface import LoggerInterface
from usecases.ports.run_episode_input_port import RunEpisodeInputPort


class RunEpisodeUseCase(RunEpisodeInputPort):
    def __init__(
        self,
        env: EnvInterface,
        agent: AgentInterface,
        logger: LoggerInterface,
        max_steps: int | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        self._env = env
        self._agent = agent
        self._logger = logger
        self._max_steps = max_steps
        self._should_stop = should_stop

    def run(self) -> EpisodeSummary:
        obs = self._env.reset()
        total_reward = 0.0
        steps = 0

        while True:
            if self._should_stop is not None and self._should_stop():
                break

            if self._max_steps is not None and steps >= self._max_steps:
                break

            cmd = self._agent.act(obs)
            result: StepResult = self._env.step(cmd)
            self._logger.save(result.obs)

            obs = result.obs
            total_reward += result.reward
            steps += 1

            if result.done:
                break

        self._logger.flush()
        return EpisodeSummary(total_steps=steps, total_reward=total_reward)
