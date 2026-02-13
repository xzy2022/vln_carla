from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from domain.entities import StepResult
from usecases.episode_info_parser import (
    choose_primary_termination_reason,
    normalize_termination_reasons,
    parse_step_info_payload,
)
from usecases.episode_types import (
    EpisodeMetrics,
    EpisodeResult,
    EpisodeSpec,
    ResetInfo,
    StepInfo,
    TerminationReason,
)
from usecases.ports.agent_interface import AgentInterface
from usecases.ports.env_interface import EnvInterface
from usecases.ports.logger_interface import LoggerInterface
from usecases.ports.run_episode_input_port import RunEpisodeInputPort

_EPSILON = 1e-6


class RunEpisodeUseCase(RunEpisodeInputPort):
    def __init__(
        self,
        env: EnvInterface,
        agent: AgentInterface,
        logger: LoggerInterface,
        should_stop: Callable[[], bool] | None = None,
        stuck_speed_threshold_mps: float = 0.2,
        stuck_steps_threshold: int = 20,
    ) -> None:
        self._env = env
        self._agent = agent
        self._logger = logger
        self._should_stop = should_stop
        self._stuck_speed_threshold_mps = stuck_speed_threshold_mps
        self._stuck_steps_threshold = stuck_steps_threshold

    def run(self, spec: EpisodeSpec) -> EpisodeResult:
        obs, reset_info = self._env.reset(spec)
        total_reward = 0.0
        steps = 0
        actual_path_length_m = 0.0
        previous_position = obs.ego.position
        low_speed_streak = 0
        usecase_stuck_count = reset_info.stuck_count
        stuck_event_recorded = False
        step_log: list[StepInfo] = []

        termination_reason = reset_info.termination_reason
        termination_reasons = normalize_termination_reasons(reset_info.termination_reasons)
        if termination_reason != TerminationReason.ONGOING and not termination_reasons:
            termination_reasons = (termination_reason,)

        try:
            while termination_reason == TerminationReason.ONGOING:
                if self._should_stop is not None and self._should_stop():
                    termination_reasons = _merge_reasons(
                        termination_reasons,
                        [TerminationReason.ERROR],
                    )
                    termination_reason = choose_primary_termination_reason(termination_reasons)
                    break

                if steps >= spec.max_steps:
                    termination_reasons = _merge_reasons(
                        termination_reasons,
                        [TerminationReason.TIMEOUT],
                    )
                    termination_reason = choose_primary_termination_reason(termination_reasons)
                    break

                cmd = self._agent.act(obs)
                result: StepResult = self._env.step(cmd)
                self._logger.save(result.obs)

                parsed_step_info = parse_step_info_payload(result.info, step_index=steps + 1)
                actual_path_length_m += _distance_between_positions(previous_position, result.obs.ego.position)
                previous_position = result.obs.ego.position

                if (
                    not parsed_step_info.reached_goal
                    and parsed_step_info.speed_mps < self._stuck_speed_threshold_mps
                ):
                    low_speed_streak += 1
                else:
                    low_speed_streak = 0
                    stuck_event_recorded = False

                step_reason_candidates: list[TerminationReason] = list(parsed_step_info.termination_reasons)
                if parsed_step_info.termination_reason != TerminationReason.ONGOING:
                    step_reason_candidates.append(parsed_step_info.termination_reason)
                if parsed_step_info.collision_count > 0:
                    step_reason_candidates.append(TerminationReason.COLLISION)
                if parsed_step_info.reached_goal:
                    step_reason_candidates.append(TerminationReason.SUCCESS)

                if low_speed_streak >= self._stuck_steps_threshold:
                    step_reason_candidates.append(TerminationReason.STUCK)
                    if not stuck_event_recorded:
                        usecase_stuck_count += 1
                        stuck_event_recorded = True

                step_reasons = normalize_termination_reasons(step_reason_candidates)
                step_reason = choose_primary_termination_reason(step_reasons)

                if result.done and step_reason == TerminationReason.ONGOING:
                    step_reasons = _merge_reasons(step_reasons, [TerminationReason.ERROR])
                    step_reason = choose_primary_termination_reason(step_reasons)

                parsed_step_info = dataclasses.replace(
                    parsed_step_info,
                    termination_reason=step_reason,
                    termination_reasons=step_reasons,
                    stuck_count=max(parsed_step_info.stuck_count, usecase_stuck_count),
                )
                step_log.append(parsed_step_info)

                obs = result.obs
                total_reward += result.reward
                steps += 1

                if step_reason != TerminationReason.ONGOING or result.done:
                    termination_reasons = _merge_reasons(termination_reasons, step_reasons)
                    termination_reason = choose_primary_termination_reason(termination_reasons)
                    break
        finally:
            self._logger.flush()

        latest_counts = _extract_latest_counts(reset_info, step_log, usecase_stuck_count)
        shortest_path_length_m = max(0.0, reset_info.shortest_path_length_m)
        sr = 1.0 if termination_reason == TerminationReason.SUCCESS else 0.0
        spl = sr * shortest_path_length_m / max(shortest_path_length_m, actual_path_length_m, _EPSILON)

        metrics = EpisodeMetrics(
            sr=sr,
            spl=spl,
            collision_count=latest_counts.collision_count,
            violation_count=latest_counts.violation_count,
            lane_invasion_count=latest_counts.lane_invasion_count,
            red_light_violation_count=latest_counts.red_light_violation_count,
            workzone_violation_count=latest_counts.workzone_violation_count,
            stuck_count=latest_counts.stuck_count,
            shortest_path_length_m=shortest_path_length_m,
            actual_path_length_m=actual_path_length_m,
            total_steps=steps,
        )
        if termination_reason != TerminationReason.ONGOING and not termination_reasons:
            termination_reasons = (termination_reason,)

        return EpisodeResult(
            spec=spec,
            metrics=metrics,
            termination_reason=termination_reason,
            termination_reasons=termination_reasons,
            reset_info=reset_info,
            step_log=step_log,
            total_reward=total_reward,
        )


def _distance_between_positions(
    previous: npt.NDArray[np.float32],
    current: npt.NDArray[np.float32],
) -> float:
    prev_x = float(previous[0])
    prev_y = float(previous[1])
    prev_z = float(previous[2])
    cur_x = float(current[0])
    cur_y = float(current[1])
    cur_z = float(current[2])
    return math.sqrt((cur_x - prev_x) ** 2 + (cur_y - prev_y) ** 2 + (cur_z - prev_z) ** 2)


def _merge_reasons(
    base: tuple[TerminationReason, ...],
    extra: list[TerminationReason] | tuple[TerminationReason, ...],
) -> tuple[TerminationReason, ...]:
    return normalize_termination_reasons([*base, *extra])


def _extract_latest_counts(
    reset_info: ResetInfo,
    step_log: list[StepInfo],
    usecase_stuck_count: int,
) -> StepInfo:
    if step_log:
        last_step = step_log[-1]
        return dataclasses.replace(
            last_step,
            violation_count=(
                last_step.lane_invasion_count
                + last_step.red_light_violation_count
                + last_step.workzone_violation_count
            ),
            stuck_count=max(last_step.stuck_count, usecase_stuck_count),
        )

    # Keep summary math simple for zero-step episodes.
    return StepInfo(
        step_index=0,
        termination_reason=reset_info.termination_reason,
        termination_reasons=reset_info.termination_reasons,
        collision_count=reset_info.collision_count,
        lane_invasion_count=reset_info.lane_invasion_count,
        red_light_violation_count=reset_info.red_light_violation_count,
        workzone_violation_count=reset_info.workzone_violation_count,
        violation_count=(
            reset_info.lane_invasion_count
            + reset_info.red_light_violation_count
            + reset_info.workzone_violation_count
        ),
        stuck_count=max(reset_info.stuck_count, usecase_stuck_count),
        reached_goal=False,
        speed_mps=0.0,
        distance_to_goal_m=float("inf"),
    )
