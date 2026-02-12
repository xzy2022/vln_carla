from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TerminationReason(str, Enum):
    ONGOING = "ONGOING"
    SUCCESS = "SUCCESS"
    TIMEOUT = "TIMEOUT"
    COLLISION = "COLLISION"
    VIOLATION = "VIOLATION"
    STUCK = "STUCK"
    ERROR = "ERROR"


@dataclass(frozen=True)
class TransformSpec:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


StartGoalRef = int | TransformSpec


@dataclass(frozen=True)
class EpisodeSpec:
    instruction: str
    start: StartGoalRef | None = None
    goal: StartGoalRef | None = None
    preferences: dict[str, float] = field(default_factory=dict)
    max_steps: int = 200
    goal_radius_m: float = 2.0


@dataclass(frozen=True)
class ResetInfo:
    termination_reason: TerminationReason = TerminationReason.ONGOING
    termination_reasons: tuple[TerminationReason, ...] = ()
    shortest_path_length_m: float = 0.0
    collision_count: int = 0
    lane_invasion_count: int = 0
    red_light_violation_count: int = 0
    violation_count: int = 0
    stuck_count: int = 0


@dataclass(frozen=True)
class StepInfo:
    step_index: int
    termination_reason: TerminationReason = TerminationReason.ONGOING
    termination_reasons: tuple[TerminationReason, ...] = ()
    collision_count: int = 0
    lane_invasion_count: int = 0
    red_light_violation_count: int = 0
    violation_count: int = 0
    stuck_count: int = 0
    reached_goal: bool = False
    speed_mps: float = 0.0
    distance_to_goal_m: float = float("inf")


@dataclass(frozen=True)
class EpisodeMetrics:
    sr: float
    spl: float
    collision_count: int
    violation_count: int
    lane_invasion_count: int
    red_light_violation_count: int
    stuck_count: int
    shortest_path_length_m: float
    actual_path_length_m: float
    total_steps: int


@dataclass(frozen=True)
class EpisodeResult:
    spec: EpisodeSpec
    metrics: EpisodeMetrics
    termination_reason: TerminationReason
    termination_reasons: tuple[TerminationReason, ...]
    reset_info: ResetInfo
    step_log: list[StepInfo]
    total_reward: float
