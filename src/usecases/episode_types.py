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


class WorkZoneSeverity(str, Enum):
    HARD = "hard"
    SOFT = "soft"


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
class WorldXYPointSpec:
    x: float
    y: float


@dataclass(frozen=True)
class WorkZoneSpec:
    id: str
    polygon_world_xy: tuple[WorldXYPointSpec, ...]
    severity: WorkZoneSeverity
    terminate_on_enter: bool = False
    cooldown_steps: int | None = None


@dataclass(frozen=True)
class WorkZoneThresholdBySeveritySpec:
    hard: int = 1
    soft: int = 999


@dataclass(frozen=True)
class ViolationThresholdsSpec:
    lane: int | None = None
    red_light: int | None = None
    workzone_by_severity: WorkZoneThresholdBySeveritySpec = field(
        default_factory=WorkZoneThresholdBySeveritySpec,
    )


@dataclass(frozen=True)
class EpisodeSpec:
    instruction: str
    start: StartGoalRef | None = None
    goal: StartGoalRef | None = None
    preferences: dict[str, float] = field(default_factory=dict)
    workzones: tuple[WorkZoneSpec, ...] = ()
    violation_thresholds: ViolationThresholdsSpec = field(default_factory=ViolationThresholdsSpec)
    workzone_default_cooldown_steps: int = 0
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
    workzone_violation_count: int = 0
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
    workzone_violation_count: int = 0
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
    workzone_violation_count: int
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
