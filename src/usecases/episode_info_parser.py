from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from usecases.episode_types import StepInfo, TerminationReason

_TERMINATION_PRIORITY: dict[TerminationReason, int] = {
    TerminationReason.ERROR: 0,
    TerminationReason.COLLISION: 1,
    TerminationReason.VIOLATION: 2,
    TerminationReason.STUCK: 3,
    TerminationReason.TIMEOUT: 4,
    TerminationReason.SUCCESS: 5,
    TerminationReason.ONGOING: 99,
}


def choose_primary_termination_reason(
    reasons: Iterable[TerminationReason],
) -> TerminationReason:
    reason_list = list(reasons)
    if not reason_list:
        return TerminationReason.ONGOING
    return min(reason_list, key=lambda reason: _TERMINATION_PRIORITY[reason])


def normalize_termination_reasons(
    reasons: Iterable[TerminationReason],
) -> tuple[TerminationReason, ...]:
    seen: set[TerminationReason] = set()
    deduped: list[TerminationReason] = []
    for reason in reasons:
        if reason in seen:
            continue
        seen.add(reason)
        deduped.append(reason)
    if not deduped:
        return ()
    return tuple(sorted(deduped, key=lambda reason: _TERMINATION_PRIORITY[reason]))


def parse_step_info_payload(
    payload: Mapping[str, object],
    *,
    step_index: int,
) -> StepInfo:
    collision_count = max(0, _as_int(payload.get("collision_count"), default=0))
    lane_invasion_count = max(0, _as_int(payload.get("lane_invasion_count"), default=0))
    red_light_violation_count = max(
        0,
        _as_int(payload.get("red_light_violation_count"), default=0),
    )
    workzone_violation_count = max(
        0,
        _as_int(payload.get("workzone_violation_count"), default=0),
    )
    violation_count = lane_invasion_count + red_light_violation_count + workzone_violation_count

    termination_reason = parse_termination_reason(payload.get("termination_reason"))
    termination_reasons = list(parse_termination_reasons(payload.get("termination_reasons")))
    if termination_reason != TerminationReason.ONGOING:
        termination_reasons.append(termination_reason)

    normalized_reasons = normalize_termination_reasons(termination_reasons)
    if termination_reason == TerminationReason.ONGOING and normalized_reasons:
        termination_reason = choose_primary_termination_reason(normalized_reasons)
    if termination_reason != TerminationReason.ONGOING and not normalized_reasons:
        normalized_reasons = (termination_reason,)

    return StepInfo(
        step_index=step_index,
        termination_reason=termination_reason,
        termination_reasons=normalized_reasons,
        collision_count=collision_count,
        lane_invasion_count=lane_invasion_count,
        red_light_violation_count=red_light_violation_count,
        workzone_violation_count=workzone_violation_count,
        violation_count=violation_count,
        stuck_count=max(0, _as_int(payload.get("stuck_count"), default=0)),
        reached_goal=_as_bool(payload.get("reached_goal"), default=False),
        speed_mps=_as_float(payload.get("speed_mps"), default=0.0),
        distance_to_goal_m=_as_float(payload.get("distance_to_goal_m"), default=float("inf")),
    )


def parse_termination_reason(value: object) -> TerminationReason:
    if isinstance(value, TerminationReason):
        return value
    if isinstance(value, str):
        try:
            return TerminationReason(value.upper())
        except ValueError:
            return TerminationReason.ONGOING
    return TerminationReason.ONGOING


def parse_termination_reasons(value: object) -> tuple[TerminationReason, ...]:
    if isinstance(value, str | TerminationReason):
        reason = parse_termination_reason(value)
        return () if reason == TerminationReason.ONGOING else (reason,)

    if not isinstance(value, Sequence):
        return ()

    reasons: list[TerminationReason] = []
    for item in value:
        reason = parse_termination_reason(item)
        if reason == TerminationReason.ONGOING:
            continue
        reasons.append(reason)
    return normalize_termination_reasons(reasons)


def _as_int(value: object, *, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _as_float(value: object, *, default: float) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return default


def _as_bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    return default
