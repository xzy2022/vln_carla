from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

_EPSILON = 1e-9

WorldXYPoint = tuple[float, float]


def validate_polygon(points: Sequence[WorldXYPoint]) -> tuple[WorldXYPoint, ...]:
    """Validate and normalize a polygon as a closed ring."""
    if len(points) < 3:
        raise ValueError("polygon requires at least 3 vertices")

    normalized: list[WorldXYPoint] = []
    for x, y in points:
        fx = float(x)
        fy = float(y)
        if not math.isfinite(fx) or not math.isfinite(fy):
            raise ValueError("polygon contains non-finite coordinate")
        normalized.append((fx, fy))

    deduped = _remove_duplicate_points(normalized)
    if len(deduped) < 3:
        raise ValueError("polygon has too many duplicate points")

    if _polygon_area(deduped) <= _EPSILON:
        raise ValueError("polygon area is too small")

    if _points_equal(deduped[0], deduped[-1]):
        return tuple(deduped)
    return tuple([*deduped, deduped[0]])


def point_in_polygon(point: WorldXYPoint, polygon: Sequence[WorldXYPoint]) -> bool:
    if len(polygon) < 4:
        return False

    x, y = point
    inside = False
    for idx in range(len(polygon) - 1):
        p1 = polygon[idx]
        p2 = polygon[idx + 1]
        if _point_on_segment(point, p1, p2):
            return True

        x1, y1 = p1
        x2, y2 = p2
        if (y1 > y) != (y2 > y):
            x_intersection = x1 + ((y - y1) * (x2 - x1) / (y2 - y1))
            if x_intersection >= x - _EPSILON:
                inside = not inside
    return inside


def inside_by_any_corner(
    corners: Sequence[WorldXYPoint],
    polygon: Sequence[WorldXYPoint],
) -> bool:
    return any(point_in_polygon(corner, polygon) for corner in corners)


def entered_zones(
    prev_inside: Sequence[str],
    curr_inside: Sequence[str],
) -> tuple[str, ...]:
    prev_set = set(prev_inside)
    return tuple(zone_id for zone_id in curr_inside if zone_id not in prev_set)


def filter_entered_by_cooldown(
    entered_zone_ids: Sequence[str],
    *,
    current_step: int,
    last_enter_step_by_zone: Mapping[str, int],
    cooldown_steps_by_zone: Mapping[str, int],
) -> tuple[str, ...]:
    effective: list[str] = []
    for zone_id in entered_zone_ids:
        cooldown_steps = max(0, cooldown_steps_by_zone.get(zone_id, 0))
        last_step = last_enter_step_by_zone.get(zone_id)
        if last_step is not None and (current_step - last_step) <= cooldown_steps:
            continue
        effective.append(zone_id)
    return tuple(effective)


def _remove_duplicate_points(points: Sequence[WorldXYPoint]) -> list[WorldXYPoint]:
    unique: list[WorldXYPoint] = []
    for point in points:
        if any(_points_equal(point, existing) for existing in unique):
            continue
        unique.append(point)
    return unique


def _polygon_area(points: Sequence[WorldXYPoint]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx in range(len(points)):
        x1, y1 = points[idx]
        x2, y2 = points[(idx + 1) % len(points)]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) * 0.5


def _point_on_segment(
    point: WorldXYPoint,
    segment_start: WorldXYPoint,
    segment_end: WorldXYPoint,
) -> bool:
    px, py = point
    x1, y1 = segment_start
    x2, y2 = segment_end

    cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
    if abs(cross) > _EPSILON:
        return False

    min_x = min(x1, x2) - _EPSILON
    max_x = max(x1, x2) + _EPSILON
    min_y = min(y1, y2) - _EPSILON
    max_y = max(y1, y2) + _EPSILON
    return min_x <= px <= max_x and min_y <= py <= max_y


def _points_equal(a: WorldXYPoint, b: WorldXYPoint) -> bool:
    return abs(a[0] - b[0]) <= _EPSILON and abs(a[1] - b[1]) <= _EPSILON
