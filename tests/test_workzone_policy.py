from __future__ import annotations

import math

import pytest

from domain.workzone_policy import (
    entered_zones,
    filter_entered_by_cooldown,
    inside_by_any_corner,
    point_in_polygon,
    validate_polygon,
)


def test_validate_polygon_auto_closes_ring() -> None:
    polygon = validate_polygon(((0.0, 0.0), (2.0, 0.0), (0.0, 2.0)))
    assert len(polygon) == 4
    assert polygon[0] == polygon[-1]


def test_validate_polygon_rejects_non_finite_points() -> None:
    with pytest.raises(ValueError):
        validate_polygon(((0.0, 0.0), (math.nan, 1.0), (1.0, 1.0)))


def test_validate_polygon_rejects_too_many_duplicates() -> None:
    with pytest.raises(ValueError):
        validate_polygon(((0.0, 0.0), (0.0, 0.0), (1.0, 1.0), (1.0, 1.0)))


def test_point_on_boundary_is_inside() -> None:
    polygon = validate_polygon(((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)))
    assert point_in_polygon((0.0, 2.0), polygon)
    assert not point_in_polygon((5.0, 2.0), polygon)


def test_inside_by_any_corner_detects_inside() -> None:
    polygon = validate_polygon(((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)))
    corners = ((5.0, 5.0), (5.0, 6.0), (3.0, 3.0), (6.0, 6.0))
    assert inside_by_any_corner(corners, polygon)


def test_entered_zones_returns_entered_ids() -> None:
    assert entered_zones(("zone_a",), ("zone_a", "zone_b", "zone_c")) == ("zone_b", "zone_c")


def test_filter_entered_by_cooldown() -> None:
    entered = ("zone_a", "zone_b", "zone_c")
    effective = filter_entered_by_cooldown(
        entered,
        current_step=10,
        last_enter_step_by_zone={"zone_a": 9, "zone_b": 6},
        cooldown_steps_by_zone={"zone_a": 2, "zone_b": 3, "zone_c": 0},
    )
    assert effective == ("zone_b", "zone_c")
