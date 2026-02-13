from __future__ import annotations

from infrastructure.carla.scene_editor_gateway import (
    clamp,
    generate_candidate_z_values,
    resolve_target_map,
    short_map_name,
)


def test_short_map_name_extracts_suffix() -> None:
    assert short_map_name("/Game/Carla/Maps/Town10HD_Opt") == "Town10HD_Opt"


def test_resolve_target_map_matches_short_name_case_insensitive() -> None:
    available = (
        "/Game/Carla/Maps/Mine_01",
        "/Game/Carla/Maps/Town10HD_Opt",
    )
    assert resolve_target_map("town10hd_opt", available) == "/Game/Carla/Maps/Town10HD_Opt"


def test_resolve_target_map_returns_none_when_ambiguous() -> None:
    available = (
        "/Game/A/Town01",
        "/Game/B/Town01",
    )
    assert resolve_target_map("Town01", available) is None


def test_clamp_caps_value_to_bounds() -> None:
    assert clamp(-1.0, 0.0, 10.0) == 0.0
    assert clamp(11.0, 0.0, 10.0) == 10.0
    assert clamp(3.0, 0.0, 10.0) == 3.0


def test_generate_candidate_z_values_respects_bounds_and_uniqueness() -> None:
    candidates = generate_candidate_z_values(base_ground_z=0.0, min_z=-1.0, max_z=1.0)
    assert candidates
    assert len(candidates) == len(set(candidates))
    assert min(candidates) >= -1.0
    assert max(candidates) <= 1.0
    assert -1.0 in candidates
    assert 1.0 in candidates
