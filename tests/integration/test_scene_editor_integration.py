from __future__ import annotations

import os
from typing import Any

import pytest

from infrastructure.carla.scene_editor_gateway import CarlaSceneEditorGateway
from usecases.scene_editor.dtos import ConnectRequest, SpawnRequest
from usecases.scene_editor.usecase import SceneEditorUseCase


@pytest.mark.integration
def test_scene_editor_connect_spawn_follow_roundtrip() -> None:
    host, port = _get_server_address_or_skip()
    carla = pytest.importorskip("carla")
    client = _build_client_or_skip(carla_module=carla, host=host, port=port)

    world = client.get_world()
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        pytest.skip("integration test skipped: no spawn points on current map")
    seed_spawn = spawn_points[0]

    usecase = SceneEditorUseCase(gateway=CarlaSceneEditorGateway())
    usecase.connect(
        ConnectRequest(
            host=host,
            port=port,
            timeout_s=5.0,
            target_map=os.getenv("SCENE_EDITOR_TEST_MAP"),
        )
    )

    spawn_result = usecase.spawn_vehicle_at(
        SpawnRequest(
            x=float(seed_spawn.location.x),
            y=float(seed_spawn.location.y),
            z=float(seed_spawn.location.z) + 0.3,
            yaw=float(seed_spawn.rotation.yaw),
        )
    )
    if not spawn_result.success or spawn_result.actor is None:
        pytest.skip("integration test skipped: vehicle spawn failed in current world state")

    follow_state = usecase.bind_topdown_follow()
    assert follow_state.bound
    spectator = usecase.tick_follow()
    assert spectator is not None
    assert abs(spectator.x - spawn_result.actor.x) < 1e-6
    assert abs(spectator.y - spawn_result.actor.y) < 1e-6

    cleanup = usecase.close(destroy_spawned=True)
    assert cleanup.destroyed_count >= 1


def _get_server_address_or_skip() -> tuple[str, int]:
    host = os.getenv("CARLA_SERVER_HOST")
    port_raw = os.getenv("CARLA_SERVER_PORT")
    if not host or not port_raw:
        pytest.skip("integration test skipped: CARLA_SERVER_HOST/CARLA_SERVER_PORT is not set")
    try:
        port = int(port_raw)
    except ValueError:
        pytest.skip("integration test skipped: CARLA_SERVER_PORT is not an integer")
    return host, port


def _build_client_or_skip(*, carla_module: Any, host: str, port: int) -> Any:
    try:
        client = carla_module.Client(host, port)
        client.set_timeout(3.0)
        client.get_world()
        return client
    except Exception as exc:
        pytest.skip(f"integration test skipped: unable to connect CARLA server ({exc})")
