from __future__ import annotations

import dataclasses

from usecases.ports.scene_editor_gateway_interface import SceneEditorGatewayInterface
from usecases.scene_editor.dtos import (
    ActorState,
    CleanupResult,
    ConnectRequest,
    ConnectionState,
    FollowRequest,
    GatewaySpawnRequest,
    QuickSpawnRequest,
    SpawnCategory,
    SpawnRequest,
    SpawnResult,
    SpectatorDelta,
    SpectatorState,
)
from usecases.scene_editor.usecase import SceneEditorUseCase


class FakeSceneEditorGateway(SceneEditorGatewayInterface):
    def __init__(self) -> None:
        self._connected = False
        self._spectator = SpectatorState(x=0.0, y=0.0, z=10.0, pitch=-90.0, yaw=0.0, roll=0.0)
        self._actors: dict[int, ActorState] = {}
        self._next_actor_id = 1

    def connect(self, host: str, port: int, timeout_s: float, target_map: str | None) -> ConnectionState:
        del timeout_s, target_map
        self._connected = True
        return ConnectionState(
            host=host,
            port=port,
            map_name="Carla/Maps/Town10HD_Opt",
            synchronous_mode=False,
            available_maps=("Carla/Maps/Town10HD_Opt",),
        )

    def get_spectator_state(self) -> SpectatorState:
        return self._spectator

    def set_spectator_topdown(self, state: SpectatorState) -> None:
        self._spectator = state

    def spawn_actor(self, req: GatewaySpawnRequest) -> SpawnResult:
        actor_id = self._next_actor_id
        self._next_actor_id += 1

        used_blueprint_id = req.blueprint_id
        if used_blueprint_id is None:
            if req.category == SpawnCategory.VEHICLE:
                used_blueprint_id = "vehicle.mini.cooper"
            else:
                used_blueprint_id = "static.prop.barrel"

        spawn_z = req.z if req.z is not None else req.fallback_base_z + 0.6
        actor = ActorState(
            actor_id=actor_id,
            type_id=used_blueprint_id,
            role_name=req.role_name,
            x=req.x,
            y=req.y,
            z=spawn_z,
            yaw=req.yaw,
        )
        self._actors[actor_id] = actor

        return SpawnResult(
            category=req.category,
            success=True,
            requested_blueprint_id=req.blueprint_id,
            used_blueprint_id=used_blueprint_id,
            actor=actor,
            ground_z_estimate=req.fallback_base_z,
            message="spawned",
        )

    def get_actor_state(self, actor_id: int) -> ActorState | None:
        return self._actors.get(actor_id)

    def find_vehicle(self, role_name: str | None, actor_id: int | None) -> ActorState | None:
        if actor_id is not None:
            actor = self._actors.get(actor_id)
            if actor is None or not actor.type_id.startswith("vehicle."):
                return None
            return actor

        if role_name is None:
            return None

        matches = [
            actor
            for actor in self._actors.values()
            if actor.type_id.startswith("vehicle.") and actor.role_name == role_name
        ]
        if len(matches) != 1:
            return None
        return matches[0]

    def destroy_actors(self, actor_ids: tuple[int, ...]) -> CleanupResult:
        destroyed = 0
        failed = 0
        for actor_id in actor_ids:
            if actor_id in self._actors:
                del self._actors[actor_id]
                destroyed += 1
            else:
                failed += 1
        return CleanupResult(destroyed_count=destroyed, failed_count=failed)

    def move_actor(self, actor_id: int, x: float, y: float) -> None:
        actor = self._actors[actor_id]
        self._actors[actor_id] = dataclasses.replace(actor, x=x, y=y)


def test_bind_follow_defaults_to_latest_spawned_vehicle() -> None:
    gateway = FakeSceneEditorGateway()
    usecase = SceneEditorUseCase(gateway=gateway)
    usecase.connect(ConnectRequest())

    first = usecase.spawn_vehicle_at(SpawnRequest(x=1.0, y=2.0, yaw=15.0))
    second = usecase.spawn_vehicle_at(SpawnRequest(x=8.0, y=9.0, yaw=25.0))

    assert first.actor is not None
    assert second.actor is not None

    follow_state = usecase.bind_topdown_follow()
    assert follow_state.bound
    assert follow_state.target_actor is not None
    assert follow_state.target_actor.actor_id == second.actor.actor_id

    spectator_state = usecase.tick_follow()
    assert spectator_state is not None
    assert spectator_state.x == second.actor.x
    assert spectator_state.y == second.actor.y
    assert spectator_state.z == 20.0


def test_bind_follow_supports_role_name_and_actor_id_fallback() -> None:
    gateway = FakeSceneEditorGateway()
    usecase = SceneEditorUseCase(gateway=gateway)
    usecase.connect(ConnectRequest())

    first = usecase.spawn_vehicle_at(SpawnRequest(x=1.0, y=1.0, role_name="dup"))
    second = usecase.spawn_vehicle_at(SpawnRequest(x=2.0, y=2.0, role_name="dup"))
    assert first.actor is not None
    assert second.actor is not None

    ambiguous = usecase.bind_topdown_follow(FollowRequest(role_name="dup"))
    assert not ambiguous.bound

    explicit = usecase.bind_topdown_follow(FollowRequest(role_name="dup", actor_id=first.actor.actor_id))
    assert explicit.bound
    assert explicit.target_actor is not None
    assert explicit.target_actor.actor_id == first.actor.actor_id


def test_quick_spawn_uses_current_spectator_xy_and_auto_role_name() -> None:
    gateway = FakeSceneEditorGateway()
    usecase = SceneEditorUseCase(gateway=gateway)
    usecase.connect(ConnectRequest())
    usecase.move_spectator(SpectatorDelta(dx=3.0, dy=-4.0, dz=1.5))

    vehicle = usecase.spawn_vehicle_at_current_xy(req=QuickSpawnRequest())
    obstacle = usecase.spawn_obstacle_at_current_xy(req=QuickSpawnRequest())

    assert vehicle.actor is not None
    assert obstacle.actor is not None
    assert vehicle.actor.x == 3.0
    assert vehicle.actor.y == -4.0
    assert obstacle.actor.x == 3.0
    assert obstacle.actor.y == -4.0
    assert vehicle.actor.role_name == "scene_editor_vehicle_1"


def test_close_with_destroy_flag_removes_spawned_actors() -> None:
    gateway = FakeSceneEditorGateway()
    usecase = SceneEditorUseCase(gateway=gateway)
    usecase.connect(ConnectRequest())
    vehicle = usecase.spawn_vehicle_at(SpawnRequest(x=1.0, y=2.0))
    obstacle = usecase.spawn_obstacle_at(SpawnRequest(x=2.0, y=3.0))
    assert vehicle.actor is not None
    assert obstacle.actor is not None

    cleanup = usecase.close(destroy_spawned=True)
    assert cleanup.destroyed_count == 2
    assert cleanup.failed_count == 0
