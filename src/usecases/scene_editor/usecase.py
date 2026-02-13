from __future__ import annotations

from usecases.ports.scene_editor_gateway_interface import SceneEditorGatewayInterface
from usecases.scene_editor.dtos import (
    ActorState,
    CleanupResult,
    ConnectRequest,
    ConnectionState,
    FollowMode,
    FollowRequest,
    FollowState,
    GatewaySpawnRequest,
    QuickSpawnRequest,
    SpawnCategory,
    SpawnRequest,
    SpawnResult,
    SpectatorDelta,
    SpectatorState,
)
from usecases.scene_editor.input_port import SceneEditorInputPort


class SceneEditorUseCase(SceneEditorInputPort):
    def __init__(self, gateway: SceneEditorGatewayInterface) -> None:
        self._gateway = gateway
        self._connection_state: ConnectionState | None = None
        self._spectator_state: SpectatorState | None = None
        self._follow_mode: FollowMode = FollowMode.NONE
        self._follow_target_actor_id: int | None = None
        self._spawned_actor_ids: list[int] = []
        self._spawned_vehicle_actor_ids: list[int] = []
        self._next_auto_vehicle_index = 1

    def connect(self, req: ConnectRequest) -> ConnectionState:
        state = self._gateway.connect(
            host=req.host,
            port=req.port,
            timeout_s=req.timeout_s,
            target_map=req.target_map,
        )
        self._connection_state = state
        self._follow_mode = FollowMode.NONE
        self._follow_target_actor_id = None
        self._spawned_actor_ids = []
        self._spawned_vehicle_actor_ids = []
        self._next_auto_vehicle_index = 1

        initial_spectator_state = SpectatorState(
            x=req.initial_spectator_x,
            y=req.initial_spectator_y,
            z=req.initial_spectator_z,
            pitch=-90.0,
            yaw=req.initial_spectator_yaw,
            roll=0.0,
        )
        self._gateway.set_spectator_topdown(initial_spectator_state)
        self._spectator_state = initial_spectator_state
        return state

    def spawn_vehicle_at(self, req: SpawnRequest) -> SpawnResult:
        spectator = self._require_spectator_state()
        auto_role_name: str | None = None
        role_name = req.role_name
        if role_name is None:
            auto_role_name = self._build_next_auto_role_name()
            role_name = auto_role_name

        result = self._gateway.spawn_actor(
            GatewaySpawnRequest(
                category=SpawnCategory.VEHICLE,
                x=req.x,
                y=req.y,
                z=req.z,
                yaw=req.yaw,
                blueprint_id=req.blueprint_id,
                role_name=role_name,
                spawn_min_z=req.spawn_min_z,
                spawn_max_z=req.spawn_max_z,
                spawn_probe_top_z=req.spawn_probe_top_z,
                spawn_probe_distance=req.spawn_probe_distance,
                fallback_base_z=spectator.z,
            )
        )

        if result.success and result.actor is not None:
            self._spawned_actor_ids.append(result.actor.actor_id)
            self._spawned_vehicle_actor_ids.append(result.actor.actor_id)
            if auto_role_name is not None:
                self._next_auto_vehicle_index += 1
            if self._follow_mode == FollowMode.AUTO_LATEST:
                self._follow_target_actor_id = result.actor.actor_id
                self._sync_spectator_to_actor(result.actor)

        return result

    def spawn_obstacle_at(self, req: SpawnRequest) -> SpawnResult:
        spectator = self._require_spectator_state()
        result = self._gateway.spawn_actor(
            GatewaySpawnRequest(
                category=SpawnCategory.OBSTACLE,
                x=req.x,
                y=req.y,
                z=req.z,
                yaw=req.yaw,
                blueprint_id=req.blueprint_id,
                role_name=req.role_name,
                spawn_min_z=req.spawn_min_z,
                spawn_max_z=req.spawn_max_z,
                spawn_probe_top_z=req.spawn_probe_top_z,
                spawn_probe_distance=req.spawn_probe_distance,
                fallback_base_z=spectator.z,
            )
        )
        if result.success and result.actor is not None:
            self._spawned_actor_ids.append(result.actor.actor_id)
        return result

    def bind_topdown_follow(self, req: FollowRequest | None = None) -> FollowState:
        self._require_spectator_state()

        target_actor: ActorState | None
        target_mode: FollowMode
        if req is None or (req.role_name is None and req.actor_id is None):
            target_actor = self._resolve_latest_vehicle_actor()
            target_mode = FollowMode.AUTO_LATEST
            if target_actor is None:
                self._follow_mode = FollowMode.NONE
                self._follow_target_actor_id = None
                return FollowState(
                    bound=False,
                    mode=FollowMode.NONE,
                    target_actor=None,
                    message="No spawned vehicle available for follow binding.",
                )
        else:
            target_actor = self._gateway.find_vehicle(
                role_name=req.role_name,
                actor_id=req.actor_id,
            )
            target_mode = FollowMode.EXPLICIT
            if target_actor is None:
                self._follow_mode = FollowMode.NONE
                self._follow_target_actor_id = None
                return FollowState(
                    bound=False,
                    mode=FollowMode.NONE,
                    target_actor=None,
                    message="Vehicle target not found by role_name/actor_id.",
                )

        self._follow_mode = target_mode
        self._follow_target_actor_id = target_actor.actor_id
        self._sync_spectator_to_actor(target_actor)
        return FollowState(
            bound=True,
            mode=target_mode,
            target_actor=target_actor,
            message="Follow binding established.",
        )

    def move_spectator(self, delta: SpectatorDelta) -> SpectatorState:
        current = self._require_spectator_state()
        next_state = current.moved(delta)
        fixed_topdown = SpectatorState(
            x=next_state.x,
            y=next_state.y,
            z=next_state.z,
            pitch=-90.0,
            yaw=next_state.yaw,
            roll=0.0,
        )
        self._gateway.set_spectator_topdown(fixed_topdown)
        self._spectator_state = fixed_topdown
        return fixed_topdown

    def spawn_vehicle_at_current_xy(self, req: QuickSpawnRequest) -> SpawnResult:
        spectator = self._require_spectator_state()
        return self.spawn_vehicle_at(
            SpawnRequest(
                x=spectator.x,
                y=spectator.y,
                z=req.z,
                yaw=req.yaw,
                blueprint_id=req.blueprint_id,
                role_name=req.role_name,
                spawn_min_z=req.spawn_min_z,
                spawn_max_z=req.spawn_max_z,
                spawn_probe_top_z=req.spawn_probe_top_z,
                spawn_probe_distance=req.spawn_probe_distance,
            )
        )

    def spawn_obstacle_at_current_xy(self, req: QuickSpawnRequest) -> SpawnResult:
        spectator = self._require_spectator_state()
        return self.spawn_obstacle_at(
            SpawnRequest(
                x=spectator.x,
                y=spectator.y,
                z=req.z,
                yaw=req.yaw,
                blueprint_id=req.blueprint_id,
                role_name=req.role_name,
                spawn_min_z=req.spawn_min_z,
                spawn_max_z=req.spawn_max_z,
                spawn_probe_top_z=req.spawn_probe_top_z,
                spawn_probe_distance=req.spawn_probe_distance,
            )
        )

    def tick_follow(self) -> SpectatorState | None:
        if self._follow_target_actor_id is None:
            return None

        target_actor = self._gateway.get_actor_state(self._follow_target_actor_id)
        if target_actor is None:
            self._follow_mode = FollowMode.NONE
            self._follow_target_actor_id = None
            return None

        self._sync_spectator_to_actor(target_actor)
        return self._spectator_state

    def close(self, destroy_spawned: bool) -> CleanupResult:
        cleanup = CleanupResult(destroyed_count=0, failed_count=0)
        if destroy_spawned and self._spawned_actor_ids:
            cleanup = self._gateway.destroy_actors(tuple(self._spawned_actor_ids))

        self._follow_mode = FollowMode.NONE
        self._follow_target_actor_id = None
        self._spawned_actor_ids = []
        self._spawned_vehicle_actor_ids = []
        self._spectator_state = None
        self._connection_state = None
        return cleanup

    def _require_spectator_state(self) -> SpectatorState:
        if self._connection_state is None or self._spectator_state is None:
            raise RuntimeError("Scene editor is not connected. Call connect() first.")
        return self._spectator_state

    def _resolve_latest_vehicle_actor(self) -> ActorState | None:
        for actor_id in reversed(self._spawned_vehicle_actor_ids):
            actor = self._gateway.get_actor_state(actor_id)
            if actor is not None:
                return actor
        return None

    def _sync_spectator_to_actor(self, actor: ActorState) -> None:
        spectator = self._require_spectator_state()
        next_state = SpectatorState(
            x=actor.x,
            y=actor.y,
            z=spectator.z,
            pitch=-90.0,
            yaw=spectator.yaw,
            roll=0.0,
        )
        self._gateway.set_spectator_topdown(next_state)
        self._spectator_state = next_state

    def _build_next_auto_role_name(self) -> str:
        return f"scene_editor_vehicle_{self._next_auto_vehicle_index}"
