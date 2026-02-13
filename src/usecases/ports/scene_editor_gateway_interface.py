from __future__ import annotations

from typing import Protocol

from usecases.scene_editor.dtos import (
    ActorState,
    CleanupResult,
    ConnectionState,
    GatewaySpawnRequest,
    SpawnResult,
    SpectatorState,
)


class SceneEditorGatewayInterface(Protocol):
    def connect(self, host: str, port: int, timeout_s: float, target_map: str | None) -> ConnectionState:
        ...

    def get_spectator_state(self) -> SpectatorState:
        ...

    def set_spectator_topdown(self, state: SpectatorState) -> None:
        ...

    def spawn_actor(self, req: GatewaySpawnRequest) -> SpawnResult:
        ...

    def get_actor_state(self, actor_id: int) -> ActorState | None:
        ...

    def find_vehicle(self, role_name: str | None, actor_id: int | None) -> ActorState | None:
        ...

    def destroy_actors(self, actor_ids: tuple[int, ...]) -> CleanupResult:
        ...
