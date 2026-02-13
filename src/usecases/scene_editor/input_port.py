from __future__ import annotations

from typing import Protocol

from usecases.scene_editor.dtos import (
    CleanupResult,
    ConnectRequest,
    ConnectionState,
    FollowRequest,
    FollowState,
    QuickSpawnRequest,
    SpawnRequest,
    SpawnResult,
    SpectatorDelta,
    SpectatorState,
)


class SceneEditorInputPort(Protocol):
    def connect(self, req: ConnectRequest) -> ConnectionState:
        ...

    def spawn_vehicle_at(self, req: SpawnRequest) -> SpawnResult:
        ...

    def spawn_obstacle_at(self, req: SpawnRequest) -> SpawnResult:
        ...

    def bind_topdown_follow(self, req: FollowRequest | None = None) -> FollowState:
        ...

    def move_spectator(self, delta: SpectatorDelta) -> SpectatorState:
        ...

    def spawn_vehicle_at_current_xy(self, req: QuickSpawnRequest) -> SpawnResult:
        ...

    def spawn_obstacle_at_current_xy(self, req: QuickSpawnRequest) -> SpawnResult:
        ...

    def tick_follow(self) -> SpectatorState | None:
        ...

    def close(self, destroy_spawned: bool) -> CleanupResult:
        ...
