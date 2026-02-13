from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SpawnCategory(str, Enum):
    VEHICLE = "vehicle"
    OBSTACLE = "obstacle"


class FollowMode(str, Enum):
    NONE = "none"
    AUTO_LATEST = "auto_latest"
    EXPLICIT = "explicit"


@dataclass(frozen=True)
class ConnectRequest:
    host: str = "127.0.0.1"
    port: int = 2000
    timeout_s: float = 10.0
    target_map: str | None = None
    initial_spectator_x: float = 0.0
    initial_spectator_y: float = 0.0
    initial_spectator_z: float = 20.0
    initial_spectator_yaw: float = 0.0


@dataclass(frozen=True)
class ConnectionState:
    host: str
    port: int
    map_name: str
    synchronous_mode: bool
    available_maps: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActorState:
    actor_id: int
    type_id: str
    role_name: str | None
    x: float
    y: float
    z: float
    yaw: float


@dataclass(frozen=True)
class SpectatorState:
    x: float
    y: float
    z: float
    pitch: float
    yaw: float
    roll: float

    def moved(self, delta: SpectatorDelta) -> SpectatorState:
        return SpectatorState(
            x=self.x + delta.dx,
            y=self.y + delta.dy,
            z=self.z + delta.dz,
            pitch=self.pitch,
            yaw=self.yaw,
            roll=self.roll,
        )


@dataclass(frozen=True)
class SpectatorDelta:
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0


@dataclass(frozen=True)
class SpawnRequest:
    x: float
    y: float
    z: float | None = None
    yaw: float = 0.0
    blueprint_id: str | None = None
    role_name: str | None = None
    spawn_min_z: float = -200.0
    spawn_max_z: float = 200.0
    spawn_probe_top_z: float = 120.0
    spawn_probe_distance: float = 300.0


@dataclass(frozen=True)
class QuickSpawnRequest:
    z: float | None = None
    yaw: float = 0.0
    blueprint_id: str | None = None
    role_name: str | None = None
    spawn_min_z: float = -200.0
    spawn_max_z: float = 200.0
    spawn_probe_top_z: float = 120.0
    spawn_probe_distance: float = 300.0


@dataclass(frozen=True)
class GatewaySpawnRequest:
    category: SpawnCategory
    x: float
    y: float
    z: float | None
    yaw: float
    blueprint_id: str | None
    role_name: str | None
    spawn_min_z: float
    spawn_max_z: float
    spawn_probe_top_z: float
    spawn_probe_distance: float
    fallback_base_z: float


@dataclass(frozen=True)
class SpawnResult:
    category: SpawnCategory
    success: bool
    requested_blueprint_id: str | None
    used_blueprint_id: str | None
    actor: ActorState | None
    ground_z_estimate: float
    message: str


@dataclass(frozen=True)
class FollowRequest:
    role_name: str | None = None
    actor_id: int | None = None


@dataclass(frozen=True)
class FollowState:
    bound: bool
    mode: FollowMode
    target_actor: ActorState | None
    message: str


@dataclass(frozen=True)
class CleanupResult:
    destroyed_count: int
    failed_count: int
