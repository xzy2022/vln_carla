from __future__ import annotations

from enum import IntFlag
from typing import Callable


class WorldSettings:
    synchronous_mode: bool
    fixed_delta_seconds: float
    no_rendering_mode: bool


class Location:
    x: float
    y: float
    z: float

    def __init__(self, x: float = ..., y: float = ..., z: float = ...) -> None: ...
    def __add__(self, other: Location) -> Location: ...
    def __mul__(self, scalar: float) -> Location: ...


class Rotation:
    roll: float
    pitch: float
    yaw: float

    def __init__(self, roll: float = ..., pitch: float = ..., yaw: float = ...) -> None: ...


class Transform:
    location: Location
    rotation: Rotation

    def __init__(self, location: Location = ..., rotation: Rotation = ...) -> None: ...
    def get_forward_vector(self) -> Location: ...


class Timestamp:
    elapsed_seconds: float


class WorldSnapshot:
    frame: int
    timestamp: Timestamp


class VehicleControl:
    def __init__(self, throttle: float = ..., steer: float = ..., brake: float = ...) -> None: ...


class Actor:
    def destroy(self) -> None: ...


class Sensor(Actor):
    def listen(self, callback: Callable[[Image], None]) -> None: ...
    def stop(self) -> None: ...


class Vehicle(Actor):
    def apply_control(self, control: VehicleControl) -> None: ...
    def get_transform(self) -> Transform: ...
    def get_velocity(self) -> Location: ...


class Spectator(Actor):
    def set_transform(self, transform: Transform) -> None: ...


class Image:
    frame: int
    width: int
    height: int
    raw_data: bytes


class ActorBlueprint:
    def set_attribute(self, key: str, value: str) -> None: ...


class BlueprintLibrary:
    def filter(self, pattern: str) -> list[ActorBlueprint]: ...
    def find(self, blueprint_id: str) -> ActorBlueprint: ...


class Map:
    name: str

    def get_spawn_points(self) -> list[Transform]: ...


class MapLayer(IntFlag):
    NONE: MapLayer
    Buildings: MapLayer
    Decals: MapLayer
    Foliage: MapLayer
    Ground: MapLayer
    ParkedVehicles: MapLayer
    Particles: MapLayer
    Props: MapLayer
    StreetLights: MapLayer
    Walls: MapLayer
    All: MapLayer


class World:
    def get_settings(self) -> WorldSettings: ...
    def apply_settings(self, settings: WorldSettings) -> None: ...
    def tick(self) -> int: ...
    def get_snapshot(self) -> WorldSnapshot: ...
    def get_blueprint_library(self) -> BlueprintLibrary: ...
    def get_map(self) -> Map: ...
    def spawn_actor(self, blueprint: ActorBlueprint, transform: Transform, attach_to: Actor | None = ...) -> Actor: ...
    def unload_map_layer(self, map_layer: MapLayer) -> None: ...
    def get_spectator(self) -> Spectator: ...


class Client:
    def __init__(self, host: str, port: int) -> None: ...
    def set_timeout(self, timeout_seconds: float) -> None: ...
    def get_world(self) -> World: ...
    def load_world(self, map_name: str) -> World: ...
