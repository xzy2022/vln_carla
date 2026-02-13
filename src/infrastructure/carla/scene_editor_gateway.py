from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath
from typing import Any, Protocol, cast

from usecases.ports.scene_editor_gateway_interface import SceneEditorGatewayInterface
from usecases.scene_editor.dtos import (
    ActorState,
    CleanupResult,
    ConnectionState,
    GatewaySpawnRequest,
    SpawnCategory,
    SpawnResult,
    SpectatorState,
)


class _LocationLike(Protocol):
    x: float
    y: float
    z: float


class _RotationLike(Protocol):
    pitch: float
    yaw: float
    roll: float


class _TransformLike(Protocol):
    location: _LocationLike
    rotation: _RotationLike


class _ActorLike(Protocol):
    id: int
    type_id: str
    attributes: Mapping[str, object]

    def get_transform(self) -> _TransformLike:
        ...


def short_map_name(map_path: str) -> str:
    return PurePosixPath(map_path).name


def resolve_target_map(requested: str, available_maps: Iterable[str]) -> str | None:
    available = tuple(available_maps)
    normalized = requested.strip().strip("/").lower()
    if not normalized:
        return None

    exact = [m for m in available if m.lower().strip("/") == normalized]
    if len(exact) == 1:
        return exact[0]

    suffix = [m for m in available if m.lower().endswith("/" + normalized)]
    if len(suffix) == 1:
        return suffix[0]

    short = [m for m in available if short_map_name(m).lower() == normalized]
    if len(short) == 1:
        return short[0]

    return None


def clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def generate_candidate_z_values(base_ground_z: float, min_z: float, max_z: float) -> list[float]:
    if min_z > max_z:
        raise ValueError("min_z must be <= max_z")

    start = clamp(base_ground_z + 0.6, min_z, max_z)
    offsets = (
        0.0,
        0.2,
        -0.2,
        0.5,
        -0.5,
        1.0,
        -1.0,
        1.5,
        -1.5,
        2.0,
        -2.0,
        3.0,
        -3.0,
        5.0,
        -5.0,
    )

    seen: set[float] = set()
    candidates: list[float] = []

    def add(z: float) -> None:
        rounded = round(clamp(z, min_z, max_z), 3)
        if rounded in seen:
            return
        seen.add(rounded)
        candidates.append(rounded)

    for delta in offsets:
        add(start + delta)

    z = start - 6.0
    while z <= start + 6.0:
        add(z)
        z += 0.4

    z = min_z
    while z <= max_z:
        add(z)
        z += 2.0

    add(min_z)
    add(max_z)
    return candidates


class CarlaSceneEditorGateway(SceneEditorGatewayInterface):
    def __init__(self) -> None:
        self._carla: Any | None = None
        self._client: Any | None = None
        self._world: Any | None = None
        self._host = ""
        self._port = 0

    def connect(self, host: str, port: int, timeout_s: float, target_map: str | None) -> ConnectionState:
        carla_mod = self._import_carla()
        client = carla_mod.Client(host, port)
        client.set_timeout(timeout_s)
        world = client.get_world()
        current_map = str(world.get_map().name)

        raw_available_maps = self._get_available_maps(client, current_map)
        resolved_target: str | None = None
        if target_map is not None:
            resolved_target = resolve_target_map(target_map, raw_available_maps)
            if resolved_target is None:
                available_list = ", ".join(raw_available_maps)
                raise RuntimeError(
                    f"Target map '{target_map}' is not available. Available maps: {available_list}"
                )

        if (
            resolved_target is not None
            and short_map_name(current_map).lower() != short_map_name(resolved_target).lower()
        ):
            world = client.load_world(resolved_target)
            current_map = str(world.get_map().name)

        settings = world.get_settings()

        self._carla = carla_mod
        self._client = client
        self._world = world
        self._host = host
        self._port = port

        return ConnectionState(
            host=host,
            port=port,
            map_name=current_map,
            synchronous_mode=bool(settings.synchronous_mode),
            available_maps=raw_available_maps,
        )

    def get_spectator_state(self) -> SpectatorState:
        world = self._require_world()
        spectator = world.get_spectator()
        get_transform = getattr(spectator, "get_transform", None)
        if not callable(get_transform):
            raise RuntimeError("CARLA spectator object does not support get_transform().")
        transform = cast(_TransformLike, get_transform())

        return SpectatorState(
            x=float(transform.location.x),
            y=float(transform.location.y),
            z=float(transform.location.z),
            pitch=float(transform.rotation.pitch),
            yaw=float(transform.rotation.yaw),
            roll=float(transform.rotation.roll),
        )

    def set_spectator_topdown(self, state: SpectatorState) -> None:
        world = self._require_world()
        carla_mod = self._require_carla()
        spectator = world.get_spectator()
        transform = carla_mod.Transform(
            carla_mod.Location(x=state.x, y=state.y, z=state.z),
            carla_mod.Rotation(pitch=state.pitch, yaw=state.yaw, roll=state.roll),
        )
        spectator.set_transform(transform)

    def spawn_actor(self, req: GatewaySpawnRequest) -> SpawnResult:
        world = self._require_world()
        carla_mod = self._require_carla()
        if req.spawn_min_z > req.spawn_max_z:
            raise RuntimeError("spawn_min_z must be <= spawn_max_z")

        blueprint, used_blueprint_id = self._pick_blueprint(
            world=world,
            category=req.category,
            preferred_blueprint_id=req.blueprint_id,
        )
        self._try_set_role_name(blueprint=blueprint, role_name=req.role_name)

        if req.z is not None:
            transform = self._build_transform(
                carla_mod=carla_mod,
                x=req.x,
                y=req.y,
                z=req.z,
                yaw=req.yaw,
            )
            actor = self._try_spawn_actor(world=world, blueprint=blueprint, transform=transform)
            if actor is None:
                return SpawnResult(
                    category=req.category,
                    success=False,
                    requested_blueprint_id=req.blueprint_id,
                    used_blueprint_id=used_blueprint_id,
                    actor=None,
                    ground_z_estimate=req.z,
                    message="Spawn failed at fixed z (likely collision).",
                )
            return SpawnResult(
                category=req.category,
                success=True,
                requested_blueprint_id=req.blueprint_id,
                used_blueprint_id=used_blueprint_id,
                actor=self._build_actor_state(actor),
                ground_z_estimate=req.z,
                message="Spawned successfully.",
            )

        ground_z = self._estimate_ground_z(
            world=world,
            carla_mod=carla_mod,
            x=req.x,
            y=req.y,
            probe_top_z=req.spawn_probe_top_z,
            probe_distance=req.spawn_probe_distance,
        )
        if ground_z is None:
            ground_z = req.fallback_base_z

        for z in generate_candidate_z_values(ground_z, req.spawn_min_z, req.spawn_max_z):
            transform = self._build_transform(
                carla_mod=carla_mod,
                x=req.x,
                y=req.y,
                z=z,
                yaw=req.yaw,
            )
            actor = self._try_spawn_actor(world=world, blueprint=blueprint, transform=transform)
            if actor is not None:
                return SpawnResult(
                    category=req.category,
                    success=True,
                    requested_blueprint_id=req.blueprint_id,
                    used_blueprint_id=used_blueprint_id,
                    actor=self._build_actor_state(actor),
                    ground_z_estimate=ground_z,
                    message="Spawned successfully.",
                )

        return SpawnResult(
            category=req.category,
            success=False,
            requested_blueprint_id=req.blueprint_id,
            used_blueprint_id=used_blueprint_id,
            actor=None,
            ground_z_estimate=ground_z,
            message="Spawn failed after auto-z collision search.",
        )

    def get_actor_state(self, actor_id: int) -> ActorState | None:
        world = self._require_world()
        actor = self._find_actor(world=world, actor_id=actor_id)
        if actor is None:
            return None
        try:
            return self._build_actor_state(actor)
        except RuntimeError:
            return None

    def find_vehicle(self, role_name: str | None, actor_id: int | None) -> ActorState | None:
        world = self._require_world()
        actors = self._list_actors(world)

        if actor_id is not None:
            actor = self._find_actor(world=world, actor_id=actor_id)
            if actor is None or not self._is_vehicle_actor(actor):
                return None
            try:
                return self._build_actor_state(actor)
            except RuntimeError:
                return None

        if role_name is None:
            return None

        matched: list[Any] = []
        for actor in actors:
            if not self._is_vehicle_actor(actor):
                continue
            actor_role_name = self._extract_actor_role_name(actor)
            if actor_role_name == role_name:
                matched.append(actor)

        if len(matched) != 1:
            return None

        try:
            return self._build_actor_state(matched[0])
        except RuntimeError:
            return None

    def destroy_actors(self, actor_ids: tuple[int, ...]) -> CleanupResult:
        world = self._require_world()
        actors = self._list_actors(world)
        by_id: dict[int, Any] = {}
        for actor in actors:
            raw_id = getattr(actor, "id", None)
            if isinstance(raw_id, int):
                by_id[raw_id] = actor

        destroyed = 0
        failed = 0
        for actor_id in actor_ids:
            actor = by_id.get(actor_id)
            if actor is None:
                failed += 1
                continue
            try:
                actor.destroy()
                destroyed += 1
            except RuntimeError:
                failed += 1

        return CleanupResult(destroyed_count=destroyed, failed_count=failed)

    def _import_carla(self) -> Any:
        if self._carla is not None:
            return self._carla
        try:
            import carla  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("Python package 'carla' is not installed in this environment.") from exc
        self._carla = carla
        return carla

    def _require_carla(self) -> Any:
        if self._carla is None:
            raise RuntimeError("CARLA module is not initialized. Call connect() first.")
        return self._carla

    def _require_world(self) -> Any:
        if self._world is None:
            raise RuntimeError("CARLA world is not initialized. Call connect() first.")
        return self._world

    def _get_available_maps(self, client: Any, fallback_current_map: str) -> tuple[str, ...]:
        get_available_maps = getattr(client, "get_available_maps", None)
        if not callable(get_available_maps):
            return (fallback_current_map,)

        raw_values = get_available_maps()
        if not isinstance(raw_values, Iterable):
            return (fallback_current_map,)
        iterable_values = cast(Iterable[object], raw_values)
        raw_maps = tuple(str(item) for item in iterable_values)
        if not raw_maps:
            return (fallback_current_map,)
        return tuple(sorted(raw_maps))

    def _pick_blueprint(self, world: Any, category: SpawnCategory, preferred_blueprint_id: str | None) -> tuple[Any, str]:
        library = world.get_blueprint_library()

        if preferred_blueprint_id:
            preferred = list(library.filter(preferred_blueprint_id))
            if preferred:
                selected = preferred[0]
                return selected, self._blueprint_id(selected, preferred_blueprint_id)

        if category == SpawnCategory.VEHICLE:
            vehicles = list(library.filter("vehicle.*"))
            if not vehicles:
                raise RuntimeError("No vehicle blueprint found in this CARLA build.")
            selected = vehicles[0]
            return selected, self._blueprint_id(selected, "vehicle.*")

        props = list(library.filter("static.prop.*"))
        if not props:
            raise RuntimeError("No static.prop.* blueprint found in this CARLA build.")

        barrel_candidates = [bp for bp in props if "barrel" in self._blueprint_id(bp, "").lower()]
        if barrel_candidates:
            selected = barrel_candidates[0]
        else:
            selected = props[0]
        return selected, self._blueprint_id(selected, "static.prop.*")

    def _try_set_role_name(self, blueprint: Any, role_name: str | None) -> None:
        if role_name is None:
            return

        set_attribute = getattr(blueprint, "set_attribute", None)
        if not callable(set_attribute):
            return

        has_attribute = getattr(blueprint, "has_attribute", None)
        if callable(has_attribute):
            try:
                if not bool(has_attribute("role_name")):
                    return
            except RuntimeError:
                return

        try:
            set_attribute("role_name", role_name)
        except RuntimeError:
            return

    def _build_transform(self, carla_mod: Any, x: float, y: float, z: float, yaw: float) -> Any:
        return carla_mod.Transform(
            carla_mod.Location(x=x, y=y, z=z),
            carla_mod.Rotation(pitch=0.0, yaw=yaw, roll=0.0),
        )

    def _try_spawn_actor(self, world: Any, blueprint: Any, transform: Any) -> Any | None:
        try_spawn_actor = getattr(world, "try_spawn_actor", None)
        if callable(try_spawn_actor):
            return try_spawn_actor(blueprint, transform)

        spawn_actor = getattr(world, "spawn_actor", None)
        if not callable(spawn_actor):
            raise RuntimeError("CARLA world does not expose spawn_actor/try_spawn_actor.")
        try:
            return spawn_actor(blueprint, transform)
        except RuntimeError:
            return None

    def _estimate_ground_z(
        self,
        world: Any,
        carla_mod: Any,
        x: float,
        y: float,
        probe_top_z: float,
        probe_distance: float,
    ) -> float | None:
        ground_projection = getattr(world, "ground_projection", None)
        if not callable(ground_projection):
            return None

        labeled = ground_projection(
            carla_mod.Location(x=x, y=y, z=probe_top_z),
            probe_distance,
        )
        if labeled is None:
            return None
        location = getattr(labeled, "location", None)
        if location is None:
            return None
        z_value = getattr(location, "z", None)
        if isinstance(z_value, float):
            return z_value
        if isinstance(z_value, int):
            return float(z_value)
        return None

    def _list_actors(self, world: Any) -> list[Any]:
        actors = world.get_actors()
        if isinstance(actors, list):
            return cast(list[Any], actors)
        if isinstance(actors, tuple):
            return list(cast(tuple[Any, ...], actors))
        if isinstance(actors, Iterable):
            iterable_actors = cast(Iterable[Any], actors)
            return [actor for actor in iterable_actors]
        raise RuntimeError("CARLA actor collection is not iterable.")

    def _find_actor(self, world: Any, actor_id: int) -> Any | None:
        for actor in self._list_actors(world):
            raw_id = getattr(actor, "id", None)
            if raw_id == actor_id:
                return actor
        return None

    def _build_actor_state(self, actor: Any) -> ActorState:
        typed_actor = cast(_ActorLike, actor)
        actor_id_raw = typed_actor.id
        type_id = typed_actor.type_id
        role_name = self._extract_actor_role_name(typed_actor)
        transform = typed_actor.get_transform()
        return ActorState(
            actor_id=actor_id_raw,
            type_id=type_id,
            role_name=role_name,
            x=float(transform.location.x),
            y=float(transform.location.y),
            z=float(transform.location.z),
            yaw=float(transform.rotation.yaw),
        )

    def _extract_actor_role_name(self, actor: _ActorLike) -> str | None:
        attributes = actor.attributes
        raw_role_name = attributes.get("role_name")
        if not isinstance(raw_role_name, str):
            return None
        stripped = raw_role_name.strip()
        if not stripped:
            return None
        return stripped

    def _is_vehicle_actor(self, actor: Any) -> bool:
        raw_type_id = getattr(actor, "type_id", "")
        if not isinstance(raw_type_id, str):
            return False
        return raw_type_id.startswith("vehicle.")

    def _blueprint_id(self, blueprint: Any, fallback: str) -> str:
        raw_id = getattr(blueprint, "id", None)
        if isinstance(raw_id, str) and raw_id:
            return raw_id
        return fallback
