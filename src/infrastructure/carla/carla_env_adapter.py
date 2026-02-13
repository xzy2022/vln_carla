from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from queue import Empty
from typing import Final

import carla
import numpy as np
import numpy.typing as npt

from domain.entities import Observation, StepResult, VehicleCommand, VehicleState
from domain.errors import EnvConnectionError, EnvStepError
from domain.workzone_policy import (
    entered_zones,
    filter_entered_by_cooldown,
    inside_by_any_corner,
    validate_polygon,
)
from infrastructure.carla.carla_conversions import (
    lh_to_rh_location,
    lh_to_rh_rotation,
    lh_to_rh_velocity,
)
from infrastructure.carla.sensor_queue import SensorQueue
from usecases.episode_types import (
    EpisodeSpec,
    ResetInfo,
    StartGoalRef,
    TerminationReason,
    TransformSpec,
    ViolationThresholdsSpec,
    WorkZoneSeverity,
)
from usecases.ports.env_interface import EnvInterface

MAP_LAYER_ALIASES: Final[dict[str, str]] = {
    "vegetation": "foliage",
    "parkedvehicles": "parkedvehicles",
    "streetlights": "streetlights",
}


@dataclass(frozen=True)
class _RuntimeWorkZone:
    id: str
    polygon_world_xy: tuple[tuple[float, float], ...]
    severity: WorkZoneSeverity
    terminate_on_enter: bool
    cooldown_steps: int


class CarlaEnvAdapter(EnvInterface):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        timeout: float = 100.0,
        fixed_dt: float = 0.1,
        sensor_timeout: float = 2.0,
        retry_attempts: int = 30,
        map_name: str = "Town04",
        spectator_follow: bool = False,
        no_rendering_mode: bool = False,
        camera_width: int = 800,
        camera_height: int = 600,
        camera_sensor_tick: float | None = None,
        unload_map_layers: tuple[str, ...] = (),
    ) -> None:
        self._host = host
        self._port = port
        self._timeout = timeout
        self._fixed_dt = fixed_dt
        self._sensor_timeout = sensor_timeout
        self._retry_attempts = retry_attempts
        self._map_name = map_name
        self._spectator_follow = spectator_follow
        self._no_rendering_mode = no_rendering_mode
        self._camera_width = camera_width
        self._camera_height = camera_height
        self._camera_sensor_tick = camera_sensor_tick
        self._unload_map_layers = unload_map_layers

        self._client: carla.Client | None = None
        self._world: carla.World | None = None
        self._original_settings: carla.WorldSettings | None = None

        self._actors: list[carla.Actor] = []
        self._vehicle: carla.Vehicle | None = None
        self._camera: carla.Sensor | None = None
        self._collision_sensor: carla.Sensor | None = None
        self._lane_invasion_sensor: carla.Sensor | None = None
        self._camera_queue: SensorQueue[carla.Image] | None = None

        self._step_index = 0
        self._collision_count = 0
        self._lane_invasion_count = 0
        self._red_light_violation_count = 0
        self._stuck_count = 0
        self._goal_location: carla.Location | None = None
        self._goal_radius_m = 2.0
        self._shortest_path_length_m = 0.0
        self._workzones_by_id: dict[str, _RuntimeWorkZone] = {}
        self._workzone_order: tuple[str, ...] = ()
        self._prev_inside_zone_ids: tuple[str, ...] = ()
        self._last_enter_step_by_zone: dict[str, int] = {}
        self._cooldown_steps_by_zone: dict[str, int] = {}
        self._workzone_violation_count = 0
        self._workzone_hard_count = 0
        self._workzone_soft_count = 0
        self._violation_thresholds = ViolationThresholdsSpec()

    def reset(self, spec: EpisodeSpec) -> tuple[Observation, ResetInfo]:
        world = self._ensure_connected_world()
        self._apply_sync_settings(world)
        self._apply_map_layer_overrides(world)
        self._destroy_actors()
        self._reset_episode_state()
        self._spawn_ego_and_sensors(world, spec.start)
        self._configure_goal(world, spec.goal, spec.goal_radius_m)
        self._configure_workzones(spec)

        try:
            frame = world.tick()
            snapshot = world.get_snapshot()
            image = self._require_camera_queue().get_for_frame(frame, self._sensor_timeout)
        except Empty as exc:
            raise EnvStepError("Sensor frame timeout on reset") from exc
        except RuntimeError as exc:
            raise EnvStepError("CARLA tick failed during reset") from exc

        self._prev_inside_zone_ids = self._detect_inside_zone_ids(self._require_vehicle())
        violation_count = (
            self._lane_invasion_count
            + self._red_light_violation_count
            + self._workzone_violation_count
        )
        return self._build_observation(snapshot, image), ResetInfo(
            termination_reason=TerminationReason.ONGOING,
            termination_reasons=(),
            shortest_path_length_m=self._shortest_path_length_m,
            collision_count=self._collision_count,
            lane_invasion_count=self._lane_invasion_count,
            red_light_violation_count=self._red_light_violation_count,
            workzone_violation_count=self._workzone_violation_count,
            violation_count=violation_count,
            stuck_count=self._stuck_count,
        )

    def step(self, cmd: VehicleCommand) -> StepResult:
        world = self._require_world()
        vehicle = self._require_vehicle()
        camera_queue = self._require_camera_queue()

        control = _to_vehicle_control(cmd.clamped())
        try:
            vehicle.apply_control(control)
            if self._spectator_follow:
                self._update_spectator(world, vehicle)
            frame = world.tick()
            snapshot = world.get_snapshot()
            image = camera_queue.get_for_frame(frame, self._sensor_timeout)
        except Empty as exc:
            raise EnvStepError("Sensor frame timeout during step") from exc
        except RuntimeError as exc:
            raise EnvStepError("CARLA tick failed during step") from exc

        self._step_index += 1
        self._red_light_violation_count += self._count_red_light_violations_stub(vehicle)
        entered_zone_ids, effective_entered_zone_ids = self._update_workzone_violations(vehicle)

        obs = self._build_observation(snapshot, image)
        speed_mps = _speed_mps(vehicle.get_velocity())
        distance_to_goal_m, reached_goal = self._goal_distance_and_reached(vehicle)
        violation_count = (
            self._lane_invasion_count
            + self._red_light_violation_count
            + self._workzone_violation_count
        )
        violation_triggered = self._is_violation_termination_triggered(effective_entered_zone_ids)
        termination_reasons = _build_termination_reasons(
            reached_goal=reached_goal,
            collision_count=self._collision_count,
            violation_triggered=violation_triggered,
        )
        termination_reason = _choose_primary_termination_reason(termination_reasons)
        done = termination_reason != TerminationReason.ONGOING
        info: dict[str, object] = {
            "step_index": self._step_index,
            "termination_reason": termination_reason.value,
            "termination_reasons": [reason.value for reason in termination_reasons],
            "collision_count": self._collision_count,
            "lane_invasion_count": self._lane_invasion_count,
            "red_light_violation_count": self._red_light_violation_count,
            "workzone_violation_count": self._workzone_violation_count,
            "workzone_hard_violation_count": self._workzone_hard_count,
            "workzone_soft_violation_count": self._workzone_soft_count,
            "entered_workzone_ids": list(entered_zone_ids),
            "effective_entered_workzone_ids": list(effective_entered_zone_ids),
            "violation_count": violation_count,
            "stuck_count": self._stuck_count,
            "reached_goal": reached_goal,
            "speed_mps": speed_mps,
            "distance_to_goal_m": distance_to_goal_m,
        }
        return StepResult(obs=obs, reward=0.0, done=done, info=info)

    def close(self) -> None:
        self._destroy_actors()
        world = self._world
        original_settings = self._original_settings
        if world is not None and original_settings is not None:
            try:
                world.apply_settings(original_settings)
            except RuntimeError:
                pass

    def _ensure_connected_world(self) -> carla.World:
        if self._world is not None:
            return self._world

        last_error: Exception | None = None
        for attempt in range(self._retry_attempts):
            try:
                client = carla.Client(self._host, self._port)
                client.set_timeout(self._timeout)
                world = client.get_world()
                if self._map_name:
                    current_map_name = world.get_map().name.split("/")[-1]
                    if current_map_name != self._map_name:
                        client.load_world(self._map_name)
                        world = client.get_world()
                        self._original_settings = None
                self._client = client
                self._world = world
                return world
            except (RuntimeError, TimeoutError, OSError) as exc:
                last_error = exc
                time.sleep(0.5 * (2**attempt))

        raise EnvConnectionError("Failed to connect to CARLA server") from last_error

    def _require_world(self) -> carla.World:
        if self._world is None:
            raise EnvConnectionError("World is not initialized. Call reset() first.")
        return self._world

    def _require_vehicle(self) -> carla.Vehicle:
        if self._vehicle is None:
            raise EnvStepError("Vehicle is not initialized. Call reset() first.")
        return self._vehicle

    def _require_camera_queue(self) -> SensorQueue[carla.Image]:
        if self._camera_queue is None:
            raise EnvStepError("Camera queue is not initialized. Call reset() first.")
        return self._camera_queue

    def _apply_sync_settings(self, world: carla.World) -> None:
        if self._original_settings is None:
            self._original_settings = world.get_settings()

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self._fixed_dt
        settings.no_rendering_mode = self._no_rendering_mode
        world.apply_settings(settings)

    def _apply_map_layer_overrides(self, world: carla.World) -> None:
        if not self._unload_map_layers:
            return

        for name in self._unload_map_layers:
            layer = _to_map_layer(name)
            if layer is None:
                continue
            try:
                world.unload_map_layer(layer)
            except RuntimeError:
                pass

    def _spawn_ego_and_sensors(self, world: carla.World, start: StartGoalRef | None) -> None:
        blueprint_library = world.get_blueprint_library()
        candidates = blueprint_library.filter("vehicle.tesla.model3")
        if not candidates:
            candidates = blueprint_library.filter("vehicle.*")
        vehicle_bp = random.choice(candidates)

        spawn_point = self._resolve_spawn_transform(world, start)
        vehicle_actor = world.spawn_actor(vehicle_bp, spawn_point)
        if not isinstance(vehicle_actor, carla.Vehicle):
            raise EnvStepError("Spawned ego actor is not a vehicle")

        self._actors.append(vehicle_actor)
        self._vehicle = vehicle_actor
        if self._spectator_follow:
            self._update_spectator(world, vehicle_actor)

        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self._camera_width))
        camera_bp.set_attribute("image_size_y", str(self._camera_height))
        if self._camera_sensor_tick is not None:
            camera_bp.set_attribute("sensor_tick", str(self._camera_sensor_tick))
        camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15.0))
        camera_actor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle_actor)
        if not isinstance(camera_actor, carla.Sensor):
            raise EnvStepError("Spawned camera actor is not a sensor")

        self._actors.append(camera_actor)
        self._camera = camera_actor

        self._camera_queue = SensorQueue[carla.Image]()
        camera_actor.listen(self._camera_queue.push)

        collision_bp = blueprint_library.find("sensor.other.collision")
        collision_actor = world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=vehicle_actor,
        )
        if not isinstance(collision_actor, carla.Sensor):
            raise EnvStepError("Spawned collision actor is not a sensor")
        self._actors.append(collision_actor)
        self._collision_sensor = collision_actor
        collision_actor.listen(self._on_collision_event)

        lane_invasion_bp = blueprint_library.find("sensor.other.lane_invasion")
        lane_invasion_actor = world.spawn_actor(
            lane_invasion_bp,
            carla.Transform(),
            attach_to=vehicle_actor,
        )
        if not isinstance(lane_invasion_actor, carla.Sensor):
            raise EnvStepError("Spawned lane invasion actor is not a sensor")
        self._actors.append(lane_invasion_actor)
        self._lane_invasion_sensor = lane_invasion_actor
        lane_invasion_actor.listen(self._on_lane_invasion_event)

    def _update_spectator(self, world: carla.World, vehicle: carla.Vehicle) -> None:
        spectator = world.get_spectator()
        transform = vehicle.get_transform()
        backward = transform.get_forward_vector() * -10.0
        up = carla.Location(z=5.0)
        location = transform.location + backward + up

        rotation = transform.rotation
        rotation.pitch = -20.0
        spectator.set_transform(carla.Transform(location, rotation))

    def _destroy_actors(self) -> None:
        for actor in self._actors:
            try:
                if isinstance(actor, carla.Sensor):
                    actor.stop()
                actor.destroy()
            except RuntimeError:
                pass

        self._actors = []
        self._vehicle = None
        self._camera = None
        self._collision_sensor = None
        self._lane_invasion_sensor = None
        self._camera_queue = None

    def _build_observation(self, snapshot: carla.WorldSnapshot, image: carla.Image) -> Observation:
        vehicle = self._require_vehicle()

        rgb = _image_to_rgb_array(image)
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()

        state = VehicleState(
            position=lh_to_rh_location(transform.location),
            rotation_rpy=lh_to_rh_rotation(transform.rotation),
            velocity=lh_to_rh_velocity(velocity),
        )

        return Observation(
            rgb=rgb,
            ego=state,
            frame=snapshot.frame,
            timestamp=snapshot.timestamp.elapsed_seconds,
        )

    def _reset_episode_state(self) -> None:
        self._step_index = 0
        self._collision_count = 0
        self._lane_invasion_count = 0
        self._red_light_violation_count = 0
        self._workzone_violation_count = 0
        self._workzone_hard_count = 0
        self._workzone_soft_count = 0
        self._prev_inside_zone_ids = ()
        self._last_enter_step_by_zone = {}
        self._workzones_by_id = {}
        self._workzone_order = ()
        self._cooldown_steps_by_zone = {}
        self._violation_thresholds = ViolationThresholdsSpec()
        self._stuck_count = 0
        self._goal_location = None
        self._goal_radius_m = 2.0
        self._shortest_path_length_m = 0.0

    def _configure_goal(
        self,
        world: carla.World,
        goal: StartGoalRef | None,
        goal_radius_m: float,
    ) -> None:
        self._goal_radius_m = max(0.1, goal_radius_m)
        if goal is None:
            self._goal_location = None
            self._shortest_path_length_m = 0.0
            return

        goal_transform = self._resolve_ref_transform(world, goal)
        self._goal_location = goal_transform.location
        vehicle = self._require_vehicle()
        self._shortest_path_length_m = _distance_between_locations(
            vehicle.get_transform().location,
            self._goal_location,
        )

    def _configure_workzones(self, spec: EpisodeSpec) -> None:
        if spec.workzone_default_cooldown_steps < 0:
            raise EnvStepError("workzone_default_cooldown_steps must be >= 0")
        _validate_violation_thresholds(spec.violation_thresholds)

        workzones_by_id: dict[str, _RuntimeWorkZone] = {}
        cooldown_steps_by_zone: dict[str, int] = {}
        zone_order: list[str] = []
        for workzone in spec.workzones:
            if workzone.id in workzones_by_id:
                raise EnvStepError(f"Duplicated workzone id: {workzone.id}")

            raw_polygon = tuple((point.x, point.y) for point in workzone.polygon_world_xy)
            try:
                polygon_world_xy = validate_polygon(raw_polygon)
            except ValueError as exc:
                raise EnvStepError(
                    f"Invalid polygon for workzone '{workzone.id}': {exc}",
                ) from exc

            cooldown_steps = (
                spec.workzone_default_cooldown_steps
                if workzone.cooldown_steps is None
                else workzone.cooldown_steps
            )
            if cooldown_steps < 0:
                raise EnvStepError(f"workzone '{workzone.id}' cooldown_steps must be >= 0")

            runtime_zone = _RuntimeWorkZone(
                id=workzone.id,
                polygon_world_xy=polygon_world_xy,
                severity=workzone.severity,
                terminate_on_enter=workzone.terminate_on_enter,
                cooldown_steps=cooldown_steps,
            )
            workzones_by_id[workzone.id] = runtime_zone
            cooldown_steps_by_zone[workzone.id] = cooldown_steps
            zone_order.append(workzone.id)

        self._workzones_by_id = workzones_by_id
        self._cooldown_steps_by_zone = cooldown_steps_by_zone
        self._workzone_order = tuple(zone_order)
        self._violation_thresholds = spec.violation_thresholds

    def _update_workzone_violations(
        self,
        vehicle: carla.Vehicle,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        curr_inside_zone_ids = self._detect_inside_zone_ids(vehicle)
        entered_zone_ids = entered_zones(self._prev_inside_zone_ids, curr_inside_zone_ids)
        effective_entered_zone_ids = filter_entered_by_cooldown(
            entered_zone_ids,
            current_step=self._step_index,
            last_enter_step_by_zone=self._last_enter_step_by_zone,
            cooldown_steps_by_zone=self._cooldown_steps_by_zone,
        )
        for zone_id in effective_entered_zone_ids:
            self._last_enter_step_by_zone[zone_id] = self._step_index
            zone = self._workzones_by_id[zone_id]
            if zone.severity == WorkZoneSeverity.HARD:
                self._workzone_hard_count += 1
            else:
                self._workzone_soft_count += 1
        self._workzone_violation_count += len(effective_entered_zone_ids)
        self._prev_inside_zone_ids = curr_inside_zone_ids
        return entered_zone_ids, effective_entered_zone_ids

    def _is_violation_termination_triggered(
        self,
        effective_entered_zone_ids: tuple[str, ...],
    ) -> bool:
        if any(
            self._workzones_by_id[zone_id].terminate_on_enter
            for zone_id in effective_entered_zone_ids
        ):
            return True

        thresholds = self._violation_thresholds
        if _is_threshold_reached(self._lane_invasion_count, thresholds.lane):
            return True
        if _is_threshold_reached(self._red_light_violation_count, thresholds.red_light):
            return True
        if _is_threshold_reached(self._workzone_hard_count, thresholds.workzone_by_severity.hard):
            return True
        if _is_threshold_reached(self._workzone_soft_count, thresholds.workzone_by_severity.soft):
            return True
        return False

    def _detect_inside_zone_ids(self, vehicle: carla.Vehicle) -> tuple[str, ...]:
        if not self._workzone_order:
            return ()

        corners_xy = _vehicle_bottom_corners_xy(vehicle)
        inside_ids: list[str] = []
        for zone_id in self._workzone_order:
            zone = self._workzones_by_id[zone_id]
            if inside_by_any_corner(corners_xy, zone.polygon_world_xy):
                inside_ids.append(zone_id)
        return tuple(inside_ids)

    def _resolve_spawn_transform(
        self,
        world: carla.World,
        start: StartGoalRef | None,
    ) -> carla.Transform:
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise EnvStepError("No spawn points available on the map")
        if start is None:
            return random.choice(spawn_points)
        return self._resolve_ref_transform(world, start, spawn_points=spawn_points)

    def _resolve_ref_transform(
        self,
        world: carla.World,
        ref: StartGoalRef,
        *,
        spawn_points: list[carla.Transform] | None = None,
    ) -> carla.Transform:
        if isinstance(ref, int):
            points = spawn_points
            if points is None:
                points = world.get_map().get_spawn_points()
            if not points:
                raise EnvStepError("No spawn points available on the map")
            if ref < 0 or ref >= len(points):
                raise EnvStepError(f"Spawn point id out of range: {ref}")
            return points[ref]
        if isinstance(ref, TransformSpec):
            return carla.Transform(
                location=carla.Location(x=ref.x, y=ref.y, z=ref.z),
                rotation=carla.Rotation(roll=ref.roll, pitch=ref.pitch, yaw=ref.yaw),
            )
        raise EnvStepError(f"Unsupported start/goal reference type: {type(ref)!r}")

    def _goal_distance_and_reached(self, vehicle: carla.Vehicle) -> tuple[float, bool]:
        if self._goal_location is None:
            return float("inf"), False
        distance_to_goal_m = _distance_between_locations(
            vehicle.get_transform().location,
            self._goal_location,
        )
        return distance_to_goal_m, distance_to_goal_m <= self._goal_radius_m

    def _on_collision_event(self, _: object) -> None:
        self._collision_count += 1

    def _on_lane_invasion_event(self, _: object) -> None:
        self._lane_invasion_count += 1

    def _count_red_light_violations_stub(self, vehicle: carla.Vehicle) -> int:
        del vehicle
        return 0


def _image_to_rgb_array(image: carla.Image) -> npt.NDArray[np.uint8]:
    bgra: npt.NDArray[np.uint8] = np.frombuffer(image.raw_data, dtype=np.uint8)
    reshaped: npt.NDArray[np.uint8] = bgra.reshape((image.height, image.width, 4))
    bgr: npt.NDArray[np.uint8] = reshaped[:, :, :3]
    rgb: npt.NDArray[np.uint8] = bgr[:, :, ::-1]
    return rgb


def _to_vehicle_control(cmd: VehicleCommand) -> carla.VehicleControl:
    return carla.VehicleControl(
        throttle=cmd.throttle,
        steer=cmd.steer,
        brake=cmd.brake,
    )


def _speed_mps(velocity: carla.Vector3D) -> float:
    return math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z)


def _distance_between_locations(a: carla.Location, b: carla.Location) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _build_termination_reasons(
    *,
    reached_goal: bool,
    collision_count: int,
    violation_triggered: bool,
) -> tuple[TerminationReason, ...]:
    reasons: list[TerminationReason] = []
    if reached_goal:
        reasons.append(TerminationReason.SUCCESS)
    if collision_count > 0:
        reasons.append(TerminationReason.COLLISION)
    if violation_triggered:
        reasons.append(TerminationReason.VIOLATION)
    return tuple(reasons)


def _is_threshold_reached(count: int, threshold: int | None) -> bool:
    if threshold is None:
        return False
    return count >= threshold


def _validate_violation_thresholds(thresholds: ViolationThresholdsSpec) -> None:
    if thresholds.lane is not None and thresholds.lane < 0:
        raise EnvStepError("lane violation threshold must be >= 0")
    if thresholds.red_light is not None and thresholds.red_light < 0:
        raise EnvStepError("red_light violation threshold must be >= 0")
    if thresholds.workzone_by_severity.hard < 0:
        raise EnvStepError("workzone hard threshold must be >= 0")
    if thresholds.workzone_by_severity.soft < 0:
        raise EnvStepError("workzone soft threshold must be >= 0")


def _vehicle_bottom_corners_xy(vehicle: carla.Vehicle) -> tuple[tuple[float, float], ...]:
    transform = vehicle.get_transform()
    vertices = vehicle.bounding_box.get_world_vertices(transform)
    if len(vertices) < 4:
        raise EnvStepError("vehicle bounding box is missing vertices")
    lowest_four = sorted(vertices, key=lambda vertex: float(vertex.z))[:4]
    corners_xy = tuple((float(vertex.x), float(vertex.y)) for vertex in lowest_four)
    if len(corners_xy) < 4:
        raise EnvStepError("vehicle bounding box bottom corners are invalid")
    return corners_xy


def _choose_primary_termination_reason(
    reasons: tuple[TerminationReason, ...],
) -> TerminationReason:
    if not reasons:
        return TerminationReason.ONGOING

    priority = {
        TerminationReason.ERROR: 0,
        TerminationReason.COLLISION: 1,
        TerminationReason.VIOLATION: 2,
        TerminationReason.STUCK: 3,
        TerminationReason.TIMEOUT: 4,
        TerminationReason.SUCCESS: 5,
        TerminationReason.ONGOING: 99,
    }
    return min(reasons, key=lambda reason: priority[reason])


def _to_map_layer(name: str) -> carla.MapLayer | None:
    normalized = name.strip().lower().replace(" ", "").replace("_", "")
    if not normalized:
        return None

    normalized = MAP_LAYER_ALIASES.get(normalized, normalized)

    if normalized == "buildings":
        return carla.MapLayer.Buildings
    if normalized == "decals":
        return carla.MapLayer.Decals
    if normalized == "foliage":
        return carla.MapLayer.Foliage
    if normalized == "ground":
        return carla.MapLayer.Ground
    if normalized == "parkedvehicles":
        return carla.MapLayer.ParkedVehicles
    if normalized == "particles":
        return carla.MapLayer.Particles
    if normalized == "props":
        return carla.MapLayer.Props
    if normalized == "streetlights":
        return carla.MapLayer.StreetLights
    if normalized == "walls":
        return carla.MapLayer.Walls
    if normalized == "all":
        return carla.MapLayer.All

    return None
