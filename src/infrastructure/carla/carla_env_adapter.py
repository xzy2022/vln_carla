from __future__ import annotations

import random
import time
from queue import Empty
from typing import Final

import carla
import numpy as np
import numpy.typing as npt

from domain.entities import Observation, StepResult, VehicleCommand, VehicleState
from domain.errors import EnvConnectionError, EnvStepError
from infrastructure.carla.carla_conversions import (
    lh_to_rh_location,
    lh_to_rh_rotation,
    lh_to_rh_velocity,
)
from infrastructure.carla.sensor_queue import SensorQueue
from usecases.ports.env_interface import EnvInterface


class CarlaEnvAdapter(EnvInterface):
    _MAP_LAYER_ALIASES: Final[dict[str, str]] = {
        "vegetation": "foliage",
        "parkedvehicles": "parkedvehicles",
        "streetlights": "streetlights",
    }

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
        self._camera_queue: SensorQueue[carla.Image] | None = None

    def reset(self) -> Observation:
        world = self._ensure_connected_world()
        self._apply_sync_settings(world)
        self._apply_map_layer_overrides(world)
        self._destroy_actors()
        self._spawn_ego_and_sensors(world)

        try:
            frame = world.tick()
            snapshot = world.get_snapshot()
            image = self._require_camera_queue().get_for_frame(frame, self._sensor_timeout)
        except Empty as exc:
            raise EnvStepError("Sensor frame timeout on reset") from exc
        except RuntimeError as exc:
            raise EnvStepError("CARLA tick failed during reset") from exc

        return self._build_observation(snapshot, image)

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

        obs = self._build_observation(snapshot, image)
        return StepResult(obs=obs, reward=0.0, done=False, info={})

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

    def _spawn_ego_and_sensors(self, world: carla.World) -> None:
        blueprint_library = world.get_blueprint_library()
        candidates = blueprint_library.filter("vehicle.tesla.model3")
        if not candidates:
            candidates = blueprint_library.filter("vehicle.*")
        vehicle_bp = random.choice(candidates)

        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise EnvStepError("No spawn points available on the map")

        spawn_point = random.choice(spawn_points)
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


def _to_map_layer(name: str) -> carla.MapLayer | None:
    normalized = name.strip().lower().replace(" ", "").replace("_", "")
    if not normalized:
        return None

    normalized = CarlaEnvAdapter._MAP_LAYER_ALIASES.get(normalized, normalized)

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
