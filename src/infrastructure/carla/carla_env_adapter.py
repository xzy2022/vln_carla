from __future__ import annotations

import random
import time
from queue import Empty

import numpy as np

import carla # type: ignore

from infrastructure.carla.carla_conversions import (
    lh_to_rh_location,
    lh_to_rh_rotation,
    lh_to_rh_velocity,
)
from infrastructure.carla.sensor_queue import SensorQueue
from domain.entities import Observation, StepResult, VehicleCommand, VehicleState
from domain.errors import EnvConnectionError, EnvStepError
from usecases.ports.env_interface import EnvInterface


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
        self._camera_queue: SensorQueue | None = None

    def reset(self) -> Observation:
        self._ensure_connected()
        self._apply_sync_settings()
        self._apply_map_layer_overrides()
        self._destroy_actors()
        self._spawn_ego_and_sensors()

        try:
            frame = self._world.tick()
            snapshot = self._world.get_snapshot()
            image = self._camera_queue.get_for_frame(frame, self._sensor_timeout)
        except Empty as exc:
            raise EnvStepError("Sensor frame timeout on reset") from exc
        except RuntimeError as exc:
            raise EnvStepError("CARLA tick failed during reset") from exc

        obs = self._build_observation(snapshot, image)
        return obs

    def step(self, cmd: VehicleCommand) -> StepResult:
        if self._world is None or self._vehicle is None or self._camera_queue is None:
            raise EnvStepError("Environment not initialized. Call reset() first.")

        control = _to_vehicle_control(cmd.clamped())
        try:
            self._vehicle.apply_control(control)
            if self._spectator_follow:
                self._update_spectator(self._vehicle)
            frame = self._world.tick()
            snapshot = self._world.get_snapshot()
            image = self._camera_queue.get_for_frame(frame, self._sensor_timeout)
        except Empty as exc:
            raise EnvStepError("Sensor frame timeout during step") from exc
        except RuntimeError as exc:
            raise EnvStepError("CARLA tick failed during step") from exc

        obs = self._build_observation(snapshot, image)
        return StepResult(obs=obs, reward=0.0, done=False, info={})

    def close(self) -> None:
        self._destroy_actors()
        if self._world and self._original_settings:
            try:
                self._world.apply_settings(self._original_settings)
            except RuntimeError:
                pass

    def _ensure_connected(self) -> None:
        if self._client and self._world:
            return

        last_error: Exception | None = None
        for attempt in range(self._retry_attempts):
            try:
                client = carla.Client(self._host, self._port)
                client.set_timeout(self._timeout)
                world = client.get_world()
                if self._map_name:
                    current_map = world.get_map().name
                    current_map_name = current_map.split("/")[-1]
                    if current_map_name != self._map_name:
                        client.load_world(self._map_name)
                        world = client.get_world()
                        self._original_settings = None
                self._client = client
                self._world = world
                return
            except (RuntimeError, TimeoutError, OSError) as exc:
                last_error = exc
                time.sleep(0.5 * (2**attempt))

        raise EnvConnectionError("Failed to connect to CARLA server") from last_error

    def _apply_sync_settings(self) -> None:
        if self._world is None:
            raise EnvConnectionError("World not available for settings")

        if self._original_settings is None:
            self._original_settings = self._world.get_settings()

        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self._fixed_dt
        settings.no_rendering_mode = self._no_rendering_mode
        self._world.apply_settings(settings)

    def _apply_map_layer_overrides(self) -> None:
        if not self._unload_map_layers or self._world is None:
            return

        for name in self._unload_map_layers:
            layer = _to_map_layer(name)
            if layer is None:
                continue
            try:
                self._world.unload_map_layer(layer)
            except RuntimeError:
                pass

    def _spawn_ego_and_sensors(self) -> None:
        """
        鍒濆鍖栨棤浜鸿溅锛圗go Vehicle锛夊強鍏朵紶鎰熷櫒绯荤粺銆?        """
        if self._world is None:
            raise EnvConnectionError("World not available for spawning")


        # 钃濆浘绛涢€夛細
        # 浠?CARLA 鐨勮摑鍥惧簱涓瓫閫夊嚭鐗瑰畾鐨勮溅鍨嬶紙浼樺厛閫夋嫨 tesla.model3锛?        # 濡傛灉娌℃湁鍒欓殢鏈洪€夋嫨浠绘剰杞﹁締钃濆浘锛夈€?        blueprint_library = self._world.get_blueprint_library()
        candidates = blueprint_library.filter("vehicle.tesla.model3")
        if not candidates:
            candidates = blueprint_library.filter("vehicle.*")
        vehicle_bp = random.choice(candidates)

        # 浣嶇疆閫夋嫨锛?        # 浠庡綋鍓嶅湴鍥剧殑鎵€鏈夊悎娉曠敓鎴愮偣锛圫pawn Points锛変腑闅忔満鎶藉彇涓€涓綅缃€?        spawn_points = self._world.get_map().get_spawn_points()
        if not spawn_points:
            raise EnvStepError("No spawn points available on the map")

        spawn_point = random.choice(spawn_points)
        vehicle = self._world.spawn_actor(vehicle_bp, spawn_point)

        # 瀹炰綋鍒涘缓锛?        # 鍦ㄩ€夊畾浣嶇疆鐢熸垚杞﹁締锛屽苟灏嗗叾璁板綍鍦?self._actors 鍒楄〃涓互渚垮悗缁粺涓€閿€姣併€?        self._actors.append(vehicle)
        self._vehicle = vehicle
        if self._spectator_follow:
            self._update_spectator(vehicle)

        # 浼犳劅鍣ㄥ垱寤猴細
        # 瀹氫箟鐩告満鐩稿浜庤溅杈嗕腑蹇冪殑浣嶇疆鍋忕疆锛坸=-5.5 绫? z=2.8 绫筹級鍜屼刊浠拌锛?15搴︼級锛?        # 瀹炵幇绫讳技鈥滆溅杞藉悗涓婃柟鈥濈殑瑙嗛噹銆?        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self._camera_width))
        camera_bp.set_attribute("image_size_y", str(self._camera_height))
        if self._camera_sensor_tick is not None:
            camera_bp.set_attribute("sensor_tick", str(self._camera_sensor_tick))
        camera_transform = carla.Transform(
            carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)
        )
        # 鐖跺瓙缁戝畾锛氫娇鐢?attach_to=vehicle 鍙傛暟灏嗙浉鏈虹墿鐞嗘寕杞藉湪鏃犱汉杞︿笂
        camera = self._world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        self._actors.append(camera)
        self._camera = camera

        self._camera_queue = SensorQueue()
        camera.listen(self._camera_queue.push)

    def _update_spectator(self, vehicle: carla.Vehicle) -> None:
        """
        渚濇嵁杞﹀瓙鐨勫疄鏃舵湭鐭ユ洿鏂拌瀵熻€呯殑浣嶇疆
        """
        if self._world is None:
            return

        spectator = self._world.get_spectator()
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
        if self._vehicle is None:
            raise EnvStepError("Vehicle not available for observation")

        rgb = _image_to_rgb_array(image)
        transform = self._vehicle.get_transform()
        velocity = self._vehicle.get_velocity()

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


def _image_to_rgb_array(image: carla.Image) -> np.ndarray:
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array


def _to_vehicle_control(cmd: VehicleCommand) -> carla.VehicleControl:
    return carla.VehicleControl(
        throttle=cmd.throttle,
        steer=cmd.steer,
        brake=cmd.brake,
    )


def _to_map_layer(name: str) -> carla.MapLayer | None:
    normalized = name.strip()
    if not normalized:
        return None
    alias_map = {
        "vegetation": "Foliage",
        "parkedvehicles": "ParkedVehicles",
        "streetlights": "StreetLights",
    }
    lower = normalized.lower().replace(" ", "").replace("_", "")
    if lower in alias_map:
        normalized = alias_map[lower]
    # Accept either canonical CARLA names or common aliases.
    candidates = [
        normalized,
        normalized.replace(" ", ""),
        normalized.replace("_", ""),
        normalized.title().replace(" ", ""),
    ]
    for candidate in candidates:
        if hasattr(carla.MapLayer, candidate):
            return getattr(carla.MapLayer, candidate)
    return None

