from __future__ import annotations

import random
import time
from queue import Empty

import numpy as np

import carla # type: ignore

from vln_carla.adapters.carla_conversions import (
    lh_to_rh_location,
    lh_to_rh_rotation,
    lh_to_rh_velocity,
)
from vln_carla.adapters.sensor_queue import SensorQueue
from vln_carla.domain.entities import Observation, StepResult, VehicleCommand, VehicleState
from vln_carla.domain.errors import EnvConnectionError, EnvStepError
from vln_carla.ports.env_interface import EnvInterface


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
    ) -> None:
        self._host = host
        self._port = port
        self._timeout = timeout
        self._fixed_dt = fixed_dt
        self._sensor_timeout = sensor_timeout
        self._retry_attempts = retry_attempts
        self._map_name = map_name
        self._spectator_follow = spectator_follow

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
        self._world.apply_settings(settings)

    def _spawn_ego_and_sensors(self) -> None:
        """
        初始化无人车（Ego Vehicle）及其传感器系统。
        """
        if self._world is None:
            raise EnvConnectionError("World not available for spawning")


        # 蓝图筛选：
        # 从 CARLA 的蓝图库中筛选出特定的车型（优先选择 tesla.model3，
        # 如果没有则随机选择任意车辆蓝图）。
        blueprint_library = self._world.get_blueprint_library()
        candidates = blueprint_library.filter("vehicle.tesla.model3")
        if not candidates:
            candidates = blueprint_library.filter("vehicle.*")
        vehicle_bp = random.choice(candidates)

        # 位置选择：
        # 从当前地图的所有合法生成点（Spawn Points）中随机抽取一个位置。
        spawn_points = self._world.get_map().get_spawn_points()
        if not spawn_points:
            raise EnvStepError("No spawn points available on the map")

        spawn_point = random.choice(spawn_points)
        vehicle = self._world.spawn_actor(vehicle_bp, spawn_point)

        # 实体创建：
        # 在选定位置生成车辆，并将其记录在 self._actors 列表中以便后续统一销毁。
        self._actors.append(vehicle)
        self._vehicle = vehicle
        if self._spectator_follow:
            self._update_spectator(vehicle)

        # 传感器创建：
        # 定义相机相对于车辆中心的位置偏置（x=-5.5 米, z=2.8 米）和俯仰角（-15度），
        # 实现类似“车载后上方”的视野。
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_transform = carla.Transform(
            carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)
        )
        # 父子绑定：使用 attach_to=vehicle 参数将相机物理挂载在无人车上
        camera = self._world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        self._actors.append(camera)
        self._camera = camera

        self._camera_queue = SensorQueue()
        camera.listen(self._camera_queue.push)

    def _update_spectator(self, vehicle: carla.Vehicle) -> None:
        """
        依据车子的实时未知更新观察者的位置
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
