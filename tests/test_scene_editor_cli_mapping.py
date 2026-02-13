from __future__ import annotations

from adapters.cli.scene_editor_cli import parse_args, run_scene_editor_cli
from adapters.cli.scene_editor_keyboard import (
    VK_B,
    VK_C,
    VK_ESC,
    VK_UP,
    KeyboardInterface,
)
from usecases.scene_editor.dtos import (
    ActorState,
    CleanupResult,
    ConnectRequest,
    ConnectionState,
    FollowMode,
    FollowRequest,
    FollowState,
    QuickSpawnRequest,
    SpawnCategory,
    SpawnRequest,
    SpawnResult,
    SpectatorDelta,
    SpectatorState,
)
from usecases.scene_editor.input_port import SceneEditorInputPort


class FakeKeyboard(KeyboardInterface):
    def __init__(self, frames: list[set[int]]) -> None:
        self._frames = frames
        self._index = -1

    def is_pressed(self, vk_code: int) -> bool:
        if vk_code == VK_ESC:
            self._index += 1
            if self._index >= len(self._frames):
                self._index = len(self._frames) - 1
        frame = self._frames[self._index]
        return vk_code in frame


class FakeSceneEditorUseCase(SceneEditorInputPort):
    def __init__(self) -> None:
        self.connect_requests: list[ConnectRequest] = []
        self.follow_requests: list[FollowRequest | None] = []
        self.move_calls: list[SpectatorDelta] = []
        self.vehicle_quick_spawns: list[QuickSpawnRequest] = []
        self.obstacle_quick_spawns: list[QuickSpawnRequest] = []
        self.close_destroy_flags: list[bool] = []
        self._spectator = SpectatorState(x=0.0, y=0.0, z=10.0, pitch=-90.0, yaw=0.0, roll=0.0)
        self._next_actor_id = 10

    def connect(self, req: ConnectRequest) -> ConnectionState:
        self.connect_requests.append(req)
        return ConnectionState(
            host=req.host,
            port=req.port,
            map_name="Carla/Maps/Town10HD_Opt",
            synchronous_mode=False,
            available_maps=("Carla/Maps/Town10HD_Opt",),
        )

    def spawn_vehicle_at(self, req: SpawnRequest) -> SpawnResult:
        del req
        raise AssertionError("spawn_vehicle_at is not expected in this test")

    def spawn_obstacle_at(self, req: SpawnRequest) -> SpawnResult:
        del req
        raise AssertionError("spawn_obstacle_at is not expected in this test")

    def bind_topdown_follow(self, req: FollowRequest | None = None) -> FollowState:
        self.follow_requests.append(req)
        return FollowState(bound=True, mode=FollowMode.EXPLICIT, target_actor=None, message="bound")

    def move_spectator(self, delta: SpectatorDelta) -> SpectatorState:
        self.move_calls.append(delta)
        self._spectator = self._spectator.moved(delta)
        return self._spectator

    def spawn_vehicle_at_current_xy(self, req: QuickSpawnRequest) -> SpawnResult:
        self.vehicle_quick_spawns.append(req)
        actor = ActorState(
            actor_id=self._next_actor_id,
            type_id=req.blueprint_id or "vehicle.mini.cooper",
            role_name="scene_editor_vehicle_1",
            x=self._spectator.x,
            y=self._spectator.y,
            z=self._spectator.z - 0.5,
            yaw=req.yaw,
        )
        self._next_actor_id += 1
        return SpawnResult(
            category=SpawnCategory.VEHICLE,
            success=True,
            requested_blueprint_id=req.blueprint_id,
            used_blueprint_id=actor.type_id,
            actor=actor,
            ground_z_estimate=self._spectator.z - 1.0,
            message="spawned",
        )

    def spawn_obstacle_at_current_xy(self, req: QuickSpawnRequest) -> SpawnResult:
        self.obstacle_quick_spawns.append(req)
        actor = ActorState(
            actor_id=self._next_actor_id,
            type_id=req.blueprint_id or "static.prop.barrel",
            role_name=None,
            x=self._spectator.x,
            y=self._spectator.y,
            z=self._spectator.z - 0.6,
            yaw=req.yaw,
        )
        self._next_actor_id += 1
        return SpawnResult(
            category=SpawnCategory.OBSTACLE,
            success=True,
            requested_blueprint_id=req.blueprint_id,
            used_blueprint_id=actor.type_id,
            actor=actor,
            ground_z_estimate=self._spectator.z - 1.0,
            message="spawned",
        )

    def tick_follow(self) -> SpectatorState | None:
        return None

    def close(self, destroy_spawned: bool) -> CleanupResult:
        self.close_destroy_flags.append(destroy_spawned)
        return CleanupResult(destroyed_count=2 if destroy_spawned else 0, failed_count=0)


class FakePerfCounter:
    def __init__(self, step: float) -> None:
        self._value = 0.0
        self._step = step

    def __call__(self) -> float:
        self._value += self._step
        return self._value


def test_cli_maps_key_events_to_usecase_calls_with_edge_trigger() -> None:
    args = parse_args(
        [
            "--tick-hz",
            "20",
            "--speed",
            "1.5",
            "--destroy-spawned-on-exit",
        ]
    )
    usecase = FakeSceneEditorUseCase()
    keyboard = FakeKeyboard(
        frames=[
            {VK_UP, VK_C},
            {VK_UP, VK_C},
            {VK_B},
            {VK_ESC},
        ]
    )
    perf_counter = FakePerfCounter(step=0.05)

    exit_code = run_scene_editor_cli(
        args=args,
        usecase=usecase,
        keyboard=keyboard,
        perf_counter=perf_counter,
        sleep=lambda _: None,
    )

    assert exit_code == 0
    assert len(usecase.connect_requests) == 1
    assert len(usecase.move_calls) >= 2
    assert len(usecase.vehicle_quick_spawns) == 1
    assert len(usecase.obstacle_quick_spawns) == 1
    assert usecase.close_destroy_flags == [True]


def test_cli_sends_follow_request_when_explicit_target_is_provided() -> None:
    args = parse_args(
        [
            "--follow-role-name",
            "hero",
            "--follow-actor-id",
            "42",
        ]
    )
    usecase = FakeSceneEditorUseCase()
    keyboard = FakeKeyboard(frames=[{VK_ESC}])
    perf_counter = FakePerfCounter(step=0.1)

    run_scene_editor_cli(
        args=args,
        usecase=usecase,
        keyboard=keyboard,
        perf_counter=perf_counter,
        sleep=lambda _: None,
    )

    assert usecase.follow_requests == [FollowRequest(role_name="hero", actor_id=42)]
