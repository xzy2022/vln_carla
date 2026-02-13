from __future__ import annotations

from collections.abc import Sequence

from adapters.cli.scene_editor_cli import parse_args, run_scene_editor_cli
from adapters.cli.scene_editor_keyboard import WindowsKeyboard
from infrastructure.carla.scene_editor_gateway import CarlaSceneEditorGateway
from usecases.scene_editor.usecase import SceneEditorUseCase


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    usecase = SceneEditorUseCase(gateway=CarlaSceneEditorGateway())
    keyboard = WindowsKeyboard()

    try:
        return run_scene_editor_cli(args=args, usecase=usecase, keyboard=keyboard)
    except RuntimeError as exc:
        print(f"[error] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
