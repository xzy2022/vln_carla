from __future__ import annotations

import argparse
from pathlib import PurePosixPath


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Connect to a running UE5 CARLA server and switch map. "
            "Relevant docs: Docs_Carla_UE5/core_world.md, "
            "Docs_Carla_UE5/tuto_first_steps.md, "
            "PythonAPI_Carla_UE5/python_api.md"
        )
    )
    parser.add_argument("--host", type=str, default="localhost", help="CARLA server host.")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server RPC port.")
    parser.add_argument("--timeout", type=float, default=10.0, help="Client timeout seconds.")
    parser.add_argument(
        "--map",
        type=str,
        default=None,
        help=(
            "Target map (full path or short name, e.g. /Game/Carla/Maps/Town01 or Town01). "
            "If omitted, auto-selects one map different from current."
        ),
    )
    parser.add_argument(
        "--keep-settings",
        action="store_true",
        help="Keep current world settings when loading map (reset_settings=False).",
    )
    parser.add_argument(
        "--list-maps-only",
        action="store_true",
        help="Only list available maps and exit.",
    )
    return parser


def short_map_name(map_path: str) -> str:
    return PurePosixPath(map_path).name


def resolve_target_map(requested: str, available_maps: list[str]) -> str | None:
    normalized = requested.strip().strip("/")
    requested_lower = normalized.lower()

    exact_matches = [m for m in available_maps if m.lower().strip("/") == requested_lower]
    if exact_matches:
        return exact_matches[0]

    by_suffix = [m for m in available_maps if m.lower().endswith("/" + requested_lower)]
    if len(by_suffix) == 1:
        return by_suffix[0]

    by_short_name = [m for m in available_maps if short_map_name(m).lower() == requested_lower]
    if len(by_short_name) == 1:
        return by_short_name[0]

    return None


def pick_alternative_map(current_map: str, available_maps: list[str]) -> str | None:
    current_short = short_map_name(current_map).lower()
    for map_path in sorted(available_maps):
        if short_map_name(map_path).lower() != current_short:
            return map_path
    return None


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        import carla  # type: ignore
    except ModuleNotFoundError:
        print("[error] Python package 'carla' is not installed in this environment.")
        return 1

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    world = client.get_world()
    current_map = world.get_map().name
    available_maps = sorted(client.get_available_maps())

    print(f"[info] connected to {args.host}:{args.port}")
    print(f"[info] current map: {current_map}")
    print("[info] available maps:")
    for map_path in available_maps:
        print(f"  - {map_path}")

    if args.list_maps_only:
        return 0

    if args.map is not None:
        target_map = resolve_target_map(args.map, available_maps)
        if target_map is None:
            print(f"[error] map '{args.map}' not found on server.")
            return 1
    else:
        target_map = pick_alternative_map(current_map, available_maps)
        if target_map is None:
            print("[error] could not auto-select a different map.")
            return 1
        print(f"[info] auto-selected map: {target_map}")

    if short_map_name(target_map).lower() == short_map_name(current_map).lower():
        print("[info] target map is already active, skip load_world.")
        return 0

    print(f"[info] loading map: {target_map}")
    print("[info] load_world will destroy actors in current world.")
    new_world = client.load_world(target_map, reset_settings=not args.keep_settings)
    print(f"[info] switched to map: {new_world.get_map().name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
