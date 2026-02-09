from __future__ import annotations

import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List all selectable vehicle blueprints from a running UE5 CARLA world."
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="CARLA server host.")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server RPC port.")
    parser.add_argument("--timeout", type=float, default=10.0, help="Client timeout seconds.")
    parser.add_argument(
        "--filter",
        type=str,
        default="vehicle.*",
        help="Blueprint wildcard filter (default: vehicle.*).",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    try:
        import carla  # type: ignore
    except ModuleNotFoundError:
        print("[error] Python package 'carla' is not installed in this environment.")
        return 1

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    try:
        world = client.get_world()
    except RuntimeError as exc:
        print(f"[error] failed to connect to CARLA server: {exc}")
        return 1

    vehicle_blueprints = sorted(
        world.get_blueprint_library().filter(args.filter),
        key=lambda bp: bp.id,
    )

    print(f"[info] connected to {args.host}:{args.port}")
    print(f"[info] map: {world.get_map().name}")
    print(f"[info] filter: {args.filter}")
    print(f"[info] total blueprints: {len(vehicle_blueprints)}")

    if not vehicle_blueprints:
        print("[warn] no vehicle blueprints found under this filter.")
        return 0

    for index, blueprint in enumerate(vehicle_blueprints, start=1):
        print(f"{index:03d}. {blueprint.id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
