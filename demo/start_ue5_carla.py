from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch UE5 CARLA server from Python. "
            "Relevant docs: Docs_Carla_UE5/start_quickstart.md, "
            "Docs_Carla_UE5/adv_rendering_options.md, "
            "Docs_Carla_UE5/adv_traffic_manager.md"
        )
    )
    parser.add_argument(
        "--carla-exe",
        type=str,
        default=r"D:\Workspace\02_Playground\Carla-0.10.0-Win64-Shipping\CarlaUnreal.exe",
        help=(
            "Path to CarlaUnreal executable. "
            "If omitted, uses CARLA_UE5_EXE env var or CarlaUnreal(.exe) in current directory."
        ),
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for startup port check.")
    parser.add_argument("--port", type=int, default=2000, help="RPC port (-carla-rpc-port).")
    parser.add_argument(
        "--quality-level",
        type=str,
        choices=["Epic", "Low"],
        default=None,
        help="Rendering quality (-quality-level).",
    )
    parser.add_argument(
        "--offscreen",
        action="store_true",
        help="Enable off-screen rendering (-RenderOffScreen).",
    )
    parser.add_argument(
        "--no-sound",
        action="store_true",
        help="Disable sound (-nosound).",
    )
    parser.add_argument(
        "--ros2",
        action="store_true",
        help="Enable native ROS2 (--ros2).",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=20.0,
        help="Seconds to wait for RPC port to open. Set <= 0 to skip check.",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Exit immediately after launch (do not manage process lifetime).",
    )
    return parser

def resolve_executable(user_input: str | None) -> Path:
    if user_input:
        candidate = Path(user_input).expanduser()
    else:
        env_path = os.environ.get("CARLA_UE5_EXE")
        if env_path:
            candidate = Path(env_path).expanduser()
        else:
            default_name = "CarlaUnreal.exe" if sys.platform.startswith("win") else "CarlaUnreal.sh"
            candidate = Path(default_name)

    if candidate.is_dir():
        default_name = "CarlaUnreal.exe" if sys.platform.startswith("win") else "CarlaUnreal.sh"
        candidate = candidate / default_name

    return candidate.resolve()


def build_launch_command(
    executable: Path,
    port: int,
    quality_level: str | None,
    offscreen: bool,
    no_sound: bool,
    ros2: bool,
) -> list[str]:
    command = [str(executable), f"-carla-rpc-port={port}"]
    if quality_level is not None:
        command.append(f"-quality-level={quality_level}")
    if offscreen:
        command.append("-RenderOffScreen")
    if no_sound:
        command.append("-nosound")
    if ros2:
        command.append("--ros2")
    return command


def wait_for_port(host: str, port: int, timeout_sec: float) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.3)
    return False


def terminate_process(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)


def main() -> int:
    args = build_arg_parser().parse_args()
    executable = resolve_executable(args.carla_exe)
    if not executable.exists():
        print(f"[error] CARLA executable not found: {executable}")
        print("Provide --carla-exe or set CARLA_UE5_EXE.")
        return 1

    command = build_launch_command(
        executable=executable,
        port=args.port,
        quality_level=args.quality_level,
        offscreen=args.offscreen,
        no_sound=args.no_sound,
        ros2=args.ros2,
    )
    print("[info] launch command:")
    print(" ", " ".join(command))

    try:
        process = subprocess.Popen(command, cwd=str(executable.parent))
    except OSError as exc:
        print(f"[error] failed to launch CARLA: {exc}")
        return 1

    print(f"[info] CARLA started, pid={process.pid}")

    if args.startup_timeout > 0:
        ready = wait_for_port(args.host, args.port, args.startup_timeout)
        if ready:
            print(f"[info] RPC port {args.host}:{args.port} is ready.")
        else:
            print(
                f"[warn] RPC port {args.host}:{args.port} did not open within "
                f"{args.startup_timeout:.1f}s."
            )

    if args.detach:
        return 0

    print("[info] press Ctrl+C to stop CARLA started by this script.")
    try:
        while process.poll() is None:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[info] stopping CARLA ...")
        terminate_process(process)

    return process.poll() or 0


if __name__ == "__main__":
    raise SystemExit(main())
