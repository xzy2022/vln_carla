from __future__ import annotations

import argparse
import atexit
import os
import re
import subprocess
import time
from importlib import metadata

from adapters.control.simple_agent import SimpleAgent
from domain.errors import EnvConnectionError, EnvStepError
from infrastructure.carla.carla_env_adapter import CarlaEnvAdapter
from infrastructure.logging.in_memory_logger import InMemoryLogger
from usecases.run_episode import RunEpisodeUseCase

DEFAULT_CARLA_EXECUTABLES = {
    "ue4": r"D:\Workspace\02_Playground\CARLA_Latest\CarlaUE4.exe",
    "ue5": r"D:\Workspace\02_Playground\Carla-0.10.0-Win64-Shipping\CarlaUnreal.exe",
}

ANSI_YELLOW_BOLD = "\033[1;33m"
ANSI_RESET = "\033[0m"

_MANAGED_PROCESSES: list[subprocess.Popen] = []
_WINDOWS_JOB_HANDLE = None

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]


def _terminal_supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    term = (os.environ.get("TERM") or "").lower()
    if term == "dumb":
        return False
    return True


def _warn(message: str) -> None:
    if _terminal_supports_color():
        print(f"{ANSI_YELLOW_BOLD}[warn]{ANSI_RESET} {message}")
        return
    print(f"[warn] {message}")


def _terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    except Exception:
        pass


def _cleanup_managed_processes() -> None:
    global _WINDOWS_JOB_HANDLE

    for proc in list(reversed(_MANAGED_PROCESSES)):
        _terminate_process(proc)
    _MANAGED_PROCESSES.clear()

    if os.name == "nt" and _WINDOWS_JOB_HANDLE is not None:
        try:
            ctypes.windll.kernel32.CloseHandle(_WINDOWS_JOB_HANDLE)
        except Exception:
            pass
        _WINDOWS_JOB_HANDLE = None


def _ensure_windows_job_object():
    global _WINDOWS_JOB_HANDLE
    if os.name != "nt":
        return None
    if _WINDOWS_JOB_HANDLE is not None:
        return _WINDOWS_JOB_HANDLE

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    job_handle = kernel32.CreateJobObjectW(None, None)
    if not job_handle:
        _warn(f"Failed to create Windows Job Object (error={ctypes.get_last_error()}).")
        return None

    limit_info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    limit_info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = kernel32.SetInformationJobObject(
        job_handle,
        JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
        ctypes.byref(limit_info),
        ctypes.sizeof(limit_info),
    )
    if not ok:
        _warn(f"Failed to configure Windows Job Object (error={ctypes.get_last_error()}).")
        kernel32.CloseHandle(job_handle)
        return None

    _WINDOWS_JOB_HANDLE = job_handle
    return _WINDOWS_JOB_HANDLE


def _register_managed_process(proc: subprocess.Popen) -> None:
    _MANAGED_PROCESSES.append(proc)

    if os.name != "nt":
        return

    job_handle = _ensure_windows_job_object()
    if job_handle is None:
        return

    process_handle = getattr(proc, "_handle", None)
    if process_handle is None:
        _warn("Could not get child process handle for lifecycle binding.")
        return

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    assigned = kernel32.AssignProcessToJobObject(job_handle, process_handle)
    if not assigned:
        _warn(f"Failed to bind CARLA process lifecycle (error={ctypes.get_last_error()}).")


def _unregister_managed_process(proc: subprocess.Popen) -> None:
    if proc in _MANAGED_PROCESSES:
        _MANAGED_PROCESSES.remove(proc)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 0 CARLA runner")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--fixed-dt", type=float, default=0.1)
    parser.add_argument("--sensor-timeout", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--throttle", type=float, default=0.3)
    parser.add_argument(
        "--carla-version",
        type=str,
        choices=["ue4", "ue5", "auto"],
        default="auto",
        help="CARLA server flavor for default executable resolution.",
    )
    parser.add_argument(
        "--carla-path",
        type=str,
        default=None,
        help="Optional CARLA executable path override. If omitted, resolved from --carla-version.",
    )
    parser.add_argument(
        "--no-launch-server",
        action="store_true",
        help="Do not launch CARLA executable; connect to an already running server.",
    )
    parser.add_argument("--quality-level", type=str, choices=["Epic", "Low"], default=None, help="Server render quality (only if we launch CARLA)")
    parser.add_argument("--map", type=str, default="Town04", help="Map to load")
    parser.add_argument("--spectator-follow", action="store_true", help="Follow ego with spectator")
    parser.add_argument("--no-rendering", action="store_true", help="Disable rendering (GPU sensors return empty data)")
    parser.add_argument("--camera-width", type=int, default=800)
    parser.add_argument("--camera-height", type=int, default=600)
    parser.add_argument("--camera-sensor-tick", type=float, default=None, help="Seconds between camera captures (e.g. 0.2 for 5Hz)")
    parser.add_argument("--unload-map-layers", type=str, default="", help="Comma-separated map layers to unload (e.g. Buildings,Vegetation,ParkedVehicles)")
    return parser


def _resolve_carla_executable(carla_version: str, explicit_path: str | None) -> tuple[str | None, str]:
    if explicit_path:
        return explicit_path, "custom"

    if carla_version in ("ue4", "ue5"):
        return DEFAULT_CARLA_EXECUTABLES[carla_version], carla_version

    # auto: keep previous behavior preference (UE4 first), fallback to UE5.
    for candidate_version in ("ue4", "ue5"):
        candidate_path = DEFAULT_CARLA_EXECUTABLES[candidate_version]
        if os.path.isfile(candidate_path):
            return candidate_path, candidate_version
    return DEFAULT_CARLA_EXECUTABLES["ue4"], "ue4"


def _detect_python_carla_version() -> str | None:
    for dist_name in ("carla", "carla-simulator"):
        try:
            return metadata.version(dist_name)
        except metadata.PackageNotFoundError:
            continue
        except Exception:
            break

    try:
        import carla  # type: ignore
    except Exception:
        return None

    for attr in ("__version__", "version", "VERSION"):
        value = getattr(carla, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _classify_python_carla_flavor(version: str | None) -> str | None:
    if version is None:
        return None
    if version == "unknown":
        return "unknown"

    match = re.search(r"(\d+)\.(\d+)", version)
    if not match:
        return "unknown"

    major = int(match.group(1))
    minor = int(match.group(2))
    if major == 0 and minor <= 9:
        return "ue4"
    if major == 0 and minor >= 10:
        return "ue5"
    return "unknown"


def _warn_if_python_env_mismatch(target_flavor: str | None) -> None:
    if target_flavor not in ("ue4", "ue5"):
        return

    env_name = os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("VIRTUAL_ENV") or "unknown"
    package_version = _detect_python_carla_version()
    package_flavor = _classify_python_carla_flavor(package_version)

    if package_version is None:
        _warn(
            f"Python env '{env_name}' has no CARLA package, expected {target_flavor} package before startup."
        )
        return

    if package_flavor == "unknown":
        _warn(
            f"Python env '{env_name}' CARLA package version '{package_version}' cannot be classified; "
            f"expected {target_flavor}."
        )
        return

    if package_flavor != target_flavor:
        _warn(
            f"Python env '{env_name}' CARLA package version '{package_version}' does not match startup target "
            f"'{target_flavor}'."
        )


def _ensure_carla_server(
    carla_exe: str | None,
    port: int,
    quality_level: str | None,
    resolved_version: str,
) -> subprocess.Popen | None:
    if not carla_exe:
        return None

    if not os.path.isfile(carla_exe):
        _warn(f"CARLA executable not found ({resolved_version}): {carla_exe}")
        return None

    print(f"Starting CARLA server ({resolved_version}) on port {port}...")
    args = [carla_exe, f"-carla-rpc-port={port}"]
    if quality_level:
        args.append(f"-quality-level={quality_level}")

    server = subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    _register_managed_process(server)
    return server


def main() -> int:
    args = build_arg_parser().parse_args()

    resolved_exe, resolved_version = _resolve_carla_executable(args.carla_version, args.carla_path)
    validation_target = args.carla_version if args.carla_version in ("ue4", "ue5") else resolved_version
    _warn_if_python_env_mismatch(validation_target)

    server_process: subprocess.Popen | None = None
    if args.no_launch_server:
        print(f"Skipping CARLA launch; connecting to running server at {args.host}:{args.port}.")
    else:
        server_process = _ensure_carla_server(resolved_exe, args.port, args.quality_level, resolved_version)

    if server_process:
        print("Waiting for CARLA server to initialize...")
        time.sleep(10)

    if args.no_rendering:
        _warn("--no-rendering will disable GPU sensors (camera frames will be empty).")

    server_exited_early = False

    def _should_stop() -> bool:
        nonlocal server_exited_early
        if server_process is None:
            return False
        if server_process.poll() is None:
            return False
        if not server_exited_early:
            _warn(f"CARLA server process exited with code {server_process.returncode}; stopping Python runner.")
        server_exited_early = True
        return True

    unload_layers = tuple(filter(None, (s.strip() for s in args.unload_map_layers.split(","))))
    env = CarlaEnvAdapter(
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        fixed_dt=args.fixed_dt,
        sensor_timeout=args.sensor_timeout,
        map_name=args.map,
        spectator_follow=args.spectator_follow,
        no_rendering_mode=args.no_rendering,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_sensor_tick=args.camera_sensor_tick,
        unload_map_layers=unload_layers,
    )
    agent = SimpleAgent(throttle=args.throttle)
    logger = InMemoryLogger()
    usecase = RunEpisodeUseCase(
        env=env,
        agent=agent,
        logger=logger,
        max_steps=args.max_steps,
        should_stop=_should_stop,
    )

    try:
        summary = usecase.run()
        if server_exited_early:
            return 1
        print(f"Episode finished: steps={summary['total_steps']} reward={summary['total_reward']:.3f}")
    except (EnvConnectionError, EnvStepError) as exc:
        print(f"[error] {exc}")
        return 1
    finally:
        env.close()
        if server_process is not None:
            _terminate_process(server_process)
            _unregister_managed_process(server_process)

    return 0


atexit.register(_cleanup_managed_processes)


if __name__ == "__main__":
    raise SystemExit(main())