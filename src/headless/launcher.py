from __future__ import annotations

import argparse
from datetime import datetime, timezone
import errno
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Callable, Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    return launch_headless_after_ui_exit(
        wait_pid=int(args.wait_pid),
        job_path=str(args.job),
        python_executable=str(args.python),
        runner_path=str(args.runner),
        poll_interval_seconds=float(args.poll_interval),
        log_level=str(args.log_level),
        validate_only=bool(args.validate_only),
    )


def launch_headless_after_ui_exit(
    *,
    wait_pid: int,
    job_path: str,
    python_executable: str = sys.executable,
    runner_path: Optional[str] = None,
    poll_interval_seconds: float = 1.0,
    log_level: str = "INFO",
    validate_only: bool = False,
    pid_alive_fn: Callable[[int], bool] = lambda pid: _pid_is_alive(pid),
    sleep_fn: Callable[[float], None] = time.sleep,
    popen_fn: Callable[..., object] = subprocess.Popen,
) -> int:
    """Wait for the UI process to exit, then detach the headless runner.

    Return 0 once the runner has been spawned successfully. The runner's own
    final status is recorded by its logs, not by this launcher process.
    """
    normalized_job_path = str(Path(job_path).expanduser())
    runner = runner_path or _default_runner_path()
    log_path = _headless_log_path(normalized_job_path)
    runner_pid_path = _runner_pid_path(normalized_job_path)
    _emit(log_path, f"Headless job queued: {normalized_job_path}")
    _emit(log_path, f"Waiting for UI process {wait_pid} to exit before loading data.")

    poll_interval = max(0.05, float(poll_interval_seconds))
    while pid_alive_fn(int(wait_pid)):
        sleep_fn(poll_interval)

    command = [
        str(python_executable),
        str(runner),
        normalized_job_path,
        "--log-level",
        str(log_level),
    ]
    if validate_only:
        command.append("--validate-only")

    _emit(log_path, "UI process exited; starting headless runner.")
    _emit(log_path, "Command: " + shlex.join(command))
    _emit(log_path, "Runner stdout/stderr will be appended to this log.")
    _remove_runner_pid(runner_pid_path)
    try:
        process = _spawn_detached_runner(
            command,
            output_log_path=log_path,
            popen_fn=popen_fn,
        )
        pid = _process_pid(process)
    except Exception as exc:
        _remove_runner_pid(runner_pid_path)
        _emit(log_path, f"Failed to start headless runner: {exc}")
        return 1

    _write_runner_pid(runner_pid_path, pid)
    _emit(log_path, f"Headless runner started with PID {pid}.")
    _emit(log_path, f"Runner PID file: {runner_pid_path}")
    _emit(log_path, "Launcher exiting; runner continues independently.")
    return 0


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for the UI process to exit, then detach a headless job"
    )
    parser.add_argument("--wait-pid", required=True, type=int, help="UI process id to wait for")
    parser.add_argument("--job", required=True, help="Path to .headless-job job.json")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run the headless runner",
    )
    parser.add_argument(
        "--runner",
        default=_default_runner_path(),
        help="Path to run_headless_job.py",
    )
    parser.add_argument(
        "--poll-interval",
        default=1.0,
        type=float,
        help="Seconds between UI process liveness checks",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level forwarded to the headless runner",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Forward --validate-only to the headless runner",
    )
    return parser.parse_args(argv)


def _default_runner_path() -> str:
    return str(Path(__file__).resolve().parents[2] / "run_headless_job.py")


def _headless_log_path(job_path: str) -> Path:
    return _job_dir_for_path(job_path) / "headless.log"


def _runner_pid_path(job_path: str) -> Path:
    return _job_dir_for_path(job_path) / "runner.pid"


def _job_dir_for_path(job_path: str) -> Path:
    job_file = Path(job_path).expanduser()
    job_dir: Optional[Path] = None
    try:
        payload = json.loads(job_file.read_text(encoding="utf-8"))
        raw_job_dir = payload.get("job_dir") if isinstance(payload, dict) else None
        if isinstance(raw_job_dir, str) and raw_job_dir.strip():
            job_dir = Path(raw_job_dir).expanduser()
    except Exception:
        job_dir = None
    if job_dir is None:
        job_dir = job_file.parent
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def _write_runner_pid(path: Path, pid: int) -> None:
    path.write_text(f"{int(pid)}\n", encoding="utf-8")


def _remove_runner_pid(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _process_pid(process: object) -> int:
    pid = int(getattr(process, "pid"))
    if pid <= 0:
        raise ValueError(f"Invalid headless runner PID: {pid}")
    return pid


def _spawn_detached_runner(
    command: Sequence[str],
    *,
    output_log_path: Path,
    popen_fn: Callable[..., object],
) -> object:
    with Path(os.devnull).open("rb") as stdin_handle, output_log_path.open(
        "a",
        encoding="utf-8",
    ) as output_handle:
        return popen_fn(
            list(command),
            stdin=stdin_handle,
            stdout=output_handle,
            stderr=output_handle,
            close_fds=True,
            start_new_session=True,
            cwd=str(Path.cwd()),
        )


def _emit(log_path: Path, message: str) -> None:
    line = f"{_timestamp()} | launcher | {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _pid_is_alive(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        if exc.errno == errno.EPERM:
            return True
        return False
    return True


if __name__ == "__main__":
    sys.exit(main())
