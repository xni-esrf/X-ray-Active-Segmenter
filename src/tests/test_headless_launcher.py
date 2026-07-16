from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest

from src.headless.launcher import launch_headless_after_ui_exit


class HeadlessLauncherTests(unittest.TestCase):
    def test_waits_for_ui_pid_then_spawns_detached_headless_command(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            job_dir = root / "headless-job"
            job_path = job_dir / "job.json"
            job_dir.mkdir()
            job_path.write_text(
                json.dumps({"job_dir": str(job_dir)}),
                encoding="utf-8",
            )
            liveness = iter((True, True, False))
            sleep_calls = []
            popen_calls = []

            def pid_alive(_pid: int) -> bool:
                return next(liveness)

            def sleep(seconds: float) -> None:
                sleep_calls.append(seconds)

            def popen(command, **kwargs):
                popen_calls.append((command, kwargs))
                return SimpleNamespace(pid=456)

            exit_code = launch_headless_after_ui_exit(
                wait_pid=123,
                job_path=str(job_path),
                python_executable="/venv/bin/python",
                runner_path="/repo/run_headless_job.py",
                poll_interval_seconds=0.25,
                log_level="DEBUG",
                validate_only=True,
                pid_alive_fn=pid_alive,
                sleep_fn=sleep,
                popen_fn=popen,
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(sleep_calls, [0.25, 0.25])
            self.assertEqual(len(popen_calls), 1)
            command, kwargs = popen_calls[0]
            self.assertEqual(
                command,
                [
                    "/venv/bin/python",
                    "/repo/run_headless_job.py",
                    str(job_path),
                    "--log-level",
                    "DEBUG",
                    "--validate-only",
                ],
            )
            self.assertTrue(kwargs["close_fds"])
            self.assertTrue(kwargs["start_new_session"])
            self.assertEqual(kwargs["cwd"], str(Path.cwd()))
            self.assertEqual(kwargs["stdout"].name, str(job_dir / "headless.log"))
            self.assertEqual(kwargs["stderr"].name, str(job_dir / "headless.log"))
            log_text = (job_dir / "headless.log").read_text(encoding="utf-8")
            self.assertIn("Waiting for UI process 123 to exit", log_text)
            self.assertIn("Runner stdout/stderr will be appended to this log", log_text)
            self.assertIn("Headless runner started with PID 456", log_text)
            self.assertIn("Launcher exiting; runner continues independently", log_text)
            self.assertEqual((job_dir / "runner.pid").read_text(encoding="utf-8"), "456\n")
            self.assertFalse((job_dir / "runner.log").exists())

    def test_uses_job_parent_for_log_when_job_dir_is_missing(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            job_path = root / "job.json"
            job_path.write_text("{}", encoding="utf-8")

            exit_code = launch_headless_after_ui_exit(
                wait_pid=123,
                job_path=str(job_path),
                python_executable="python",
                runner_path="runner.py",
                pid_alive_fn=lambda _pid: False,
                sleep_fn=lambda _seconds: None,
                popen_fn=lambda _command, **_kwargs: SimpleNamespace(pid=789),
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((root / "headless.log").exists())
            self.assertFalse((root / "runner.log").exists())
            self.assertEqual((root / "runner.pid").read_text(encoding="utf-8"), "789\n")

    def test_returns_nonzero_and_does_not_write_pid_when_spawn_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            job_path = root / "job.json"
            job_path.write_text("{}", encoding="utf-8")
            stale_pid = root / "runner.pid"
            stale_pid.write_text("111\n", encoding="utf-8")

            def fail_spawn(_command, **_kwargs):
                raise OSError("spawn failed")

            exit_code = launch_headless_after_ui_exit(
                wait_pid=123,
                job_path=str(job_path),
                python_executable="python",
                runner_path="runner.py",
                pid_alive_fn=lambda _pid: False,
                sleep_fn=lambda _seconds: None,
                popen_fn=fail_spawn,
            )

            self.assertEqual(exit_code, 1)
            self.assertFalse(stale_pid.exists())
            log_text = (root / "headless.log").read_text(encoding="utf-8")
            self.assertIn("Failed to start headless runner: spawn failed", log_text)

    def test_returns_nonzero_and_does_not_write_pid_when_spawned_pid_is_invalid(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            job_path = root / "job.json"
            job_path.write_text("{}", encoding="utf-8")

            exit_code = launch_headless_after_ui_exit(
                wait_pid=123,
                job_path=str(job_path),
                python_executable="python",
                runner_path="runner.py",
                pid_alive_fn=lambda _pid: False,
                sleep_fn=lambda _seconds: None,
                popen_fn=lambda _command, **_kwargs: SimpleNamespace(pid=0),
            )

            self.assertEqual(exit_code, 1)
            self.assertFalse((root / "runner.pid").exists())
            log_text = (root / "headless.log").read_text(encoding="utf-8")
            self.assertIn("Failed to start headless runner: Invalid headless runner PID: 0", log_text)


if __name__ == "__main__":
    unittest.main()
