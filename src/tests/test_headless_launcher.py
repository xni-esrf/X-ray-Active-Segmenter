from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest

from src.headless.launcher import launch_headless_after_ui_exit


class HeadlessLauncherTests(unittest.TestCase):
    def test_waits_for_ui_pid_then_runs_headless_command(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            job_dir = root / ".headless-job"
            job_path = job_dir / "job.json"
            job_dir.mkdir()
            job_path.write_text(
                json.dumps({"job_dir": str(job_dir)}),
                encoding="utf-8",
            )
            liveness = iter((True, True, False))
            sleep_calls = []
            run_calls = []

            def pid_alive(_pid: int) -> bool:
                return next(liveness)

            def sleep(seconds: float) -> None:
                sleep_calls.append(seconds)

            def run(command):
                run_calls.append(command)
                return SimpleNamespace(returncode=7)

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
                run_fn=run,
            )

            self.assertEqual(exit_code, 7)
            self.assertEqual(sleep_calls, [0.25, 0.25])
            self.assertEqual(
                run_calls,
                [
                    [
                        "/venv/bin/python",
                        "/repo/run_headless_job.py",
                        str(job_path),
                        "--log-level",
                        "DEBUG",
                        "--validate-only",
                    ]
                ],
            )
            log_text = (job_dir / "headless.log").read_text(encoding="utf-8")
            self.assertIn("Waiting for UI process 123 to exit", log_text)
            self.assertIn("Headless runner exited with code 7", log_text)

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
                run_fn=lambda _command: SimpleNamespace(returncode=0),
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((root / "headless.log").exists())


if __name__ == "__main__":
    unittest.main()
