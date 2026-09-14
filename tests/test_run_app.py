"""Deterministic checks for the one-command local launcher."""

from __future__ import annotations

import subprocess
import sys

import run_app


class FakeProcess:
    def __init__(self, exit_code=None):
        self.exit_code = exit_code
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.exit_code

    def terminate(self):
        self.terminated = True
        self.exit_code = 0

    def wait(self, timeout=None):
        return self.exit_code

    def kill(self):
        self.killed = True
        self.exit_code = -9


def test_commands_use_active_python_and_fixed_local_ports():
    backend, frontend = run_app.build_commands()

    assert backend[:3] == [sys.executable, "-m", "uvicorn"]
    assert backend[-2:] == ["--port", "8000"]
    assert frontend[:3] == [sys.executable, "-m", "streamlit"]
    assert "--server.port=8501" in frontend


def test_commands_support_alternate_smoke_ports():
    backend, frontend = run_app.build_commands(8010, 8510)

    assert backend[-2:] == ["--port", "8010"]
    assert "--server.port=8510" in frontend


def test_frontend_process_does_not_receive_gemini_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-that-must-not-cross-boundary")

    environment = run_app.frontend_environment("http://127.0.0.1:8010")

    assert "GEMINI_API_KEY" not in environment
    assert environment["BACKEND_URL"] == "http://127.0.0.1:8010"


def test_stop_process_terminates_a_running_child():
    process = FakeProcess()

    run_app.stop_process(process)  # type: ignore[arg-type]

    assert process.terminated is True
    assert process.killed is False


def test_stop_process_ignores_a_completed_child():
    process = FakeProcess(exit_code=0)

    run_app.stop_process(process)  # type: ignore[arg-type]

    assert process.terminated is False


def test_existing_application_is_reused_without_spawning(monkeypatch):
    monkeypatch.setattr(run_app, "url_is_ready", lambda _url: True)

    def unexpected_spawn(*_args, **_kwargs):
        raise AssertionError("The launcher must not duplicate running services.")

    monkeypatch.setattr(run_app.subprocess, "Popen", unexpected_spawn)

    assert run_app.run(open_browser=False) == 0
