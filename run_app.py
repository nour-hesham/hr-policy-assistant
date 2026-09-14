"""Start the FastAPI backend and Streamlit frontend with one command."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent
STARTUP_TIMEOUT_SECONDS = 180


def build_commands(
    backend_port: int = 8000,
    frontend_port: int = 8501,
) -> tuple[list[str], list[str]]:
    """Use the active interpreter so the launcher works without activation."""
    backend = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.api:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(backend_port),
    ]
    frontend = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "frontend/app.py",
        "--server.address=127.0.0.1",
        f"--server.port={frontend_port}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]
    return backend, frontend


def frontend_environment(backend_url: str = "http://127.0.0.1:8000") -> dict[str, str]:
    """Keep the backend-only Gemini credential out of Streamlit's process."""
    environment = os.environ.copy()
    environment.pop("GEMINI_API_KEY", None)
    environment["BACKEND_URL"] = backend_url
    return environment


def url_is_ready(url: str) -> bool:
    """Return whether a local HTTP endpoint is already available."""
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError):
        return False


def wait_until_ready(
    url: str,
    process: subprocess.Popen[bytes],
    *,
    timeout_seconds: float = STARTUP_TIMEOUT_SECONDS,
) -> None:
    """Wait for an HTTP endpoint or fail when its process exits/times out."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(
                f"A local application process exited during startup (code {exit_code})."
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 300:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for {url}.")


def stop_process(process: subprocess.Popen[bytes] | None) -> None:
    """Stop one child process, escalating only when graceful exit fails."""
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run(
    *,
    open_browser: bool = True,
    backend_port: int = 8000,
    frontend_port: int = 8501,
) -> int:
    backend_url = f"http://127.0.0.1:{backend_port}"
    frontend_url = f"http://127.0.0.1:{frontend_port}"
    backend_command, frontend_command = build_commands(backend_port, frontend_port)
    backend: subprocess.Popen[bytes] | None = None
    frontend: subprocess.Popen[bytes] | None = None

    try:
        backend_is_running = url_is_ready(f"{backend_url}/health")
        frontend_is_running = url_is_ready(frontend_url)
        if backend_is_running and frontend_is_running:
            print(f"HR Policy Assistant is already running at {frontend_url}")
            if open_browser:
                webbrowser.open(frontend_url)
            return 0
        if backend_is_running or frontend_is_running:
            raise RuntimeError(
                "Port 8000 or 8501 is already in use. Stop the existing local "
                "service, then run this command again."
            )

        print("Starting the policy index and API (first run can take a few minutes)...")
        backend = subprocess.Popen(backend_command, cwd=REPOSITORY_ROOT)
        wait_until_ready(f"{backend_url}/health", backend)

        print("Starting the Streamlit interface...")
        frontend = subprocess.Popen(
            frontend_command,
            cwd=REPOSITORY_ROOT,
            env=frontend_environment(backend_url),
        )
        wait_until_ready(frontend_url, frontend)

        print(f"HR Policy Assistant is ready at {frontend_url}")
        print("Press Ctrl+C here to stop both services.")
        if open_browser:
            webbrowser.open(frontend_url)

        while True:
            if backend.poll() is not None:
                raise RuntimeError("The backend stopped unexpectedly.")
            if frontend.poll() is not None:
                raise RuntimeError("The frontend stopped unexpectedly.")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping the HR Policy Assistant...")
        return 0
    except RuntimeError as exc:
        print(f"Unable to start the application: {exc}", file=sys.stderr)
        return 1
    finally:
        stop_process(frontend)
        stop_process(backend)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Start FastAPI and Streamlit together for local review."
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start both services without opening the default browser.",
    )
    parser.add_argument("--backend-port", type=int, default=8000, help=argparse.SUPPRESS)
    parser.add_argument("--frontend-port", type=int, default=8501, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    return run(
        open_browser=not arguments.no_browser,
        backend_port=arguments.backend_port,
        frontend_port=arguments.frontend_port,
    )


if __name__ == "__main__":
    raise SystemExit(main())
