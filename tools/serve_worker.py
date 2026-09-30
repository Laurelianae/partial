"""Keep the server and its children tied to the lifetime of the SSH connection."""

from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import threading
from pathlib import Path


def main() -> int:
    server_args = sys.argv[1:]
    watch_stdin = server_args[:1] == ["--watch-stdin"]
    if watch_stdin:
        server_args = server_args[1:]
    # SSH does not activate the venv. JIT builds still need its ninja executable on PATH.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    signal.signal(signal.SIGINT, lambda *_: stopping.set())

    process = subprocess.Popen(
        [sys.executable, "-m", "minisgl", *server_args],
        start_new_session=True,
        stdin=subprocess.DEVNULL if watch_stdin else None,
    )
    connection_thread = None
    if watch_stdin:

        def watch_connection() -> None:
            # Raw reads avoid holding a BufferedReader lock during interpreter shutdown.
            descriptor = sys.stdin.fileno()
            while not stopping.is_set():
                readable, _, _ = select.select([descriptor], [], [], 0.2)
                if readable and not os.read(descriptor, 1):
                    stopping.set()
                    return

        connection_thread = threading.Thread(target=watch_connection, name="ssh-lifetime")
        connection_thread.start()

    while process.poll() is None and not stopping.wait(0.2):
        pass
    stopping.set()
    if connection_thread is not None:
        connection_thread.join()
    if process.poll() is None:
        # Give the server parent time to send ExitMsg and stop its own children.
        process.terminate()
        try:
            process.wait(timeout=12.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    # Also clean up children if the server parent exited abruptly before its finally block.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return process.returncode if process.returncode >= 0 else 128 - process.returncode


if __name__ == "__main__":
    sys.exit(main())
