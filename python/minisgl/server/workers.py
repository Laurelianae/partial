from __future__ import annotations

import multiprocessing as mp
import os
import queue
import signal
import threading
import time
from typing import Callable


class WorkerProcesses:
    """Own local children and stop the frontend when any worker fails."""

    def __init__(self) -> None:
        self.processes: list[mp.Process] = []
        self.stopping = threading.Event()
        self.failure: str | None = None
        self.monitor: threading.Thread | None = None

    def start(self, **kwargs) -> None:
        process = mp.Process(daemon=False, **kwargs)
        process.start()
        self.processes.append(process)

    def check_alive(self) -> None:
        for process in self.processes:
            if process.exitcode is not None:
                self.failure = f"{process.name} exited with code {process.exitcode}"
                raise RuntimeError(self.failure)

    def wait_ready(self, acknowledgments: mp.Queue, count: int, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        for _ in range(count):
            while True:
                self.check_alive()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Workers did not become ready before --startup-timeout")
                try:
                    acknowledgments.get(timeout=min(0.2, remaining))
                    break
                except queue.Empty:
                    continue

    def monitor_frontend(self) -> None:
        def watch() -> None:
            while not self.stopping.wait(0.2):
                try:
                    self.check_alive()
                except RuntimeError as error:
                    self.failure = str(error)
                    # Uvicorn handles SIGTERM by closing connections and exiting.
                    os.kill(os.getpid(), signal.SIGTERM)
                    return

        self.monitor = threading.Thread(target=watch, name="worker-monitor", daemon=True)
        self.monitor.start()

    def stop(self, request_exit: Callable[[], None] | None = None) -> None:
        self.stopping.set()
        if self.monitor is not None:
            self.monitor.join()
        deadline = time.monotonic() + 10.0
        if self.failure is not None:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
        if request_exit is not None and self.failure is None:
            try:
                request_exit()
            except Exception:
                # A failed worker may have already closed its request socket.
                pass
        for process in self.processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=0.2)
            if process.is_alive():
                process.kill()
                process.join()
