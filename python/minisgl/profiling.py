"""Bounded, opt-in CPU/CUDA traces for sequential Naive API requests."""

from __future__ import annotations

import gzip
import json
import os
import shutil
from contextlib import nullcontext
from functools import wraps
from pathlib import Path

import torch

_ACTIVE = False


def region(name: str):
    return torch.profiler.record_function("naive::" + name) if _ACTIVE else nullcontext()


def traced(name: str):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with region(name):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def capture_phase(request: int, step: int, warmup: int, repeat: int) -> str | None:
    """Step zero produces the first token; steps 1 onward are decode forwards."""
    if request % (warmup + repeat) < warmup:
        return None
    if step == 0:
        return "prefill"
    return "decode" if 8 <= step <= 11 else None


class RequestProfiler:
    def __init__(self, rank: int, directory: str, warmup: int, repeat: int):
        self.rank = rank
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.warmup, self.repeat = warmup, repeat
        self.request = -1
        self.uid: int | None = None
        self.step = 0
        self.profiler = None
        self.phase = None

    @classmethod
    def from_env(cls, rank: int):
        directory = os.environ.get("MINISGL_PROFILE_DIR")
        if not directory:
            return None
        return cls(
            rank,
            directory,
            int(os.environ["MINISGL_PROFILE_WARMUP"]),
            int(os.environ["MINISGL_PROFILE_REPEAT"]),
        )

    def prime(self) -> None:
        # On the Sparks' torch 2.9.1/cu130 build, the first CUPTI capture can
        # contain CPU events only. Initialize it before server readiness;
        # measured captures still independently require CUDA kernel events.
        value = torch.ones(1, device="cuda")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ):
            value.add_(1)
            torch.cuda.synchronize()

    def begin(self, uid: int) -> None:
        if self.uid is not None:
            raise RuntimeError("Profiling requires sequential requests")
        self.request += 1
        self.uid, self.step = uid, 0

    def before_step(self) -> None:
        global _ACTIVE
        if self.uid is None:
            return
        phase = capture_phase(self.request, self.step, self.warmup, self.repeat)
        if phase and self.profiler is None:
            self.phase = phase
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )
            self.profiler.start()
            _ACTIVE = True

    def after_step(self, finished: bool) -> None:
        if self.uid is None:
            return
        if self.profiler is not None and self.step in (0, 11):
            self.stop(complete=True)
        self.step += 1
        if finished:
            self.uid = None

    def stop(self, complete: bool = False) -> None:
        global _ACTIVE
        _ACTIVE = False
        profiler, self.profiler = self.profiler, None
        if profiler is None:
            return
        profiler.stop()
        path = self.directory / f"rank-{self.rank}-request-{self.request}-{self.phase}.json"
        profiler.export_chrome_trace(str(path))
        with (
            path.open("rb") as source,
            gzip.open(str(path) + ".gz", "wb", compresslevel=1) as target,
        ):
            shutil.copyfileobj(source, target)
        path.unlink()
        print(
            "NAIVE_RESULT="
            + json.dumps(
                {
                    "kind": "profile_trace",
                    "rank": self.rank,
                    "uid": self.uid,
                    "request": self.request,
                    "phase": self.phase,
                    "steps": [0] if self.phase == "prefill" else [8, 9, 10, 11],
                    "complete": complete,
                    "path": str(path) + ".gz",
                }
            ),
            flush=True,
        )

    def abort(self) -> None:
        self.stop()
        self.uid = None
