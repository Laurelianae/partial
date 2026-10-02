"""Opt-in single-request Naive baseline records; no additional CUDA synchronization."""

from __future__ import annotations

import json
import time

import torch


class BaselineTelemetry:
    def __init__(self, rank: int):
        self.rank = rank
        self.record: dict | None = None

    def runtime(self, config, engine) -> None:
        from importlib.metadata import PackageNotFoundError, version

        packages = {}
        for name in ("torch", "transformers", "triton", "flashinfer-python", "safetensors"):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        print(
            "NAIVE_RESULT="
            + json.dumps(
                {
                    "kind": "runtime",
                    "rank": self.rank,
                    "settings": config.shared_inference_settings(),
                    "resolved": {
                        "dtype": str(engine.dtype),
                        "num_pages": engine.num_pages,
                        "max_seq_len": engine.max_seq_len,
                        "attention": type(engine.attn_backend).__name__,
                        "overlap": False,
                        "cuda_graphs": False,
                    },
                    "packages": packages,
                    "nccl": torch.cuda.nccl.version(),
                    "float32_matmul_precision": torch.get_float32_matmul_precision(),
                    "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                }
            ),
            flush=True,
        )

    @staticmethod
    def memory() -> dict:
        free, total = torch.cuda.mem_get_info()
        return {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
            "device_free": free,
            "device_total": total,
        }

    def begin(self, msg) -> None:
        if self.record is not None:
            raise RuntimeError("Baseline telemetry requires sequential requests")
        torch.cuda.reset_peak_memory_stats()
        self.record = {
            "kind": "telemetry",
            "rank": self.rank,
            "uid": msg.uid,
            "input_ids": msg.input_ids.tolist(),
            "output_ids": [],
            "token_seconds": [],
            "resident_bytes": self.memory(),
        }
        self.started = time.perf_counter()

    def token(self, uid: int, token: int) -> None:
        elapsed = time.perf_counter() - self.started
        record = self.record
        assert record is not None and record["uid"] == uid
        record["output_ids"].append(token)
        record["token_seconds"].append(elapsed)
        if len(record["output_ids"]) == 1:
            record["prefill_seconds"] = elapsed
            record["prefill_memory_bytes"] = self.memory()
            torch.cuda.reset_peak_memory_stats()

    def finish(self) -> None:
        assert self.record is not None
        self.record["decode_memory_bytes"] = self.memory()
        print("NAIVE_RESULT=" + json.dumps(self.record), flush=True)
        self.record = None
