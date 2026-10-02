"""Collect and attribute bounded Naive CPU/CUDA traces without summing nested ranges."""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path


def validate_manifest(records: list[dict], tp: int) -> list[dict]:
    samples = [r for r in records if r.get("kind") == "sample" and not r["warmup"]]
    expected = {
        (s["uid"], rank, phase)
        for s in samples
        for rank in range(tp)
        for phase in ("prefill", "decode")
    }
    traces = [r for r in records if r.get("kind") == "profile_trace"]
    keys = {(r["uid"], r["rank"], r["phase"]) for r in traces}
    if keys != expected or len(keys) != len(traces) or any(not r["complete"] for r in traces):
        raise ValueError("Missing, duplicate, unexpected, or incomplete profile traces")
    by_uid = {s["uid"]: s for s in samples}
    all_samples = [r for r in records if r.get("kind") == "sample"]
    order = {s["uid"]: i for i, s in enumerate(all_samples)}
    for trace in traces:
        steps = [0] if trace["phase"] == "prefill" else [8, 9, 10, 11]
        if trace["steps"] != steps or trace["request"] != order[trace["uid"]]:
            raise ValueError("Profile request/step association mismatch")
        trace.update(case=by_uid[trace["uid"]]["case"], index=by_uid[trace["uid"]]["index"])
    return traces


def union_us(intervals: list[tuple[float, float]]) -> float:
    end = float("-inf")
    total = 0.0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def analyze_trace(path: Path) -> dict:
    with gzip.open(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    cpu = defaultdict(list)
    kernels, runtime = [], []
    for event in events:
        if event.get("ph") != "X" or "dur" not in event:
            continue
        cat = event.get("cat", "")
        if cat in ("cpu_op", "user_annotation"):
            cpu[(event["pid"], event["tid"])].append(event)
        elif cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            kernels.append(event)
        elif cat in ("cuda_runtime", "cuda_driver"):
            runtime.append(event)
    if not kernels or not any(e.get("cat") == "kernel" for e in kernels):
        raise ValueError(f"No CUDA kernel events in {path}")
    external = {}
    cpu_self = defaultdict(float)
    operators = defaultdict(float)
    steps = []
    for thread in cpu.values():
        stack = []
        for event in sorted(thread, key=lambda e: (e["ts"], -e["dur"])):
            start, end = event["ts"], event["ts"] + event["dur"]
            while stack and start >= stack[-1][0]:
                stack.pop()
            parent_category = stack[-1][1] if stack else "other"
            name = event["name"]
            category = parent_category
            if name.startswith("naive::"):
                label = name.removeprefix("naive::")
                if label.startswith("step/"):
                    steps.append((start, end, int(label.split("/")[1])))
                elif not label.startswith("layer/"):
                    category = label
            # Exclusive CPU time: subtract direct children's full intervals only.
            cpu_self[category] += event["dur"]
            if stack:
                cpu_self[parent_category] -= event["dur"]
                operators[stack[-1][2]] -= event["dur"]
            operators[name] += event["dur"]
            identifier = event.get("args", {}).get("External id")
            if identifier is not None:
                external[identifier] = category
            stack.append((end, category, name))
    correlations = {
        e.get("args", {}).get("correlation"): external.get(
            e.get("args", {}).get("External id"), "other"
        )
        for e in runtime
    }
    gpu = defaultdict(list)
    kernel_names = defaultdict(float)
    for event in kernels:
        args = event.get("args", {})
        category = external.get(
            args.get("External id"), correlations.get(args.get("correlation"), "other")
        )
        gpu[category].append((event["ts"], event["ts"] + event["dur"]))
        kernel_names[event["name"]] += event["dur"]
    all_intervals = [pair for values in gpu.values() for pair in values]
    busy = union_us(all_intervals)
    step_span = union_us([(a, b) for a, b, _ in steps])
    # Clip kernels to captured steps for an idle estimate, excluding between-step gaps.
    clipped = [
        (max(a, c), min(b, d))
        for a, b in all_intervals
        for c, d, _ in steps
        if max(a, c) < min(b, d)
    ]
    runtime_names = defaultdict(float)
    for e in runtime:
        runtime_names[e["name"]] += e["dur"]

    def seconds(values):
        return {k: v / 1e6 for k, v in sorted(values.items(), key=lambda p: -p[1])}

    return {
        "steps": sorted(step for _, _, step in steps),
        "cpu_exclusive_seconds": seconds(cpu_self),
        "cpu_operator_self_seconds": seconds(operators),
        "cuda_runtime_seconds": seconds(runtime_names),
        "gpu_category_busy_seconds": seconds({k: union_us(v) for k, v in gpu.items()}),
        "gpu_category_kernel_sum_seconds": seconds(
            {k: sum(b - a for a, b in v) for k, v in gpu.items()}
        ),
        "gpu_kernel_sum_seconds": seconds(kernel_names),
        "gpu_busy_seconds": busy / 1e6,
        "step_wall_seconds": step_span / 1e6,
        "step_gpu_idle_seconds": (step_span - union_us(clipped)) / 1e6,
        "kernel_events": sum(e.get("cat") == "kernel" for e in kernels),
    }


def collect_profiles(
    records: list[dict], tp: int, remote_directory: str, directory: Path, timeout: float
) -> list[dict]:
    manifest = validate_manifest(records, tp)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    for trace in manifest:
        remote = Path(trace["path"])
        expected_name = f"rank-{trace['rank']}-request-{trace['request']}-{trace['phase']}.json.gz"
        if remote.parent != Path(remote_directory) or remote.name != expected_name:
            raise ValueError("Unexpected remote trace path")
        local = directory / remote.name
        with local.open("wb") as stream:
            subprocess.run(
                ["bash", "tools/remote.sh", str(trace["rank"]), "exec", "cat", str(remote)],
                stdout=stream,
                check=True,
                timeout=timeout,
            )
        trace["local_path"] = str(local)
        trace["sha256"] = hashlib.sha256(local.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest, indent=2))
    for trace in manifest:
        try:
            trace["analysis"] = analyze_trace(Path(trace["local_path"]))
            if trace["analysis"]["steps"] != trace["steps"]:
                raise ValueError("Trace events disagree with declared capture steps")
        except BaseException as exc:
            trace["error"] = repr(exc)
            raise
        finally:
            manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest
