"""Reproducible streaming API baseline on the two Sparks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from chunk_stability_remote import preflight
from naive_profile import collect_profiles
from naive_results import emit
from naive_runner import command, run_job


def source_identity() -> dict:
    files = []
    for root in ("python", "tools", "tests", "benchmark", "docs"):
        files.extend(
            p
            for p in Path(root).rglob("*")
            if p.is_file()
            and p.suffix
            in (".py", ".json", ".md", ".cu", ".cuh", ".h", ".cpp", ".sh", ".hpp", ".cc")
        )
    files.extend(
        p for p in (Path("justfile"), Path("pyproject.toml"), Path("uv.lock")) if p.exists()
    )
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def distribution(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "min": ordered[0],
        "max": ordered[-1],
    }


def validate_results(
    records: list[dict], tp: int, warmup: int, repeat: int, tokens: int, telemetry: bool = True
) -> dict:
    workloads = [r for r in records if r.get("kind") == "workload"]
    if len(workloads) != 1 or sum(r.get("kind") == "client_complete" for r in records) != 1:
        raise ValueError("Missing or duplicate workload/completion record")
    cases = {c["name"]: c for c in workloads[0]["cases"]}
    samples = [r for r in records if r.get("kind") == "sample"]
    expected = {(name, i) for name in cases for i in range(warmup + repeat)}
    if len(samples) != len(expected) or {(s["case"], s["index"]) for s in samples} != expected:
        raise ValueError("Incomplete or duplicate request samples")
    if len({s["uid"] for s in samples}) != len(samples):
        raise ValueError("Duplicate request IDs")
    traces = [r for r in records if r.get("kind") == "telemetry"]
    by_key = {(r["uid"], r["rank"]): r for r in traces}
    if telemetry and (len(traces) != len(samples) * tp or len(by_key) != len(traces)):
        raise ValueError("Missing or duplicate rank telemetry")
    for sample in samples:
        if len(sample["token_seconds"]) != tokens or sample["warmup"] != (sample["index"] < warmup):
            raise ValueError("Invalid sample token count or warmup marker")
        if not telemetry:
            continue
        ranks = [by_key.get((sample["uid"], rank)) for rank in range(tp)]
        if any(r is None for r in ranks):
            raise ValueError("Missing request rank telemetry")
        for rank in ranks:
            if rank["input_ids"] != cases[sample["case"]]["input_ids"]:
                raise ValueError("Prompt tokens differ from saved workload")
            if len(rank["output_ids"]) != tokens or len(rank["token_seconds"]) != tokens:
                raise ValueError("Telemetry token count mismatch")
            if rank["output_ids"] != ranks[0]["output_ids"]:
                raise ValueError("TP output token disagreement")
            for phase in ("resident_bytes", "prefill_memory_bytes", "decode_memory_bytes"):
                if any(
                    rank[phase][key] < 0
                    for key in (
                        "allocated",
                        "reserved",
                        "peak_allocated",
                        "peak_reserved",
                        "device_free",
                        "device_total",
                    )
                ):
                    raise ValueError("Invalid memory record")
            times = rank["token_seconds"]
            if any(b <= a for a, b in zip([0.0, *times], times)):
                raise ValueError("Nonmonotonic rank timing")
    result = {}
    for name in cases:
        measured = [s for s in samples if s["case"] == name and not s["warmup"]]
        result[name] = {
            key: distribution([s[key] for s in measured])
            for key in (
                "ttft_seconds",
                "decode_tokens_per_second",
                "end_to_end_seconds",
                "end_to_end_tokens_per_second",
            )
        }
        result[name]["decode_latency_seconds"] = distribution(
            [latency for s in measured for latency in s["decode_latency_seconds"]]
        )
        if telemetry:
            result[name]["ranks"] = {}
            for rank in range(tp):
                rows = [by_key[(s["uid"], rank)] for s in measured]
                result[name]["ranks"][rank] = {
                    "prefill_seconds": distribution([r["prefill_seconds"] for r in rows]),
                    "peak_allocated_bytes": max(
                        r[p]["peak_allocated"]
                        for r in rows
                        for p in ("prefill_memory_bytes", "decode_memory_bytes")
                    ),
                    "peak_reserved_bytes": max(
                        r[p]["peak_reserved"]
                        for r in rows
                        for p in ("prefill_memory_bytes", "decode_memory_bytes")
                    ),
                }
            outputs = [by_key[(s["uid"], 0)]["output_ids"] for s in measured]
            result[name]["repeated_tokens_identical"] = all(o == outputs[0] for o in outputs)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", default="~/models/Naive-N0.5-Flash-Int4")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--tp-size", type=int, choices=(1, 2), default=2)
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=28800)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--without-telemetry", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(".cache/naive-baseline"))
    args = parser.parse_args()
    if args.preflight:
        preflight(Path(args.model), args.fixture)
        emit({"kind": "baseline_source", "files": source_identity()})
        return
    if (
        min(args.sessions, args.repeat) < 1
        or args.tokens < 2
        or args.warmup < 0
        or any(not math.isfinite(t) or t <= 0 for t in (args.timeout, args.request_timeout))
    ):
        parser.error("Invalid counts or timeouts")
    if not args.fixture and (args.tp_size != 2 or args.without_telemetry):
        parser.error("Production baseline requires TP=2 and telemetry")
    if args.profile and (args.without_telemetry or args.tokens < 13):
        parser.error(
            "Profiling requires telemetry and at least 13 output tokens (one after trace export)"
        )
    if args.profile and args.output == Path(".cache/naive-baseline"):
        args.output = Path(".cache/naive-profile")
    directory = args.output / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8])
    directory.mkdir(parents=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    os.environ["MINISGL_VALIDATION_COMMIT"] = commit
    report = {
        "status": "running",
        "command": sys.argv,
        "commit": commit,
        "source_files": source_identity(),
        "settings": {**vars(args), "output": str(args.output)},
        "jobs": [],
        "summaries": [],
    }
    (directory / "source.diff").write_text(
        subprocess.check_output(["git", "diff", "HEAD"], text=True)
    )
    (directory / "git-status.txt").write_text(
        subprocess.check_output(["git", "status", "--short"], text=True)
    )
    for name in report["source_files"]:
        target = directory / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(name).read_bytes())

    def save():
        (directory / "results.json").write_text(json.dumps(report, indent=2))

    def job(name, commands, service_ranks=()):
        target = directory / name
        target.mkdir()
        entry = {"name": name, "commands": commands, "status": "running"}
        report["jobs"].append(entry)
        save()
        print(f"Running {name}; logs: {target}", flush=True)
        try:
            rows = run_job(commands, target, args.timeout, service_ranks)
            entry.update(status="passed", results=rows)
            return rows
        except BaseException as exc:
            entry.update(status="failed", error=repr(exc))
            raise
        finally:
            save()

    try:
        extra = ["--fixture"] if args.fixture else []
        identities = job(
            "preflight",
            [
                command(node, "tools/naive_baseline.py", [args.model, "--preflight", *extra])
                for node in range(args.tp_size)
            ],
        )
        checkpoints = [r for r in identities if r.get("kind") == "preflight"]
        sources = [r for r in identities if r.get("kind") == "baseline_source"]
        if len(checkpoints) != args.tp_size or any(
            r["fixture"]["files"] != checkpoints[0]["fixture"]["files"] for r in checkpoints
        ):
            raise ValueError("Missing or mismatched checkpoint identities")
        if len(sources) != args.tp_size or any(
            r["files"] != report["source_files"] for r in sources
        ):
            raise ValueError("Remote source differs from saved source")
        extra += [
            "--warmup",
            str(args.warmup),
            "--repeat",
            str(args.repeat),
            "--tokens",
            str(args.tokens),
            "--request-timeout",
            str(args.request_timeout),
        ]
        if args.without_telemetry:
            extra.append("--without-telemetry")
        for session in range(args.sessions):
            remote_profile = f".cache/naive-profile/{directory.name}/session-{session}"
            profile_args = ["--profile-dir", remote_profile] if args.profile else []
            records = job(
                f"session-{session}",
                [
                    command(
                        node,
                        "tools/naive_baseline_worker.py",
                        (
                            [*extra, *profile_args]
                            if args.tp_size == 2
                            else ["--model", args.model, *extra, *profile_args]
                        ),
                        args.model,
                        args.tp_size,
                    )
                    for node in range(args.tp_size)
                ],
                (1,) if args.tp_size == 2 else (),
            )
            report["summaries"].append(
                validate_results(
                    records,
                    args.tp_size,
                    args.warmup,
                    args.repeat,
                    args.tokens,
                    not args.without_telemetry,
                )
            )
            if args.profile:
                report.setdefault("profiles", []).append(
                    collect_profiles(
                        records,
                        args.tp_size,
                        remote_profile,
                        directory / f"session-{session}" / "traces",
                        args.timeout,
                    )
                )
            save()
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        save()
        print(f"Baseline {report['status']}: {directory / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
