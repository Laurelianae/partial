"""Supervised TP=2 chunk investigation, with identity/resource checks before loading."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from naive_runner import command, run_job


def source_identity() -> dict:
    files = sorted(
        [
            *Path("python").rglob("*.py"),
            *Path("tools").glob("*.py"),
            *Path("tests").rglob("*.py"),
            Path("justfile"),
            Path("pyproject.toml"),
        ]
    )
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def preflight(model: Path, fixture: bool) -> None:
    from naive_results import emit, metadata

    processes = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    memory = {
        line.split(":")[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.split()[1].isdigit()
    }
    if processes:
        raise RuntimeError(f"GPU processes already active; left untouched: {processes}")
    minimum = 2 * 1024**3 if fixture else 90 * 1024**3
    if memory["MemAvailable"] < minimum:
        raise RuntimeError(f"Insufficient available memory: {memory['MemAvailable']} < {minimum}")
    emit(
        {
            "kind": "preflight",
            **metadata(model.expanduser()),
            "source_files": source_identity(),
            "gpu_processes": processes,
            "mem_available": memory["MemAvailable"],
            "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
        }
    )


def validate_results(results: list[dict]) -> list[dict]:
    completed = sorted((r for r in results if r.get("kind") == "complete"), key=lambda r: r["rank"])
    if [r["rank"] for r in completed] != [0, 1]:
        raise AssertionError("Missing or duplicate TP completion records")
    if (
        not all(r["rank_agreement"] for r in completed)
        or completed[0]["cases"] != completed[1]["cases"]
    ):
        raise AssertionError("TP results disagree")
    return completed


def findings(report: dict) -> str:
    lines = [
        "# Chunk-size stability investigation",
        "",
        f"Status: {report['status']}",
        "",
        "Batch size 1; TP=2; BF16 native execution; identical physical cache locations.",
        "Each run poisons and rebuilds KV/index storage. Eight teacher-forced decode steps",
        "use baseline tokens; eight independently greedy tokens are also compared.",
        "",
    ]
    completed = report.get("completed", [])
    if completed:
        lines += [
            "| History | Chunk | Max logit error | First divergent generated token | Changed prediction indices |",
            "|---|---:|---:|---:|---|",
        ]
        for case in completed[0]["cases"]:
            for chunk in case["chunks"]:
                lines.append(
                    f"| {case['name']} ({len(case['ids'])}) | {chunk['size']} | "
                    f"{max(m['max_abs'] for m in chunk['forced_history']):.8g} | "
                    f"{chunk['first_divergent_token']} | "
                    f"{[i for i, m in enumerate(chunk['forced_history']) if m['winner_changed']]} |"
                )
        repeat = max(m["max_abs"] for c in completed[0]["cases"] for m in c["baseline_repeat"])
        lines += [
            "",
            "Indices are zero-based. Prediction 0 uses prompt logits; prediction 8 follows",
            "eight decoded inputs and predicts token nine, beyond the independent eight-token window.",
            f"Repeated full-prefill baseline maximum error: {repeat:.8g}.",
            "Both TP ranks agreed exactly on measured logits and result summaries.",
        ]
        for rank in completed:
            trace = rank.get("trace")
            if trace:
                lines += [
                    "",
                    f"Rank {rank['rank']} traced {trace['case']}, chunk {trace['chunk']}:",
                    f"- First numerical difference: `{trace['first_numerical_difference']}`",
                    f"- First selection change: `{trace['first_selection_change']}`",
                ]
                for name, replay in trace["replays"].items():
                    lines.append(
                        f"- `{name}` identical-input shape error: BF16 "
                        f"{replay['bf16']['max_abs']:.8g}; FP32 {replay['fp32']['max_abs']:.8g}."
                    )
        lines += [
            "",
            "FP32 replays disable TF32 and promote existing BF16 values, including runtime",
            "INT4 reconstruction. They supply component-level arithmetic evidence, not an",
            "unquantized or full FP32 trained-model reference. Selection changes are observations;",
            "causal attribution beyond the replayed operations remains inconclusive.",
            "This focused sample does not estimate the frequency of instability in traffic.",
        ]
    if "error" in report:
        lines += ["", f"Failure: {report['error']}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--skip-trace", action="store_true")
    parser.add_argument("--case", help="Run only this named history")
    parser.add_argument(
        "--chunks", type=int, nargs="+", choices=(13, 64, 128), default=[13, 64, 128]
    )
    parser.add_argument("--timeout", type=float, default=14400)
    parser.add_argument("--output", type=Path, default=Path(".cache/chunk-stability"))
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")
    if len(set(args.chunks)) != len(args.chunks):
        parser.error("Chunk sizes must be unique")
    if args.preflight:
        preflight(args.model, args.fixture)
        return
    directory = args.output / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8])
    directory.mkdir(parents=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    os.environ["MINISGL_VALIDATION_COMMIT"] = commit
    report = {
        "status": "running",
        "command": sys.argv,
        "commit": commit,
        "source_files": source_identity(),
        "jobs": [],
    }
    (directory / "source.diff").write_text(
        subprocess.check_output(["git", "diff", "HEAD"], text=True)
    )
    # Include untracked diagnostic tools, which a git diff alone would omit.
    snapshot = directory / "source"
    for name in report["source_files"]:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(name).read_bytes())

    def save() -> None:
        (directory / "results.json").write_text(json.dumps(report, indent=2))
        (directory / "findings.md").write_text(findings(report))

    def job(name: str, commands: list[list[str]]) -> list[dict]:
        target = directory / name
        target.mkdir()
        record = {"name": name, "commands": commands, "status": "running"}
        report["jobs"].append(record)
        save()
        print(f"Running {name}; logs: {target}", flush=True)
        try:
            result = run_job(commands, target, args.timeout)
            record.update(status="passed", results=result)
            return result
        except BaseException as exc:
            record.update(status="failed", error=repr(exc))
            raise
        finally:
            save()

    try:
        extra = ["--fixture"] if args.fixture else []
        identities = job(
            "preflight",
            [
                command(
                    node,
                    "tools/chunk_stability_remote.py",
                    [str(args.model), "--preflight", *extra],
                )
                for node in (0, 1)
            ],
        )
        if (
            len(identities) != 2
            or identities[0]["fixture"]["files"] != identities[1]["fixture"]["files"]
        ):
            raise AssertionError("Checkpoint hashes differ or identities missing")
        if any(identity["source_files"] != report["source_files"] for identity in identities):
            raise AssertionError("Remote source differs from local source snapshot")
        if args.skip_trace:
            extra.append("--skip-trace")
        if args.case:
            extra.extend(["--case", args.case])
        extra.extend(["--chunks", *map(str, args.chunks)])
        result = job(
            "investigation",
            [
                command(node, "tools/chunk_stability.py", extra, str(args.model), 2)
                for node in (0, 1)
            ],
        )
        report["completed"] = validate_results(result)
        if (
            args.fixture
            and identities[0]["fixture"]["quantized"]
            and args.case in (None, "known-int4-132")
            and 13 in args.chunks
        ):
            known = next(
                c for c in report["completed"][0]["cases"] if c["name"] == "known-int4-132"
            )
            chunk13 = next(c for c in known["chunks"] if c["size"] == 13)
            if chunk13["forced_history"][0]["max_abs"] <= 0:
                raise AssertionError("Known chunk-dependent fixture was not detected")
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        save()
        print(f"Report: {directory / 'findings.md'}", flush=True)


if __name__ == "__main__":
    main()
