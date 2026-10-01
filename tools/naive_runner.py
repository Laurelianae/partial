"""Supervise reproducible remote regression and native-only measurement jobs."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path
from uuid import uuid4

from serve import remote_command, stop_nodes


def command(
    node: int, script: str, args: list[str], model: str | None = None, tp: int = 1
) -> list[str]:
    if tp == 2:
        cmd = remote_command(node, model, args, True)
        cmd.insert(
            cmd.index(".venv/bin/python"),
            "MINISGL_VALIDATION_COMMIT=" + os.environ.get("MINISGL_VALIDATION_COMMIT", "unknown"),
        )
        index = cmd.index("tools/serve_worker.py")
        cmd[index + 2 : index + 2] = ["--run-command", script]
        return cmd
    return [
        "bash",
        "tools/remote.sh",
        str(node),
        "exec",
        "env",
        "MINISGL_VALIDATION_COMMIT=" + os.environ.get("MINISGL_VALIDATION_COMMIT", "unknown"),
        ".venv/bin/python",
        "tools/serve_worker.py",
        "--watch-stdin",
        "--run-command",
        script,
        *args,
    ]


def run_job(
    commands: list[list[str]], directory: Path, timeout: float, service_ranks: tuple[int, ...] = ()
) -> list[dict]:
    """Wait for every successful rank; fail immediately if any rank fails."""
    processes, logs = [], []
    started = time.monotonic()
    try:
        for rank, cmd in enumerate(commands):
            log = (directory / f"rank-{rank}.log").open("w")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
            )
        while True:
            codes = [process.poll() for process in processes]
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError(f"Worker failure: {codes}")
            if all(code == 0 for rank, code in enumerate(codes) if rank not in service_ranks):
                break
            if time.monotonic() - started > timeout:
                raise TimeoutError(f"Job exceeded {timeout}s")
            time.sleep(0.1)
    finally:
        stop_nodes(processes)
        for log in logs:
            log.close()
    results = []
    for rank in range(len(commands)):
        for line in (directory / f"rank-{rank}.log").read_text().splitlines():
            if line.startswith("NAIVE_RESULT="):
                results.append(json.loads(line.split("=", 1)[1]))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("regression", "measure"))
    parser.add_argument("fixture")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--tp-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path, default=Path(".cache/naive-results"))
    args, extra = parser.parse_known_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.mode == "regression" and extra:
        parser.error("unknown regression arguments: " + " ".join(extra))
    if args.mode == "measure":
        if args.quick:
            parser.error("--quick applies only to regression")
        workload = argparse.ArgumentParser()
        workload.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
        for name, default in [
            ("prompt-length", 128),
            ("batch-size", 1),
            ("warmup", 2),
            ("repeat", 5),
        ]:
            workload.add_argument("--" + name, type=int, default=default)
        settings = workload.parse_args(extra)
        if (
            min(settings.prompt_length, settings.batch_size, settings.repeat) < 1
            or settings.warmup < 0
        ):
            parser.error("workload sizes/repeat must be positive; warmup must be nonnegative")
        extra = [
            value
            for name, value in vars(settings).items()
            for value in ("--" + name.replace("_", "-"), str(value))
        ]
    output = args.output / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "mode": args.mode,
        "fixture": args.fixture,
        "jobs": [],
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }

    os.environ["MINISGL_VALIDATION_COMMIT"] = report["commit"]

    def job(
        name: str, commands: list[list[str]], service_ranks: tuple[int, ...] = ()
    ) -> list[dict]:
        directory = output / name
        directory.mkdir()
        record = {"name": name, "commands": commands, "status": "running"}
        report["jobs"].append(record)
        print(f"Running {name}; logs: {directory}", flush=True)
        try:
            record["results"] = run_job(commands, directory, args.timeout, service_ranks)
            record["status"] = "passed"
            return record["results"]
        except BaseException as error:
            record.update(status="failed", error=repr(error))
            raise
        finally:
            (output / "results.json").write_text(json.dumps(report, indent=2))

    def execute_jobs() -> None:
        nodes = (0,) if args.quick or (args.mode == "measure" and args.tp_size == 1) else (0, 1)
        identities = []
        for node in nodes:
            result = job(
                f"identity-{node}", [command(node, "tools/naive_results.py", [args.fixture])]
            )
            if len(result) != 1:
                raise RuntimeError("Missing fixture identity")
            identities.append(result[0])
        if any(identity["files"] != identities[0]["files"] for identity in identities):
            raise RuntimeError("Fixture hashes differ across nodes")
        if args.mode == "measure":
            results = job(
                "measurement",
                [
                    command(
                        node,
                        "tools/measure_naive.py",
                        ["--model", args.fixture, *extra] if args.tp_size == 1 else extra,
                        args.fixture,
                        args.tp_size,
                    )
                    for node in range(args.tp_size)
                ],
            )
            if sorted(r["rank"] for r in results) != list(range(args.tp_size)):
                raise RuntimeError("Incomplete measurement ranks")
            report["slowest_rank_seconds"] = {
                phase: [
                    max(r["measurements"][phase]["seconds"][i] for r in results)
                    for i in range(len(results[0]["measurements"][phase]["seconds"]))
                ]
                for phase in ("prefill", "decode")
            }
            (output / "results.json").write_text(json.dumps(report, indent=2))
            return
        job(
            "core",
            [
                command(
                    0,
                    "-m",
                    [
                        "pytest",
                        "tests/core",
                        "tests/misc/test_multinode.py",
                        "tests/misc/test_naive_runner.py",
                        "--no-cov",
                        "-q",
                    ],
                )
            ],
        )
        for tp in ((1,) if args.quick else (1, 2)):
            for dtype in (
                ("bfloat16",)
                if args.quick or identities[0].get("quantized")
                else ("float32", "bfloat16")
            ):
                parity_args = ["--dtype", dtype]
                if not args.quick:
                    parity_args += ["--production-top-k", "--page-size", "4"]
                results = job(
                    f"parity-tp{tp}-{dtype}",
                    [
                        command(
                            node,
                            "tests/misc/check_naive_parity.py",
                            ["--model", args.fixture, *parity_args] if tp == 1 else parity_args,
                            args.fixture,
                            tp,
                        )
                        for node in range(tp)
                    ],
                )
                if sorted(r["rank"] for r in results) != list(range(tp)):
                    raise RuntimeError("Incomplete parity ranks")
        if not args.quick:
            job(
                "naive-serving",
                [command(0, "tests/misc/check_local_serving.py", ["--model", args.fixture])],
            )
            serving_args = [
                "--port",
                "1930",
                "--num-pages",
                "512",
                "--max-prefill-length",
                "64",
                "--max-running-requests",
                "8",
                "--graph",
                "0",
            ]
            job(
                "naive-serving-tp2",
                [
                    command(
                        0, "tests/misc/check_local_serving.py", ["--port", "1930"], args.fixture, 2
                    ),
                    remote_command(1, args.fixture, serving_args, True),
                ],
                service_ranks=(1,),
            )
            job(
                "qwen-serving",
                [
                    command(
                        0,
                        "tests/misc/check_local_serving.py",
                        ["--model", "Qwen/Qwen3-0.6B", "--graph", "4", "--cache-type", "naive"],
                    )
                ],
            )

    try:
        execute_jobs()
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        (output / "results.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
