"""Start a supervised single-node server, exercise its API, and always stop its workers."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=1929)
    parser.add_argument("--graph", type=int, default=0)
    parser.add_argument("--cache-type", choices=("radix", "naive"), default="radix")
    parser.add_argument("--no-overlap", action="store_true")
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dist-init-addr")
    args = parser.parse_args()
    topology = []
    if args.nnodes > 1:
        topology = [
            "--nnodes",
            str(args.nnodes),
            "--node-rank",
            str(args.node_rank),
            "--tp-size",
            str(args.tp_size),
            "--dist-init-addr",
            args.dist_init_addr,
        ]
    url = f"http://127.0.0.1:{args.port}"
    try:
        with urlopen(url + "/v1/models", timeout=1):
            raise RuntimeError("Test port is already serving; choose another --port")
    except URLError:
        pass
    with tempfile.TemporaryFile(mode="w+") as log:
        import os

        environment = os.environ.copy()
        if args.no_overlap:
            environment["MINISGL_DISABLE_OVERLAP_SCHEDULING"] = "1"
        process = subprocess.Popen(
            [
                sys.executable,
                "tools/serve_worker.py",
                "--watch-stdin",
                "--model",
                args.model,
                "--port",
                str(args.port),
                "--graph",
                str(args.graph),
                "--num-pages",
                "512",
                "--max-prefill-length",
                "64",
                "--max-running-requests",
                "8",
                "--cache-type",
                args.cache_type,
                *topology,
            ],
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 120
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Server exited before readiness")
                try:
                    with urlopen(url + "/v1/models", timeout=1):
                        break
                except URLError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Server did not become ready")
                    time.sleep(0.2)
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("check_serving.py")),
                    "--url",
                    url,
                    "--model",
                    args.model,
                ],
                check=True,
            )
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            log.seek(0)
            print(log.read(), flush=True)


if __name__ == "__main__":
    main()
