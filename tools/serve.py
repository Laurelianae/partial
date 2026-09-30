"""Launch one or both Sparks using the SSH settings loaded by just."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time

FORWARDED_ENV = (
    "NCCL_SOCKET_IFNAME",
    "NCCL_IB_HCA",
    "GLOO_SOCKET_IFNAME",
    "NCCL_DEBUG",
    "MINISGL_DISABLE_OVERLAP_SCHEDULING",
    "MINISGL_FLASHINFER_USE_TENSOR_CORES",
)


def remote_command(node: int, model: str, extra_args: list[str], watch_stdin: bool) -> list[str]:
    master = os.environ.get("PARTIAL_MASTER_ADDR")
    if not master:
        raise ValueError("Set PARTIAL_MASTER_ADDR in .env.local to node 0's ConnectX address")
    environment = ["PYTHONUNBUFFERED=1"]
    for name in FORWARDED_ENV:
        value = os.environ.get(f"PARTIAL_NODE_{node}_{name}", os.environ.get(name))
        if value:
            environment.append(f"{name}={value}")
    command = [
        "bash",
        "tools/remote.sh",
        str(node),
        "exec",
        "env",
        *environment,
        ".venv/bin/python",
        "tools/serve_worker.py",
    ]
    if watch_stdin:
        command.append("--watch-stdin")
    return command + [
        "--model",
        model,
        *extra_args,
        "--nnodes",
        "2",
        "--node-rank",
        str(node),
        "--tp-size",
        "2",
        "--dist-init-addr",
        f"{master}:{os.environ.get('PARTIAL_MASTER_PORT', '29500')}",
    ]


def print_node_output(node: int, process: subprocess.Popen) -> None:
    for line in process.stdout:
        print(f"[node {node}] {line}", end="", flush=True)


def stop_nodes(processes: list[subprocess.Popen]) -> None:
    # EOF tells the remote wrapper to stop its managed process group, even over SSH.
    for process in processes:
        process.stdin.close()
    deadline = time.monotonic() + 15.0
    for process in processes:
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", type=int, choices=(0, 1))
    parser.add_argument("model")
    parser.add_argument("server_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.node is not None:
        return subprocess.call(remote_command(args.node, args.model, args.server_args, False))

    processes = []
    readers = []
    try:
        for node in (0, 1):
            process = subprocess.Popen(
                remote_command(node, args.model, args.server_args, True),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            processes.append(process)
            reader = threading.Thread(target=print_node_output, args=(node, process), daemon=True)
            reader.start()
            readers.append(reader)
        while True:
            for node, process in enumerate(processes):
                code = process.poll()
                if code is not None:
                    print(f"[node {node}] launcher exited with code {code}", flush=True)
                    return code if code >= 0 else 128 - code
            time.sleep(0.2)
    except KeyboardInterrupt:
        return 130
    finally:
        stop_nodes(processes)
        for reader in readers:
            reader.join(timeout=1.0)


if __name__ == "__main__":
    sys.exit(main())
