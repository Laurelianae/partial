"""Run one TP parity worker with the same node transport settings as serving."""

from __future__ import annotations

import subprocess
import sys

from serve import remote_command


def main() -> int:
    node, model, dtype, *extra_args = sys.argv[1:]
    command = remote_command(int(node), model, ["--dtype", dtype, *extra_args], False)
    worker = command.index("tools/serve_worker.py")
    command[worker + 1 : worker + 1] = ["--check-naive-parity"]
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
