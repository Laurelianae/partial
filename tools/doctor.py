import platform
import sys

import torch


def main() -> None:
    print("python:", platform.python_version())
    print("executable:", sys.executable)
    print("architecture:", platform.machine())
    print("torch:", torch.__version__)
    print("cuda:", torch.version.cuda)

    available = torch.cuda.is_available()
    print("available:", available)

    if not available:
        raise SystemExit("ERROR: CUDA is unavailable in this environment.")

    torch.cuda.set_device(0)
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))

    # Exercise an actual GPU operation, not just device discovery.
    result = torch.arange(
        4, dtype=torch.float32, device="cuda:0"
    ).sum().item()

    if result != 6.0:
        raise SystemExit(
            f"ERROR: CUDA calculation returned {result}, expected 6.0."
        )

    print("CUDA calculation: OK")


if __name__ == "__main__":
    main()