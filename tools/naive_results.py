"""Shared identity and diagnostics for offline Naive checks."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path


def fixture_identity(path: str | Path) -> dict:
    root = Path(path)
    files = {}
    for file in sorted(root.rglob("*")):
        if file.is_file() and file.suffix != ".pyc" and "__pycache__" not in file.parts:
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            files[str(file.relative_to(root))] = digest.hexdigest()
    if not files or not any(name.endswith(".safetensors") for name in files):
        raise ValueError(f"No checkpoint tensors in {root}")
    return {"path": str(path), "files": files}


def metadata(model: str | Path) -> dict:
    import torch
    import transformers

    commit = os.environ.get("MINISGL_VALIDATION_COMMIT")
    if commit is None:
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL, timeout=5
            ).strip()
        except (OSError, subprocess.SubprocessError):
            commit = "unknown"
    return {
        "commit": commit,
        "fixture": fixture_identity(model),
        "hardware": {"device": torch.cuda.get_device_name(), "platform": platform.platform()},
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
        },
    }


def emit(result: dict) -> None:
    print("NAIVE_RESULT=" + json.dumps(result), flush=True)


if __name__ == "__main__":
    import sys

    emit(fixture_identity(sys.argv[1]))
