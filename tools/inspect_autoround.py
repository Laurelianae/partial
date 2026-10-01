"""Inspect Naive AutoRound headers without reading tensor payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minisgl.models.autoround import inspect_checkpoint
from minisgl.models.naive_config import NaiveN05FlashConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--tp-size", type=int, default=2)
    args = parser.parse_args()
    config = NaiveN05FlashConfig(**json.loads((args.model / "config.json").read_text()))
    report = inspect_checkpoint(args.model, config, args.tp_size)
    report.pop("entries")
    report["tp_size"] = args.tp_size
    report["qualification"] = "pending trained-weight projection and generation checks"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
