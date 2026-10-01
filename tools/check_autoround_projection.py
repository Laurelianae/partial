"""Compare one trained packed expert projection with the independent reference on CUDA."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from autoround_reference import reference_projection
from minisgl.distributed import set_tp_info
from minisgl.models.autoround import dequantize, inspect_checkpoint
from minisgl.models.naive import output_shard_linear
from minisgl.models.naive_config import NaiveN05FlashConfig
from safetensors import safe_open


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument(
        "--projection", choices=("gate_proj", "up_proj", "down_proj"), required=True
    )
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--tokens", type=int, default=13)
    args = parser.parse_args()
    if args.tokens <= 0:
        parser.error("--tokens must be positive")
    set_tp_info(args.rank, args.tp_size)
    config = NaiveN05FlashConfig(**json.loads((args.model / "config.json").read_text()))
    report = inspect_checkpoint(args.model, config, args.tp_size)
    prefix = f"model.layers.{args.layer}.mlp.experts.{args.expert}.{args.projection}."
    parts, local = {}, {}
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for component in ("qweight", "qzeros", "scales"):
        name = prefix + component
        if name not in report["entries"]:
            parser.error(f"Projection not found: {name}")
        with safe_open(report["entries"][name]["file"], framework="pt", device="cpu") as reader:
            parts[component] = reader.get_tensor(name).cuda()
            view = reader.get_slice(name)
            width = view.get_shape()[1] // args.tp_size
            local[component] = view[:, args.rank * width : (args.rank + 1) * width].cuda()
    reference = reference_projection(parts)
    actual = dequantize(**local)
    torch.testing.assert_close(actual, reference.chunk(args.tp_size, 0)[args.rank], atol=0, rtol=0)
    torch.manual_seed(42)
    states = torch.randn(args.tokens, actual.shape[1], device="cuda", dtype=torch.bfloat16)
    wanted = torch.nn.functional.linear(states, reference).chunk(args.tp_size, -1)[args.rank]
    out = output_shard_linear(states, actual)
    torch.testing.assert_close(out, wanted, atol=0.025, rtol=0.025)
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "model": str(args.model),
                "projection": prefix,
                "rank": args.rank,
                "tp_size": args.tp_size,
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "elapsed_seconds": time.monotonic() - started,
                "max_projection_error": (out.float() - wanted.float()).abs().max().item(),
                "weight_reconstruction_exact": True,
                "reference_check_peak_pytorch_bytes": torch.cuda.max_memory_allocated(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
