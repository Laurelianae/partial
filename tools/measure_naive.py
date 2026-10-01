"""Native-only fixture prefill/decode measurements; no reference allocations."""

from __future__ import annotations

import argparse
import time

import torch
from minisgl.core import Batch, Req
from minisgl.distributed import DistributedInfo
from minisgl.engine import Engine, EngineConfig
from naive_results import emit, metadata


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--nnodes", type=int, choices=(1, 2), default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--tp-size", type=int)
    parser.add_argument("--dist-init-addr", default="127.0.0.1:29610")
    for name, default in [("prompt-length", 128), ("batch-size", 1), ("warmup", 2), ("repeat", 5)]:
        parser.add_argument("--" + name, type=int, default=default)
    args = parser.parse_args()
    if min(args.prompt_length, args.batch_size, args.repeat) < 1 or args.warmup < 0:
        parser.error("workload sizes/repeat must be positive; warmup must be nonnegative")
    if args.node_rank not in range(args.nnodes) or args.tp_size not in (None, args.nnodes):
        parser.error("Measurement requires one TP rank per node")
    engine = Engine(
        EngineConfig(
            model_path=args.model,
            dtype=getattr(torch, args.dtype),
            tp_info=DistributedInfo(args.node_rank, args.nnodes),
            nnodes=args.nnodes,
            local_gpu_index=0,
            dist_init_addr=args.dist_init_addr,
            use_pynccl=False,
            num_page_override=args.batch_size * (args.prompt_length + 1),
            max_running_req=args.batch_size,
        )
    )
    try:
        if not engine.model_config.is_naive or engine.dtype != getattr(torch, args.dtype):
            raise ValueError("Measurement requires native Naive execution in the requested dtype")

        def execute(decode: bool = False) -> None:
            length = args.prompt_length + int(decode)
            reqs = [
                Req(
                    torch.full((length,), 100, dtype=torch.int32),
                    row,
                    args.prompt_length if decode else 0,
                    1,
                    row,
                    None,
                    None,
                )
                for row in range(args.batch_size)
            ]
            batch = Batch(reqs, "decode" if decode else "prefill")
            batch.padded_reqs = reqs
            for row in range(args.batch_size):
                engine.page_table[row, :length] = torch.arange(
                    row * (args.prompt_length + 1),
                    row * (args.prompt_length + 1) + length,
                    device=engine.device,
                )
            batch.input_ids = torch.full(
                (args.batch_size * (1 if decode else length),),
                100,
                dtype=torch.int32,
                device=engine.device,
            )
            batch.positions = (
                torch.full(
                    (args.batch_size,), args.prompt_length, dtype=torch.int32, device=engine.device
                )
                if decode
                else torch.arange(length, dtype=torch.int32, device=engine.device).repeat(
                    args.batch_size
                )
            )
            batch.out_loc = torch.cat(
                [engine.page_table[r.table_idx, r.cached_len : r.device_len] for r in reqs]
            )
            engine.attn_backend.prepare_metadata(batch)
            with engine.ctx.forward_batch(batch):
                engine.model.forward()

        measurements = {}
        for phase in ("prefill", "decode"):
            execute()  # Populate cached history before decode.
            for _ in range(args.warmup):
                execute(phase == "decode")
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            resident = {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
            }
            samples = []
            for _ in range(args.repeat):
                if args.nnodes > 1:
                    torch.distributed.barrier(group=engine.tp_cpu_group)
                torch.cuda.synchronize()
                start = time.perf_counter()
                execute(phase == "decode")
                torch.cuda.synchronize()
                samples.append(time.perf_counter() - start)
            free, total = torch.cuda.mem_get_info()
            measurements[phase] = {
                "seconds": samples,
                "resident_pytorch_bytes": resident,
                "peak_pytorch_bytes": {
                    "allocated": torch.cuda.max_memory_allocated(),
                    "reserved": torch.cuda.max_memory_reserved(),
                },
                "device_bytes": {"free": free, "total": total},
            }
        emit(
            {
                **metadata(args.model),
                "rank": args.node_rank,
                "tp_size": args.nnodes,
                "dtype": args.dtype,
                "backend": "naive-eager",
                "workload": vars(args),
                "measurements": measurements,
            }
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
