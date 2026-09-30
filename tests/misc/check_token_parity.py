"""Capture actual generated token IDs with a small scheduler harness on remote GPUs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.message import UserMsg
from minisgl.scheduler import Scheduler, SchedulerConfig


class RequestsFinished(Exception):
    pass


class TokenCheckScheduler(Scheduler):
    """Replace frontend I/O with fixed requests while keeping real scheduling and inference."""

    def offline_receive_msg(self, blocking=False):
        if self.pending:
            pending, self.pending = self.pending, []
            return pending
        if blocking:
            raise RequestsFinished
        return []

    def offline_send_result(self, replies):
        for reply in replies:
            self.output_ids[reply.uid].append(reply.next_token)

    def generate_tokens(self, prompts, params):
        self.pending = [
            UserMsg(
                uid=uid,
                input_ids=torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int32),
                sampling_params=params,
            )
            for uid, prompt in enumerate(prompts)
        ]
        self.output_ids = {uid: [] for uid in range(len(prompts))}
        try:
            self.run_forever()
        except RequestsFinished:
            return list(self.output_ids.values())


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nnodes", type=int, choices=(1, 2), default=1)
    parser.add_argument("--node-rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--dist-init-addr", default="127.0.0.1:29600")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--write-baseline", type=Path)
    args = parser.parse_args()
    scheduler = TokenCheckScheduler(
        SchedulerConfig(
            model_path="Qwen/Qwen3-0.6B",
            dtype=torch.bfloat16,
            tp_info=DistributedInfo(args.node_rank, args.nnodes),
            nnodes=args.nnodes,
            local_gpu_index=0,
            dist_init_addr=args.dist_init_addr,
            use_pynccl=False,
            offline_mode=True,
            cache_type="naive",
            num_page_override=1024,
            cuda_graph_max_bs=8,
            max_running_req=4,
            max_extend_tokens=128,
        )
    )
    try:
        prompts = [
            "The capital of France is",
            "The capital of Germany is",
            "The capital of Italy is",
        ]
        greedy = scheduler.generate_tokens(
            prompts, SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
        )
        if args.baseline:
            assert greedy == json.loads(args.baseline.read_text()), "TP=1 and TP=2 token IDs differ"
        if args.write_baseline and args.node_rank == 0:
            args.write_baseline.write_text(json.dumps(greedy))
        sampled = scheduler.generate_tokens(
            prompts,
            SamplingParams(temperature=0.7, top_k=20, top_p=0.9, max_tokens=8, ignore_eos=True),
        )
        if args.nnodes == 2:
            outputs = [None, None]
            torch.distributed.all_gather_object(
                outputs, sampled, group=scheduler.engine.tp_cpu_group
            )
            assert outputs[0] == outputs[1], "TP ranks used different sampled tokens"
        print(f"rank {args.node_rank}: greedy token IDs {greedy}", flush=True)
        print(f"rank {args.node_rank}: token parity checks passed", flush=True)
    finally:
        scheduler.shutdown()


if __name__ == "__main__":
    main()
