"""Offline, batch-one Naive chunk sensitivity experiment (no serving changes)."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import torch
from minisgl.core import Batch, Req
from minisgl.distributed import DistributedInfo
from minisgl.engine import Engine, EngineConfig
from naive_results import emit


def error_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict:
    a, b = reference.double(), actual.double()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Mismatched shapes or nonfinite diagnostic values")
    delta = b - a
    return {
        "max_abs": delta.abs().max().item(),
        "relative_l2": (delta.norm() / a.norm().clamp_min(1e-300)).item(),
    }


def logit_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict:
    result = error_metrics(reference, actual)
    for name, values in (("baseline", reference), ("candidate", actual)):
        top = values.float().topk(2)
        # topk's tie ordering need not match argmax's first-index greedy rule.
        result[name + "_winner"] = int(values.argmax())
        result[name + "_margin"] = (top.values[0] - top.values[1]).item()
    result["winner_changed"] = result["baseline_winner"] != result["candidate_winner"]
    return result


def first_divergence(a: list[int], b: list[int]) -> int | None:
    if len(a) != len(b):
        raise ValueError("Unequal generation lengths")
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


def logit_digest(outputs: list[torch.Tensor]) -> str:
    return hashlib.sha256(
        b"".join(x.view(torch.uint8).numpy().tobytes() for x in outputs)
    ).hexdigest()


def trace_target(cases: list[dict]) -> tuple[int, int]:
    candidates = []
    for index, case in enumerate(cases):
        for chunk in case["chunks"]:
            metrics = chunk["forced_history"]
            changed = chunk["first_divergent_token"] is not None or any(
                m["winner_changed"] for m in metrics
            )
            candidates.append(
                (
                    not changed,
                    len(case["ids"]) if changed else 0,
                    -max(m["max_abs"] for m in metrics),
                    index,
                    chunk["size"],
                )
            )
    return min(candidates)[-2:]


def require_rank_agreement(engine: Engine, value: object) -> None:
    if engine.tp_info.size > 1:
        values = [None] * engine.tp_info.size
        torch.distributed.all_gather_object(values, value, group=engine.tp_cpu_group)
        if any(v != values[0] for v in values):
            raise AssertionError("TP ranks disagree")


def prompts(model: Path, fixture: bool) -> list[dict]:
    if fixture:
        return [
            {"name": "fixture-129", "ids": [100 + i % 37 for i in range(129)]},
            {"name": "fixture-257", "ids": [100 + i % 37 for i in range(257)]},
            {"name": "fixture-2049", "ids": [100 + i % 37 for i in range(2049)]},
            {"name": "known-int4-132", "ids": [100 + i % 37 for i in range(129)] + [203, 204, 205]},
        ]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    specifications = [
        (
            "factual",
            129,
            "Paris is the capital of France. Berlin is the capital of Germany. "
            "Rome is the capital of Italy. Madrid is the capital of Spain. ",
            "\nQuestion: What is the capital of France?\nAnswer:",
        ),
        (
            "code",
            257,
            "def square(x):\n    return x * x\n\ndef add(a, b):\n    return a + b\n\n",
            "\n# Complete the Python function.\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return",
        ),
        (
            "retrieval",
            2049,
            "Archive record: the gardener planted blue flowers near the stone wall. ",
            "\nQuestion: What is the secret archive code stated at the beginning?\nAnswer:",
        ),
    ]
    result = []
    for name, length, body, suffix in specifications:
        prefix = "The secret archive code is ORCHID-731.\n" if name == "retrieval" else ""
        start = tokenizer.encode(prefix, add_special_tokens=False)
        end = tokenizer.encode(suffix, add_special_tokens=False)
        middle = tokenizer.encode(body * length, add_special_tokens=False)
        ids = start + middle[: length - len(start) - len(end)] + end
        assert len(ids) == length
        result.append({"name": name, "ids": ids, "decoded_text": tokenizer.decode(ids)})
    return result


class History:
    """Each run rebuilds every visible KV/index entry at identical physical locations."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.ids: list[int] = []

    def reset(self) -> None:
        self.ids = []
        cache = self.engine.ctx.kv_cache
        # Poison all storage: an accidental stale-history read becomes a hard failure.
        for bank in (cache._keys, cache._values, cache._index):
            for tensor in bank:
                if tensor is not None:
                    tensor.fill_(float("nan"))

    def append(self, ids: list[int], decode: bool = False) -> torch.Tensor:
        e, cached = self.engine, len(self.ids)
        self.ids.extend(ids)
        total = len(self.ids)
        e.page_table[0, :total] = torch.arange(total, device=e.device)
        req = Req(torch.tensor(self.ids, dtype=torch.int32), 0, cached, 1, 0, None, None)
        batch = Batch([req], "decode" if decode else "prefill")
        batch.padded_reqs = batch.reqs
        batch.input_ids = torch.tensor(ids, dtype=torch.int32, device=e.device)
        batch.positions = torch.arange(cached, total, dtype=torch.int32, device=e.device)
        batch.out_loc = e.page_table[0, cached:total]
        e.attn_backend.prepare_metadata(batch)
        with e.ctx.forward_batch(batch):
            logits = e.model.forward()[0].detach().cpu()
        if not torch.isfinite(logits).all():
            raise AssertionError("Nonfinite logits, possibly stale cache access")
        require_rank_agreement(
            e, hashlib.sha256(logits.view(torch.uint8).numpy().tobytes()).hexdigest()
        )
        return logits

    def run(self, prompt: list[int], chunk: int, forced: list[int] | None = None) -> dict:
        self.reset()
        started = time.monotonic()
        for offset in range(0, len(prompt), chunk):
            logits = self.append(prompt[offset : offset + chunk])
            consumed = min(offset + chunk, len(prompt))
            if (offset // chunk + 1) % 16 == 0 or consumed == len(prompt):
                print(
                    f"rank={self.engine.tp_info.rank} chunk={chunk} "
                    f"history={'baseline-forced' if forced is not None else 'greedy'} "
                    f"prefill={consumed}/{len(prompt)} elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
        outputs, generated = [logits], []
        for step in range(8):
            token = int(logits.argmax()) if forced is None else forced[step]
            generated.append(token)
            logits = self.append([token], decode=True)
            outputs.append(logits)
        print(f"history complete elapsed={time.monotonic() - started:.1f}s", flush=True)
        return {"logits": outputs, "tokens": generated}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--nnodes", type=int, choices=(1, 2), default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--tp-size", type=int)
    parser.add_argument("--dist-init-addr", default="127.0.0.1:29610")
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--skip-trace", action="store_true")
    parser.add_argument("--case", help="Run only this named history")
    parser.add_argument(
        "--chunks", type=int, nargs="+", choices=(13, 64, 128), default=[13, 64, 128]
    )
    args = parser.parse_args()
    if args.node_rank not in range(args.nnodes) or args.tp_size not in (None, args.nnodes):
        parser.error("Requires one rank per node")
    args.model = args.model.expanduser()
    cases = prompts(args.model, args.fixture)
    if args.case:
        cases = [case for case in cases if case["name"] == args.case]
        if not cases:
            parser.error("Unknown history for this checkpoint mode")
    if len(set(args.chunks)) != len(args.chunks):
        parser.error("Chunk sizes must be unique")
    engine = Engine(
        EngineConfig(
            model_path=str(args.model),
            dtype=torch.bfloat16,
            tp_info=DistributedInfo(args.node_rank, args.nnodes),
            nnodes=args.nnodes,
            local_gpu_index=0,
            dist_init_addr=args.dist_init_addr,
            use_pynccl=False,
            num_page_override=2060,
            max_running_req=1,
            distributed_timeout=600,
        )
    )
    try:
        history = History(engine)
        require_rank_agreement(engine, cases)
        results = []
        for case in cases:
            ids = case["ids"]
            print(f"case={case['name']} length={len(ids)} baseline", flush=True)
            baseline = history.run(ids, len(ids))
            repeat = history.run(ids, len(ids), baseline["tokens"])
            record = {
                **case,
                "baseline_tokens": baseline["tokens"],
                "baseline_logit_digest": logit_digest(baseline["logits"]),
                "baseline_repeat": [
                    logit_metrics(a, b) for a, b in zip(baseline["logits"], repeat["logits"])
                ],
                "chunks": [],
            }
            for chunk in args.chunks:
                print(f"case={case['name']} chunk={chunk}", flush=True)
                forced = history.run(ids, chunk, baseline["tokens"])
                independent = history.run(ids, chunk)
                metrics = [
                    logit_metrics(a, b) for a, b in zip(baseline["logits"], forced["logits"])
                ]
                divergence = first_divergence(baseline["tokens"], independent["tokens"])
                record["chunks"].append(
                    {
                        "size": chunk,
                        "forced_history": metrics,
                        "forced_logit_digest": logit_digest(forced["logits"]),
                        "independent_tokens": independent["tokens"],
                        "first_divergent_token": divergence,
                    }
                )
            results.append(record)
            emit({"kind": "case", "rank": args.node_rank, "case": record})
        require_rank_agreement(engine, results)
        trace = None
        if not args.skip_trace:
            from chunk_trace import trace_case

            case_id, chunk = trace_target(results)
            case = results[case_id]
            trace = trace_case(history, case["ids"], case["baseline_tokens"], chunk)
            measured = next(c for c in case["chunks"] if c["size"] == chunk)
            if (
                trace["baseline_logit_digest"] != case["baseline_logit_digest"]
                or trace["candidate_logit_digest"] != measured["forced_logit_digest"]
            ):
                raise AssertionError("Traced outputs differ from uninstrumented measurement")
            trace.update(case=case["name"], chunk=chunk)
        emit(
            {
                "kind": "complete",
                "rank": args.node_rank,
                "tp_size": args.nnodes,
                "rank_agreement": True,
                "cases": results,
                "trace": trace,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                "tf32_during_measurement": torch.backends.cuda.matmul.allow_tf32,
            }
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
