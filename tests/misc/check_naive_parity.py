"""Compare cached Mini-SGLang execution with Naive's pinned upstream eager reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from minisgl.attention.naive import sparse_mask
from minisgl.core import Batch, Req
from minisgl.distributed import DistributedInfo
from minisgl.engine import Engine, EngineConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from naive_reference import load_reference
from naive_results import emit, metadata


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--nnodes", type=int, choices=(1, 2), default=1)
    parser.add_argument("--node-rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--tp-size", type=int)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--production-top-k", action="store_true")
    parser.add_argument("--dist-init-addr", default="127.0.0.1:29610")
    args = parser.parse_args()
    if args.tp_size is not None and args.tp_size != args.nnodes:
        parser.error("This harness requires one TP worker per node")
    dtype = getattr(torch, args.dtype)
    engine = Engine(
        EngineConfig(
            model_path=str(args.model),
            dtype=dtype,
            tp_info=DistributedInfo(args.node_rank, args.nnodes),
            nnodes=args.nnodes,
            local_gpu_index=0,
            dist_init_addr=args.dist_init_addr,
            use_pynccl=False,
            num_page_override=4096,
            page_size=args.page_size,
            max_running_req=4,
        )
    )
    assert engine.dtype == dtype, f"Requested {dtype}, but engine uses {engine.dtype}"
    reference = load_reference(args.model, dtype, engine.device)
    max_error = 0.0
    max_relative_l2 = 0.0
    reference_caches = []
    reference_features = {}
    for name, module in reference.named_modules():
        if name.endswith(("input_layernorm", "post_attention_layernorm", "self_attn", "mlp")):
            module.register_forward_hook(
                lambda module, inputs, output, name=name: reference_features.__setitem__(
                    name, output.detach()
                )
            )

    def diagnose(batch):
        with engine.ctx.forward_batch(batch):
            states = engine.model.model.embed_tokens.forward(batch.input_ids)
            for i, layer in enumerate(engine.model.model.layers.op_list):
                norm = layer.input_layernorm.forward(states)
                attn = layer.self_attn.forward(norm)
                post = layer.post_attention_layernorm.forward(states + attn)
                mlp = layer.mlp.forward(post)
                for suffix, value in (
                    ("input_layernorm", norm),
                    ("self_attn", attn),
                    ("post_attention_layernorm", post),
                    ("mlp", mlp),
                ):
                    target = reference_features[f"model.layers.{i}.{suffix}"].reshape_as(value)
                    print(
                        f"layer={i} {suffix} max_error={(value.float()-target.float()).abs().max().item()}",
                        flush=True,
                    )
                states = states + attn + mlp

    def run(prompts, chunks, cached_prefix=0, location_offset=0, fragmented=False):
        nonlocal max_error, max_relative_l2, reference_caches
        if cached_prefix == 0:
            reference_caches = [None] * len(prompts)
        offset = location_offset
        for row, prompt in enumerate(prompts):
            engine.page_table[row, : len(prompt)] = torch.arange(
                offset, offset + len(prompt), device=engine.device
            )
            if fragmented:
                locations = engine.page_table[row, : len(prompt)]
                engine.page_table[row, : len(prompt)] = (
                    locations // args.page_size * 3 + 3
                ) * args.page_size + locations % args.page_size
            offset += len(prompt)
        consumed = [cached_prefix] * len(prompts)
        last = None
        for chunk in chunks:
            reqs, ids, positions = [], [], []
            expected = []
            for row, prompt in enumerate(prompts):
                cached = consumed[row]
                total = min(cached + chunk, len(prompt))
                if total == cached:
                    continue
                consumed[row] = total
                reqs.append(
                    Req(
                        torch.tensor(prompt[:total], dtype=torch.int32),
                        row,
                        cached,
                        1,
                        row,
                        None,
                        None,
                    )
                )
                ids.extend(prompt[cached:total])
                positions.extend(range(cached, total))
                output = reference(
                    input_ids=torch.tensor([prompt[cached:total]], device=engine.device),
                    past_key_values=reference_caches[row],
                    use_cache=True,
                )
                reference_caches[row] = output.past_key_values
                expected.append(output.logits[0, -1])
            if not reqs:
                continue
            batch = Batch(reqs, "decode" if chunk == 1 else "prefill")
            batch.padded_reqs = batch.reqs
            batch.input_ids = torch.tensor(ids, dtype=torch.int32, device=engine.device)
            batch.positions = torch.tensor(positions, dtype=torch.int32, device=engine.device)
            batch.out_loc = torch.cat(
                [engine.page_table[r.table_idx, r.cached_len : r.device_len] for r in reqs]
            )
            engine.attn_backend.prepare_metadata(batch)
            with engine.ctx.forward_batch(batch):
                actual = engine.model.forward()
            wanted = torch.stack(expected)
            error = (actual.float() - wanted.float()).abs().max().item()
            max_error = max(max_error, error)
            relative_l2 = ((actual.float() - wanted.float()).norm() / wanted.float().norm()).item()
            max_relative_l2 = max(max_relative_l2, relative_l2)
            tolerance = 2e-5 if dtype == torch.float32 else 0.025
            try:
                assert relative_l2 < (
                    2e-5 if dtype == torch.float32 else 0.05
                ), f"relative_l2={relative_l2}"
                torch.testing.assert_close(
                    actual.float(),
                    wanted.float(),
                    atol=tolerance,
                    rtol=tolerance,
                    msg=lambda message: f"requests={batch.attn_metadata.requests} chunk={chunk}\n{message}",
                )
            except AssertionError:
                if len(reqs) == 1:
                    diagnose(batch)
                raise
            assert torch.equal(actual.argmax(-1), wanted.argmax(-1)), "Greedy token IDs differ"
            last = actual
        return last

    def check_components():
        states = (
            torch.arange(
                17 * reference.config.hidden_size, device=engine.device, dtype=torch.float32
            )
            .sin()
            .view(17, -1)
            .to(dtype)
        )
        positions = torch.arange(17, device=engine.device, dtype=torch.int32)
        allowed = positions[:, None] >= positions[None, :]
        native_layers = engine.model.model.layers.op_list
        upstream_layers = reference.model.layers
        iq, ik, iw = native_layers[0].self_attn.indexer.forward(states, positions)
        scores = (
            (iq.transpose(0, 1).float() @ ik.float().T).relu() * iw.T.unsqueeze(-1).float()
        ).sum(0)
        embeddings = upstream_layers[0].self_attn.rotary_emb(states[None], positions[None])
        expected_scores = upstream_layers[0].self_attn.indexer(states[None], embeddings)[0]
        tolerance = 2e-5 if dtype == torch.float32 else 0.025
        torch.testing.assert_close(scores, expected_scores, atol=tolerance, rtol=tolerance)
        expected_indices = expected_scores.masked_fill(~allowed, -torch.inf).argsort(
            dim=-1, descending=True, stable=True
        )[:, :16]
        expected_mask = allowed & torch.zeros_like(allowed).scatter(-1, expected_indices, True)
        assert torch.equal(sparse_mask(scores, allowed, 16), expected_mask)
        selected, weights = native_layers[1].mlp.gate.forward(states)
        _, expected_weights, expected_selected = upstream_layers[1].mlp.gate(states[None])
        assert torch.equal(selected, expected_selected)
        torch.testing.assert_close(weights, expected_weights)
        actual_experts = native_layers[1].mlp.experts.forward(states, selected, weights)
        expected_experts = upstream_layers[1].mlp.experts(
            states, expected_selected, expected_weights
        )
        torch.testing.assert_close(actual_experts, expected_experts, atol=tolerance, rtol=tolerance)
        for layer_id in (0, 1):
            req = Req(torch.zeros(17, dtype=torch.int32), 0, 0, 1, 0, None, None)
            batch = Batch([req], "prefill")
            batch.padded_reqs, batch.positions = batch.reqs, positions
            batch.out_loc = positions
            engine.page_table[0, :17] = positions
            engine.attn_backend.prepare_metadata(batch)
            with engine.ctx.forward_batch(batch):
                actual_attention = native_layers[layer_id].self_attn.forward(states)
            expected_attention = upstream_layers[layer_id].self_attn(
                states[None], positions[None], allowed[None]
            )[0]
            torch.testing.assert_close(
                actual_attention, expected_attention, atol=tolerance, rtol=tolerance
            )
        print(
            f"rank={args.node_rank}: indexer, sparse selections, router, experts, and attention components passed",
            flush=True,
        )

    try:
        check_components()
        # Boundary prompts, unequal lengths, and token IDs from the actual vocabulary.
        for length in (15, 16, 17, 127, 128, 129, 257):
            prompt = [100 + i % 37 for i in range(length)]
            full = run([prompt], [length])
            chunked = run([prompt], [min(13, length)] * (length // 13 + 1))
            if dtype == torch.float32:
                torch.testing.assert_close(full.float(), chunked.float(), atol=2e-5, rtol=2e-5)
        if args.production_top_k:
            engine.model_config.naive_config.index_top_k = 2048
            reference.config.index_top_k = 2048
            run([[100 + i % 37 for i in range(2049)]], [2049])
            engine.model_config.naive_config.index_top_k = 16
            reference.config.index_top_k = 16
        run(
            [[100 + i % 37 for i in range(130)], [200 + i % 41 for i in range(129)]], [64, 64, 1, 1]
        )
        prefix = [100 + i % 37 for i in range(129)]
        continuation = prefix + [203, 204, 205]
        run([prefix], [129])
        reused = run([continuation], [1, 1, 1], cached_prefix=129)
        relocated = run([continuation], [132], location_offset=1000)
        if dtype == torch.float32:
            torch.testing.assert_close(reused.float(), relocated.float(), atol=2e-5, rtol=2e-5)
        fresh = run([continuation], [132])
        fragmented = run([continuation], [13] * 11, fragmented=True)
        tolerance = 2e-5 if dtype == torch.float32 else 0.025
        torch.testing.assert_close(
            fresh.float(), fragmented.float(), atol=tolerance, rtol=tolerance
        )
        # Cached greedy generation, using identical histories in both implementations.
        prompt = [100, 101, 102, 103, 104]
        greedy = []
        for step in range(8):
            logits = run(
                [prompt],
                [len(prompt)] if step == 0 else [1],
                cached_prefix=0 if step == 0 else len(prompt) - 1,
            )
            token = int(logits.argmax(-1).item())
            prompt.append(token)
            greedy.append(token)
        if args.nnodes == 2:
            outputs = [None, None]
            torch.distributed.all_gather_object(outputs, greedy, group=engine.tp_cpu_group)
            assert outputs[0] == outputs[1]
        print(
            f"rank={args.node_rank} dtype={dtype} max_logit_error={max_error:.8g} "
            f"max_relative_l2={max_relative_l2:.8g} greedy={greedy}",
            flush=True,
        )
        emit(
            {
                **metadata(args.model),
                "rank": args.node_rank,
                "tp_size": args.nnodes,
                "dtype": args.dtype,
                "backend": "naive-eager",
                "max_logit_error": max_error,
                "max_relative_l2": max_relative_l2,
                "generated_token_ids": greedy,
            }
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
