"""Opt-in operation tracing and identical-input precision replays for chunk_stability.

Only this process is patched, with restoration in finally. Traces live on CPU and
are released after one case. No change to model/serving implementations is needed.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from chunk_stability import error_metrics, logit_digest
from minisgl.layers import BaseOP
from minisgl.models.autoround import GPTQProjection
from minisgl.models.naive import gather_last_dim


@contextmanager
def fp32_arithmetic():
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


def replay_linear(x: torch.Tensor, weight: torch.Tensor, bias, groups: list[torch.Tensor]) -> dict:
    """Partition rows without changing their values, weights, or output width."""
    result = {}
    with fp32_arithmetic():
        for name, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
            inputs, w = x.to(dtype), weight.to(dtype)
            b = None if bias is None else bias.to(dtype)
            full = F.linear(inputs, w, b)
            chunked = torch.empty_like(full)
            for rows in groups:
                if rows.numel():
                    chunked[rows] = F.linear(inputs[rows], w, b)
            result[name] = error_metrics(full, chunked)
    result.update(
        input_shape=list(x.shape),
        weight_shape=list(weight.shape),
        group_sizes=[g.numel() for g in groups],
        tf32=False,
    )
    return result


def aligned_rows(positions: torch.Tensor, stored: torch.Tensor) -> torch.Tensor:
    if positions.unique().numel() != positions.numel():
        raise AssertionError("Duplicate absolute trace positions")
    if positions.numel() and (positions.min() < 0 or positions.max() >= len(stored)):
        raise AssertionError("Trace position outside baseline")
    return stored[positions.long()]


class Trace:
    def __init__(self, engine, length: int, prompt_length: int, chunk: int) -> None:
        self.engine, self.length, self.prompt_length, self.chunk = (
            engine,
            length,
            prompt_length,
            chunk,
        )
        self.reference: dict[str, torch.Tensor] = {}
        self.coverage: dict[str, torch.Tensor] = {}
        self.inputs: dict[str, torch.Tensor] = {}
        self.order: list[str] = []
        self.differences: dict[str, dict] = {}
        self.replays: dict[str, dict] = {}
        self.modules: dict[str, BaseOP] = {}
        self.baseline = True
        self.patches = []
        self.attn_layer = ""
        self.sparse_offset = 0
        self.attention_offset = 0
        self.attention_inputs = {}

    def positions(self) -> torch.Tensor:
        return self.engine.ctx.batch.positions.cpu().long()

    def record(self, name, value, positions=None, selection=False, inputs=None) -> None:
        positions = self.positions() if positions is None else positions.cpu().long()
        value = value.detach().cpu()
        if value.shape[0] != len(positions):
            raise AssertionError(f"Unaligned trace: {name}")
        if self.baseline:
            if name not in self.reference:
                self.order.append(name)
                self.reference[name] = torch.empty(
                    (self.length, *value.shape[1:]), dtype=value.dtype
                )
                self.coverage[name] = torch.zeros(self.length, dtype=torch.bool)
            if self.coverage[name][positions].any():
                raise AssertionError(f"Repeated baseline positions: {name}")
            self.reference[name][positions] = value
            self.coverage[name][positions] = True
            if inputs is not None:
                if name not in self.inputs:
                    self.inputs[name] = torch.empty(
                        (self.length, *inputs.shape[1:]), dtype=inputs.dtype
                    )
                self.inputs[name][positions] = inputs.detach().cpu()
        else:
            if self.coverage[name][positions].any():
                raise AssertionError(f"Repeated candidate positions: {name}")
            target = aligned_rows(positions, self.reference[name])
            different = (target != value).reshape(len(positions), -1).any(-1)
            if different.any():
                absolute = int(positions[different].min())
                stats = self.differences.setdefault(
                    name,
                    {
                        "operation": name,
                        "first_position": absolute,
                        "selection": selection,
                        "max_abs": 0.0,
                        "max_batch_relative_l2": 0.0,
                    },
                )
                stats["first_position"] = min(stats["first_position"], absolute)
                metric = error_metrics(target, value)
                stats["max_abs"] = max(stats["max_abs"], metric["max_abs"])
                stats["max_batch_relative_l2"] = max(
                    stats["max_batch_relative_l2"], metric["relative_l2"]
                )
            self.coverage[name][positions] = True

    def patch(self, obj, attr, replacement) -> None:
        old = getattr(obj, attr)
        self.patches.append((obj, attr, old, attr in vars(obj)))
        setattr(obj, attr, replacement)

    def groups(self, count: int, device) -> list[torch.Tensor]:
        return [
            torch.arange(start, min(start + self.chunk, count), device=device)
            for start in range(0, count, self.chunk)
        ]

    def linear_replay(self, name, module) -> None:
        if (
            name in self.replays
            or name not in self.differences
            or sum(r.get("kind") == "linear" for r in self.replays.values()) >= 3
        ):
            return
        if name not in self.inputs or not hasattr(module, "weight") or module.weight.ndim != 2:
            return
        # Output-sharded row projections include a gather/padding operation; replay
        # their underlying full input separately, rather than changing TP semantics.
        x = self.inputs[name][: self.prompt_length].to(module.weight.device)
        weight, bias = module.weight, getattr(module, "bias", None)
        if getattr(module, "_output_sharded", False):
            tp = self.engine.tp_info
            width = weight.shape[0]
            padding = (tp.rank * width, (tp.size - tp.rank - 1) * width)
            weight = F.pad(weight, (0, 0, *padding))
            if bias is not None:
                bias = F.pad(bias, padding)
        if x.shape[-1] != module.weight.shape[-1]:
            return
        self.replays[name] = replay_linear(x, weight, bias, self.groups(len(x), x.device))
        self.replays[name]["kind"] = "linear"

    def replay_linears(self) -> None:
        # An early operation's difference can first appear in the last chunk.
        # Select in aligned operation order once all chunks have been compared.
        for name in self.order:
            if name in self.modules:
                self.linear_replay(name, self.modules[name])

    def expert_replay(self, name, module) -> None:
        key = name + ".expert_gate_up"
        if any(k.endswith("expert_gate_up") for k in self.replays) or name not in self.differences:
            return
        router = name.removesuffix("experts") + "gate.selection"
        selected = self.reference[router][: self.prompt_length]
        x = self.inputs[name][: self.prompt_length]
        # Pick an expert used in the differing row, then retain its full slot-major
        # token order. Decode-only differences use the last prompt row as a probe.
        position = min(self.differences[name]["first_position"], self.prompt_length - 1)
        expert = int(selected[position, 0])
        _, token = torch.where(selected.T == expert)
        if isinstance(getattr(module, "gate_proj", None), GPTQProjection):
            # Runtime decode performs FP16 dequantization then rounds to BF16.
            # Promote those exact reconstructed values; never recover original weights.
            weight = torch.cat((module.gate_proj.forward(expert), module.up_proj.forward(expert)))
        else:
            weight = module.gate_up_proj[expert]
        inputs = x[token].to(weight.device)
        groups = [
            torch.where((token >= start) & (token < start + self.chunk))[0].to(weight.device)
            for start in range(0, self.prompt_length, self.chunk)
        ]
        replay = replay_linear(inputs, weight, None, groups)
        replay.update(
            expert=expert,
            weight_values="runtime BF16 before FP32 promotion",
            scope="local fused gate/up projection; routing held fixed",
        )
        self.replays[key] = replay
        del weight, inputs

    def install(self) -> None:
        import minisgl.attention.naive as attention_module

        sparse = attention_module.sparse_mask
        attention = attention_module.attention

        def attention_wrapper(q, k, v, allowed, value_scale, sink):
            result = attention(q, k, v, allowed, value_scale, sink)
            positions = self.positions()[self.attention_offset : self.attention_offset + q.shape[1]]
            self.attention_offset += q.shape[1]
            name = self.attn_layer + ".core_attention"
            self.record(name, result.transpose(0, 1), positions)
            if self.baseline and int(positions[0]) < self.prompt_length:
                if name not in self.attention_inputs:
                    self.attention_inputs[name] = {
                        "q": torch.empty(
                            (q.shape[0], self.prompt_length, q.shape[-1]), dtype=q.dtype
                        ),
                        "k": k.cpu(),
                        "v": v.cpu(),
                        "mask": torch.empty(
                            (self.prompt_length, self.prompt_length), dtype=torch.bool
                        ),
                        "scale": value_scale,
                        "sink": None if sink is None else sink.cpu(),
                    }
                saved = self.attention_inputs[name]
                saved["q"][:, positions] = q.cpu()
                saved["mask"][positions] = allowed.cpu()
            elif (
                not self.baseline
                and name in self.differences
                and not any(k.endswith("core_attention") for k in self.replays)
            ):
                saved = self.attention_inputs[name]
                replay = {}
                with fp32_arithmetic():
                    for label, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
                        sq, sk, sv = [
                            saved[key].to(device=q.device, dtype=dtype) for key in ("q", "k", "v")
                        ]
                        mask = saved["mask"].to(q.device)
                        ss = (
                            None
                            if saved["sink"] is None
                            else saved["sink"].to(device=q.device, dtype=dtype)
                        )
                        outputs = []
                        for size in (self.prompt_length, self.chunk):
                            parts = []
                            for start in range(0, self.prompt_length, size):
                                total = min(start + size, self.prompt_length)
                                inner_size = (
                                    self.engine.model_config.naive_config.attention_chunk_size
                                )
                                for inner in range(start, total, inner_size):
                                    end = min(inner + inner_size, total)
                                    parts.append(
                                        attention(
                                            sq[:, inner:end],
                                            sk[:, :total],
                                            sv[:, :total],
                                            mask[inner:end, :total],
                                            saved["scale"],
                                            ss,
                                        )
                                    )
                            outputs.append(torch.cat(parts, dim=1))
                        replay[label] = error_metrics(*outputs)
                replay.update(
                    tf32=False,
                    scope="attention; identical Q/K/V, frozen baseline sparse mask",
                    input_shape=list(saved["q"].shape),
                )
                self.replays[name] = replay
            return result

        def sparse_wrapper(scores, allowed, top_k):
            result = sparse(scores, allowed, top_k)
            positions = self.positions()[self.sparse_offset : self.sparse_offset + len(scores)]
            self.sparse_offset += len(scores)
            padded = F.pad(result, (0, self.length - result.shape[-1]))
            self.record(self.attn_layer + ".sparse_selection", padded, positions, selection=True)
            return result

        self.patch(attention_module, "sparse_mask", sparse_wrapper)
        self.patch(attention_module, "attention", attention_wrapper)

        def walk(module, name):
            if isinstance(module, GPTQProjection):
                return
            self.modules[name] = module
            for child_name, child in list(vars(module).items()):
                if isinstance(child, BaseOP):
                    walk(child, name + "." + child_name)
            original = module.forward

            def forward(*args, **kwargs):
                if name.endswith("self_attn"):
                    self.attn_layer, self.sparse_offset = name, 0
                    self.attention_offset = 0
                # Residual entering post-attention normalization is separately visible.
                if name.endswith("post_attention_layernorm"):
                    self.record(name + ".residual_input", args[0])
                result = original(*args, **kwargs)
                if name.endswith("mlp.gate"):
                    self.record(name + ".selection", result[0], selection=True)
                    self.record(name + ".selection_set", result[0].sort(-1).values, selection=True)
                    self.record(name + ".weights", result[1])
                elif isinstance(result, torch.Tensor):
                    save_input = (
                        hasattr(module, "weight") and module.weight.ndim == 2
                    ) or name.endswith("experts")
                    captured = args[0] if save_input else None
                    if (
                        self.baseline
                        and save_input
                        and hasattr(module, "weight")
                        and getattr(module, "_output_sharded", False)
                    ):
                        captured = gather_last_dim(captured, module._comm)
                    self.record(name, result, inputs=captured)
                    if not self.baseline:
                        if name.endswith("experts"):
                            self.expert_replay(name, module)
                return result

            self.patch(module, "forward", forward)

        for i, layer in enumerate(self.engine.model.model.layers.op_list):
            walk(layer, f"layer.{i}")
        walk(self.engine.model.model.norm, "final_norm")

    def finish_pass(self) -> None:
        for name, seen in self.coverage.items():
            if not seen.all():
                raise AssertionError(f"Missing absolute trace positions: {name}")
            seen.zero_()

    def close(self) -> None:
        for obj, attr, original, owned in reversed(self.patches):
            if owned:
                setattr(obj, attr, original)
            else:
                delattr(obj, attr)
        self.patches.clear()
        self.reference.clear()
        self.inputs.clear()
        self.modules.clear()
        self.attention_inputs.clear()


def trace_case(history, ids: list[int], tokens: list[int], chunk: int) -> dict:
    trace = Trace(history.engine, len(ids) + 8, len(ids), chunk)
    try:
        trace.install()
        baseline = history.run(ids, len(ids), tokens)
        trace.finish_pass()
        trace.baseline = False
        candidate = history.run(ids, chunk, tokens)
        trace.finish_pass()
        trace.replay_linears()
        differences = [trace.differences[name] for name in trace.order if name in trace.differences]
        return {
            "baseline_logit_digest": logit_digest(baseline["logits"]),
            "candidate_logit_digest": logit_digest(candidate["logits"]),
            "alignment": "absolute token positions; complete coverage asserted for every operation",
            "ordering": "forward operation order, then earliest absolute position within operation",
            "first_numerical_difference": next(
                (d for d in differences if not d["selection"]), None
            ),
            "first_selection_change": next((d for d in differences if d["selection"]), None),
            "differences": differences,
            "replays": trace.replays,
            "prompt_logit_error": error_metrics(baseline["logits"][0], candidate["logits"][0]),
            "attribution": "Component replays isolate arithmetic on identical inputs. They do not establish end-to-end FP32 parity or exclude upstream implementation errors.",
        }
    finally:
        trace.close()
