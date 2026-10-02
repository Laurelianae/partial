"""Native AutoRound GPTQ storage, checkpoint inspection, and eager decoding.

Packing is independent of the execution backend. No decoded expert is cached.
"""

from __future__ import annotations

import json
import math
import re
import struct
from pathlib import Path

import torch
from minisgl.layers import BaseOP
from minisgl.profiling import traced

EXPERT = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.(qweight|qzeros|scales|g_idx)$"
)
_DTYPES = {
    "I32": (torch.int32, 4),
    "F16": (torch.float16, 2),
    "BF16": (torch.bfloat16, 2),
    "F32": (torch.float32, 4),
}


def validate_quantization(config) -> bool:
    q = getattr(config, "quantization_config", None)
    if not q:
        return False
    if not isinstance(q, dict):
        raise ValueError("AutoRound quantization_config must be an object")
    if q.get("quant_method") not in ("auto-round", "auto_round"):
        raise ValueError("Unsupported Naive quant_method; expected auto-round")
    # AutoRound exports packing_format. Keep legacy fixture aliases as fallbacks;
    # a runtime backend selection must not override the serialized weight layout.
    format_field = next((key for key in ("packing_format", "format", "backend") if key in q), None)
    packing_format = q[format_field] if format_field is not None else None
    if packing_format != "auto_round:auto_gptq":
        raise ValueError(
            "Unsupported AutoRound format; expected auto_round:auto_gptq, "
            f"received {format_field or 'missing packing_format'}={packing_format!r}"
        )
    for field, expected in (("bits", 4), ("group_size", 128), ("sym", True)):
        if q.get(field) != expected:
            raise ValueError(f"Unsupported AutoRound {field}; expected {expected}")
    if q.get("desc_act", False) or q.get("act_bits", 16) != 16:
        raise ValueError("AutoRound activation quantization/order mapping is unsupported")
    if config.hidden_size % 128 or config.moe_intermediate_size % 128:
        raise ValueError("AutoRound expert input dimensions must be divisible by 128")
    return True


def projection_shapes(inputs: int, outputs: int) -> dict[str, tuple[int, int]]:
    return {
        "qweight": (inputs // 8, outputs),
        "qzeros": (inputs // 128, outputs // 8),
        "scales": (inputs // 128, outputs),
    }


@traced("int4_reconstruct")
def dequantize(qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """GPTQ input-major packing, zero + 1, and FP16 multiply before BF16 cast."""
    shifts = torch.arange(8, device=qweight.device, dtype=torch.int32) * 4
    values = (
        ((qweight[:, None, :] >> shifts[None, :, None]) & 15)
        .reshape(-1, qweight.shape[1])
        .to(torch.float16)
    )
    zeros = (((qzeros[:, :, None] >> shifts) & 15) + 1).reshape(qzeros.shape[0], -1)
    # Group views avoid materializing expanded scales or zero points.
    values = values.view(scales.shape[0], 128, -1)
    values.sub_(zeros[:, None, :].to(torch.float16))
    values.mul_(scales[:, None, :])
    return values.reshape(-1, scales.shape[1]).T.to(torch.bfloat16).contiguous()


class GPTQProjection(BaseOP):
    """Packed per-expert tensor state exposed through BaseOP's strict loader."""

    def __init__(self, experts: int, inputs: int, outputs: int) -> None:
        for name, shape in projection_shapes(inputs, outputs).items():
            setattr(
                self,
                name,
                torch.empty(
                    (experts, *shape), dtype=torch.float16 if name == "scales" else torch.int32
                ),
            )

    def forward(self, expert: int) -> torch.Tensor:
        return dequantize(self.qweight[expert], self.qzeros[expert], self.scales[expert])


def execution_workspace(config, tp_size: int) -> int:
    if not validate_quantization(config):
        return 0
    h, m = config.hidden_size, config.moe_intermediate_size
    # Decode FP16 + BF16 + contiguous transpose, gate/up concatenation, and
    # full-output padded down GEMM. Conservative bound, independent of expert count.
    return 16 * h * m + 8 * h * m // tp_size


def dense_shapes(c) -> dict[str, tuple[int, ...]]:
    """Full checkpoint schema for the split-attention Naive architecture."""
    shapes = {
        "model.embed_tokens.weight": (c.vocab_size, c.hidden_size),
        "model.norm.weight": (c.hidden_size,),
    }
    if not c.tie_word_embeddings:
        shapes["lm_head.weight"] = (c.vocab_size, c.hidden_size)
    for layer in range(c.num_hidden_layers):
        root = f"model.layers.{layer}."
        shapes[root + "input_layernorm.weight"] = (c.hidden_size,)
        shapes[root + "post_attention_layernorm.weight"] = (c.hidden_size,)
        prefix = "swa_" if c.hybrid_layer_pattern[layer] else ""
        heads = getattr(c, prefix + "num_attention_heads")
        kv = getattr(c, prefix + "num_key_value_heads")
        dim, vdim = getattr(c, prefix + "head_dim"), getattr(c, prefix + "v_head_dim")
        attn = root + "self_attn."
        for name, out in (("q_proj", heads * dim), ("k_proj", kv * dim), ("v_proj", kv * vdim)):
            shapes[attn + name + ".weight"] = (out, c.hidden_size)
            if c.attention_bias:
                shapes[attn + name + ".bias"] = (out,)
        shapes[attn + "o_proj.weight"] = (c.hidden_size, heads * vdim)
        if getattr(c, "add_swa_attention_sink_bias" if prefix else "add_full_attention_sink_bias"):
            shapes[attn + "attention_sink_bias"] = (heads,)
        if not prefix:
            for name, out in (
                ("wq", c.index_n_heads * c.index_head_dim),
                ("wk", c.index_head_dim),
                ("weights_proj", c.index_n_heads),
            ):
                shapes[attn + "indexer." + name + ".weight"] = (out, c.hidden_size)
            for component in ("weight", "bias"):
                shapes[attn + "indexer.k_norm." + component] = (c.index_head_dim,)
        mlp = root + "mlp."
        if c.moe_layer_freq[layer]:
            shapes[mlp + "gate.weight"] = (c.n_routed_experts, c.hidden_size)
            shapes[mlp + "gate.e_score_correction_bias"] = (c.n_routed_experts,)
        else:
            for name in ("gate_proj", "up_proj"):
                shapes[mlp + name + ".weight"] = (c.intermediate_size, c.hidden_size)
            shapes[mlp + "down_proj.weight"] = (c.hidden_size, c.intermediate_size)
    return shapes


def inspect_checkpoint(folder: str | Path, config, tp_size: int = 1) -> dict:
    """Read safetensors JSON headers only, including cross-file index validation."""
    try:
        return _inspect_checkpoint(folder, config, tp_size)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError, struct.error) as error:
        raise ValueError(f"Malformed checkpoint metadata: {error}") from error


def _inspect_checkpoint(folder: str | Path, config, tp_size: int) -> dict:
    if not validate_quantization(config):
        raise ValueError("Inspection requires an AutoRound Naive checkpoint")
    config.validate()
    if (
        tp_size <= 0
        or config.hidden_size % (8 * tp_size)
        or config.moe_intermediate_size % (8 * tp_size)
    ):
        raise ValueError("TP size must divide expert output dimensions and zero-point words")
    if config.intermediate_size % tp_size:
        raise ValueError("TP size must divide dense MLP output dimensions")
    for prefix in ("", "swa_"):
        heads = getattr(config, prefix + "num_attention_heads")
        kv = getattr(config, prefix + "num_key_value_heads")
        if heads % tp_size or (kv % tp_size if kv >= tp_size else tp_size % kv):
            raise ValueError("TP size is incompatible with attention heads")
    folder = Path(folder)
    entries = {}
    for path in sorted(folder.glob("*.safetensors")):
        if path.name == "consolidated.safetensors":
            continue
        with path.open("rb") as stream:
            length_data = stream.read(8)
            if len(length_data) != 8:
                raise ValueError(f"Truncated safetensors header: {path.name}")
            length = struct.unpack("<Q", length_data)[0]
            if length > 100_000_000:
                raise ValueError(f"Invalid safetensors header length: {path.name}")

            def unique_pairs(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError(f"Duplicate header entry: {key}")
                    result[key] = value
                return result

            header = json.loads(stream.read(length), object_pairs_hook=unique_pairs)
        end = 0
        for name, spec in sorted(
            ((k, v) for k, v in header.items() if k != "__metadata__"),
            key=lambda item: item[1]["data_offsets"][0],
        ):
            if name in entries:
                raise ValueError(f"Duplicate checkpoint tensor: {name}")
            shape, dt = spec["shape"], spec["dtype"]
            if dt not in _DTYPES or any(type(d) is not int or d <= 0 for d in shape):
                raise ValueError(f"Invalid tensor shape/dtype: {name}")
            start, stop = spec["data_offsets"]
            if (
                type(start) is not int
                or type(stop) is not int
                or start != end
                or stop - start != math.prod(shape) * _DTYPES[dt][1]
            ):
                raise ValueError(f"Invalid tensor offsets: {name}")
            end = stop
            entries[name] = {**spec, "file": str(path)}
        if 8 + length + end != path.stat().st_size:
            raise ValueError(f"Invalid safetensors payload length: {path.name}")
    if not entries:
        raise ValueError("No safetensors tensors found")
    index = folder / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text(), object_pairs_hook=unique_pairs)["weight_map"]
        if set(weight_map) != set(entries):
            raise ValueError("Checkpoint index entries do not match tensor headers")
        for name, filename in weight_map.items():
            if filename != Path(entries[name]["file"]).name:
                raise ValueError(f"Checkpoint index points to wrong shard: {name}")
    dense = dense_shapes(config)
    for name, shape in dense.items():
        if name not in entries or tuple(entries[name]["shape"]) != shape:
            raise ValueError(f"Missing or malformed Naive tensor: {name}; expected {shape}")
    required = set()
    for layer, routed in enumerate(config.moe_layer_freq):
        if not routed:
            continue
        for expert in range(config.n_routed_experts):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                inputs, outputs = (
                    (config.moe_intermediate_size, config.hidden_size)
                    if projection == "down_proj"
                    else (config.hidden_size, config.moe_intermediate_size)
                )
                prefix = f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
                for component, shape in projection_shapes(inputs, outputs).items():
                    name = prefix + "." + component
                    required.add(name)
                    spec = entries.get(name)
                    dt = "F16" if component == "scales" else "I32"
                    if spec is None or tuple(spec["shape"]) != shape or spec["dtype"] != dt:
                        raise ValueError(
                            f"Missing or malformed AutoRound tensor: {name}; expected {shape} {dt}"
                        )
    for name, spec in entries.items():
        if ".experts." in name and name not in required:
            raise ValueError(f"Unsupported AutoRound expert entry (including g_idx): {name}")
        if name not in required:
            if name not in dense:
                raise ValueError(f"Unexpected Naive checkpoint tensor: {name}")
            router = name.endswith(("mlp.gate.weight", "e_score_correction_bias"))
            if spec["dtype"] != ("F32" if router else "BF16"):
                raise ValueError(f"Expected {'FP32 router' if router else 'BF16'} tensor: {name}")
    total = sum(math.prod(e["shape"]) * _DTYPES[e["dtype"]][1] for e in entries.values())
    index_total_size = None
    if index.exists():
        metadata = json.loads(index.read_text()).get("metadata", {})
        index_total_size = metadata.get("total_size")
        # AutoRound's streamed exporter can leave total_size at zero. The validated
        # tensor headers and file extents supply the actual payload size instead.
        if index_total_size is not None and (
            type(index_total_size) is not int or index_total_size not in (0, total)
        ):
            raise ValueError("Checkpoint index total_size does not match headers")
    # Non-expert sharding follows the BF16 Naive loader; indexer/norm/router replicated.
    residents = [0] * tp_size
    for name, spec in entries.items():
        nbytes = math.prod(spec["shape"]) * _DTYPES[spec["dtype"]][1]
        split = name in required or (
            ".indexer." not in name
            and ".mlp.gate." not in name
            and any(
                s in name for s in ("_proj.", "embed_tokens.", "lm_head.", "attention_sink_bias")
            )
        )
        if any(p in name for p in (".k_proj.", ".v_proj.")) and ".indexer." not in name:
            layer = int(name.split(".layers.")[1].split(".")[0])
            heads = (
                config.swa_num_key_value_heads
                if config.hybrid_layer_pattern[layer]
                else config.num_key_value_heads
            )
            nbytes = nbytes // min(heads, tp_size)
        elif split:
            nbytes = nbytes // tp_size
        for rank in range(tp_size):
            rank_bytes = nbytes
            if name in ("model.embed_tokens.weight", "lm_head.weight"):
                width = (config.vocab_size + tp_size - 1) // tp_size
                rows = max(0, min(width, config.vocab_size - rank * width))
                rank_bytes = rows * config.hidden_size * _DTYPES[spec["dtype"]][1]
            residents[rank] += rank_bytes
    resident = max(residents)
    workspace = execution_workspace(config, tp_size)
    return {
        "checkpoint_bytes": total,
        "index_total_size_bytes": index_total_size,
        "per_rank_resident_bytes": resident,
        "rank_resident_bytes": residents,
        "execution_workspace_bytes": workspace,
        "per_rank_peak_weight_bytes": resident + workspace,
        "entries": entries,
    }
