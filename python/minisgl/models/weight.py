from __future__ import annotations

import glob
import re
from typing import Dict, Iterator, Tuple

import safetensors
import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import cached_load_hf_config, div_ceil, download_hf_weight
from tqdm import tqdm

_SPLIT_DIM_0 = [".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj"]
_SPLIT_DIM_1 = [".o_proj", ".down_proj"]

# Merge groups: individual projections -> fused projection
_MERGE_GROUPS = {
    ".q_proj": (".qkv_proj", ("q", "k", "v")),
    ".k_proj": (".qkv_proj", ("q", "k", "v")),
    ".v_proj": (".qkv_proj", ("q", "k", "v")),
    ".gate_proj": (".gate_up_proj", ("gate", "up")),
    ".up_proj": (".gate_up_proj", ("gate", "up")),
}
_SLOT_NAMES = {
    ".q_proj": "q",
    ".k_proj": "k",
    ".v_proj": "v",
    ".gate_proj": "gate",
    ".up_proj": "up",
}
_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")


def _shard_tensor(key: str, value: torch.Tensor, r: int, n: int, num_kv_heads: int):
    """Extract rank r's shard from a single tensor. Returns a contiguous copy."""
    if any(key.count(sub) for sub in _SPLIT_DIM_0):
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        if is_kv_proj and num_kv_heads is not None and num_kv_heads < n:
            head_dim = value.shape[0] // num_kv_heads
            head_idx = r * num_kv_heads // n
            return value[head_idx * head_dim : (head_idx + 1) * head_dim].clone()
        return value.chunk(n, dim=0)[r].clone()
    elif any(key.count(sub) for sub in _SPLIT_DIM_1):
        return value.chunk(n, dim=1)[r].clone()
    elif key.count("lm_head") or key.count("embed_tokens"):
        num_embeddings = value.shape[0]
        num_embeddings_per_partition = div_ceil(num_embeddings, n)
        vocab_start_idx = r * num_embeddings_per_partition
        vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
        return value[vocab_start_idx:vocab_end_idx, :].clone()
    else:
        return value


def _get_merge_info(key: str):
    """If key belongs to a merge group, return (merged_key, slot, all_slots). Else None."""
    for suffix, (fused_suffix, slots) in _MERGE_GROUPS.items():
        if key.count(suffix):
            return key.replace(suffix, fused_suffix), _SLOT_NAMES[suffix], slots
    return None


def _get_expert_stack_info(key: str) -> tuple[str, int] | None:
    """Map an expert-scoped checkpoint key to the packed runtime key."""
    match = _EXPERT_PATTERN.match(key)
    if match is None:
        return None

    packed_name = match.group("name")
    if packed_name.endswith(".weight"):
        packed_name = packed_name.removesuffix(".weight")
    return f"{match.group('prefix')}.{packed_name}", int(match.group("idx"))


def load_weight(
    model_path: str, device: torch.device, *, dtype: torch.dtype | None = None
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming weight loader. Yields (name, tensor) pairs already sharded, merged,
    and on device. Peak CPU memory: one full tensor + a small merge buffer.
    The execution dtype selects Naive's BF16 output-row sharding; it does not cast weights.
    """
    from .config import ModelConfig

    model_folder = download_hf_weight(model_path)
    config = ModelConfig.from_hf(cached_load_hf_config(model_path))
    files = glob.glob(f"{model_folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    tp_info = get_tp_info()

    if config.is_naive:
        yield from _load_naive_weight(files, config, device, dtype=dtype)
        return

    # Buffer for merge groups: merged_key -> {slot: tensor}
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    expert_buf: Dict[str, Dict[int, torch.Tensor]] = {}
    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for name in f.keys():
                # Strip multimodal wrapper prefix, skip vision/projector weights
                if name.startswith(("vision_tower.", "multi_modal_projector.")):
                    continue
                raw = f.get_tensor(name)
                name = name.removeprefix("language_model.")
                tensor = _shard_tensor(name, raw, tp_info.rank, tp_info.size, config.num_kv_heads)
                del raw

                if (info := _get_merge_info(name)) is None:
                    out = (name, tensor)
                else:
                    merged_key, slot, all_slots = info
                    merge_buf.setdefault(merged_key, {})[slot] = tensor
                    if not all(s in merge_buf[merged_key] for s in all_slots):
                        continue
                    parts = [merge_buf[merged_key][s] for s in all_slots]
                    del merge_buf[merged_key]
                    out = (merged_key, torch.cat(parts, dim=0))

                if config.is_moe and (expert_info := _get_expert_stack_info(out[0])) is not None:
                    packed_key, expert_idx = expert_info
                    slots = expert_buf.setdefault(packed_key, {})
                    slots[expert_idx] = out[1]
                    if len(slots) != config.num_experts:
                        continue
                    experts = [slots[idx] for idx in range(config.num_experts)]
                    del expert_buf[packed_key]
                    yield packed_key, torch.stack(experts, dim=0)
                else:  # Normal dense model
                    yield out[0], out[1]

    assert not merge_buf, f"Incomplete merge groups in checkpoint: {list(merge_buf.keys())}"
    assert not expert_buf, f"Incomplete expert tensors in checkpoint: {list(expert_buf.keys())}"


def _load_naive_weight(files, config, device, *, dtype: torch.dtype | None = None):
    """Naive keeps split attention/dense projections and packs only expert matrices."""
    c = config.naive_config
    assert c is not None
    from .autoround import validate_quantization

    if validate_quantization(c):
        if dtype != torch.bfloat16:
            raise ValueError("AutoRound Naive inference requires bfloat16")
        yield from _load_autoround_weight(files, config, device)
        return
    rank, size = get_tp_info().rank, get_tp_info().size
    experts = {}
    packed_parts = {}
    seen = set()
    emitted = set()

    def shard(name, tensor):
        if name.endswith(".experts.gate_up_proj"):
            gate, up = tensor.chunk(2, dim=1)
            return torch.cat((gate.chunk(size, dim=1)[rank], up.chunk(size, dim=1)[rank]), dim=1)
        if name.endswith(".experts.down_proj"):
            return tensor.chunk(size, dim=1 if dtype == torch.bfloat16 else 2)[rank]
        if dtype == torch.bfloat16 and name.endswith(
            (".o_proj.weight", ".o_proj.bias", ".down_proj.weight", ".down_proj.bias")
        ):
            return tensor.chunk(size, dim=0)[rank]
        if ".indexer." in name or ".mlp.gate." in name:
            return tensor
        if name.endswith(".attention_sink_bias"):
            return tensor.chunk(size)[rank]
        if ".self_attn." in name:
            layer = int(name.split(".layers.")[1].split(".")[0])
            prefix = "swa_" if c.hybrid_layer_pattern[layer] else ""
            heads = getattr(c, prefix + "num_key_value_heads")
            return _shard_tensor(name, tensor, rank, size, heads)
        return _shard_tensor(name, tensor, rank, size, c.num_key_value_heads)

    def output(name, tensor):
        if name in emitted:
            raise ValueError(f"Duplicate Naive logical tensor: {name}")
        emitted.add(name)
        dtype = (
            torch.float32
            if name.endswith((".mlp.gate.weight", ".e_score_correction_bias"))
            else tensor.dtype
        )
        return name, shard(name, tensor).contiguous().to(device=device, dtype=dtype)

    def pack(name, tensor):
        # Transformers 5 checkpoints can store already-packed experts without .weight.
        if ".experts." in name:
            name = name.removesuffix(".weight")
            projection = name.rsplit(".", 1)[-1]
            expected = {
                "gate_up_proj": (c.n_routed_experts, 2 * c.moe_intermediate_size, c.hidden_size),
                "gate_proj": (c.n_routed_experts, c.moe_intermediate_size, c.hidden_size),
                "up_proj": (c.n_routed_experts, c.moe_intermediate_size, c.hidden_size),
                "down_proj": (c.n_routed_experts, c.hidden_size, c.moe_intermediate_size),
            }.get(projection)
            if expected is None or tuple(tensor.shape) != expected:
                raise ValueError(f"Malformed Naive packed expert tensor: {name} {tensor.shape}")
            if name.endswith((".gate_proj", ".up_proj")):
                prefix, projection = name.rsplit(".", 1)
                parts = packed_parts.setdefault(prefix, {})
                if projection in parts:
                    raise ValueError(f"Duplicate Naive packed expert entry: {name}")
                parts[projection] = tensor
                if len(parts) == 2:
                    del packed_parts[prefix]
                    return output(
                        prefix + ".gate_up_proj",
                        torch.cat((parts["gate_proj"], parts["up_proj"]), 1),
                    )
                return None
        return output(name, tensor)

    for file in sorted(files):
        with safetensors.safe_open(file, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if name in seen:
                    raise ValueError(f"Duplicate Naive checkpoint tensor: {name}")
                seen.add(name)
                tensor = reader.get_tensor(name)
                if (
                    tensor.dtype
                    not in (torch.float32, torch.bfloat16, torch.float16, torch.float64)
                    or "scale_inv" in name
                ):
                    raise ValueError("Quantized Naive weights require a quantized loader")
                match = _EXPERT_PATTERN.match(name)
                if match:
                    prefix = match.group("prefix")
                    projection = match.group("name").removesuffix(".weight")
                    key = prefix + "." + projection
                    parts = experts.setdefault(key, {})
                    idx = int(match.group("idx"))
                    if idx >= c.n_routed_experts:
                        raise ValueError(f"Invalid Naive expert index: {idx}")
                    expected = (
                        (c.hidden_size, c.moe_intermediate_size)
                        if projection == "down_proj"
                        else (c.moe_intermediate_size, c.hidden_size)
                    )
                    if (
                        projection not in ("gate_proj", "up_proj", "down_proj")
                        or tuple(tensor.shape) != expected
                    ):
                        raise ValueError(f"Malformed Naive expert tensor: {name} {tensor.shape}")
                    if idx in parts:
                        raise ValueError(f"Duplicate Naive expert entry: {name}")
                    parts[idx] = tensor
                    if len(parts) != c.n_routed_experts:
                        continue
                    tensor = torch.stack([parts[i] for i in range(c.n_routed_experts)])
                    del experts[key]
                    name = key
                result = pack(name, tensor)
                if result is not None:
                    yield result
    if experts or packed_parts:
        raise ValueError("Incomplete Naive expert tensors in checkpoint")


def _load_autoround_weight(files, config, device):
    """Fill final expert arrays from rank-local CPU slices, without stacking copies."""
    from pathlib import Path

    from .autoround import _DTYPES, EXPERT, inspect_checkpoint

    c = config.naive_config
    rank, size = get_tp_info().rank, get_tp_info().size
    if not files:
        raise ValueError("No AutoRound checkpoint shards found")
    report = inspect_checkpoint(Path(files[0]).parent, c, size)
    expected_files = {str(Path(e["file"]).resolve()) for e in report["entries"].values()}
    if len(set(files)) != len(files) or {str(Path(f).resolve()) for f in files} != expected_files:
        raise ValueError("Duplicate or incomplete AutoRound checkpoint shard list")
    buffers = {}
    remaining = {}
    # Preallocate each final array once. Components can reside in different files.
    for name, spec in report["entries"].items():
        match = EXPERT.match(name)
        if not match:
            continue
        layer, expert, projection, component = match.groups()
        key = f"model.layers.{layer}.mlp.experts.{projection}.{component}"
        if key not in buffers:
            shape = (c.n_routed_experts, spec["shape"][0], spec["shape"][1] // size)
            buffers[key] = torch.empty(shape, dtype=_DTYPES[spec["dtype"]][0], device=device)
            remaining[key] = c.n_routed_experts
    for file in sorted(files):
        with safetensors.safe_open(file, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                match = EXPERT.match(name)
                if match:
                    layer, expert, projection, component = match.groups()
                    key = f"model.layers.{layer}.mlp.experts.{projection}.{component}"
                    target = buffers[key][int(expert)]
                    width = target.shape[1]
                    sliced = reader.get_slice(name)[:, rank * width : (rank + 1) * width]
                    target.copy_(sliced)
                    del sliced, target
                    remaining[key] -= 1
                    if not remaining[key]:
                        yield key, buffers.pop(key)
                else:
                    # Reuse established BF16 sharding for all non-expert tensors.
                    tensor = reader.get_tensor(name)
                    if name.endswith((".o_proj.weight", ".down_proj.weight")):
                        tensor = tensor.chunk(size, dim=0)[rank]
                    elif ".indexer." in name or ".mlp.gate." in name:
                        pass
                    elif name.endswith(".attention_sink_bias"):
                        tensor = tensor.chunk(size)[rank]
                    else:
                        heads = c.num_key_value_heads
                        if ".self_attn." in name:
                            layer = int(name.split(".layers.")[1].split(".")[0])
                            if c.hybrid_layer_pattern[layer]:
                                heads = c.swa_num_key_value_heads
                        tensor = _shard_tensor(name, tensor, rank, size, heads)
                    yield name, tensor.contiguous().to(device=device)
