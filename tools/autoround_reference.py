"""Independent fixture encoder/reference decoder. Never imports the runtime decoder."""

from __future__ import annotations

import re

import torch

QUANTIZATION = {
    "quant_method": "auto-round",
    "packing_format": "auto_round:auto_gptq",
    "bits": 4,
    "group_size": 128,
    "sym": True,
    "desc_act": False,
}


def pack_projection(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    outputs, inputs = weight.shape
    grouped = weight.float().reshape(outputs, inputs // 128, 128)
    scales = (grouped.abs().amax(-1).clamp_min(1e-6) / 7).half()
    values = (grouped / scales.float()[:, :, None]).round().add(8).clamp(0, 15)
    values = values.reshape(outputs, inputs).to(torch.int64)
    packed = torch.zeros(inputs // 8, outputs, dtype=torch.int64)
    for nibble in range(8):
        packed |= values[:, nibble::8].T << (4 * nibble)
    return {
        "qweight": packed.int().contiguous(),
        "qzeros": torch.full((inputs // 128, outputs // 8), 0x77777777, dtype=torch.int32),
        "scales": scales.T.contiguous(),
    }


def reference_projection(parts: dict[str, torch.Tensor]) -> torch.Tensor:
    """Unpack independently using unsigned int64 words and strided assignment."""
    packed = parts["qweight"].long() & 0xFFFFFFFF
    packed_zeros = parts["qzeros"].long() & 0xFFFFFFFF
    inputs, outputs = packed.shape[0] * 8, packed.shape[1]
    values = torch.empty(inputs, outputs, device=packed.device, dtype=torch.int16)
    zeros = torch.empty(parts["scales"].shape, device=packed.device, dtype=torch.int16)
    for nibble in range(8):
        values[nibble::8] = (packed // (16**nibble)) % 16
        zeros[:, nibble::8] = (packed_zeros // (16**nibble)) % 16 + 1
    residual = values - zeros.repeat_interleave(128, dim=0)
    return (residual * parts["scales"].repeat_interleave(128, dim=0)).T.bfloat16().contiguous()


def quantize_experts(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = {}
    for name, tensor in tensors.items():
        if ".experts." not in name:
            result[name] = tensor
            continue
        root, projection = name.removesuffix(".weight").rsplit(".", 1)
        projections = (
            zip(("gate_proj", "up_proj"), tensor.chunk(2, dim=1))
            if projection == "gate_up_proj"
            else [(projection, tensor)]
        )
        for projection, bank in projections:
            for expert, weight in enumerate(bank):
                for component, packed in pack_projection(weight).items():
                    result[f"{root}.{expert}.{projection}.{component}"] = packed
    return result


def reconstruct_experts(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result, groups, banks = {}, {}, {}
    pattern = re.compile(r"^(.+\.experts)\.(\d+)\.(\w+_proj)\.(qweight|qzeros|scales)$")
    for name, tensor in tensors.items():
        match = pattern.match(name)
        if match:
            root, expert, projection, component = match.groups()
            groups.setdefault((root, int(expert), projection), {})[component] = tensor
        else:
            result[name] = tensor
    for (root, expert, projection), parts in groups.items():
        banks.setdefault(root, {}).setdefault(projection, {})[expert] = reference_projection(parts)
    for root, projections in banks.items():
        stacked = {
            p: torch.stack([experts[i] for i in range(len(experts))])
            for p, experts in projections.items()
        }
        result[root + ".gate_up_proj"] = torch.cat((stacked["gate_proj"], stacked["up_proj"]), 1)
        result[root + ".down_proj"] = stacked["down_proj"]
    return result
