"""Load the revision-pinned upstream reference shipped with a synthetic checkpoint."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

from safetensors.torch import load_file

UPSTREAM_MODEL = "NaiveAI/Naive-N0.5-Flash"
UPSTREAM_REVISION = "0235b3b5ff27422b1f57cdc2acddfaf643e08356"


def reference_module(folder: Path):
    package = "_naive_fixture_reference"
    module = types.ModuleType(package)
    module.__path__ = [str(folder.resolve())]
    sys.modules[package] = module
    return importlib.import_module(package + ".modeling_naive_n05_flash")


def load_reference(folder: Path, dtype, device):
    module = reference_module(folder)
    config = module.NaiveN05FlashConfig.from_pretrained(folder)
    model = module.NaiveN05FlashForCausalLM(config).to(device=device, dtype=dtype)
    model.set_experts_implementation("eager")
    # HF checkpoint loading retains FP32 rotary buffers. A blanket model.to(dtype)
    # would round the inverse frequencies and change sparse token selection.
    for layer in model.model.layers:
        rope = layer.self_attn.rotary_emb
        layer.self_attn.rotary_emb = type(rope)(rope.config).to(device=device)
    # The checkpoint is unquantized; retain upstream's strict FP32 router parameters.
    for layer in model.model.layers:
        if hasattr(layer.mlp, "gate"):
            layer.mlp.gate.float()
    tensors = load_file(str(folder / "model.safetensors"), device=str(device))
    if getattr(config, "quantization_config", None):
        from autoround_reference import reconstruct_experts

        tensors = reconstruct_experts(tensors)
    model.load_state_dict(tensors)
    return model.eval()
