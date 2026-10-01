"""Generate one small checkpoint shared by the server, TP ranks, and upstream reference."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from naive_reference import UPSTREAM_MODEL, UPSTREAM_REVISION, reference_module
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--int4", action="store_true", help="Pack routed experts as AutoRound GPTQ")
    args = parser.parse_args()
    folder = args.output
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("Fixture output must be empty; use a new directory to regenerate")
    source = Path(
        snapshot_download(
            UPSTREAM_MODEL,
            revision=UPSTREAM_REVISION,
            allow_patterns=[
                "*.py",
                "config.json",
                "tokenizer*",
                "vocab*",
                "merges.txt",
                "special_tokens_map.json",
                "chat_template.jinja",
            ],
        )
    )
    folder.mkdir(parents=True, exist_ok=True)
    for file in source.iterdir():
        if file.is_file():
            shutil.copyfile(file, folder / file.name)
    config_data = json.loads((folder / "config.json").read_text())
    config_data.update(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=3,
        num_attention_heads=8,
        num_key_value_heads=4,
        swa_num_attention_heads=8,
        swa_num_key_value_heads=8,
        n_routed_experts=16,
        moe_intermediate_size=256,
        num_experts_per_tok=8,
        hybrid_layer_pattern=[0, 1, 0],
        moe_layer_freq=[0, 1, 1],
        index_top_k=16,
        max_position_embeddings=4096,
        attention_chunk_size=128,
    )
    module = reference_module(folder)
    config = module.NaiveN05FlashConfig(**config_data)
    config.save_pretrained(folder)
    torch.manual_seed(args.seed)
    model = module.NaiveN05FlashForCausalLM(config)
    # Initialize explicitly: custom upstream expert modules need not follow HF initialization.
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            if name.endswith("e_score_correction_bias"):
                tensor.copy_(torch.linspace(-0.015, 0.015, tensor.numel()))
            elif name.endswith("attention_sink_bias"):
                tensor.copy_(torch.linspace(-0.2, 0.2, tensor.numel()))
            elif name.endswith(".bias"):
                tensor.zero_()
            elif "layernorm.weight" in name or name.endswith(("norm.weight", "k_norm.weight")):
                tensor.fill_(1.0)
            else:
                tensor.normal_(mean=0, std=0.02)
    tensors = {
        name: tensor.detach().contiguous().cpu() for name, tensor in model.state_dict().items()
    }
    for name in tensors:
        if not name.endswith(("mlp.gate.weight", "e_score_correction_bias")):
            tensors[name] = tensors[name].to(torch.bfloat16)
    if args.int4:
        from autoround_reference import QUANTIZATION, quantize_experts

        tensors = quantize_experts(tensors)
        config_data = json.loads((folder / "config.json").read_text())
        config_data["quantization_config"] = QUANTIZATION
        (folder / "config.json").write_text(json.dumps(config_data, indent=2) + "\n")
    save_file(tensors, str(folder / "model.safetensors"), metadata={"format": "pt"})
    (folder / "fixture_manifest.json").write_text(
        json.dumps(
            {
                "upstream_model": UPSTREAM_MODEL,
                "upstream_revision": UPSTREAM_REVISION,
                "seed": args.seed,
                "synthetic": True,
                "quantized": args.int4,
                "torch": torch.__version__,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Synthetic Naive checkpoint: {folder} ({sum(t.numel() * t.element_size() for t in tensors.values()) / 2**20:.1f} MiB)"
    )


if __name__ == "__main__":
    main()
