from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from transformers import PretrainedConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    model_type: str
    architectures: list[str]
    naive_config: PretrainedConfig | None = None

    @property
    def is_naive(self) -> bool:
        return self.model_type == "naive_n05_flash"

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        if hasattr(config, "text_config") and config.text_config is not None:
            top = config
            config = config.text_config
            for attr in ("architectures", "rope_theta", "rope_scaling"):
                if not getattr(config, attr, None) and getattr(top, attr, None):
                    setattr(config, attr, getattr(top, attr))

        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = (
            getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        )
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        model_type = getattr(config, "model_type", "llama")
        num_experts = getattr(
            config,
            "num_local_experts",
            getattr(config, "n_routed_experts", getattr(config, "num_experts", 0)),
        )
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
        norm_topk_prob = getattr(config, "norm_topk_prob", False)
        architectures = getattr(config, "architectures", ["LlamaForCausalLM"])

        # Llama/Qwen: rope_theta is a direct attr; Mistral: it's inside rope_scaling dict
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_theta = getattr(config, "rope_theta", None) or rope_scaling["rope_theta"]

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=getattr(
                config, "rms_norm_eps", getattr(config, "layernorm_epsilon", 1e-6)
            ),
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=int(head_dim * getattr(config, "partial_rotary_factor", 1.0)),
                max_position=config.max_position_embeddings,
                base=rope_theta,
                scaling=rope_scaling,
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            model_type=model_type,
            architectures=architectures,
            naive_config=config if model_type == "naive_n05_flash" else None,
        )

    def kv_bytes_per_token(self, tp_size: int, itemsize: int) -> int:
        from minisgl.utils import div_even

        if self.naive_config is None:
            return (
                2
                * self.num_layers
                * div_even(self.num_kv_heads, tp_size, allow_replicate=True)
                * self.head_dim
                * itemsize
            )
        c = self.naive_config
        size = 0
        for swa in c.hybrid_layer_pattern:
            prefix = "swa_" if swa else ""
            heads = div_even(
                getattr(c, prefix + "num_key_value_heads"), tp_size, allow_replicate=True
            )
            size += (
                heads
                * (getattr(c, prefix + "head_dim") + getattr(c, prefix + "v_head_dim"))
                * itemsize
            )
            if not swa:
                size += c.index_head_dim * 4  # rounded indexer keys retained in FP32
        return size
