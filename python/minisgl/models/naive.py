from __future__ import annotations

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.layers import (
    BaseOP,
    LinearOProj,
    LinearReplicated,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from minisgl.utils import div_even

from .base import BaseLLMModel
from .config import ModelConfig
from .naive_config import NaiveN05FlashConfig


def round_indexer_fp8(states: torch.Tensor) -> torch.Tensor:
    states = states.float()
    scale = states.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4) / 448.0
    return (states / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale


def rotary(states: torch.Tensor, positions: torch.Tensor, dim: int, base: float) -> torch.Tensor:
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=states.device, dtype=torch.float32) / dim))
    freqs = positions.float()[:, None] * inv[None, :]
    angles = torch.cat((freqs, freqs), dim=-1)
    cos, sin = angles.cos().to(states.dtype)[:, None], angles.sin().to(states.dtype)[:, None]
    rotated = states[..., :dim]
    first, second = rotated.chunk(2, dim=-1)
    rotated = rotated * cos + torch.cat((-second, first), dim=-1) * sin
    return torch.cat((rotated, states[..., dim:]), dim=-1)


class NaiveRMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.weight = torch.empty(size)
        self._eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states = x.float()
        states = states * torch.rsqrt(states.pow(2).mean(-1, keepdim=True) + self._eps)
        return self.weight * states.to(x.dtype)


class NaiveLayerNorm(BaseOP):
    def __init__(self, size: int) -> None:
        self.weight, self.bias = torch.empty(size), torch.empty(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, 1e-5)


class NaiveRowLinear(LinearOProj):
    """Round once after combining TP dot products, rather than once per shard."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._tp_size == 1:
            return F.linear(x, self.weight, self.bias)
        partial = F.linear(x.float(), self.weight.float())
        return self._comm.all_reduce(partial).to(x.dtype)


class NaiveColumnLinear(LinearReplicated):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if get_tp_info().size == 1:
            return F.linear(x, self.weight, self.bias)
        return F.linear(
            x.float(), self.weight.float(), None if self.bias is None else self.bias.float()
        ).to(x.dtype)


class NaiveIndexer(BaseOP):
    def __init__(self, c: NaiveN05FlashConfig) -> None:
        self.wq = LinearReplicated(c.hidden_size, c.index_n_heads * c.index_head_dim, False)
        self.wk = LinearReplicated(c.hidden_size, c.index_head_dim, False)
        self.k_norm = NaiveLayerNorm(c.index_head_dim)
        self.weights_proj = LinearReplicated(c.hidden_size, c.index_n_heads, False)
        self._config = c

    def forward(
        self, states: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self._config
        dim = int(c.head_dim * c.partial_rotary_factor)
        q = self.wq.forward(states).view(-1, c.index_n_heads, c.index_head_dim)
        k = self.k_norm.forward(self.wk.forward(states)).unsqueeze(1)
        q, k = rotary(q, positions, dim, c.rope_theta), rotary(k, positions, dim, c.rope_theta)
        if c.indexer_activation_dtype == "fp8_e4m3":
            q, k = round_indexer_fp8(q), round_indexer_fp8(k)
        weights = self.weights_proj.forward(states) * c.index_n_heads**-0.5
        return q, k.squeeze(1), weights


class NaiveAttention(BaseOP):
    def __init__(self, c: NaiveN05FlashConfig, layer_id: int) -> None:
        prefix = "swa_" if c.hybrid_layer_pattern[layer_id] else ""
        heads = getattr(c, prefix + "num_attention_heads")
        kv_heads = getattr(c, prefix + "num_key_value_heads")
        self._heads = div_even(heads, get_tp_info().size)
        self._kv_heads = div_even(kv_heads, get_tp_info().size, allow_replicate=True)
        self._dim, self._v_dim = getattr(c, prefix + "head_dim"), getattr(c, prefix + "v_head_dim")
        self._base = getattr(c, prefix + "rope_theta")
        self._rotary_dim = int(self._dim * c.partial_rotary_factor)
        self._layer_id = layer_id
        self.q_proj = NaiveColumnLinear(c.hidden_size, self._heads * self._dim, c.attention_bias)
        self.k_proj = NaiveColumnLinear(c.hidden_size, self._kv_heads * self._dim, c.attention_bias)
        self.v_proj = NaiveColumnLinear(
            c.hidden_size, self._kv_heads * self._v_dim, c.attention_bias
        )
        self.o_proj = NaiveRowLinear(heads * self._v_dim, c.hidden_size, False)
        self.indexer = None if prefix else NaiveIndexer(c)
        sink = c.add_swa_attention_sink_bias if prefix else c.add_full_attention_sink_bias
        self.attention_sink_bias = torch.empty(self._heads) if sink else None

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        pos = ctx.batch.positions
        q = self.q_proj.forward(states).view(-1, self._heads, self._dim)
        k = self.k_proj.forward(states).view(-1, self._kv_heads, self._dim)
        v = self.v_proj.forward(states).view(-1, self._kv_heads, self._v_dim)
        q, k = rotary(q, pos, self._rotary_dim, self._base), rotary(
            k, pos, self._rotary_dim, self._base
        )
        kwargs = {}
        if self.indexer is not None:
            iq, ik, iw = self.indexer.forward(states, pos)
            kwargs = {"index_query": iq, "index_key": ik, "index_weights": iw}
        out = ctx.attn_backend.forward(
            q, k, v, self._layer_id, ctx.batch, sink=self.attention_sink_bias, **kwargs
        )
        return self.o_proj.forward(out.reshape(-1, self._heads * self._v_dim))


class NaiveMLP(BaseOP):
    def __init__(self, c: NaiveN05FlashConfig) -> None:
        width = div_even(c.intermediate_size, get_tp_info().size)
        self.gate_proj = NaiveColumnLinear(c.hidden_size, width, False)
        self.up_proj = NaiveColumnLinear(c.hidden_size, width, False)
        self.down_proj = NaiveRowLinear(c.intermediate_size, c.hidden_size, False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(F.silu(self.gate_proj.forward(x)) * self.up_proj.forward(x))


class NaiveRouter(BaseOP):
    def __init__(self, c: NaiveN05FlashConfig) -> None:
        self.weight = torch.empty(c.n_routed_experts, c.hidden_size, dtype=torch.float32)
        self.e_score_correction_bias = torch.empty(c.n_routed_experts, dtype=torch.float32)
        self._config = c

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = self._config
        scores = F.linear(states.float(), self.weight).sigmoid()
        selected = (
            (scores + self.e_score_correction_bias)
            .topk(c.num_experts_per_tok, sorted=False)
            .indices
        )
        weights = scores.gather(-1, selected)
        if c.norm_topk_prob:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return selected, weights * c.routed_scaling_factor


class NaiveExperts(BaseOP):
    def __init__(self, c: NaiveN05FlashConfig) -> None:
        width = div_even(c.moe_intermediate_size, get_tp_info().size)
        self.gate_up_proj = torch.empty(c.n_routed_experts, 2 * width, c.hidden_size)
        self.down_proj = torch.empty(c.n_routed_experts, c.hidden_size, width)
        self._comm = DistributedCommunicator()

    def forward(
        self, states: torch.Tensor, selected: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        result = torch.zeros_like(states)
        for expert in range(self.gate_up_proj.shape[0]):
            # Match the reference's slot-major token ordering for BF16 GEMMs.
            slot, token = torch.where(selected.T == expert)
            if token.numel() == 0:
                continue
            if get_tp_info().size > 1:
                gate_up = F.linear(states[token].float(), self.gate_up_proj[expert].float()).to(
                    states.dtype
                )
            else:
                gate_up = F.linear(states[token], self.gate_up_proj[expert])
            gate, up = gate_up.chunk(2, dim=-1)
            activated = F.silu(gate) * up
            if get_tp_info().size > 1:
                partial = F.linear(activated.float(), self.down_proj[expert].float())
                out = self._comm.all_reduce(partial).to(states.dtype)
            else:
                out = F.linear(activated, self.down_proj[expert])
            out = (out * weights[token, slot, None]).to(states.dtype)
            result.index_add_(0, token, out)
        return result


class NaiveMoE(BaseOP):
    def __init__(self, c: NaiveN05FlashConfig) -> None:
        self.gate, self.experts = NaiveRouter(c), NaiveExperts(c)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        selected, weights = self.gate.forward(states)
        return self.experts.forward(states, selected, weights)


class NaiveDecoderLayer(BaseOP):
    def __init__(self, c: NaiveN05FlashConfig, layer_id: int) -> None:
        self.self_attn = NaiveAttention(c, layer_id)
        self.mlp = NaiveMoE(c) if c.moe_layer_freq[layer_id] else NaiveMLP(c)
        self.input_layernorm = NaiveRMSNorm(c.hidden_size, c.layernorm_epsilon)
        self.post_attention_layernorm = NaiveRMSNorm(c.hidden_size, c.layernorm_epsilon)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        states = states + self.self_attn.forward(self.input_layernorm.forward(states))
        return states + self.mlp.forward(self.post_attention_layernorm.forward(states))


class NaiveModel(BaseOP):
    def __init__(self, c: NaiveN05FlashConfig) -> None:
        self.embed_tokens = VocabParallelEmbedding(c.vocab_size, c.hidden_size)
        self.layers = OPList([NaiveDecoderLayer(c, i) for i in range(c.num_hidden_layers)])
        self.norm = NaiveRMSNorm(c.hidden_size, c.layernorm_epsilon)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        states = self.embed_tokens.forward(input_ids)
        for layer in self.layers.op_list:
            states = layer.forward(states)
        return self.norm.forward(states)


class NaiveN05FlashForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig) -> None:
        c = config.naive_config
        assert c is not None
        self.model = NaiveModel(c)
        self.lm_head = ParallelLMHead(
            c.vocab_size,
            c.hidden_size,
            tie_word_embeddings=c.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if c.tie_word_embeddings else None,
        )

    def forward(self) -> torch.Tensor:
        return self.lm_head.forward(self.model.forward(get_global_ctx().batch.input_ids))
