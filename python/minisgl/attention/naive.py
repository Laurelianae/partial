from __future__ import annotations

from dataclasses import dataclass

import torch
from minisgl.core import Batch, get_global_ctx
from minisgl.kvcache.naive_pool import NaiveKVCache
from minisgl.models.config import ModelConfig

from .base import BaseAttnBackend, BaseAttnMetadata


def sparse_mask(scores: torch.Tensor, allowed: torch.Tensor, top_k: int) -> torch.Tensor:
    scores = scores.masked_fill(~allowed, -torch.inf)
    selected = scores.argsort(dim=-1, descending=True, stable=True)[..., :top_k]
    return allowed & torch.zeros_like(allowed).scatter(-1, selected, True)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    allowed: torch.Tensor,
    value_scale: float | None,
    sink: torch.Tensor | None,
) -> torch.Tensor:
    """Heads-first eager attention with an optional zero-value softmax sink."""
    k = k.repeat_interleave(q.shape[0] // k.shape[0], dim=0)
    v = v.repeat_interleave(q.shape[0] // v.shape[0], dim=0)
    if value_scale is not None:
        v = v * value_scale
    logits = (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
    logits = logits.masked_fill(~allowed.unsqueeze(0), -torch.inf)
    if sink is not None:
        logits = torch.cat((logits, sink[:, None, None].expand(-1, q.shape[1], 1)), dim=-1)
    probabilities = logits.float().softmax(-1).nan_to_num(0.0)[..., : k.shape[1]].to(v.dtype)
    return probabilities @ v


@dataclass
class NaiveMetadata(BaseAttnMetadata):
    # Snapshot host lengths before the engine advances each request.
    requests: list[tuple[int, int, int]]  # table row, cached length, total length
    last_indices: torch.Tensor

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.last_indices[:bs]


class NaiveAttentionBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config.naive_config
        assert self.config is not None
        self.cache = get_global_ctx().kv_cache
        assert isinstance(self.cache, NaiveKVCache)

    def prepare_metadata(self, batch: Batch) -> None:
        requests = [(r.table_idx, r.cached_len, r.device_len) for r in batch.padded_reqs]
        lengths = torch.tensor(
            [total - cached for _, cached, total in requests],
            device=self.cache.device,
            dtype=torch.int64,
        )
        batch.attn_metadata = NaiveMetadata(requests, lengths.cumsum(0) - 1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        *,
        index_query: torch.Tensor | None = None,
        index_key: torch.Tensor | None = None,
        index_weights: torch.Tensor | None = None,
        sink: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config
        metadata = batch.attn_metadata
        assert isinstance(metadata, NaiveMetadata)
        self.cache.store_kv(k, v, batch.out_loc, layer_id)
        swa = bool(c.hybrid_layer_pattern[layer_id])
        if not swa:
            assert index_key is not None
            self.cache.store_index(index_key, batch.out_loc, layer_id)
        outputs = []
        offset = 0
        for row, cached, total in metadata.requests:
            length = total - cached
            loc = get_global_ctx().page_table[row, :total].long()
            keys = self.cache.k_cache(layer_id).flatten(0, 1)[loc].transpose(0, 1)
            values = self.cache.v_cache(layer_id).flatten(0, 1)[loc].transpose(0, 1)
            # Bound the eager score matrix for long prefill chunks.
            for start in range(0, length, c.attention_chunk_size):
                end = min(start + c.attention_chunk_size, length)
                sl = slice(offset + start, offset + end)
                positions = batch.positions[sl]
                distance = positions[:, None] - torch.arange(total, device=q.device)[None, :]
                allowed = distance >= 0
                if swa:
                    allowed &= distance < c.sliding_window
                else:
                    assert index_query is not None and index_weights is not None
                    ik = self.cache.index_cache(layer_id)[loc]
                    scores = (index_query[sl].transpose(0, 1).float() @ ik.float().T).relu()
                    scores = (scores * index_weights[sl].T.unsqueeze(-1).float()).sum(0)
                    allowed = sparse_mask(scores, allowed, c.index_top_k)
                out = attention(
                    q[sl].transpose(0, 1), keys, values, allowed, c.attention_value_scale, sink
                )
                outputs.append(out.transpose(0, 1))
            offset += length
        return torch.cat(outputs, dim=0)

    def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
        raise RuntimeError("Naive eager attention does not support CUDA graphs")

    def prepare_for_capture(self, batch: Batch) -> None:
        raise RuntimeError("Naive eager attention does not support CUDA graphs")

    def prepare_for_replay(self, batch: Batch) -> None:
        raise RuntimeError("Naive eager attention does not support CUDA graphs")
