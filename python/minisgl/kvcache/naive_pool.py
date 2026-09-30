from __future__ import annotations

import torch
from minisgl.distributed import get_tp_info
from minisgl.models.config import ModelConfig
from minisgl.utils import div_even

from .base import BaseKVCachePool


class NaiveKVCache(BaseKVCachePool):
    """Full per-layer histories sharing scheduler locations, including DSA indexer keys."""

    def __init__(
        self,
        config: ModelConfig,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        c = config.naive_config
        assert c is not None
        self._device, self._dtype = device, dtype
        self._keys, self._values, self._index = [], [], []
        for swa in c.hybrid_layer_pattern:
            prefix = "swa_" if swa else ""
            heads = div_even(
                getattr(c, prefix + "num_key_value_heads"), get_tp_info().size, allow_replicate=True
            )
            shape = (num_pages, page_size, heads)
            self._keys.append(
                torch.empty(*shape, getattr(c, prefix + "head_dim"), dtype=dtype, device=device)
            )
            self._values.append(
                torch.empty(*shape, getattr(c, prefix + "v_head_dim"), dtype=dtype, device=device)
            )
            self._index.append(
                None
                if swa
                else torch.empty(
                    num_pages * page_size, c.index_head_dim, dtype=torch.float32, device=device
                )
            )

    def k_cache(self, index: int) -> torch.Tensor:
        return self._keys[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._values[index]

    def index_cache(self, index: int) -> torch.Tensor:
        result = self._index[index]
        assert result is not None
        return result

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        for cache, states in ((self._keys[layer_id], k), (self._values[layer_id], v)):
            flat = cache.flatten(0, 1)
            flat.index_copy_(0, out_loc.long(), states.reshape(-1, *flat.shape[1:]))

    def store_index(self, key: torch.Tensor, out_loc: torch.Tensor, layer_id: int) -> None:
        self.index_cache(layer_id).index_copy_(0, out_loc.long(), key.float())

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def num_layers(self) -> int:
        return len(self._keys)
