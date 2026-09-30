from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from minisgl.distributed import DistributedInfo
from minisgl.utils import cached_load_hf_config

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 256
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    distributed_timeout: float = 60.0
    startup_timeout: float = 600.0
    # TP rank selects a model shard; local GPU index selects a device on this host.
    local_gpu_index: int | None = None
    nnodes: int = 1
    dist_init_addr: str | None = None
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        from minisgl.models import ModelConfig

        return ModelConfig.from_hf(self.hf_config)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        if self.dist_init_addr is not None:
            return f"tcp://{self.dist_init_addr}"
        return "tcp://127.0.0.1:2333"

    @property
    def gpu_index(self) -> int:
        return self.tp_info.rank if self.local_gpu_index is None else self.local_gpu_index

    def shared_inference_settings(self) -> dict[str, object]:
        """Settings that must agree so all TP workers execute the same operations."""
        names = (
            "model_path",
            "dtype",
            "max_running_req",
            "attention_backend",
            "moe_backend",
            "cuda_graph_bs",
            "cuda_graph_max_bs",
            "page_size",
            "memory_ratio",
            "use_dummy_weight",
            "use_pynccl",
            "max_seq_len_override",
            "num_page_override",
            "distributed_timeout",
            "startup_timeout",
        )
        settings = {name: getattr(self, name) for name in names}
        settings["dtype"] = str(self.dtype)
        settings["model_config"] = self.hf_config.to_dict()
        settings["model_revision"] = getattr(self.hf_config, "_commit_hash", None)
        from minisgl.env import ENV

        settings["disable_overlap_scheduling"] = ENV.DISABLE_OVERLAP_SCHEDULING.value
        settings["flashinfer_use_tensor_cores"] = ENV.FLASHINFER_USE_TENSOR_CORES.value
        return settings
