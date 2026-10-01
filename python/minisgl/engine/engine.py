from __future__ import annotations

from datetime import timedelta
from time import monotonic
from typing import Any, Dict, NamedTuple, Tuple

import torch
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, set_global_ctx
from minisgl.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info
from minisgl.kvcache import create_kvcache_pool
from minisgl.layers import set_rope_device
from minisgl.models import create_model, load_weight
from minisgl.moe import create_moe_backend
from minisgl.utils import init_logger, is_sm90_supported, is_sm100_supported, torch_dtype

from ..utils.arch import is_sm121
from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)


class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event


class Engine:
    def __init__(self, config: EngineConfig):
        assert not torch.cuda.is_initialized()
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        _adjust_config(config)
        self.model_config = config.model_config

        self.device = torch.device(f"cuda:{config.gpu_index}")
        self.tp_info = config.tp_info
        self.multi_node = config.nnodes > 1
        torch.cuda.set_device(self.device)
        torch.manual_seed(42)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)

        self.tp_cpu_group = self._init_communication(config)
        if self.multi_node:
            from minisgl.distributed.check import check_matching_settings

            settings = [{} for _ in range(config.tp_info.size)]
            torch.distributed.all_gather_object(
                settings, config.shared_inference_settings(), group=self.tp_cpu_group
            )
            check_matching_settings(settings)
        # Every worker must select the same graph sizes and a cache that fits all nodes.
        initial_memory_range = self._sync_get_memory()
        self.initial_local_free_memory = self.local_free_memory
        init_free_memory = initial_memory_range[0 if self.multi_node else 1]
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # ======================= Model initialization ========================
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_config)
        load_started = monotonic()
        self.model.load_state_dict(self._load_weight_state_dict(config))
        torch.cuda.synchronize(self.device)
        self.weight_loading_seconds = monotonic() - load_started
        self.weight_resident_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in self.model.state_dict().values()
        )
        self.weight_loading_peak_bytes = torch.cuda.max_memory_allocated(self.device)
        logger.info(
            f"Weights loaded in {self.weight_loading_seconds:.2f}s; "
            f"resident tensors {mem_GB(self.weight_resident_bytes)}, "
            f"loading peak PyTorch allocation {mem_GB(self.weight_loading_peak_bytes)}"
        )

        # ======================= KV cache initialization ========================
        self.num_pages = self._determine_num_pages(init_free_memory, config)
        num_tokens = self.num_pages * config.page_size
        self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            page_size=config.page_size,
            device=self.device,
            dtype=self.dtype,
        )

        # ======================= Page table initialization ========================
        # NOTE: 1. aligned to 128 bytes; 2. store raw locations instead of pages
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = _align_up_32(self.max_seq_len)
        self.ctx.page_table = self.page_table = torch.zeros(  # + 1 for dummy request
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )

        # ======================= Attention & MoE backend initialization ========================
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(
            config.attention_backend, config.model_config
        )
        if config.model_config.is_moe and not config.model_config.is_naive:
            self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)

        # ======================= Sampler initialization ========================
        self.sampler = Sampler(self.device, config.model_config.vocab_size)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # ======================= Graph capture initialization ========================
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)  # point to dummy page
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
        )
        if self.multi_node:
            # Weight loading and JIT compilation use the startup deadline. Once ready,
            # request coordination gets its shorter runtime failure-detection timeout.
            self.tp_cpu_group = torch.distributed.new_group(
                backend="gloo", timeout=timedelta(seconds=config.distributed_timeout)
            )

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        if config.tp_info.size == 1 or (config.use_pynccl and not self.multi_node):
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            max_bytes = (
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            )
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.new_group(
                backend="gloo",
                timeout=timedelta(
                    seconds=(
                        config.startup_timeout if self.multi_node else config.distributed_timeout
                    )
                ),
            )
            assert tp_cpu_group is not None
        return tp_cpu_group

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        if config.use_dummy_weight:
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            return {
                k: v.to(
                    torch.float32
                    if config.model_config.is_naive
                    and (k.endswith("mlp.gate.weight") or k.endswith("e_score_correction_bias"))
                    else (
                        v.dtype
                        if config.model_config.is_naive
                        and k.endswith((".qweight", ".qzeros", ".scales"))
                        else self.dtype
                    )
                )
                for k, v in load_weight(config.model_path, self.device, dtype=self.dtype)
            }

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        memory_range = self._sync_get_memory()
        new_free_memory = memory_range[0 if self.multi_node else 1]
        cache_per_page = (
            config.model_config.kv_bytes_per_token(config.tp_info.size, self.dtype.itemsize)
            * config.page_size
        )
        from minisgl.models.autoround import execution_workspace

        workspace = (
            execution_workspace(config.model_config.naive_config, config.tp_info.size)
            if config.model_config.is_naive
            else 0
        )
        if workspace:
            logger.info_rank0(f"Packed expert execution reserve: {mem_GB(workspace)}")
        num_pages = config.num_page_override
        if num_pages is None:
            if self.multi_node:
                # Compute each host's budget locally, then choose the smallest page count.
                model_memory = self.initial_local_free_memory - self.local_free_memory
                available_memory = (
                    int(config.memory_ratio * self.initial_local_free_memory) - model_memory
                )
            else:
                model_memory = old_free_memory - new_free_memory
                available_memory = int(config.memory_ratio * old_free_memory) - model_memory
            num_pages = (available_memory - workspace) // cache_per_page

        if workspace and num_pages * cache_per_page + workspace > self.local_free_memory:
            raise ValueError("KV cache override leaves insufficient packed-expert workspace")
        if self.multi_node:
            pages = torch.tensor(num_pages, dtype=torch.int64)
            torch.distributed.all_reduce(
                pages, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
            )
            num_pages = int(pages.item())

        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
        num_tokens = num_pages * config.page_size
        real_kv_size = num_pages * cache_per_page
        logger.info(f"Allocating {num_tokens} tokens for KV cache, K + V = {mem_GB(real_kv_size)}")
        return num_pages

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        self.local_free_memory = free_memory
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            log = logger.warning if self.multi_node else logger.error
            log(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            if not self.multi_node:
                raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream
        with self.ctx.forward_batch(batch):
            if self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
            else:
                logits = self.model.forward()

        for req in batch.reqs:
            req.complete_one()

        if self.multi_node:
            if self.tp_info.is_primary():
                next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
            else:
                next_tokens_gpu = torch.empty(batch.size, dtype=torch.int32, device=self.device)
            # One sampling decision keeps request completion and future batches identical.
            torch.distributed.broadcast(next_tokens_gpu, src=0)
        else:
            next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def shutdown(self) -> None:
        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()


def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32


def _adjust_config(config: EngineConfig):
    def override(attr: str, value: Any):  # this is dangerous, use with caution
        object.__setattr__(config, attr, value)

    if config.model_config.is_naive:
        if config.attention_backend not in ("auto", "naive"):
            raise ValueError("Naive requires the eager 'naive' attention backend")
        if config.moe_backend != "auto":
            raise ValueError("Naive uses its own eager sigmoid MoE implementation")
        if config.use_dummy_weight:
            raise ValueError("Use tools/make_naive_fixture.py instead of --dummy-weight for Naive")
        from minisgl.models.autoround import validate_quantization

        if validate_quantization(config.hf_config) and config.dtype != torch.bfloat16:
            raise ValueError("AutoRound Naive inference requires bfloat16")
        if config.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("Naive eager execution supports BF16 and FP32")
        override("attention_backend", "naive")
        override("cuda_graph_bs", [])
        override("cuda_graph_max_bs", 0)
        logger.info_rank0("Naive eager execution: CUDA graphs and overlap scheduling disabled")
        return

    if config.attention_backend == "auto":
        if is_sm121():
            backend = "fi"
        elif is_sm100_supported():
            backend = "trtllm"
        elif is_sm90_supported():
            backend = "fa,fi"
        else:
            backend = "fi"
        override("attention_backend", backend)
        logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")

    if "trtllm" in config.attention_backend and config.page_size not in [16, 32, 64]:
        override("page_size", 64)
        logger.warning_rank0("Page size is overridden to 64 for TRTLLM backend")

    if config.model_config.is_moe and config.moe_backend == "auto":
        override("moe_backend", "fused")
        logger.info_rank0(f"Auto-selected MoE backend: {config.moe_backend}")
