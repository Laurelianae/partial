from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from minisgl.attention.naive import attention, sparse_mask
from minisgl.models.config import ModelConfig
from minisgl.models.naive import (
    NaiveRouter,
    NaiveRowLinear,
    gather_last_dim,
    rotary,
    round_indexer_fp8,
)
from minisgl.models.naive_config import NaiveN05FlashConfig
from minisgl.models.weight import _load_naive_weight
from minisgl.utils import torch_dtype
from safetensors.torch import save_file


def config(**kwargs):
    return NaiveN05FlashConfig(
        num_hidden_layers=3, hybrid_layer_pattern=[0, 1, 0], moe_layer_freq=[0, 1, 1], **kwargs
    )


def test_config_and_cache_budget():
    c = ModelConfig.from_hf(config())
    assert c.is_naive and c.is_moe and c.num_experts == 256
    assert c.rotary_config.rotary_dim == 64
    assert c.rms_norm_eps == 1e-5
    # Two DSA layers (4 KV heads), one SWA layer (8), replicated FP32 indexer keys.
    assert c.kv_bytes_per_token(2, 2) == (2 * 2 + 4) * (192 + 128) * 2 + 2 * 128 * 4


def test_invalid_config():
    with pytest.raises(ValueError, match="rotary"):
        config(partial_rotary_factor=0.33)
    with pytest.raises(ValueError, match="positive"):
        config(attention_chunk_size=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_naive_tp_correctness_settings(monkeypatch, dtype, tp_size):
    import minisgl.distributed.info as distributed_info
    from minisgl.distributed import DistributedInfo
    from minisgl.engine.engine import _adjust_config

    monkeypatch.setattr(distributed_info, "_TP_INFO", DistributedInfo(0, tp_size))
    c = config()
    settings = SimpleNamespace(
        model_config=ModelConfig.from_hf(c),
        hf_config=c,
        attention_backend="auto",
        moe_backend="auto",
        use_dummy_weight=False,
        dtype=dtype,
        tp_info=SimpleNamespace(size=tp_size),
        cuda_graph_bs=[1, 2],
        cuda_graph_max_bs=2,
    )
    _adjust_config(settings)
    assert settings.dtype == dtype
    assert settings.attention_backend == "naive"
    assert settings.cuda_graph_bs == [] and settings.cuda_graph_max_bs == 0
    settings.attention_backend = "fi"
    with pytest.raises(ValueError, match="eager"):
        _adjust_config(settings)


def test_tokenizer_workers_register_native_config(monkeypatch):
    from minisgl.utils import load_tokenizer
    from transformers import AutoTokenizer
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    tokenizer = SimpleNamespace(chat_template="native template")

    def from_pretrained(path):
        assert path == "fixture"
        assert CONFIG_MAPPING["naive_n05_flash"] is NaiveN05FlashConfig
        return tokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", from_pretrained)
    assert load_tokenizer("fixture") is tokenizer


def test_partial_rope_leaves_tail_and_preserves_norm():
    x = torch.randn(5, 2, 192)
    out = rotary(x, torch.arange(5), 64, 10000)
    torch.testing.assert_close(out[0], x[0])
    assert torch.equal(out[..., 64:], x[..., 64:])
    torch.testing.assert_close(out[..., :64].square().sum(-1), x[..., :64].square().sum(-1))


def test_sink_is_a_zero_value_softmax_entry():
    q = torch.zeros(2, 1, 192)
    k = torch.zeros(1, 2, 192)
    v = torch.ones(1, 2, 128)
    allowed = torch.ones(1, 2, dtype=torch.bool)
    # Two zero-logit real keys and a sink of exp(log(2)) have total denominator four.
    out = attention(q, k, v, allowed, 0.5, torch.full((2,), torch.log(torch.tensor(2.0))))
    torch.testing.assert_close(out, torch.full_like(out, 0.25))
    empty = attention(q, k, v, torch.zeros_like(allowed), None, None)
    assert torch.equal(empty, torch.zeros_like(empty))


@pytest.mark.parametrize("top_k,length", [(2, 4), (2048, 2050)])
def test_sparse_selection_ties_and_causality(top_k, length):
    scores = torch.zeros(2, length)
    allowed = torch.ones_like(scores, dtype=torch.bool)
    allowed[0, 1:] = False
    mask = sparse_mask(scores, allowed, top_k)
    assert mask[0].sum() == 1
    assert mask[1, :top_k].all() and not mask[1, top_k:].any()


def test_indexer_fp8_rounding_uses_vector_scales():
    x = torch.tensor([[0.0, 0.0, 0.0], [0.001, 0.45, 10.0], [0.1, 45.0, 1000.0]])
    out = round_indexer_fp8(x)
    assert torch.equal(out[0], torch.zeros(3))
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[1] * 100, out[2])
    assert out[1, -1] == 10.0


def test_router_bias_changes_choices_but_not_weights():
    router = NaiveRouter(
        config(hidden_size=4, n_routed_experts=4, num_experts_per_tok=2, routed_scaling_factor=2.0)
    )
    router.weight = torch.zeros(4, 4)
    router.e_score_correction_bias = torch.tensor([0.0, 1.0, 2.0, 3.0])
    selected, weights = router.forward(torch.zeros(1, 4))
    assert set(selected[0].tolist()) == {2, 3}
    torch.testing.assert_close(weights, torch.ones_like(weights))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("packed", [False, True])
def test_naive_loader_shards_layer_heads_and_experts(tmp_path, monkeypatch, packed, dtype):
    import minisgl.models.weight as loader

    c = ModelConfig.from_hf(
        config(hidden_size=4, n_routed_experts=2, num_experts_per_tok=1, moe_intermediate_size=4)
    )
    monkeypatch.setattr(loader, "get_tp_info", lambda: SimpleNamespace(rank=1, size=2))
    gate = torch.arange(2 * 4 * 4).reshape(2, 4, 4).float()
    up = gate + 100
    down = gate.transpose(1, 2).contiguous()
    tensors = {
        "model.layers.0.self_attn.k_proj.weight": torch.arange(4 * 192 * 4)
        .reshape(4 * 192, 4)
        .float(),
        "model.layers.1.self_attn.k_proj.weight": torch.arange(8 * 192 * 4)
        .reshape(8 * 192, 4)
        .float(),
        "model.layers.0.self_attn.indexer.wq.weight": torch.ones(16 * 128, 4),
        "model.layers.1.mlp.gate.weight": torch.ones(2, 4),
        "model.layers.0.self_attn.o_proj.weight": torch.arange(16).reshape(4, 4).float(),
        "model.layers.0.mlp.down_proj.weight": torch.arange(16).reshape(4, 4).float(),
    }
    prefix = "model.layers.1.mlp.experts."
    if packed:
        tensors[prefix + "gate_up_proj"] = torch.cat((gate, up), 1)
        tensors[prefix + "down_proj"] = down
    else:
        for i in range(2):
            for name, value in (("gate", gate), ("up", up), ("down", down)):
                tensors[f"{prefix}{i}.{name}_proj.weight"] = value[i].contiguous()
    file = tmp_path / "weights.safetensors"
    save_file(tensors, str(file))
    result = dict(_load_naive_weight([str(file)], c, torch.device("cpu"), dtype=dtype))
    assert result["model.layers.0.self_attn.k_proj.weight"].shape == (2 * 192, 4)
    assert result["model.layers.1.self_attn.k_proj.weight"].shape == (4 * 192, 4)
    assert result["model.layers.0.self_attn.indexer.wq.weight"].shape == (16 * 128, 4)
    assert result["model.layers.1.mlp.gate.weight"].shape == (2, 4)
    torch.testing.assert_close(
        result[prefix + "gate_up_proj"], torch.cat((gate[:, 2:], up[:, 2:]), 1)
    )
    torch.testing.assert_close(
        result[prefix + "down_proj"], down[:, 2:, :] if dtype == torch.bfloat16 else down[:, :, 2:]
    )
    for name in ("model.layers.0.self_attn.o_proj.weight", "model.layers.0.mlp.down_proj.weight"):
        expected = tensors[name][2:, :] if dtype == torch.bfloat16 else tensors[name][:, 2:]
        torch.testing.assert_close(result[name], expected)


@pytest.mark.parametrize("size", [1, 2, 4])
def test_gather_last_dim_preserves_token_and_rank_order(monkeypatch, size):
    import minisgl.models.naive as naive

    monkeypatch.setattr(naive, "get_tp_info", lambda: SimpleNamespace(rank=0, size=size))
    full = torch.arange(3 * 2 * size).reshape(3, 2 * size)
    shards = full.chunk(size, dim=-1)

    class Communicator:
        def all_gather(self, value):
            assert value.is_contiguous()
            assert torch.equal(value, shards[0])
            return torch.cat(shards, dim=0)

    assert torch.equal(gather_last_dim(shards[0], Communicator()), full)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BF16 GEMM regression requires CUDA")
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("tokens", [1, 13, 127])
@pytest.mark.parametrize("has_bias", [False, True])
def test_bf16_tp_output_projection_matches_unsharded_gemm(monkeypatch, rank, tokens, has_bias):
    import minisgl.models.naive as naive

    monkeypatch.setattr(naive, "get_tp_info", lambda: SimpleNamespace(rank=rank, size=2))
    generator = torch.Generator(device="cuda").manual_seed(42)
    states = torch.randn(tokens, 1024, device="cuda", generator=generator).bfloat16()
    weight = (torch.randn(256, 1024, device="cuda", generator=generator) * 0.02).bfloat16()
    bias = torch.randn(256, device="cuda", generator=generator).bfloat16() if has_bias else None
    expected = torch.nn.functional.linear(states, weight, bias)
    inputs, outputs = states.chunk(2, dim=-1), expected.chunk(2, dim=-1)
    with torch_dtype(torch.bfloat16):
        projection = NaiveRowLinear(1024, 256, has_bias)
    projection.weight = weight.chunk(2, dim=0)[rank].contiguous()
    projection.bias = None if bias is None else bias.chunk(2)[rank].contiguous()

    class Communicator:
        calls = 0

        def all_gather(self, value):
            self.calls += 1
            if self.calls == 1:
                assert torch.equal(value, inputs[rank])
                return torch.cat(inputs, dim=0)
            torch.testing.assert_close(value, outputs[rank], atol=0, rtol=0)
            shards = list(outputs)
            shards[rank] = value
            return torch.cat(shards, dim=0)

    projection._comm = Communicator()
    actual = projection.forward(inputs[rank])
    assert projection.weight.dtype == actual.dtype == torch.bfloat16
    assert projection.weight.numel() == weight.numel() // 2
    assert projection._comm.calls == 2
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
