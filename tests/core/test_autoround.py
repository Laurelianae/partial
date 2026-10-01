from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from minisgl.models.autoround import (
    dense_shapes,
    dequantize,
    execution_workspace,
    inspect_checkpoint,
    validate_quantization,
)
from minisgl.models.config import ModelConfig
from minisgl.models.naive_config import NaiveN05FlashConfig
from minisgl.models.weight import _load_naive_weight
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from autoround_reference import QUANTIZATION, pack_projection, reference_projection  # noqa: E402


def config():
    return NaiveN05FlashConfig(
        vocab_size=32,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=256,
        n_routed_experts=2,
        num_experts_per_tok=1,
        num_hidden_layers=2,
        hybrid_layer_pattern=[0, 1],
        moe_layer_freq=[0, 1],
        quantization_config=QUANTIZATION.copy(),
    )


def checkpoint(c):
    tensors = {
        name: torch.zeros(
            shape,
            dtype=(
                torch.float32
                if name.endswith(("mlp.gate.weight", "e_score_correction_bias"))
                else torch.bfloat16
            ),
        )
        for name, shape in dense_shapes(c).items()
    }
    for expert in range(c.n_routed_experts):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            shape = (
                (c.hidden_size, c.moe_intermediate_size)
                if projection == "down_proj"
                else (c.moe_intermediate_size, c.hidden_size)
            )
            for component, tensor in pack_projection(torch.randn(shape) * 0.02).items():
                tensors[f"model.layers.1.mlp.experts.{expert}.{projection}.{component}"] = tensor
    return tensors


def test_signed_words_order_groups_and_half_rounding():
    # All sixteen nibbles, signed words, different groups and every zero value.
    words = torch.tensor([0x76543210, -0x01234568], dtype=torch.int32)
    qweight = words.repeat(32, 4)
    qzeros = torch.tensor([[0x76543210], [-0x01234568]], dtype=torch.int32)
    scales = torch.tensor([[0.10004] * 8, [0.3333] * 8], dtype=torch.float16)
    parts = {"qweight": qweight, "qzeros": qzeros, "scales": scales}
    actual = dequantize(**parts)
    torch.testing.assert_close(actual, reference_projection(parts), atol=0, rtol=0)
    assert actual.dtype == torch.bfloat16 and actual.is_contiguous()
    assert actual[0, 0] == torch.tensor(-float(scales[0, 0]), dtype=torch.bfloat16)
    assert not torch.equal(actual[:, :128], actual[:, 128:])


@pytest.mark.parametrize(
    "field,value",
    [
        ("bits", 8),
        ("sym", False),
        ("group_size", 64),
        ("desc_act", True),
        ("act_bits", 8),
        ("packing_format", "exl3"),
        ("quant_method", "gptq"),
    ],
)
def test_unsupported_quantization(field, value):
    c = config()
    c.quantization_config[field] = value
    with pytest.raises(ValueError, match="Unsupported|unsupported"):
        validate_quantization(c)


@pytest.mark.parametrize(
    "fault", [None, "missing", "duplicate", "shape", "dtype", "index", "g_idx", "dense", "extra"]
)
def test_headers_across_shards(tmp_path, fault):
    c = config()
    tensors = checkpoint(c)
    name = "model.layers.1.mlp.experts.0.gate_proj.qweight"
    if fault == "missing":
        del tensors[name]
    elif fault == "shape":
        tensors[name] = tensors[name][:-1]
    elif fault == "dtype":
        tensors[name] = tensors[name].float()
    elif fault == "g_idx":
        tensors[name.replace("qweight", "g_idx")] = torch.arange(c.hidden_size).int()
    elif fault == "dense":
        del tensors["model.norm.weight"]
    elif fault == "extra":
        tensors["unexpected.weight"] = torch.ones(2, dtype=torch.bfloat16)
    items = list(tensors.items())
    # Interleave components and experts across files, unlike whole-expert shards.
    save_file(dict(items[::2]), str(tmp_path / "a.safetensors"))
    save_file(dict(items[1::2]), str(tmp_path / "b.safetensors"))
    index = {
        "weight_map": {
            k: "a.safetensors" if i % 2 == 0 else "b.safetensors" for i, (k, _) in enumerate(items)
        }
    }
    if fault == "index":
        index["weight_map"][name] = "wrong.safetensors"
    if fault == "duplicate":
        save_file({name: tensors[name]}, str(tmp_path / "c.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    if fault:
        with pytest.raises(ValueError):
            inspect_checkpoint(tmp_path, c, 2)
    else:
        report = inspect_checkpoint(tmp_path, c, 2)
        assert report["checkpoint_bytes"] == sum(
            t.numel() * t.element_size() for t in tensors.values()
        )
        assert report["per_rank_peak_weight_bytes"] == report[
            "per_rank_resident_bytes"
        ] + execution_workspace(c, 2)


@pytest.mark.parametrize("rank", [0, 1])
def test_streaming_tp_reconstruction(tmp_path, monkeypatch, rank):
    import minisgl.models.weight as loader

    c = config()
    tensors = checkpoint(c)
    items = list(tensors.items())
    files = [tmp_path / "a.safetensors", tmp_path / "b.safetensors"]
    for i, file in enumerate(files):
        save_file(dict(items[i::2]), str(file))
    monkeypatch.setattr(loader, "get_tp_info", lambda: SimpleNamespace(rank=rank, size=2))
    loaded = dict(
        _load_naive_weight(
            [str(f) for f in files],
            ModelConfig.from_hf(c),
            torch.device("cpu"),
            dtype=torch.bfloat16,
        )
    )
    for projection in ("gate_proj", "up_proj", "down_proj"):
        prefix = f"model.layers.1.mlp.experts.{projection}."
        for expert in range(c.n_routed_experts):
            original = {
                p: tensors[f"model.layers.1.mlp.experts.{expert}.{projection}.{p}"]
                for p in ("qweight", "qzeros", "scales")
            }
            local = {p: loaded[prefix + p][expert] for p in original}
            torch.testing.assert_close(
                dequantize(**local),
                reference_projection(original).chunk(2, 0)[rank],
                atol=0,
                rtol=0,
            )
            assert local["scales"].dtype == torch.float16
            assert local["qweight"].dtype == torch.int32


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Production projection requires remote CUDA"
)
@pytest.mark.parametrize("shape", [(2048, 4096), (4096, 2048)])
@pytest.mark.parametrize("rank", [0, 1])
def test_production_projection(shape, rank, monkeypatch):
    import minisgl.models.naive as naive

    monkeypatch.setattr(naive, "get_tp_info", lambda: SimpleNamespace(rank=rank, size=2))

    torch.manual_seed(42)
    parts = {k: v.cuda() for k, v in pack_projection(torch.randn(shape) * 0.02).items()}
    reference = reference_projection(parts)
    local = {k: v.chunk(2, 1)[rank].contiguous() for k, v in parts.items()}
    weight = dequantize(**local)
    torch.testing.assert_close(weight, reference.chunk(2, 0)[rank], atol=0, rtol=0)
    states = torch.randn(13, shape[1], device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(
        naive.output_shard_linear(states, weight),
        torch.nn.functional.linear(states, reference).chunk(2, -1)[rank],
        atol=0.025,
        rtol=0.025,
    )


def test_fp16_multiply_rounds_before_bf16():
    scale = torch.tensor(4.178285598754883e-05, dtype=torch.float16)
    parts = {
        "qweight": torch.full((16, 8), 0x44444444, dtype=torch.int32),
        "qzeros": torch.zeros(1, 1, dtype=torch.int32),
        "scales": scale.expand(1, 8).contiguous(),
    }
    actual = dequantize(**parts)
    assert torch.equal(actual, (scale * 3).bfloat16().expand(8, 128))
    assert actual[0, 0] != (scale.float() * 3).bfloat16()


def test_cache_reserves_dequant_workspace(monkeypatch):
    import minisgl.distributed.info as distributed_info
    import minisgl.models.autoround as autoround
    from minisgl.distributed import DistributedInfo
    from minisgl.engine.engine import Engine

    monkeypatch.setattr(distributed_info, "_TP_INFO", DistributedInfo(0, 1))
    engine = object.__new__(Engine)
    engine.multi_node = False
    engine.dtype = torch.bfloat16
    engine.local_free_memory = 800
    engine._sync_get_memory = lambda: (800, 800)
    c = SimpleNamespace(
        model_config=SimpleNamespace(
            is_naive=True, naive_config=config(), kv_bytes_per_token=lambda size, itemsize: 4
        ),
        tp_info=SimpleNamespace(size=1),
        page_size=1,
        memory_ratio=0.9,
        num_page_override=None,
    )
    monkeypatch.setattr(autoround, "execution_workspace", lambda config, size: 100)
    assert engine._determine_num_pages(1000, c) == 150
    c.num_page_override = 200
    with pytest.raises(ValueError, match="workspace"):
        engine._determine_num_pages(1000, c)


@pytest.mark.parametrize("fault", ["duplicate_header", "offsets", "truncated", "missing_field"])
def test_malformed_headers(tmp_path, fault):
    import struct

    spec = {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}
    if fault == "offsets":
        spec["data_offsets"] = [1, 3]
    if fault == "missing_field":
        del spec["shape"]
    header = json.dumps({"x": spec})
    if fault == "duplicate_header":
        header = '{"x":' + json.dumps(spec) + ',"x":' + json.dumps(spec) + "}"
    payload = header.encode()
    data = struct.pack("<Q", len(payload)) + payload + b"\x00\x00"
    if fault == "truncated":
        data = data[:7]
    (tmp_path / "a.safetensors").write_bytes(data)
    with pytest.raises(ValueError):
        inspect_checkpoint(tmp_path, config())


def test_packed_operator_strict_state_and_no_decoded_cache():
    from minisgl.models.autoround import GPTQProjection

    op = GPTQProjection(2, 128, 128)
    state = {name: tensor.clone() for name, tensor in op.state_dict().items()}
    state["qweight"] = state["qweight"].bfloat16()
    with pytest.raises(AssertionError):
        op.load_state_dict(state)
    parts = pack_projection(torch.randn(128, 128) * 0.02)
    op.load_state_dict(
        {name: tensor.expand(2, *tensor.shape).contiguous() for name, tensor in parts.items()}
    )
    result = op.forward(1)
    torch.testing.assert_close(result, reference_projection(parts), atol=0, rtol=0)
    assert set(op.state_dict()) == {"qweight", "qzeros", "scales"}


@pytest.mark.parametrize("field", ["packing_format", "format", "backend"])
def test_quantization_format_metadata_fields(field):
    c = config()
    packing_format = c.quantization_config.pop("packing_format")
    c.quantization_config[field] = packing_format
    assert validate_quantization(c)


def test_serialized_packing_format_takes_precedence_over_backend():
    c = config()
    c.quantization_config["backend"] = "auto_round:torch_zp"
    assert validate_quantization(c)
    c.quantization_config["packing_format"] = "auto_round:gptqmodel"
    c.quantization_config["format"] = "auto_round:auto_gptq"
    c.quantization_config["backend"] = "auto_round:auto_gptq"
    with pytest.raises(ValueError, match="packing_format='auto_round:gptqmodel'"):
        validate_quantization(c)


def test_missing_packing_format_reports_received_metadata():
    c = config()
    del c.quantization_config["packing_format"]
    with pytest.raises(ValueError, match="missing packing_format=None"):
        validate_quantization(c)


@pytest.mark.parametrize("index_size", [0, 1, -1, "0"])
def test_checkpoint_index_size_placeholder(tmp_path, index_size):
    c = config()
    tensors = checkpoint(c)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    index = {
        "metadata": {"total_size": index_size},
        "weight_map": dict.fromkeys(tensors, "model.safetensors"),
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    if index_size == 0:
        report = inspect_checkpoint(tmp_path, c, 2)
        assert report["index_total_size_bytes"] == 0
        assert report["checkpoint_bytes"] == sum(
            t.numel() * t.element_size() for t in tensors.values()
        )
    else:
        with pytest.raises(ValueError, match="total_size"):
            inspect_checkpoint(tmp_path, c, 2)
