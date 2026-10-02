"""Diagnostic invariants. CUDA/TP integration runs through naive-chunk-stability."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from chunk_stability import first_divergence, logit_metrics
from chunk_stability_remote import validate_results
from chunk_trace import aligned_rows, replay_linear


def test_intentionally_perturbed_output():
    reference = torch.tensor([1.0, 3.0, 2.0])
    candidate = torch.tensor([1.0, 3.0, 4.0])
    metrics = logit_metrics(reference, candidate)
    assert metrics["winner_changed"]
    assert metrics["max_abs"] == 2
    assert metrics["baseline_margin"] == metrics["candidate_margin"] == 1
    assert first_divergence([1, 2, 3], [1, 4, 3]) == 1
    assert first_divergence([1], [1]) is None
    with pytest.raises(ValueError):
        logit_metrics(reference, candidate * float("nan"))


def test_absolute_position_alignment():
    full = torch.arange(259 * 2).view(259, 2)
    for start in range(0, 259, 13):
        positions = torch.arange(start, min(start + 13, 259))
        assert torch.equal(aligned_rows(positions, full), full[start : start + 13])
    with pytest.raises(AssertionError):
        aligned_rows(torch.tensor([3, 3]), full)
    with pytest.raises(AssertionError):
        aligned_rows(torch.tensor([259]), full)


def test_winner_uses_greedy_tie_breaking():
    logits = torch.ones(20)
    metrics = logit_metrics(logits, logits)
    assert metrics["baseline_winner"] == metrics["candidate_winner"] == 0
    assert metrics["baseline_margin"] == metrics["candidate_margin"] == 0


def test_rank_disagreement_and_missing_completion():
    base = {"kind": "complete", "rank": 0, "rank_agreement": True, "cases": [1]}
    assert len(validate_results([base, {**base, "rank": 1}])) == 2
    for records in ([base], [base, base], [base, {**base, "rank": 1, "cases": [2]}]):
        with pytest.raises(AssertionError):
            validate_results(records)


def test_identical_input_replay_and_tf32_restoration():
    x, w = torch.arange(18).reshape(6, 3).bfloat16(), torch.ones(2, 3).bfloat16()
    before = torch.backends.cuda.matmul.allow_tf32
    result = replay_linear(x, w, None, [torch.arange(0, 2), torch.arange(2, 6)])
    assert result["bf16"]["max_abs"] == result["fp32"]["max_abs"] == 0
    assert torch.backends.cuda.matmul.allow_tf32 == before


def test_cache_reset_poisons_all_histories():
    from types import SimpleNamespace

    from chunk_stability import History

    banks = [[torch.ones(7, 2)], [torch.ones(7, 3)], [None, torch.ones(7, 4)]]
    cache = SimpleNamespace(_keys=banks[0], _values=banks[1], _index=banks[2])
    history = History(SimpleNamespace(ctx=SimpleNamespace(kv_cache=cache)))
    history.ids = [10, 11]
    previous_ids = history.ids
    history.reset()
    assert history.ids == [] and previous_ids == [10, 11]
    assert all(torch.isnan(tensor).all() for bank in banks for tensor in bank if tensor is not None)


def test_trace_perturbation_coverage_and_restoration():
    from types import SimpleNamespace

    from chunk_trace import Trace

    trace = Trace(None, 4, 4, 2)
    positions = torch.arange(4)
    trace.record("norm", torch.ones(4, 2), positions)
    trace.finish_pass()
    trace.baseline = False
    trace.record("norm", torch.ones(2, 2), positions[:2])
    with pytest.raises(AssertionError, match="Missing"):
        trace.finish_pass()
    trace.record("norm", torch.tensor([[1.0, 1.0], [1.0, 2.0]]), positions[2:])
    assert trace.differences["norm"]["first_position"] == 3
    assert trace.differences["norm"]["max_abs"] == 1
    with pytest.raises(AssertionError, match="Repeated"):
        trace.record("norm", torch.ones(2, 2), positions[2:])
    obj = SimpleNamespace(value=1)
    trace.patch(obj, "value", 2)
    trace.close()
    assert obj.value == 1 and not trace.reference and not trace.inputs


def test_trace_prefers_shortest_changed_prediction_including_ninth():
    from chunk_stability import trace_target

    cases = [
        {
            "ids": [0] * length,
            "chunks": [
                {
                    "size": chunk,
                    "first_divergent_token": None,
                    "forced_history": [
                        {"winner_changed": changed and step == 8, "max_abs": error}
                        for step in range(9)
                    ],
                }
            ],
        }
        for length, chunk, changed, error in [
            (129, 13, False, 10),
            (257, 128, True, 1),
            (2049, 64, True, 50),
        ]
    ]
    assert trace_target(cases) == (1, 128)
    for case in cases:
        for metric in case["chunks"][0]["forced_history"]:
            metric["winner_changed"] = False
    assert trace_target(cases) == (2, 64)


def test_linear_replay_prioritizes_aligned_order_after_late_difference():
    from types import SimpleNamespace

    from chunk_trace import Trace

    trace = Trace(None, 4, 4, 2)
    trace.order = ["q", "k", "v", "index"]
    trace.differences = dict.fromkeys(["k", "v", "index", "q"], {})
    trace.replays = {"attention": {}, "expert": {}}
    for name in trace.order:
        trace.inputs[name] = torch.ones(4, 2).bfloat16()
        trace.modules[name] = SimpleNamespace(weight=torch.ones(2, 2).bfloat16())
    trace.replay_linears()
    assert [k for k, v in trace.replays.items() if v.get("kind") == "linear"] == ["q", "k", "v"]
