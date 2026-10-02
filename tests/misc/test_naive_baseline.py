"""Baseline metric semantics, completeness checks, and opt-in telemetry lifecycle."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from naive_baseline import validate_results
from naive_baseline_worker import request_sample, summarize_events


def events():
    def event(seconds, content=None, finish=None):
        return {
            "seconds": seconds,
            "data": json.dumps(
                {
                    "id": "cmpl-7",
                    "choices": [
                        {"delta": {"content": content} if content else {}, "finish_reason": finish}
                    ],
                }
            ),
        }

    return [
        event(2),
        event(3, "hello"),
        event(5, " world"),
        event(5.1, finish="stop"),
        {"seconds": 5.2, "data": "[DONE]"},
    ]


def records():
    sample = {
        "kind": "sample",
        "case": "test",
        "index": 0,
        "warmup": False,
        **summarize_events(events(), 6),
    }
    memory = dict.fromkeys(
        ("allocated", "reserved", "peak_allocated", "peak_reserved", "device_free", "device_total"),
        1,
    )
    traces = [
        {
            "kind": "telemetry",
            "uid": 7,
            "rank": rank,
            "input_ids": [42],
            "output_ids": [1, 2, 3],
            "token_seconds": [1, 2, 4],
            "prefill_seconds": 1,
            **dict.fromkeys(
                ("resident_bytes", "prefill_memory_bytes", "decode_memory_bytes"), memory
            ),
        }
        for rank in (0, 1)
    ]
    return [
        {"kind": "workload", "cases": [{"name": "test", "input_ids": [42]}]},
        {"kind": "client_complete"},
        sample,
        *traces,
    ]


def test_empty_text_is_still_a_token():
    result = summarize_events(events(), 6)
    assert result["ttft_seconds"] == 2
    assert result["first_text_seconds"] == 3
    assert result["token_seconds"] == [2, 3, 5]
    assert result["decode_latency_seconds"] == [1, 2]
    assert result["decode_tokens_per_second"] == 2 / 3
    assert result["end_to_end_tokens_per_second"] == 0.5


@pytest.mark.parametrize("omit", [3, 4])
def test_incomplete_stream_rejected(omit):
    data = events()
    del data[omit]
    with pytest.raises(ValueError, match="Incomplete"):
        summarize_events(data, 6)


def test_missing_duplicate_and_disagreeing_ranks():
    data = records()
    assert validate_results(data, 2, 0, 1, 3)["test"]["repeated_tokens_identical"]
    for malformed in (data[:-1], data + [data[-1]]):
        with pytest.raises(ValueError, match="rank telemetry"):
            validate_results(malformed, 2, 0, 1, 3)
    bad = copy.deepcopy(data)
    bad[-1]["output_ids"][-1] = 4
    with pytest.raises(ValueError, match="disagreement"):
        validate_results(bad, 2, 0, 1, 3)
    bad[-1]["output_ids"][-1] = 3
    bad[-1]["input_ids"] = [43]
    with pytest.raises(ValueError, match="Prompt tokens"):
        validate_results(bad, 2, 0, 1, 3)


def test_request_failure_retains_partial_events(monkeypatch):
    import naive_baseline_worker as worker

    response = Mock()
    response.__enter__ = Mock(return_value=iter([b"data: [DONE]\n"]))
    response.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(worker, "urlopen", Mock(return_value=response))
    emit = Mock()
    monkeypatch.setattr(worker, "emit", emit)
    with pytest.raises(ValueError):
        request_sample("http://localhost", {"max_tokens": 3}, 10)
    assert emit.call_args.args[0]["kind"] == "partial_request"
    assert emit.call_args.args[0]["events"][0]["data"] == "[DONE]"


def test_request_deadline(monkeypatch):
    import naive_baseline_worker as worker

    response = Mock()
    response.__enter__ = Mock(return_value=iter([b"data: [DONE]\n"]))
    response.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(worker, "urlopen", Mock(return_value=response))
    monkeypatch.setattr(worker.time, "perf_counter", Mock(side_effect=[0, 11]))
    with pytest.raises(TimeoutError):
        request_sample("http://localhost", {"max_tokens": 3}, 10)


def test_telemetry_phase_memory_and_tokens(monkeypatch, capsys):
    import minisgl.scheduler.telemetry as telemetry
    from minisgl.scheduler.telemetry import BaselineTelemetry

    reset = Mock()
    monkeypatch.setattr(telemetry.torch.cuda, "reset_peak_memory_stats", reset)
    monkeypatch.setattr(BaselineTelemetry, "memory", staticmethod(lambda: {"allocated": 4}))
    monkeypatch.setattr(telemetry.time, "perf_counter", Mock(side_effect=[10, 12, 13]))
    recorder = BaselineTelemetry(1)
    msg = SimpleNamespace(uid=7, input_ids=SimpleNamespace(tolist=lambda: [42]))
    recorder.begin(msg)
    with pytest.raises(RuntimeError, match="sequential"):
        recorder.begin(msg)
    recorder.token(7, 5)
    recorder.token(7, 6)
    recorder.finish()
    row = json.loads(capsys.readouterr().out.split("=", 1)[1])
    assert row["input_ids"] == [42]
    assert row["output_ids"] == [5, 6]
    assert row["token_seconds"] == [2, 3]
    assert row["prefill_seconds"] == 2
    assert reset.call_count == 2
    assert recorder.record is None
