"""Capture bounds, trace identity, attribution, and interrupted lifecycle."""

import copy
import gzip
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from naive_profile import analyze_trace, validate_manifest


def test_capture_windows():
    from minisgl.profiling import capture_phase

    for request in range(6):
        phases = [capture_phase(request, step, 1, 2) for step in range(32)]
        if request in (0, 3):
            assert phases == [None] * 32
        else:
            assert [i for i, phase in enumerate(phases) if phase == "prefill"] == [0]
            assert [i for i, phase in enumerate(phases) if phase == "decode"] == [8, 9, 10, 11]


def manifest_records():
    return [
        {"kind": "sample", "uid": 9, "case": "fixture", "index": 0, "warmup": True},
        {"kind": "sample", "uid": 10, "case": "fixture", "index": 1, "warmup": False},
        *[
            {
                "kind": "profile_trace",
                "uid": 10,
                "request": 1,
                "rank": rank,
                "phase": phase,
                "steps": steps,
                "complete": True,
            }
            for rank in (0, 1)
            for phase, steps in (("prefill", [0]), ("decode", [8, 9, 10, 11]))
        ],
    ]


def test_manifest_validation():
    records = manifest_records()
    assert len(validate_manifest(records, 2)) == 4
    for broken in (records[:-1], records + [records[-1]]):
        with pytest.raises(ValueError, match="profile traces"):
            validate_manifest(broken, 2)
    for field, value in (
        ("rank", 2),
        ("uid", 9),
        ("complete", False),
        ("steps", [7, 8, 9, 10]),
        ("request", 0),
    ):
        broken = copy.deepcopy(records)
        broken[-1][field] = value
        with pytest.raises(ValueError):
            validate_manifest(broken, 2)


@pytest.mark.parametrize("launch_category", ["cuda_runtime", "cuda_driver"])
def test_trace_exclusive_attribution_and_overlap(tmp_path, launch_category):
    def event(name, cat, start, duration, args=None):
        return {
            "ph": "X",
            "name": name,
            "cat": cat,
            "ts": start,
            "dur": duration,
            "pid": 1,
            "tid": 1,
            "args": args or {},
        }

    events = [
        event("naive::step/0", "user_annotation", 0, 100),
        event("naive::experts", "user_annotation", 10, 80),
        event("naive::int4_reconstruct", "user_annotation", 20, 30),
        event("aten::mul", "cpu_op", 25, 10, {"External id": 1}),
        event("cudaLaunchKernel", launch_category, 27, 1, {"External id": 1, "correlation": 8}),
        event("mul", "kernel", 30, 20, {"correlation": 8}),
        event("mul", "kernel", 40, 20, {"External id": 1}),
    ]
    path = tmp_path / "trace.json.gz"
    with gzip.open(path, "wt") as stream:
        json.dump({"traceEvents": events}, stream)
    result = analyze_trace(path)
    assert result["steps"] == [0]
    assert result["gpu_busy_seconds"] == 30 / 1e6
    assert result["step_gpu_idle_seconds"] == 70 / 1e6
    assert result["cpu_exclusive_seconds"]["experts"] == 50 / 1e6
    assert result["cpu_exclusive_seconds"]["int4_reconstruct"] == 30 / 1e6
    assert result["gpu_category_busy_seconds"]["int4_reconstruct"] == 30 / 1e6
    with gzip.open(path, "wt") as stream:
        json.dump({"traceEvents": events[:4]}, stream)
    with pytest.raises(ValueError, match="No CUDA"):
        analyze_trace(path)


def test_abort_exports_incomplete_and_disables_ranges(tmp_path, monkeypatch, capsys):
    import minisgl.profiling as module

    profiler = Mock()
    profiler.export_chrome_trace.side_effect = lambda path: Path(path).write_text("{}")
    monkeypatch.setattr(module.torch.profiler, "profile", Mock(return_value=profiler))
    capture = module.RequestProfiler(1, str(tmp_path), 0, 1)
    capture.begin(5)
    capture.before_step()
    assert module._ACTIVE
    capture.abort()
    assert not module._ACTIVE
    assert capture.uid is None
    assert capture.profiler is None
    profiler.stop.assert_called_once()
    row = json.loads(capsys.readouterr().out.split("=", 1)[1])
    assert not row["complete"]
    assert row["rank"] == 1
    assert Path(row["path"]).exists()
    capture.abort()  # Idempotent shutdown.
    profiler.stop.assert_called_once()
