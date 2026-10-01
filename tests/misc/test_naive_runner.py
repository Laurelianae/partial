"""Supervisor behavior without SSH or CUDA."""

import io
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import naive_runner
from serve import stop_nodes


@pytest.mark.parametrize(
    "codes,exception", [([0, 0], None), ([1, None], RuntimeError), ([None, None], TimeoutError)]
)
def test_job_outcomes(tmp_path, monkeypatch, codes, exception):
    processes = [Mock(stdin=io.StringIO(), poll=Mock(return_value=code)) for code in codes]
    monkeypatch.setattr(naive_runner.subprocess, "Popen", Mock(side_effect=processes))
    cleanup = Mock()
    monkeypatch.setattr(naive_runner, "stop_nodes", cleanup)
    if exception:
        with pytest.raises(exception):
            naive_runner.run_job([["worker"]] * 2, tmp_path, -1)
    else:
        assert naive_runner.run_job([["worker"]] * 2, tmp_path, 1) == []
    cleanup.assert_called_once_with(processes)


def test_interruption_cleans_up(tmp_path, monkeypatch):
    process = Mock(stdin=io.StringIO(), poll=Mock(side_effect=KeyboardInterrupt))
    monkeypatch.setattr(naive_runner.subprocess, "Popen", Mock(return_value=process))
    cleanup = Mock()
    monkeypatch.setattr(naive_runner, "stop_nodes", cleanup)
    with pytest.raises(KeyboardInterrupt):
        naive_runner.run_job([["worker"]], tmp_path, 1)
    cleanup.assert_called_once_with([process])


def test_cleanup_escalates_with_bounded_waits():
    process = Mock(stdin=io.StringIO())
    process.wait.side_effect = [
        subprocess.TimeoutExpired("worker", 15),
        subprocess.TimeoutExpired("worker", 1),
        0,
    ]
    stop_nodes([process])
    assert process.stdin.closed
    process.terminate.assert_called_once()
    process.kill.assert_called_once()


def test_tp_command_keeps_lifetime_wrapper(monkeypatch):
    monkeypatch.setenv("PARTIAL_MASTER_ADDR", "192.0.2.1")
    cmd = naive_runner.command(1, "tools/measure_naive.py", [], "fixture", 2)
    index = cmd.index("tools/serve_worker.py")
    assert cmd[index + 1 : index + 4] == [
        "--watch-stdin",
        "--run-command",
        "tools/measure_naive.py",
    ]
    assert cmd[cmd.index("--node-rank") + 1] == "1"


def test_waits_for_slower_successful_rank(tmp_path, monkeypatch):
    processes = [
        Mock(stdin=io.StringIO(), poll=Mock(return_value=0)),
        Mock(stdin=io.StringIO(), poll=Mock(side_effect=[None, 0])),
    ]
    monkeypatch.setattr(naive_runner.subprocess, "Popen", Mock(side_effect=processes))
    monkeypatch.setattr(naive_runner, "stop_nodes", Mock())
    monkeypatch.setattr(naive_runner.time, "sleep", Mock())
    naive_runner.run_job([["worker"]] * 2, tmp_path, 1)
    assert processes[1].poll.call_count == 2


def test_normal_serving_peer_shutdown(tmp_path, monkeypatch):
    processes = [
        Mock(stdin=io.StringIO(), poll=Mock(side_effect=[None, 0])),
        Mock(stdin=io.StringIO(), poll=Mock(return_value=0)),
    ]
    monkeypatch.setattr(naive_runner.subprocess, "Popen", Mock(side_effect=processes))
    monkeypatch.setattr(naive_runner, "stop_nodes", Mock())
    monkeypatch.setattr(naive_runner.time, "sleep", Mock())
    naive_runner.run_job([["worker"]] * 2, tmp_path, 1, service_ranks=(1,))


def test_partial_launch_failure_cleans_up(tmp_path, monkeypatch):
    process = Mock(stdin=io.StringIO())
    monkeypatch.setattr(naive_runner.subprocess, "Popen", Mock(side_effect=[process, OSError()]))
    cleanup = Mock()
    monkeypatch.setattr(naive_runner, "stop_nodes", cleanup)
    with pytest.raises(OSError):
        naive_runner.run_job([["worker"]] * 2, tmp_path, 1)
    cleanup.assert_called_once_with([process])


def test_fixture_mismatch_prevents_execution(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(sys, "argv", ["runner", "regression", "fixture", "--output", str(tmp_path)])
    monkeypatch.setattr(naive_runner.subprocess, "check_output", Mock(return_value="commit"))
    jobs = Mock(
        side_effect=[
            [{"files": {"model.safetensors": "one"}}],
            [{"files": {"model.safetensors": "two"}}],
        ]
    )
    monkeypatch.setattr(naive_runner, "run_job", jobs)
    with pytest.raises(RuntimeError, match="hashes differ"):
        naive_runner.main()
    assert jobs.call_count == 2
    report = json.loads(next(tmp_path.glob("*/results.json")).read_text())
    assert report["status"] == "failed"
