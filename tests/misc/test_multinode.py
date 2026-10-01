from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest
import torch
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.distributed.check import check_matching_settings
from minisgl.message import AbortBackendMsg, ExitMsg, UserMsg
from minisgl.scheduler.io import SchedulerIOMixin
from minisgl.server.args import ServerArgs, parse_args
from minisgl.server.launch import scheduler_arguments
from minisgl.server.workers import WorkerProcesses


def server_args(**kwargs) -> ServerArgs:
    return ServerArgs(
        model_path="unused",
        dtype=torch.bfloat16,
        tp_info=DistributedInfo(0, 2),
        nnodes=2,
        dist_init_addr="192.0.2.1:29500",
        **kwargs,
    )


@pytest.mark.parametrize("node", [0, 1])
def test_each_spark_uses_local_gpu_zero(node):
    args = server_args(node_rank=node)
    args.validate_topology()
    workers = scheduler_arguments(args)
    assert len(workers) == 1
    assert workers[0].tp_info == DistributedInfo(node, 2)
    assert workers[0].gpu_index == 0
    assert workers[0].distributed_addr == "tcp://192.0.2.1:29500"


def test_single_host_rank_mapping():
    args = ServerArgs(model_path="unused", dtype=torch.bfloat16, tp_info=DistributedInfo(0, 4))
    assert [worker.gpu_index for worker in scheduler_arguments(args)] == [0, 1, 2, 3]
    assert args.distributed_addr == "tcp://127.0.0.1:1920"


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"nnodes": 3}, "--nnodes"),
        ({"node_rank": 2}, "--node-rank"),
        ({"tp_info": DistributedInfo(0, 1)}, "--tp-size 2"),
        ({"dist_init_addr": None}, "--dist-init-addr"),
        ({"dist_init_addr": "127.0.0.1:29500"}, "reachable"),
        ({"dist_init_addr": "host:70000"}, "HOST:PORT"),
        ({"distributed_timeout": 0}, "positive"),
        ({"distributed_timeout": float("nan")}, "positive"),
        ({"startup_timeout": float("inf")}, "positive"),
        ({"startup_timeout": -1}, "positive"),
    ],
)
def test_invalid_topology(changes, message):
    from dataclasses import replace

    with pytest.raises(ValueError, match=message):
        replace(server_args(), **changes).validate_topology()


def test_invalid_cli_fails_before_model_download():
    with pytest.raises(SystemExit) as error:
        parse_args(["--model", "does-not-exist", "--nnodes", "2", "--tp", "2"])
    assert error.value.code == 2


def test_cli_two_node_defaults():
    args, _ = parse_args(
        [
            "--model",
            "unused",
            "--dtype",
            "bfloat16",
            "--nnodes",
            "2",
            "--node-rank",
            "1",
            "--tp",
            "2",
            "--dist-init-addr",
            "host:29500",
        ]
    )
    assert args.distributed_timeout == 120
    assert args.startup_timeout == 600
    assert not args.use_pynccl


def test_configuration_mismatch_names_fields():
    check_matching_settings([{"dtype": "bf16"}, {"dtype": "bf16"}])
    with pytest.raises(ValueError, match="rank 1.*cache_type, dtype"):
        check_matching_settings(
            [
                {"dtype": "bf16", "cache_type": "radix"},
                {"dtype": "fp16", "cache_type": "naive"},
            ]
        )


def test_peer_receives_ordered_requests_and_control_messages(monkeypatch):
    import msgpack

    messages = [
        UserMsg(
            uid=7,
            input_ids=torch.tensor([1, 2], dtype=torch.int32),
            sampling_params=SamplingParams(max_tokens=4),
        ),
        AbortBackendMsg(uid=7),
        ExitMsg(),
    ]
    raw = [msgpack.packb(message.encoder(), use_bin_type=True) for message in messages]

    def broadcast(payload, src, group):
        assert src == 0
        payload[0] = raw

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast)
    peer = object.__new__(SchedulerIOMixin)
    peer._is_primary = False
    peer.tp_cpu_group = object()
    received = peer._recv_msg_multi_node(blocking=True)
    assert received[0].input_ids.tolist() == [1, 2]
    assert received[1] == AbortBackendMsg(uid=7)
    assert isinstance(received[2], ExitMsg)


def test_idle_rank_zero_broadcasts_empty_batch(monkeypatch):
    polls, batches = [], []
    primary = object.__new__(SchedulerIOMixin)
    primary._is_primary = True
    primary.tp_cpu_group = object()
    primary._recv_from_tokenizer = SimpleNamespace(
        socket=SimpleNamespace(poll=lambda timeout: polls.append(timeout)),
        empty=lambda: True,
    )
    monkeypatch.setattr(
        torch.distributed,
        "broadcast_object_list",
        lambda payload, **kwargs: batches.append(payload[0]),
    )
    assert primary._recv_msg_multi_node(blocking=True) == []
    assert polls == [100]
    assert batches == [[]]


def test_readiness_timeout_and_failed_child():
    workers = WorkerProcesses()
    with pytest.raises(TimeoutError, match="startup-timeout"):
        workers.wait_ready(queue.Queue(), count=1, timeout=0.01)
    workers.processes = [SimpleNamespace(name="scheduler", exitcode=1)]
    with pytest.raises(RuntimeError, match="scheduler exited with code 1"):
        workers.wait_ready(queue.Queue(), count=1, timeout=1)


def test_readiness_waits_for_every_local_worker():
    acknowledgments = queue.Queue()
    acknowledgments.put("scheduler ready")
    workers = WorkerProcesses()
    with pytest.raises(TimeoutError):
        workers.wait_ready(acknowledgments, count=2, timeout=0.01)


def test_worker_monitor_stops_frontend(monkeypatch):
    import signal
    import threading

    signaled = threading.Event()
    monkeypatch.setattr(
        "minisgl.server.workers.os.kill",
        lambda pid, sig: signaled.set() if sig == signal.SIGTERM else None,
    )
    workers = WorkerProcesses()
    workers.processes = [SimpleNamespace(name="scheduler", exitcode=1)]
    workers.monitor_frontend()
    assert signaled.wait(timeout=2)
    workers.stopping.set()
    workers.monitor.join()
    assert "scheduler" in workers.failure


@pytest.mark.parametrize("rank", [0, 1])
def test_only_rank_zero_samples_before_tokens_are_copied_to_host(monkeypatch, rank):
    from contextlib import nullcontext

    from minisgl.engine.engine import Engine

    engine = object.__new__(Engine)
    engine.multi_node = True
    engine.tp_info = DistributedInfo(rank, 2)
    engine.device = torch.device("cpu")
    engine.stream = object()
    engine.ctx = SimpleNamespace(forward_batch=lambda batch: nullcontext())
    engine.graph_runner = SimpleNamespace(can_use_cuda_graph=lambda batch: False)
    engine.model = SimpleNamespace(forward=lambda: torch.zeros((2, 32)))
    operations = []

    def sample(logits, args):
        assert rank == 0, "Peer must not make an independent sampling decision"
        operations.append("sample")
        return torch.tensor([5, 7])

    def broadcast(tokens, src):
        assert src == 0
        operations.append("broadcast")
        if rank == 1:
            tokens.copy_(torch.tensor([5, 7], dtype=torch.int32))

    engine.sampler = SimpleNamespace(sample=sample)
    batch = SimpleNamespace(size=2, reqs=[SimpleNamespace(complete_one=lambda: None)] * 2)
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: engine.stream)
    monkeypatch.setattr(
        torch.cuda,
        "Event",
        lambda: SimpleNamespace(record=lambda stream: operations.append("copy complete")),
    )
    output = engine.forward_batch(batch, args=None)
    assert output.next_tokens_cpu.tolist() == [5, 7]
    assert operations == (["sample"] if rank == 0 else []) + ["broadcast", "copy complete"]


def test_ssh_wrapper_exits_when_server_fails_with_stdin_open(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    # Keep stdin open while the child exits, reproducing the interpreter-shutdown race.
    (tmp_path / "minisgl.py").write_text("raise SystemExit(7)\n")
    wrapper = Path(__file__).resolve().parents[2] / "tools" / "serve_worker.py"
    process = subprocess.Popen(
        [sys.executable, str(wrapper), "--watch-stdin"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
    )
    try:
        assert process.wait(timeout=10) == 7
        assert b"Fatal Python error" not in process.stderr.read()
    finally:
        process.stdin.close()
        if process.poll() is None:
            process.kill()
            process.wait()


def test_ssh_wrapper_stops_server_on_connection_close(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    (tmp_path / "minisgl.py").write_text(
        "import time\nprint('ready', flush=True)\ntime.sleep(60)\n"
    )
    wrapper = Path(__file__).resolve().parents[2] / "tools" / "serve_worker.py"
    process = subprocess.Popen(
        [sys.executable, str(wrapper), "--watch-stdin"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
    )
    try:
        assert process.stdout.readline().strip() == b"ready"
        process.stdin.close()
        assert process.wait(timeout=5) == 143
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_config_copy_preserves_revision_without_mutating_cached_config(monkeypatch):
    from minisgl.utils.hf import cached_load_hf_config
    from transformers import PretrainedConfig

    source = PretrainedConfig(_commit_hash="test-revision")
    monkeypatch.setattr("minisgl.utils.hf._load_hf_config", lambda _: source)
    copy = cached_load_hf_config("unused")
    assert copy is not source
    assert copy._commit_hash == "test-revision"


def test_offline_llm_rejects_two_nodes_before_initializing_cuda():
    from minisgl.llm import LLM

    with pytest.raises(ValueError, match="single-node"):
        LLM("unused", nnodes=2)


def test_configuration_mismatch_detects_missing_optional_field():
    with pytest.raises(ValueError, match="num_page_override"):
        check_matching_settings([{"num_page_override": None}, {}])


def test_cache_uses_local_memory_budget_then_the_smallest_peer_page_count(monkeypatch):
    from minisgl.engine.engine import Engine

    engine = object.__new__(Engine)
    engine.multi_node = True
    engine.dtype = torch.bfloat16
    engine.initial_local_free_memory = 1000
    engine.tp_cpu_group = object()

    def memory_range():
        engine.local_free_memory = 800
        return 600, 800

    engine._sync_get_memory = memory_range
    config = SimpleNamespace(
        num_page_override=None,
        memory_ratio=0.9,
        page_size=1,
        tp_info=DistributedInfo(0, 2),
        model_config=SimpleNamespace(
            is_naive=False, kv_bytes_per_token=lambda tp_size, itemsize: 2 * itemsize
        ),
    )

    def reduce_pages(pages, op, group):
        # Local budget: 900 - (1000 - 800) = 700 bytes, four bytes per page.
        assert pages.item() == 175
        assert op == torch.distributed.ReduceOp.MIN
        pages.fill_(150)

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce_pages)
    assert engine._determine_num_pages(old_free_memory=900, config=config) == 150
