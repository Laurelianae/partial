from __future__ import annotations

import logging
import multiprocessing as mp
import signal
import sys
import time
from dataclasses import replace
from typing import TYPE_CHECKING

from minisgl.distributed import DistributedInfo
from minisgl.utils import init_logger

from .workers import WorkerProcesses

if TYPE_CHECKING:
    from .args import ServerArgs


def scheduler_arguments(args: ServerArgs) -> list[ServerArgs]:
    """Global TP ranks select shards; each Spark has only local GPU zero."""
    if args.nnodes == 2:
        return [replace(args, tp_info=DistributedInfo(args.node_rank, 2), local_gpu_index=0)]
    return [
        replace(args, tp_info=DistributedInfo(rank, args.tp_info.size), local_gpu_index=rank)
        for rank in range(args.tp_info.size)
    ]


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue) -> None:
    import torch
    from minisgl.scheduler import Scheduler

    with torch.inference_mode():
        scheduler = Scheduler(args)
        scheduler.sync_all_ranks()
        ack_queue.put(f"Scheduler rank {args.tp_info.rank} is ready")
        if args.silent_output:
            logging.disable(logging.INFO)
        try:
            scheduler.run_forever()
        except KeyboardInterrupt:
            scheduler.shutdown()


def _start_tokenizers(args: ServerArgs, workers: WorkerProcesses, ack_queue: mp.Queue) -> None:
    from minisgl.tokenizer import tokenize_worker

    for tokenizer_id in range(args.num_tokenizer + 1):
        detokenizer = tokenizer_id == args.num_tokenizer
        workers.start(
            target=tokenize_worker,
            kwargs={
                "tokenizer_path": args.model_path,
                "addr": args.zmq_detokenizer_addr if detokenizer else args.zmq_tokenizer_addr,
                "backend_addr": args.zmq_backend_addr,
                "frontend_addr": args.zmq_frontend_addr,
                "local_bs": 1,
                "create": args.tokenizer_create_addr,
                "tokenizer_id": tokenizer_id,
                "ack_queue": ack_queue,
            },
            name=f"minisgl-{'detokenizer' if detokenizer else 'tokenizer'}-{tokenizer_id}",
        )


def _request_scheduler_exit(args: ServerArgs) -> None:
    import msgpack
    import zmq
    from minisgl.message import ExitMsg

    # Use rank zero's request path so both schedulers see the same exit.
    with zmq.Context() as context:
        with context.socket(zmq.PUSH) as socket:
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SNDTIMEO, 1000)
            socket.connect(args.zmq_backend_addr)
            socket.send(msgpack.packb(ExitMsg().encoder(), use_bin_type=True))
            time.sleep(0.1)


def launch_server(run_shell: bool = False) -> None:
    from .api_server import run_api_server
    from .args import parse_args

    args, run_shell = parse_args(sys.argv[1:], run_shell)
    logger = init_logger(__name__, f"node-{args.node_rank}")
    workers = WorkerProcesses()
    mp.set_start_method("spawn", force=True)
    ack_queue = mp.Queue()

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    previous_interrupt_handler = signal.getsignal(signal.SIGINT)

    def start_backend() -> None:
        local_schedulers = scheduler_arguments(args)
        for config in local_schedulers:
            workers.start(
                target=_run_scheduler,
                args=(config, ack_queue),
                name=f"minisgl-TP{config.tp_info.rank}-scheduler",
            )
        expected_acks = len(local_schedulers)
        if args.node_rank == 0:
            _start_tokenizers(args, workers, ack_queue)
            expected_acks += args.num_tokenizer + 1
        workers.wait_ready(ack_queue, expected_acks, args.startup_timeout)
        logger.info("All local workers and TP peers are ready")

    try:
        if args.node_rank == 0:

            def start_frontend_backend() -> None:
                start_backend()
                workers.monitor_frontend()

            run_api_server(args, start_frontend_backend, run_shell=run_shell)
        else:
            start_backend()
            scheduler_process = workers.processes[0]
            while scheduler_process.is_alive():
                scheduler_process.join(timeout=0.2)
            if scheduler_process.exitcode != 0:
                raise RuntimeError(f"Scheduler exited with code {scheduler_process.exitcode}")
    except KeyboardInterrupt:
        logger.info("Stopping local workers")
    finally:
        # A second stop signal must not interrupt child cleanup halfway through.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        request_exit = (lambda: _request_scheduler_exit(args)) if args.node_rank == 0 else None
        workers.stop(request_exit)
        ack_queue.close()
        signal.signal(signal.SIGTERM, previous_handler)
        signal.signal(signal.SIGINT, previous_interrupt_handler)
    if workers.failure is not None:
        raise RuntimeError(workers.failure)


if __name__ == "__main__":
    launch_server()
