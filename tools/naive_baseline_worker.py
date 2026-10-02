"""Serve and measure sequential streaming requests on the rank-zero loopback interface."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from naive_results import emit
from serve import stop_nodes


def summarize_events(events: list[dict], elapsed: float) -> dict:
    tokens = []
    first_text = None
    uid = None
    done = False
    finished = False
    for event in events:
        if event["data"] == "[DONE]":
            done = True
            continue
        payload = json.loads(event["data"])
        if uid is not None and uid != payload["id"]:
            raise ValueError("Response IDs changed within a request")
        uid = payload["id"]
        choice = payload["choices"][0]
        if choice["finish_reason"] is not None:
            finished = True
            continue
        # This server emits one event per token, including empty decoded strings.
        tokens.append(event["seconds"])
        if first_text is None and choice["delta"].get("content"):
            first_text = event["seconds"]
    if not done or not finished or len(tokens) < 2:
        raise ValueError("Incomplete streaming response or fewer than two tokens")
    duration = tokens[-1] - tokens[0]
    if duration <= 0:
        raise ValueError("Nonpositive decode duration")
    return {
        "uid": int(uid.removeprefix("cmpl-")),
        "ttft_seconds": tokens[0],
        "first_text_seconds": first_text,
        "token_seconds": tokens,
        "decode_latency_seconds": [b - a for a, b in zip(tokens, tokens[1:])],
        "decode_tokens_per_second": (len(tokens) - 1) / duration,
        "end_to_end_seconds": elapsed,
        "end_to_end_tokens_per_second": len(tokens) / elapsed,
    }


def request_sample(url: str, body: dict, timeout: float) -> dict:
    events = []
    started = time.perf_counter()
    try:
        request = Request(
            url + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=timeout) as response:
            for raw in response:
                elapsed = time.perf_counter() - started
                if elapsed > timeout:
                    raise TimeoutError("Request exceeded its deadline")
                line = raw.decode().strip()
                if not line.startswith("data: "):
                    continue
                events.append({"seconds": elapsed, "data": line[6:]})
                if line == "data: [DONE]":
                    break
        result = summarize_events(events, time.perf_counter() - started)
        if len(result["token_seconds"]) != body["max_tokens"]:
            raise ValueError("Unexpected generated token count")
        return {**result, "events": events}
    except BaseException:
        emit({"kind": "partial_request", "body": body, "events": events})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dist-init-addr")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--port", type=int, default=1939)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--without-telemetry", action="store_true")
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    if args.tokens < 2 or args.repeat < 1 or args.warmup < 0:
        parser.error("Need >=2 tokens, >=1 repetition and >=0 warmups")
    model = str(Path(args.model).expanduser())
    url = f"http://127.0.0.1:{args.port}"
    if args.node_rank == 0:
        try:
            with urlopen(url + "/v1/models", timeout=1):
                raise RuntimeError("Baseline port is already occupied")
        except URLError:
            pass
    server_args = [
        "--model",
        model,
        "--dtype",
        "bfloat16",
        "--port",
        str(args.port),
        "--tp-size",
        str(args.tp_size),
        "--nnodes",
        str(args.nnodes),
        "--node-rank",
        str(args.node_rank),
        "--cache-type",
        "naive",
        "--page-size",
        "1",
        "--num-pages",
        "4096",
        "--max-prefill-length",
        "4096",
        "--max-running-requests",
        "1",
        "--graph",
        "0",
        "--startup-timeout",
        str(args.startup_timeout),
    ]
    if args.dist_init_addr:
        server_args.extend(["--dist-init-addr", args.dist_init_addr])
    environment = os.environ.copy()
    environment["MINISGL_BASELINE_TELEMETRY"] = "0" if args.without_telemetry else "1"
    process = subprocess.Popen(
        [sys.executable, "tools/serve_worker.py", "--watch-stdin", *server_args],
        stdin=subprocess.PIPE,
        env=environment,
        start_new_session=True,
    )
    started = time.monotonic()
    try:
        if args.node_rank != 0:
            code = process.wait()
            if code:
                raise RuntimeError(f"Peer server exited: {code}")
            return
        from minisgl.utils import load_tokenizer
        from transformers import AutoConfig

        tokenizer = load_tokenizer(model)
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        cases = json.loads(Path("benchmark/online/naive-baseline-prompts.json").read_text())
        if args.fixture:
            cases = [{"name": "fixture", "prompt": "Explain why the sky is blue."}]
        for case in cases:
            case["input_ids"] = tokenizer.encode(case["prompt"])
            if len(case["input_ids"]) + args.tokens > min(4096, config.max_position_embeddings):
                raise ValueError("Workload exceeds configured sequence/cache capacity")
        emit(
            {"kind": "workload", "cases": cases, "server_args": server_args, "settings": vars(args)}
        )
        while True:
            if process.poll() is not None:
                raise RuntimeError("Server exited before readiness")
            try:
                with urlopen(url + "/v1/models", timeout=1):
                    break
            except URLError:
                if time.monotonic() - started > args.startup_timeout:
                    raise TimeoutError("Server readiness deadline exceeded")
                time.sleep(0.2)
        emit({"kind": "ready", "startup_seconds": time.monotonic() - started})
        for case in cases:
            for index in range(args.warmup + args.repeat):
                body = {
                    "model": model,
                    "prompt": case["prompt"],
                    "stream": True,
                    "max_tokens": args.tokens,
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": -1,
                    "ignore_eos": True,
                }
                print(
                    f"Baseline {case['name']} request {index + 1}/" f"{args.warmup + args.repeat}",
                    flush=True,
                )
                sample = request_sample(url, body, args.request_timeout)
                emit(
                    {
                        "kind": "sample",
                        "case": case["name"],
                        "index": index,
                        "warmup": index < args.warmup,
                        "body": body,
                        **sample,
                    }
                )
        emit({"kind": "client_complete"})
    finally:
        stop_nodes([process])


if __name__ == "__main__":
    main()
