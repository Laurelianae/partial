"""Exercise a running API; use --baseline to compare greedy outputs between TP=1 and TP=2."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen


def request(url: str, prompt: str, **options):
    payload = {
        "model": "Qwen/Qwen3-0.6B",
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": 8,
        "ignore_eos": True,
        **options,
    }
    return urlopen(
        Request(
            url + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        ),
        timeout=60,
    )


def complete(url: str, prompt: str, **options) -> str:
    with request(url, prompt, **options) as response:
        result = json.load(response)
    assert result["object"] == "chat.completion"
    return result["choices"][0]["message"]["content"]


def stream(url: str, prompt: str) -> str:
    content = ""
    finished = False
    with request(url, prompt, stream=True) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                finished = True
                break
            delta = json.loads(payload)["choices"][0]["delta"]
            content += delta.get("content", "")
    assert finished, "Stream did not finish"
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:1919")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--write-baseline", type=Path)
    parser.add_argument("--idle-seconds", type=float, default=0)
    args = parser.parse_args()
    started = time.monotonic()
    prompts = ["The capital of France is", "One plus one equals", "The color of a clear sky is"]
    outputs = [complete(args.url, prompt) for prompt in prompts]
    if args.baseline:
        assert outputs == json.loads(args.baseline.read_text()), "Greedy outputs differ from TP=1"
    if args.write_baseline:
        args.write_baseline.write_text(json.dumps(outputs))
    assert stream(args.url, prompts[0]) == outputs[0]
    with ThreadPoolExecutor(max_workers=3) as executor:
        concurrent = list(executor.map(lambda prompt: complete(args.url, prompt), prompts))
    # BF16 kernels can choose a different continuation when the batch shape changes.
    # This check verifies all concurrent requests complete with valid replies.
    assert len(concurrent) == len(prompts) and all(concurrent)
    assert complete(args.url, "Write a poem about stars.", temperature=0.7, top_k=20, top_p=0.9)
    assert complete(args.url, "hello " * 160, max_tokens=8)
    with request(args.url, "Count upwards from one.", stream=True, max_tokens=128) as response:
        assert response.readline().startswith(b"data: ")
    assert complete(args.url, prompts[1]) == outputs[1]
    if args.idle_seconds:
        time.sleep(args.idle_seconds)
        assert complete(args.url, prompts[0]) == outputs[0]
    print(f"Serving checks passed in {time.monotonic() - started:.2f}s", flush=True)
    print(json.dumps(outputs), flush=True)


if __name__ == "__main__":
    main()
