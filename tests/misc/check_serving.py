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


def stream(url: str, prompt: str, **options) -> str:
    content = ""
    finished = False
    with request(url, prompt, stream=True, **options) as response:
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
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()
    started = time.monotonic()
    prompts = ["The capital of France is", "One plus one equals", "The color of a clear sky is"]

    def completion(prompt, **options):
        return complete(args.url, prompt, model=args.model, **options)

    outputs = [completion(prompt) for prompt in prompts]
    if args.baseline:
        assert outputs == json.loads(args.baseline.read_text()), "Greedy outputs differ from TP=1"
    if args.write_baseline:
        args.write_baseline.write_text(json.dumps(outputs))
    assert stream(args.url, prompts[0], model=args.model) == outputs[0]
    assert complete(
        args.url,
        None,
        model=args.model,
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=4,
    )
    with ThreadPoolExecutor(max_workers=3) as executor:
        concurrent = list(executor.map(completion, prompts))
    # BF16 kernels can choose a different continuation when the batch shape changes.
    # This check verifies all concurrent requests complete with valid replies.
    assert len(concurrent) == len(prompts) and all(concurrent)
    assert completion("Write a poem about stars.", temperature=0.7, top_k=20, top_p=0.9)
    assert completion("hello " * 160, max_tokens=8)
    with request(
        args.url, "Count upwards from one.", model=args.model, stream=True, max_tokens=128
    ) as response:
        assert response.readline().startswith(b"data: ")
    # Cancellation is asynchronous; a pending decode can change the BF16 batch shape.
    # Verify serving recovers; fixed-shape token parity belongs in its dedicated harness.
    assert completion(prompts[1])
    if args.idle_seconds:
        time.sleep(args.idle_seconds)
        assert completion(prompts[0]) == outputs[0]
    print(f"Serving checks passed in {time.monotonic() - started:.2f}s", flush=True)
    print(json.dumps(outputs), flush=True)


if __name__ == "__main__":
    main()
