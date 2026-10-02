# Naive compact baseline findings

On 2026-10-02, the real `~/models/Naive-N0.5-Flash-Int4` checkpoint completed a
TP=2 streaming API baseline on the two NVIDIA GB10 Sparks. The compact suite
used one server session, one warmup and two measured requests per prompt, with
32 generated tokens per request. All nine requests completed. All six measured
requests passed token-count, prompt-identity, rank-agreement, and memory-record
checks; repeated measured outputs were identical for each prompt.

Total wall time was about 12 minutes 31 seconds including full checkpoint hashing
and startup. Model startup took 99.0 seconds; the nine requests took 468.5 seconds.
Use this as an initial optimization baseline. Two repetitions do not establish
reliable tail latency, and one session does not measure restart variability.

## Measured results

Medians of two requests; decode latency pools the 62 inter-token intervals per
prompt. TTFT and throughput use client event arrival times on Spark 0 loopback.
Warmups are excluded.

| Prompt | Input tokens | TTFT (s) | Decode latency (s/token) | Decode tokens/s | End-to-end (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| assistant | 127 | 13.520 | 1.059 | 0.942 | 46.419 |
| coding | 248 | 17.884 | 1.066 | 0.936 | 51.022 |
| retrieval | 2172 | 24.144 | 1.086 | 0.919 | 57.879 |

| Prompt | TTFT range (s) | Decode tokens/s range | Server prefill, rank 0 / 1 (s) |
| --- | ---: | ---: | ---: |
| assistant | 13.493–13.547 | 0.940–0.944 | 13.510 / 13.509 |
| coding | 17.871–17.898 | 0.935–0.936 | 17.875 / 17.874 |
| retrieval | 24.081–24.208 | 0.919–0.919 | 24.132 / 24.131 |

Server prefill measures scheduler acceptance through first-token completion,
including batch preparation and sampling. These are separate rank-local clocks;
do not subtract timestamps across ranks. The 2,172-token retrieval prompt exceeds
the model's 2,048-key sparse-selection limit.

| Prompt | Peak allocated per rank (GiB) | Peak reserved per rank (GiB) |
| --- | ---: | ---: |
| assistant | 79.707 | 79.842 |
| coding | 79.715 | 79.842 |
| retrieval | 79.840 | 80.205 |

These PyTorch high-water counters were identical on both ranks. They exclude
some external allocations and must not be added to device memory snapshots.
Phase-separated counters and device free/total snapshots are retained in the raw records.

## Configuration and artifacts

- Source base: `3726ddf`; the synchronized uncommitted baseline implementation is
  identified by the saved source hashes, snapshot, Git status, and `source.diff`.
- BF16 execution of INT4 weights, eager attention, TF32 disabled, no CUDA graphs
  or overlap; naive cache, page size 1, 4,096 cache pages and prefill limit, one request.
- Greedy sampling, `ignore_eos=true`, literal prompt strings without a chat template.
- Both ranks: PyTorch `2.9.1+cu130`, CUDA `13.0`, Transformers `5.17.0`,
  Triton `3.5.1`, FlashInfer `0.7.0`, NCCL `2.27.7`, Safetensors `0.8.0`.
- Rank 0: Python `3.12.13`, Linux `6.17.0-1031-nvidia`; rank 1: Python `3.12.3`,
  Linux `6.17.0-1029-nvidia`. Full driver/hardware records are saved in preflight.

```bash
just naive-baseline '~/models/Naive-N0.5-Flash-Int4'
# Equivalent explicit compact settings:
just naive-baseline '~/models/Naive-N0.5-Flash-Int4' --sessions 1 --warmup 1 --repeat 2 --tokens 32
```

- [Complete raw results](../.cache/naive-baseline/20261002-181913-9c3c3b05/results.json).
- [Rank 0 log](../.cache/naive-baseline/20261002-181913-9c3c3b05/session-0/rank-0.log) and [rank 1 log](../.cache/naive-baseline/20261002-181913-9c3c3b05/session-0/rank-1.log).
- The run directory also contains full checkpoint hashes, exact commands and
  requests, actual input/output IDs, raw SSE events, source snapshot, and warmups.

## Longer-run cross-check and interrupted attempts

The original two-session, 128-output-token suite was stopped at the user’s request
after two assistant warmups and one measured assistant request. The measured
request decoded at 0.937 tokens/s; the compact assistant median was 0.942 tokens/s
(about 0.6% higher). Both compact measured outputs exactly matched the first 32
IDs of that 128-token output. This supports the shorter window for an initial
baseline on this prompt; it does not establish long-generation behavior for every prompt.

- Longer attempt: `.cache/naive-baseline/20261002-180616-9344a043/`.
  Its supervisor status is `failed` with `KeyboardInterrupt` because the full
  requested suite was intentionally stopped. Completed samples remain in rank logs.
- Earlier attempt: `.cache/naive-baseline/20261002-180152-c3640927/`.
  Stopped before token samples to extend prompts, ensuring the retrieval case
  exceeds the sparse-selection limit. Worker cleanup succeeded before restarting.

## Validation and next step

- 85 focused tests passed on Spark 0:
  `just run 0 .venv/bin/python -m pytest tests/misc/test_naive_baseline.py tests/misc/test_naive_runner.py tests/misc/test_multinode.py tests/core/test_autoround.py --no-cov -q`.
- INT4 fixture API baselines passed at TP=1 and TP=2. All three raw streaming
  responses at each TP size matched exactly with telemetry enabled and disabled.
  Fixture artifacts: `20261002-175531-54575e0b`, `20261002-175811-ead78500`,
  `20261002-175918-20f03a63`, and `20261002-180115-0654c8f6` under
  `.cache/naive-baseline/`.
- Normal API smoke passed on Spark 0, covering streaming, concurrency, sampling,
  chunked prefill, and cancellation:
  `just run 0 .venv/bin/python tests/misc/check_local_serving.py --model .cache/naive-int4-fixture`.
- Black, Ruff, and `git diff --check` passed locally.

Roadmap step 1 is complete for the compact baseline. Next, profile the same
workloads to attribute prefill and decode costs. Similar decode rates across
prompts do not by themselves establish a kernel or communication bottleneck.
No inference optimization or numerical-tolerance change was made.
