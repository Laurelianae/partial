# Naive INT4 cost profiling findings

Roadmap step 2 is complete. On 2026-10-02, all 24 real-model traces captured CUDA
kernels: three prompts × two measured requests × two phases × two GB10 ranks.
**The first optimization should eliminate full-matrix INT4 reconstruction from
expert execution.** Reconstruction was the largest attributed GPU category in
every trace, including traces with substantial collective waits.

Use packed INT4 expert execution that reconstructs tiles as needed instead of
materializing whole BF16 matrices. Preserve the current eager path as the
reference and retain its FP16 dequantization arithmetic, BF16 rounding, projection
padding behavior, and validation tolerances. A fused implementation still needs
projection, fixture, TP=2, and end-to-end validation in roadmap step 3; this
profiling work makes no inference optimization or speedup claim.

## Workload and control

The checkpoint was `~/models/Naive-N0.5-Flash-Int4`, served at TP=2 across the two
NVIDIA GB10 Sparks. Both suites used the saved 127/248/2,172-token prompts, greedy
32-token outputs, one warmup and two measured requests per prompt in one session.
Settings match the [compact baseline](naive-baseline.md): BF16 execution, eager
attention, no CUDA graphs or overlap, and one request at a time.

Control and profile snapshots matched across all 166 source/workload files, based
on commit `2b00e207c8ef2ed0810d19741d306a29cdbc99ce` plus the saved profiling diff.
Both ranks' checkpoint hashes and source identities passed preflight. Runtime
versions remained PyTorch 2.9.1+cu130, CUDA 13.0 and NCCL 2.27.7; complete hardware,
software, commands and identities are retained in the reports. Final edits after
measurement were formatting of CLI error messages, an attribution test extension,
and documentation; inference and capture code were unchanged.

| Prompt | Control TTFT (s) | Control decode tokens/s | Control end-to-end (s) | Profiled end-to-end (s) |
| --- | ---: | ---: | ---: | ---: |
| assistant | 13.410 | 0.945 | 46.201 | 86.631 |
| coding | 17.878 | 0.942 | 50.802 | 105.383 |
| retrieval | 23.326 | 0.949 | 55.994 | 129.408 |

These are medians of two requests. All nine output sequences matched between
control and profiling, on both ranks, and every raw SSE payload sequence matched.
The control also matched all original baseline output IDs. Its decode rates were
approximately 0.4%, 0.6%, and 3.3% above the original baseline; this is run variation,
not an implemented optimization. PyTorch peak allocation remained approximately
79.707/79.715/79.840 GiB per rank. These counters exclude host profiler storage.

## Measured attribution

Capture covers prefill and decode steps 8–11. The tables below take medians over
the four rank/request observations per prompt and phase; the two ranks are paired
observations, not four independent repetitions. Decode durations are divided by
four to give seconds per token. GPU category busy time merges overlapping events
within that category. CPU and GPU times overlap and must not be added together.

| Prompt | Prefill wall (s) | GPU busy (s) | INT4 reconstruction (s) | Reconstruction / GPU busy |
| --- | ---: | ---: | ---: | ---: |
| assistant | 14.656 | 12.756 | 9.692 | 75.5% |
| coding | 19.684 | 17.360 | 12.748 | 74.5% |
| retrieval | 27.143 | 23.212 | 14.263 | 61.4% |

| Prompt | Decode wall (s/token) | GPU busy (s/token) | INT4 reconstruction (s/token) | Reconstruction / GPU busy | GPU idle / wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| assistant | 1.564 | 0.844 | 0.540 | 64.0% | 46.1% |
| coding | 1.589 | 0.852 | 0.541 | 63.6% | 46.4% |
| retrieval | 1.640 | 0.867 | 0.522 | 60.2% | 46.9% |

Fractions are medians of per-trace ratios, so ratios of table medians can differ.
Across individual traces, reconstruction occupied 44.3–77.3% of GPU busy time in
prefill and 57.8–65.2% in decode. It remained the largest category in all 24.

Other GPU costs, in seconds (prefill / per decoded token):

| Category | assistant | coding | retrieval |
| --- | ---: | ---: | ---: |
| Output projections, including padding | 1.083 / 0.101 | 1.438 / 0.101 | 1.863 / 0.098 |
| Weight concatenation | 0.885 / 0.050 | 1.172 / 0.050 | 1.300 / 0.048 |
| TP gathers, including waits | 0.659 / 0.043 | 1.026 / 0.046 | 2.405 / 0.057 |
| Expert gate/up GEMMs | 0.206 / 0.010 | 0.276 / 0.010 | 0.470 / 0.011 |
| Attention compute | 0.004 / 0.002 | 0.015 / 0.003 | 1.866 / 0.029 |
| Sparse selection | 0.0002 / 0.0002 | 0.0005 / 0.0002 | 0.013 / 0.0002 |

Output projections include expert down projections and attention/dense projections;
they are not a measurement of padding alone. Attention projections, score
construction, cache work and sparse selection have separate categories in the
machine-readable results. The longer retrieval prefill makes attention more
visible, but it remains smaller than reconstruction.

## CPU dispatch, communication, and uncertainty

Expert dispatch is the largest exclusive CPU category: about 1.10–1.12 s per
profiled decode step. This is host elapsed time, including waits for queued GPU
work, not pure Python compute. The representative assistant decode trace on each
rank contains 48,128 `aten::nonzero` calls and 48,132 `cudaStreamSynchronize` calls
across four steps: 12,032 expert scans per token, consistent with 47 MoE layers
scanning all 256 experts. Only 4,512 reconstruction calls occur across those four
steps, corresponding to three projections for eight selected experts per layer.

On rank 0, those four steps include 3.388 s of `aten::nonzero` self time and
2.105 s of CUDA stream synchronization API time. The latter overlaps the former;
both can wait for earlier reconstruction or collectives. Dispatch batching and
reducing synchronization are the second optimization candidate, especially for
decode. Their benefit must be measured after reducing reconstruction cost.

Both ranks show the same reconstruction dominance. Decode step durations are
closely matched; median TP-gather GPU time per token is 0.050/0.037 s on ranks
0/1 for assistant, 0.053/0.043 for coding, and 0.071/0.054 for retrieval. These
include peer waits and should not be interpreted as network-transfer times.

Prefill has visible imbalance and outliers. The second coding request contains a
2.822 s embedding all-reduce on rank 0. The second retrieval request contains a
7.257 s embedding all-reduce on rank 1, near the beginning of its capture, and
rank-local prefill wall durations of 28.475/40.967 s. Profiler initialization and
host/export overhead can skew progress between ranks; these instrumented waits
do not establish a production network bottleneck. Raw traces and per-layer detail
are retained. Matching uses request, phase, layer, and collective ordering, never
subtraction of absolute timestamps across hosts.

Instrumentation is substantial: captured decode steps take 1.56–1.64 s/token,
versus 1.054–1.056 s/token for the same control window, an increase of roughly
48–56%. Trace stopping, serialization and compression also delay subsequent
uncaptured tokens, explaining the much larger end-to-end times above. The
reconstruction ranking is consistent across prompts, repetitions and ranks;
precise production CPU shares and speedup predictions cannot be inferred from
these instrumented timings. Use the unprofiled control to validate step 3.

The investment order supported by these traces is:

1. Packed INT4 expert execution to avoid full BF16 reconstruction and its
   intermediate tensor traffic, while preserving validated arithmetic.
2. Dispatch and synchronization reduction, followed by communication batching
   where measured collective costs justify it.
3. Reassess output projection/padding and long-context attention after re-profiling.
   Keep existing projection shapes and tolerances until replacements pass the
   [trained projection](naive-projection-findings.md) and
   [chunk-stability](chunk-stability-findings.md) checks.

## Reproduction, artifacts and validation

```bash
just naive-baseline '~/models/Naive-N0.5-Flash-Int4'
just naive-profile '~/models/Naive-N0.5-Flash-Int4'
```

See [profiling instructions](naive-profile.md) for capture semantics and fixtures.

- [Unprofiled control](../.cache/naive-baseline/20261002-185807-7924b051/results.json).
- [Profile results](../.cache/naive-profile/20261002-191031-f130c128/results.json).
- [Trace manifest and 24 trace hashes](../.cache/naive-profile/20261002-191031-f130c128/session-0/traces/manifest.json),
  with 2,948,896,625 bytes of compressed traces retained in that directory.
- [Aggregated analysis](../.cache/naive-profile/20261002-191031-f130c128/analysis.json)
  and its `summarize.py` script in the same run directory.
- [Representative per-step/layer details and collective outliers](../.cache/naive-profile/20261002-191031-f130c128/trace-details.json)
  and the `inspect_details.py` extraction script.

The first TP=1 fixture profile, `20261002-184956-dee9edc7`, was intentionally
rejected by validation because the first capture had no CUDA events. A minimal
probe reproduced this cold-start behavior on the installed PyTorch/CUPTI build;
subsequent captures contained CUDA kernels. A small profiler initialization pass
before server readiness resolved it without changing packages. Every measured
trace still independently requires GPU events.

Successful profiled fixtures were `20261002-185454-f2e85b65` (TP=1) and
`20261002-185623-7d898c8e` (TP=2), with 16 output tokens. Their controls were
`20261002-185603-37e28848` and `20261002-185649-7d20bfd8`, respectively. All
output IDs and raw SSE payloads matched with profiling enabled and disabled.

The focused suite passed 90 tests on Spark 0:

```bash
just run 0 .venv/bin/python -m pytest tests/misc/test_naive_profile.py tests/misc/test_naive_baseline.py tests/misc/test_naive_runner.py tests/misc/test_multinode.py tests/core/test_autoround.py --no-cov -q
just run 0 .venv/bin/python tests/misc/check_local_serving.py --model .cache/naive-int4-fixture
```

The normal serving smoke passed streaming, concurrency, sampling, chunked prefill,
and cancellation. The final five profiler tests also passed separately after
extending CUDA-driver correlation coverage. Black, Ruff and `git diff --check`
passed locally. GPU worker cleanup was checked on both Sparks.
