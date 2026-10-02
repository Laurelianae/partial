# Naive performance roadmap

Improve real-model INT4 inference on the two NVIDIA GB10 Sparks: profile first,
optimize the largest target-model cost, then evaluate draft support. The first
milestone is a reproducible real-model profile and one measured improvement;
draft evaluation does not need to wait until every kernel is optimized.

Work through the steps in order. Check off tasks only when their evidence is
recorded in the results log below. Run all CUDA and multinode measurements and
tests on the remote machines through the repository's `just` recipes.

## 1. Establish a production baseline

- [x] Record checkpoint identity, source revision and working-tree changes,
  software versions, hardware, and serving settings.
- [x] Choose and save representative prompts, token counts, generation settings,
  warmup, and repetition counts. Start with one request at TP=2.
- [x] Measure prompt processing and sustained autoregressive decoding separately.
- [x] Record time to first token, per-token decode latency, output tokens/second,
  and peak memory on both ranks.
- [x] Save exact commands, workloads, raw samples, and result artifacts so later
  changes can be compared under the same conditions.

**Complete when:** Repeated real-model runs provide a reproducible baseline.

Completed on 2026-10-02 with the [compact baseline](naive-baseline-findings.md):
one warmup and two measured 32-token requests per prompt at TP=2. This initial
baseline does not establish tail latency or variation between server restarts.
See the [reproduction instructions](naive-baseline.md).

The existing `just naive-measure` harness is useful for diagnostics, but its
repeated fixed-history forward passes are not sustained autoregressive decoding.
Existing fixture timings do not establish production performance. See
[measurement details](naive.md#reproducible-regression-and-measurement).

## 2. Identify the dominant costs

- [x] Profile INT4 weight reconstruction, expert GEMMs and dispatch, TP
  communication, attention/indexing, and CPU overhead.
- [x] Inspect both ranks for communication waits and imbalance; distinguish
  prompt-processing costs from decode costs.
- [x] Rank optimization candidates by their measured contribution and save the
  traces supporting the first choice.

**Initial hypothesis:** Eager INT4 MoE execution is the leading candidate. It
reconstructs selected expert weights into BF16 on every forward, executes experts
sequentially, and communicates per expert. This is a code-based hypothesis, not
a measured bottleneck.

Other candidates include BF16 output projections that compute padded zero rows,
and attention that sorts scores and computes over the full history despite a
sparse or sliding-window mask.

**Complete when:** Saved traces support a specific first optimization.

Completed on 2026-10-02: [profiling findings](naive-profile-findings.md) record
24 validated CPU/CUDA traces and a matching unprofiled control. INT4
reconstruction was the largest GPU category in every trace. First optimize packed
expert execution to avoid full BF16 reconstruction; reassess dispatch and
synchronization next. Profiling overhead and rank imbalance are recorded explicitly.

## 3. Implement and validate one target-model improvement

- [ ] Address the largest measured cost. If MoE dominates, prioritize packed
  INT4 execution that avoids reconstructing full BF16 weight matrices.
- [ ] Assess expert dispatch and communication batching next if measurements
  justify them.
- [ ] Retain the eager path as a correctness reference.
- [ ] Run relevant projection, fixture parity, and serving regressions remotely,
  including TP=2 coverage for distributed execution changes.
- [ ] Repeat the baseline workloads and record speed, memory, and numerical
  results alongside the baseline.

**Constraint:** BF16 projection padding preserves observed numerical behavior.
Changing GEMM shapes can affect rounding, expert routing, and sparse selection.
Do not remove the padding or loosen tolerances merely to obtain speed; validate
any replacement against the existing correctness checks. Consult the
[Naive validation notes](naive.md),
[trained projection findings](naive-projection-findings.md), and
[chunk-stability findings](chunk-stability-findings.md).

**Complete when:** Correctness checks pass and repeated end-to-end measurements
demonstrate an improvement. Record regressions or inconclusive results explicitly.

## 4. Evaluate speculative decoding

- [ ] Identify the draft repository, architecture, tokenizer compatibility,
  integration requirements, and memory footprint on the two Sparks.
- [ ] Measure target verification for blocks of 2, 4, and 8 proposed tokens.
- [ ] Use a focused prototype to measure drafting cost and acceptance on
  representative prompts.
- [ ] Compare total time per accepted token with ordinary decoding, accounting
  for drafting, verification, and acceptance/cache-management overhead.
- [ ] Use the results to choose full draft integration or another target-model
  optimization, and record the decision and evidence.

**Complete when:** Measurements support the next investment. A compatible draft
alone does not establish a speedup: verification may activate more distinct
experts and increase reconstruction and communication costs.

Full draft support would require acceptance/sampling logic, KV-cache rollback,
and scheduler integration. Revisit those details after the evaluation rather
than committing to a draft architecture before inspecting its repository.

## Results log

Steps 1 and 2 have a compact production baseline and measured cost attribution.
Target-model optimization remains pending. Keep measured findings separate from
hypotheses and link to detailed artifacts, including interrupted or unsuccessful
attempts.

| Date | Step | Source revision / working-tree changes | Workload | Commands / artifacts | Findings | Next action |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-10-02 | 1: preliminary attempt | `3726ddf` + saved baseline source snapshot | Initial prompt suite | `.cache/naive-baseline/20261002-180152-c3640927/` | Stopped before token samples to extend retrieval beyond the sparse-selection limit | Use finalized prompts |
| 2026-10-02 | 1: longer attempt | `3726ddf` + saved baseline source snapshot | 127-token assistant, 128 output tokens | `.cache/naive-baseline/20261002-180616-9344a043/` | One measured request: 0.937 decode tokens/s; stopped at user request to shorten suite | Run compact suite |
| 2026-10-02 | 1: compact baseline complete | `3726ddf` + saved baseline source snapshot | TP=2; prompts 127/248/2,172 tokens; 32 output tokens; one warmup + two repetitions | `just naive-baseline`; [findings and artifacts](naive-baseline-findings.md) | Median TTFT 13.520/17.884/24.144 s; decode 0.942/0.936/0.919 tokens/s; rank token agreement and repeatability passed | Step 2: attribute costs with profiling |
| 2026-10-02 | 2: profiling complete | `2b00e20` + saved profiling source snapshot | Same compact TP=2 workload; prefill and decode steps 8–11 on both ranks | `just naive-baseline`; `just naive-profile`; [findings and artifacts](naive-profile-findings.md) | 24 valid traces; identical tokens/SSE; reconstruction largest GPU category in every trace; median GPU shares 61–76% prefill and 60–64% decode | Step 3: avoid full BF16 expert reconstruction with validated packed INT4 execution |
