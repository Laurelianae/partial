# Naive-N0.5-Flash correctness milestone

For the step-by-step optimization checklist and results log, see the
[Naive performance roadmap](naive-performance-roadmap.md).

The server recognizes `NaiveN05FlashForCausalLM` and executes its SWA/DSA attention,
partial RoPE, attention sinks, FP8-rounded indexer, dense first layer, and sigmoid
MoE router using eager PyTorch operations. Transformers is pinned to `5.17.0`.
The native configuration is adapted from the upstream implementation at revision
`0235b3b5ff27422b1f57cdc2acddfaf643e08356`, retaining its Apache-2.0 notices.

## Generate the development checkpoint

Run on each Spark, with identical software versions:

```bash
just run 0 .venv/bin/python tools/make_naive_fixture.py --output .cache/naive-fixture
just run 1 .venv/bin/python tools/make_naive_fixture.py --output .cache/naive-fixture
```

The generator downloads only pinned upstream code, configuration, and tokenizer
files. It creates a 170.2 MiB BF16 checkpoint with FP32 routers, seed 42, three
layers (DSA/SWA/DSA), dense/MoE/MoE feed-forward layers, hidden width 256, 16 experts,
top-eight routing, and sparse top-k 16. Production head dimensions (192/128), rotary
dimensions (64), indexer dimensions, and SWA window (128) are preserved. The vocabulary
and chat template are unchanged. Outputs are random and are not useful language generation.

The output directory must be empty. `.cache/` is excluded from remote synchronization,
so fixtures survive `just sync`. Compare checkpoint hashes before testing TP.
The manifest records the upstream revision and seed. The fixture's Python files are
used only by the validation tools; normal serving uses the server's native model class.

`--dummy-weight` is rejected for Naive: it allocates full model shapes and does not
provide a common checkpoint for reference and TP comparisons.

## Serve and validate

```bash
just run 0 .venv/bin/python tools/serve_worker.py --model .cache/naive-fixture \
    --num-pages 512 --max-prefill-length 64
just serve-two .cache/naive-fixture --num-pages 512 --max-prefill-length 64
```

Naive selects the eager backend and disables CUDA graphs and overlap scheduling.
An explicit incompatible attention/MoE backend or quantized checkpoint is rejected.
BF16 and FP32 are supported on one GPU and at TP=2. BF16 requests retain BF16 weights,
activations, and KV caches; routers and indexer score accumulation retain their existing
FP32 precision. The parity harness rejects any override of the requested execution dtype.

BF16 output projections shard output rows and gather input activations, preserving the
full dot-product reduction dimension. Temporary zero rows around the local weight shard
also preserve the upstream GEMM output dimension: changing either dimension can change
cuBLAS reduction order and move a result across a BF16 rounding boundary. Such small
differences can change expert routing or sparse token selection in later layers.
Weights remain sharded, and the upstream reference and CUDA precision defaults are unchanged.
FP32 keeps its existing input-sharded output projections and FP32 reductions.

This BF16 correctness path adds activation gathers, temporary buffers of a full output
projection's size, and GEMM work on zero rows. It does not provide TP compute scaling for
these projections. Throughput and full-model memory capacity remain unqualified.

The cache retains full per-layer histories, including DSA indexer keys, in scheduler
page locations. Memory budgeting includes the separate key/value shapes and FP32
indexer storage. This supports chunked prefill and prefix reuse but does not implement
SWA ring buffers or million-token memory optimization.

```bash
just run 0 .venv/bin/python -m pytest tests/core/test_naive.py tests/misc/test_multinode.py --no-cov -q
just run 0 bash -c 'export PATH="$PWD/.venv/bin:$PATH"; .venv/bin/python tests/misc/check_naive_parity.py --model .cache/naive-fixture --dtype float32 --production-top-k'
just run 0 bash -c 'export PATH="$PWD/.venv/bin:$PATH"; .venv/bin/python tests/misc/check_naive_parity.py --model .cache/naive-fixture --dtype bfloat16 --production-top-k'
just run 0 .venv/bin/python tests/misc/check_local_serving.py --model .cache/naive-fixture
```

Run these concurrently in separate terminals for two-node parity:

```bash
just naive-parity 0 .cache/naive-fixture bfloat16 --production-top-k
just naive-parity 1 .cache/naive-fixture bfloat16 --production-top-k
```

Repeat with `float32` for the FP32 regression. The parity recipes accept additional harness
arguments and use the existing rendezvous and transport settings. The reference
explicitly selects eager experts and retains FP32 rotary buffers. Comparisons cover
full and chunked prefill, cached decode, SWA and sparse-selection boundaries, unequal
request lengths, prefix reuse, relocated cache slots, and eight greedy tokens. FP32
uses `atol=rtol=2e-5`; BF16 TP=1 and TP=2 use `atol=rtol=0.025`. Relative L2 error is
bounded by `2e-5` for FP32 and `0.05` for BF16, and greedy token IDs must match.
`--production-top-k` additionally exercises
2,049 tokens with top-k 2,048, using the small model.

## Validation — 2026-09-30

Both Sparks used NVIDIA GB10 GPUs, PyTorch `2.9.1+cu130`, CUDA 13.0, and
Transformers `5.17.0`. The generated checkpoint SHA-256 matched on both nodes:
`1bcdb57de566bdf4ad9291ccf231ade5dd1db3fcad113e5a51a21c7bf6b78512`.

| Check | Result |
| --- | --- |
| Core tests, BF16 projection rounding, and multinode regressions | 67 tests passed |
| BF16 TP=1 and TP=2, standard fixture cases | Maximum logit error 0; both ranks and reference greedy IDs matched |
| BF16 TP=1 and TP=2, including production sparse top-k boundary | Maximum logit error `0.0078125`, relative L2 `0.001443`; greedy IDs matched |
| FP32 TP=1, including production sparse top-k boundary | Maximum logit error `1.22e-6`; greedy IDs matched |
| FP32 TP=2, including production sparse top-k boundary | Maximum logit error `1.19e-6`; both ranks and reference greedy IDs matched |
| Naive API, one and two nodes | Streaming, chat template, concurrency, stochastic sampling, chunked prefill, cancellation passed |
| Small Qwen API regression after dependency upgrade | Passed with CUDA graphs enabled and naive prefix cache |

Before the BF16 fix, bypassing promotion at TP=2 failed chunked prefill with maximum
logit error `0.072021484375`, exceeding the existing BF16 bounds. The fix passes those
same cases without changing the tolerances, reference implementation, or precision flags.

Test servers were stopped after validation. These are correctness checks; their elapsed
times do not establish sustained throughput. Qwen's existing BF16 behavior can change
continuations when cached prefixes or pending cancellation change batch shapes, so its
smoke check validates recovery after cancellation rather than requiring identical text.

## Limits

Synthetic parity establishes implementation behavior for the tested configurations.
It does not qualify full trained-weight loading, quantization, long-context capacity,
BF16 TP beyond the tested fixture and hardware, speculative decoding, or throughput.
Full BF16 weights require roughly
618 GB; FP8 alone is still too large for the two Sparks. AutoRound INT4 routed-expert
loading is implemented as described below; full trained-weight qualification is pending.

## Reproducible regression and measurement

For sustained trained-model API measurements, use the [production baseline](naive-baseline.md).
The diagnostic harness below retains its fixed-history semantics.

With the same fixture installed on both Sparks:

```bash
just naive-regression .cache/naive-fixture
just naive-regression .cache/naive-fixture --quick
just naive-measure .cache/naive-fixture --tp-size 1 --prompt-length 128 --batch-size 2 --warmup 2 --repeat 5
just naive-measure .cache/naive-fixture --tp-size 2 --dtype bfloat16 --prompt-length 128 --batch-size 2 --warmup 2 --repeat 5
```

The full regression checks fixture file hashes across nodes before execution, runs
core and supervisor tests, FP32/BF16 TP=1/TP=2 parity including production sparse
top-k and page size four, Naive API checks at TP=1/TP=2, and a Qwen3-0.6B API smoke
check. `--quick` runs core tests and BF16 TP=1 parity only. Both TP workers launch
automatically; failures, a per-job `--timeout` (default 1,800 seconds), and Ctrl-C
close the SSH lifetime pipes and stop managed workers. The fixture must already
exist; these commands do not generate or download it.

Results are written locally to `.cache/naive-results/<timestamp>/results.json`
and per-job `rank-*.log` files. `--output` changes the parent directory. Parity
results include the local source commit, checkpoint/config/tokenizer hashes,
GPU and software versions, error metrics, and generated token IDs. The local
commit is forwarded explicitly because remote synchronization excludes `.git`;
logs and results describe the synchronized working tree, which may include
uncommitted changes.

Measurement launches a separate native-only process and never loads the upstream
reference. It measures complete eager forward workloads including batch construction
and attention metadata: full prefill, then one cached decode step with an already
populated prompt history. Each repetition uses the same synthetic token history;
it is not an autoregressive throughput benchmark. CUDA is synchronized around
each sample. TP results retain each rank's samples and report the slowest rank
for each repetition, separately for prefill and decode.

Memory fields use bytes: `resident_pytorch_bytes` is allocated/reserved memory
after warmup, `peak_pytorch_bytes` is allocated/reserved high-water memory during
repetitions, and `device_bytes` is CUDA device free/total memory after that phase.
These counters describe different scopes and must not be added together. PyTorch
counters do not include all driver, communication, or external allocator memory.
The cache allocation scales with batch size and prompt length. Fixture timings
are diagnostics only, not full-model baselines, capacity estimates, or performance
targets; no timing thresholds are enforced.

Production projection tests use full attention output dimensions and individual
expert projection dimensions, with synthetic weights allocated one projection at
a time. Storage layout assertions cover the existing eager loader only. Future
backends should test logical weight reconstruction and execution rather than
inherit its packing and sharding assumptions. Optimized INT4 kernels, other packing formats, and full-model capacity qualification
remain separate work.

The supervised full regression was validated on both GB10 Sparks on 2026-10-01
with PyTorch `2.9.1+cu130`, CUDA `13.0`, and Transformers `5.17.0`: expanded core
and supervisor tests, all four parity combinations, Naive API checks at TP=1/TP=2,
and the Qwen smoke check passed. Native-only measurement was validated at TP=1
and TP=2 with prompt length 32, batch size two, one warmup, and two repetitions;
JSON phase separation, memory scopes, rank completeness, and slowest-rank
aggregation were checked. These runs establish harness behavior, not a performance
baseline.


## AutoRound INT4 eager path

Existing model-path loading automatically recognizes `auto-round` / `auto_round`,
`packing_format: auto_round:auto_gptq` (with legacy `format`/`backend` aliases), four bits, group size 128, symmetric quantization,
and no activation order mapping. Only individual routed expert projections are
packed; dense/attention tensors must be BF16 and routers FP32. `g_idx`, activation
quantization, other formats (including Exl3), and FP32 quantized execution are rejected.

```bash
just run 0 .venv/bin/python tools/inspect_autoround.py /path/to/checkpoint --tp-size 2
just run 0 .venv/bin/python tools/make_naive_fixture.py --output .cache/naive-int4-fixture --int4
just run 1 .venv/bin/python tools/make_naive_fixture.py --output .cache/naive-int4-fixture --int4
just naive-regression .cache/naive-int4-fixture
```

Inspection reads only safetensors headers and config/index JSON, validates the full
expected tensor schema, and reports checkpoint payload bytes, each rank's resident
weight bytes, a conservative temporary expert workspace reserve, and their sum.
An index `total_size` of zero is treated as an exporter placeholder; payload sizes
are computed from validated tensor headers. Nonzero index size mismatches are rejected.
The peak weight estimate excludes KV cache, activations, driver/communication memory,
and allocator overhead. The engine subtracts the reserve before allocating KV cache,
and checks explicit page overrides against available memory.

The loader preallocates final per-rank packed expert arrays and fills them from CPU
safetensors slices; components may live in different shards. It retains INT32 words
and FP16 scales. Gate/up and down projections shard output channels, preserving the
existing BF16 gathers and output projection behavior. The eager backend reconstructs
only the expert currently selected, uses FP16 scale multiplication before BF16
conversion, and releases reconstructed matrices after their GEMMs. There is no
persistent decoded expert cache or converted checkpoint. Packing/inspection live
separately from the eager execution operator so future formats/backends can adapt
without changing routing or accumulation.

The synthetic fixture uses an independent encoder and reference decoder in
`tools/autoround_reference.py`. Its supervised regression automatically selects
BF16 for quantized fixtures, retaining TP=1/TP=2 prefill, chunked prefill, cached
decode, multiple requests, API generation, and the unquantized Qwen smoke check.

After the same trained checkpoint is staged on both Sparks, compare sampled real
projections (choose multiple layers/experts and all three projection types):

```bash
just run 0 .venv/bin/python tools/check_autoround_projection.py /path/to/checkpoint --layer 1 --expert 0 --projection gate_proj --rank 0
just run 1 .venv/bin/python tools/check_autoround_projection.py /path/to/checkpoint --layer 1 --expert 0 --projection down_proj --rank 1
just serve-two /path/to/checkpoint --dtype bfloat16 --max-running-requests 1 --max-seq-len-override 4096
```

Record
both ranks' loading seconds, resident tensor bytes, peak loading allocation (logged
by the engine), peak generation memory, and deterministic generated token IDs.
The projection helper retains one full reference projection solely for validation;
its memory result describes that check, not runtime memory. Sampled trained
projection checks, TP=2 startup, and repeated deterministic generation have now
passed; see the results below. These checks do not establish independent end-to-end
trained-model parity. Model transfer and storage cleanup are outside this workflow.


INT4 fixture validation on both NVIDIA GB10 Sparks (2026-10-01, PyTorch
`2.9.1+cu130`, CUDA `13.0`, Transformers `5.17.0`) passed 155 core/supervisor tests,
BF16 TP=1/TP=2 reference parity, both API serving checks, and the Qwen smoke check.
Maximum reference logit error was `0.0078125`, maximum relative L2 `0.002001`,
and greedy tokens matched at TP=1 and on both TP=2 ranks. The independent reference
and native implementation both showed a `0.09619140625` full-vs-chunked prefill
logit difference for one synthetic history. Quantized placement checks therefore
hold chunk shapes fixed; every prefill/decode workload still compares against the
independent reference with the original tolerances. Existing unquantized
cross-chunk checks remain in place. Fixture parity does not promise chunk-size
invariant BF16 generation.

The fixture header report at TP=2 was 169,128,592 checkpoint bytes, 85,705,352
resident weight bytes per rank, and a 1,310,720-byte expert workspace reserve.
The sampled fixture gate projection qualification command reconstructed weights
exactly and produced zero projection error. These are synthetic implementation
checks; the subsequent trained sample is recorded below.

The final unquantized regression also passed all 155 tests, FP32/BF16 parity at
TP=1/TP=2, both Naive API checks, and Qwen serving smoke on the same Sparks.
Local validation records are in
`.cache/naive-int4-results/20261001-181616-080880db/results.json` and
`.cache/naive-int4-unquantized-results/20261001-181917-b02927bd/results.json`.

With the real checkpoint staged at `~/models/Naive-N0.5-Flash-Int4` on both Sparks,
header validation passed on both hosts after recognizing AutoRound's canonical
`packing_format` field and its zero-valued index size placeholder. Validated payload
size was 169,538,945,408 bytes, with 84,949,268,416 resident weight bytes per rank
and a 167,772,160-byte expert workspace reserve. Regression tests cover both metadata
cases; incorrect nonzero index sizes remain rejected.

The user subsequently reported successful TP=2 startup and streamed generation from
the real checkpoint through `/generate` and `/v1/chat/completions`. The chat smoke
answered "Paris." and terminated with `finish_reason: stop`. These establish basic
trained-model API serving. Subsequent trained runs recorded loading and memory
diagnostics and bit-identical repeated full-prefill baselines in the
[chunk investigation](chunk-stability-findings.md).

On 2026-10-02, all 36 sampled trained expert projection checks passed: layers
1, 24, and 47; experts 0 and 255; gate/up/down projections; both TP=2 output shards.
Reconstructed weights matched the independent decoder exactly, and every measured
projection error was zero. The supervised run took 297.46 seconds including full
checkpoint hashing. See [the projection validation report](naive-projection-findings.md)
for commands, artifacts, timings, and coverage limits. Independent end-to-end
trained-model reference parity and broader correctness qualification remain pending.

For a focused trained-model comparison of full prefill with chunks of 13, 64,
and 128, see [the chunk-size investigation](chunk-stability.md). Its standalone
TP=2 harness records fixed token histories, baseline repeats, forced-history logits,
independent greedy generation, aligned operation traces, and component FP32 replays
without changing serving behavior or parity tolerances.
