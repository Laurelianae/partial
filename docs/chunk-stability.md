# Focused Naive chunk-size investigation

Run the existing INT4 checkpoint across both Sparks:

```sh
just naive-chunk-stability '~/models/Naive-N0.5-Flash-Int4' --timeout 14400
```

The supervisor uses the existing stdin lifetime wrapper and timeout cleanup. It
checks available host memory and active GPU compute processes before loading,
hashes checkpoint files on both nodes, and verifies remote Python source against
a local snapshot. It refuses an occupied GPU without terminating other workloads.
Results, exact commands, source snapshots, checkpoint identities, software versions,
and rank logs are saved below `.cache/chunk-stability/<timestamp>-<id>/`.
`findings.md` summarizes `results.json`; logs preserve completed cases if a later
case fails. The timeout applies to each supervised job.
Workers report elapsed time and token progress every 16 prefill chunks, plus each
history's completion time. Small chunks can take much longer on this eager,
cross-node implementation because expert reconstruction and collectives repeat
for every forward pass.

The trained-model workload contains three fixed token histories: factual text
(129 tokens), Python code (257), and retrieval of a code placed at the beginning
of a long context (2,049). Their exact IDs and decoded text are recorded. These
are diagnostic histories, not a quality benchmark or traffic sample. The worker
uses a local tokenizer without downloading checkpoint code.

For each history, compare full prefill with chunks of 13, 64, and 128. Batch size
is one, page size is one, and absolute token position maps to the same physical
cache location in every run. Before each new history, poison every KV and index
cache with NaNs, then rebuild the entire visible history. Nonfinite logits fail
the run. Repeat full prefill to distinguish chunk effects from run variability.

Each comparison records the final prompt logits plus eight teacher-forced decode
outputs on the baseline's generated history. Separately, each chunk size generates
eight greedy tokens independently. A token can first diverge at index zero (the
first generated token). Decode-output index zero in `forced_history` denotes final
prompt logits; indices one through eight follow the supplied baseline tokens.
Maximum absolute error, relative L2 error, top-two margins, winners, and first
greedy divergence are recorded without applying a parity tolerance. Every forward's
logits must hash identically across TP ranks, and their complete case summaries
must agree.

The worker traces the shortest history that changes independent greedy output or
any measured next-token winner (including the prediction after eight decode steps);
if none changes, it traces the largest measured logit difference. For ties in
history length it selects the larger logit difference. Temporary wrappers record
normalization, attention projections and attention core, residuals, MLP outputs,
expert selections (ordered and sorted), and sparse-attention masks. CPU traces
align by absolute token position, including decode positions. Complete coverage
and absence of duplicate positions are asserted. Trace output logits must match
the uninstrumented run exactly. Wrappers and buffers are released in `finally`.

Trace differences are ordered by forward operation, then earliest position within
that operation. This is a layer/operation ordering, not wall-clock order across
prefill chunks. An ordered expert selection change can be only a permutation;
`selection_set` distinguishes membership changes. Sparse masks use absolute key
positions and pad future keys with false.

Precision replays use identical captured baseline inputs, preserving row order
and chunk boundaries. They cover a bounded set of divergent linear projections,
the first divergent attention core (with baseline sparse selections frozen), and
one expert gate/up projection in the first divergent expert block. TP output
projections retain the runtime's padded output width. Only the selected expert's
needed matrices are reconstructed, using the runtime's INT4 decoder and BF16
weight values before FP32 promotion. Replays disable TF32 and restore its previous
settings afterward. These tests isolate component arithmetic; they do not provide
an unquantized or end-to-end FP32 trained-model reference. Decode-only differences
may use the last prompt row to choose an expert probe and remain inconclusive
about the specific decode operation.

Validate with existing fixtures and invariant tests:

```sh
just run 0 .venv/bin/python -m pytest tests/misc/test_chunk_stability.py tests/misc/test_naive_runner.py --no-cov -q
just naive-chunk-stability .cache/naive-int4-fixture --fixture --timeout 1800
just naive-chunk-stability .cache/naive-fixture --fixture --timeout 1800
```

Fixture mode adds the established 132-token INT4 chunk-dependent history and
requires that its chunk-13 prompt logit difference is detected. It uses explicit
fixture IDs instead of a tokenizer. Invariant tests check deliberate perturbation,
absolute-position alignment, missing/duplicate coverage, poisoned independent
histories, TP disagreement, cleanup, and precision-setting restoration. All
CUDA/distributed execution takes place remotely. Serving code, routing policy,
precision, and parity tolerances are unchanged.

For a focused repeat, add `--case code --chunks 128` (or another recorded history
name and chunk size). Defaults retain the complete three-history, three-chunk matrix.
