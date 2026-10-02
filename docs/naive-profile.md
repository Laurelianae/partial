# Naive cost profiling

Run a control and a profiled compact suite on the same source revision:

```bash
just sync-dry 0
just sync-dry 1
just naive-baseline '~/models/Naive-N0.5-Flash-Int4'
just naive-profile '~/models/Naive-N0.5-Flash-Int4'
```

`naive-profile` invokes the existing baseline harness with `--profile`. It retains
TP=2, the three saved prompts, 32 generated tokens, one warmup and two measured
requests per prompt. All GPU work runs on the remote Sparks. Profiled throughput
includes instrumentation and trace-export overhead; use the unprofiled control
for performance comparisons.

Each scheduler captures CPU and CUDA activities for prefill (step 0) and decode
steps 8–11. The first forward after prefill is decode step 1. Shape, stack, and
memory profiling are disabled; baseline memory telemetry remains enabled.
Ranges identify the request step, layer, INT4 reconstruction, dispatch, GEMMs,
weight concatenation, TP gathers, attention, sparse selection, and scheduling.
Warmup requests are not captured. The profiler runs a tiny initialization pass
before readiness because the installed torch 2.9.1/cu130 build can omit GPU
activities from its first capture. No synchronization is added inside measured
regions; starting/stopping the profiler itself can synchronize.

Traces are exported and compressed after each captured window. At least 13
output tokens are required, so export finishes before the final response and
worker cleanup. Initialization and export can disturb timing, including the next
rank's collective waits. GPU durations and CPU timings must be interpreted with
that overhead in mind.

## Artifacts and interpretation

`.cache/naive-profile/<run-id>/` contains the same source/checkpoint identities,
raw requests, samples, memory records, and worker logs as the baseline. Each
session also contains a `traces/manifest.json` with request/rank/phase identities,
SHA-256 hashes, local trace paths, and attribution summaries. The gzip-compressed
Chrome traces can be decompressed for a trace viewer. Copies remain remotely in
the matching `.cache/naive-profile/` directory.

Validation rejects missing/duplicate captures, wrong request or step identities,
incomplete streams, token disagreement between ranks, and traces without CUDA
kernels. Partial logs and already collected artifacts survive failure.

CPU category totals are exclusive across nested CPU operators and annotations;
CPU operator self times are also retained. GPU work is associated with the
innermost named range through external IDs and launch correlations. GPU category
busy times merge overlapping intervals within each category. Different categories
can overlap, so their totals need not sum to overall device busy time. Kernel
summed durations are reported separately. Idle time is captured-step wall time
minus the union of GPU activity inside those steps. CUDA API durations overlap
CPU operator time and must not be added to it. Collective kernel duration can
include waiting for a peer and is not a pure network-transfer measurement.

Compare ranks by request, phase, layer and collective order. Absolute timestamps
from different hosts are separate clock domains. Report uncertainty and keep
unattributed work explicit instead of treating it as zero.

## Fixture validation

Run these sequentially, repeating with `--tp-size 2`:

```bash
just naive-baseline .cache/naive-int4-fixture --fixture --tp-size 1 --tokens 16 --timeout 600
just naive-profile .cache/naive-int4-fixture --fixture --tp-size 1 --tokens 16 --timeout 600
```

Compare all output token IDs and raw SSE payloads between the two runs. GPU and
multinode tests must use remote `just` recipes, including the focused profiler,
baseline, runner, multinode and AutoRound tests.
