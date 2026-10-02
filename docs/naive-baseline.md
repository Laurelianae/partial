# Naive production baseline

See the [first compact baseline results](naive-baseline-findings.md).

Run the trained INT4 checkpoint on both Sparks with one request at a time:

```bash
just sync-dry 0
just sync-dry 1
just naive-baseline '~/models/Naive-N0.5-Flash-Int4'
```

This starts one server session and runs the saved assistant, coding, and
document-retrieval prompts, with one warmup and two measured requests per prompt.
Generation is greedy with `ignore_eos=true` and 32 output tokens. This compact
suite is an initial optimization baseline; it does not establish tail latency
or variability between server restarts.
The client runs on Spark 0 against its loopback streaming chat-completions endpoint.
Prompts are sent as literal `prompt` strings, without a chat template. Actual input
IDs and lengths are saved, rather than inferred from text or requested lengths.

Settings are TP=2, BF16 execution of INT4 weights, eager attention, no overlap or
CUDA graphs, naive cache (no cross-request prefix reuse), page size 1, 4,096 pages,
4,096 prefill tokens, and one running request. The suite fits a single prefill
batch. The worker rejects workloads exceeding the model or cache capacity.
These settings describe a controlled warm single-request baseline, not maximum
concurrency or capacity. Model loading and readiness time are reported separately.

`--warmup`, `--repeat`, `--tokens`, `--sessions`, `--request-timeout` (seconds,
default 1,800), `--timeout` (seconds per supervised job, default 28,800), and
`--output` are configurable and saved in the results. For the longer suite, use
`--sessions 2 --warmup 2 --repeat 5 --tokens 128`. Count changes produce a
different workload; comparisons should use the same settings and saved prompts.

## Timing and memory semantics

The API emits one event for every generated token, even if that token produces
no visible text. The client timestamps these events and excludes the final
finish event and `[DONE]` from token counts. It retains the complete SSE payloads.

- TTFT measures request submission through arrival of the first token event.
  First visible text has a separate timestamp.
- Decode latency is the interval between successive token events. Sustained
  decode throughput is `(N - 1) / (last_token_time - first_token_time)`.
- End-to-end latency includes receiving the terminal stream marker; end-to-end
  throughput is `N / end_to_end_seconds`.
- Each scheduler separately records request acceptance through first-token
  completion, including scheduling, batch preparation, forward and sampling.
  This is server-side prompt processing, not a kernel-only measurement.
- Scheduler token timestamps use the existing token-copy completion boundary.
  No extra CUDA synchronization or TP barrier is inserted. Client timings and
  each rank's monotonic timestamps belong to separate clock domains.

`MINISGL_BASELINE_TELEMETRY=1` enables the scheduler records. It is off by default
and requires Naive with one running request. It records input and output IDs,
first-token duration, token completion times, and per-rank memory. Records are
buffered until request completion. Memory queries and recording incur a small
instrumentation cost that is included in the measured run.

PyTorch allocated/reserved peaks reset at request start and after the first token,
separating prefill and decode. Resident counters are sampled at request start;
device free/total memory is sampled at phase boundaries. Reserved memory can
retain warmup allocations. These scopes must not be added together; PyTorch
counters omit some driver, communication, and external allocations. Device free
memory is a snapshot, not a measured device-wide high-water mark.

## Artifacts and acceptance

`.cache/naive-baseline/<timestamp>-<id>/` contains `results.json`, per-job rank logs,
`source.diff`, Git status, and a source snapshot including the workload file.
Results contain exact commands, checkpoint SHA-256 hashes, hardware/driver and
package versions, effective runtime settings, warmups, raw measured samples,
per-rank records, and median/p95/min/max summaries. P95 uses the nearest rank;
two repetitions do not establish a reliable p95. Decode summaries also
retain all individual intervals.

Preflight checks source and checkpoint agreement across nodes and refuses to
start over active GPU processes. Existing processes are left untouched. Errors,
Ctrl-C, and deadlines stop managed workers; logs retain partial request data.
Incomplete requests or missing/duplicate/mismatched rank records fail validation.
Repeated token differences are reported explicitly, not silently averaged away.

Step 1 uses repeated complete trained-model requests and a recorded interpretation
of their variation. The compact suite has two repetitions per prompt in one
server session; longer sustained runs and independent restarts remain optional
follow-ups. Fixture runs validate the harness only. Profiling and model
optimizations remain subsequent roadmap steps.

## Fixture and regression validation

```bash
just run 0 .venv/bin/python -m pytest tests/misc/test_naive_baseline.py tests/misc/test_naive_runner.py tests/misc/test_multinode.py tests/core/test_autoround.py --no-cov -q
just naive-baseline .cache/naive-int4-fixture --fixture --sessions 1 --warmup 1 --repeat 2 --tokens 8 --timeout 300
just naive-baseline .cache/naive-int4-fixture --fixture --tp-size 1 --sessions 1 --warmup 1 --repeat 2 --tokens 8 --timeout 300
```

The fixture-only `--without-telemetry` option runs the same API workload through
the normal serving path to compare raw streaming responses with instrumentation.
