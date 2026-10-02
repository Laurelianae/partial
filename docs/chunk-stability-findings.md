# Naive INT4 chunk-size findings

The trained `~/models/Naive-N0.5-Flash-Int4` checkpoint was tested on two NVIDIA
GB10 Sparks with TP=2, batch size one, BF16 execution, and identical cache
locations. PyTorch was `2.9.1+cu130`, CUDA `13.0`, and Transformers `5.17.0`.
Each rank loaded 79.12 GiB of weights. Production inference code and precision
settings were not changed.

## Observed behavior

All three fixed histories produced the same first eight independent greedy tokens
for full prefill and chunks of 13, 64, and 128. Every repeated full-prefill baseline
was bit-identical. Both TP ranks agreed exactly on measured logits and summaries.

However, the code history changed its **ninth-token prediction** at chunk 128.
After the shared continuation ` n * factorial(n - 1)\n\n`, full prefill predicted
`def` (token 750), while chunk 128 predicted `#` (token 2). Their top-two margins
were 0.25 and 0.125, respectively. This comparison follows eight decoded inputs;
it is just beyond the separate eight-token generation window. A focused repeat
reproduced the original baseline and chunked logit digests exactly. There were no
top-two ties anywhere in the full trained measurement matrix.

Maximum absolute logit error across final prompt logits and eight forced-history
decode outputs:

| History | Tokens | Chunk 13 | Chunk 64 | Chunk 128 |
|---|---:|---:|---:|---:|
| Factual | 129 | 2.0 | 2.6875 | 2.1875 |
| Code | 257 | 4.03125 | 3.125 | 3.28125 |
| Retrieval | 2,049 | 2.64453125 | 2.4730224609375 | 2.515625 |

These are measured differences, not relaxed parity tolerances. This small,
fixed sample does not estimate instability frequency in real traffic.

## Tracing and precision evidence

The code/chunk-128 trace first differs in forward-operation order at
`layer.0.self_attn.q_proj`, absolute prompt position 256. Maximum error there is
0.0001220703125 on rank 0 and 0.000244140625 on rank 1. Key/value projections also
differ. The first expert-membership change is in `layer.1.mlp.gate` at position 51
on both ranks. Operation order takes precedence over token position: a later
operation can differ at an earlier token. No sparse-attention selection changed
in this traced short history, whose length is below the 2,048-key selection limit.

Identical-input replays isolate arithmetic from propagated input and selection
changes. Replaying the first differing query projection reduced the rank-0/rank-1
shape errors from 0.0001220703125/0.000244140625 to approximately
3.58e-6/5.25e-6. Replay priority follows aligned operation order, so late-position
differences in early operations are included. For the key projection, BF16 full-versus-chunk error was 0.0078125 and
0.015625 on ranks 0 and 1; FP32 reduced these to approximately 9.54e-7 and 1.19e-6.
For the sampled fused expert gate/up projection, error fell from 0.0625 to
approximately 7.63e-6 and 6.68e-6. Attention replays held the baseline sparse mask
fixed. TF32 was disabled during all precision replays.

Expert replays used the runtime INT4 decoder and retained its BF16 reconstructed
weight values before promotion to FP32. They do not recover unquantized weights.
The component evidence supports arithmetic sensitivity. It does not establish
end-to-end FP32 parity, prove which accumulated changes caused the final token
flip, or exclude every trained-model implementation error. A full FP32 trained
reference was not loaded.

The original full-matrix run also traced code/chunk-13, the largest logit-error
case: its first numerical difference was the first layer's key projection at
position 0, followed by expert-membership changes in layer 1 at position 82.
The final selector also considers the prediction after eight decode steps, so
it selects code/chunk-128 for the observed decision change.

## Reproducible records

- Full trained matrix: `.cache/chunk-stability/20261002-145859-bd37c02f/results.json`.
- Focused decision-change repeat: `.cache/chunk-stability/20261002-162032-8b6f7074/results.json`.
- Final aligned-priority precision replay: `.cache/chunk-stability/20261002-165331-c692cdad/results.json`.
- INT4 fixture validation: `.cache/chunk-stability/20261002-170334-6cbb49e5/results.json`.
- Unquantized fixture validation: `.cache/chunk-stability/20261002-170406-77f2f98c/results.json`.

Each run directory contains commands, per-rank logs, full checkpoint hashes,
software/hardware metadata, a source snapshot, exact token IDs, and machine-readable
metrics. All 19 diagnostic/supervisor tests passed on Spark 0; lint and Black formatting
checks passed locally. Both final fixture runs completed with the same source
identity as the final trained replay. The known INT4 fixture's chunk-13 prompt error reproduced exactly at
0.09619140625. The harness checks poisoned independent cache histories, complete
absolute-position trace coverage, exact traced/uninstrumented logit agreement,
rank agreement, intentional perturbation detection, and worker cleanup.

See [the diagnostic workflow](chunk-stability.md) for commands and measurement
semantics. The initial long retrieval chunk-13 comparison took about 30.5 minutes
for both histories; chunk 64 took about 9.3 minutes. These are diagnostic job
wall times, not serving-throughput benchmarks. The final harness logs progress
and elapsed time every 16 prefill chunks.
