# Sampled trained INT4 projection validation

On 2026-10-02, all **36 checks passed** on the two NVIDIA GB10 Sparks using
`~/models/Naive-N0.5-Flash-Int4`. Every sampled rank-local reconstructed BF16
weight matched the independent decoder exactly, and every projection comparison
had maximum absolute error **0**. These checks identified no calculation change
needed for the sampled expert projections.

## Coverage and method

The fixed sample was layers **1, 24, 47**, experts **0, 255**, and **gate, up,
down** projections, with both output shards at TP size 2: 3 × 2 × 3 × 2 = 36.
Layer numbers and expert indices are zero-based. Layer 0 has a dense MLP;
layer 1 is the first expert layer. Each projection has 8,388,608 full-weight
elements; each rank's 4,194,304 reconstructed elements were compared exactly.

The existing `tools/check_autoround_projection.py` was run unchanged. It compares
the runtime decoder against the independently implemented
`tools/autoround_reference.py`, using the actual checkpoint's packed weights,
zero points, and scales. Each check uses 13 seeded random BF16 input vectors
(seed 42) and compares `output_shard_linear` against the corresponding output
slice of a full reference projection. Weight tolerances remain zero; projection
tolerances remain `atol=0.025, rtol=0.025`, although all observed errors were zero.

Both machines' complete checkpoint SHA-256 manifests matched, and their source
hashes matched the saved local snapshot. Preflight checks found no active GPU
compute processes and sufficient available memory. Each rank ran on its own
Spark concurrently with the other rank. The existing `naive_runner.run_job` and
`serve_worker.py` mechanisms supplied timeout and SSH-lifetime cleanup. Both
postflight GPU process lists were empty.

## Results and timing

| Sample | Checks | Exact reconstructed weights | Maximum projection error |
|---|---:|---|---:|
| Layer 1, experts 0 and 255 | 12 | All matched | 0 |
| Layer 24, experts 0 and 255 | 12 | All matched | 0 |
| Layer 47, experts 0 and 255 | 12 | All matched | 0 |

Total supervised wall time was **297.46 seconds** (4 minutes 57 seconds).
Checkpoint/source preflight took 219.59 seconds. The 18 paired projection jobs
took approximately 77 seconds including process startup and checkpoint-header
inspection. The helper's narrower component timer was 0.626–0.876 seconds per
check. Maximum reference-check PyTorch allocation was 82,231,296 bytes
(78.42 MiB); this is not full-model inference memory.

Software: PyTorch `2.9.1+cu130`, CUDA `13.0`, Transformers `5.17.0`; Python
`3.12.13` on rank 0 and `3.12.3` on rank 1.

## Reproduction and artifacts

Machine-readable results, exact commands, checkpoint and source hashes, source
snapshot, launcher, and per-rank logs are retained locally under:

`.cache/projection-qualification/20261002-172140/`

The aggregate result is `results.json`. The saved launcher was invoked as:

```bash
just --command python3 .cache/projection-qualification/run.py
```

An individual check can be reproduced with:

```bash
just run 0 .venv/bin/python tools/check_autoround_projection.py /home/laura/models/Naive-N0.5-Flash-Int4 --layer 1 --expert 0 --projection gate_proj --tp-size 2 --rank 0 --tokens 13
```

Repeat over the sample above, using node 1 for rank 1. The saved launcher runs
both ranks concurrently and includes the full identity and resource checks.

## Interpretation

This closes the previously pending sampled trained-projection checks. It supports
the correctness of the tested INT4 reconstruction and output-shard projections.
Combined with the existing synthetic reference comparisons and trained chunk
investigation, it provides no evidence requiring an inference arithmetic change.

The check evaluates each rank's shard independently; it does not exercise live
distributed collectives or the runtime's fused gate/up operation. It does not
compare end-to-end trained-model logits against an independent implementation,
cover every expert, measure answer quality, or establish chunk-invariant output.
The measured chunk-dependent token change in the
[chunk investigation](chunk-stability-findings.md) remains a numerical-stability
finding. Production code and arithmetic were unchanged by this validation.
