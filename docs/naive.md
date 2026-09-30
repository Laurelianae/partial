# Naive-N0.5-Flash correctness milestone

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
BF16 and FP32 are supported on one GPU. **TP execution promotes weights and activations
to FP32**, with a startup warning. Native BF16 TP did not meet the reference error bounds
on the stress fixture: small rounding differences changed discrete expert and sparse
token selection. This milestone keeps strict FP32 parity for the distributed path.

The cache retains full per-layer histories, including DSA indexer keys, in scheduler
page locations. Memory budgeting includes the separate key/value shapes and FP32
indexer storage. This supports chunked prefill and prefix reuse but does not implement
SWA ring buffers or million-token memory optimization.

```bash
just run 0 .venv/bin/python -m pytest tests/core/test_naive.py tests/misc/test_multinode.py --no-cov -q
just run 0 bash -c 'export PATH="$PWD/.venv/bin:$PATH"; .venv/bin/python tests/misc/check_naive_parity.py --model .cache/naive-fixture --dtype float32 --production-top-k'
just run 0 bash -c 'export PATH="$PWD/.venv/bin:$PATH"; .venv/bin/python tests/misc/check_naive_parity.py --model .cache/naive-fixture --dtype bfloat16'
just run 0 .venv/bin/python tests/misc/check_local_serving.py --model .cache/naive-fixture
```

Run these concurrently in separate terminals for two-node parity:

```bash
just naive-parity 0 .cache/naive-fixture float32
just naive-parity 1 .cache/naive-fixture float32
```

The parity recipes use the existing rendezvous and transport settings. The reference
explicitly selects eager experts and retains FP32 rotary buffers. Comparisons cover
full and chunked prefill, cached decode, SWA and sparse-selection boundaries, unequal
request lengths, prefix reuse, relocated cache slots, and eight greedy tokens. FP32
uses `atol=rtol=2e-5`; BF16 TP=1 uses `atol=rtol=0.025`. Relative L2 error is also
bounded, and greedy token IDs must match. `--production-top-k` additionally exercises
2,049 tokens with top-k 2,048, using the small model.

## Validation — 2026-09-30

Both Sparks used NVIDIA GB10 GPUs, PyTorch `2.9.1+cu130`, CUDA 13.0, and
Transformers `5.17.0`. The generated checkpoint SHA-256 matched on both nodes:
`1bcdb57de566bdf4ad9291ccf231ade5dd1db3fcad113e5a51a21c7bf6b78512`.

| Check | Result |
| --- | --- |
| Configuration, loader, tokenizer, and multinode regressions | 41 tests passed |
| BF16 TP=1 model parity | Maximum logit error 0; greedy IDs matched |
| FP32 TP=1, including production sparse top-k boundary | Maximum logit error `1.22e-6`; greedy IDs matched |
| FP32 TP=2, including promotion from requested BF16 | Maximum logit error `1.19e-6`; both ranks and reference greedy IDs matched |
| Naive API, one and two nodes | Streaming, chat template, concurrency, stochastic sampling, chunked prefill, cancellation passed |
| Small Qwen API regression after dependency upgrade | Passed with CUDA graphs enabled and naive prefix cache |

Test servers were stopped after validation. These are correctness checks; their elapsed
times do not establish sustained throughput. Qwen's existing BF16 behavior can change
continuations when cached prefixes or pending cancellation change batch shapes, so its
smoke check validates recovery after cancellation rather than requiring identical text.

## Limits

Synthetic parity establishes implementation behavior for the tested configurations.
It does not qualify full trained-weight loading, quantization, long-context capacity,
native BF16 TP, speculative decoding, or throughput. Full BF16 weights require roughly
618 GB; FP8 alone is still too large for the two Sparks. Quantized loading and memory
placement remain separate work.
