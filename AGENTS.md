# Repository Guidelines

## Project Structure & Module Organization

Mini-SGLang is a Python LLM inference server. Source lives in `python/minisgl/`: `server/` provides the API, `scheduler/` coordinates requests, `engine/` runs inference, and `models/`, `layers/`, `attention/`, and `kvcache/` implement model execution. GPU kernels live in `kernel/`, with C++/CUDA sources under `kernel/csrc/` and Triton code under `kernel/triton/`.

Tests are grouped into `tests/core/`, `tests/kernel/`, and `tests/misc/`. Performance scripts live in `benchmark/offline/` and `benchmark/online/`. Consult `docs/structures.md` for architecture and `docs/features.md` for serving options. `assets/` contains the logo; `tools/` contains remote development and diagnostic scripts.

## Build, Test, and Development Commands

Runtime development requires Linux, an NVIDIA GPU, and the CUDA Toolkit; macOS cannot run the CUDA dependencies natively.

- `uv venv --python=3.12` followed by `source .venv/bin/activate`: create and activate an environment.
- `uv pip install -e '.[dev]'`: install the package and development tools.
- `python -m minisgl --model Qwen/Qwen3-0.6B`: launch the API server; append `--shell` for terminal chat.
- `docker build -t minisgl .`: build the CUDA container.
- `pytest`: run collected tests with terminal and HTML coverage reports.
- `pre-commit run --all-files`: run formatting and repository hygiene checks.
- `just doctor 0`: synchronize source and check Python, PyTorch, and CUDA on node 0. Remote recipes require `.env.local` and a preconfigured remote virtual environment.

## Coding Style & Naming Conventions

Use four-space Python indentation, type annotations, `snake_case` functions/modules, and `PascalCase` classes. Target Python 3.10+ and a 100-character line length. Black formats Python; Ruff checks imports and selected lint rules. Pre-commit also invokes clang-format for C++/CUDA; follow surrounding native-code style.

## Testing Guidelines

Use `test_*.py` files and `test_*` functions. Pytest and pytest-cov are configured without a minimum coverage threshold. Many tests require CUDA; some are executable integration or performance scripts, such as `python tests/kernel/test_index.py`. Run relevant tests on supported hardware and compare kernel outputs against reference implementations. Report hardware and commands when validating performance changes.

All testing that requires CUDA or a multinode setup must run on the remote machines. Use the `just` recipes, for example `just run 0 .venv/bin/python -m pytest` or `just run 0 .venv/bin/python tests/kernel/test_index.py`. For the two-node connectivity test, run `just dist-smoke 0` and `just dist-smoke 1` concurrently in separate terminals.

## Commit & Pull Request Guidelines

History uses concise subjects, including plain descriptions and `[Fix]`, `[Feature]`, or `[Minor]` prefixes. Follow these patterns and keep commits focused. In PRs, describe the behavior change, link relevant issues, and report validation commands/results; include GPU and model details for inference changes.

## Configuration & Remote Sync

Keep SSH node settings in ignored `.env.local`; never commit credentials. Preview remote changes with `just sync-dry 0`. Synchronization uses `rsync --delete`, so configure `PARTIAL_REMOTE_DIR` as a dedicated project directory.
