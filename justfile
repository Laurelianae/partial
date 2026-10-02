set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
set positional-arguments
set dotenv-path := ".env.local"
set dotenv-required

# List available commands.
default:
    @just --list

# Create the dedicated remote project directory.
# Does not install Python or create a venv.
init node="0":
    @bash tools/remote.sh "$1" init

# Synchronize source to one node.
sync node="0":
    @bash tools/remote.sh "$1" sync

# Preview changes and deletions without applying them.
sync-dry node="0":
    @bash tools/remote.sh "$1" sync --dry-run

# Synchronize both nodes.
sync-all: (sync "0") (sync "1")

# Open Bash with the venv and its prompt decoration enabled.
shell node="0": (sync node)
    @bash tools/remote.sh "$1" shell

# Locate the Mini-SGLang entry point in the remote environment.
check-remote node="0": (sync node)
    @bash tools/remote.sh "$1" exec .venv/bin/python -c \
        'import importlib.util; spec = importlib.util.find_spec("minisgl.__main__"); assert spec is not None and spec.origin is not None, "Cannot locate minisgl.__main__"; print("Mini-SGLang entry point:", spec.origin)'

check-all: (check-remote "0") (check-remote "1")

# Check Python, PyTorch, and a small CUDA calculation on one node.
doctor node="0": (sync node)
    @bash tools/remote.sh "$1" exec .venv/bin/python tools/doctor.py

doctor-all: (doctor "0") (doctor "1")

# Execute a command inside the remote project directory.
run node +args: (sync node)
    @bash tools/remote.sh "$1" exec "${@:2}"

# Launch ONE side of the distributed connectivity test.
dist-smoke node: (sync node)
    @bash tools/remote.sh "$1" exec .venv/bin/python -m torch.distributed.run \
        --nnodes=2 --nproc-per-node=1 --node-rank="$1" \
        --master-addr="${PARTIAL_MASTER_ADDR:?Set PARTIAL_MASTER_ADDR in .env.local}" \
        --master-port="${PARTIAL_MASTER_PORT:-29500}" \
        tools/distributed_smoke.py

# Serve one model across both Sparks; Ctrl-C stops both nodes.
serve-two model *args: sync-all
    @python3 tools/serve.py "$1" "${@:2}"

# Start one side in a separate terminal for debugging.
serve-node node model *args: (sync node)
    @python3 tools/serve.py --node "$1" "$2" "${@:3}"

# Run one side of Naive reference parity; invoke both nodes concurrently.
naive-parity node model dtype="float32" *args: (sync node)
    @python3 tools/naive_parity_remote.py "$1" "$2" "$3" "${@:4}"

# Supervised fixture regression; --quick runs core and BF16 TP=1 only.
naive-regression fixture *args: sync-all
    @python3 tools/naive_runner.py regression "$1" "${@:2}"

# Native-only prefill/decode measurement, with --tp-size 1 or 2.
naive-measure fixture *args: sync-all
    @python3 tools/naive_runner.py measure "$1" "${@:2}"

# Diagnose chunk-dependent logits/greedy tokens across both Sparks, without serving changes.
naive-chunk-stability model *args: sync-all
    @python3 tools/chunk_stability_remote.py "$1" "${@:2}"

# Compact real-model streaming API baseline (TP=2, one session).
naive-baseline model="~/models/Naive-N0.5-Flash-Int4" *args: sync-all
    @python3 tools/naive_baseline.py "$1" "${@:2}"

# Capture prefill and decode steps 8-11 on both ranks of the compact API suite.
naive-profile model="~/models/Naive-N0.5-Flash-Int4" *args: sync-all
    @python3 tools/naive_baseline.py "$1" --profile "${@:2}"
