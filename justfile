set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

remote := env_var_or_default("PARTIAL_REMOTE", env_var("SPARK_ADDRESS"))
remote_user := env_var_or_default("PARTIAL_USER", env_var("USER"))
ssh_target := remote_user + "@" + remote

remote_dir := env_var_or_default(
    "PARTIAL_REMOTE_DIR",
    "~/Development/Projects/partial",
)

sync:
    ./tools/sync.sh

shell: sync
    ssh -t {{ssh_target}} \
        'cd {{remote_dir}} && exec bash --rcfile <(printf "%s\n" "source ~/.bashrc" "source .venv/bin/activate") -i'

check-remote: sync
    ssh {{ssh_target}} \
        'cd {{remote_dir}} && .venv/bin/python -c "import minisgl; print(\"Mini-SGLang import OK\")"'

doctor: sync
    ssh {{ssh_target}} 'cd {{remote_dir}} && \
        source .venv/bin/activate && \
        python -c "import torch; \
        print(\"torch:\", torch.__version__); \
        print(\"cuda:\", torch.version.cuda); \
        print(\"available:\", torch.cuda.is_available()); \
        print(\"device:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); \
        print(\"capability:\", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"'

smoke: sync
    ssh -t {{ssh_target}} 'cd {{remote_dir}} && \
        source .venv/bin/activate && \
        python -m minisgl \
            --model Qwen/Qwen3-0.6B \
            --shell'

serve-smoke: sync
    ssh -t {{ssh_target}} 'cd {{remote_dir}} && \
        source .venv/bin/activate && \
        python -m minisgl \
            --model Qwen/Qwen3-0.6B \
            --host 0.0.0.0 \
            --port 1919'