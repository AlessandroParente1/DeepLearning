#!/usr/bin/env bash
# One-time setup on the cluster login node:  bash hpc/setup_env.sh
# Installs uv (if missing), the evaluation environment and downloads the ESMFold weights.
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/env.sh

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Installing the environment (uv sync --group eval) ..."
uv sync --group eval

# LSF opens the job's output files before the script runs, so a missing directory fails the job
# rather than being created by it. The .out/.err files themselves are gitignored.
mkdir -p hpc/logs

# Only download the files: loading the 8 GB model needs more memory than a login node allows.
# The model itself is loaded in the GPU job, which asks for enough memory.
echo "Downloading ESMFold weights to $HF_HOME (about 8 GB, only the first time) ..."
uv run --group eval python - <<'PY'
from huggingface_hub import snapshot_download

print("ESMFold weights are cached in", snapshot_download("facebook/esmfold_v1"))
PY

echo
echo "Setup done. Score a set of samples on a GPU with either of:"
echo "    bsub < hpc/eval.sh                                       # every set, as a batch job"
echo "    bash hpc/run_eval.sh results/sequences/mock_data.csv     # one set, in a GPU session"
