# Sourced by the other scripts in hpc/. Not run directly. Edit the settings below if your
# cluster differs.

# Optional: modules to load, e.g. EVAL_MODULES="python3/3.12.4". Python is installed by uv,
# and the torch wheels ship their own CUDA libraries, so usually none are needed.
EVAL_MODULES="${EVAL_MODULES:-}"

# LSF settings (DTU HPC). Check `bqueues` / the DTU HPC GPU docs for the current names.
# torch on Linux is the CUDA 12.6 build (see pyproject.toml), which runs on every Nvidia queue:
# gpuv100, gpua100, gpua10, gpua40, gpul40s.
QUEUE="${QUEUE:-gpuv100}"
WALLTIME="${WALLTIME:-02:00}"
CORES="${CORES:-4}"
MEM_PER_CORE="${MEM_PER_CORE:-8GB}"

for module_name in $EVAL_MODULES; do
    module load "$module_name"
done

export PATH="$HOME/.local/bin:$PATH"  # where the uv installer puts uv

# Keep the big caches out of the small home quota: the ESMFold weights (~8 GB) and uv's
# package cache (several GB of torch/CUDA wheels). They go to /work3/$USER if it exists,
# otherwise to .cache/ inside this repository. Set HF_HOME / UV_CACHE_DIR to override.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"  # -P: real path, not via symlinks
# The scripts that source this file run under `set -u`, where a bare $USER would abort them in any
# environment that does not export it - a batch job started from a bare shell, for one.
WHO="${USER:-$(id -un)}"
if [ -d "/work3/$WHO" ]; then
    CACHE_ROOT="/work3/$WHO"
else
    CACHE_ROOT="$REPO_DIR/.cache"
fi
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR"

# uv unpacks every wheel into its cache and then puts the files in .venv. pyproject.toml asks for
# copies, because a checkout in a OneDrive folder cannot take hardlinks; here they can, and the
# CUDA libraries are large enough that the copy doubles the environment - about 9 GB of the same
# bytes twice. Hardlinks need one filesystem for both, so this only applies when the cache and the
# repository are on the same one; otherwise uv falls back to copying without saying so, and the
# warning below is the only hint you would get.
if [ "$(uname -s)" != "Linux" ]; then
    :   # a laptop checkout, where pyproject.toml asks for copies for a reason
elif [ "$(df -P "$UV_CACHE_DIR" | awk 'NR==2 {print $1}')" \
     = "$(df -P "$REPO_DIR"    | awk 'NR==2 {print $1}')" ]; then
    export UV_LINK_MODE="${UV_LINK_MODE:-hardlink}"
else
    echo "note: $UV_CACHE_DIR and $REPO_DIR are on different filesystems, so .venv will be a" >&2
    echo "      full copy of the cache (~9 GB extra). Clone under $CACHE_ROOT to avoid it, or" >&2
    echo "      run 'uv cache clean' once the environment is installed." >&2
fi
