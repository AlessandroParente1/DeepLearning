#!/usr/bin/env bash
# Score one set of generated sequences, structures included:
#
#     bash hpc/run_eval.sh results/sequences/mock_data.csv          # -> results/eval/mock_data/
#     bash hpc/run_eval.sh results/sequences/mock_data.csv my/dir   # -> my/dir/
#
# The argument is one CSV with a `sequence` column, such as what your sampling code wrote.
# This needs a GPU, because it folds every valid sample with ESMFold. Run it inside an interactive
# GPU session (`sxm2sh`, `voltash`, ...) or let hpc/eval.sh submit it for you. Without a GPU, drop
# the --fold and the figures line below: the sequence metrics need no GPU at all.
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/env.sh

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: bash hpc/run_eval.sh <samples.csv> [output_dir]" >&2
    exit 1
fi
SAMPLES="$1"
if [ ! -f "$SAMPLES" ]; then
    echo "no such file: $SAMPLES" >&2
    exit 1
fi

# The same default eval.py would pick, computed here because the figures need the path too.
if [ "$#" -eq 2 ]; then
    OUT="$2"
else
    NAME="$(basename "$SAMPLES")"
    OUT="results/eval/${NAME%.csv}"
fi

echo "== GPU"
# A plain torch.cuda.get_device_name() on a machine without a GPU raises an AssertionError, which
# is a confusing way to learn that the job did not ask for one.
uv run --group eval python - <<'PY'
import sys

import torch

print(f"torch {torch.__version__}")
if not torch.cuda.is_available():
    sys.exit("No GPU here. Fold on a GPU node - 'bsub < hpc/eval.sh', or an interactive session - "
             "or score the sequences alone with:\n"
             "    uv run python eval/eval.py --samples <samples.csv>")
print(f"  {torch.cuda.get_device_name()}, compute capability "
      f"{'.'.join(str(n) for n in torch.cuda.get_device_capability())}")
PY

echo "== Metrics and ESMFold structures -> $OUT"
uv run --group eval python eval/eval.py --samples "$SAMPLES" --out "$OUT" --fold

echo "== Figures for the first 5 structures"
uv run --group eval python eval/plot_structure.py "$OUT/pdb" --limit 5 --out "$OUT/figures"

echo "Done. $OUT holds eval.csv, summary.csv, pdb/ and figures/"
