#!/bin/sh
#BSUB -q gpuv100
#BSUB -J fold_w34
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 6:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o hpc/logs/fold_w34.out
#BSUB -e hpc/logs/fold_w34.err

# Fold the guided samples and the controls with ESMFold (no training, the samples were made on the laptop):
#     bsub < hpc/fold_week3_4.sh                 # 64 sequences per setting: fast first pass
#     FOLD_SIZE=256 bsub < hpc/fold_week3_4.sh   # all 256 per setting: about 4x longer
#
# Needs `bash hpc/setup_env.sh` once (environment + ESMFold weights), and these files in results/sequences/:
#     control_real.csv control_shuffled.csv guided_small_b_HU_n256.csv cfg_small_b_HU_n256.csv
# Each set is skipped if it already has a pLDDT summary, so the job can be resubmitted after a timeout.
# At the end the small result files are packed into fold_results_<FOLD_SIZE>.tar.gz (no .pdb files) to copy back.

source hpc/env.sh
set -eu

FOLD_SIZE="${FOLD_SIZE:-64}"
SEQ=results/sequences
SETS="control_real control_shuffled guided_small_b_HU_n256 cfg_small_b_HU_n256"

for base in $SETS; do
    if [ ! -f "$SEQ/$base.csv" ]; then
        echo "MISSING $SEQ/$base.csv, skipping"
        continue
    fi
    name="${base}_f${FOLD_SIZE}"
    # keep the first FOLD_SIZE sequences of every setting (a CSV without a setting column is one setting)
    uv run python -c "
import sys, pandas as pd
df = pd.read_csv('$SEQ/$base.csv')
df = df.groupby('setting', sort=False).head($FOLD_SIZE) if 'setting' in df else df.head($FOLD_SIZE)
df.to_csv('$SEQ/$name.csv', index=False)
print('$name:', len(df), 'sequences')
"
    if [ -f "results/eval/$name/summary.csv" ] && grep -q mean_plddt "results/eval/$name/summary.csv"; then
        echo "$name already folded, skipping"
    else
        uv run --group eval python eval/eval.py --samples "$SEQ/$name.csv" --fold
    fi
done

tar -czf "fold_results_${FOLD_SIZE}.tar.gz" $(ls results/eval/*_f${FOLD_SIZE}/summary.csv results/eval/*_f${FOLD_SIZE}/eval.csv 2>/dev/null)
echo "done: fold_results_${FOLD_SIZE}.tar.gz"
