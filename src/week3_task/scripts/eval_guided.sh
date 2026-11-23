#!/bin/sh
#BSUB -q gpuv100
#BSUB -J eval_guided
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o hpc/logs/eval_guided.out
#BSUB -e hpc/logs/eval_guided.err

# Fold and score the guided sample sets and the two controls:   bsub < hpc/eval_guided.sh
# Needs `bash hpc/setup_env.sh` to have been run once. Each CSV has a `setting` column (e.g. HU_gamma2), so
# eval.py groups it and summary.csv has one row per gamma. Missing CSVs are skipped.

source hpc/env.sh

for name in control_real control_shuffled guided_small_b_HU; do
    csv=results/sequences/$name.csv
    if [ -f "$csv" ]; then
        uv run --group eval python eval/eval.py --samples "$csv" --fold
    else
        echo "skipping $name: $csv not found"
    fi
done
