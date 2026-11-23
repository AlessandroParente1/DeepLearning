#!/bin/sh
#BSUB -q gpuv100
#BSUB -J week3
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 6:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o hpc/logs/week3.out
#BSUB -e hpc/logs/week3.err

# Whole week-3 pipeline on one GPU, from a fresh clone:   bsub < hpc/run_week3.sh
#
# 1. train the small transformer (small_b) and the family head on it     (minutes)
# 2. classifier-guided sampling toward HU, 256 sequences per gamma        (~10-20 min)
# 3. independent family check (nearest training protein)                  (CPU, few min)
# 4. train the conditional model (label dropout) and sample with classifier-free guidance, 256 per w   (week 4)
# 5. fold both controls and both guided sets with ESMFold                (the slow part)
#
# Needs `bash hpc/setup_env.sh` to have been run once (environment + ESMFold weights).
# Checkpoints are not in git, so they are rebuilt here. Seeds are fixed, but GPU kernels are not bit-exact, so
# numbers can differ slightly from a laptop run.
# Each step is skipped if its output already exists, so the job can be resubmitted after a timeout.

source hpc/env.sh
set -eu

SEQ=results/sequences
GUIDED=guided_small_b_HU_n256
CFG=cfg_small_b_HU_n256

mkdir -p checkpoints results/sequences

if [ ! -f checkpoints/small_b.pt ]; then
    uv run python src/train_transformer.py --d-model 256 --heads 4 --blocks 4 --ffn 1024 \
        --dropout 0.3 --weight-decay 0.1 --lr 5e-4 --name small_b --seed 0
fi

if [ ! -f checkpoints/family_head_small_b.pt ]; then
    uv run python src/train_classifier_head.py --backbone checkpoints/small_b.pt \
        --out checkpoints/family_head_small_b.pt --results results/classifier_small_b --seed 0
fi

if [ ! -f $SEQ/$GUIDED.csv ]; then
    uv run python src/generate_guided.py --head checkpoints/family_head_small_b.pt --target HU \
        --gammas 0 0.5 1 2 4 8 --n 256 --batch-size 64 --seed 0 --out $SEQ/$GUIDED.csv
fi

if [ ! -f $SEQ/${GUIDED}_family_check.csv ]; then
    uv run python eval/family_check.py --samples $SEQ/$GUIDED.csv
fi

if [ ! -f checkpoints/small_b_cfg.pt ]; then
    uv run python src/train_transformer.py --d-model 256 --heads 4 --blocks 4 --ffn 1024         --dropout 0.3 --weight-decay 0.1 --lr 5e-4 --conditional --p-drop 0.15 --name small_b_cfg --seed 0
fi

if [ ! -f $SEQ/$CFG.csv ]; then
    uv run python src/generate_cfg.py --model checkpoints/small_b_cfg.pt --judge-head checkpoints/family_head_small_b.pt         --target HU --ws 0 0.5 1 2 4 8 16 --n 256 --batch-size 128 --seed 0 --out $SEQ/$CFG.csv
fi

if [ ! -f $SEQ/${CFG}_family_check.csv ]; then
    uv run python eval/family_check.py --samples $SEQ/$CFG.csv
fi

# 256 real and 256 shuffled proteins, the same size as each guided set
if [ ! -f $SEQ/control_real.csv ]; then
    uv run python eval/make_controls.py --num 256
fi

for name in control_real control_shuffled $GUIDED $CFG; do
    if [ ! -f results/eval/$name/summary.csv ] || ! grep -q mean_plddt results/eval/$name/summary.csv; then
        uv run --group eval python eval/eval.py --samples $SEQ/$name.csv --fold
    fi
done

echo "done. Results: results/eval/*/summary.csv, results/sequences/${GUIDED}_family_check.csv"
