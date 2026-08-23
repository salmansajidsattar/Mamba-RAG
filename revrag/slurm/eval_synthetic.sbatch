#!/bin/bash
#SBATCH --job-name=revrag_synthetic
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/synthetic_%j.out
#SBATCH --error=logs/synthetic_%j.err
#SBATCH --array=0-2

module load cuda/12.1
module load python/3.11
source ~/.venvs/revrag/bin/activate
cd $SLURM_SUBMIT_DIR

SEEDS=(42 123 456)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# NOTE: synthetic benchmark is SUPPLEMENTARY / diagnostic analysis only.
# Primary evaluation uses real benchmarks (contractnli, cuad_amendments,
# multihop_rag, loco, legalbench) via eval_all.sbatch.
# Run this only for error analysis and verifier ablation figures.

for MODEL in bm25 bge bgem3 mamba revrag_no_verifier revrag; do
    echo "Synthetic eval: $MODEL (seed=$SEED)"
    python evaluate/run_eval.py \
        --config configs/default.yaml \
        --model $MODEL \
        --dataset synthetic \
        --seeds $SEED \
        --device cuda
done

echo "Synthetic evaluation complete for seed=$SEED"
