#!/bin/bash
#SBATCH --job-name=revrag_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --array=0-2        # 3 seeds

module load cuda/12.1
module load python/3.11
source ~/.venvs/revrag/bin/activate
cd $SLURM_SUBMIT_DIR

SEEDS=(42 123 456)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

export LANGFUSE_PUBLIC_KEY="${LANGFUSE_PUBLIC_KEY}"
export LANGFUSE_SECRET_KEY="${LANGFUSE_SECRET_KEY}"

# Primary real benchmark datasets (no synthetic generation):
#   contractnli   — expert-annotated NLI spans in real NDAs (retrieval framing)
#   cuad_amendments — real amendment pairs from CUAD expert annotations
#   multihop_rag  — peer-reviewed multi-hop retrieval benchmark (Li et al., 2024)
#   loco          — long-context retrieval (Chang et al., 2024)
#   legalbench    — character-level retrieval over CUAD/MAUD/PrivacyQA
#
# Supplementary (diagnostic only, not primary claim):
#   synthetic     — run separately via: sbatch slurm/eval_synthetic.sbatch

for MODEL in bm25 bge bgem3 mamba revrag_no_verifier revrag; do
    for DATASET in legalbench contractnli cuad_amendments multihop_rag loco; do
        echo "Running $MODEL on $DATASET (seed=$SEED)"
        python evaluate/run_eval.py \
            --config configs/default.yaml \
            --model $MODEL \
            --dataset $DATASET \
            --seeds $SEED \
            --device cuda
    done
done

echo "Evaluation complete for seed=$SEED"
