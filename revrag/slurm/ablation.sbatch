#!/bin/bash
#SBATCH --job-name=revrag_ablation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=logs/ablation_%j.out
#SBATCH --error=logs/ablation_%j.err

module load cuda/12.1
module load python/3.11
source ~/.venvs/revrag/bin/activate
cd $SLURM_SUBMIT_DIR

# Hyperparameter ablation: lambda × NLI threshold grid on synthetic benchmark
python experiments/ablation_hyperparams.py \
    --config configs/default.yaml \
    --dataset-path data/synthetic/cross_page_bench.jsonl \
    --device cuda \
    --max-items 500 \
    --output-dir outputs/ablation

echo "Ablation complete."
