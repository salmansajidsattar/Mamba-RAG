#!/bin/bash
#SBATCH --job-name=revrag_efficiency
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/efficiency_%j.out
#SBATCH --error=logs/efficiency_%j.err

module load cuda/12.1
module load python/3.11
source ~/.venvs/revrag/bin/activate
cd $SLURM_SUBMIT_DIR

python experiments/run_efficiency.py \
    --config configs/default.yaml \
    --device cuda

python plotting/plot_scaling.py \
    --results outputs/efficiency/efficiency_results.json \
    --output-dir outputs/figures

echo "Efficiency sweep and plots complete."
