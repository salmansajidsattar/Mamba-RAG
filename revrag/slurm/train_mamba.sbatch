#!/bin/bash
#SBATCH --job-name=revrag_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@university.edu

# ── Environment setup ──────────────────────────────────────────────────────
module load cuda/12.1
module load python/3.11

source ~/.venvs/revrag/bin/activate
cd $SLURM_SUBMIT_DIR

# Verify GPU
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# ── Langfuse keys (set in your ~/.bashrc or load from secret file) ─────────
export LANGFUSE_PUBLIC_KEY="${LANGFUSE_PUBLIC_KEY}"
export LANGFUSE_SECRET_KEY="${LANGFUSE_SECRET_KEY}"

# ── Training ───────────────────────────────────────────────────────────────
python train/train_mamba.py \
    --config configs/default.yaml \
    --output-dir outputs/checkpoints/mamba2_finetuned \
    --epochs 3 \
    --batch-size 32 \
    --lr 2e-5 \
    --seed 42

echo "Training complete. Checkpoint saved to outputs/checkpoints/mamba2_finetuned"
