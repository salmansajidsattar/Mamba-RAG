#!/bin/bash
#SBATCH --job-name=revrag_geneval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=logs/geneval_%j.out
#SBATCH --error=logs/geneval_%j.err

module load cuda/12.1
module load python/3.11
source ~/.venvs/revrag/bin/activate
cd $SLURM_SUBMIT_DIR

# End-to-end generation evaluation: compare all retrieval models
# using GPT-4o-mini as the reader (requires OPENAI_API_KEY)
export OPENAI_API_KEY="${OPENAI_API_KEY}"

python evaluate/run_generation_eval.py \
    --config configs/default.yaml \
    --dataset-path data/synthetic/cross_page_bench.jsonl \
    --backend openai \
    --model-name gpt-4o-mini \
    --n-queries 200 \
    --top-k 5 \
    --compare-all \
    --device cuda \
    --output-dir outputs/generation_eval

echo "Generation evaluation complete."
