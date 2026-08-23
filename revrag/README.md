# RevRAG: Revision-Aware Retrieval-Augmented Generation

**Mamba-2 bi-encoder + cross-page consistency verification for long multi-page documents.**

Paper: *RevRAG: Linear-Time Long-Context Retrieval with Intra-Document Cross-Page Consistency Verification* (submission pending)

---

## Setup

```bash
# 1. Create virtualenv (Python 3.11 recommended)
python -m venv ~/.venvs/revrag
source ~/.venvs/revrag/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install mamba-ssm (requires CUDA 12.x + compatible GPU)
pip install mamba-ssm causal-conv1d

# 4. Set Langfuse keys (optional — set enabled: false in config to skip)
export LANGFUSE_PUBLIC_KEY="your_key"
export LANGFUSE_SECRET_KEY="your_key"
```

---

## Quick start — smoke test (CPU, no GPU needed)

```bash
cd revrag/
python tests/smoke_test.py
```

Expected: all 5 tests pass in < 60 seconds.

---

## Running tests

```bash
cd revrag/
pytest tests/ -v
```

---

## Data preparation

### 1. TREC DL / MS MARCO
```bash
# Download MS MARCO document collection
wget https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz
gunzip msmarco-docs.tsv.gz
mv msmarco-docs.tsv data/raw/msmarco/collection.tsv

# Download TREC DL queries and qrels
# See https://microsoft.github.io/msmarco/TREC-Deep-Learning.html
```

### 2. LegalBench-RAG
```bash
git clone https://github.com/zeroentropy-ai/legalbenchrag data/raw/legalbench_rag
```

### 3. Generate synthetic cross-page benchmark
```bash
python scripts/generate_synthetic.py \
    --config configs/default.yaml \
    --output data/synthetic/cross_page_bench.jsonl
```

---

## Training the Mamba-2 retriever

**Local (single GPU):**
```bash
python train/train_mamba.py \
    --config configs/default.yaml \
    --output-dir outputs/checkpoints/mamba2_finetuned
```

**SLURM cluster:**
```bash
mkdir -p logs
sbatch slurm/train_mamba.sbatch
```

---

## Evaluation

**Single model, single dataset:**
```bash
python evaluate/run_eval.py \
    --config configs/default.yaml \
    --model revrag \
    --dataset legalbench \
    --seeds 42 123 456 \
    --device cuda
```

**All models, all datasets (SLURM array job):**
```bash
sbatch slurm/eval_all.sbatch
```

Results saved to `outputs/<model>/<dataset>/summary.json`.

---

## Efficiency sweep

```bash
# Local
python experiments/run_efficiency.py --config configs/default.yaml

# SLURM
sbatch slurm/efficiency.sbatch
```

---

## Plotting

```bash
# Scaling curves (Figure 1)
python plotting/plot_scaling.py \
    --results outputs/efficiency/efficiency_results.json

# Main results bar chart (Figure 2)
python plotting/plot_results.py --results-dir outputs
```

Figures saved to `outputs/figures/` as `.pdf`.

---

## Repository structure

```
revrag/
├── configs/           YAML configuration files
├── data/              Data loaders (TREC DL, LegalBench-RAG, synthetic)
├── models/            Retriever implementations
│   ├── bm25_retriever.py        BM25 baseline
│   ├── bge_retriever.py         BGE chunked dense baseline
│   ├── mamba_retriever.py       Mamba-2 bi-encoder (no chunking)
│   └── revrag.py                Full RevRAG pipeline
├── modules/
│   └── consistency_verifier.py  ← Core contribution
├── metrics/
│   └── eval_metrics.py          Recall@k, NDCG, MRR, ECS, CRA, RR
├── train/             Mamba-2 fine-tuning script
├── evaluate/          Main evaluation runner
├── experiments/       Efficiency sweep
├── plotting/          Figure generation scripts
├── tracing/           Langfuse integration
├── tests/             pytest unit tests + smoke test
└── slurm/             SLURM job scripts
```

---

## Key config options

| Parameter | Default | Description |
|---|---|---|
| `model.mamba2.checkpoint` | `state-spaces/mamba2-130m` | Pretrained Mamba-2 checkpoint |
| `model.mamba2.max_length` | `8192` | Max tokens (no chunking) |
| `model.consistency_verifier.checkpoint` | `cross-encoder/nli-deberta-v3-small` | NLI model for pairwise scoring |
| `model.consistency_verifier.nli_contradiction_threshold` | `0.5` | Contradiction confidence threshold |
| `model.consistency_verifier.page_order_weight` | `0.3` | λ weight for document-order rule |
| `retrieval.top_k` | `20` | Candidates before reranking |
| `retrieval.final_k` | `5` | Passages returned to generator |

---

## Reproducibility

All experiments use 3 seeds (42, 123, 456). Results reported as mean ± std.
Statistical significance tested with paired t-test (α = 0.05).
All configs and per-run results are logged to `outputs/` as JSON.
Retrieval pipeline runs are traced via Langfuse (set `langfuse.enabled: false` to disable).

---

## Citing

```bibtex
@inproceedings{revrag2026,
  title   = {{RevRAG}: Linear-Time Long-Context Retrieval with Intra-Document
             Cross-Page Consistency Verification},
  author  = {AUTHOR},
  year    = {2026},
}
```
