# MambaRAG

Revision-aware retrieval for long, multi-page documents.

MambaRAG (the method is called RevRAG in the code and paper) reads a full document with a Mamba-2 encoder (no chunking), then checks the top results for contradictions across pages. When two passages disagree, the later page wins. This helps with contracts where an amendment on page 22 overrides a clause on page 4.

## The problem

A normal RAG system finds the old clause, misses the amendment, and the generator gives a confident wrong answer.

```
Page 4 : "Monthly payment is $1,000."      <- retrieved
Page 22: "Amendment: payment is now $1,500." <- missed
Answer : "$1,000"                            <- wrong
```

## How it works

```
query -> Mamba-2 bi-encoder -> FAISS top-k -> NLI verifier -> final top-k -> generator
```

1. **Mamba-2 bi-encoder.** Encodes the whole document in one linear-time pass. No chunking. (Falls back to BERT on CPU.)
2. **FAISS search.** Gets top-k candidate passages.
3. **Cross-page verifier.** An NLI model (`cross-encoder/nli-deberta-v3-small`) scores every passage pair. If contradiction is above `tau_c`, the passage from the later page gets a boost.
4. **Final score.** `final = lambda * page_order_boost + (1 - lambda) * retrieval_score`

Both parts can be turned on or off, so each one can be tested alone.

## Project layout

```
config.py            settings for the experiment pipeline
run_experiment.py    runs the full pipeline (steps 1 to 11)
train.py             train the bi-encoder
evaluate.py          evaluate all models, ablation, scaling
smoke_test.py        quick CPU check
steps/               one file per pipeline step
src/
  config.py          model, training, data, eval settings
  data/              ContractNLI, CUAD, LegalBench-RAG, MultiHop-RAG, LoCo loaders
  models/            encoder, bi_encoder, verifier, revrag
  training/          InfoNCE and MarginMSE losses, trainer
  evaluation/        metrics and evaluator
  visualization/     plots
revrag/              older standalone version with SLURM scripts and tests
backends/            local Hugging Face LLM for generation eval
paper/               LaTeX paper and references
literature_review*.md  literature notes
kaggle_*.py, package_for_kaggle.py, preprocess_local.py   Kaggle helpers
```

## Setup

Python 3.11 is recommended.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r revrag/requirements.txt
pip install mamba-ssm causal-conv1d   # needs CUDA 12.x and an NVIDIA GPU
```

Without a GPU, the code runs with the BERT fallback encoder.

## Quick start

```bash
python smoke_test.py
```

## Data

Benchmarks used:

| Dataset | Used for |
|---|---|
| ContractNLI | contract retrieval, training triplets |
| CUAD (amendments) | amendment pairs, hard negatives |
| LegalBench-RAG | character-level retrieval on legal docs |
| MultiHop-RAG | multi-hop retrieval |
| LoCoV1 | long-context retrieval |
| Synthetic cross-page set | ablation and error analysis only |

Download and prepare:

```bash
python download_datasets_local.py
python preprocess_local.py
```

For Kaggle, use `kaggle_download_datasets.py` or `package_for_kaggle.py`, then run `kaggle_revrag.ipynb`.

## Train

```bash
python train.py
python train.py --device cpu
python train.py --smoke
```

Training uses InfoNCE loss with hard negatives. The best checkpoint is saved to `outputs/checkpoints/best/`.

## Evaluate

```bash
python evaluate.py
python evaluate.py --checkpoint outputs/checkpoints/best
python evaluate.py --dataset contractnli
python evaluate.py --smoke
```

Results go to `outputs/results/` and figures to `outputs/figures/`.

Run the whole pipeline (download, eval, ablation, generation, plots):

```bash
python run_experiment.py
```

## Models compared

BM25, BGE (chunked), BGE-M3, Mamba-2 retriever, RevRAG without verifier, RevRAG.

## Metrics

Recall@1/5/20, NDCG@10, MRR, MAP, and three metrics for conflicts:

- **CRA@1**: is the correct (later) passage ranked above the outdated one?
- **DPR**: how often the outdated passage ends up at rank 1 (lower is better)
- **ECS**: share of passage pairs with no contradiction

## Main settings

| Setting | Default |
|---|---|
| Encoder | `state-spaces/mamba2-130m` |
| Max tokens | 4096 |
| Verifier | `cross-encoder/nli-deberta-v3-small` |
| `lambda` (page order weight) | 0.3 |
| `tau_c` (contradiction threshold) | 0.5 |
| Candidates / returned | 20 / 5 |
| Seeds | 42, 123, 456 |

The ablation sweeps `lambda` from 0.0 to 1.0 and `tau_c` from 0.30 to 0.70.

## Generation eval

Uses a free local model, no API key. Pick one in `config.py`: `phi3` (default), `mistral`, `flan`, `smollm`.

## Status

This project is under active experiments. Code and results can change. Final benchmark numbers are not available yet.

## Citation

```bibtex
@inproceedings{revrag2026,
  title  = {RevRAG: Linear-Time Long-Context Retrieval with Intra-Document Cross-Page Consistency Verification},
  author = {Salman Sajid},
  year   = {2026}
}
```

## Contact

Salman Sajid, salmansajidsattar@gmail.com
