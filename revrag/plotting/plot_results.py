"""
Plot main results table as a grouped bar chart (Figure 2 in the paper).

Reads summary.json files from outputs/<model>/<dataset>/summary.json
and produces comparison plots across all baselines and datasets.

Usage:
    python plotting/plot_results.py --output-dir outputs/figures
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "figure.dpi": 300,
})

MODELS = ["bm25", "bge", "mamba", "revrag_no_verifier", "revrag"]
MODEL_LABELS = {
    "bm25": "BM25",
    "bge": "BGE",
    "mamba": "Mamba",
    "revrag_no_verifier": "RevRAG\n(no verifier)",
    "revrag": "RevRAG\n(ours)",
}
COLORS = ["#aaaaaa", "#e41a1c", "#377eb8", "#ff7f00", "#4daf4a"]

DATASETS = ["trec_dl", "legalbench", "synthetic"]
DATASET_LABELS = {
    "trec_dl": "TREC DL",
    "legalbench": "LegalBench-RAG",
    "synthetic": "Synthetic (CRA)",
}

# Metric to show per dataset
PRIMARY_METRIC = {
    "trec_dl": "ndcg@10_mean",
    "legalbench": "char_recall@5_mean",
    "synthetic": "cra@1_mean",
}
PRIMARY_METRIC_LABEL = {
    "trec_dl": "NDCG@10",
    "legalbench": "Char Recall@5",
    "synthetic": "CRA@1",
}


def load_results(base_dir: Path) -> Dict:
    results = {}
    for model in MODELS:
        results[model] = {}
        for dataset in DATASETS:
            summary = base_dir / model / dataset / "summary.json"
            if summary.exists():
                with open(summary) as f:
                    data = json.load(f)
                results[model][dataset] = data
    return results


def plot_main_results(results: Dict, output_dir: Path) -> None:
    n_datasets = len(DATASETS)
    n_models = len(MODELS)
    x = np.arange(n_datasets)
    width = 0.15
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for i, (model, color) in enumerate(zip(MODELS, COLORS)):
        means, errs = [], []
        for dataset in DATASETS:
            metric_key = PRIMARY_METRIC[dataset]
            err_key = metric_key.replace("_mean", "_std")
            val = results.get(model, {}).get(dataset, {}).get(metric_key, 0.0)
            err = results.get(model, {}).get(dataset, {}).get(err_key, 0.0)
            means.append(val if val else 0.0)
            errs.append(err if err else 0.0)

        bars = ax.bar(
            x + offsets[i],
            means,
            width,
            label=MODEL_LABELS[model],
            color=color,
            yerr=errs,
            capsize=3,
            error_kw={"linewidth": 1},
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([
        f"{DATASET_LABELS[d]}\n({PRIMARY_METRIC_LABEL[d]})"
        for d in DATASETS
    ])
    ax.set_ylabel("Score")
    ax.set_title("Retrieval Performance Across Datasets (mean ± std, 3 seeds)")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()

    out = output_dir / "main_results.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="outputs")
    parser.add_argument("--output-dir", default="outputs/figures")
    args = parser.parse_args()

    base_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(base_dir)
    plot_main_results(results, output_dir)


if __name__ == "__main__":
    main()
