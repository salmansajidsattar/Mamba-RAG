"""
Plot latency and memory scaling curves (Figure 1 in the paper).

Reads outputs/efficiency/efficiency_results.json and produces:
    - latency_scaling.pdf  (ms/query vs. document length)
    - memory_scaling.pdf   (peak GPU MB vs. document length)

Run AFTER experiments/run_efficiency.py has completed.

Usage:
    python plotting/plot_scaling.py --results outputs/efficiency/efficiency_results.json
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Paper-quality style settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "text.usetex": False,
})

MODEL_STYLES = {
    "bge":    {"color": "#e41a1c", "marker": "o", "linestyle": "--",  "label": "BGE (chunked Transformer)"},
    "mamba":  {"color": "#377eb8", "marker": "s", "linestyle": "-.",  "label": "Mamba Retriever (no verifier)"},
    "revrag": {"color": "#4daf4a", "marker": "^", "linestyle": "-",   "label": "RevRAG (ours)"},
}


def plot_latency(results: dict, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    lengths = sorted(int(k) for k in results.keys())

    for model_name, style in MODEL_STYLES.items():
        means, stds = [], []
        for length in lengths:
            entry = results[str(length)].get(model_name, {})
            means.append(entry.get("mean_ms", np.nan))
            stds.append(entry.get("std_ms", np.nan))

        means = np.array(means, dtype=float)
        stds = np.array(stds, dtype=float)
        valid = ~np.isnan(means)

        ax.plot(
            np.array(lengths)[valid],
            means[valid],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            label=style["label"],
            linewidth=1.8,
            markersize=6,
        )
        ax.fill_between(
            np.array(lengths)[valid],
            (means - stds)[valid],
            (means + stds)[valid],
            color=style["color"],
            alpha=0.15,
        )

    ax.set_xlabel("Document length (tokens)")
    ax.set_ylabel("Latency (ms / query)")
    ax.set_title("Query Latency vs. Document Length")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    plt.tight_layout()

    out = output_dir / "latency_scaling.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_memory(results: dict, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    lengths = sorted(int(k) for k in results.keys())

    for model_name, style in MODEL_STYLES.items():
        mems = []
        for length in lengths:
            entry = results[str(length)].get(model_name, {})
            mems.append(entry.get("peak_memory_mb", np.nan))
        mems = np.array(mems, dtype=float)
        valid = ~np.isnan(mems)
        ax.plot(
            np.array(lengths)[valid],
            mems[valid],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            label=style["label"],
            linewidth=1.8,
            markersize=6,
        )

    ax.set_xlabel("Document length (tokens)")
    ax.set_ylabel("Peak GPU memory (MB)")
    ax.set_title("Peak Memory vs. Document Length")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    plt.tight_layout()

    out = output_dir / "memory_scaling.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="outputs/efficiency/efficiency_results.json",
    )
    parser.add_argument("--output-dir", default="outputs/figures")
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_latency(results, output_dir)
    plot_memory(results, output_dir)


if __name__ == "__main__":
    main()
