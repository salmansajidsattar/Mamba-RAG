"""
src/visualization/plots.py
==========================
All paper-ready figures for RevRAG.

Figures produced:
  1. training_loss_curve.pdf       — train vs val loss per epoch
  2. main_results_bar.pdf          — Recall@5 per model per dataset
  3. scaling_curve.pdf             — encoding time vs sequence length (linear vs quadratic)
  4. ablation_lambda.pdf           — CRA@1 vs λ (page_order_weight)
  5. ablation_threshold.pdf        — CRA@1 vs τ_c (contradiction threshold)
  6. ablation_heatmap.pdf          — λ × τ_c grid heatmap
  7. pr_curve.pdf                  — Precision-Recall curves per model
  8. label_breakdown.pdf           — ContractNLI results split by NLI label

All saved as both .pdf (for LaTeX \includegraphics) and .png (for preview).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

logger = logging.getLogger(__name__)

# ── Style ─────────────────────────────────────────────────────────────────────
STYLE = {
    "figure.figsize":      (7, 4),
    "figure.dpi":          150,
    "font.size":           11,
    "axes.titlesize":      12,
    "axes.labelsize":      11,
    "legend.fontsize":     9,
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "grid.alpha":          0.3,
    "lines.linewidth":     1.8,
    "lines.markersize":    5,
}
plt.rcParams.update(STYLE)

MODEL_COLORS = {
    "BM25":              "#8B9DA7",
    "BGE":               "#5BA4CF",
    "BGE-M3":            "#3490DC",
    "Mamba (no verif.)": "#F39C12",
    "RevRAG (no verif.)":"#E67E22",
    "RevRAG":            "#27AE60",
}
MODEL_ORDER = list(MODEL_COLORS.keys())
LINESTYLES  = ["-", "--", "-.", ":", "-", "--"]
MARKERS     = ["o", "s", "^", "D", "v", "P"]


def _save(fig, path: Path, name: str):
    """Save figure as both PDF and PNG."""
    path.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(path / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  ✓ {name}.pdf + .png → {path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Training loss curve
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_training_loss(history: List[Dict], out_dir: Path):
    """history = list of {epoch, train_loss, val_loss} dicts."""
    epochs     = [h["epoch"]      for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]

    fig, ax = plt.subplots()
    ax.plot(epochs, train_loss, label="Train loss", color="#2980B9", marker="o")
    ax.plot(epochs, val_loss,   label="Val loss",   color="#E74C3C", marker="s",
            linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("InfoNCE Loss")
    ax.set_title("RevRAG Bi-Encoder Training")
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    _save(fig, out_dir, "training_loss_curve")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Main results bar chart
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_main_results(
    results: Dict[str, Dict[str, float]],   # {model: {dataset: metric_value}}
    metric:  str,
    out_dir: Path,
    title:   str = "",
):
    """
    Grouped bar chart: models × datasets.

    Parameters
    ----------
    results : {model_name: {dataset_name: score}}
    metric  : metric name for y-axis label (e.g. "Recall@5")
    """
    models   = [m for m in MODEL_ORDER if m in results]
    datasets = sorted({ds for m in models for ds in results[m]})

    x    = np.arange(len(datasets))
    w    = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(max(8, 2 * len(datasets)), 4.5))

    for i, model in enumerate(models):
        scores = [results[model].get(ds, 0.0) for ds in datasets]
        offset = (i - len(models) / 2 + 0.5) * w
        bars   = ax.bar(
            x + offset, scores, w * 0.9,
            label=model,
            color=MODEL_COLORS.get(model, "#95A5A6"),
        )
        # Annotate bars
        for bar, score in zip(bars, scores):
            if score > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{score:.3f}",
                    ha="center", va="bottom", fontsize=7, rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1.05)
    ax.set_title(title or f"RevRAG — {metric} by Dataset")
    ax.legend(loc="upper right", ncol=2)
    _save(fig, out_dir, f"main_results_{metric.lower().replace('@','_at_')}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Scaling curve (encoding time vs sequence length)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_scaling_curve(
    seq_lengths:   List[int],
    time_mamba:    List[float],   # seconds per token — Mamba-2 (linear)
    time_bert:     List[float],   # seconds per token — BERT (quadratic)
    out_dir:       Path,
):
    """
    Encoding time vs sequence length.
    Demonstrates Mamba-2's O(T) advantage over BERT's O(T²).
    """
    fig, ax = plt.subplots()
    ax.plot(seq_lengths, time_mamba, label="Mamba-2 (RevRAG)", color="#27AE60",
            marker="o")
    ax.plot(seq_lengths, time_bert,  label="BERT/Transformer", color="#E74C3C",
            marker="s", linestyle="--")

    # Fit reference lines
    x = np.array(seq_lengths, dtype=float)
    # Linear fit for Mamba
    a_lin = np.polyfit(x, time_mamba, 1)
    # Quadratic fit for BERT
    a_qua = np.polyfit(x, time_bert, 2)
    xs = np.linspace(min(x), max(x), 200)
    ax.plot(xs, np.polyval(a_lin, xs), color="#27AE60", alpha=0.3,
            linestyle="-", label="O(T) fit")
    ax.plot(xs, np.polyval(a_qua, xs), color="#E74C3C", alpha=0.3,
            linestyle="-", label="O(T²) fit")

    ax.set_xlabel("Sequence Length (tokens)")
    ax.set_ylabel("Encoding Time (s)")
    ax.set_title("Encoding Efficiency: Mamba-2 vs. Transformer")
    ax.legend()
    _save(fig, out_dir, "scaling_curve")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4 & 5. Ablation: λ and τ_c sensitivity
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_ablation_lambda(
    lambda_values: List[float],
    cra_scores:    List[float],
    out_dir:       Path,
    best_lambda:   Optional[float] = None,
):
    fig, ax = plt.subplots()
    ax.plot(lambda_values, cra_scores, marker="o", color="#8E44AD")
    if best_lambda is not None:
        ax.axvline(best_lambda, color="#E74C3C", linestyle="--", alpha=0.7,
                   label=f"Best λ={best_lambda}")
        ax.legend()
    ax.set_xlabel("λ (page_order_weight)")
    ax.set_ylabel("CRA@1")
    ax.set_title("Ablation: Page-Order Weight (λ)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1.05)
    _save(fig, out_dir, "ablation_lambda")


def plot_ablation_threshold(
    threshold_values: List[float],
    cra_scores:       List[float],
    out_dir:          Path,
    best_tau:         Optional[float] = None,
):
    fig, ax = plt.subplots()
    ax.plot(threshold_values, cra_scores, marker="s", color="#16A085")
    if best_tau is not None:
        ax.axvline(best_tau, color="#E74C3C", linestyle="--", alpha=0.7,
                   label=f"Best τ_c={best_tau}")
        ax.legend()
    ax.set_xlabel("τ_c (contradiction threshold)")
    ax.set_ylabel("CRA@1")
    ax.set_title("Ablation: NLI Contradiction Threshold (τ_c)")
    ax.set_ylim(0, 1.05)
    _save(fig, out_dir, "ablation_threshold")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Ablation heatmap (λ × τ_c)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_ablation_heatmap(
    lambda_values:    List[float],
    threshold_values: List[float],
    cra_grid:         np.ndarray,   # shape (len(lambda), len(tau))
    out_dir:          Path,
):
    """
    Heatmap of CRA@1 over λ × τ_c grid.
    cra_grid[i, j] = CRA@1 for lambda_values[i], threshold_values[j].
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        cra_grid, aspect="auto", cmap="RdYlGn",
        vmin=0, vmax=1,
        origin="lower",
    )
    ax.set_xticks(range(len(threshold_values)))
    ax.set_xticklabels([f"{t:.2f}" for t in threshold_values], rotation=45)
    ax.set_yticks(range(len(lambda_values)))
    ax.set_yticklabels([f"{l:.1f}" for l in lambda_values])
    ax.set_xlabel("τ_c (NLI contradiction threshold)")
    ax.set_ylabel("λ (page_order_weight)")
    ax.set_title("CRA@1 Ablation: λ × τ_c")

    plt.colorbar(im, ax=ax, label="CRA@1", shrink=0.8)

    # Annotate cells
    for i in range(len(lambda_values)):
        for j in range(len(threshold_values)):
            ax.text(j, i, f"{cra_grid[i, j]:.2f}",
                    ha="center", va="center", fontsize=6,
                    color="white" if cra_grid[i, j] < 0.5 else "black")

    _save(fig, out_dir, "ablation_heatmap")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Precision-Recall curve
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_pr_curve(
    model_pr: Dict[str, Tuple[List[float], List[float]]],   # {model: (precision, recall)}
    out_dir:  Path,
    title:    str = "Precision-Recall Curve",
):
    """model_pr: {model_name: (precisions_list, recalls_list)}"""
    fig, ax = plt.subplots()
    for (model, (precisions, recalls)), ls, mk in zip(
        model_pr.items(), LINESTYLES, MARKERS
    ):
        ax.plot(
            recalls, precisions,
            label=model,
            color=MODEL_COLORS.get(model, "#95A5A6"),
            linestyle=ls, marker=mk, markevery=max(1, len(recalls) // 8),
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(loc="upper right")
    _save(fig, out_dir, "pr_curve")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. ContractNLI label breakdown
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_label_breakdown(
    results_by_label: Dict[str, Dict[str, float]],   # {label: {model: score}}
    metric:  str,
    out_dir: Path,
):
    """
    Grouped bars: NLI label (Entailment / Contradiction) × model.
    Shows RevRAG verifier specifically helps on Contradiction examples.
    """
    labels = list(results_by_label.keys())
    models = MODEL_ORDER

    x = np.arange(len(labels))
    w = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(6, 4))

    for i, model in enumerate(models):
        if not any(model in results_by_label[lb] for lb in labels):
            continue
        scores = [results_by_label[lb].get(model, 0.0) for lb in labels]
        offset = (i - len(models) / 2 + 0.5) * w
        ax.bar(x + offset, scores, w * 0.9,
               label=model, color=MODEL_COLORS.get(model, "#95A5A6"))

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1.05)
    ax.set_title("ContractNLI: Results by NLI Label")
    ax.legend(ncol=2, fontsize=8)
    _save(fig, out_dir, "label_breakdown")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Convenience: generate all plots from saved results JSON
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_all_plots(results_dir: Path, out_dir: Path):
    """
    Load results JSON files from results_dir and produce all paper figures.
    Called by evaluate.py after evaluation is complete.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training loss
    hist_path = results_dir / "training_history.json"
    if hist_path.exists():
        with open(hist_path) as f:
            history = json.load(f)
        plot_training_loss(history, out_dir)

    # Main results
    main_path = results_dir / "main_results.json"
    if main_path.exists():
        with open(main_path) as f:
            main = json.load(f)
        for metric in ["Recall@5", "CRA@1", "NDCG@10"]:
            if any(metric.lower().replace("@", "_at_") in str(v)
                   for v in main.values()):
                plot_main_results(main, metric, out_dir)

    # Ablation
    abl_path = results_dir / "ablation_results.json"
    if abl_path.exists():
        with open(abl_path) as f:
            abl = json.load(f)
        if "lambda_values" in abl:
            plot_ablation_lambda(
                abl["lambda_values"], abl["cra_scores_lambda"],
                out_dir, best_lambda=abl.get("best_lambda"),
            )
        if "threshold_values" in abl:
            plot_ablation_threshold(
                abl["threshold_values"], abl["cra_scores_threshold"],
                out_dir, best_tau=abl.get("best_tau"),
            )
        if "grid" in abl:
            plot_ablation_heatmap(
                abl["lambda_values"], abl["threshold_values"],
                np.array(abl["grid"]), out_dir,
            )

    logger.info(f"All plots saved → {out_dir}")
