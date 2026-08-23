"""
evaluate.py
===========
Top-level evaluation entry point for RevRAG.

Usage:
    python evaluate.py                              # full eval on all datasets
    python evaluate.py --device cpu                 # CPU mode
    python evaluate.py --checkpoint outputs/checkpoints/best
    python evaluate.py --smoke                      # quick eval (20 queries/dataset)
    python evaluate.py --skip_ablation              # skip λ×τ_c grid
    python evaluate.py --skip_scaling               # skip encoding speed test
    python evaluate.py --dataset contractnli        # single dataset only

Output files (in outputs/results/):
    main_results.json     — Recall@5 per model per dataset
    full_metrics.json     — all metrics (R@1/5/20, NDCG@10, MRR, MAP, CRA@1, DPR, ECS)
    ablation_results.json — λ × τ_c grid + best params
    scaling_results.json  — encoding time vs sequence length

Output figures (in outputs/figures/):
    main_results_recall_at_5.pdf
    ablation_heatmap.pdf
    ablation_lambda.pdf
    ablation_threshold.pdf
    scaling_curve.pdf
    training_loss_curve.pdf    (if training_history.json exists)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")

# Auto-detect Kaggle and patch paths
if Path("/kaggle/working").exists():
    try:
        import importlib, sys
        sys.path.insert(0, "/kaggle/working")
        importlib.import_module("kaggle_config")
    except ModuleNotFoundError:
        logger.warning("Running on Kaggle but kaggle_config.py not found. "
                       "Run kaggle_download_datasets.py first.")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RevRAG")
    p.add_argument("--checkpoint",    default=None,
                   help="Path to trained BiEncoder checkpoint (dir with bi_encoder.pt)")
    p.add_argument("--device",        default=None,
                   help="cuda | cpu (auto-detect if omitted)")
    p.add_argument("--results_dir",   default=None,
                   help="Output directory for JSON results")
    p.add_argument("--figures_dir",   default=None,
                   help="Output directory for figures")
    p.add_argument("--smoke",         action="store_true",
                   help="Quick eval: 20 queries per dataset")
    p.add_argument("--skip_ablation", action="store_true",
                   help="Skip λ × τ_c ablation grid")
    p.add_argument("--skip_scaling",  action="store_true",
                   help="Skip encoding speed experiment")
    p.add_argument("--dataset",       default=None,
                   choices=["contractnli", "multihop", "loco", "all"],
                   help="Which dataset to evaluate on (default: all)")
    p.add_argument("--processed_dir", default=None,
                   help="Path to revrag_processed/ dir (skips raw dataset loading)")
    return p.parse_args()


def main():
    args = parse_args()

    import torch
    from src.config import CFG
    from src.evaluation.evaluator import RevRAGEvaluator

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ── Paths ─────────────────────────────────────────────────────────────────
    ckpt_path   = Path(args.checkpoint) if args.checkpoint else (
        CFG.out_dir / "checkpoints" / "best"
    )
    results_dir = Path(args.results_dir) if args.results_dir else (
        CFG.out_dir / "results"
    )
    figures_dir = Path(args.figures_dir) if args.figures_dir else (
        CFG.out_dir / "figures"
    )

    max_q = 20 if args.smoke else None

    # ── Run evaluation ────────────────────────────────────────────────────────
    ev = RevRAGEvaluator(
        checkpoint_path=ckpt_path if ckpt_path.exists() else None,
        results_dir=results_dir,
        device=device,
        skip_scaling=args.skip_scaling or args.smoke,
        skip_ablation=args.skip_ablation or args.smoke,
        max_queries=max_q,
    )

    processed_dir = Path(args.processed_dir) if args.processed_dir else None

    if args.dataset and args.dataset != "all":
        _run_single_dataset(ev, args.dataset, max_q, processed_dir=processed_dir)
    else:
        ev.run_all()

    # ── Generate figures ──────────────────────────────────────────────────────
    logger.info("Generating figures …")
    try:
        from src.visualization.plots import generate_all_plots
        generate_all_plots(results_dir, figures_dir)
        logger.info(f"✓ Figures → {figures_dir}")
    except Exception as e:
        logger.warning(f"Figure generation failed: {e}")

    logger.info(f"\n✓ Evaluation complete. Results → {results_dir}")


def _run_single_dataset(ev, dataset_name: str, max_q, processed_dir=None):
    """Run evaluation on a single dataset."""
    from src.config import CFG

    # ── Pre-processed JSON path (Kaggle / fast) ───────────────────────────────
    if processed_dir and processed_dir.exists():
        from src.data.processed_loader import ProcessedDataLoader
        loader = ProcessedDataLoader(processed_dir)

        if dataset_name == "contractnli":
            raw_queries = loader.contractnli_queries("test")
            if max_q:
                raw_queries = raw_queries[:max_q]
            ids, texts = loader.contractnli_passages()
            queries = [q["query"] for q in raw_queries]
            qids    = [q["query_id"] for q in raw_queries]
            # qrels: passage contains at least one gold_text substring
            qrels = {}
            for q in raw_queries:
                gold_set = set()
                for gt in q.get("gold_texts", []):
                    for pid, pt in zip(ids, texts):
                        if gt and gt[:80] in pt:
                            gold_set.add(pid)
                qrels[q["query_id"]] = gold_set or ({ids[0]} if ids else set())

        elif dataset_name == "multihop":
            raw_queries = loader.multihop_queries("test")
            if max_q:
                raw_queries = raw_queries[:max_q]
            ids, texts = loader.multihop_passages()
            queries    = [q["query"] for q in raw_queries]
            qids       = [q["query_id"] for q in raw_queries]
            qrels      = {q["query_id"]: set(q["gold_ids"]) for q in raw_queries}

        elif dataset_name == "loco":
            raw_queries = loader.loco_queries()
            if max_q:
                raw_queries = raw_queries[:max_q]
            ids, texts = loader.loco_passages()
            queries    = [q["query"] for q in raw_queries]
            qids       = [q["query_id"] for q in raw_queries]
            qrels      = {q["query_id"]: set(q["gold_ids"]) for q in raw_queries}

        else:
            logger.error(f"Unknown dataset: {dataset_name}")
            return

    # ── Raw dataset path (local) ─────────────────────────────────────────────
    else:
        if dataset_name == "contractnli":
            from src.data.contractnli import ContractNLIDataset
            ds  = ContractNLIDataset(CFG.data.contractnli_dir, split="test",
                                     seed=CFG.data.seed)
            ids, texts = ds.get_passages()
            queries = [e["hypothesis"] for e in ds.get_contradiction_queries()
                       ][:max_q or 9999]
            qids    = [f"cnli_{i}" for i in range(len(queries))]
            all_ids, all_texts = ids, texts
            qrels = {}
            for i, ex in enumerate(ds.get_contradiction_queries()[:len(queries)]):
                gold  = {pid for pid, pt in zip(all_ids, all_texts)
                         if ex.get("gold_text", "") in pt}
                qrels[f"cnli_{i}"] = gold or ({all_ids[0]} if all_ids else set())

        elif dataset_name == "multihop":
            from src.data.multihop import MultiHopRAGDataset
            ds  = MultiHopRAGDataset(CFG.data.hf_cache_dir, split="test",
                                     max_queries=max_q, seed=CFG.data.seed)
            ids, texts = ds.get_all_passages()
            queries    = ds.get_queries()
            qids       = [e["query_id"] for e in ds.examples]
            qrels      = ds.get_qrels()

        elif dataset_name == "loco":
            from src.data.loco import LoCoV1Dataset
            ds  = LoCoV1Dataset(CFG.data.hf_cache_dir, split="test",
                                max_queries=max_q, seed=CFG.data.seed)
            ids, texts = ds.get_all_passages()
            queries    = ds.get_queries()
            qids       = [e["query_id"] for e in ds.examples]
            qrels      = ds.get_qrels()

        else:
            logger.error(f"Unknown dataset: {dataset_name}")
            return

    results = ev.eval_dataset(dataset_name, ids, texts, queries, qids, qrels)

    print(f"\n{'='*55}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"{'Model':<25} {'R@1':>6} {'R@5':>6} {'R@20':>6} {'NDCG@10':>8} {'MRR':>6}")
    print("-" * 55)
    for model, m in results.items():
        print(
            f"{model:<25} "
            f"{m.get('recall@1',0):>6.3f} "
            f"{m.get('recall@5',0):>6.3f} "
            f"{m.get('recall@20',0):>6.3f} "
            f"{m.get('ndcg@10',0):>8.3f} "
            f"{m.get('mrr',0):>6.3f}"
        )
    print("=" * 55)


if __name__ == "__main__":
    main()
