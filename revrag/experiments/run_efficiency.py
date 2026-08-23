"""
Efficiency experiment: latency (ms/query) and peak GPU memory vs. document length.

Sweeps across doc_length_buckets (tokens) for:
    - BGE chunked Transformer (quadratic baseline)
    - Mamba-2 retriever (linear)
    - RevRAG = Mamba-2 + verifier (linear + overhead)

Produces the scaling curve plot data saved to outputs/efficiency/.

This is the figure reviewers will remember. Run on GPU for realistic numbers.

Usage:
    python experiments/run_efficiency.py --config configs/default.yaml
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _generate_synthetic_doc(length_tokens: int, vocab_size: int = 10000) -> str:
    """Generate a random synthetic document of approximately `length_tokens` tokens."""
    words_per_token = 0.75  # rough conversion
    n_words = int(length_tokens * words_per_token)
    rng = np.random.default_rng(42)
    words = [f"word{rng.integers(0, vocab_size)}" for _ in range(n_words)]
    return " ".join(words)


def measure_latency(
    model,
    doc: str,
    query: str = "What is the payment amount?",
    warmup: int = 3,
    timed: int = 10,
    device: str = "cpu",
) -> Dict:
    """
    Measure inference latency and peak memory for a single (doc, query) pair.

    Returns dict with mean_ms, std_ms, peak_memory_mb.
    """
    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.reset_peak_memory_stats()

    # Warmup
    for _ in range(warmup):
        if hasattr(model, "retrieve"):
            model.retrieve(query, top_k=5)
        elif hasattr(model, "batch_retrieve"):
            model.batch_retrieve([("q0", query)], top_k=5)

    # Timed runs
    latencies = []
    for _ in range(timed):
        t0 = time.perf_counter()
        if hasattr(model, "retrieve"):
            model.retrieve(query, top_k=5)
        elif hasattr(model, "batch_retrieve"):
            model.batch_retrieve([("q0", query)], top_k=5)
        latencies.append((time.perf_counter() - t0) * 1000)

    peak_mem_mb = 0.0
    if torch.cuda.is_available() and device != "cpu":
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "peak_memory_mb": peak_mem_mb,
    }


def run_efficiency_sweep(cfg: Dict, output_dir: Path, device: str) -> None:
    from models.bge_retriever import BGERetriever
    from models.mamba_retriever import MambaRetriever
    from models.revrag import RevRAG

    buckets = cfg["efficiency"]["doc_length_buckets"]
    warmup = cfg["efficiency"]["warmup_runs"]
    timed = cfg["efficiency"]["timed_runs"]

    results = {b: {} for b in buckets}
    query = "What is the payment amount specified in this agreement?"

    for length in buckets:
        logger.info(f"=== Document length: {length} tokens ===")
        doc = _generate_synthetic_doc(length)
        documents = [("synthetic_doc", doc)]

        # BGE
        logger.info("  Benchmarking BGE...")
        try:
            bge = BGERetriever(
                model_name=cfg["model"]["bge"]["checkpoint"],
                chunk_size=cfg["data"]["trec_dl"]["chunk_size"],
                batch_size=cfg["retrieval"]["batch_size"],
                device=device,
            )
            bge.index_documents(documents)
            results[length]["bge"] = measure_latency(
                bge, doc, query, warmup, timed, device
            )
        except Exception as e:
            logger.warning(f"BGE benchmark failed: {e}")
            results[length]["bge"] = {"mean_ms": None, "std_ms": None, "peak_memory_mb": None}

        # Mamba
        logger.info("  Benchmarking Mamba-2...")
        try:
            mamba = MambaRetriever(
                checkpoint=cfg["model"]["mamba2"]["checkpoint"],
                max_length=cfg["model"]["mamba2"]["max_length"],
                batch_size=cfg["retrieval"]["batch_size"],
                device=device,
            )
            mamba.index_documents(documents)
            results[length]["mamba"] = measure_latency(
                mamba, doc, query, warmup, timed, device
            )
        except Exception as e:
            logger.warning(f"Mamba benchmark failed: {e}")
            results[length]["mamba"] = {"mean_ms": None, "std_ms": None, "peak_memory_mb": None}

        # RevRAG
        logger.info("  Benchmarking RevRAG...")
        try:
            revrag = RevRAG(
                mamba_checkpoint=cfg["model"]["mamba2"]["checkpoint"],
                nli_checkpoint=cfg["model"]["consistency_verifier"]["checkpoint"],
                max_doc_length=cfg["model"]["mamba2"]["max_length"],
                top_k=cfg["retrieval"]["top_k"],
                device=device,
                use_verifier=True,
            )
            revrag.index_documents(documents)
            results[length]["revrag"] = measure_latency(
                revrag, doc, query, warmup, timed, device
            )
        except Exception as e:
            logger.warning(f"RevRAG benchmark failed: {e}")
            results[length]["revrag"] = {"mean_ms": None, "std_ms": None, "peak_memory_mb": None}

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "efficiency_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Efficiency results saved to {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg["project"]["output_dir"]) / "efficiency"
    run_efficiency_sweep(cfg, output_dir, device)


if __name__ == "__main__":
    main()
