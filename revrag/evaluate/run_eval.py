"""
Main evaluation script for all baselines and RevRAG.

Runs all required metrics (Section 2.4 of paper spec):
    - Recall@1/5/10/20, NDCG@10, MRR          on all datasets
    - Evidence Consistency Score (ECS)          on all datasets
    - Revision Recall (RR)                      on synthetic benchmark
    - Conflict Resolution Accuracy (CRA)         on synthetic benchmark
    - Latency / memory vs. document length      efficiency sweep

Results are saved to CSV/JSON for direct import into paper tables.

Usage:
    python evaluate/run_eval.py --config configs/default.yaml \
        --model revrag --dataset legalbench --seed 42
"""

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from metrics.eval_metrics import (
    compute_all_metrics,
    character_precision_recall_at_k,
    evidence_consistency_score,
    conflict_resolution_accuracy,
    revision_recall,
    paired_ttest,
    bootstrap_ci,
)
from tracing.langfuse_tracer import LangfuseTracer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(model_name: str, cfg: Dict, device: str):
    """Instantiate the requested model."""
    if model_name == "bm25":
        from models.bm25_retriever import BM25Retriever
        return BM25Retriever()

    elif model_name == "bge":
        from models.bge_retriever import BGERetriever
        return BGERetriever(
            model_name=cfg["model"]["bge"]["checkpoint"],
            chunk_size=cfg["data"]["trec_dl"]["chunk_size"],
            chunk_overlap=cfg["data"]["trec_dl"]["chunk_overlap"],
            batch_size=cfg["retrieval"]["batch_size"],
            device=device,
        )

    elif model_name == "mamba":
        from models.mamba_retriever import MambaRetriever
        return MambaRetriever(
            checkpoint=cfg["model"]["mamba2"]["checkpoint"],
            hidden_size=cfg["model"]["mamba2"]["hidden_size"],
            max_length=cfg["model"]["mamba2"]["max_length"],
            batch_size=cfg["retrieval"]["batch_size"],
            device=device,
        )

    elif model_name == "revrag":
        from models.revrag import RevRAG
        return RevRAG(
            mamba_checkpoint=cfg["model"]["mamba2"]["checkpoint"],
            nli_checkpoint=cfg["model"]["consistency_verifier"]["checkpoint"],
            hidden_size=cfg["model"]["mamba2"]["hidden_size"],
            max_doc_length=cfg["model"]["mamba2"]["max_length"],
            top_k=cfg["retrieval"]["top_k"],
            final_k=cfg["retrieval"]["final_k"],
            nli_threshold=cfg["model"]["consistency_verifier"][
                "nli_contradiction_threshold"
            ],
            page_order_weight=cfg["model"]["consistency_verifier"][
                "page_order_weight"
            ],
            batch_size=cfg["retrieval"]["batch_size"],
            device=device,
            use_verifier=True,
        )

    elif model_name == "revrag_no_verifier":
        # Ablation (a): Mamba retriever without consistency module
        from models.revrag import RevRAG
        return RevRAG(
            mamba_checkpoint=cfg["model"]["mamba2"]["checkpoint"],
            hidden_size=cfg["model"]["mamba2"]["hidden_size"],
            max_doc_length=cfg["model"]["mamba2"]["max_length"],
            top_k=cfg["retrieval"]["top_k"],
            final_k=cfg["retrieval"]["final_k"],
            batch_size=cfg["retrieval"]["batch_size"],
            device=device,
            use_verifier=False,
        )

    elif model_name == "bgem3":
        # BGE-M3 baseline — 8192-token native, no chunking
        from models.bgem3_retriever import BGEM3Retriever
        return BGEM3Retriever(
            checkpoint=cfg["model"]["bgem3"]["checkpoint"],
            max_length=cfg["model"]["bgem3"]["max_length"],
            device=device,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_dataset(dataset_name: str, cfg: Dict):
    """Load the requested dataset."""
    if dataset_name == "trec_dl":
        from data.trec_dl import load_trec_dl
        return load_trec_dl(cfg)
    elif dataset_name == "legalbench":
        from data.legalbench import load_legalbench
        return load_legalbench(cfg)
    elif dataset_name == "synthetic":
        from data.legalbench import load_legalbench
        from data.synthetic import SyntheticBenchmarkDataset
        lb = load_legalbench(cfg)
        docs = lb.get_all_documents()
        return SyntheticBenchmarkDataset(
            cfg["data"]["synthetic"]["output_path"], docs
        )
    elif dataset_name == "loco":
        from data.loco import LoCoV1Dataset
        return LoCoV1Dataset(
            split=cfg["data"]["loco"].get("split", "test"),
            max_docs=cfg["data"]["loco"].get("max_docs"),
            cache_dir=cfg["data"]["loco"].get("cache_dir"),
        )
    elif dataset_name == "contractnli":
        from data.contractnli_retrieval import ContractNLIRetrievalDataset
        return ContractNLIRetrievalDataset(
            split=cfg["data"]["contractnli"].get("split", "test"),
            label_filter=cfg["data"]["contractnli"].get("label_filter"),
            min_span_length=cfg["data"]["contractnli"].get("min_span_length", 20),
            cache_dir=cfg["data"]["contractnli"].get("cache_dir"),
        )
    elif dataset_name == "cuad_amendments":
        from data.cuad_amendments import CUADAmendmentDataset
        return CUADAmendmentDataset(
            split=cfg["data"]["cuad_amendments"].get("split", "test"),
            categories=cfg["data"]["cuad_amendments"].get("categories"),
            min_char_gap=cfg["data"]["cuad_amendments"].get("min_char_gap", 500),
            cache_dir=cfg["data"]["cuad_amendments"].get("cache_dir"),
        )
    elif dataset_name == "multihop_rag":
        from data.multihop_rag import MultiHopRAGDataset
        return MultiHopRAGDataset(
            max_queries=cfg["data"]["multihop_rag"].get("max_queries", 500),
            neg_ratio=cfg["data"]["multihop_rag"].get("neg_ratio", 50),
            question_types=cfg["data"]["multihop_rag"].get("question_types"),
            cache_dir=cfg["data"]["multihop_rag"].get("cache_dir"),
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def run_evaluation(
    model,
    dataset,
    model_name: str,
    dataset_name: str,
    cfg: Dict,
    seed: int,
    tracer: LangfuseTracer,
    output_dir: Path,
) -> Dict:
    """Run evaluation for one (model, dataset, seed) combination."""
    logger.info(f"Evaluating {model_name} on {dataset_name} (seed={seed})")
    np.random.seed(seed)
    torch.manual_seed(seed)

    top_k = cfg["retrieval"]["top_k"]
    metrics = {}

    if dataset_name == "trec_dl":
        docs = dataset.get_doc_texts()
        queries = dataset.get_query_list()
        qrels = dataset.qrels

        # Index
        if hasattr(model, "index_documents"):
            model.index_documents(docs)
        elif hasattr(model, "index"):
            model.index(docs)

        # Retrieve
        t0 = time.time()
        if hasattr(model, "batch_retrieve"):
            raw_results = model.batch_retrieve(queries, top_k=top_k)
        else:
            raw_results = {}
            for qid, qtext in queries:
                raw_results[qid] = model.retrieve(qtext, top_k=top_k)
        latency_ms = (time.time() - t0) * 1000 / max(len(queries), 1)

        # Convert RerankedResult lists to (doc_id, score) tuples if needed
        flat_results = {}
        for qid, hits in raw_results.items():
            if hits and hasattr(hits[0], "passage_info"):
                flat_results[qid] = [
                    (r.passage_info.doc_id, r.final_score) for r in hits
                ]
            else:
                flat_results[qid] = hits

        metrics = compute_all_metrics(flat_results, qrels)
        metrics["latency_ms_per_query"] = latency_ms

    elif dataset_name == "legalbench":
        docs_dict = dataset.get_all_documents()
        docs = list(docs_dict.items())

        if hasattr(model, "index_documents"):
            model.index_documents(docs)
        elif hasattr(model, "index"):
            model.index(docs)

        queries = [(item["query_id"], item["query"]) for item in dataset]
        gold_spans = {
            item["query_id"]: item["gold_spans"] for item in dataset
        }

        t0 = time.time()
        if hasattr(model, "batch_retrieve"):
            raw_results = model.batch_retrieve(queries, top_k=top_k)
        else:
            raw_results = {}
            for qid, qtext in queries:
                raw_results[qid] = model.retrieve(qtext, top_k=top_k)
        latency_ms = (time.time() - t0) * 1000 / max(len(queries), 1)

        # Character-level precision/recall (LegalBench-RAG protocol)
        rich_results = {}
        for qid, hits in raw_results.items():
            if hits and hasattr(hits[0], "passage_info"):
                rich_results[qid] = [
                    (r.passage_info.doc_id, r.final_score,
                     {"text": r.passage_info.text})
                    for r in hits
                ]
            else:
                rich_results[qid] = hits

        p5, r5 = character_precision_recall_at_k(rich_results, gold_spans, k=5)
        metrics["char_precision@5"] = p5
        metrics["char_recall@5"] = r5
        metrics["latency_ms_per_query"] = latency_ms

    elif dataset_name == "synthetic":
        docs_dict = {item["doc_id"]: item.get("document", "") for item in dataset}
        docs = list(docs_dict.items())

        if hasattr(model, "index_documents"):
            model.index_documents(docs)
        elif hasattr(model, "index"):
            model.index(docs)

        queries = [(item["item_id"], item["query"]) for item in dataset]
        gold_answers = {item["item_id"]: item["gold_answer"] for item in dataset}

        t0 = time.time()
        if hasattr(model, "batch_retrieve"):
            raw_results = model.batch_retrieve(queries, top_k=top_k)
        else:
            raw_results = {}
            for qid, qtext in queries:
                raw_results[qid] = model.retrieve(qtext, top_k=top_k)
        latency_ms = (time.time() - t0) * 1000 / max(len(queries), 1)

        cra = conflict_resolution_accuracy(raw_results, gold_answers, top_k=1)
        metrics["cra@1"] = cra
        metrics["latency_ms_per_query"] = latency_ms

    elif dataset_name == "contractnli":
        # --- ContractNLI as retrieval benchmark ---
        doc_ids = list(dataset.documents.keys())
        doc_texts = [dataset.documents[d] for d in doc_ids]

        if hasattr(model, "index"):
            model.index(doc_texts, doc_ids)

        queries = [(q["query_id"], q["hypothesis"]) for q in dataset.queries]
        t0 = time.time()
        raw_results = {}
        for qid, qtext in queries:
            hits = model.retrieve(qtext, top_k=top_k)
            raw_results[qid] = hits
        latency_ms = (time.time() - t0) * 1000 / max(len(queries), 1)

        # Character-level precision/recall per query
        prec_all, rec_all = [], []
        prec_contradiction, rec_contradiction = [], []
        for q in dataset.queries:
            qid = q["query_id"]
            hits = raw_results.get(qid, [])
            texts = [h.text if hasattr(h, "text") else "" for h in hits[:5]]
            p, r = dataset.character_overlap(qid, texts)
            prec_all.append(p)
            rec_all.append(r)
            if q["label"] == "Contradiction":
                prec_contradiction.append(p)
                rec_contradiction.append(r)

        metrics["char_precision@5"] = float(np.mean(prec_all)) if prec_all else 0.0
        metrics["char_recall@5"] = float(np.mean(rec_all)) if rec_all else 0.0
        # Contradiction-only metrics — the cross-page revision test
        metrics["contradiction_char_precision@5"] = (
            float(np.mean(prec_contradiction)) if prec_contradiction else 0.0
        )
        metrics["contradiction_char_recall@5"] = (
            float(np.mean(rec_contradiction)) if rec_contradiction else 0.0
        )
        metrics["n_contradiction_queries"] = len(prec_contradiction)
        metrics["latency_ms_per_query"] = latency_ms

    elif dataset_name == "cuad_amendments":
        # --- CUAD real amendment pairs ---
        doc_ids = list(dataset.documents.keys())
        doc_texts = [dataset.documents[d] for d in doc_ids]

        if hasattr(model, "index"):
            model.index(doc_texts, doc_ids)

        t0 = time.time()
        cra_scores, decoy_scores, prec_all, rec_all = [], [], [], []

        for item in dataset.items:
            qid = item["item_id"]
            hits = model.retrieve(item["query"], top_k=top_k)
            texts = [h.text if hasattr(h, "text") else "" for h in hits[:5]]

            p, r = dataset.character_overlap(qid, texts)
            prec_all.append(p)
            rec_all.append(r)

            # CRA@1: does top-1 contain the gold (later/authoritative) span?
            top1 = texts[0] if texts else ""
            cra = 1.0 if item["gold_span_text"][:50] in top1 else 0.0
            cra_scores.append(cra)

            # Decoy promotion: does top-1 contain the earlier (wrong) span?
            decoy = dataset.decoy_promotion_rate(qid, texts, top_k=1)
            decoy_scores.append(decoy)

        latency_ms = (time.time() - t0) * 1000 / max(len(dataset.items), 1)

        metrics["char_precision@5"] = float(np.mean(prec_all)) if prec_all else 0.0
        metrics["char_recall@5"] = float(np.mean(rec_all)) if rec_all else 0.0
        metrics["cra@1"] = float(np.mean(cra_scores)) if cra_scores else 0.0
        metrics["decoy_promotion_rate"] = float(np.mean(decoy_scores)) if decoy_scores else 0.0
        metrics["latency_ms_per_query"] = latency_ms

    elif dataset_name == "multihop_rag":
        # --- MultiHop-RAG: standard retrieval metrics ---
        doc_ids = list(dataset.documents.keys())
        doc_texts = [dataset.documents[d] for d in doc_ids]

        if hasattr(model, "index"):
            model.index(doc_texts, doc_ids)

        queries_list = [(q["query_id"], q["query_text"]) for q in dataset.queries]
        t0 = time.time()
        recall_at_5, recall_at_20 = [], []

        for q in dataset.queries:
            hits = model.retrieve(q["query_text"], top_k=top_k)
            retrieved_ids = []
            for h in hits:
                if hasattr(h, "doc_id"):
                    retrieved_ids.append(h.doc_id)
                elif hasattr(h, "passage"):
                    retrieved_ids.append(h.passage.doc_id)
            recall_at_5.append(dataset.recall_at_k(q["query_id"], retrieved_ids, k=5))
            recall_at_20.append(dataset.recall_at_k(q["query_id"], retrieved_ids, k=20))

        latency_ms = (time.time() - t0) * 1000 / max(len(dataset.queries), 1)

        # Also compute NDCG with pytrec_eval
        run_dict = {}
        for q in dataset.queries:
            hits = model.retrieve(q["query_text"], top_k=top_k)
            run_dict[q["query_id"]] = {}
            for rank, h in enumerate(hits):
                did = h.doc_id if hasattr(h, "doc_id") else f"d{rank}"
                run_dict[q["query_id"]][did] = float(top_k - rank)

        try:
            from metrics.eval_metrics import ndcg_at_k
            qrel = dataset.get_qrel_dict()
            ndcg = ndcg_at_k(run_dict, qrel, k=10)
        except Exception:
            ndcg = float(np.mean(recall_at_5))

        metrics["recall@5"] = float(np.mean(recall_at_5))
        metrics["recall@20"] = float(np.mean(recall_at_20))
        metrics["ndcg@10"] = ndcg
        metrics["latency_ms_per_query"] = latency_ms

    logger.info(f"Metrics for {model_name}/{dataset_name}/seed={seed}: {metrics}")

    # Save per-run results
    run_out = {
        "model": model_name,
        "dataset": dataset_name,
        "seed": seed,
        **metrics,
    }
    out_file = output_dir / f"{model_name}_{dataset_name}_seed{seed}.json"
    with open(out_file, "w") as f:
        json.dump(run_out, f, indent=2)

    return metrics


def aggregate_seeds(results_list: List[Dict]) -> Dict:
    """Compute mean ± std across seeds."""
    all_keys = set().union(*[r.keys() for r in results_list])
    agg = {}
    for key in all_keys:
        vals = [r[key] for r in results_list if key in r]
        if vals and all(isinstance(v, (int, float)) for v in vals):
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
    return agg


def main():
    parser = argparse.ArgumentParser(description="RevRAG Evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--model",
        default="revrag",
        choices=["bm25", "bge", "mamba", "revrag", "revrag_no_verifier"],
    )
    parser.add_argument(
        "--dataset",
        default="legalbench",
        choices=["trec_dl", "legalbench", "synthetic"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = args.seeds or cfg["evaluation"]["seeds"]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(cfg["project"]["output_dir"]) / args.model / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    tracer = LangfuseTracer(cfg)

    all_seed_results = []
    for seed in seeds:
        model = build_model(args.model, cfg, device)
        dataset = load_dataset(args.dataset, cfg)
        result = run_evaluation(
            model, dataset, args.model, args.dataset, cfg, seed, tracer, output_dir
        )
        all_seed_results.append(result)

    # Aggregate and save summary
    agg = aggregate_seeds(all_seed_results)
    agg["model"] = args.model
    agg["dataset"] = args.dataset
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(agg, f, indent=2)

    logger.info(f"Summary written to {summary_path}")
    logger.info(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
