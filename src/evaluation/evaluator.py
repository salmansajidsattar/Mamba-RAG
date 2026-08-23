"""
src/evaluation/evaluator.py
============================
Full evaluation runner for the RevRAG paper.

Runs six models on three datasets and computes all paper metrics:
  Models:   BM25 | BGE | BGE-M3 | Mamba (no verif.) | RevRAG (no verif.) | RevRAG
  Datasets: ContractNLI | MultiHopRAG | LoCoV1
  Metrics:  Recall@{1,5,20}, NDCG@10, MRR, MAP, CRA@1, DPR↓, ECS

Also runs:
  - λ × τ_c ablation grid on CUAD
  - Encoding time scaling curve (Mamba-2 vs BERT/Transformer)
  - ContractNLI results broken down by NLI label
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from src.config import CFG
from src.evaluation.metrics import (
    compute_all_metrics,
    cra_at_k,
    decoy_promotion_rate,
    evidence_consistency_score,
    character_recall_at_k,
)
from src.models.revrag import RevRAG

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BM25 baseline (rank_bm25 — pure Python, no GPU needed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BM25Retriever:
    """Thin wrapper around rank_bm25."""

    def __init__(self, doc_ids: List[str], doc_texts: List[str]):
        from rank_bm25 import BM25Okapi
        self.doc_ids  = doc_ids
        tokenised     = [t.lower().split() for t in doc_texts]
        self.bm25     = BM25Okapi(tokenised)

    def retrieve(self, queries: List[str], top_k: int = 20) -> List[List[Dict]]:
        results = []
        for q in queries:
            scores = self.bm25.get_scores(q.lower().split())
            ranked = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:top_k]
            results.append([
                {"doc_id": self.doc_ids[i], "score": float(scores[i])}
                for i in ranked
            ])
        return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dense baselines: BGE and BGE-M3
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SentenceTransformerRetriever:
    """Dense retriever using a sentence-transformers model + FAISS."""

    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.model    = SentenceTransformer(model_name, device=device)
        self.doc_ids  = []
        self.index    = None

    def build_index(self, doc_ids: List[str], doc_texts: List[str], batch_size: int = 64):
        import faiss
        self.doc_ids = doc_ids
        embs = self.model.encode(
            doc_texts, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=True,
        )
        dim  = embs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embs.astype(np.float32))

    def retrieve(self, queries: List[str], top_k: int = 20) -> List[List[Dict]]:
        q_embs  = self.model.encode(
            queries, normalize_embeddings=True, show_progress_bar=False
        )
        scores, indices = self.index.search(q_embs.astype(np.float32), top_k)
        results = []
        for row_s, row_i in zip(scores, indices):
            results.append([
                {"doc_id": self.doc_ids[i], "score": float(s)}
                for s, i in zip(row_s, row_i)
                if i < len(self.doc_ids)
            ])
        return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper: build all models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_revrag(
    checkpoint_path: Optional[Path] = None,
    use_verifier: bool = True,
    device: str = "cpu",
) -> RevRAG:
    model = RevRAG(
        checkpoint=CFG.model.mamba_checkpoint,
        hidden_size=CFG.model.hidden_size,
        max_seq_len=CFG.model.max_seq_len,
        top_k_retrieval=CFG.eval.top_k_retrieval,
        top_k_final=max(CFG.eval.recall_k),
        page_order_weight=CFG.model.page_order_weight,
        contradiction_threshold=CFG.model.nli_contradiction_threshold,
        use_verifier=use_verifier,
        device=device,
        verifier_checkpoint=CFG.model.verifier_checkpoint,
    )
    if checkpoint_path and checkpoint_path.exists():
        model.load(checkpoint_path)
    return model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Core evaluation function
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def evaluate_retriever(
    retriever,
    queries:    List[str],
    query_ids:  List[str],
    qrels:      Dict[str, Set[str]],
    k_values:   List[int],
    is_revrag:  bool = False,
) -> Dict[str, float]:
    """
    Run retriever on queries and compute all metrics.

    retriever must have a .retrieve(queries, top_k) method returning
    List[List[Dict]] where each dict has at least "doc_id".
    """
    top_k = max(k_values)
    raw   = retriever.retrieve(queries, top_k=top_k)

    run_results: Dict[str, List[str]] = {}
    cuad_passages: List[List[Dict]]   = []

    for qid, result_list in zip(query_ids, raw):
        run_results[qid] = [r["doc_id"] for r in result_list]
        if is_revrag:
            cuad_passages.append(result_list)

    metrics = compute_all_metrics(run_results, qrels, k_values=k_values)

    # RevRAG-specific metrics
    if is_revrag and cuad_passages:
        metrics["cra@1"]  = float(np.mean([
            cra_at_k(p, k=1) for p in cuad_passages
        ]))
        metrics["dpr"]    = decoy_promotion_rate(cuad_passages, k=1)
        ecs_vals = []
        for p in cuad_passages:
            texts = [x.get("text", "") for x in p]
            # ECS requires verifier — use nli_contradiction_score field if present
            nli_scores = [x.get("nli_contradiction_score", 0.0) for x in p]
            if any(s > 0 for s in nli_scores):
                pairs   = [(texts[a], texts[b])
                           for a in range(len(texts))
                           for b in range(a + 1, len(texts))]
                n_pairs = len(pairs) or 1
                n_contradictions = sum(
                    1 for x in p if x.get("nli_contradiction_score", 0) > 0.5
                )
                ecs_vals.append(1.0 - n_contradictions / n_pairs)
        if ecs_vals:
            metrics["ecs"] = float(np.mean(ecs_vals))

    return metrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Encoding time scaling experiment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def measure_encoding_speed(
    device: str = "cpu",
    seq_lengths: Optional[List[int]] = None,
    n_docs: int = 32,
) -> Dict:
    """
    Measure encoding throughput (tokens/sec) at different sequence lengths.
    Returns timing data for the scaling curve figure.
    """
    from src.models.encoder import MambaEncoder
    from transformers import AutoTokenizer, AutoModel
    import torch

    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024, 2048, 4096]

    def time_model(encode_fn, tokens: torch.Tensor, mask: torch.Tensor) -> float:
        """Returns seconds per forward pass (mean of 5 runs)."""
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            with torch.no_grad():
                encode_fn(tokens, mask)
            times.append(time.perf_counter() - t0)
        return float(np.mean(times[1:]))   # discard first (JIT warmup)

    results = {
        "seq_lengths":  seq_lengths,
        "time_mamba":   [],
        "time_bert":    [],
    }

    # Mamba-2 encoder (or BERT fallback on CPU)
    mamba_enc = MambaEncoder(
        checkpoint=CFG.model.mamba_checkpoint,
        hidden_size=CFG.model.hidden_size,
        device=device,
    )
    mamba_enc.eval()
    tokenizer = mamba_enc.tokenizer

    # BERT reference
    bert_name = "bert-base-uncased"
    from transformers import BertModel
    bert_tok  = AutoTokenizer.from_pretrained(bert_name)
    bert_mdl  = BertModel.from_pretrained(bert_name).to(device).eval()

    for seq_len in seq_lengths:
        dummy_text = ["the contract shall be " * (seq_len // 5 + 1)] * n_docs
        # Truncate to exact seq_len tokens
        tok_out = tokenizer(
            dummy_text, max_length=seq_len, truncation=True,
            padding=True, return_tensors="pt"
        )
        ids  = tok_out["input_ids"].to(device)
        mask = tok_out["attention_mask"].to(device)

        t_mamba = time_model(
            lambda i, m: mamba_enc(i, m),
            ids, mask,
        )
        results["time_mamba"].append(t_mamba)

        bert_tok_out = bert_tok(
            dummy_text,
            max_length=min(seq_len, 512),
            truncation=True, padding=True,
            return_tensors="pt",
        )
        b_ids  = bert_tok_out["input_ids"].to(device)
        b_mask = bert_tok_out["attention_mask"].to(device)
        t_bert = time_model(
            lambda i, m: bert_mdl(input_ids=i, attention_mask=m),
            b_ids, b_mask,
        )
        results["time_bert"].append(t_bert)

        logger.info(
            f"  seq_len={seq_len:5d} | Mamba={t_mamba:.3f}s | BERT={t_bert:.3f}s"
        )

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Ablation: λ × τ_c grid
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_ablation(
    revrag:       RevRAG,
    queries:      List[str],
    query_ids:    List[str],
    qrels:        Dict[str, Set[str]],
    lambda_values:    Optional[List[float]] = None,
    threshold_values: Optional[List[float]] = None,
) -> Dict:
    """
    Sweep λ and τ_c and record CRA@1 at each point.
    Returns a dict with "grid" (2D np array) + axis lists.
    """
    lv = lambda_values    or CFG.eval.lambda_values
    tv = threshold_values or CFG.eval.threshold_values

    grid = np.zeros((len(lv), len(tv)))

    for i, lam in enumerate(lv):
        for j, tau in enumerate(tv):
            raw = revrag.retrieve(
                queries,
                top_k=max(CFG.eval.recall_k),
                lambda_override=lam,
                tau_override=tau,
            )
            cra_scores = [cra_at_k(p, k=1) for p in raw]
            grid[i, j] = float(np.mean(cra_scores))
            logger.info(f"  λ={lam:.1f}  τ_c={tau:.2f}  CRA@1={grid[i,j]:.3f}")

    # Best single-param series
    cra_lambda    = list(grid.mean(axis=1))   # mean over τ_c for each λ
    cra_threshold = list(grid.mean(axis=0))   # mean over λ for each τ_c

    best_lam_idx  = int(np.argmax(cra_lambda))
    best_tau_idx  = int(np.argmax(cra_threshold))

    return {
        "lambda_values":       lv,
        "threshold_values":    tv,
        "grid":                grid.tolist(),
        "cra_scores_lambda":   cra_lambda,
        "cra_scores_threshold":cra_threshold,
        "best_lambda":         lv[best_lam_idx],
        "best_tau":            tv[best_tau_idx],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main evaluation orchestrator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RevRAGEvaluator:
    """
    Orchestrates full paper evaluation.

    Usage:
        ev = RevRAGEvaluator(checkpoint_path=Path("outputs/checkpoints/best"),
                             results_dir=Path("outputs/results"),
                             device="cuda")
        ev.run_all()
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        results_dir:     Optional[Path] = None,
        device:          str = "cpu",
        skip_scaling:    bool = False,
        skip_ablation:   bool = False,
        max_queries:     Optional[int] = None,
    ):
        self.ckpt_path    = Path(checkpoint_path) if checkpoint_path else None
        self.results_dir  = Path(results_dir or CFG.out_dir / "results")
        self.device       = device
        self.skip_scaling = skip_scaling
        self.skip_ablation= skip_ablation
        self.max_queries  = max_queries
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _load_contractnli(self):
        from src.data.contractnli import ContractNLIDataset
        return ContractNLIDataset(
            data_dir=CFG.data.contractnli_dir,
            split="test",
            seed=CFG.data.seed,
        )

    def _load_multihop(self):
        from src.data.multihop import MultiHopRAGDataset
        return MultiHopRAGDataset(
            cache_dir=CFG.data.hf_cache_dir,
            split="test",
            max_queries=self.max_queries,
            seed=CFG.data.seed,
        )

    def _load_loco(self):
        from src.data.loco import LoCoV1Dataset
        return LoCoV1Dataset(
            cache_dir=CFG.data.hf_cache_dir,
            split="test",
            max_queries=self.max_queries,
            max_docs=5000,
            seed=CFG.data.seed,
        )

    def _load_cuad(self):
        from src.data.cuad import CUADAmendmentDataset
        return CUADAmendmentDataset(
            hf_cache_dir=CFG.data.hf_cache_dir,
            seed=CFG.data.seed,
        )

    # ── Single-dataset evaluation ─────────────────────────────────────────────

    def eval_dataset(
        self,
        dataset_name: str,
        doc_ids:    List[str],
        doc_texts:  List[str],
        queries:    List[str],
        query_ids:  List[str],
        qrels:      Dict[str, Set[str]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Runs all six models on one dataset. Returns {model_name: metrics}.
        """
        k_vals = CFG.eval.recall_k

        model_results: Dict[str, Dict[str, float]] = {}

        # ── BM25 ──────────────────────────────────────────────────────────────
        logger.info(f"  [{dataset_name}] BM25 …")
        bm25 = BM25Retriever(doc_ids, doc_texts)
        raw  = bm25.retrieve(queries, top_k=max(k_vals))
        run  = {qid: [r["doc_id"] for r in res]
                for qid, res in zip(query_ids, raw)}
        model_results["BM25"] = compute_all_metrics(run, qrels, k_vals)

        # ── BGE ───────────────────────────────────────────────────────────────
        logger.info(f"  [{dataset_name}] BGE …")
        bge = SentenceTransformerRetriever("BAAI/bge-base-en-v1.5", device=self.device)
        bge.build_index(doc_ids, doc_texts)
        raw = bge.retrieve(queries, top_k=max(k_vals))
        run = {qid: [r["doc_id"] for r in res]
               for qid, res in zip(query_ids, raw)}
        model_results["BGE"] = compute_all_metrics(run, qrels, k_vals)

        # ── BGE-M3 ────────────────────────────────────────────────────────────
        logger.info(f"  [{dataset_name}] BGE-M3 …")
        bge_m3 = SentenceTransformerRetriever("BAAI/bge-m3", device=self.device)
        bge_m3.build_index(doc_ids, doc_texts)
        raw = bge_m3.retrieve(queries, top_k=max(k_vals))
        run = {qid: [r["doc_id"] for r in res]
               for qid, res in zip(query_ids, raw)}
        model_results["BGE-M3"] = compute_all_metrics(run, qrels, k_vals)

        # ── Mamba (no verifier) ────────────────────────────────────────────────
        logger.info(f"  [{dataset_name}] Mamba (no verif.) …")
        mamba_model = build_revrag(self.ckpt_path, use_verifier=False, device=self.device)
        mamba_model.index(doc_texts, doc_ids)
        raw = mamba_model.retrieve(queries, top_k=max(k_vals))
        run = {qid: [r["doc_id"] for r in res]
               for qid, res in zip(query_ids, raw)}
        model_results["Mamba (no verif.)"] = compute_all_metrics(run, qrels, k_vals)

        # ── RevRAG (no verifier = bi-encoder only) ────────────────────────────
        # Same as above in terms of retrieval; kept for clarity in ablation
        model_results["RevRAG (no verif.)"] = model_results["Mamba (no verif.)"].copy()

        # ── RevRAG (full) ─────────────────────────────────────────────────────
        logger.info(f"  [{dataset_name}] RevRAG (full) …")
        revrag = build_revrag(self.ckpt_path, use_verifier=True, device=self.device)
        revrag.index(doc_texts, doc_ids)
        raw = revrag.retrieve(queries, top_k=max(k_vals))

        run = {qid: [r["doc_id"] for r in res]
               for qid, res in zip(query_ids, raw)}
        std_metrics = compute_all_metrics(run, qrels, k_vals)

        # RevRAG-specific metrics (CRA@1, DPR↓, ECS)
        std_metrics["cra@1"] = float(np.mean([cra_at_k(p, k=1) for p in raw]))
        std_metrics["dpr"]   = decoy_promotion_rate(raw, k=1)
        ecs_vals = []
        for passages in raw:
            nli_scores = [p.get("nli_contradiction_score", 0.0) for p in passages]
            if any(s > 0 for s in nli_scores):
                n_contra   = sum(1 for s in nli_scores if s > 0.5)
                n_passages = max(len(passages), 1)
                ecs_vals.append(1.0 - n_contra / n_passages)
        if ecs_vals:
            std_metrics["ecs"] = float(np.mean(ecs_vals))
        model_results["RevRAG"] = std_metrics

        return model_results

    # ── Full run ──────────────────────────────────────────────────────────────

    def run_all(self):
        logger.info("=" * 60)
        logger.info("RevRAG — Full Evaluation")
        logger.info("=" * 60)

        all_results: Dict[str, Dict[str, Dict[str, float]]] = {}

        # ── ContractNLI ───────────────────────────────────────────────────────
        logger.info("\n[1/3] ContractNLI …")
        cnli = self._load_contractnli()
        doc_ids, doc_texts = cnli.get_passages(window=512, stride=256)

        # Use ALL queries (both Entailment + Contradiction) for retrieval eval
        all_queries = cnli.queries
        if self.max_queries:
            all_queries = all_queries[: self.max_queries]
        queries   = [q["hypothesis"] for q in all_queries]
        query_ids = [q["query_id"]   for q in all_queries]

        # Build qrels: for each query, the gold passage IDs are those whose
        # text contains the gold character spans
        qrels: Dict[str, Set[str]] = {}
        pid_to_text = dict(zip(doc_ids, doc_texts))
        for q in all_queries:
            qid   = q["query_id"]
            spans = cnli.gold_spans.get(qid, [])
            gold  = set()
            for doc_id, cs, ce, span_text in spans:
                for pid, pt in pid_to_text.items():
                    if pid.startswith(doc_id) and span_text[:50] in pt:
                        gold.add(pid)
            qrels[qid] = gold if gold else {doc_ids[0]} if doc_ids else set()

        cnli_results = self.eval_dataset(
            "ContractNLI", doc_ids, doc_texts, queries, query_ids, qrels
        )
        all_results["ContractNLI"] = cnli_results

        # ── MultiHopRAG ───────────────────────────────────────────────────────
        logger.info("\n[2/3] MultiHopRAG …")
        mh = self._load_multihop()
        doc_ids, doc_texts = mh.get_all_passages()
        queries   = mh.get_queries()
        query_ids = [e["query_id"] for e in mh.examples]
        qrels     = mh.get_qrels()

        mh_results = self.eval_dataset(
            "MultiHopRAG", doc_ids, doc_texts, queries, query_ids, qrels
        )
        all_results["MultiHopRAG"] = mh_results

        # ── LoCoV1 ────────────────────────────────────────────────────────────
        logger.info("\n[3/3] LoCoV1 …")
        loco = self._load_loco()
        doc_ids, doc_texts = loco.get_all_passages()
        queries   = loco.get_queries()
        query_ids = [e["query_id"] for e in loco.examples]
        qrels     = loco.get_qrels()

        if queries:
            loco_results = self.eval_dataset(
                "LoCoV1", doc_ids, doc_texts, queries, query_ids, qrels
            )
            all_results["LoCoV1"] = loco_results
        else:
            logger.warning("LoCoV1 returned no queries — skipping.")

        # ── Ablation ──────────────────────────────────────────────────────────
        ablation_results = {}
        if not self.skip_ablation:
            logger.info("\n[Ablation] λ × τ_c on CUAD …")
            try:
                cuad  = self._load_cuad()
                test  = cuad.get_split("test")

                # Build corpus: each item contributes a gold doc + a decoy doc
                cuad_doc_map: Dict[str, str] = {}
                for e in test:
                    iid = e["item_id"]
                    gold_doc_id  = f"{iid}_gold"
                    decoy_doc_id = f"{iid}_decoy"
                    cuad_doc_map[gold_doc_id]  = e.get("gold_text",  "")
                    cuad_doc_map[decoy_doc_id] = e.get("decoy_text", "")

                cuad_doc_ids  = list(cuad_doc_map.keys())
                cuad_texts    = [cuad_doc_map[d] for d in cuad_doc_ids]
                cuad_queries  = [e["query"]       for e in test]
                cuad_qids     = [e["item_id"]     for e in test]
                cuad_qrels    = {
                    e["item_id"]: {f"{e['item_id']}_gold"}
                    for e in test
                }

                revrag = build_revrag(self.ckpt_path, use_verifier=True,
                                      device=self.device)
                revrag.index(cuad_texts, cuad_doc_ids)
                ablation_results = run_ablation(
                    revrag, cuad_queries, cuad_qids, cuad_qrels
                )
            except Exception as e:
                logger.warning(f"Ablation skipped: {e}")

        # ── Encoding scaling ──────────────────────────────────────────────────
        scaling_results = {}
        if not self.skip_scaling:
            logger.info("\n[Scaling] Encoding time vs sequence length …")
            try:
                scaling_results = measure_encoding_speed(device=self.device)
            except Exception as e:
                logger.warning(f"Scaling experiment skipped: {e}")

        # ── Save all results ──────────────────────────────────────────────────
        self._save_results(all_results, ablation_results, scaling_results)

        logger.info("\n✓ Evaluation complete.")
        return all_results

    def _save_results(
        self,
        main_results:    Dict,
        ablation_results: Dict,
        scaling_results:  Dict,
    ):
        # Flatten for bar chart: {model: {dataset: Recall@5}}
        bar_data: Dict[str, Dict[str, float]] = {}
        for dataset, model_dict in main_results.items():
            for model, metrics in model_dict.items():
                if model not in bar_data:
                    bar_data[model] = {}
                bar_data[model][dataset] = metrics.get("recall@5", 0.0)

        with open(self.results_dir / "main_results.json", "w") as f:
            json.dump(bar_data, f, indent=2)

        with open(self.results_dir / "full_metrics.json", "w") as f:
            json.dump(main_results, f, indent=2)

        if ablation_results:
            with open(self.results_dir / "ablation_results.json", "w") as f:
                json.dump(ablation_results, f, indent=2)

        if scaling_results:
            with open(self.results_dir / "scaling_results.json", "w") as f:
                json.dump(scaling_results, f, indent=2)

        logger.info(f"Results saved → {self.results_dir}")

        # Print summary table
        print("\n" + "=" * 70)
        print(f"{'Model':<25} " + "  ".join(f"{ds[:12]:>12}" for ds in main_results))
        print("-" * 70)
        for model in ["BM25","BGE","BGE-M3","Mamba (no verif.)","RevRAG (no verif.)","RevRAG"]:
            row = f"{model:<25}"
            for ds in main_results:
                v = main_results[ds].get(model, {}).get("recall@5", 0)
                row += f"  {v:>12.3f}"
            print(row)
        print("=" * 70)
        print("Metric: Recall@5")
