"""
End-to-end generation evaluation for RevRAG.

Feeds top-k retrieved passages to a language model and measures whether the
generated answer is correct, compared to a gold answer string.

Two LLM backends are supported:
  1. OpenAI API  (requires OPENAI_API_KEY; model default: gpt-4o-mini)
  2. HuggingFace (local; model default: meta-llama/Llama-3.2-3B-Instruct)

This addresses the key reviewer concern: "Does better retrieval actually
produce fewer wrong answers?"  Even a small-scale experiment (n=200 queries)
is sufficient to show the trend.

Metrics:
  - Exact Match (EM): gold_answer appears verbatim in generated response
  - Soft Match (F1): token-level F1 between gold_answer and generated response
  - Hallucination Rate: model generates the *distractor* answer when the gold
    is available in context (measures retrieval quality → generation quality link)

Usage:
    # OpenAI backend
    export OPENAI_API_KEY="sk-..."
    python evaluate/run_generation_eval.py \\
        --config configs/default.yaml \\
        --backend openai \\
        --model-name gpt-4o-mini \\
        --dataset-path data/synthetic/cross_page_bench.jsonl \\
        --retrieval-model revrag \\
        --device cuda \\
        --n-queries 200

    # HuggingFace backend (local)
    python evaluate/run_generation_eval.py \\
        --config configs/default.yaml \\
        --backend hf \\
        --model-name meta-llama/Llama-3.2-3B-Instruct \\
        --dataset-path data/synthetic/cross_page_bench.jsonl \\
        --retrieval-model revrag \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a legal contract analyst. Answer the user's question using ONLY "
    "the provided context passages. If later passages contradict earlier ones, "
    "the LATER passage is authoritative. Be concise: respond with the specific "
    "value or phrase that answers the question, not a full sentence."
)

def _build_prompt(query: str, passages: List[str]) -> str:
    context = "\n\n".join(
        f"[Passage {i+1}]\n{p}" for i, p in enumerate(passages)
    )
    return (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer (use only the context above):"
    )


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _token_f1(prediction: str, gold: str) -> float:
    pred_tokens = _tokenize(prediction)
    gold_tokens = _tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction: str, gold: str) -> float:
    """1.0 if gold appears verbatim (case-insensitive) in prediction."""
    return 1.0 if gold.lower().strip() in prediction.lower() else 0.0


def _hallucination_check(prediction: str, gold: str, distractor: str) -> float:
    """1.0 if distractor appears in prediction but gold does NOT (hallucination)."""
    pred_lower = prediction.lower()
    distractor_present = distractor.lower().strip() in pred_lower
    gold_present = gold.lower().strip() in pred_lower
    return 1.0 if (distractor_present and not gold_present) else 0.0


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

class OpenAIBackend:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai")
        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model_name = model_name

    def generate(self, query: str, passages: List[str]) -> str:
        prompt = _build_prompt(query, passages)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=100,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()


class HuggingFaceBackend:
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
                 device: str = "cpu"):
        logger.info(f"Loading LLM: {model_name}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
        )
        self.model.eval()
        self.device = device
        self.model_name = model_name

    def generate(self, query: str, passages: List[str]) -> str:
        import torch
        prompt = f"{SYSTEM_PROMPT}\n\n{_build_prompt(query, passages)}"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Retrieval model builder
# ---------------------------------------------------------------------------

def _build_retrieval_model(model_name: str, cfg: Dict, device: str):
    if model_name == "bm25":
        from models.bm25_retriever import BM25Retriever
        return BM25Retriever()
    elif model_name == "bge":
        from models.bge_retriever import BGERetriever
        return BGERetriever(device=device)
    elif model_name == "bgem3":
        from models.bgem3_retriever import BGEM3Retriever
        return BGEM3Retriever(device=device)
    elif model_name == "mamba":
        from models.mamba_retriever import MambaRetriever
        return MambaRetriever(
            checkpoint=cfg["model"]["mamba2"]["checkpoint"],
            hidden_size=cfg["model"]["mamba2"]["hidden_size"],
            max_length=cfg["model"]["mamba2"]["max_length"],
            device=device,
        )
    elif model_name == "revrag":
        from models.revrag import RevRAG
        return RevRAG(
            mamba_checkpoint=cfg["model"]["mamba2"]["checkpoint"],
            hidden_size=cfg["model"]["mamba2"]["hidden_size"],
            max_length=cfg["model"]["mamba2"]["max_length"],
            nli_checkpoint=cfg["model"]["consistency_verifier"]["checkpoint"],
            nli_contradiction_threshold=cfg["model"]["consistency_verifier"]["nli_contradiction_threshold"],
            page_order_weight=cfg["model"]["consistency_verifier"]["page_order_weight"],
            device=device,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_generation(
    items: List[Dict],
    retrieval_model,
    llm_backend,
    top_k: int = 5,
) -> Dict:
    em_scores: List[float] = []
    f1_scores: List[float] = []
    hallucination_scores: List[float] = []
    latencies: List[float] = []

    for item in tqdm(items, desc="Generation eval"):
        doc_text = item.get("document", "")
        if not doc_text:
            continue
        query = item["query"]
        gold = item["gold_answer"]
        distractor = item.get("distractor_answer", "")

        # Retrieve passages
        t0 = time.perf_counter()
        try:
            results = retrieval_model.retrieve(query, top_k=top_k)
            passages = [r.text if hasattr(r, "text") else str(r) for r in results[:top_k]]
        except Exception as e:
            logger.warning(f"Retrieval failed for item {item.get('item_id')}: {e}")
            continue

        # Generate answer
        try:
            generated = llm_backend.generate(query, passages)
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            continue

        latencies.append((time.perf_counter() - t0) * 1000)

        em = _exact_match(generated, gold)
        f1 = _token_f1(generated, gold)
        hal = _hallucination_check(generated, gold, distractor) if distractor else 0.0

        em_scores.append(em)
        f1_scores.append(f1)
        hallucination_scores.append(hal)

        logger.debug(
            f"[{item.get('item_id', '?')}] gold={gold!r}  "
            f"gen={generated[:60]!r}  EM={em:.0f}  F1={f1:.3f}  HAL={hal:.0f}"
        )

    return {
        "n": len(em_scores),
        "exact_match": float(np.mean(em_scores)) if em_scores else 0.0,
        "token_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "hallucination_rate": float(np.mean(hallucination_scores)) if hallucination_scores else 0.0,
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end generation evaluation for RevRAG."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--dataset-path",
        default="data/synthetic/cross_page_bench.jsonl",
    )
    parser.add_argument(
        "--retrieval-model",
        choices=["bm25", "bge", "bgem3", "mamba", "revrag"],
        default="revrag",
    )
    parser.add_argument("--backend", choices=["openai", "hf"], default="openai")
    parser.add_argument("--model-name", default="gpt-4o-mini",
                        help="LLM model name (OpenAI model ID or HF model ID).")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--n-queries", type=int, default=200,
                        help="Max queries to evaluate.")
    parser.add_argument("--output-dir", default="outputs/generation_eval")
    parser.add_argument("--compare-all", action="store_true",
                        help="Run all 5 retrieval models and compare side-by-side.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load items
    items = []
    with open(args.dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if args.n_queries and len(items) > args.n_queries:
        import random
        random.seed(42)
        items = random.sample(items, args.n_queries)
    logger.info(f"Evaluating on {len(items)} items.")

    # Build LLM backend
    if args.backend == "openai":
        llm = OpenAIBackend(model_name=args.model_name)
    else:
        llm = HuggingFaceBackend(model_name=args.model_name, device=args.device)

    models_to_run = (
        ["bm25", "bge", "bgem3", "mamba", "revrag"]
        if args.compare_all
        else [args.retrieval_model]
    )

    all_results = {}
    for model_name in models_to_run:
        logger.info(f"\n{'='*50}\nEvaluating retrieval model: {model_name}\n{'='*50}")
        retriever = _build_retrieval_model(model_name, cfg, args.device)

        # Index documents (unique docs in the item set)
        doc_map = {}
        for item in items:
            doc_id = item["doc_id"]
            doc_text = item.get("document", "")
            if doc_id not in doc_map and doc_text:
                doc_map[doc_id] = doc_text

        if hasattr(retriever, "index"):
            doc_ids = list(doc_map.keys())
            doc_texts = [doc_map[d] for d in doc_ids]
            retriever.index(doc_texts, doc_ids)

        result = evaluate_generation(
            items=items,
            retrieval_model=retriever,
            llm_backend=llm,
            top_k=args.top_k,
        )
        result["retrieval_model"] = model_name
        result["llm_backend"] = args.backend
        result["llm_model"] = args.model_name
        all_results[model_name] = result

        logger.info(
            f"{model_name}: EM={result['exact_match']:.4f}  "
            f"F1={result['token_f1']:.4f}  "
            f"HAL={result['hallucination_rate']:.4f}"
        )

    # Save
    output_file = output_dir / f"generation_eval_{args.backend}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved → {output_file}")

    # Print comparison table
    print("\n" + "=" * 65)
    print(f"{'Model':<12} {'EM':>8} {'F1':>8} {'Hallucination':>14} {'N':>6}")
    print("-" * 65)
    for name, r in all_results.items():
        print(
            f"{name:<12} {r['exact_match']:>8.4f} {r['token_f1']:>8.4f} "
            f"{r['hallucination_rate']:>14.4f} {r['n']:>6}"
        )
    print("=" * 65)


if __name__ == "__main__":
    main()
