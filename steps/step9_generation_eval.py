"""Step 9 — End-to-end generation evaluation (free, no API key needed).

Uses a local HuggingFace LLM (phi-3-mini by default) to answer questions
using retrieved passages as context.

Metrics:
  Exact Match (EM)      — strict string match after normalisation
  Token F1              — token overlap between prediction and gold answer
  Hallucination Rate    — fraction where top-1 retrieved = distractor (not gold)

All models are compared side-by-side on the same N queries.

Model options (set in config.py → GenerationConfig.model_key):
  "phi3"    → microsoft/phi-3-mini-4k-instruct   ~8 GB VRAM  (default)
  "mistral" → mistralai/Mistral-7B-Instruct-v0.2 ~14 GB VRAM
  "flan"    → google/flan-t5-large               ~3 GB VRAM
  "smollm"  → HuggingFaceTB/SmolLM2-1.7B-Instruct ~4 GB VRAM
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Attempt to use the unified CLI runner first; fall back to inline if missing
_CLI = "evaluate/run_generation_eval.py"


def run(work_dir: str, cfg=None, skip: bool = False):
    """
    Parameters
    ----------
    work_dir : str
        Path to the revrag/ package directory.
    cfg : GenerationConfig | None
        If None, imports from config.py automatically.
    skip : bool
        Pass True to skip this step entirely.
    """
    if skip:
        print("  [skip] generation eval (skip=True in config)")
        return

    # ── Resolve config ─────────────────────────────────────────────────────
    if cfg is None:
        sys.path.insert(0, str(Path(work_dir).parent))
        from config import cfg as _cfg
        cfg = _cfg.generation

    syn_path = Path(work_dir) / "data" / "synthetic" / "cross_page_bench.jsonl"
    if not syn_path.exists():
        print("  ⚠ Synthetic benchmark not found — run step 4 first.")
        return

    # ── Try CLI runner ─────────────────────────────────────────────────────
    cli_path = Path(work_dir) / _CLI
    if cli_path.exists():
        _run_cli(work_dir, cfg, syn_path)
    else:
        print(f"  ⚠ {_CLI} not found — running inline generation eval …")
        _run_inline(work_dir, cfg, syn_path)


# ── CLI path ───────────────────────────────────────────────────────────────

def _run_cli(work_dir: str, cfg, syn_path: Path):
    cmd = [
        sys.executable, _CLI,
        "--config",       "configs/default.yaml",
        "--dataset-path", str(syn_path),
        "--backend",      "hf",
        "--model-name",   cfg.model_name,
        "--n-queries",    str(cfg.n_queries),
        "--compare-all",
        "--device",       "cuda",
        "--output-dir",   str(cfg.output_dir),
    ]
    if cfg.load_in_4bit:
        cmd.append("--load-in-4bit")

    result = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True,
                            timeout=7200)
    print(result.stdout[-2000:] if result.stdout else "")
    if result.returncode != 0:
        print(result.stderr[-800:])
        print("  ⚠ Generation eval CLI failed — see above.")
    else:
        print(f"  ✓ Generation eval complete → {cfg.output_dir}")


# ── Inline fallback (no CLI script needed) ────────────────────────────────

def _run_inline(work_dir: str, cfg, syn_path: Path):
    """Minimal inline generation eval using backends/hf_llm.py."""
    import sys, json, re
    from pathlib import Path
    from collections import Counter

    sys.path.insert(0, str(Path(work_dir).parent))
    sys.path.insert(0, work_dir)

    from backends.hf_llm import HFLLMBackend      # type: ignore
    from evaluate.run_eval import build_model       # type: ignore

    print(f"  LLM: {cfg.model_name}")
    llm = HFLLMBackend(
        model_name=cfg.model_name,
        device="cuda",
        load_in_4bit=cfg.load_in_4bit,
        max_new_tokens=cfg.max_new_tokens,
    )

    # Load dataset
    items = []
    with open(syn_path) as f:
        for line in f:
            items.append(json.loads(line))
    items = items[: cfg.n_queries]

    import yaml
    with open(Path(work_dir) / "configs" / "default.yaml") as f:
        yaml_cfg = yaml.safe_load(f)

    MODELS = ["bm25", "bge", "bgem3", "mamba", "revrag_no_verifier", "revrag"]
    results = {}

    for model_name in MODELS:
        print(f"    Evaluating retrieval model: {model_name} …")
        retriever = build_model(model_name, yaml_cfg, "cuda")

        em_scores, f1_scores, halluc = [], [], []
        for item in items:
            query   = item["query"]
            gold    = item.get("gold_answer", "")
            passages = retriever.retrieve(query, top_k=5)

            context = "\n\n".join(
                f"[Passage {i+1}] {p['text']}"
                for i, p in enumerate(passages)
            )
            pred = llm.generate(query, context)

            em_scores.append(int(_normalize(pred) == _normalize(gold)))
            f1_scores.append(_token_f1(_normalize(pred), _normalize(gold)))

            # Hallucination: top-1 is a distractor (decoy text in item)
            top1_text = passages[0]["text"] if passages else ""
            decoy     = item.get("decoy_text", "")
            gold_text = item.get("gold_text", "")
            if decoy and gold_text:
                is_halluc = (
                    _overlap(top1_text, decoy) > 0.5 and
                    _overlap(top1_text, gold_text) <= 0.5
                )
                halluc.append(int(is_halluc))

        n = len(em_scores)
        results[model_name] = {
            "em":              sum(em_scores) / n,
            "token_f1":        sum(f1_scores) / n,
            "hallucination":   sum(halluc) / len(halluc) if halluc else None,
            "n_queries":       n,
        }
        r = results[model_name]
        print(f"      EM={r['em']:.3f}  F1={r['token_f1']:.3f}  "
              f"Halluc={r['hallucination']}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_file = cfg.output_dir / "generation_results.json"
    with open(out_file, "w") as f:
        json.dump({"model": cfg.model_name, "results": results}, f, indent=2)
    print(f"  ✓ Generation eval complete → {out_file}")


# ── Helpers ────────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    import re
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def _token_f1(pred: str, gold: str) -> float:
    from collections import Counter
    p_toks = pred.split()
    g_toks = gold.split()
    if not p_toks or not g_toks:
        return 0.0
    common = Counter(p_toks) & Counter(g_toks)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(p_toks)
    recall    = n_common / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def _overlap(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    return len(set_a & set_b) / max(len(set_a), len(set_b))
