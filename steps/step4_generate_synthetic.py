"""Step 4 — Generate supplementary synthetic benchmark.

Role: diagnostic / ablation only.
NOT used for main retrieval claims — real datasets (ContractNLI, CUAD,
MultiHop-RAG, LoCoV1) are the primary evaluation.

Used in:
  - λ × τ hyperparameter ablation sweep (step 6)
  - Error analysis / failure categorisation
"""
import sys
from pathlib import Path


def run(work_dir: str, cfg: dict):
    from data.legalbench import LegalBenchRAGDataset  # type: ignore
    from data.synthetic import generate_synthetic_benchmark  # type: ignore

    out_path = Path(work_dir) / "data" / "synthetic" / "cross_page_bench.jsonl"
    if out_path.exists():
        import json
        n = sum(1 for _ in out_path.open())
        print(f"  ✓ Synthetic benchmark already exists ({n} items) — skipping generation.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    syn_cfg = cfg.get("synthetic", {})
    lb = LegalBenchRAGDataset(cfg)
    source_docs = lb.get_all_documents()
    print(f"  Source documents: {len(source_docs)}")

    items = generate_synthetic_benchmark(
        source_documents=source_docs,
        output_path=str(out_path),
        num_documents=syn_cfg.get("num_documents", 400),
        contradictions_per_doc=syn_cfg.get("contradictions_per_doc", 5),
        validation_sample=syn_cfg.get("validation_sample", 100),
        seed=42,
    )
    print(f"  ✓ {len(items)} synthetic items written → {out_path}")
    print("    (supplementary only — not used for main paper results)")
