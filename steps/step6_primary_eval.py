"""Step 6 — Primary evaluation on real benchmark datasets.

All models × all real datasets × N seeds.

Estimated runtimes on T4 GPU (16 GB):
  legalbench      ~8 min / model  → ~25 min total (6 models × 3 seeds)
  contractnli     ~10 min / model → ~30 min total
  cuad_amendments ~12 min / model → ~35 min total
  multihop_rag    ~15 min / model → ~45 min total
  loco            ~20 min / model → ~60 min total
  ─────────────────────────────────────────────────────
  Total                           ~3.5 hours

TREC DL is skipped (MS MARCO 20 GB — run via slurm/eval_all.sbatch on cluster).
"""
import subprocess, sys, json
from pathlib import Path


MODELS = ["bm25", "bge", "bgem3", "mamba", "revrag_no_verifier", "revrag"]

# Primary: all real, peer-reviewed benchmark datasets
REAL_DATASETS = [
    "legalbench",        # char-level retrieval over CUAD/MAUD/ContractNLI/PrivacyQA
    "contractnli",       # expert-annotated NLI spans in real NDAs (Koreeda & Manning, 2021)
    "cuad_amendments",   # real amendment pairs from CUAD expert annotations (Hendrycks et al., 2021)
    "multihop_rag",      # peer-reviewed multi-hop retrieval (Li et al., 2024)
    "loco",              # long-context retrieval up to 32K tokens (Chang et al., 2024)
]


def _run_one(work_dir: str, model: str, dataset: str, seed: int,
             device: str, timeout: int) -> dict:
    cmd = [
        sys.executable, "evaluate/run_eval.py",
        "--config",  "configs/default.yaml",
        "--model",   model,
        "--dataset", dataset,
        "--seeds",   str(seed),
        "--device",  device,
    ]
    result = subprocess.run(
        cmd, cwd=work_dir, capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        return {"error": result.stderr[-600:]}

    # Print last 5 lines (metric summary)
    lines = result.stdout.strip().split("\n")
    for line in lines[-5:]:
        print(f"    {line}")
    return {"ok": True}


def run(work_dir: str, models=None, datasets=None, seeds=None,
        device: str = "cuda", skip_loco: bool = False,
        run_synthetic_supplementary: bool = True):

    models   = models   or MODELS
    seeds    = seeds    or [42, 123, 456]
    datasets = datasets or (
        [d for d in REAL_DATASETS if d != "loco"] if skip_loco else REAL_DATASETS
    )

    failed = []
    total  = len(models) * len(datasets) * len(seeds)
    done   = 0

    for model in models:
        for dataset in datasets:
            # Generous timeouts per dataset
            timeout = {"loco": 2400, "cuad_amendments": 1800}.get(dataset, 1200)
            for seed in seeds:
                done += 1
                tag = f"{model} | {dataset} | seed={seed}"
                print(f"\n  [{done}/{total}] {tag}")
                r = _run_one(work_dir, model, dataset, seed, device, timeout)
                if "error" in r:
                    print(f"  ❌ FAILED: {tag}")
                    print(r["error"])
                    failed.append(tag)

    print(f"\n  ✓ Primary evaluation complete: {done - len(failed)}/{done} runs OK.")
    if failed:
        print(f"  ⚠ {len(failed)} runs failed:")
        for f in failed:
            print(f"      {f}")

    # ── Supplementary synthetic eval (optional) ────────────────────────────
    if run_synthetic_supplementary:
        print("\n  Running supplementary synthetic eval (diagnostic) …")
        syn_failed = []
        for model in models:
            for seed in seeds:
                tag = f"{model} | synthetic | seed={seed}"
                print(f"    {tag}")
                r = _run_one(work_dir, model, "synthetic", seed, device, 600)
                if "error" in r:
                    syn_failed.append(tag)
        if syn_failed:
            print(f"  ⚠ {len(syn_failed)} synthetic runs failed.")
        else:
            print("  ✓ Supplementary synthetic eval complete.")
