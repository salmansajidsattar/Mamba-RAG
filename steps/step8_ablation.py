"""Step 8 — Hyperparameter ablation sweep (λ × τ_c).

99 configurations (λ ∈ {0.0,…,1.0} × τ_c ∈ {0.30,…,0.70}) evaluated on
the supplementary synthetic benchmark (max 500 items).

Outputs: outputs/ablation/{ablation_lambda.pdf, ablation_threshold.pdf,
                            ablation_heatmap.pdf, ablation_results.json}
Runtime: ~30 min on T4.
"""
import subprocess, sys
from pathlib import Path


def run(work_dir: str, device: str = "cuda", max_items: int = 500,
        skip: bool = False):
    if skip:
        print("  [skip] ablation sweep (skip_ablation=True)")
        return

    syn_path = Path(work_dir) / "data" / "synthetic" / "cross_page_bench.jsonl"
    if not syn_path.exists():
        print("  ⚠ Synthetic benchmark not found — run step 4 first.")
        return

    cmd = [
        sys.executable, "experiments/ablation_hyperparams.py",
        "--config",      "configs/default.yaml",
        "--dataset-path", str(syn_path),
        "--device",       device,
        "--max-items",    str(max_items),
        "--output-dir",  "outputs/ablation",
    ]
    result = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True,
                            timeout=3600)
    print(result.stdout[-1000:] if result.stdout else "")
    if result.returncode != 0:
        print(result.stderr[-600:])
        print("  ⚠ Ablation sweep failed.")
    else:
        print("  ✓ Ablation complete → outputs/ablation/")
