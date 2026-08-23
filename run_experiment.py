"""
run_experiment.py — RevRAG experiment pipeline
================================================
Run with:
    python run_experiment.py

All settings live in config.py — edit that file, not this one.
This file only wires steps together.
"""
import sys
import time
from pathlib import Path

# ── Make project root importable ─────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "revrag"))

from config import cfg           # all settings in one place
from steps import (              # one module per step
    step1_verify_gpu,
    step2_install_deps,
    step3_download_datasets,
    step4_generate_synthetic,
    step5_smoke_test,
    step6_primary_eval,
    step7_aggregate,
    step8_ablation,
    step9_generation_eval,
    step10_plots,
    step11_package,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _section(n: int, title: str):
    bar = "━" * 62
    print(f"\n{bar}\n  STEP {n}: {title}\n{bar}")


def _elapsed(t0: float) -> str:
    s = int(time.time() - t0)
    return f"{s // 60}m {s % 60}s"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    t_start = time.time()
    work    = str(cfg.work_dir)

    # ── 1. GPU ────────────────────────────────────────────────────────────────
    _section(1, "Verify GPU / CUDA")
    t = time.time()
    cuda_ok = step1_verify_gpu.run()
    device  = cfg.env.device if cuda_ok else "cpu"
    print(f"  [{_elapsed(t)}]  device = {device}")

    # ── 2. Deps ───────────────────────────────────────────────────────────────
    _section(2, "Install dependencies")
    t = time.time()
    step2_install_deps.run(skip=cfg.env.skip_install)
    print(f"  [{_elapsed(t)}]")

    # ── 3. Datasets ───────────────────────────────────────────────────────────
    _section(3, "Download benchmark datasets")
    t = time.time()
    step3_download_datasets.run(work, skip_loco=cfg.datasets.skip_loco)
    print(f"  [{_elapsed(t)}]")

    # ── 4. Synthetic (supplementary) ─────────────────────────────────────────
    _section(4, "Generate supplementary synthetic benchmark")
    t = time.time()
    step4_generate_synthetic.run(work, vars(cfg.synthetic))
    print(f"  [{_elapsed(t)}]")

    # ── 5. Smoke test ─────────────────────────────────────────────────────────
    _section(5, "Smoke test")
    t = time.time()
    step5_smoke_test.run(work)
    print(f"  [{_elapsed(t)}]")

    # ── 6. Primary eval ───────────────────────────────────────────────────────
    _section(6, "Primary evaluation (real benchmark datasets)")
    datasets = [
        d for d in cfg.datasets.primary
        if not (cfg.datasets.skip_loco and d == "loco")
    ]
    print(f"  Models:   {cfg.eval.models}")
    print(f"  Datasets: {datasets}")
    print(f"  Seeds:    {cfg.eval.seeds}")
    t = time.time()
    step6_primary_eval.run(
        work,
        models=cfg.eval.models,
        datasets=datasets,
        seeds=cfg.eval.seeds,
        device=device,
        run_synthetic_supplementary=cfg.eval.run_synthetic,
    )
    print(f"  [{_elapsed(t)}]")

    # ── 7. Aggregate ──────────────────────────────────────────────────────────
    _section(7, "Aggregate results (mean ± std)")
    t = time.time()
    step7_aggregate.run(work)
    print(f"  [{_elapsed(t)}]")

    # ── 8. Ablation ───────────────────────────────────────────────────────────
    _section(8, "Hyperparameter ablation sweep (λ × τ_c)")
    t = time.time()
    step8_ablation.run(
        work,
        device=device,
        max_items=cfg.ablation.max_items,
        skip=cfg.ablation.skip,
    )
    print(f"  [{_elapsed(t)}]")

    # ── 9. Generation eval (free local LLM) ──────────────────────────────────
    _section(9, f"Generation eval — {cfg.generation.model_name}")
    print(f"  backend:  HuggingFace (local, no API key)")
    print(f"  model:    {cfg.generation.model_name}")
    print(f"  4-bit:    {cfg.generation.load_in_4bit}")
    t = time.time()
    step9_generation_eval.run(
        work,
        cfg=cfg.generation,
        skip=cfg.generation.skip,
    )
    print(f"  [{_elapsed(t)}]")

    # ── 10. Figures ───────────────────────────────────────────────────────────
    _section(10, "Generate paper figures")
    t = time.time()
    step10_plots.run(work, skip=cfg.plots.skip)
    print(f"  [{_elapsed(t)}]")

    # ── 11. Package ───────────────────────────────────────────────────────────
    _section(11, "Package outputs")
    t = time.time()
    step11_package.run(work)
    print(f"  [{_elapsed(t)}]")

    # ── Done ──────────────────────────────────────────────────────────────────
    print(f"\n{'━'*62}")
    print(f"  ✓ Pipeline complete in {_elapsed(t_start)}")
    print(f"  Results → {cfg.out_dir}")
    print(f"{'━'*62}")


if __name__ == "__main__":
    main()
