"""Step 5 — CPU smoke test.

Always runs on CPU regardless of the DEVICE setting in config.py.
This is intentional — the smoke test must pass BEFORE committing GPU time.

Exit behaviour:
  sys.exit(0) → all required checks passed → proceed to GPU eval
  sys.exit(1) → a required check failed    → pipeline aborts here

Tests run (< 2 min on CPU):
  config YAML, all dataset loaders, BM25, BGE, Mamba stub,
  RevRAG pipeline, NLI verifier, Char-overlap, CRA@1, Recall@k
"""
import subprocess
import sys
from pathlib import Path


def run(work_dir: str):
    test_script = Path(work_dir) / "tests" / "smoke_test.py"

    if not test_script.exists():
        print(f"  ⚠ {test_script} not found — skipping smoke test.")
        return

    print("  Running on CPU (device=cpu is always used for smoke test) …\n")

    result = subprocess.run(
        [sys.executable, str(test_script)],
        cwd=work_dir,
        capture_output=False,   # stream output directly so user sees progress
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "\n  ⛔ Smoke test FAILED — required checks did not pass.\n"
            "  Fix the errors above before running GPU eval.\n"
            "  (GPU eval would waste session time on a broken pipeline.)"
        )
    # exit 0 = all required checks passed (optional ones may have failed)
