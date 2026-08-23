"""Step 11 — Package all outputs into a zip for download.

Excludes model checkpoints (large) and raw dataset caches.
"""
import zipfile, os
from pathlib import Path
from datetime import datetime


def run(work_dir: str):
    work_dir  = Path(work_dir)
    ts        = datetime.now().strftime("%Y%m%d_%H%M")
    zip_path  = work_dir.parent / f"revrag_results_{ts}.zip"

    include_dirs = [
        work_dir / "outputs",
    ]
    exclude_patterns = {"checkpoints", "__pycache__", ".git"}

    total_files = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for inc_dir in include_dirs:
            if not inc_dir.exists():
                continue
            for fpath in inc_dir.rglob("*"):
                if any(ex in fpath.parts for ex in exclude_patterns):
                    continue
                if fpath.is_file():
                    arcname = fpath.relative_to(work_dir.parent)
                    zf.write(fpath, arcname)
                    total_files += 1

    size_mb = zip_path.stat().st_size / 1e6
    print(f"  ✓ {total_files} files packaged → {zip_path} ({size_mb:.1f} MB)")
    print("    Download this file from the Kaggle Output panel (if on Kaggle).")
    return str(zip_path)
