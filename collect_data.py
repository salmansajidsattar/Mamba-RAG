import os
import shutil
import zipfile
import time
from pathlib import Path

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def section(title):
    print(f"\n{'─'*55}\n  {title}\n{'─'*55}")


def copy_dir(src: Path, dst: Path, label: str):
    if not src.exists():
        print(f"  ⚠ Not found: {src}")
        return
    if dst.exists():
        print(f"  ✓ {label} (already in data/)")
        return
    print(f"  Copying {label} …", end=" ", flush=True)
    shutil.copytree(str(src), str(dst))
    size_mb = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e6
    print(f"✓  ({size_mb:.0f} MB)")


# ── 1. Git-cloned datasets ────────────────────────────────────────────────────
section("1. Git-cloned datasets")

# LegalBench-RAG — could be in revrag/data/raw/ or downloaded fresh
lb_src = ROOT / "revrag" / "data" / "raw" / "legalbench_rag"
if not lb_src.exists():
    print("  LegalBench-RAG not found — cloning now …")
    import subprocess
    lb_src.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([
        "git", "clone", "--depth=1",
        "https://github.com/zeroentropy-ai/legalbenchrag",
        str(lb_src),
    ])
copy_dir(lb_src, DATA_DIR / "legalbench_rag", "LegalBench-RAG")

# ContractNLI
cnli_src = ROOT / "revrag" / "data" / "raw" / "contractnli"
if not cnli_src.exists():
    print("  ContractNLI not found — cloning now …")
    import subprocess
    cnli_src.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([
        "git", "clone", "--depth=1",
        "https://github.com/stanfordnlp/contract-nli",
        str(cnli_src),
    ])
copy_dir(cnli_src, DATA_DIR / "contractnli", "ContractNLI")


# ── 2. HuggingFace cached datasets ───────────────────────────────────────────
section("2. HuggingFace cache → data/hf_cache/")

# Map: HF cache folder name → friendly label
HF_DATASETS = {
    "datasets--theatticusproject--cuad":          "CUAD",
    "datasets--yixuantt--MultiHopRAG":            "MultiHop-RAG",
    "datasets--hazyresearch--LoCoV1-Documents":   "LoCoV1-Documents",
    "datasets--hazyresearch--LoCoV1-Queries":     "LoCoV1-Queries",
}

if not HF_CACHE.exists():
    print(f"  ⚠ HuggingFace cache not found at {HF_CACHE}")
    print("    Run download_datasets_local.py first.")
else:
    hf_dst_root = DATA_DIR / "hf_cache"
    hf_dst_root.mkdir(exist_ok=True)
    for folder_name, label in HF_DATASETS.items():
        src = HF_CACHE / folder_name
        dst = hf_dst_root / folder_name
        copy_dir(src, dst, label)


# ── 3. Summary ────────────────────────────────────────────────────────────────
section("3. Contents of data/")
total_mb = 0
for item in sorted(DATA_DIR.rglob("*")):
    if item.is_file():
        total_mb += item.stat().st_size / 1e6
top_level = [p.name for p in DATA_DIR.iterdir()]
print(f"  Folders: {top_level}")
print(f"  Total size: {total_mb:.0f} MB")


# ── 4. Zip for Kaggle ─────────────────────────────────────────────────────────
section("4. Creating revrag_datasets.zip …")
t        = time.time()
zip_path = ROOT / "revrag_datasets.zip"
EXCLUDE  = {".git", "__pycache__", "*.pyc"}

count = 0
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for fpath in DATA_DIR.rglob("*"):
        if any(ex.lstrip("*") in str(fpath) for ex in EXCLUDE):
            continue
        if fpath.is_file():
            arcname = fpath.relative_to(ROOT)
            zf.write(fpath, arcname)
            count += 1

size_mb = zip_path.stat().st_size / 1e6
elapsed = int(time.time() - t)
print(f"  ✓ {count} files → revrag_datasets.zip  "
      f"({size_mb:.0f} MB)  [{elapsed}s]")

print(f"""
  Next steps:
  ─────────────────────────────────────────────────
  1. Go to kaggle.com → Datasets → New Dataset
  2. Upload: {zip_path}
  3. Name it: revrag-datasets  (set to Private)
  4. In your Kaggle notebook:
       Add Data → Your Datasets → revrag-datasets
  5. Data will be at: /kaggle/input/revrag-datasets/data/
  ─────────────────────────────────────────────────
""")
