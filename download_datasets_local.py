"""
download_datasets_local.py — Download all benchmark datasets on CPU locally.

Run this ONCE on your machine, then zip and upload to Kaggle.
No GPU required. Internet required.

Usage:
    python download_datasets_local.py

Output:
    revrag/data/raw/          ← all datasets cached here
    revrag_datasets.zip       ← ready to upload to Kaggle
"""
import os
import sys
from pathlib import Path

# Force HuggingFace to cache inside the project folder, not C:\Users\...\.cache
# Must be set BEFORE importing datasets or huggingface_hub
_HF_CACHE = str(Path(__file__).parent / "revrag" / "data" / "raw" / "hf_cache")
os.environ["HF_HOME"]                        = _HF_CACHE
os.environ["HF_DATASETS_CACHE"]              = _HF_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"]          = _HF_CACHE
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import subprocess
import zipfile
import time
from pathlib import Path

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "revrag" / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def section(title):
    print(f"\n{'─'*55}\n  {title}\n{'─'*55}")


def elapsed(t0):
    s = int(time.time() - t0)
    return f"{s//60}m {s%60}s"


# ── 1. LegalBench-RAG ─────────────────────────────────────────────────────────
section("1/5  LegalBench-RAG (git clone ~50 MB)")
t = time.time()
lb_dir = DATA_DIR / "legalbench_rag"
if not lb_dir.exists():
    subprocess.check_call([
        "git", "clone", "--depth=1",
        "https://github.com/zeroentropy-ai/legalbenchrag",
        str(lb_dir),
    ])
    print(f"  ✓ LegalBench-RAG  [{elapsed(t)}]")
else:
    print("  ✓ LegalBench-RAG (already exists)")


# ── 2. ContractNLI ────────────────────────────────────────────────────────────
section("2/5  ContractNLI — Stanford NLP GitHub (git clone ~50 MB)")
t = time.time()
cnli_dir = DATA_DIR / "contractnli"
if not cnli_dir.exists():
    subprocess.check_call([
        "git", "clone", "--depth=1",
        "https://github.com/stanfordnlp/contract-nli",
        str(cnli_dir),
    ])
    # Verify data files
    missing = [f for f in ["train.json", "dev.json", "test.json"]
               if not (cnli_dir / "data" / f).exists()]
    if missing:
        print(f"  ⚠ Data files not in repo: {missing}")
        print("    Download manually from https://stanfordnlp.github.io/contract-nli/")
    else:
        print(f"  ✓ ContractNLI  [{elapsed(t)}]")
else:
    print("  ✓ ContractNLI (already exists)")


# ── 3. CUAD ───────────────────────────────────────────────────────────────────
section("3/5  CUAD — theatticusproject/cuad (~300 MB)")
t = time.time()
cuad_dir = DATA_DIR / "cuad"
if not cuad_dir.exists():
    from datasets import load_dataset
    load_dataset("theatticusproject/cuad", cache_dir=str(cuad_dir))
    print(f"  ✓ CUAD  [{elapsed(t)}]")
else:
    print("  ✓ CUAD (already exists)")


# ── 4. MultiHop-RAG ───────────────────────────────────────────────────────────
section("4/5  MultiHop-RAG — yixuantt/MultiHopRAG (~200 MB)")
t = time.time()
mh_dir = DATA_DIR / "multihop_rag"
if not mh_dir.exists():
    from datasets import load_dataset
    load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train", cache_dir=str(mh_dir))
    print(f"  ✓ MultiHop-RAG  [{elapsed(t)}]")
else:
    print("  ✓ MultiHop-RAG (already exists)")


# ── 5. LoCoV1 (optional — large) ──────────────────────────────────────────────
section("5/5  LoCoV1 — hazyresearch/LoCoV1 (~500 MB, optional)")
SKIP_LOCO = "--skip-loco" in sys.argv
if SKIP_LOCO:
    print("  ─ Skipped (pass without --skip-loco to include)")
else:
    t = time.time()
    loco_dir = DATA_DIR / "loco_v1"
    if not loco_dir.exists():
        from datasets import load_dataset
        load_dataset("hazyresearch/LoCoV1-Documents", cache_dir=str(loco_dir / "documents"))
        load_dataset("hazyresearch/LoCoV1-Queries",   cache_dir=str(loco_dir / "queries"))
        print(f"  ✓ LoCoV1  [{elapsed(t)}]")
    else:
        print("  ✓ LoCoV1 (already exists)")


# ── Zip for Kaggle upload ─────────────────────────────────────────────────────
section("Zipping for Kaggle upload …")
t       = time.time()
zip_out = ROOT / "revrag_datasets.zip"
EXCLUDE = {"__pycache__", ".git", "*.pyc"}

total = 0
with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for fpath in DATA_DIR.rglob("*"):
        if any(ex.lstrip("*") in str(fpath) for ex in EXCLUDE):
            continue
        if fpath.is_file():
            arcname = fpath.relative_to(ROOT)
            zf.write(fpath, arcname)
            total += 1

size_mb = zip_out.stat().st_size / 1e6
print(f"  ✓ {total} files → {zip_out.name}  ({size_mb:.0f} MB)  [{elapsed(t)}]")
print(f"\n  Next steps:")
print(f"  1. Go to kaggle.com → Datasets → New Dataset")
print(f"  2. Upload: {zip_out}")
print(f"  3. Name it: revrag-datasets  (private)")
print(f"  4. In your Kaggle notebook, it will be at:")
print(f"     /kaggle/input/revrag-datasets/revrag/data/raw/")
