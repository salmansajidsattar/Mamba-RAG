"""
package_for_kaggle.py
=====================
Run this on your LOCAL machine to create a zip ready for Kaggle upload.

Usage:
    cd F:\MambaRAG
    python package_for_kaggle.py

Output:
    F:\MambaRAG\revrag_kaggle_data.zip   (~400-500 MB)

Then upload that zip to Kaggle:
    kaggle.com -> Datasets -> New Dataset -> Upload -> revrag_kaggle_data.zip
    Name it: revrag-datasets
"""
import os, zipfile
from pathlib import Path

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_ZIP  = ROOT / "revrag_kaggle_data.zip"

# Files to include
INCLUDE = [
    # ContractNLI - only the 3 JSON files (not the whole git repo)
    ("data/contractnli/resources/contract-nli/train.json",
     "contractnli/train.json"),
    ("data/contractnli/resources/contract-nli/dev.json",
     "contractnli/dev.json"),
    ("data/contractnli/resources/contract-nli/test.json",
     "contractnli/test.json"),
]

# HF cache dirs to include entirely
HF_DIRS = [
    "data/hf_cache/datasets--theatticusproject--cuad",
    "data/hf_cache/datasets--yixuantt--MultiHopRAG",
    "data/hf_cache/datasets--hazyresearch--LoCoV1-Documents",
    "data/hf_cache/datasets--hazyresearch--LoCoV1-Queries",
]

print(f"Packaging RevRAG datasets -> {OUT_ZIP}")
print(f"Source: {DATA_DIR}")

total_files = 0
total_bytes = 0

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED,
                     compresslevel=1) as zf:

    # ContractNLI JSON files
    print("\nAdding ContractNLI ...")
    for src_rel, arc_name in INCLUDE:
        src = ROOT / src_rel
        if src.exists():
            zf.write(src, arc_name)
            total_files += 1
            total_bytes += src.stat().st_size
            print(f"  + {arc_name}  ({src.stat().st_size/1e6:.1f} MB)")
        else:
            print(f"  MISSING: {src}")

    # HuggingFace cache dirs
    for hf_dir_rel in HF_DIRS:
        hf_dir = ROOT / hf_dir_rel
        if not hf_dir.exists():
            print(f"\nSkipping (not found): {hf_dir_rel}")
            continue
        dir_name = hf_dir.name
        print(f"\nAdding {dir_name} ...")
        n = 0
        for fpath in hf_dir.rglob("*"):
            if fpath.is_file():
                arc_path = f"hf_cache/{dir_name}/{fpath.relative_to(hf_dir)}"
                zf.write(fpath, arc_path)
                total_files += 1
                total_bytes += fpath.stat().st_size
                n += 1
        print(f"  {n} files added")

print(f"\nDone!")
print(f"  Total files : {total_files}")
print(f"  Source size : {total_bytes/1e6:.0f} MB")
print(f"  Zip size    : {OUT_ZIP.stat().st_size/1e6:.0f} MB")
print(f"  Output      : {OUT_ZIP}")
print()
print("Next steps:")
print("  1. Go to https://kaggle.com/datasets")
print("  2. Click 'New Dataset'")
print("  3. Upload revrag_kaggle_data.zip")
print("  4. Name it exactly:  revrag-datasets")
print("  5. Make it Public (or Private)")
print("  6. In your Kaggle notebook, click 'Add Data' -> search 'revrag-datasets'")
print("  7. Run:  !python kaggle_setup_from_dataset.py")
