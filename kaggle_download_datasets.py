"""
kaggle_download_datasets.py
Run this first in your Kaggle notebook to download all RevRAG datasets.
Usage:  !python kaggle_download_datasets.py
"""
import os, shutil, subprocess, sys, zipfile
from pathlib import Path

WORKING_DIR  = Path("/kaggle/working")
DATA_DIR     = WORKING_DIR / "data"
HF_CACHE_DIR = DATA_DIR / "hf_cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"]                        = str(HF_CACHE_DIR)
os.environ["HF_DATASETS_CACHE"]              = str(HF_CACHE_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"]          = str(HF_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"]             = str(HF_CACHE_DIR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

print("=" * 60)
print("RevRAG Dataset Downloader - Kaggle Edition")
print(f"Data directory: {DATA_DIR}")
print("=" * 60)


# ---------------------------------------------------------------------------
# 0. Install dependencies
# ---------------------------------------------------------------------------
def install_deps():
    print("\n[0/4] Installing dependencies ...")
    pkgs = [
        "datasets", "huggingface_hub", "transformers",
        "torch", "faiss-cpu", "rank_bm25",
        "sentence_transformers", "matplotlib", "numpy",
        "scikit-learn", "requests",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install"] + pkgs + ["-q"], check=False)
    print("  OK")


# ---------------------------------------------------------------------------
# 1. ContractNLI  — branch is "master" (not "main")
# ---------------------------------------------------------------------------
def download_contractnli():
    import requests
    print("\n[1/4] ContractNLI ...")

    json_root = DATA_DIR / "contractnli" / "resources" / "contract-nli"
    json_root.mkdir(parents=True, exist_ok=True)

    if (json_root / "train.json").exists():
        print(f"  Already exists at {json_root}")
        _verify_cnli(json_root)
        return

    HEADERS = {"User-Agent": "RevRAG/1.0"}

    # Method A: individual JSON files — branch = master
    RAW_BASE = "https://raw.githubusercontent.com/stanfordnlp/contract-nli/master/resources/contract-nli"
    print("  Method A: downloading JSON files (branch=master) ...")
    for split in ("train", "dev", "test"):
        url = f"{RAW_BASE}/{split}.json"
        out = json_root / f"{split}.json"
        try:
            print(f"    {split}.json ...", end=" ", flush=True)
            r = requests.get(url, headers=HEADERS, timeout=180)
            r.raise_for_status()
            out.write_bytes(r.content)
            print(f"OK ({len(r.content)/1e6:.1f} MB)")
        except Exception as e:
            print(f"FAILED: {e}")

    if (json_root / "train.json").exists():
        _verify_cnli(json_root)
        return

    # Method B: zip — branch = master
    print("  Method B: zip download (branch=master) ...")
    ZIP_URL  = "https://github.com/stanfordnlp/contract-nli/archive/refs/heads/master.zip"
    zip_path = DATA_DIR / "cnli.zip"
    dest_dir = DATA_DIR / "contractnli"
    try:
        print("    Downloading zip ...", end=" ", flush=True)
        r = requests.get(ZIP_URL, headers=HEADERS, stream=True, timeout=300, allow_redirects=True)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
        print(f"OK ({zip_path.stat().st_size/1e6:.1f} MB)")

        print("    Extracting ...", end=" ", flush=True)
        tmp = DATA_DIR / "_cnli_tmp"
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp)
        subdirs = [d for d in tmp.iterdir() if d.is_dir()]
        if subdirs:
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            shutil.move(str(subdirs[0]), str(dest_dir))
            print(f"OK -> {dest_dir}")
        shutil.rmtree(tmp, ignore_errors=True)
        zip_path.unlink(missing_ok=True)
    except Exception as e:
        print(f"FAILED: {e}")

    _verify_cnli(json_root)


def _verify_cnli(json_root):
    import json
    for split in ("train", "dev", "test"):
        p = json_root / f"{split}.json"
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            print(f"  OK {split}.json - {len(data.get('documents', []))} documents")
        else:
            print(f"  MISSING {split}.json")


# ---------------------------------------------------------------------------
# 2. CUAD  — no trust_remote_code, verification_mode=no_checks
# ---------------------------------------------------------------------------
def download_cuad():
    print("\n[2/4] CUAD (theatticusproject/cuad) ...")
    from datasets import load_dataset
    try:
        ds = load_dataset(
            "theatticusproject/cuad",
            split="train",
            cache_dir=str(HF_CACHE_DIR),
            verification_mode="no_checks",
        )
        print(f"  OK - {len(ds)} examples")
    except Exception as e:
        print(f"  FAILED: {e}")


# ---------------------------------------------------------------------------
# 3. MultiHopRAG  — no trust_remote_code
# ---------------------------------------------------------------------------
def download_multihop():
    print("\n[3/4] MultiHopRAG (yixuantt/MultiHopRAG) ...")
    from datasets import load_dataset
    try:
        ds = load_dataset(
            "yixuantt/MultiHopRAG",
            "MultiHopRAG",
            split="train",
            cache_dir=str(HF_CACHE_DIR),
        )
        print(f"  OK - {len(ds)} examples")
    except Exception as e:
        print(f"  FAILED: {e}")


# ---------------------------------------------------------------------------
# 4. LoCoV1  — only "test" split exists on HuggingFace
# ---------------------------------------------------------------------------
def download_loco():
    print("\n[4/4] LoCoV1 ...")
    from datasets import load_dataset
    for repo in ["hazyresearch/LoCoV1-Documents", "hazyresearch/LoCoV1-Queries"]:
        try:
            ds = load_dataset(repo, split="test", cache_dir=str(HF_CACHE_DIR))
            print(f"  OK {repo} [test] - {len(ds)} rows")
        except Exception as e:
            print(f"  FAILED {repo}: {e}")


# ---------------------------------------------------------------------------
# Write kaggle_config.py
# ---------------------------------------------------------------------------
def write_kaggle_config():
    lines = [
        "# kaggle_config.py - auto-generated",
        "# Usage: import kaggle_config  (before from src.config import CFG)",
        "import src.config as _cfg",
        "from pathlib import Path",
        f'_DATA = Path("{DATA_DIR}")',
        f'_HF   = Path("{HF_CACHE_DIR}")',
        '_CNLI = _DATA / "contractnli" / "resources" / "contract-nli"',
        "_cfg.CFG.data.data_dir        = _DATA",
        "_cfg.CFG.data.hf_cache        = _HF",
        "_cfg.CFG.data.hf_cache_dir    = _HF",
        "_cfg.CFG.data.contractnli_dir = _CNLI",
        "_cfg.CFG.data.cuad_cache      = _HF",
        "_cfg.CFG.data.multihop_cache  = _HF",
        "_cfg.CFG.data.loco_cache      = _HF",
        '_cfg.CFG.out_dir              = Path("/kaggle/working/outputs")',
        "_cfg.CFG.out_dir.mkdir(parents=True, exist_ok=True)",
        'print("[kaggle_config] Paths configured for Kaggle.")',
    ]
    out = WORKING_DIR / "kaggle_config.py"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  OK kaggle_config.py -> {out}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary():
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    checks = [
        ("ContractNLI train.json", DATA_DIR / "contractnli/resources/contract-nli/train.json"),
        ("ContractNLI dev.json",   DATA_DIR / "contractnli/resources/contract-nli/dev.json"),
        ("ContractNLI test.json",  DATA_DIR / "contractnli/resources/contract-nli/test.json"),
        ("CUAD",                   HF_CACHE_DIR / "datasets--theatticusproject--cuad"),
        ("MultiHopRAG",            HF_CACHE_DIR / "datasets--yixuantt--MultiHopRAG"),
        ("LoCoV1-Documents",       HF_CACHE_DIR / "datasets--hazyresearch--LoCoV1-Documents"),
        ("LoCoV1-Queries",         HF_CACHE_DIR / "datasets--hazyresearch--LoCoV1-Queries"),
    ]
    all_ok = True
    for name, path in checks:
        exists = Path(path).exists()
        print(f"  {'OK ' if exists else 'MISSING'} {name}")
        if not exists:
            all_ok = False
    print("=" * 60)
    if all_ok:
        print("All datasets ready.  Next: !python smoke_test.py")
    else:
        print("Some datasets missing - check errors above.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    install_deps()
    download_contractnli()
    download_cuad()
    download_multihop()
    download_loco()
    write_kaggle_config()
    print_summary()
