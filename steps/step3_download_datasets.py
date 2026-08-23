"""Step 3 — Datasets: use pre-uploaded Kaggle zip OR download fresh.

Two modes (auto-detected):

  MODE A — Kaggle pre-uploaded (recommended):
    You ran download_datasets_local.py on your CPU, uploaded revrag_datasets.zip
    to Kaggle as a dataset named 'revrag-datasets'.
    → Copies from /kaggle/input/revrag-datasets/ in ~10 seconds. No network.

  MODE B — Download on the fly:
    No Kaggle dataset attached. Downloads from GitHub / HuggingFace.
    → Takes ~8 min. Requires internet ON in Kaggle settings.

Detection: if KAGGLE_INPUT_DIR exists and has the data, use Mode A.
           otherwise fall back to Mode B automatically.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Where Kaggle mounts uploaded datasets
KAGGLE_INPUT = Path("/kaggle/input/revrag-datasets/data")


def run(work_dir: str, skip_loco: bool = False):
    data_dir = Path(work_dir) / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    if KAGGLE_INPUT.exists():
        _mode_a_copy(data_dir, skip_loco)
    else:
        _mode_b_download(data_dir, skip_loco)

    print("  ✓ All datasets ready.")


# ── Mode A: copy from pre-uploaded Kaggle dataset ─────────────────────────────

def _mode_a_copy(data_dir: Path, skip_loco: bool):
    print("  Mode A: copying from pre-uploaded Kaggle dataset …")
    datasets = ["legalbench_rag", "contractnli", "cuad", "multihop_rag"]
    if not skip_loco:
        datasets.append("loco_v1")

    for name in datasets:
        src = KAGGLE_INPUT / name
        dst = data_dir / name
        if dst.exists():
            print(f"  ✓ {name} (already in working dir)")
            continue
        if not src.exists():
            print(f"  ⚠ {name} not found in Kaggle input — will download.")
            _download_one(name, data_dir, skip_loco)
            continue
        print(f"  Copying {name} …", end=" ", flush=True)
        shutil.copytree(str(src), str(dst))
        print("✓")


# ── Mode B: download fresh ─────────────────────────────────────────────────────

def _mode_b_download(data_dir: Path, skip_loco: bool):
    print("  Mode B: downloading from GitHub / HuggingFace …")
    print("  (Tip: run download_datasets_local.py locally, upload zip to Kaggle")
    print("   as 'revrag-datasets' to skip this step next time.)\n")

    for name in ["legalbench_rag", "contractnli", "cuad", "multihop_rag"] + (
        [] if skip_loco else ["loco_v1"]
    ):
        _download_one(name, data_dir, skip_loco)


def _download_one(name: str, data_dir: Path, skip_loco: bool):
    dst = data_dir / name
    if dst.exists():
        print(f"  ✓ {name} (cached)")
        return

    if name == "legalbench_rag":
        print("  Cloning LegalBench-RAG …")
        subprocess.check_call([
            "git", "clone", "--depth=1",
            "https://github.com/zeroentropy-ai/legalbenchrag", str(dst),
        ])

    elif name == "contractnli":
        print("  Cloning ContractNLI (Stanford NLP) …")
        subprocess.check_call([
            "git", "clone", "--depth=1",
            "https://github.com/stanfordnlp/contract-nli", str(dst),
        ])
        missing = [f for f in ["train.json", "dev.json", "test.json"]
                   if not (dst / "data" / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"ContractNLI JSON files missing: {missing}\n"
                f"Download manually from https://stanfordnlp.github.io/contract-nli/"
            )

    elif name == "cuad":
        print("  Downloading CUAD (~300 MB) …")
        from datasets import load_dataset
        load_dataset("theatticusproject/cuad", cache_dir=str(dst))

    elif name == "multihop_rag":
        print("  Downloading MultiHop-RAG (~200 MB) …")
        from datasets import load_dataset
        load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train", cache_dir=str(dst))

    elif name == "loco_v1":
        if skip_loco:
            print("  ─ LoCoV1 skipped (skip_loco=True)")
            return
        print("  Downloading LoCoV1 (~500 MB) …")
        from datasets import load_dataset
        # LoCoV1 is split into two separate HF datasets
        load_dataset("hazyresearch/LoCoV1-Documents", cache_dir=str(dst / "documents"))
        load_dataset("hazyresearch/LoCoV1-Queries",   cache_dir=str(dst / "queries"))

    print(f"  ✓ {name}")
