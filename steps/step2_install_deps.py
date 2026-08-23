"""Step 2 — Install Python dependencies.

Only needed when running on a fresh Kaggle session or clean venv.
Set SKIP_INSTALL=True in run_experiment.py if deps are already installed.
"""
import subprocess, sys


CORE_PACKAGES = [
    "transformers==4.44.0",
    "datasets",
    "faiss-gpu",       # falls back to faiss-cpu if no GPU
    "rank-bm25",
    "sentence-transformers",
    "FlagEmbedding",
    "pytrec-eval-terrier",
    "tqdm",
    "pyyaml",
    "numpy",
    "scipy",
    "matplotlib",
    "openai",
]


def _pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])


def run(skip: bool = False):
    if skip:
        print("  [skip] dependency installation (SKIP_INSTALL=True)")
        return

    print("  Installing core packages …")
    _pip(*CORE_PACKAGES)
    print("  ✓ Core packages installed")

    # faiss-gpu may fail in CPU-only envs — fall back to faiss-cpu
    try:
        import faiss  # noqa
    except ImportError:
        print("  faiss-gpu failed — installing faiss-cpu fallback …")
        _pip("faiss-cpu")

    # mamba-ssm: try pip wheel first, then source build
    print("  Installing mamba-ssm (may take ~8 min if built from source) …")
    try:
        _pip("causal-conv1d")
        _pip("mamba-ssm")
        import mamba_ssm  # noqa
        print("  ✓ mamba-ssm installed via pip wheel")
    except Exception:
        print("  pip wheel failed — building from source …")
        try:
            _pip("ninja", "packaging")
            _pip("git+https://github.com/state-spaces/mamba.git")
            import mamba_ssm  # noqa
            print("  ✓ mamba-ssm built from source")
        except Exception as e:
            print(f"  ⚠ mamba-ssm unavailable: {e}")
            print("    GRU stub encoder will be used — Mamba model results are invalid.")
