"""
config.py — Central configuration for RevRAG experiments
=========================================================

Single source of truth. Import anywhere:

    from config import cfg

Edit the values here instead of scattering settings across multiple files.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ── Root paths ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
WORK_DIR  = ROOT_DIR / "revrag"
DATA_DIR  = WORK_DIR / "data"
OUT_DIR   = WORK_DIR / "outputs"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sub-configs (each section maps to one step)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class EnvConfig:
    """Hardware / environment settings."""
    device:       str  = "cuda"   # "cuda" | "cpu"
    skip_install: bool = False    # True if deps already installed


@dataclass
class DatasetConfig:
    """Which datasets to download and evaluate on."""

    # Primary benchmarks (real, peer-reviewed — used for main paper claims)
    primary: List[str] = field(default_factory=lambda: [
        "legalbench",       # character-level retrieval over real legal docs
        "contractnli",      # expert-annotated NDA retrieval (Koreeda & Manning, 2021)
        "cuad_amendments",  # real amendment pairs from CUAD (Hendrycks et al., NeurIPS 2021)
        "multihop_rag",     # multi-hop retrieval (Li et al., 2024)
        "loco",             # long-context retrieval (Chang et al., 2024)
    ])

    skip_loco: bool = False  # True to skip LoCoV1 if session time is tight

    # HuggingFace cache locations
    hf_cache: Path = DATA_DIR / "raw"


@dataclass
class EvalConfig:
    """Evaluation settings."""
    models: List[str] = field(default_factory=lambda: [
        "bm25", "bge", "bgem3", "mamba", "revrag_no_verifier", "revrag",
    ])
    seeds:  List[int] = field(default_factory=lambda: [42, 123, 456])

    # Supplementary synthetic eval (diagnostic / ablation — not main claims)
    run_synthetic: bool = True


@dataclass
class SyntheticConfig:
    """Supplementary synthetic benchmark settings."""
    num_documents:       int = 400
    contradictions_per_doc: int = 5
    validation_sample:   int = 100
    seed:                int = 42
    output_path: Path = DATA_DIR / "synthetic" / "cross_page_bench.jsonl"


@dataclass
class AblationConfig:
    """λ × τ_c hyperparameter sweep."""
    skip:       bool = False
    max_items:  int  = 500   # items from synthetic benchmark to use
    output_dir: Path = OUT_DIR / "ablation"


# ── Free LLM options for generation eval ─────────────────────────────────────
# All run locally via HuggingFace Transformers — no API key required.
#
#  Model                               VRAM     Speed (T4)   Quality
#  ─────────────────────────────────── ──────── ──────────── ──────────
#  microsoft/phi-3-mini-4k-instruct    ~8 GB    ~40 tok/s    ★★★★
#  mistralai/Mistral-7B-Instruct-v0.2  ~14 GB   ~25 tok/s    ★★★★★
#  google/flan-t5-large                ~3 GB    ~80 tok/s    ★★★
#  HuggingFaceTB/SmolLM2-1.7B-Instruct ~4 GB   ~60 tok/s    ★★★

FREE_LLM_OPTIONS = {
    "phi3":     "microsoft/phi-3-mini-4k-instruct",   # recommended default
    "mistral":  "mistralai/Mistral-7B-Instruct-v0.2", # best quality
    "flan":     "google/flan-t5-large",               # lightweight
    "smollm":   "HuggingFaceTB/SmolLM2-1.7B-Instruct",# fastest
}


@dataclass
class GenerationConfig:
    """End-to-end generation evaluation settings."""
    skip:       bool = False

    # Free local LLM — no API key needed
    backend:    str  = "hf"             # always "hf" (free)
    model_key:  str  = "phi3"           # key into FREE_LLM_OPTIONS above
    n_queries:  int  = 200
    max_new_tokens: int = 256
    load_in_4bit:   bool = True         # 4-bit quant to fit 7B on T4 16 GB
    output_dir: Path = OUT_DIR / "generation_eval"

    @property
    def model_name(self) -> str:
        return FREE_LLM_OPTIONS[self.model_key]


@dataclass
class PlotConfig:
    skip:       bool = False
    output_dir: Path = OUT_DIR / "figures"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Top-level config object
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class RevRAGConfig:
    env:        EnvConfig        = field(default_factory=EnvConfig)
    datasets:   DatasetConfig    = field(default_factory=DatasetConfig)
    eval:       EvalConfig       = field(default_factory=EvalConfig)
    synthetic:  SyntheticConfig  = field(default_factory=SyntheticConfig)
    ablation:   AblationConfig   = field(default_factory=AblationConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    plots:      PlotConfig       = field(default_factory=PlotConfig)

    work_dir: Path = WORK_DIR
    out_dir:  Path = OUT_DIR


# Singleton — import and override fields as needed
cfg = RevRAGConfig()
