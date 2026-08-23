"""
src/config.py — Central configuration for RevRAG.
All hyperparameters in one place. Import anywhere with:
    from src.config import CFG
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

ROOT     = Path(__file__).resolve().parents[1]   # F:\MambaRAG\
DATA_DIR = ROOT / "data"
OUT_DIR  = ROOT / "outputs"


@dataclass
class ModelConfig:
    # ── Encoder ──────────────────────────────────────────────────────────────
    # GPU (Kaggle): real Mamba-2 SSM, linear-time O(T)
    mamba_checkpoint:  str = "state-spaces/mamba2-130m"
    # CPU fallback for local development (same interface, swap automatically)
    cpu_fallback:      str = "bert-base-uncased"
    hidden_size:       int = 768
    max_seq_len:       int = 4096   # Mamba-2 handles long seqs natively
    encode_batch_size: int = 8

    # ── Verifier (NLI cross-encoder) ─────────────────────────────────────────
    # DeBERTa-v3-small: SOTA NLI, fast at inference, NOT trained on ContractNLI
    # → no data leakage when evaluating on ContractNLI contradiction subset
    verifier_checkpoint: str   = "cross-encoder/nli-deberta-v3-small"
    nli_batch_size:      int   = 32

    # ── Retrieval hyperparameters ─────────────────────────────────────────────
    top_k_retrieval: int   = 20    # candidates before reranking
    top_k_final:     int   = 5     # passages returned to user

    # ── Verifier hyperparameters (ablated in paper) ───────────────────────────
    page_order_weight:          float = 0.3   # λ: weight of page-order score
    nli_contradiction_threshold: float = 0.5  # τ_c: score above = contradiction


@dataclass
class TrainingConfig:
    # ── Loss ─────────────────────────────────────────────────────────────────
    loss:             str   = "infonce"
    temperature:      float = 0.02
    hard_neg_weight:  float = 1.0   # weight for hard negatives vs in-batch

    # ── Optimisation ─────────────────────────────────────────────────────────
    learning_rate:    float = 2e-5
    weight_decay:     float = 0.01
    warmup_steps:     int   = 500
    max_epochs:       int   = 10
    train_batch_size: int   = 32
    eval_batch_size:  int   = 64
    grad_clip:        float = 1.0
    fp16:             bool  = True    # mixed precision on GPU

    # ── Checkpointing ────────────────────────────────────────────────────────
    save_dir:         Path  = OUT_DIR / "checkpoints"
    save_every_n:     int   = 1       # epochs
    early_stop_patience: int = 3


@dataclass
class DataConfig:
    data_dir:     Path = DATA_DIR
    hf_cache:     Path = DATA_DIR / "hf_cache"
    hf_cache_dir: Path = DATA_DIR / "hf_cache"   # alias used by loaders

    # ContractNLI (official splits in JSON)
    # Actual JSON files are at: data/contract-nli/
    contractnli_dir: Path = DATA_DIR / "contract-nli"
    cnli_label_filter: List[str] = field(
        default_factory=lambda: ["Entailment", "Contradiction"]
    )
    cnli_min_span_len: int = 20

    # CUAD — local JSON takes priority over HF cache
    cuad_json:     Path = DATA_DIR / "CUAD_v1" / "CUAD_v1.json"
    cuad_cache:    Path = DATA_DIR / "hf_cache"
    cuad_min_gap:  int  = 500     # min char gap between amendment spans
    cuad_categories: List[str] = field(default_factory=lambda: [
        "Amendment Exception",
        "Renewal Term",
        "Expiration Date",
        "Anti-Assignment",
        "Termination For Convenience",
        "Notice Period To Terminate Renewal",
    ])

    # MultiHopRAG
    multihop_cache:      Path = DATA_DIR / "hf_cache"
    multihop_max_queries: int = 500

    # LegalBench-RAG
    legalbench_dir: Path = DATA_DIR / "legalbench_rag"

    # LoCoV1
    loco_cache:     Path = DATA_DIR / "hf_cache"

    # Hard negatives per positive during training
    hard_neg_per_pos: int = 3
    # Random seed for reproducible splits
    seed: int = 42


@dataclass
class EvalConfig:
    # k values to evaluate at
    recall_k:    List[int] = field(default_factory=lambda: [1, 5, 20])
    precision_k: List[int] = field(default_factory=lambda: [1, 5])
    ndcg_k:      List[int] = field(default_factory=lambda: [10])

    # λ and τ_c search space for ablation
    lambda_values: List[float] = field(
        default_factory=lambda: [round(x * 0.1, 1) for x in range(11)]
    )
    threshold_values: List[float] = field(
        default_factory=lambda: [round(0.30 + x * 0.05, 2) for x in range(9)]
    )


@dataclass
class RevRAGConfig:
    model:    ModelConfig    = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data:     DataConfig     = field(default_factory=DataConfig)
    eval:     EvalConfig     = field(default_factory=EvalConfig)
    device:   str            = "cuda"   # overridden to "cpu" locally
    seed:     int            = 42
    out_dir:  Path           = OUT_DIR


CFG = RevRAGConfig()
