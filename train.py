"""
train.py
========
Top-level training entry point for RevRAG bi-encoder.

Usage:
    python train.py                        # default: ContractNLI + CUAD, GPU if available
    python train.py --device cpu           # force CPU
    python train.py --max_epochs 5         # quick test
    python train.py --smoke                # 2-epoch smoke test on 200 samples

What this does:
  1. Load ContractNLI train/val splits → (query, positive, hard_neg) triplets
  2. Load CUAD amendment train/val splits → additional hard negative triplets
  3. Combine triplets and train BiEncoder (Mamba-2 or BERT fallback)
  4. Save best checkpoint to outputs/checkpoints/best/
  5. Save training_history.json and training loss curve
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")

# Auto-detect Kaggle and patch paths
if Path("/kaggle/working").exists():
    try:
        import importlib, sys
        sys.path.insert(0, "/kaggle/working")
        importlib.import_module("kaggle_config")
    except ModuleNotFoundError:
        logger.warning("Running on Kaggle but kaggle_config.py not found. "
                       "Run kaggle_download_datasets.py first.")


def parse_args():
    p = argparse.ArgumentParser(description="Train RevRAG bi-encoder")
    p.add_argument("--device",        default=None,   help="cuda | cpu (auto-detect if omitted)")
    p.add_argument("--max_epochs",    type=int,   default=None, help="Override max epochs")
    p.add_argument("--batch_size",    type=int,   default=None, help="Override train batch size")
    p.add_argument("--lr",            type=float, default=None, help="Override learning rate")
    p.add_argument("--out_dir",       default=None,  help="Output directory for checkpoints")
    p.add_argument("--smoke",         action="store_true", help="Quick 2-epoch smoke test")
    p.add_argument("--no_cuad",       action="store_true", help="Skip CUAD data")
    p.add_argument("--processed_dir", default=None,
                   help="Path to revrag_processed/ dir (skips raw dataset loading)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Imports ───────────────────────────────────────────────────────────────
    import torch
    from src.config import CFG
    from src.models.bi_encoder import BiEncoder
    from src.training.trainer  import BiEncoderTrainer

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ── Config overrides ──────────────────────────────────────────────────────
    if args.max_epochs:
        CFG.training.max_epochs = args.max_epochs
    if args.batch_size:
        CFG.training.train_batch_size = args.batch_size
    if args.lr:
        CFG.training.learning_rate = args.lr
    if args.out_dir:
        CFG.out_dir = Path(args.out_dir)
    if args.smoke:
        CFG.training.max_epochs       = 2
        CFG.training.early_stop_patience = 99
        logger.info("Smoke mode: 2 epochs, no early stopping")

    out_dir = CFG.out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load training data ────────────────────────────────────────────────────
    processed_dir = Path(args.processed_dir) if args.processed_dir else None

    if processed_dir and processed_dir.exists():
        # ── Fast path: load pre-computed JSON files (Kaggle mode) ─────────────
        logger.info(f"Loading from pre-processed dir: {processed_dir}")
        from src.data.processed_loader import ProcessedDataLoader
        loader = ProcessedDataLoader(processed_dir)

        raw_train = loader.all_train_triplets(include_cuad=not args.no_cuad)
        raw_val   = loader.all_val_triplets(include_cuad=not args.no_cuad)

        # Convert dict→tuple for trainer compatibility
        train_triplets = [(t["query"], t["positive"], t["hard_neg"]) for t in raw_train]
        val_triplets   = [(t["query"], t["positive"], t["hard_neg"]) for t in raw_val]
        logger.info(f"  Loaded {len(train_triplets)} train, {len(val_triplets)} val triplets")

    else:
        # ── Raw path: load directly from HuggingFace / JSON files ────────────
        logger.info("Loading ContractNLI …")
        from src.data.contractnli import ContractNLIDataset

        train_cnli = ContractNLIDataset(
            data_dir=CFG.data.contractnli_dir,
            split="train",
            seed=CFG.data.seed,
        )
        val_cnli = ContractNLIDataset(
            data_dir=CFG.data.contractnli_dir,
            split="dev",
            seed=CFG.data.seed,
        )

        train_triplets = train_cnli.get_triplets()
        val_triplets   = val_cnli.get_triplets()
        logger.info(f"  ContractNLI: {len(train_triplets)} train, {len(val_triplets)} val triplets")

        if not args.no_cuad:
            logger.info("Loading CUAD …")
            try:
                from src.data.cuad import CUADAmendmentDataset
                cuad = CUADAmendmentDataset(
                    hf_cache_dir=CFG.data.hf_cache_dir,
                    seed=CFG.data.seed,
                )
                cuad_train_trip = [
                    (e["query"], e.get("gold_text", ""), e.get("decoy_text", ""))
                    for e in cuad.get_split("train")
                    if e.get("gold_text") and e.get("decoy_text")
                ]
                cuad_val_trip = [
                    (e["query"], e.get("gold_text", ""), e.get("decoy_text", ""))
                    for e in cuad.get_split("val")
                    if e.get("gold_text") and e.get("decoy_text")
                ]
                if args.smoke:
                    cuad_train_trip = cuad_train_trip[:100]
                    cuad_val_trip   = cuad_val_trip[:25]
                train_triplets += cuad_train_trip
                val_triplets   += cuad_val_trip
                logger.info(
                    f"  CUAD: +{len(cuad_train_trip)} train, +{len(cuad_val_trip)} val triplets"
                )
            except Exception as e:
                logger.warning(f"  CUAD skipped: {e}")

    if args.smoke:
        train_triplets = train_triplets[:200]
        val_triplets   = val_triplets[:50]

    logger.info(f"Total: {len(train_triplets)} train, {len(val_triplets)} val triplets")

    # ── Build model ───────────────────────────────────────────────────────────
    logger.info("Building BiEncoder …")
    bi_encoder = BiEncoder(
        checkpoint=CFG.model.mamba_checkpoint,
        hidden_size=CFG.model.hidden_size,
        max_seq_len=CFG.model.max_seq_len,
        share_weights=True,
        device=device,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = BiEncoderTrainer(
        bi_encoder=bi_encoder,
        train_triplets=train_triplets,
        val_triplets=val_triplets,
        out_dir=out_dir,
        device=device,
    )
    history = trainer.train()

    # ── Plot training loss ────────────────────────────────────────────────────
    try:
        from src.visualization.plots import plot_training_loss
        plot_dir = CFG.out_dir / "figures"
        plot_training_loss(history, plot_dir)
        logger.info(f"Training loss curve → {plot_dir}")
    except Exception as e:
        logger.warning(f"Could not generate training loss plot: {e}")

    logger.info(f"\n✓ Training complete. Best checkpoint → {out_dir / 'best'}")


if __name__ == "__main__":
    main()
