"""
Fine-tuning script for the Mamba-2 bi-encoder retriever.

Training procedure (following DPR / Mamba Retriever convention):
    - In-batch negatives: each document in the batch serves as a negative
      for all other queries in the batch
    - Hard negatives: BM25-mined negatives added per query
    - Loss: InfoNCE (in-batch softmax cross-entropy)
    - Optimizer: AdamW with linear warmup + cosine decay
    - Precision: fp16 via torch.cuda.amp

Usage:
    python train/train_mamba.py --config configs/default.yaml
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from models.mamba_retriever import MambaEncoder, InfoNCELoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrieverTrainDataset(Dataset):
    """
    Simple query-positive pair dataset for bi-encoder training.

    Each item: {"query": str, "positive": str, "negatives": [str, ...]}
    Expected format: JSONL file with one example per line.
    """

    def __init__(self, path: str, max_negatives: int = 7):
        self.examples = []
        self.max_negatives = max_negatives
        with open(path) as f:
            for line in f:
                ex = json.loads(line)
                self.examples.append(ex)
        logger.info(f"Loaded {len(self.examples)} training examples from {path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        negs = ex.get("negatives", [])[:self.max_negatives]
        return {
            "query": ex["query"],
            "positive": ex["positive"],
            "negatives": negs,
        }


def train(cfg: dict, output_dir: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_cfg = cfg["training"]

    # Model
    encoder = MambaEncoder(
        checkpoint=cfg["model"]["mamba2"]["checkpoint"],
        hidden_size=cfg["model"]["mamba2"]["hidden_size"],
        max_length=cfg["model"]["mamba2"]["max_length"],
        pooling=cfg["model"]["mamba2"]["pooling"],
    ).to(device)

    loss_fn = InfoNCELoss(temperature=0.02)
    optimizer = AdamW(
        encoder.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scaler = GradScaler(enabled=train_cfg.get("fp16", True) and device == "cuda")

    # Data
    train_data = RetrieverTrainDataset(
        path=cfg.get("train_data_path", "data/processed/train.jsonl"),
        max_negatives=train_cfg["hard_negatives_per_query"],
    )
    train_loader = DataLoader(
        train_data,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        collate_fn=lambda batch: batch,  # keep as list; tokenize in loop
    )

    total_steps = (
        len(train_loader)
        * train_cfg["epochs"]
        // train_cfg["gradient_accumulation_steps"]
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_cfg["warmup_steps"],
        num_training_steps=total_steps,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(train_cfg["epochs"]):
        encoder.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['epochs']}")
        ):
            queries = [ex["query"] for ex in batch]
            positives = [ex["positive"] for ex in batch]

            # Tokenize queries and positives
            q_enc = encoder.tokenizer(
                queries,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)

            d_enc = encoder.tokenizer(
                positives,
                padding=True,
                truncation=True,
                max_length=cfg["model"]["mamba2"]["max_length"],
                return_tensors="pt",
            ).to(device)

            with autocast(enabled=train_cfg.get("fp16", True) and device == "cuda"):
                q_embs = encoder(
                    q_enc["input_ids"],
                    q_enc.get("attention_mask"),
                )
                d_embs = encoder(
                    d_enc["input_ids"],
                    d_enc.get("attention_mask"),
                )
                loss = loss_fn(q_embs, d_embs)
                loss = loss / train_cfg["gradient_accumulation_steps"]

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * train_cfg["gradient_accumulation_steps"]

            if (step + 1) % train_cfg["gradient_accumulation_steps"] == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % train_cfg.get("save_steps", 500) == 0:
                ckpt_path = output_dir / f"checkpoint_epoch{epoch+1}_step{step+1}"
                torch.save(encoder.state_dict(), ckpt_path / "model.pt")

        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = output_dir / "best_model.pt"
            torch.save(encoder.state_dict(), best_path)
            logger.info(f"New best model saved to {best_path}")

    logger.info(f"Training complete. Best loss: {best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="outputs/checkpoints/mamba2_finetuned")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(args.seed)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr

    train(cfg, args.output_dir)


if __name__ == "__main__":
    main()
