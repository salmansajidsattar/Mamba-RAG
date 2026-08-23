"""
src/training/trainer.py
=======================
Training loop for RevRAG bi-encoder.

Training data: ContractNLI (train split) + CUAD amendment hard negatives.
Loss:          InfoNCE with temperature τ=0.02.
Optimiser:     AdamW with linear warmup + cosine decay.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.bi_encoder import BiEncoder
from src.training.losses   import InfoNCELoss
from src.config            import CFG

logger = logging.getLogger(__name__)


# ── PyTorch Dataset for (query, positive, hard_neg) triplets ─────────────────

class TripletDataset(Dataset):
    """
    Wraps a list of (query_text, positive_text, hard_neg_text) tuples.
    Tokenisation is deferred to the collate function.
    """

    def __init__(self, triplets: List[Tuple[str, str, str]]):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]   # (query, pos, hard_neg)


def collate_fn(tokenizer, max_len: int):
    """Returns a collate function that tokenises on-the-fly."""
    def _collate(batch):
        queries   = [b[0] for b in batch]
        positives = [b[1] for b in batch]
        hard_negs = [b[2] for b in batch]

        def tok(texts):
            return tokenizer(
                texts,
                max_length=max_len,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )

        return tok(queries), tok(positives), tok(hard_negs)
    return _collate


# ── Trainer ───────────────────────────────────────────────────────────────────

class BiEncoderTrainer:
    """
    Trains the BiEncoder with InfoNCE loss.

    Parameters
    ----------
    bi_encoder     : BiEncoder instance (Mamba-2 or BERT fallback)
    train_triplets : (query, positive, hard_neg) tuples — ContractNLI + CUAD
    val_triplets   : validation triplets for early stopping
    out_dir        : checkpoint save directory
    device         : "cuda" or "cpu"
    """

    def __init__(
        self,
        bi_encoder:     BiEncoder,
        train_triplets: List[Tuple[str, str, str]],
        val_triplets:   List[Tuple[str, str, str]],
        out_dir:        Path = CFG.out_dir / "checkpoints",
        device:         str  = "cuda",
    ):
        self.model          = bi_encoder
        self.train_triplets = train_triplets
        self.val_triplets   = val_triplets
        self.out_dir        = Path(out_dir)
        self.device         = device
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.loss_fn  = InfoNCELoss(
            temperature=CFG.training.temperature,
            hard_neg_weight=CFG.training.hard_neg_weight,
        )
        self.history: List[dict] = []

    def _build_optimizer(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=CFG.training.learning_rate,
            weight_decay=CFG.training.weight_decay,
        )

    def _build_scheduler(self, optimizer, total_steps: int):
        from torch.optim.lr_scheduler import OneCycleLR
        return OneCycleLR(
            optimizer,
            max_lr=CFG.training.learning_rate,
            total_steps=total_steps,
            pct_start=CFG.training.warmup_steps / total_steps,
        )

    def train(self):
        tokenizer  = self.model.query_encoder.tokenizer
        max_len    = min(
            CFG.model.max_seq_len,
            512 if not self.model.query_encoder._is_mamba else CFG.model.max_seq_len,
        )

        train_ds = TripletDataset(self.train_triplets)
        val_ds   = TripletDataset(self.val_triplets)

        train_loader = DataLoader(
            train_ds,
            batch_size=CFG.training.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn(tokenizer, max_len),
            num_workers=0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=CFG.training.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn(tokenizer, max_len),
            num_workers=0,
        )

        optimizer    = self._build_optimizer()
        total_steps  = len(train_loader) * CFG.training.max_epochs
        scheduler    = self._build_scheduler(optimizer, total_steps)
        scaler       = torch.cuda.amp.GradScaler() if (
            CFG.training.fp16 and self.device == "cuda"
        ) else None

        best_val_loss = float("inf")
        patience_cnt  = 0

        logger.info(
            f"Training: {len(self.train_triplets)} triplets, "
            f"{CFG.training.max_epochs} epochs, device={self.device}"
        )

        for epoch in range(1, CFG.training.max_epochs + 1):
            t_start = time.time()

            # ── Train ────────────────────────────────────────────────────────
            self.model.train()
            train_loss = 0.0

            for step, (q_enc, d_enc, hn_enc) in enumerate(train_loader, 1):
                q_enc  = {k: v.to(self.device) for k, v in q_enc.items()}
                d_enc  = {k: v.to(self.device) for k, v in d_enc.items()}
                hn_enc = {k: v.to(self.device) for k, v in hn_enc.items()}

                optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    q_emb, d_emb = self.model(
                        q_enc["input_ids"], q_enc["attention_mask"],
                        d_enc["input_ids"], d_enc["attention_mask"],
                    )
                    # Encode hard negatives separately
                    hn_emb = self.model.doc_encoder(
                        hn_enc["input_ids"], hn_enc["attention_mask"]
                    )
                    loss = self.loss_fn(q_emb, d_emb, hn_emb)

                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), CFG.training.grad_clip
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), CFG.training.grad_clip
                    )
                    optimizer.step()

                scheduler.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # ── Validate ─────────────────────────────────────────────────────
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for q_enc, d_enc, hn_enc in val_loader:
                    q_enc  = {k: v.to(self.device) for k, v in q_enc.items()}
                    d_enc  = {k: v.to(self.device) for k, v in d_enc.items()}
                    hn_enc = {k: v.to(self.device) for k, v in hn_enc.items()}
                    q_emb, d_emb = self.model(
                        q_enc["input_ids"], q_enc["attention_mask"],
                        d_enc["input_ids"], d_enc["attention_mask"],
                    )
                    hn_emb = self.model.doc_encoder(
                        hn_enc["input_ids"], hn_enc["attention_mask"]
                    )
                    val_loss += self.loss_fn(q_emb, d_emb, hn_emb).item()
            val_loss /= max(len(val_loader), 1)

            elapsed = time.time() - t_start
            record  = {
                "epoch": epoch, "train_loss": train_loss,
                "val_loss": val_loss, "elapsed_s": elapsed,
            }
            self.history.append(record)
            logger.info(
                f"Epoch {epoch:3d}/{CFG.training.max_epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s"
            )

            # ── Checkpoint + early stopping ───────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_cnt  = 0
                self.model.save(self.out_dir / "best")
                logger.info(f"  ✓ Best model saved (val_loss={val_loss:.4f})")
            else:
                patience_cnt += 1
                if patience_cnt >= CFG.training.early_stop_patience:
                    logger.info(f"  Early stopping at epoch {epoch}.")
                    break

            if epoch % CFG.training.save_every_n == 0:
                self.model.save(self.out_dir / f"epoch_{epoch:03d}")

        # Save training history
        hist_path = self.out_dir / "training_history.json"
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training complete. History → {hist_path}")
        return self.history
