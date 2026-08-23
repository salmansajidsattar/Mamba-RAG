"""
src/training/losses.py
======================
InfoNCE contrastive loss for bi-encoder training.

InfoNCE (van den Oord et al., 2018) — used in DPR, ANCE, SimCSE.
In-batch negatives: every other document in the batch is a negative.
Hard negatives: explicitly paired negatives (contradiction spans, superseded amendments).

Loss for a batch of B (query, positive, hard_neg) triplets:

  L = -1/B * Σ log [ exp(q_i · d_i / τ) /
                     (Σ_j exp(q_i · d_j / τ) + exp(q_i · hn_i / τ)) ]

where q_i, d_i are L2-normalised embeddings, τ is temperature.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss with in-batch negatives and hard negatives.

    Parameters
    ----------
    temperature     : τ (default 0.02 — standard for dense retrieval)
    hard_neg_weight : weight applied to hard negative logits (default 1.0)
    """

    def __init__(self, temperature: float = 0.02, hard_neg_weight: float = 1.0):
        super().__init__()
        self.temperature     = temperature
        self.hard_neg_weight = hard_neg_weight

    def forward(
        self,
        q_emb:  torch.Tensor,           # (B, D)
        d_emb:  torch.Tensor,           # (B, D)  positives
        hn_emb: torch.Tensor | None = None,  # (B, D) hard negatives (optional)
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        q_emb  : L2-normalised query embeddings (B, D)
        d_emb  : L2-normalised positive doc embeddings (B, D)
        hn_emb : L2-normalised hard negative embeddings (B, D), optional

        Returns
        -------
        scalar loss
        """
        B = q_emb.size(0)

        # Similarity matrix: q × all_docs — shape (B, B)
        sim_matrix = torch.matmul(q_emb, d_emb.T) / self.temperature

        # Append hard negatives as extra columns → (B, B + B)
        if hn_emb is not None:
            hn_sim = torch.matmul(q_emb, hn_emb.T) / self.temperature
            hn_sim = hn_sim * self.hard_neg_weight
            # Only include the diagonal hard negatives (each query's own hard neg)
            hn_diag = hn_sim.diag().unsqueeze(1)      # (B, 1)
            sim_matrix = torch.cat([sim_matrix, hn_diag], dim=1)  # (B, B+1)

        # Labels: diagonal = correct (index i = positive for query i)
        labels = torch.arange(B, device=q_emb.device)

        loss = F.cross_entropy(sim_matrix, labels)
        return loss


class MarginMSELoss(nn.Module):
    """
    Margin-MSE loss (Hofstätter et al., 2021) — alternative to InfoNCE.
    Useful when you have graded relevance scores from a teacher model.

    L = MSE( score(q, d+) - score(q, d-), margin_target )
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        q_emb:  torch.Tensor,   # (B, D)
        pos_emb: torch.Tensor,  # (B, D)
        neg_emb: torch.Tensor,  # (B, D)
    ) -> torch.Tensor:
        pos_scores = (q_emb * pos_emb).sum(dim=-1)   # (B,)
        neg_scores = (q_emb * neg_emb).sum(dim=-1)   # (B,)
        target     = torch.full_like(pos_scores, self.margin)
        return F.mse_loss(pos_scores - neg_scores, target)
