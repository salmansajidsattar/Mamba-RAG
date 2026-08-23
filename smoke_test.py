"""
smoke_test.py
=============
CPU smoke test — verifies every module imports and core logic runs.
No GPU, no HuggingFace downloads required (uses synthetic data for model tests).
ContractNLI JSON files must exist at the configured path for the data tests.

Usage:
    python smoke_test.py

Exits with code 0 if all checks pass, 1 on any failure.
"""
from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

PASS = "✓"
FAIL = "✗"
results = []


def check(name: str, fn):
    try:
        fn()
        print(f"  {PASS} {name}")
        results.append((name, True, None))
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        print(f"  {FAIL} {name}")
        print(f"       {msg}")
        results.append((name, False, msg))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[1] Config")

def _config_loads():
    from src.config import CFG
    assert CFG.model.hidden_size == 768
    assert CFG.training.temperature == 0.02
    assert isinstance(CFG.eval.recall_k, list)

check("CFG singleton loads", _config_loads)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Metrics (no model, no data)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[2] Metrics")

def _recall():
    from src.evaluation.metrics import recall_at_k
    assert recall_at_k({"a","b"}, ["a","c","b"], 1) == 0.5
    assert recall_at_k({"a","b"}, ["a","b","c"], 2) == 1.0

def _ndcg():
    from src.evaluation.metrics import ndcg_at_k
    v = ndcg_at_k({"a"}, ["a","b","c"], 3)
    assert 0 < v <= 1

def _mrr():
    from src.evaluation.metrics import mrr
    assert mrr({"b"}, ["a","b","c"]) == 0.5

def _cra():
    from src.evaluation.metrics import cra_at_k
    passages = [{"doc_id":"gold","is_gold":True,"is_decoy":False},
                {"doc_id":"dec","is_gold":False,"is_decoy":True}]
    assert cra_at_k(passages, k=1) == 1.0
    passages_bad = [{"doc_id":"dec","is_gold":False,"is_decoy":True},
                    {"doc_id":"gold","is_gold":True,"is_decoy":False}]
    assert cra_at_k(passages_bad, k=1) == 0.0

def _dpr():
    from src.evaluation.metrics import decoy_promotion_rate
    bad = [[{"doc_id":"d","is_gold":False,"is_decoy":True}]]
    assert decoy_promotion_rate(bad, k=1) == 1.0

def _char_recall():
    from src.evaluation.metrics import character_recall_at_k
    spans = [(0, 5, "hello")]
    texts = ["hello world"]
    assert character_recall_at_k(spans, texts, k=1) == 1.0

def _compute_all():
    from src.evaluation.metrics import compute_all_metrics
    run = {"q1": ["d1","d2"], "q2": ["d2","d1"]}
    qrels = {"q1": {"d1"}, "q2": {"d1"}}
    m = compute_all_metrics(run, qrels, k_values=[1,2])
    assert "recall@1" in m
    assert "recall@2" in m

check("recall_at_k",          _recall)
check("ndcg_at_k",            _ndcg)
check("mrr",                  _mrr)
check("cra_at_k",             _cra)
check("decoy_promotion_rate", _dpr)
check("character_recall_at_k",_char_recall)
check("compute_all_metrics",  _compute_all)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Loss functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[3] Loss functions")

def _infonce():
    import torch
    import torch.nn.functional as F
    from src.training.losses import InfoNCELoss
    B, D = 4, 64
    q  = F.normalize(torch.randn(B, D), dim=-1)
    d  = F.normalize(torch.randn(B, D), dim=-1)
    hn = F.normalize(torch.randn(B, D), dim=-1)
    loss_fn = InfoNCELoss(temperature=0.07)
    loss = loss_fn(q, d, hn)
    assert loss.item() > 0
    loss_no_hn = loss_fn(q, d, None)
    assert loss_no_hn.item() > 0

def _margin_mse():
    import torch
    import torch.nn.functional as F
    from src.training.losses import MarginMSELoss
    B, D = 4, 64
    q   = F.normalize(torch.randn(B, D), dim=-1)
    pos = F.normalize(torch.randn(B, D), dim=-1)
    neg = F.normalize(torch.randn(B, D), dim=-1)
    loss = MarginMSELoss()(q, pos, neg)
    assert loss.item() >= 0

check("InfoNCELoss",   _infonce)
check("MarginMSELoss", _margin_mse)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. MambaEncoder (CPU fallback to BERT)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[4] MambaEncoder (CPU / BERT fallback)")

def _encoder_init():
    from src.models.encoder import MambaEncoder
    enc = MambaEncoder(device="cpu")
    return enc

def _encoder_encode():
    from src.models.encoder import MambaEncoder
    import torch
    enc  = MambaEncoder(device="cpu")
    embs = enc.encode(["the agreement shall terminate", "force majeure applies"], batch_size=2)
    assert embs.shape[0] == 2
    assert embs.shape[1] == enc.hidden_size
    # L2 norm ≈ 1
    norms = (embs ** 2).sum(dim=-1).sqrt()
    assert all(abs(n.item() - 1.0) < 1e-4 for n in norms), f"Not unit-normed: {norms}"

check("MambaEncoder init",   _encoder_init)
check("MambaEncoder encode", _encoder_encode)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. BiEncoder + FAISS index + retrieve
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[5] BiEncoder + FAISS")

def _biencoder_index_retrieve():
    from src.models.bi_encoder import BiEncoder
    be = BiEncoder(device="cpu", share_weights=True)
    docs = [
        "The agreement shall terminate upon written notice.",
        "Force majeure clauses excuse non-performance.",
        "Indemnification provisions allocate risk between parties.",
        "Governing law determines which jurisdiction's rules apply.",
    ]
    ids  = [f"doc_{i}" for i in range(len(docs))]
    be.build_index(docs, ids, batch_size=4)
    results = be.retrieve(["what terminates the agreement?"], top_k=2)
    assert len(results) == 1
    assert len(results[0]) == 2
    assert all("doc_id" in r for r in results[0])
    assert all("score"  in r for r in results[0])

def _biencoder_save_load():
    import tempfile
    from src.models.bi_encoder import BiEncoder
    be = BiEncoder(device="cpu")
    with tempfile.TemporaryDirectory() as tmp:
        be.save(Path(tmp))
        be2 = BiEncoder(device="cpu")
        be2.load(Path(tmp))

check("BiEncoder index+retrieve", _biencoder_index_retrieve)
check("BiEncoder save/load",      _biencoder_save_load)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. CrossPageVerifier
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[6] CrossPageVerifier")

def _verifier_score_pair():
    from src.models.verifier import CrossPageVerifier
    v = CrossPageVerifier(device="cpu")
    s = v.score_pair(
        "The contract expires on December 31.",
        "The contract expires on December 31."
    )
    assert 0.0 <= s <= 1.0

def _verifier_rerank():
    from src.models.verifier import CrossPageVerifier
    v = CrossPageVerifier(device="cpu")
    passages = [
        {"doc_id": "p1", "text": "Termination shall occur upon 30-day notice.",
         "score": 0.9, "page": 5},
        {"doc_id": "p2", "text": "Termination shall occur upon 60-day notice.",
         "score": 0.85, "page": 2},
    ]
    reranked = v.rerank(passages, top_k=2)
    assert len(reranked) == 2

check("CrossPageVerifier.score_pair", _verifier_score_pair)
check("CrossPageVerifier.rerank",     _verifier_rerank)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. RevRAG end-to-end (CPU, synthetic data)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[7] RevRAG end-to-end (CPU, synthetic)")

def _revrag_e2e():
    from src.models.revrag import RevRAG
    model = RevRAG(device="cpu", use_verifier=True)
    docs = [
        "The licensee shall pay royalties quarterly.",
        "Royalties are due annually under the amended agreement.",
        "Force majeure applies to acts of god and natural disasters.",
    ]
    ids = [f"d{i}" for i in range(len(docs))]
    model.index(docs, ids)
    results = model.retrieve(["when are royalties due?"], top_k=2)
    assert len(results) == 1
    assert len(results[0]) <= 2
    for r in results[0]:
        assert "doc_id" in r

check("RevRAG end-to-end", _revrag_e2e)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. ContractNLI data loader
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[8] ContractNLI data loader")

def _contractnli_loads():
    from src.config import CFG
    from src.data.contractnli import ContractNLIDataset
    data_dir = CFG.data.contractnli_dir
    if not (data_dir / "train.json").exists():
        raise FileNotFoundError(
            f"ContractNLI not found at {data_dir}. "
            "Run: git clone https://github.com/stanfordnlp/contract-nli data/contractnli"
        )
    ds = ContractNLIDataset(data_dir, split="train", seed=42)
    assert len(ds) > 0
    trip = ds.get_triplets()
    assert len(trip) > 0
    q, pos, hn = trip[0]
    assert isinstance(q,   str) and q
    assert isinstance(pos, str) and pos

check("ContractNLI loader", _contractnli_loads)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. Visualization (no display — Agg backend)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[9] Visualization (Agg backend)")

def _plot_loss():
    import tempfile, numpy as np
    from src.visualization.plots import plot_training_loss
    history = [{"epoch": i, "train_loss": 1.0/i, "val_loss": 1.2/i}
               for i in range(1, 6)]
    with tempfile.TemporaryDirectory() as tmp:
        plot_training_loss(history, Path(tmp))
        assert (Path(tmp) / "training_loss_curve.pdf").exists()

def _plot_main():
    import tempfile
    from src.visualization.plots import plot_main_results
    results = {
        "RevRAG": {"ContractNLI": 0.85, "MultiHopRAG": 0.70},
        "BM25":   {"ContractNLI": 0.55, "MultiHopRAG": 0.45},
    }
    with tempfile.TemporaryDirectory() as tmp:
        plot_main_results(results, "Recall@5", Path(tmp))
        assert (Path(tmp) / "main_results_recall_at_5.pdf").exists()

def _plot_heatmap():
    import tempfile, numpy as np
    from src.visualization.plots import plot_ablation_heatmap
    lv = [0.0, 0.3, 0.6, 1.0]
    tv = [0.3, 0.4, 0.5, 0.6, 0.7]
    grid = np.random.rand(len(lv), len(tv))
    with tempfile.TemporaryDirectory() as tmp:
        plot_ablation_heatmap(lv, tv, grid, Path(tmp))
        assert (Path(tmp) / "ablation_heatmap.pdf").exists()

check("plot_training_loss", _plot_loss)
check("plot_main_results",  _plot_main)
check("plot_ablation_heatmap", _plot_heatmap)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. Trainer (micro: 2 triplets, 1 epoch)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[10] Trainer (micro: 2 triplets, 1 epoch)")

def _trainer_micro():
    import tempfile
    from src.models.bi_encoder import BiEncoder
    from src.training.trainer  import BiEncoderTrainer
    from src.config import CFG

    triplets = [
        ("royalties are due quarterly",
         "the licensee pays royalties every quarter",
         "the agreement terminates upon written notice"),
        ("governing law is New York",
         "this agreement is governed by New York law",
         "force majeure applies to natural disasters"),
    ]

    CFG.training.max_epochs        = 1
    CFG.training.early_stop_patience = 99
    CFG.training.train_batch_size  = 2
    CFG.training.eval_batch_size   = 2

    be = BiEncoder(device="cpu")
    with tempfile.TemporaryDirectory() as tmp:
        trainer = BiEncoderTrainer(
            bi_encoder=be,
            train_triplets=triplets,
            val_triplets=triplets,
            out_dir=Path(tmp),
            device="cpu",
        )
        history = trainer.train()
        assert len(history) == 1
        assert "train_loss" in history[0]
        assert "val_loss"   in history[0]

check("Trainer micro-run", _trainer_micro)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 60)
passed = [r for r in results if r[1]]
failed = [r for r in results if not r[1]]
print(f"PASSED: {len(passed)}/{len(results)}")

if failed:
    print(f"\nFAILED ({len(failed)}):")
    for name, _, msg in failed:
        print(f"  {FAIL} {name}: {msg}")
    print("=" * 60)
    sys.exit(1)
else:
    print("All checks passed!")
    print("=" * 60)
    sys.exit(0)
