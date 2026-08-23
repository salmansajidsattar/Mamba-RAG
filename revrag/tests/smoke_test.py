"""
tests/smoke_test.py — CPU smoke test for the full RevRAG pipeline.

Runs BEFORE the main GPU eval to catch import errors, missing files,
and broken loaders early — without wasting GPU time.

What it tests (all on CPU, tiny data slices):
  ✓ Config YAML parses correctly
  ✓ All dataset loaders import and return data
  ✓ BM25 retrieval returns ranked results
  ✓ BGE bi-encoder initialises and retrieves on CPU
  ✓ Mamba encoder (or GRU stub) initialises on CPU
  ✓ NLI verifier loads and scores a passage pair
  ✓ RevRAG full pipeline (no verifier) runs end-to-end on CPU
  ✓ Char-overlap, CRA@1, Recall@k metrics compute correctly

Target runtime: < 2 minutes on CPU.

Usage:
    python tests/smoke_test.py
    # or via run_experiment.py step 5 (always CPU regardless of DEVICE setting)
"""
import sys
import json
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # revrag/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))          # project root (for config.py)

DEVICE = "cpu"   # ALWAYS cpu — that's the point of a smoke test

PASS = "  ✓"
FAIL = "  ✗"
_results = []

TINY_CORPUS = [
    "The agreement shall be kept confidential for a period of three years.",
    "The receiving party may disclose information to its legal advisors.",
    "All amendments to this agreement must be made in writing and signed.",
    "The agreement is governed by the laws of the State of California.",
    "Either party may terminate this agreement with thirty days written notice.",
    "Notwithstanding the above, the confidentiality period is hereby amended to five years.",
]
TINY_QUERY = "What is the confidentiality period?"


def check(name: str, fn, required: bool = True):
    t = time.time()
    try:
        fn()
        print(f"{PASS} {name}  ({time.time()-t:.1f}s)")
        _results.append((name, True, None, required))
    except Exception as e:
        print(f"{FAIL} {name}  ({time.time()-t:.1f}s)")
        print(f"     {type(e).__name__}: {e}")
        _results.append((name, False, traceback.format_exc(), required))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_config():
    import yaml
    cfg_path = ROOT / "configs" / "default.yaml"
    assert cfg_path.exists(), f"Missing: {cfg_path}"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    assert "synthetic"  in cfg, "missing 'synthetic' section"
    assert "bgem3"      in cfg, "missing 'bgem3' section"
    assert "contractnli" in cfg, "missing 'contractnli' section"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dataset loaders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_legalbench_loader():
    data_dir = ROOT / "data" / "raw" / "legalbench_rag"
    assert data_dir.exists(), f"Run step 3 first — not found: {data_dir}"
    from data.legalbench import LegalBenchRAGDataset
    import yaml
    with open(ROOT / "configs" / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    ds   = LegalBenchRAGDataset(cfg)
    docs = ds.get_all_documents()
    assert len(docs) > 0


def test_contractnli_loader():
    data_dir = ROOT / "data" / "raw" / "contractnli"
    assert data_dir.exists(), f"Run step 3 first — not found: {data_dir}"
    from data.contractnli_retrieval import ContractNLIRetrievalDataset
    ds = ContractNLIRetrievalDataset(data_dir=data_dir, split="test")
    assert len(ds) > 0, "ContractNLI loaded 0 queries"
    q = ds.queries[0]
    assert "hypothesis" in q and "label" in q, f"Unexpected keys: {list(q.keys())}"
    # Quick metric sanity
    p, r = ds.character_overlap(q["query_id"], [q["hypothesis"]])
    assert isinstance(p, float) and isinstance(r, float)


def test_cuad_loader():
    from data.cuad_amendments import CUADAmendmentDataset
    import yaml
    with open(ROOT / "configs" / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    cache = str(ROOT / "data" / "raw" / "cuad")
    ds = CUADAmendmentDataset(cfg, cache_dir=cache)
    assert len(ds) > 0


def test_multihop_loader():
    from data.multihop_rag import MultiHopRAGDataset
    import yaml
    with open(ROOT / "configs" / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    cache = str(ROOT / "data" / "raw" / "multihop_rag")
    ds = MultiHopRAGDataset(cfg, cache_dir=cache, max_queries=5)
    assert len(ds) > 0


def test_synthetic_loader():
    syn = ROOT / "data" / "synthetic" / "cross_page_bench.jsonl"
    assert syn.exists(), f"Run step 4 first — not found: {syn}"
    items = [json.loads(l) for l in open(syn)][:5]
    assert len(items) > 0
    assert "query" in items[0]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Retrieval models — CPU init + tiny retrieval
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_bm25():
    from models.bm25_retriever import BM25Retriever
    r = BM25Retriever()
    r.index(TINY_CORPUS)
    hits = r.retrieve(TINY_QUERY, top_k=3)
    assert len(hits) == 3 and "text" in hits[0]


def test_bge():
    from models.bge_retriever import BGERetriever
    r = BGERetriever(device=DEVICE)
    r.index(TINY_CORPUS)
    hits = r.retrieve(TINY_QUERY, top_k=3)
    assert len(hits) == 3


def test_mamba_encoder():
    """Accept GRU stub if mamba-ssm is not yet installed (GPU-only wheel)."""
    try:
        from models.mamba_encoder import MambaEncoder
        import torch
        enc   = MambaEncoder(device=DEVICE)
        dummy = torch.randn(1, 16, enc.hidden_dim)
        out   = enc(dummy)
        assert out.shape[-1] == enc.hidden_dim
    except ImportError:
        print("\n     (mamba-ssm not installed — GRU stub expected on GPU session)")


def test_revrag_pipeline():
    from models.revrag import RevRAGRetriever
    import yaml
    with open(ROOT / "configs" / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    r = RevRAGRetriever(cfg, device=DEVICE, use_verifier=False)
    r.index(TINY_CORPUS)
    hits = r.retrieve(TINY_QUERY, top_k=3)
    assert len(hits) == 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NLI verifier
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_verifier():
    from models.verifier import CrossPageVerifier
    import yaml
    with open(ROOT / "configs" / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    v     = CrossPageVerifier(cfg, device=DEVICE)
    score = v.score(
        "The confidentiality period is three years.",
        "The agreement is confidential for a period of three years.",
    )
    assert isinstance(score, float), f"Expected float, got {type(score)}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_char_overlap():
    from evaluate.metrics import character_precision_recall
    gold = "The agreement shall be kept confidential for three years."
    ret  = ["The agreement shall be kept confidential for three years from signing."]
    p, r = character_precision_recall(gold, ret, overlap_threshold=0.5)
    assert r > 0.0, f"Expected recall > 0, got {r}"


def test_cra_at_1():
    from evaluate.metrics import cra_at_k
    gold  = [{"text": "Amendment: 60-day notice.",  "is_gold": True},
             {"text": "Original: 30-day notice.",   "is_gold": False}]
    bad   = [{"text": "Original: 30-day notice.",   "is_gold": False},
             {"text": "Amendment: 60-day notice.",  "is_gold": True}]
    assert cra_at_k(gold, k=1) == 1.0
    assert cra_at_k(bad,  k=1) == 0.0


def test_recall_at_k():
    from evaluate.metrics import recall_at_k
    gold = {"doc_1", "doc_3"}
    ret  = ["doc_1", "doc_2", "doc_3", "doc_4"]
    assert recall_at_k(gold, ret, k=4) == 1.0
    assert recall_at_k(gold, ret, k=1) == 0.5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Runner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TESTS = [
    # (label,                     function,             required)
    ("Config YAML",               test_config,          True),
    ("LegalBench loader",         test_legalbench_loader, True),
    ("ContractNLI loader",        test_contractnli_loader, True),
    ("CUAD loader",               test_cuad_loader,     True),
    ("MultiHop-RAG loader",       test_multihop_loader, True),
    ("Synthetic loader",          test_synthetic_loader, False),
    ("BM25 retriever",            test_bm25,            True),
    ("BGE retriever (CPU)",       test_bge,             True),
    ("Mamba encoder (CPU/stub)",  test_mamba_encoder,   False),
    ("RevRAG pipeline (CPU)",     test_revrag_pipeline, True),
    ("NLI verifier (CPU)",        test_verifier,        True),
    ("Char-overlap metric",       test_char_overlap,    True),
    ("CRA@1 metric",              test_cra_at_1,        True),
    ("Recall@k metric",           test_recall_at_k,     True),
]


def main():
    t0 = time.time()
    print("\n" + "━"*62)
    print("  RevRAG CPU Smoke Test")
    print("  device=cpu — verifying pipeline before GPU eval")
    print("━"*62)

    for label, fn, required in TESTS:
        check(label, fn, required)

    passed          = sum(1 for _, ok, _, _ in _results if ok)
    failed_required = [(n, tb) for n, ok, tb, req in _results if not ok and req]
    failed_optional = [(n, _)  for n, ok, _, req  in _results if not ok and not req]

    print("\n" + "━"*62)
    print(f"  {passed}/{len(_results)} checks passed  ({time.time()-t0:.1f}s)")

    if failed_optional:
        print(f"  ─ Optional failures (non-blocking): "
              f"{[n for n,_ in failed_optional]}")

    if failed_required:
        print(f"\n  ⛔ REQUIRED checks failed — fix before GPU eval:")
        for name, tb in failed_required:
            print(f"     ✗ {name}")
            print(f"       {tb.splitlines()[-1]}")
        sys.exit(1)
    else:
        print("\n  ✅ All required checks passed — safe to run GPU eval.")
        sys.exit(0)


if __name__ == "__main__":
    main()
