"""
preprocess_local.py
====================
Run on your LOCAL machine. Reads raw datasets from F:\MambaRAG\data\,
produces compact JSON files, then zips them for Kaggle upload.

Output: F:\MambaRAG\revrag_processed\
  contractnli/
    train_triplets.json      [{query, positive, hard_neg}]
    val_triplets.json
    test_queries.json        [{query_id, query, label, doc_id, gold_texts}]
    passages.json            {passage_id: text}
  cuad/
    train_triplets.json
    val_triplets.json
    test_queries.json        [{query_id, query, gold_doc_id, decoy_doc_id, ...}]
    passages.json
  multihop/
    train_queries.json
    val_queries.json
    test_queries.json        [{query_id, query, gold_ids}]
    passages.json
  loco/
    test_queries.json
    passages.json

Upload zip to Kaggle Datasets as "revrag-processed".
"""
import json, sys, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("preprocess")

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR  = ROOT / "revrag_processed"
OUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))


def save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    size_mb = path.stat().st_size / 1e6
    n = len(obj) if isinstance(obj, (list, dict)) else "?"
    logger.info(f"  Saved {path.name}: {n} items ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# 1. ContractNLI
# ---------------------------------------------------------------------------
def process_contractnli():
    logger.info("\n[1/4] ContractNLI ...")
    from src.data.contractnli import ContractNLIDataset

    out        = OUT_DIR / "contractnli"
    cnli_dir   = DATA_DIR / "contractnli" / "resources" / "contract-nli"
    all_docs   = {}

    # ---- Train triplets ----
    ds_train = ContractNLIDataset(data_dir=cnli_dir, split="train", seed=42)
    all_docs.update(ds_train.documents)
    triplets = [{"query": q, "positive": p, "hard_neg": n}
                for q, p, n in ds_train.get_triplets()]
    save(triplets, out / "train_triplets.json")

    # ---- Val (dev) triplets — build manually since _build_triplets only runs for train ----
    ds_dev = ContractNLIDataset(data_dir=cnli_dir, split="dev", seed=42)
    all_docs.update(ds_dev.documents)
    # For dev we make simple (hypothesis, gold_span) pairs (no hard neg mining needed for val)
    val_triplets = []
    for q in ds_dev.queries:
        if q["label"] != "Entailment" or not q["has_gold_span"]:
            continue
        qid   = q["query_id"]
        spans = ds_dev.gold_spans.get(qid, [])
        if not spans:
            continue
        pos_text = spans[0][3]   # (doc_id, start, end, text) -> text
        # soft negative: random span from same doc
        doc_text = ds_dev.documents.get(q["doc_id"], "")
        import random
        rng = random.Random(42)
        rand_s = rng.randint(0, max(0, len(doc_text) - 200))
        hard_neg = doc_text[rand_s: rand_s + 200]
        val_triplets.append({"query": q["hypothesis"], "positive": pos_text,
                             "hard_neg": hard_neg})
    save(val_triplets, out / "val_triplets.json")

    # ---- Test queries with gold texts ----
    ds_test = ContractNLIDataset(data_dir=cnli_dir, split="test", seed=42)
    all_docs.update(ds_test.documents)
    test_queries = []
    for q in ds_test.queries:
        spans = ds_test.gold_spans.get(q["query_id"], [])
        test_queries.append({
            "query_id":   q["query_id"],
            "query":      q["hypothesis"],
            "label":      q["label"],
            "doc_id":     q["doc_id"],
            "gold_texts": [s[3] for s in spans],   # list of gold span texts
        })
    save(test_queries, out / "test_queries.json")

    # ---- Passages (all splits) ----
    logger.info("  Building passage index (all documents)...")
    passages = {}
    window, stride = 512, 256
    for doc_id, text in all_docs.items():
        for i, start in enumerate(range(0, max(1, len(text) - window + 1), stride)):
            pid = f"{doc_id}_p{i}"
            passages[pid] = text[start: start + window]
    save(passages, out / "passages.json")


# ---------------------------------------------------------------------------
# 2. CUAD
# ---------------------------------------------------------------------------
def process_cuad():
    logger.info("\n[2/4] CUAD ...")
    from src.data.cuad import CUADAmendmentDataset

    ds  = CUADAmendmentDataset(hf_cache_dir=DATA_DIR / "hf_cache", seed=42)
    out = OUT_DIR / "cuad"

    for split in ("train", "val", "test"):
        items = ds.get_split(split)

        triplets = [
            {"query": it["query"], "positive": it["gold_text"],
             "hard_neg": it["decoy_text"]}
            for it in items
            if it.get("gold_text") and it.get("decoy_text")
        ]
        save(triplets, out / f"{split}_triplets.json")

        queries = [
            {
                "query_id":     it["item_id"],
                "query":        it["query"],
                "category":     it["category"],
                "gold_text":    it.get("gold_text", ""),
                "decoy_text":   it.get("decoy_text", ""),
                "gold_doc_id":  f"{it['item_id']}_gold",
                "decoy_doc_id": f"{it['item_id']}_decoy",
            }
            for it in items
        ]
        save(queries, out / f"{split}_queries.json")

    # Passage index: each (item_id)_gold and _decoy as a "document"
    passages = {}
    for it in ds.items:
        passages[f"{it['item_id']}_gold"]  = it.get("gold_text", "")
        passages[f"{it['item_id']}_decoy"] = it.get("decoy_text", "")
    save(passages, out / "passages.json")


# ---------------------------------------------------------------------------
# 3. MultiHopRAG
# ---------------------------------------------------------------------------
def process_multihop():
    logger.info("\n[3/4] MultiHopRAG ...")
    from src.data.multihop import MultiHopRAGDataset

    out          = OUT_DIR / "multihop"
    all_passages = {}

    for split in ("train", "val", "test"):
        ds = MultiHopRAGDataset(cache_dir=DATA_DIR / "hf_cache",
                                split=split, seed=42)
        queries = [{"query_id": e["query_id"], "query": e["query"],
                    "gold_ids": e["gold_ids"]}
                   for e in ds.examples]
        save(queries, out / f"{split}_queries.json")
        all_passages.update(ds.corpus)

    save(all_passages, out / "passages.json")


# ---------------------------------------------------------------------------
# 4. LoCoV1
# ---------------------------------------------------------------------------
def process_loco():
    logger.info("\n[4/4] LoCoV1 ...")
    from src.data.loco import LoCoV1Dataset

    out = OUT_DIR / "loco"

    # Load all queries (use split="train" to get the large chunk, then save all)
    # We instantiate with split="test" but save only "test" queries;
    # the corpus is the same regardless of split.
    ds = LoCoV1Dataset(cache_dir=DATA_DIR / "hf_cache",
                       split="test",
                       max_queries=None,
                       max_docs=None,
                       seed=42)
    queries = [{"query_id": e["query_id"], "query": e["query"],
                "gold_ids": e["gold_ids"]}
               for e in ds.examples]
    save(queries, out / "test_queries.json")
    save(ds.get_corpus(), out / "passages.json")


# ---------------------------------------------------------------------------
# Zip
# ---------------------------------------------------------------------------
def create_zip():
    import zipfile
    zip_path = ROOT / "revrag_processed.zip"
    logger.info(f"\nZipping → {zip_path}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for fpath in sorted(OUT_DIR.rglob("*.json")):
            arc = fpath.relative_to(ROOT)
            zf.write(fpath, arc)
            logger.info(f"  + {arc}  ({fpath.stat().st_size/1e6:.1f} MB raw)")
    mb = zip_path.stat().st_size / 1e6
    logger.info(f"\nZip size: {mb:.1f} MB  →  {zip_path}")
    print("\n" + "=" * 60)
    print("Upload instructions")
    print("=" * 60)
    print("  1. kaggle.com → Datasets → New Dataset")
    print("  2. Drag in  revrag_processed.zip")
    print("  3. Name:  revrag-processed")
    print("  4. In notebook: Add Data → revrag-processed")
    print("     (appears at /kaggle/input/revrag-processed/)")
    print("  5. First cell: !python kaggle_load_processed.py")
    print("     Then:        !python train.py  or  !python evaluate.py")
    print("=" * 60)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    process_contractnli()
    process_cuad()
    process_multihop()
    process_loco()
    create_zip()
    logger.info("\nDone.")
