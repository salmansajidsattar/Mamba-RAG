"""Step 7 — Aggregate results: mean ± std across seeds, print summary table."""
import json, math
from collections import defaultdict
from pathlib import Path


KEY_METRICS = {
    "legalbench":      ("char_recall@5",              "Char-R@5"),
    "contractnli":     ("contradiction_char_recall@5", "Contra-R@5"),
    "cuad_amendments": ("cra@1",                       "CRA@1"),
    "multihop_rag":    ("recall@5",                    "R@5"),
    "loco":            ("recall@5",                    "R@5"),
    "synthetic":       ("cra@1",                       "CRA@1"),
}

SECONDARY_METRICS = {
    "legalbench":      ("char_precision@5",            "Char-P@5"),
    "contractnli":     ("contradiction_char_precision@5","Contra-P@5"),
    "cuad_amendments": ("decoy_promotion_rate",         "DPR↓"),
    "multihop_rag":    ("ndcg@10",                     "NDCG@10"),
    "loco":            ("ndcg@10",                     "NDCG@10"),
}


def _mean_std(vals):
    if not vals:
        return float("nan"), float("nan")
    n = len(vals)
    mean = sum(vals) / n
    var  = sum((v - mean) ** 2 for v in vals) / n if n > 1 else 0.0
    return mean, math.sqrt(var)


def run(work_dir: str) -> dict:
    results_dir = Path(work_dir) / "outputs"
    all_runs    = list(results_dir.glob("*_seed*.json"))
    print(f"  Found {len(all_runs)} result files in {results_dir}")

    grouped: dict = defaultdict(list)
    for f in all_runs:
        with open(f) as fh:
            r = json.load(fh)
        key = (r.get("model", "?"), r.get("dataset", "?"))
        grouped[key].append(r)

    summary = {}
    for (model, dataset), runs in sorted(grouped.items()):
        agg = {}
        all_keys = set().union(*[set(r.keys()) for r in runs])
        for k in all_keys:
            vals = [r[k] for r in runs if isinstance(r.get(k), (int, float))]
            if vals:
                mean, std = _mean_std(vals)
                agg[k] = {"mean": mean, "std": std, "n": len(vals)}
        summary[f"{model}/{dataset}"] = agg

    # Print table
    col = 42
    print(f"\n  {'Model / Dataset':<{col}} {'Metric':<20} {'Mean':>8}  {'±Std':>8}")
    print("  " + "─" * (col + 42))

    for key, metrics in sorted(summary.items()):
        _, dataset = key.split("/", 1)
        if dataset == "synthetic":
            continue  # print synthetic separately at the end

        m_key, m_name = KEY_METRICS.get(dataset, ("char_recall@5", "Char-R@5"))
        m = metrics.get(m_key, {})
        mean, std = m.get("mean", float("nan")), m.get("std", float("nan"))
        print(f"  {key:<{col}} {m_name:<20} {mean:>8.4f}  {std:>8.4f}")

    # Supplementary synthetic
    print("\n  --- Supplementary synthetic (diagnostic) ---")
    for key, metrics in sorted(summary.items()):
        _, dataset = key.split("/", 1)
        if dataset != "synthetic":
            continue
        m_key, m_name = KEY_METRICS["synthetic"]
        m = metrics.get(m_key, {})
        mean, std = m.get("mean", float("nan")), m.get("std", float("nan"))
        print(f"  {key:<{col}} {m_name:<20} {mean:>8.4f}  {std:>8.4f}")

    # Save
    out = results_dir / "summary_all.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  ✓ Summary saved → {out}")
    return summary
