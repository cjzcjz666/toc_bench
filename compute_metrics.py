#!/usr/bin/env python3
"""
TOC-Bench Metrics
=================
Reads benchmark JSON + one prediction JSONL, scores predictions, prints
the standard report:

  1. Overall accuracy
  2. Macro-by-dim accuracy (mean of per-dim accuracies)
  3. Tier-weighted accuracy (Tier1×0.32 + Tier2×0.43 + Tier3×0.25)
  4. Hallucination Detection Accuracy (HDA), with per-variant break-down

Plus per-dim, per-format, and per-source tables.

Scoring:
  - mcq_4 / sp / numerical: extracted == correct_answer (string equality)
  - ordering: extracted == correct_order (list equality, both order-sensitive)
  - extracted is None or extraction failed → counted as INCORRECT (not abstain)

Usage:
    python compute_metrics.py \\
        --bench /path/to/toc_bench_clean_v2.json \\
        --preds /path/to/preds_<model>.jsonl

    # Compare multiple models in one table:
    python compute_metrics.py \\
        --bench .../toc_bench_clean_v2.json \\
        --preds preds_random.jsonl preds_frequency.jsonl preds_qwen7b.jsonl
"""

import argparse
import json
from collections import defaultdict, Counter
from pathlib import Path


# Tier weights (from TOC-Bench design: Tier1 32%, Tier2 43%, Tier3 25%)
TIER_WEIGHTS = {
    "tier1": 0.32,
    "tier2": 0.43,
    "tier3": 0.25,
}

# Random-chance baselines per format
RANDOM_CHANCE = {
    "mcq_4": 0.25,
    "sp": 0.50,
    "numerical": 0.25,
    "ordering_3": 1 / 6,    # 3! = 6 permutations
    "ordering_4": 1 / 24,   # 4! = 24 permutations
}


def is_correct(extracted, item):
    """Compare extracted answer to ground truth. Lists for ordering, str for others."""
    if extracted is None:
        return False
    fmt = item["format"]
    if fmt.startswith("ordering"):
        # Both should be lists
        gt = item.get("correct_order")
        if not gt:
            return False
        return list(extracted) == list(gt)
    return str(extracted) == str(item.get("correct_answer", ""))


def compute_one_model(items, preds_by_qa, model_name):
    """Compute all metrics for one model.

    Returns a dict: { 'overall', 'macro_dim', 'tier_weighted',
                      'per_dim', 'per_tier', 'per_format', 'per_source',
                      'hallucination', 'invalid_rate', ... }
    """
    n = len(items)
    if n == 0:
        return {}

    correct_total = 0
    invalid_count = 0
    by_dim = defaultdict(lambda: {"correct": 0, "total": 0})
    by_tier = defaultdict(lambda: {"correct": 0, "total": 0})
    by_format = defaultdict(lambda: {"correct": 0, "total": 0})
    by_source = defaultdict(lambda: {"correct": 0, "total": 0})

    # Hallucination break-down
    hallu_buckets = {
        "clean": {"correct": 0, "total": 0},
        "variant_a": {"correct": 0, "total": 0},
        "variant_b": {"correct": 0, "total": 0},
        "distractor_only": {"correct": 0, "total": 0},
    }

    missing_preds = 0

    for it in items:
        qa_id = it["qa_id"]
        if qa_id not in preds_by_qa:
            missing_preds += 1
            continue
        pred = preds_by_qa[qa_id]
        extracted = pred.get("extracted_answer")
        ok = is_correct(extracted, it)
        if extracted is None:
            invalid_count += 1

        correct_total += int(ok)

        meta = it.get("metadata", {})
        dim = meta.get("dim", "unknown")
        tier = meta.get("tier", "unknown")
        fmt = it["format"]
        source = it["video_id"].split("_")[0] or "unknown"

        for bucket, key in [
            (by_dim, dim), (by_tier, tier), (by_format, fmt), (by_source, source)
        ]:
            bucket[key]["total"] += 1
            bucket[key]["correct"] += int(ok)

        # Hallucination bucket
        hallu = meta.get("hallucination")
        if hallu == "variant_a":
            tag = "variant_a"
        elif hallu == "variant_b":
            tag = "variant_b"
        elif meta.get("has_hallucination_distractor"):
            tag = "distractor_only"
        else:
            tag = "clean"
        hallu_buckets[tag]["total"] += 1
        hallu_buckets[tag]["correct"] += int(ok)

    n_scored = n - missing_preds
    overall = correct_total / max(1, n_scored)

    per_dim = {d: v["correct"] / max(1, v["total"]) for d, v in by_dim.items()}
    per_tier = {t: v["correct"] / max(1, v["total"]) for t, v in by_tier.items()}
    per_format = {f: v["correct"] / max(1, v["total"]) for f, v in by_format.items()}
    per_source = {s: v["correct"] / max(1, v["total"]) for s, v in by_source.items()}

    # Macro-by-dim: mean of per-dim accuracies (skip dims with <10 items
    # to avoid noise; TVBench-style)
    valid_dims = [d for d in per_dim if by_dim[d]["total"] >= 10]
    macro_dim = (sum(per_dim[d] for d in valid_dims) / max(1, len(valid_dims))
                 if valid_dims else 0.0)

    # Tier-weighted accuracy
    tier_weighted = sum(
        per_tier.get(t, 0.0) * w for t, w in TIER_WEIGHTS.items())

    # HDA: only on hallu items (variant_a + variant_b)
    hda_total = (hallu_buckets["variant_a"]["total"]
                 + hallu_buckets["variant_b"]["total"])
    hda_correct = (hallu_buckets["variant_a"]["correct"]
                   + hallu_buckets["variant_b"]["correct"])
    hda = hda_correct / max(1, hda_total) if hda_total else None

    return {
        "model": model_name,
        "n_total": n,
        "n_scored": n_scored,
        "n_missing_preds": missing_preds,
        "n_invalid_extraction": invalid_count,
        "invalid_rate": invalid_count / max(1, n_scored),
        "overall": overall,
        "macro_dim": macro_dim,
        "tier_weighted": tier_weighted,
        "hda": hda,
        "per_dim": per_dim,
        "per_dim_counts": {d: v for d, v in by_dim.items()},
        "per_tier": per_tier,
        "per_tier_counts": {t: v for t, v in by_tier.items()},
        "per_format": per_format,
        "per_format_counts": {f: v for f, v in by_format.items()},
        "per_source": per_source,
        "per_source_counts": {s: v for s, v in by_source.items()},
        "hallucination_buckets": {
            k: {**v, "accuracy": v["correct"] / max(1, v["total"])}
            for k, v in hallu_buckets.items()
        },
    }


def load_preds(path):
    """Load JSONL into {qa_id -> pred_record}."""
    preds = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            qa_id = rec.get("qa_id")
            if not qa_id:
                continue
            # Last record per qa_id wins (in case of resume duplicates)
            preds[qa_id] = rec
    return preds


def fmt_pct(v, decimals=1):
    if v is None:
        return "  N/A "
    return f"{v*100:>5.{decimals}f}%"


def print_main_table(results):
    """Print the main comparison table: one row per model."""
    print("=" * 80)
    print("  Main Results (sorted by Overall Accuracy)")
    print("=" * 80)
    print(f"  {'Model':<25s} {'Overall':>9s} {'MacroDim':>9s} "
          f"{'TierWtd':>9s} {'HDA':>9s} {'Invalid%':>9s}")
    print(f"  {'-' * 25:<25s} {'-' * 9:>9s} {'-' * 9:>9s} "
          f"{'-' * 9:>9s} {'-' * 9:>9s} {'-' * 9:>9s}")
    rows = sorted(results, key=lambda r: -r["overall"])
    for r in rows:
        print(f"  {r['model']:<25s} {fmt_pct(r['overall']):>9s} "
              f"{fmt_pct(r['macro_dim']):>9s} {fmt_pct(r['tier_weighted']):>9s} "
              f"{fmt_pct(r['hda']):>9s} {fmt_pct(r['invalid_rate']):>9s}")
    print()


def print_per_dim_table(results):
    """Per-dim accuracy table: one row per dim, columns = models."""
    if not results:
        return

    # Collect all dims actually present
    all_dims = set()
    for r in results:
        all_dims.update(r["per_dim"].keys())
    dims_ordered = sorted(all_dims)

    print("=" * 100)
    print("  Per-Dimension Accuracy")
    print("=" * 100)
    header = f"  {'Dim':<28s} " + "".join(
        f"{r['model']:>15s}" for r in results)
    print(header)
    print(f"  {'-' * 28:<28s} " + "".join("-" * 15 for _ in results))

    for dim in dims_ordered:
        n_items = next(
            (r["per_dim_counts"].get(dim, {}).get("total", 0)
             for r in results if dim in r["per_dim_counts"]), 0)
        cells = []
        for r in results:
            v = r["per_dim"].get(dim)
            cnt = r["per_dim_counts"].get(dim, {}).get("total", 0)
            cells.append(f"{fmt_pct(v):>8s} ({cnt:>3d})")
        print(f"  {dim:<28s} " + "".join(f"{c:>15s}" for c in cells))
    print()


def print_per_tier_table(results):
    print("=" * 80)
    print("  Per-Tier Accuracy")
    print("=" * 80)
    header = f"  {'Tier':<10s} " + "".join(f"{r['model']:>15s}" for r in results)
    print(header)
    print(f"  {'-' * 10:<10s} " + "".join("-" * 15 for _ in results))
    for tier in ["tier1", "tier2", "tier3"]:
        cells = []
        for r in results:
            v = r["per_tier"].get(tier)
            cnt = r["per_tier_counts"].get(tier, {}).get("total", 0)
            cells.append(f"{fmt_pct(v):>8s} ({cnt:>3d})")
        print(f"  {tier:<10s} " + "".join(f"{c:>15s}" for c in cells))
    print()


def print_hallucination_table(results):
    print("=" * 80)
    print("  Hallucination Detection Break-down")
    print("=" * 80)
    header = f"  {'Bucket':<20s} " + "".join(f"{r['model']:>15s}" for r in results)
    print(header)
    print(f"  {'-' * 20:<20s} " + "".join("-" * 15 for _ in results))
    for bucket in ["clean", "distractor_only", "variant_a", "variant_b"]:
        cells = []
        for r in results:
            b = r["hallucination_buckets"].get(bucket, {})
            v = b.get("accuracy")
            cnt = b.get("total", 0)
            cells.append(f"{fmt_pct(v):>8s} ({cnt:>3d})")
        print(f"  {bucket:<20s} " + "".join(f"{c:>15s}" for c in cells))
    print()


def print_per_format_table(results):
    print("=" * 80)
    print("  Per-Format Accuracy (with random-chance baselines)")
    print("=" * 80)
    header = (f"  {'Format':<14s} {'Chance':>8s} "
              + "".join(f"{r['model']:>15s}" for r in results))
    print(header)
    print(f"  {'-' * 14:<14s} {'-' * 8:>8s} "
          + "".join("-" * 15 for _ in results))
    for fmt in ["mcq_4", "sp", "numerical", "ordering_3", "ordering_4"]:
        cells = []
        for r in results:
            v = r["per_format"].get(fmt)
            cnt = r["per_format_counts"].get(fmt, {}).get("total", 0)
            cells.append(f"{fmt_pct(v):>8s} ({cnt:>3d})")
        chance = fmt_pct(RANDOM_CHANCE.get(fmt))
        print(f"  {fmt:<14s} {chance:>8s} " + "".join(f"{c:>15s}" for c in cells))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True,
                    help="path to toc_bench_clean_v2.json")
    ap.add_argument("--preds", nargs="+", required=True,
                    help="One or more prediction JSONL files (one per model)")
    ap.add_argument("--save-summary", default=None,
                    help="Optional: write all metrics as JSON")
    args = ap.parse_args()

    with open(args.bench) as f:
        bench = json.load(f)
    items = bench["items"]

    results = []
    for pred_path in args.preds:
        preds = load_preds(pred_path)
        # Infer model name: prefer first record's "model" field
        model_name = next(iter(preds.values())).get("model") if preds else None
        if not model_name:
            model_name = Path(pred_path).stem.replace("preds_", "")
        r = compute_one_model(items, preds, model_name)
        results.append(r)

    print()
    print_main_table(results)
    print_per_format_table(results)
    print_per_tier_table(results)
    print_hallucination_table(results)
    print_per_dim_table(results)

    if args.save_summary:
        with open(args.save_summary, "w") as f:
            # Strip the embedded counter dicts (not JSON-friendly)
            clean = []
            for r in results:
                rc = {k: v for k, v in r.items()
                      if not k.endswith("_counts")}
                clean.append(rc)
            json.dump(clean, f, indent=2)
        print(f"Wrote summary to {args.save_summary}")


if __name__ == "__main__":
    main()