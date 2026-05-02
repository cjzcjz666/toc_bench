#!/usr/bin/env python3
"""
TOC-Bench Step 4b: Sample for Human Verification
==================================================
Splits the combined filter output (_combined.json) into three tiered files:

  1. toc_bench_verified.json    — target ~5000 items for human annotation,
                                   stratified by dim with per-dim caps and
                                   small dims retained fully
  2. toc_bench_large.json       — the complement (unverified items, kept
                                   for scale studies and reported as
                                   TOC-Bench-Large in the paper)
  3. toc_bench_quality_sample.json  — stratified subsample of Large used
                                       to estimate the noise level of the
                                       unverified subset after verification
                                       is complete

  4. _sampling_summary.json     — full provenance: targets, realized counts,
                                   per-dim ratios, seed

Video-level constraint:
  Sampling is at the qa-item level, not the video level. This means
  Verified and Large may share videos (different QA items on the same
  video). That's fine — Verified is the evaluation set; Large is the
  scale set; the same underlying videos can serve both.

Usage:
    python step4b_sample_for_verification.py
    python step4b_sample_for_verification.py --seed 42 --verified-target 5000
    python step4b_sample_for_verification.py --quality-sample-size 500
    python step4b_sample_for_verification.py --dim-targets-json caps.json
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import QA_DIR, DIMENSIONS

FILTER_DIR = QA_DIR / "filtered_natural"
OUT_DIR = QA_DIR / "human_verification_natural"


# ============================================================
# Default per-dim targets for Verified subset
# ============================================================
# Rationale:
#   - Small dims (≤ ~300 items) kept entirely — already rare, can't
#     afford to lose statistical power.
#   - Large dims downsampled, but each dim gets at least ~300 items for
#     meaningful per-dim ablation.
#   - Total target: ~5000 items (tracks NeurIPS D&B benchmark scale:
#     TVBench=2654, Video-MME=2700, MVBench=4000).
#
# Adjust via --dim-targets-json if needed.

DEFAULT_DIM_TARGETS = {
    "reappear_identity":      37,    # keep all (small + under review)
    "event_ordering":         175,   # keep all (tier2, C-critical, small)
    "cross_object_order":     298,   # keep all (tier3, C-critical, small)
    "reappear_or_disappear":  243,   # keep all (tier2 SP, small)
    "event_existence":        360,   # downsample from 438
    "relative_spatial_change": 440,  # downsample from 870 (low-survival dim)
    "event_count":            500,   # downsample from 969 (tier2 C-critical)
    "conditional_state":      700,   # downsample from 1941 (tier3 C-critical)
    "duration_category":      800,   # downsample from 4463 (tier1)
    "temporal_location":     1447,   # downsample from 8466 (largest dim)
}
# Sum = 37+175+298+243+360+440+500+700+800+1447 = 5000

DEFAULT_VERIFIED_TARGET = 5000
DEFAULT_QUALITY_SAMPLE_SIZE = 500


# ============================================================
# Sampling
# ============================================================

def sample_verified(items, dim_targets, seed):
    """Per-dim stratified sample. Returns (verified_items, large_items,
    per_dim_stats)."""
    rng = random.Random(seed)

    # Group by dim
    by_dim = defaultdict(list)
    for it in items:
        by_dim[it.get("dim", "unknown")].append(it)

    verified = []
    large = []
    stats = {}

    for dim, pool in by_dim.items():
        target = dim_targets.get(dim, len(pool))  # dims w/o target = keep all
        pool_copy = list(pool)
        rng.shuffle(pool_copy)

        if target >= len(pool_copy):
            # Keep all, nothing for large from this dim
            verified.extend(pool_copy)
            stats[dim] = {
                "available": len(pool_copy),
                "target": target,
                "verified": len(pool_copy),
                "large": 0,
            }
        else:
            verified.extend(pool_copy[:target])
            large.extend(pool_copy[target:])
            stats[dim] = {
                "available": len(pool_copy),
                "target": target,
                "verified": target,
                "large": len(pool_copy) - target,
            }

    # Final shuffle so that adjacent items in output aren't clustered
    # by dim (makes Excel / annotation more natural)
    rng.shuffle(verified)
    rng.shuffle(large)

    return verified, large, stats


def sample_quality(large_items, n, seed):
    """Proportional stratified sample from Large, for quality estimation.
    Each dim contributes proportionally to its size in Large."""
    if not large_items or n <= 0:
        return []

    rng = random.Random(seed + 1)  # different seed from verified sampling

    by_dim = defaultdict(list)
    for it in large_items:
        by_dim[it.get("dim", "unknown")].append(it)

    total_large = sum(len(v) for v in by_dim.values())
    if total_large == 0:
        return []

    # Allocate proportional quotas; largest-remainder rounding to hit n exactly
    raw_quotas = {
        dim: n * len(pool) / total_large for dim, pool in by_dim.items()
    }
    integer_quotas = {dim: int(q) for dim, q in raw_quotas.items()}
    shortfall = n - sum(integer_quotas.values())
    # Distribute shortfall to dims with largest fractional remainder
    remainders = sorted(
        ((raw_quotas[d] - integer_quotas[d], d) for d in integer_quotas),
        reverse=True,
    )
    for _, dim in remainders[:shortfall]:
        integer_quotas[dim] += 1

    sample = []
    for dim, quota in integer_quotas.items():
        pool = by_dim[dim]
        rng.shuffle(pool)
        quota = min(quota, len(pool))
        sample.extend(pool[:quota])

    rng.shuffle(sample)
    return sample


# ============================================================
# Reporting
# ============================================================

def print_sampling_report(stats, verified, large, quality):
    print("\n" + "=" * 72)
    print("  Sampling Report")
    print("=" * 72)

    print(f"\n  Verified size:       {len(verified):>6,d}")
    print(f"  Large size:          {len(large):>6,d}")
    print(f"  Quality sample size: {len(quality):>6,d}")
    print(f"  Total input:         {len(verified) + len(large):>6,d}")

    print(f"\n  Per-dim breakdown (Verified subset):")
    print(f"    {'dim':<28s} {'avail':>7s} {'target':>7s} "
          f"{'verif':>7s} {'large':>7s}")
    print(f"    {'-'*28} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    # Sort by verified count descending
    for dim in sorted(stats, key=lambda d: -stats[d]["verified"]):
        s = stats[dim]
        c_mark = "★" if DIMENSIONS.get(dim, {}).get("c_critical") else " "
        print(f"    {c_mark} {dim:<26s} {s['available']:>7,d} "
              f"{s['target']:>7,d} {s['verified']:>7,d} "
              f"{s['large']:>7,d}")

    # Tier breakdown
    print(f"\n  Tier distribution:")
    ver_tier = Counter(it.get("tier", "?") for it in verified)
    lar_tier = Counter(it.get("tier", "?") for it in large)
    print(f"    {'tier':<8s} {'verified':>10s} {'large':>10s}")
    for tier in ["tier1", "tier2", "tier3"]:
        v = ver_tier.get(tier, 0)
        l = lar_tier.get(tier, 0)
        vp = 100 * v / max(1, len(verified))
        lp = 100 * l / max(1, len(large))
        print(f"    {tier:<8s} {v:>6,d}({vp:>4.1f}%) {l:>6,d}({lp:>4.1f}%)")

    # Format breakdown
    print(f"\n  Format distribution:")
    ver_fmt = Counter(it.get("format", "?") for it in verified)
    for fmt, cnt in ver_fmt.most_common():
        pct = 100 * cnt / max(1, len(verified))
        print(f"    {fmt:<12s} {cnt:>6,d}  ({pct:>4.1f}%)")

    # Hallucination coverage in verified
    hallu_ver = Counter()
    for it in verified:
        tag = it.get("hallucination")
        if it.get("has_hallucination_distractor"):
            tag = tag or "distractor_only"
        hallu_ver[tag or "none"] += 1
    print(f"\n  Hallucination coverage in Verified:")
    for tag, cnt in sorted(hallu_ver.items()):
        pct = 100 * cnt / max(1, len(verified))
        print(f"    {str(tag):<24s} {cnt:>6,d}  ({pct:>4.1f}%)")

    # Unique videos
    ver_vids = len(set(it["video_id"] for it in verified))
    lar_vids = len(set(it["video_id"] for it in large))
    overlap_vids = len(
        set(it["video_id"] for it in verified) &
        set(it["video_id"] for it in large)
    )
    qual_vids = len(set(it["video_id"] for it in quality))
    print(f"\n  Unique videos:")
    print(f"    Verified: {ver_vids:>6,d}")
    print(f"    Large:    {lar_vids:>6,d}")
    print(f"    Overlap:  {overlap_vids:>6,d}  "
          f"(same video in both subsets with different QAs — this is fine)")
    print(f"    Quality:  {qual_vids:>6,d}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="TOC-Bench Step 4b: Sample for human verification"
    )
    parser.add_argument("--input", type=str, default=None,
                        help=f"Input _combined.json (default: "
                             f"{FILTER_DIR}/_combined.json)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help=f"Output directory (default: {OUT_DIR})")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verified-target", type=int,
                        default=DEFAULT_VERIFIED_TARGET,
                        help="Approximate target for Verified subset. "
                             "Per-dim targets are summed; deviation from "
                             "this number is reported but not enforced.")
    parser.add_argument("--quality-sample-size", type=int,
                        default=DEFAULT_QUALITY_SAMPLE_SIZE,
                        help="Number of items to sample from Large for "
                             "quality estimation (default 500).")
    parser.add_argument("--dim-targets-json", type=str, default=None,
                        help="Path to a JSON file overriding per-dim "
                             "targets. Schema: {\"dim_name\": int_count}. "
                             "Dims not in the file use DEFAULT_DIM_TARGETS "
                             "or keep all if unspecified there either.")
    args = parser.parse_args()

    print("=" * 72)
    print("  TOC-Bench Step 4b: Sampling for Human Verification")
    print("=" * 72)

    # Resolve paths
    input_path = (Path(args.input) if args.input
                  else FILTER_DIR / "_combined.json")
    out_dir = Path(args.output_dir) if args.output_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"  [ERROR] Input not found: {input_path}")
        print(f"  Run step4 --combine first.")
        sys.exit(1)

    # Load
    with open(input_path) as f:
        data = json.load(f)
    items = data.get("qa_items", [])
    print(f"  Loaded {len(items):,} items from {input_path.name}")

    # Per-dim targets
    dim_targets = dict(DEFAULT_DIM_TARGETS)
    if args.dim_targets_json:
        with open(args.dim_targets_json) as f:
            user_targets = json.load(f)
        dim_targets.update(user_targets)
        print(f"  Loaded {len(user_targets)} per-dim overrides from "
              f"{args.dim_targets_json}")

    expected_sum = sum(dim_targets.values())
    print(f"  Verified target (from per-dim targets): {expected_sum:,}")
    if abs(expected_sum - args.verified_target) > 100:
        print(f"  [WARN] Per-dim targets sum ({expected_sum}) differs from "
              f"--verified-target ({args.verified_target}) by more than 100.")

    # Sample verified + large
    print(f"\n  Running stratified sampling (seed={args.seed})...")
    verified, large, stats = sample_verified(items, dim_targets, args.seed)

    # Sample quality subset from large
    quality = sample_quality(large, args.quality_sample_size, args.seed)
    print(f"  Quality sample: {len(quality):,} items from Large "
          f"(target {args.quality_sample_size})")

    # Report
    print_sampling_report(stats, verified, large, quality)

    # Save
    verified_path = out_dir / "toc_bench_verified.json"
    large_path = out_dir / "toc_bench_large.json"
    quality_path = out_dir / "toc_bench_quality_sample.json"
    summary_path = out_dir / "_sampling_summary.json"

    with open(verified_path, "w") as f:
        json.dump({
            "subset": "verified",
            "description": (
                "Items selected for human verification. Per-dim stratified "
                "sampling with caps to balance dimensions while preserving "
                "small dims entirely."
            ),
            "total_items": len(verified),
            "qa_items": verified,
        }, f, indent=2)

    with open(large_path, "w") as f:
        json.dump({
            "subset": "large",
            "description": (
                "Items NOT selected for human verification — forms "
                "TOC-Bench-Large for scale studies. Quality estimated via "
                "toc_bench_quality_sample.json after verification."
            ),
            "total_items": len(large),
            "qa_items": large,
        }, f, indent=2)

    with open(quality_path, "w") as f:
        json.dump({
            "subset": "quality_sample",
            "description": (
                "Stratified subsample of Large, proportional to per-dim "
                "size. Used after annotators verify it to estimate noise "
                "level of TOC-Bench-Large as a whole."
            ),
            "total_items": len(quality),
            "sampled_from": "toc_bench_large.json",
            "qa_items": quality,
        }, f, indent=2)

    # Summary provenance
    summary = {
        "input": str(input_path),
        "input_total_items": len(items),
        "seed": args.seed,
        "verified_target": args.verified_target,
        "quality_sample_size": args.quality_sample_size,
        "dim_targets": dim_targets,
        "realized": {
            "verified_count": len(verified),
            "large_count": len(large),
            "quality_count": len(quality),
        },
        "per_dim_stats": stats,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Final message
    print(f"\n{'='*72}")
    print(f"  Files written")
    print(f"{'='*72}")
    print(f"  {verified_path.name:<36s}  {len(verified):>6,d} items")
    print(f"  {large_path.name:<36s}  {len(large):>6,d} items")
    print(f"  {quality_path.name:<36s}  {len(quality):>6,d} items")
    print(f"  {summary_path.name:<36s}  (provenance)")

    print(f"\n  Next steps:")
    print(f"    1. Run step5a with --input {verified_path}")
    print(f"       → produces annotation_sheet.xlsx for the 5000 "
          f"verified items")
    print(f"    2. Run step5a with --input {quality_path} "
          f"--output-suffix _quality")
    print(f"       → produces a second Excel for the 500 quality-check items")
    print(f"    3. After annotation, step5c merges both back into final "
          f"benchmark files.")


if __name__ == "__main__":
    main()