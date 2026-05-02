#!/usr/bin/env python3
"""
TOC-Bench Step 2: Phenomenon-First Video Selection
====================================================
Selects ~1,000 videos from the 13,000+ that passed Step 1c filtering.

Design principles (in priority order):
  1. Phenomenon balance  — every temporal phenomenon slot has enough videos
  2. Duration balance    — short / medium / long videos are represented
  3. Object density      — sparse / moderate / dense scenes are represented
  4. Source balance      — no single source dominates (soft cap)

Algorithm: greedy weighted set cover.
  Each round picks the video that fills the largest deficit across
  unfilled phenomenon slots, with duration / density / source bonuses
  and penalties layered on top.

Input:
    filtered_videos.json  (from Step 1c, contains passed videos with
                           phenomenon_profile, quality_score, stats)

Output:
    selected_videos.json  (selected subset with selection metadata)

Usage:
    python step2_select_videos.py
    python step2_select_videos.py --total 800
    python step2_select_videos.py --dry-run          # print stats only
    python step2_select_videos.py --analyze-only      # analyze existing selection
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import ROOT, FILTERED_DIR

# ============================================================
# Selection Configuration
# ============================================================

# --- Phenomenon slots (Priority 1: hard targets) ---
# Each slot defines a phenomenon the benchmark must evaluate.
# A video can satisfy multiple slots simultaneously.
PHENOMENON_SLOTS = {
    # ========== v1 slots (legacy, retained for backward compat) ==========
    "occlusion_reappear": {
        "description": "Object fully occluded then reappears",
        "profile_key": "occlusion_reappear",
        "target_videos": 200,
        "weight": 1.0,         # all slots equal weight by default
    },
    "partial_occlusion": {
        "description": "Object partially blocked but still visible",
        "profile_key": "partial_occlusion",
        "target_videos": 200,
        "weight": 1.0,
    },
    "exit_reenter": {
        "description": "Object exits frame then re-enters or reappears",
        "profile_key": "exit_reenter",
        "target_videos": 150,
        "weight": 1.0,
    },
    "appear_disappear": {
        "description": "Object appears mid-video or disappears permanently",
        "profile_key": "appear_disappear",
        "target_videos": 120,
        "weight": 0.8,
    },
    "state_change": {
        "description": "Sudden area or position jump",
        "profile_key": "state_change",
        "target_videos": 100,
        "weight": 0.6,
    },
    "interaction": {
        "description": "Two objects interact (sustained bbox overlap)",
        "profile_key": "interaction",
        "target_videos": 250,
        "weight": 1.0,
    },
    "multi_phenomenon": {
        "description": "Video contains >= 3 distinct meaningful event types",
        "profile_key": "multi_phenomenon",
        "target_videos": 150,
        "weight": 1.2,        # bonus: compound videos are most valuable
    },

    # ========== v2 slots (new) — support Tier 2/3 feasibility ==========
    "v2_reappear_identity": {
        "description": "Has sibling-based reappear identity question candidate (T2.3)",
        "profile_key": "has_reappear_identity_candidate",
        "target_videos": 300,
        "weight": 1.5,        # higher weight: this dim is rare & valuable
    },
    "v2_conditional_state": {
        "description": "Has cross-base-noun covisible pair for T3.1",
        "profile_key": "has_conditional_state_pair",
        "target_videos": 400,
        "weight": 1.3,
    },
    "v2_countable_event": {
        "description": "Has at least one repeated event type for T2.1 event_count",
        "profile_key": "has_countable_event",
        "target_videos": 300,
        "weight": 1.3,
    },
    "v2_cross_event_chain": {
        "description": "Has an object with ≥3 distinct event types (T2.2 ordering)",
        "profile_key": "has_cross_event_chain",
        "target_videos": 300,
        "weight": 1.2,
    },
    "v2_hi_confidence": {
        "description": "≥2 objects with instance_confidence ≥ 0.7 (C-gate)",
        "profile_key": "has_high_confidence_objects",
        "target_videos": 500,
        "weight": 1.0,
    },
}

# --- Duration buckets (Priority 2: soft targets) ---
DURATION_TARGETS = {
    "short":  {"range": (5, 15),   "ratio": 0.15},
    "medium": {"range": (15, 40),  "ratio": 0.55},
    "long":   {"range": (40, 90),  "ratio": 0.30},
}

# --- Object density buckets (Priority 3: soft targets) ---
DENSITY_TARGETS = {
    "sparse":   {"range": (2, 4),   "ratio": 0.10},
    "moderate": {"range": (5, 8),   "ratio": 0.45},
    "dense":    {"range": (9, 999), "ratio": 0.45},
}

# --- Source cap (Priority 4: prevent extreme skew) ---
MAX_SOURCE_RATIO = 0.45

# --- Scoring weights ---
PHENOMENON_SCORE_SCALE = 10.0   # base scale for phenomenon deficit
DURATION_BONUS_SCALE = 2.0
DENSITY_BONUS_SCALE = 1.5
SOURCE_PENALTY = -3.0


# ============================================================
# Classification Helpers
# ============================================================

def classify_duration_bucket(duration_sec):
    """Classify video duration into a bucket name."""
    for name, cfg in DURATION_TARGETS.items():
        lo, hi = cfg["range"]
        if lo <= duration_sec < hi:
            return name
    # Edge case: exactly at upper boundary of long
    if duration_sec >= DURATION_TARGETS["long"]["range"][0]:
        return "long"
    return "short"


def classify_density_bucket(num_objects):
    """Classify tracked object count into a density bucket."""
    for name, cfg in DENSITY_TARGETS.items():
        lo, hi = cfg["range"]
        if lo <= num_objects <= hi:
            return name
    if num_objects >= DENSITY_TARGETS["dense"]["range"][0]:
        return "dense"
    return "sparse"


# ============================================================
# Core Selection Algorithm
# ============================================================

def score_video(video, selected, slot_fill, slots,
                duration_counts, density_counts, source_counts):
    """
    Score a candidate video for selection.

    Returns a float: higher = more valuable to select next.
    Negative scores are possible (source over-represented, all slots full).
    """
    total_selected = len(selected)
    score = 0.0

    # ---- Priority 1: Phenomenon slot deficit ----
    profile = video.get("phenomenon_profile", {})
    for slot_name, slot_cfg in slots.items():
        target = slot_cfg["target_videos"]
        current = slot_fill[slot_name]
        if current >= target:
            continue
        if profile.get(slot_cfg["profile_key"], False):
            deficit_ratio = (target - current) / target
            score += deficit_ratio * slot_cfg["weight"] * PHENOMENON_SCORE_SCALE

    # ---- Priority 2: Duration diversity ----
    duration = video.get("stats", {}).get("duration", 0)
    dur_bucket = classify_duration_bucket(duration)
    if total_selected > 0:
        dur_ratio = duration_counts.get(dur_bucket, 0) / total_selected
        dur_target = DURATION_TARGETS[dur_bucket]["ratio"]
        score += (dur_target - dur_ratio) * DURATION_BONUS_SCALE

    # ---- Priority 3: Density diversity ----
    n_objects = video.get("stats", {}).get("num_objects", 0)
    den_bucket = classify_density_bucket(n_objects)
    if total_selected > 0:
        den_ratio = density_counts.get(den_bucket, 0) / total_selected
        den_target = DENSITY_TARGETS[den_bucket]["ratio"]
        score += (den_target - den_ratio) * DENSITY_BONUS_SCALE

    # ---- Priority 4: Source over-representation penalty ----
    source = video.get("source", "unknown")
    if total_selected > 0:
        src_ratio = source_counts.get(source, 0) / total_selected
        if src_ratio > MAX_SOURCE_RATIO:
            score += SOURCE_PENALTY

    return score


def select_videos(candidates, total_target=1000, seed=42):
    """
    Greedy weighted set cover selection.

    Phase 1: greedily pick the highest-scoring video each round
             until all phenomenon slots are filled or we hit total_target.
    Phase 2: if total_target not yet reached and slots are full,
             fill remaining with highest quality_score videos for diversity.
    """
    rng = random.Random(seed)
    # Shuffle to break ties randomly (greedy is deterministic on equal scores)
    candidates = list(candidates)
    rng.shuffle(candidates)

    selected = []
    selected_ids = set()

    # Tracking counters
    slot_fill = {name: 0 for name in PHENOMENON_SLOTS}
    duration_counts = Counter()
    density_counts = Counter()
    source_counts = Counter()

    remaining = list(candidates)

    # ---- Phase 1: Phenomenon-driven greedy selection ----
    while len(selected) < total_target and remaining:
        best_video = None
        best_score = -float("inf")
        best_idx = -1

        for idx, v in enumerate(remaining):
            s = score_video(
                v, selected, slot_fill, PHENOMENON_SLOTS,
                duration_counts, density_counts, source_counts,
            )
            if s > best_score:
                best_score = s
                best_video = v
                best_idx = idx

        if best_video is None:
            break

        # If all slots are satisfied and score is non-positive, switch to Phase 2
        all_filled = all(
            slot_fill[name] >= cfg["target_videos"]
            for name, cfg in PHENOMENON_SLOTS.items()
        )
        if all_filled and best_score <= 0:
            break

        # Select this video
        selected.append(best_video)
        selected_ids.add(best_video["video_id"])
        remaining.pop(best_idx)

        # Update counters
        profile = best_video.get("phenomenon_profile", {})
        for slot_name, slot_cfg in PHENOMENON_SLOTS.items():
            if profile.get(slot_cfg["profile_key"], False):
                slot_fill[slot_name] += 1

        dur = best_video.get("stats", {}).get("duration", 0)
        duration_counts[classify_duration_bucket(dur)] += 1

        n_obj = best_video.get("stats", {}).get("num_objects", 0)
        density_counts[classify_density_bucket(n_obj)] += 1

        source_counts[best_video.get("source", "unknown")] += 1

    phase1_count = len(selected)

    # ---- Phase 2: Fill remaining quota by quality_score ----
    if len(selected) < total_target and remaining:
        remaining.sort(key=lambda v: v.get("quality_score", 0), reverse=True)
        for v in remaining:
            if len(selected) >= total_target:
                break
            if v["video_id"] in selected_ids:
                continue
            selected.append(v)
            selected_ids.add(v["video_id"])

            # Update counters for reporting
            profile = v.get("phenomenon_profile", {})
            for slot_name, slot_cfg in PHENOMENON_SLOTS.items():
                if profile.get(slot_cfg["profile_key"], False):
                    slot_fill[slot_name] += 1
            dur = v.get("stats", {}).get("duration", 0)
            duration_counts[classify_duration_bucket(dur)] += 1
            n_obj = v.get("stats", {}).get("num_objects", 0)
            density_counts[classify_density_bucket(n_obj)] += 1
            source_counts[v.get("source", "unknown")] += 1

    return {
        "selected": selected,
        "slot_fill": slot_fill,
        "duration_counts": dict(duration_counts),
        "density_counts": dict(density_counts),
        "source_counts": dict(source_counts),
        "phase1_count": phase1_count,
    }


# ============================================================
# Analysis & Reporting
# ============================================================

def print_selection_report(result, total_pool):
    """Print a detailed report of the selection outcome."""
    selected = result["selected"]
    slot_fill = result["slot_fill"]
    n = len(selected)

    print(f"\n{'='*70}")
    print(f"  Selection Report")
    print(f"{'='*70}")
    print(f"  Pool size:        {total_pool}")
    print(f"  Selected:         {n}")
    print(f"  Phase 1 (greedy): {result['phase1_count']}")
    print(f"  Phase 2 (fill):   {n - result['phase1_count']}")

    # ---- Phenomenon slot coverage ----
    print(f"\n  Phenomenon Slot Coverage:")
    print(f"  {'Slot':<25s}  {'Selected':>8s}  {'Target':>6s}  {'Status':>8s}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*6}  {'-'*8}")
    all_met = True
    for slot_name, slot_cfg in PHENOMENON_SLOTS.items():
        current = slot_fill[slot_name]
        target = slot_cfg["target_videos"]
        status = "OK" if current >= target else "SHORT"
        if status == "SHORT":
            all_met = False
        print(f"  {slot_name:<25s}  {current:>8d}  {target:>6d}  {status:>8s}")
    if all_met:
        print(f"\n  All phenomenon slots satisfied.")
    else:
        print(f"\n  WARNING: Some slots under-filled (pool may be too small).")

    # ---- Duration distribution ----
    print(f"\n  Duration Distribution:")
    dur_counts = result["duration_counts"]
    for bucket, cfg in DURATION_TARGETS.items():
        count = dur_counts.get(bucket, 0)
        actual_pct = count / n * 100 if n else 0
        target_pct = cfg["ratio"] * 100
        lo, hi = cfg["range"]
        print(f"    {bucket:<10s} ({lo:>2d}-{hi:>2d}s):  {count:>5d}"
              f"  ({actual_pct:5.1f}%  target {target_pct:.0f}%)")

    # ---- Density distribution ----
    print(f"\n  Object Density Distribution:")
    den_counts = result["density_counts"]
    for bucket, cfg in DENSITY_TARGETS.items():
        count = den_counts.get(bucket, 0)
        actual_pct = count / n * 100 if n else 0
        target_pct = cfg["ratio"] * 100
        lo, hi = cfg["range"]
        hi_str = f"{hi}" if hi < 999 else "20+"
        print(f"    {bucket:<10s} ({lo:>2d}-{hi_str:>3s} obj):"
              f"  {count:>5d}"
              f"  ({actual_pct:5.1f}%  target {target_pct:.0f}%)")

    # ---- Source distribution ----
    print(f"\n  Source Distribution:")
    src_counts = result["source_counts"]
    for src, count in sorted(src_counts.items(),
                             key=lambda x: x[1], reverse=True):
        pct = count / n * 100 if n else 0
        over = " *** OVER CAP" if pct > MAX_SOURCE_RATIO * 100 else ""
        print(f"    {src:<20s}  {count:>5d}  ({pct:5.1f}%){over}")

    # ---- Event distribution ----
    print(f"\n  Event Distribution (selected videos):")
    total_ec = Counter()
    for v in selected:
        for etype, cnt in v.get("stats", {}).get("event_counts", {}).items():
            total_ec[etype] += cnt
    for etype, cnt in total_ec.most_common():
        avg = cnt / n if n else 0
        print(f"    {etype:<25s}  total={cnt:>6d}  avg/video={avg:.1f}")

    # ---- Quality score distribution ----
    scores = [v.get("quality_score", 0) for v in selected]
    if scores:
        print(f"\n  Quality Score Distribution:")
        print(f"    min={min(scores)}  median={sorted(scores)[len(scores)//2]}"
              f"  max={max(scores)}  mean={sum(scores)/len(scores):.1f}")

    # ---- Difficulty tiers (by quality score percentile) ----
    if scores:
        scores_sorted = sorted(scores)
        p30 = scores_sorted[int(len(scores_sorted) * 0.30)]
        p75 = scores_sorted[int(len(scores_sorted) * 0.75)]
        easy = sum(1 for s in scores if s <= p30)
        hard = sum(1 for s in scores if s >= p75)
        medium = n - easy - hard
        print(f"\n  Difficulty Tiers (by quality score percentile):")
        print(f"    easy   (score <= {p30}):  {easy:>5d}  ({easy/n*100:.0f}%)")
        print(f"    medium:               {medium:>5d}  ({medium/n*100:.0f}%)")
        print(f"    hard   (score >= {p75}):  {hard:>5d}  ({hard/n*100:.0f}%)")


def print_pool_analysis(candidates):
    """Analyze the input pool before selection."""
    n = len(candidates)
    print(f"\n{'='*70}")
    print(f"  Input Pool Analysis ({n} videos)")
    print(f"{'='*70}")

    # Phenomenon slot availability
    print(f"\n  Phenomenon slot availability in pool:")
    for slot_name, slot_cfg in PHENOMENON_SLOTS.items():
        key = slot_cfg["profile_key"]
        count = sum(
            1 for v in candidates
            if v.get("phenomenon_profile", {}).get(key, False)
        )
        target = slot_cfg["target_videos"]
        status = "OK" if count >= target else "TIGHT" if count >= target * 0.8 else "LOW"
        print(f"    {slot_name:<25s}  available={count:>6d}"
              f"  target={target:>4d}  {status}")

    # Source distribution in pool
    print(f"\n  Source distribution in pool:")
    src = Counter(v.get("source", "unknown") for v in candidates)
    for s, c in src.most_common():
        print(f"    {s:<20s}  {c:>6d}  ({c/n*100:.1f}%)")

    # Duration distribution in pool
    print(f"\n  Duration distribution in pool:")
    for bucket in DURATION_TARGETS:
        count = sum(
            1 for v in candidates
            if classify_duration_bucket(
                v.get("stats", {}).get("duration", 0)
            ) == bucket
        )
        print(f"    {bucket:<10s}  {count:>6d}  ({count/n*100:.1f}%)")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="TOC-Bench Step 2: Phenomenon-first video selection"
    )
    parser.add_argument(
        "--total", type=int, default=1000,
        help="Target number of videos to select (default: 1000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for tie-breaking (default: 42)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Analyze pool and show projected selection, but don't save"
    )
    parser.add_argument(
        "--analyze-only", action="store_true",
        help="Analyze an existing selected_videos.json"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to filtered_videos.json (default: auto)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save selected_videos.json (default: auto)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  TOC-Bench Step 2: Phenomenon-First Video Selection")
    print("=" * 70)

    output_path = Path(args.output) if args.output else FILTERED_DIR / "selected_videos.json"

    # ---- Analyze existing selection ----
    if args.analyze_only:
        if not output_path.exists():
            print(f"  [ERROR] {output_path} not found.")
            sys.exit(1)
        with open(output_path) as f:
            data = json.load(f)
        # Reconstruct result dict for reporting
        selected = data["videos"]
        # Recompute counters
        slot_fill = {name: 0 for name in PHENOMENON_SLOTS}
        duration_counts = Counter()
        density_counts = Counter()
        source_counts = Counter()
        for v in selected:
            profile = v.get("phenomenon_profile", {})
            for sn, sc in PHENOMENON_SLOTS.items():
                if profile.get(sc["profile_key"], False):
                    slot_fill[sn] += 1
            dur = v.get("stats", {}).get("duration", 0)
            duration_counts[classify_duration_bucket(dur)] += 1
            n_obj = v.get("stats", {}).get("num_objects", 0)
            density_counts[classify_density_bucket(n_obj)] += 1
            source_counts[v.get("source", "unknown")] += 1

        result = {
            "selected": selected,
            "slot_fill": slot_fill,
            "duration_counts": dict(duration_counts),
            "density_counts": dict(density_counts),
            "source_counts": dict(source_counts),
            "phase1_count": data.get("phase1_count", len(selected)),
        }
        print_selection_report(result, data.get("total_pool", "?"))
        return

    # ---- Load pool ----
    input_path = Path(args.input) if args.input else FILTERED_DIR / "filtered_videos.json"
    if not input_path.exists():
        print(f"  [ERROR] {input_path} not found. Run step1c first.")
        sys.exit(1)

    with open(input_path) as f:
        filtered = json.load(f)

    candidates = filtered.get("videos", [])
    # Only consider videos that passed filtering
    candidates = [v for v in candidates if v.get("passed", True)]

    print(f"  Loaded {len(candidates)} passed videos from {input_path}")

    if not candidates:
        print("  [ERROR] No passed videos found.")
        sys.exit(1)

    # ---- Check for phenomenon_profile ----
    has_profile = sum(
        1 for v in candidates if "phenomenon_profile" in v
    )
    if has_profile == 0:
        print("  [ERROR] No phenomenon_profile found in filtered_videos.json.")
        print("  Please re-run step1c with the updated version to generate")
        print("  phenomenon profiles.")
        sys.exit(1)
    if has_profile < len(candidates):
        print(f"  [WARN] {len(candidates) - has_profile} videos missing"
              f" phenomenon_profile (will be treated as all-False)")

    # ---- Pool analysis ----
    print_pool_analysis(candidates)

    # ---- Run selection ----
    print(f"\n  Running selection (target={args.total}, seed={args.seed})...")
    result = select_videos(candidates, total_target=args.total, seed=args.seed)

    # ---- Report ----
    print_selection_report(result, len(candidates))

    # ---- Save ----
    if args.dry_run:
        print(f"\n  [DRY-RUN] Not saving. Use without --dry-run to save.")
        return

    selected = result["selected"]
    # Sort final output by video_id for reproducibility
    selected.sort(key=lambda v: v.get("video_id", ""))

    output_data = {
        "total_pool": len(candidates),
        "total_selected": len(selected),
        "phase1_count": result["phase1_count"],
        "phase2_count": len(selected) - result["phase1_count"],
        "selection_config": {
            "total_target": args.total,
            "seed": args.seed,
            "max_source_ratio": MAX_SOURCE_RATIO,
            "phenomenon_slots": {
                name: {"target": cfg["target_videos"], "filled": result["slot_fill"][name]}
                for name, cfg in PHENOMENON_SLOTS.items()
            },
        },
        "distribution": {
            "phenomenon_slot_fill": result["slot_fill"],
            "duration_counts": result["duration_counts"],
            "density_counts": result["density_counts"],
            "source_counts": result["source_counts"],
        },
        "videos": selected,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n  Saved {len(selected)} selected videos to {output_path}")

    # Also save a compact video_id list for quick reference
    id_list_path = output_path.parent / "selected_video_ids.txt"
    with open(id_list_path, "w") as f:
        for v in selected:
            f.write(v["video_id"] + "\n")
    print(f"  Saved video ID list to {id_list_path}")


if __name__ == "__main__":
    main()