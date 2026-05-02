#!/usr/bin/env python3
"""
TOC-Bench Step 1b Post-processing: SAM3 Track Repair + Confidence
=================================================================
Implements 方案 C (user decision): SAM3 text-prompt tracking has known
instance-stability issues — ID splits (same object assigned 2 IDs across
time), ID swaps (different objects sharing an ID), and spurious fragments.
This module runs AFTER step1b and AUGMENTS tracks.json with:

  1. Fragment cleanup      — drop obj_ids visible for too few frames
  2. ID-split repair       — merge same-label fragments that are clearly
                             the same physical object (non-overlapping in
                             time, spatially continuous)
  3. ID-swap detection     — flag obj_ids with sudden position jumps that
                             aren't explained by occlusion (DOES NOT split
                             the track; just lowers confidence)
  4. Instance confidence   — 0-1 score per final obj_id that downstream
                             modules (step1c, step3a) use to gate C-dim QA

Input:  TRACKS_DIR/<video_id>.json  (from step1b)
Output: TRACKS_DIR/<video_id>.json  (overwritten with `objects` augmented
                                     with `instance_confidence` and merge
                                     history; `repair_log/<video_id>.json`
                                     for audit)

Usage:
    python step1b_postprocess.py                  # process all tracks
    python step1b_postprocess.py --video-id X     # single video
    python step1b_postprocess.py --dry-run        # report without writing
    python step1b_postprocess.py --stats          # aggregate summary
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *a, **k): return x

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import TRACKS_DIR, SAM3_POSTPROC_CONFIG


# ============================================================================
# Geometry helpers
# ============================================================================

def bbox_center(box):
    """bbox = [x1, y1, x2, y2] → (cx, cy)."""
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def bbox_is_valid(box):
    """Check bbox isn't the [0,0,0,0] sentinel for invisible frames."""
    return box and (box[2] - box[0] > 0) and (box[3] - box[1] > 0)


def euclid(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def image_diagonal(width, height):
    return math.hypot(width, height)


# ============================================================================
# Per-track analysis helpers
# ============================================================================

def compute_visible_frame_range(track):
    """Returns (first_visible_frame_idx, last_visible_frame_idx, visible_count)
    or (None, None, 0) if no visible frames."""
    vis_frames = [i for i, v in enumerate(track["visible"]) if v]
    if not vis_frames:
        return None, None, 0
    return vis_frames[0], vis_frames[-1], len(vis_frames)


def first_last_visible_bbox(track):
    """Returns (first_bbox_center, last_bbox_center) among visible frames."""
    first_idx, last_idx, _ = compute_visible_frame_range(track)
    if first_idx is None:
        return None, None
    first_box = track["bboxes"][first_idx]
    last_box = track["bboxes"][last_idx]
    if not (bbox_is_valid(first_box) and bbox_is_valid(last_box)):
        return None, None
    return bbox_center(first_box), bbox_center(last_box)


def mean_confidence(track):
    """Mean SAM3 confidence across visible frames."""
    scores = [c for c, v in zip(track["confidence"], track["visible"]) if v]
    return sum(scores) / len(scores) if scores else 0.0


def visible_ratio(track, n_frames):
    _, _, cnt = compute_visible_frame_range(track)
    return cnt / max(1, n_frames)


# ============================================================================
# Step 1: Fragment cleanup
# ============================================================================

def filter_short_fragments(tracks, n_frames, cfg):
    """Drop obj_ids with too few visible frames."""
    min_abs = cfg["min_fragment_frames"]
    min_ratio = cfg["min_fragment_duration_ratio"]
    min_frames_required = max(min_abs, int(min_ratio * n_frames))

    kept, dropped = [], []
    for t in tracks:
        _, _, vis_count = compute_visible_frame_range(t)
        if vis_count >= min_frames_required:
            kept.append(t)
        else:
            dropped.append({
                "obj_id": t["obj_id"],
                "label": t["label"],
                "visible_frames": vis_count,
                "threshold": min_frames_required,
            })
    return kept, dropped


# ============================================================================
# Step 2: ID-split repair (merge same-label fragments)
# ============================================================================

def find_mergeable_pairs(tracks, fps, img_diag, cfg):
    """Find pairs of obj_ids that look like the SAME physical object split
    into two tracker IDs. Criteria:
        - Same raw label
        - Their visible time windows don't overlap
        - Gap between end of one and start of next ≤ max_time_gap_sec
        - Spatial jump (last center of earlier → first center of later)
          ≤ max_spatial_jump_ratio × img_diag
    Returns list of (earlier_track, later_track) pairs.
    """
    max_gap_frames = cfg["split_repair_max_time_gap_sec"] * fps
    max_jump = cfg["split_repair_max_spatial_jump_ratio"] * img_diag

    # Group tracks by label (only merge within same label)
    by_label = defaultdict(list)
    for t in tracks:
        by_label[t["label"]].append(t)

    pairs = []
    for label, group in by_label.items():
        if len(group) < 2:
            continue
        # Precompute visible range per track
        ranges = []
        for t in group:
            first, last, cnt = compute_visible_frame_range(t)
            if first is not None:
                ranges.append((t, first, last))
        # Sort by first_visible_frame
        ranges.sort(key=lambda r: r[1])

        for i in range(len(ranges)):
            for j in range(i + 1, len(ranges)):
                t_a, first_a, last_a = ranges[i]
                t_b, first_b, last_b = ranges[j]

                # Must not overlap in time
                if last_a >= first_b:
                    continue
                # Gap must be within threshold
                if first_b - last_a > max_gap_frames:
                    continue
                # Spatial continuity
                a_last_box = t_a["bboxes"][last_a]
                b_first_box = t_b["bboxes"][first_b]
                if not (bbox_is_valid(a_last_box) and bbox_is_valid(b_first_box)):
                    continue
                jump = euclid(bbox_center(a_last_box), bbox_center(b_first_box))
                if jump > max_jump:
                    continue
                pairs.append((t_a, t_b, jump))

    return pairs


def greedy_merge_fragments(tracks, pairs):
    """Given candidate merge pairs, greedily apply them.
    Returns (merged_tracks, merge_log).

    Each track can be merged into at most one other (first-fit).
    """
    # id → current track (mutable). When we merge B into A, we rewrite
    # id_to_track[B.obj_id] = A, so subsequent references follow.
    id_to_track = {t["obj_id"]: t for t in tracks}
    # Track which obj_ids have been absorbed (the "losers" of merges)
    absorbed = set()
    merge_log = []

    # Sort pairs by spatial jump (smaller jump = more confident merge)
    pairs_sorted = sorted(pairs, key=lambda p: p[2])

    for t_a, t_b, jump in pairs_sorted:
        if t_a["obj_id"] in absorbed or t_b["obj_id"] in absorbed:
            continue  # one side already absorbed in earlier merge

        # Merge t_b into t_a: combine per-frame arrays
        n = len(t_a["frames"])
        for k in range(n):
            if not t_a["visible"][k] and t_b["visible"][k]:
                # take from t_b where t_a is invisible
                t_a["bboxes"][k] = t_b["bboxes"][k]
                t_a["visible"][k] = True
                t_a["confidence"][k] = t_b["confidence"][k]
                t_a["mask_areas"][k] = t_b["mask_areas"][k]
            # if both visible at same frame: keep t_a (should be rare; pairs
            # are filtered to non-overlap)

        # Record merge
        t_a.setdefault("merged_from", []).append(t_b["obj_id"])
        absorbed.add(t_b["obj_id"])
        merge_log.append({
            "into_obj_id": t_a["obj_id"],
            "absorbed_obj_id": t_b["obj_id"],
            "label": t_a["label"],
            "spatial_jump_px": round(jump, 2),
        })

    # Keep only unabsorbed tracks
    merged = [t for t in tracks if t["obj_id"] not in absorbed]
    return merged, merge_log


# ============================================================================
# Step 3: ID-swap detection (flag suspicious jumps within one obj_id)
# ============================================================================

def detect_swaps_in_track(track, img_diag, cfg):
    """Detect sudden position jumps within a single obj_id that look like
    the tracker reassigned the ID to a different object.

    A jump is flagged only if:
        - Both the before-jump and after-jump frames are visible
        - The two visible frames are within `min_visible_gap` of each other
          (so we're not just seeing movement across a long occlusion)
        - The bbox center distance exceeds threshold
    """
    threshold = cfg["swap_detect_position_jump_ratio"] * img_diag
    max_gap = cfg["swap_detect_min_visible_gap"]

    swap_events = []
    last_visible_idx = None
    last_visible_center = None

    for i, (box, vis) in enumerate(zip(track["bboxes"], track["visible"])):
        if not vis or not bbox_is_valid(box):
            continue
        cur_center = bbox_center(box)
        if last_visible_idx is not None and i - last_visible_idx <= max_gap:
            d = euclid(last_visible_center, cur_center)
            if d > threshold:
                swap_events.append({
                    "frame_before": last_visible_idx,
                    "frame_after": i,
                    "jump_px": round(d, 2),
                })
        last_visible_idx = i
        last_visible_center = cur_center

    return swap_events


# ============================================================================
# Step 4: Instance confidence scoring
# ============================================================================

def compute_instance_confidence(track, n_frames, num_merges, num_swaps):
    """Combine signals into a single [0, 1] score per obj_id.

    Formula (from config):
        base         = mean SAM3 confidence across visible frames
        split_penalty = 1.0 - 0.15 * num_merges                (clipped ≥ 0)
        swap_penalty  = 1.0 - 0.25 * num_swaps                 (clipped ≥ 0)
        coverage      = 1.0 if visible_ratio >= 0.3 else 0.8
        confidence    = base * split_penalty * swap_penalty * coverage
    """
    base = mean_confidence(track)
    split_penalty = max(0.0, 1.0 - 0.15 * num_merges)
    swap_penalty = max(0.0, 1.0 - 0.25 * num_swaps)
    coverage = 1.0 if visible_ratio(track, n_frames) >= 0.3 else 0.8
    return round(base * split_penalty * swap_penalty * coverage, 4)


# ============================================================================
# Main per-video pipeline
# ============================================================================

def postprocess_video(tracks_data, cfg):
    """Run the full post-processing pipeline on one video's tracks.

    Args:
        tracks_data: dict loaded from tracks.json
        cfg: SAM3_POSTPROC_CONFIG dict

    Returns:
        (new_tracks_data, repair_log)
    """
    metadata = tracks_data.get("metadata", {})
    width = metadata.get("width", 1280)
    height = metadata.get("height", 720)
    fps = metadata.get("fps", 3)
    n_frames = len(tracks_data.get("timestamps", []))
    if n_frames == 0:
        n_frames = metadata.get("n_frames", 1)
    img_diag = image_diagonal(width, height)

    raw_tracks = list(tracks_data["objects"])
    n_raw = len(raw_tracks)

    # --- Step 1: drop short fragments ---
    kept, dropped = filter_short_fragments(raw_tracks, n_frames, cfg)

    # --- Step 2: find and apply ID-split merges ---
    pairs = find_mergeable_pairs(kept, fps, img_diag, cfg)
    merged, merge_log = greedy_merge_fragments(kept, pairs)

    # --- Step 3: detect within-track swaps and compute confidence ---
    per_track_swaps = {}
    for t in merged:
        swaps = detect_swaps_in_track(t, img_diag, cfg)
        per_track_swaps[t["obj_id"]] = swaps

    for t in merged:
        num_merges = len(t.get("merged_from", []))
        num_swaps = len(per_track_swaps.get(t["obj_id"], []))
        t["instance_confidence"] = compute_instance_confidence(
            t, n_frames, num_merges, num_swaps
        )
        if num_swaps > 0:
            t["suspected_swaps"] = per_track_swaps[t["obj_id"]]

    # --- Build output ---
    new_tracks_data = dict(tracks_data)
    new_tracks_data["objects"] = merged
    new_tracks_data["num_objects"] = len(merged)
    new_tracks_data["postprocessed"] = True

    repair_log = {
        "video_id": tracks_data.get("video_id"),
        "raw_object_count": n_raw,
        "after_fragment_filter": len(kept),
        "after_merge": len(merged),
        "dropped_fragments": dropped,
        "merges_applied": merge_log,
        "total_swaps_flagged": sum(len(s) for s in per_track_swaps.values()),
        "obj_confidences": [
            {"obj_id": t["obj_id"], "label": t["label"],
             "confidence": t["instance_confidence"]}
            for t in merged
        ],
    }

    return new_tracks_data, repair_log


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print results without writing files")
    ap.add_argument("--stats", action="store_true",
                    help="Print aggregate statistics over all postprocessed videos")
    args = ap.parse_args()

    cfg = SAM3_POSTPROC_CONFIG
    repair_dir = TRACKS_DIR / "repair_log"
    if not args.dry_run and cfg.get("save_repair_log", True):
        repair_dir.mkdir(parents=True, exist_ok=True)

    # Collect targets
    if args.video_id:
        targets = [TRACKS_DIR / f"{args.video_id}.json"]
    else:
        targets = sorted([
            f for f in TRACKS_DIR.glob("*.json")
            if not f.name.startswith("_")
        ])

    if args.stats:
        _print_stats(targets)
        return

    print(f"Post-processing {len(targets)} track files...")
    agg = {"total_videos": 0, "total_raw_obj": 0, "total_dropped": 0,
           "total_merged": 0, "total_swaps": 0,
           "confidence_above_0.7": 0, "total_obj_after": 0}

    for tf in tqdm(targets):
        try:
            with open(tf) as f:
                tracks_data = json.load(f)
        except Exception as e:
            print(f"  [SKIP] {tf.name}: {e}")
            continue

        if tracks_data.get("postprocessed"):
            continue  # already done

        new_data, log = postprocess_video(tracks_data, cfg)

        agg["total_videos"] += 1
        agg["total_raw_obj"] += log["raw_object_count"]
        agg["total_dropped"] += len(log["dropped_fragments"])
        agg["total_merged"] += len(log["merges_applied"])
        agg["total_swaps"] += log["total_swaps_flagged"]
        agg["total_obj_after"] += new_data["num_objects"]
        agg["confidence_above_0.7"] += sum(
            1 for c in log["obj_confidences"] if c["confidence"] >= 0.7
        )

        if args.dry_run:
            continue

        # Write back tracks.json (overwrite) and save repair log
        with open(tf, "w") as f:
            json.dump(new_data, f, indent=2)
        if cfg.get("save_repair_log", True):
            with open(repair_dir / tf.name, "w") as f:
                json.dump(log, f, indent=2)

    print("\n" + "=" * 60)
    print("  Post-processing summary")
    print("=" * 60)
    print(f"  Videos processed:           {agg['total_videos']}")
    print(f"  Raw obj_ids (input):        {agg['total_raw_obj']}")
    print(f"  Dropped (short fragments):  {agg['total_dropped']}")
    print(f"  Merged (ID-split repaired): {agg['total_merged']}")
    print(f"  Swap events flagged:        {agg['total_swaps']}")
    print(f"  Final obj_ids:              {agg['total_obj_after']}")
    print(f"  High-confidence (≥0.7):     {agg['confidence_above_0.7']} "
          f"({agg['confidence_above_0.7'] / max(1, agg['total_obj_after']) * 100:.1f}%)")


def _print_stats(targets):
    """Read repair logs and summarize."""
    log_dir = TRACKS_DIR / "repair_log"
    if not log_dir.exists():
        print("No repair logs found. Run post-processing first.")
        return

    total = {"videos": 0, "raw": 0, "dropped": 0, "merged": 0, "swaps": 0,
             "obj_after": 0, "hi_conf": 0}
    conf_dist = []

    for log_file in log_dir.glob("*.json"):
        try:
            with open(log_file) as f:
                log = json.load(f)
        except Exception:
            continue
        total["videos"] += 1
        total["raw"] += log.get("raw_object_count", 0)
        total["dropped"] += len(log.get("dropped_fragments", []))
        total["merged"] += len(log.get("merges_applied", []))
        total["swaps"] += log.get("total_swaps_flagged", 0)
        confs = [c["confidence"] for c in log.get("obj_confidences", [])]
        total["obj_after"] += len(confs)
        total["hi_conf"] += sum(1 for c in confs if c >= 0.7)
        conf_dist.extend(confs)

    print("=" * 60)
    print("  Aggregate post-processing stats")
    print("=" * 60)
    for k, v in total.items():
        print(f"  {k:<20s} {v}")
    if conf_dist:
        conf_dist.sort()
        n = len(conf_dist)
        print(f"\n  Confidence distribution (n={n}):")
        print(f"    min    = {conf_dist[0]:.3f}")
        print(f"    median = {conf_dist[n // 2]:.3f}")
        print(f"    mean   = {sum(conf_dist) / n:.3f}")
        print(f"    max    = {conf_dist[-1]:.3f}")
        # Buckets
        buckets = [0.0, 0.3, 0.5, 0.7, 0.85, 1.0]
        print(f"  Confidence buckets:")
        for i in range(len(buckets) - 1):
            lo, hi = buckets[i], buckets[i + 1]
            cnt = sum(1 for c in conf_dist if lo <= c < hi)
            print(f"    [{lo:.2f}, {hi:.2f})  {cnt:>6d}  ({cnt / n * 100:.1f}%)")


if __name__ == "__main__":
    main()