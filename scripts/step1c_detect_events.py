#!/usr/bin/env python3
"""
TOC-Bench Step 1c: Event Detection & Video Filtering
=====================================================
9 event types detected from SAM 3.1 object tracks:

Per-object (8):
  appear            - object first detected after frame 0, not near border
  enter_frame       - object enters from frame border
  exit_frame        - object leaves via frame border
  partial_occlusion - visible area drops to 10-50% of baseline (partially blocked)
  full_occlusion    - visible area drops below 10% of baseline, NOT near border
  reappear          - recovers from full_occlusion or exit_frame
  disappear         - full_occlusion or exit_frame with no recovery until video end
  state_change      - sudden area or position jump (flipped, opened, fast motion)

Pairwise (1):
  interaction       - two objects' bboxes overlap for sustained period

Changes from original:
  - Every event carries skeleton-ready metadata:
    temporal_position, spatial_position, timestamp, event_id
  - Interaction events carry mean_iou and per-object spatial positions
  - detect_all_events builds per-object timeline summaries for cross-event QA
  - filter_video outputs phenomenon_profile for Step 2 video selection
  - find_occluder_candidates links full_occlusion to likely occluding objects

Usage:
    python step1c_detect_events.py [--source perception_test] [--stats]
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
try:
    from tqdm import tqdm
except ImportError:  # graceful fallback if tqdm isn't installed
    def tqdm(iterable, *args, **kwargs):
        return iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    ROOT, TRACKS_DIR, EVENTS_DIR, FILTERED_DIR,
    EVENT_CONFIG, FILTER_CONFIG,
    # v2 additions: bucket definitions, stats config, ref-expr modifiers
    TIME_BUCKETS, DURATION_BUCKETS,
    EVENT_STATS_CONFIG, REF_EXPR_MODIFIERS,
)

import re


# ============================================================
# Utilities
# ============================================================

def compute_bbox_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    a2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def bbox_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def bbox_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def is_near_border(box, width, height, margin=0.05):
    mx, my = width * margin, height * margin
    return box[0] < mx or box[1] < my or box[2] > width - mx or box[3] > height - my


def compute_baseline_area(mask_areas, visible, idx, window=10):
    """Rolling median of last `window` visible frames' areas as baseline."""
    past_areas = []
    for j in range(max(0, idx - window), idx):
        if visible[j] and mask_areas[j] > 0:
            past_areas.append(mask_areas[j])
    if not past_areas:
        # No history — use first visible area as baseline
        for j in range(len(visible)):
            if visible[j] and mask_areas[j] > 0:
                return mask_areas[j]
        return 0
    return float(np.median(past_areas))


# ============================================================
# Skeleton-Ready Classification Helpers
# ============================================================

def classify_temporal_position(frame, total_frames):
    """Classify a frame index into a temporal bucket from config.TIME_BUCKETS.

    v2: 4 buckets均分 (user decision Q1):
        beginning (0-25%) / early (25-50%) / late (50-75%) / end (75-100%).

    The function reads TIME_BUCKETS from config, so changing the config
    automatically changes behavior here.
    """
    if total_frames <= 1:
        # Edge: single-frame video — use the first bucket
        return next(iter(TIME_BUCKETS.keys()))
    ratio = frame / (total_frames - 1)
    # Clamp to [0, 1) to avoid right-edge misclassification
    ratio = max(0.0, min(0.9999, ratio))
    for bucket_name, (lo, hi) in TIME_BUCKETS.items():
        if lo <= ratio < hi:
            return bucket_name
    # Fallback: last bucket
    return list(TIME_BUCKETS.keys())[-1]


def classify_spatial_position(bbox, img_width, img_height):
    """Classify a bbox into a coarse spatial position label.

    NOTE v2: this function is retained for EVENT metadata (useful for
    debugging / analytics), but it is NO LONGER used by the QA generation
    pipeline. Spatial_location dim has been replaced by relative_spatial_change.
    Kept for backward compatibility with existing analysis scripts.
    """
    if img_width <= 0 or img_height <= 0:
        return "center"
    cx, cy = bbox_center(bbox)
    # Horizontal
    if cx < img_width * 0.33:
        x_pos = "left"
    elif cx > img_width * 0.67:
        x_pos = "right"
    else:
        x_pos = "center"
    # Vertical
    if cy < img_height * 0.33:
        y_pos = "upper"
    elif cy > img_height * 0.67:
        y_pos = "lower"
    else:
        y_pos = "middle"
    if y_pos == "middle":
        return x_pos
    if x_pos == "center":
        return y_pos
    return f"{y_pos}-{x_pos}"


def classify_duration_category(duration_frames, fps):
    """Classify a duration in frames into a human-readable category.

    v2: 4 buckets from config.DURATION_BUCKETS (user decision Q2: v1 style).
    Reads thresholds from config so changes propagate automatically.
    """
    if fps <= 0:
        fps = 6.0
    seconds = duration_frames / fps
    for bucket_name, (lo, hi) in DURATION_BUCKETS.items():
        if lo <= seconds < hi:
            return bucket_name
    # Fallback: last bucket (for values at exactly the upper boundary of the last range)
    return list(DURATION_BUCKETS.keys())[-1]


def frame_to_timestamp(frame_idx, timestamps):
    """Convert frame index to timestamp in seconds.  Falls back to None."""
    if timestamps and 0 <= frame_idx < len(timestamps):
        return round(timestamps[frame_idx], 3)
    return None


# ============================================================
# Occluder Candidate Detection
# ============================================================

def find_occluder_candidates(obj_id, invisible_start, tracks,
                             iou_threshold=0.10):
    """
    For a full_occlusion event, find other tracked objects whose bboxes
    overlap with the subject's last-visible bbox around the moment it
    goes invisible.  Returns list of (obj_id, label, iou) sorted by iou.
    """
    subject = None
    for t in tracks:
        if t["obj_id"] == obj_id:
            subject = t
            break
    if subject is None:
        return []

    # Last visible bbox before going invisible
    last_vis = invisible_start - 1
    while last_vis >= 0 and not subject["visible"][last_vis]:
        last_vis -= 1
    if last_vis < 0:
        return []
    ref_bbox = subject["bboxes"][last_vis]

    candidates = []
    for t in tracks:
        if t["obj_id"] == obj_id:
            continue
        # Check a small window around the occlusion onset
        for fi in range(max(0, invisible_start - 2),
                        min(len(t["visible"]), invisible_start + 3)):
            if fi < len(t["visible"]) and t["visible"][fi]:
                iou = compute_bbox_iou(ref_bbox, t["bboxes"][fi])
                if iou > iou_threshold:
                    candidates.append(
                        (t["obj_id"], t["label"], round(iou, 3))
                    )
                    break  # one hit per object is enough

    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


# ============================================================
# Per-Object Event Detection
# ============================================================

def detect_object_events(track, config, img_width, img_height,
                         timestamps=None, tracks_all=None):
    """
    Detect all per-object events for a single track.

    Every event dict now includes skeleton-ready metadata:
      - event_id:           assigned later in detect_all_events
      - temporal_position:  "beginning"/"early"/"middle"/"late"/"end"
      - spatial_position:   "left"/"center"/"right"/"upper-left"/...
      - timestamp / timestamp_start / timestamp_end:  seconds (or None)
      - duration_category:  "briefly"/"for several seconds"/... (ranged)
    """
    events = []
    obj_id = track["obj_id"]
    label = track["label"]
    visible = track["visible"]
    areas = track["mask_areas"]
    bboxes = track["bboxes"]
    n = len(visible)

    if n == 0:
        return events

    window = config.get("baseline_window", 10)
    partial_ratio = config.get("partial_occlusion_ratio", 0.50)
    full_ratio = config.get("full_occlusion_ratio", 0.10)
    min_partial = config.get("min_partial_frames", 2)
    min_full = config.get("min_full_occlusion_frames", 2)
    min_reappear = config.get("min_reappear_frames", 2)
    border_margin = config.get("border_margin", 0.05)
    state_area_ratio = config.get("state_area_change_ratio", 0.5)
    state_pos_ratio = config.get("state_position_change_ratio", 0.25)
    img_diag = (np.sqrt(img_width**2 + img_height**2)
                if img_width > 0 else 1.0)

    # Estimate fps from timestamps for duration classification
    fps_est = 6.0
    if timestamps and len(timestamps) >= 2:
        total_dur = timestamps[-1] - timestamps[0]
        if total_dur > 0:
            fps_est = (len(timestamps) - 1) / total_dur

    # ---- Classify each frame's visibility state ----
    frame_states = []
    for i in range(n):
        if not visible[i] or areas[i] == 0:
            frame_states.append("invisible")
        else:
            baseline = compute_baseline_area(areas, visible, i, window)
            if baseline > 0:
                ratio = areas[i] / baseline
                if ratio < full_ratio:
                    frame_states.append("invisible")
                elif ratio < partial_ratio:
                    frame_states.append("partial")
                else:
                    frame_states.append("normal")
            else:
                frame_states.append("normal")

    # --------------------------------------------------------
    # Helper: build a point-event dict with skeleton metadata
    # --------------------------------------------------------
    def _point_event(event_type, frame, **extra):
        bbox = bboxes[frame] if frame < len(bboxes) else [0, 0, 0, 0]
        ev = {
            "type": event_type,
            "obj_id": obj_id,
            "label": label,
            "frame": frame,
            "temporal_position": classify_temporal_position(frame, n),
            "spatial_position": classify_spatial_position(
                bbox, img_width, img_height
            ),
            "timestamp": frame_to_timestamp(frame, timestamps),
        }
        ev.update(extra)
        return ev

    # --------------------------------------------------------
    # Helper: build a ranged-event dict with skeleton metadata
    # --------------------------------------------------------
    def _range_event(event_type, start_frame, end_frame, **extra):
        bbox_s = (bboxes[start_frame]
                  if start_frame < len(bboxes) else [0, 0, 0, 0])
        dur = end_frame - start_frame + 1
        mid = (start_frame + end_frame) // 2
        ev = {
            "type": event_type,
            "obj_id": obj_id,
            "label": label,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "duration_frames": dur,
            "duration_category": classify_duration_category(dur, fps_est),
            "temporal_position": classify_temporal_position(mid, n),
            "spatial_position": classify_spatial_position(
                bbox_s, img_width, img_height
            ),
            "timestamp_start": frame_to_timestamp(start_frame, timestamps),
            "timestamp_end": frame_to_timestamp(end_frame, timestamps),
        }
        ev.update(extra)
        return ev

    # ---- 1. Appear / enter_frame: first visible frame > 0 ----
    first_visible = next(
        (i for i in range(n) if frame_states[i] != "invisible"), None
    )
    if first_visible is not None and first_visible > 2:
        if is_near_border(bboxes[first_visible],
                          img_width, img_height, border_margin):
            events.append(_point_event("enter_frame", first_visible))
        else:
            events.append(_point_event("appear", first_visible))

    # ---- 2. Scan for partial_occlusion, full_occlusion, exit_frame,
    #         reappear, disappear ----
    i = 0
    while i < n:
        state = frame_states[i]

        # --- Partial occlusion segment ---
        if state == "partial":
            seg_start = i
            while i < n and frame_states[i] == "partial":
                i += 1
            seg_len = i - seg_start
            if seg_len >= min_partial:
                events.append(
                    _range_event("partial_occlusion", seg_start, i - 1)
                )
            continue

        # --- Invisible segment (full_occlusion or exit_frame) ---
        if state == "invisible":
            seg_start = i
            while i < n and frame_states[i] == "invisible":
                i += 1
            seg_end = i   # first non-invisible frame (or n)
            seg_len = seg_end - seg_start

            if seg_len < min_full:
                continue

            # Was the object visible before this segment?
            was_visible_before = seg_start > 0 and any(
                s != "invisible" for s in frame_states[:seg_start]
            )
            if not was_visible_before:
                continue

            # Last known bbox before going invisible
            last_visible_idx = seg_start - 1
            while (last_visible_idx >= 0
                   and frame_states[last_visible_idx] == "invisible"):
                last_visible_idx -= 1

            if last_visible_idx >= 0:
                near_border = is_near_border(
                    bboxes[last_visible_idx],
                    img_width, img_height, border_margin
                )
            else:
                near_border = False

            event_type = "exit_frame" if near_border else "full_occlusion"

            extra = {}
            # For full_occlusion: find likely occluding objects
            if event_type == "full_occlusion" and tracks_all is not None:
                occluders = find_occluder_candidates(
                    obj_id, seg_start, tracks_all
                )
                if occluders:
                    extra["occluder_candidates"] = [
                        {"obj_id": oid, "label": lbl, "iou": iou_val}
                        for oid, lbl, iou_val in occluders
                    ]

            events.append(
                _range_event(event_type, seg_start, seg_end - 1, **extra)
            )

            # Check reappear vs disappear
            if seg_end < n:
                vis_after = sum(
                    1 for s in frame_states[
                        seg_end:min(seg_end + min_reappear + 2, n)
                    ]
                    if s != "invisible"
                )
                if vis_after >= min_reappear:
                    # Reappear — also check if re-enters from border
                    if is_near_border(bboxes[seg_end],
                                      img_width, img_height, border_margin):
                        events.append(_point_event(
                            "enter_frame", seg_end,
                            after_event=event_type,
                            after_frame=seg_start,
                        ))
                    events.append(_point_event(
                        "reappear", seg_end,
                        after_event=event_type,
                        invisible_duration=seg_len,
                        invisible_duration_category=(
                            classify_duration_category(seg_len, fps_est)
                        ),
                    ))
            else:
                # Never came back → disappear
                events.append(_point_event(
                    "disappear", seg_start,
                    cause=event_type,
                ))
            continue

        # --- Normal frame, just advance ---
        i += 1

    # ---- 3. State change ----
    min_sc_gap = config.get("min_state_change_gap", 5)
    last_sc_frame = -999

    prev_vis_idx = None
    for i in range(n):
        if frame_states[i] == "invisible":
            prev_vis_idx = None
            continue
        if prev_vis_idx is not None and (i - last_sc_frame) >= min_sc_gap:
            # Area jump
            a_prev = areas[prev_vis_idx]
            a_curr = areas[i]
            if a_prev > 0:
                area_change = abs(a_curr - a_prev) / a_prev
                if area_change > state_area_ratio:
                    events.append(_point_event(
                        "state_change", i,
                        subtype="area_jump",
                        change_ratio=round(area_change, 3),
                    ))
                    last_sc_frame = i

            # Position jump (only if no area_jump already fired)
            if last_sc_frame != i:
                cx1, cy1 = bbox_center(bboxes[prev_vis_idx])
                cx2, cy2 = bbox_center(bboxes[i])
                pos_shift = np.sqrt(
                    (cx2 - cx1)**2 + (cy2 - cy1)**2
                )
                if img_diag > 0 and pos_shift / img_diag > state_pos_ratio:
                    events.append(_point_event(
                        "state_change", i,
                        subtype="position_jump",
                        shift_ratio=round(pos_shift / img_diag, 3),
                    ))
                    last_sc_frame = i
        prev_vis_idx = i

    return events


# ============================================================
# Pairwise Event Detection
# ============================================================

def detect_interactions(tracks, config, img_width, img_height,
                        timestamps=None):
    """
    Detect pairwise object interactions (sustained bbox overlap).
    Filters out interactions spanning >max_duration_ratio of the video.

    Each interaction event now includes:
      - temporal_position, spatial_position, timestamps
      - mean_iou across the overlap segment
    """
    events = []
    iou_thresh = config.get("interaction_iou_threshold", 0.05)
    min_frames = config.get("min_interaction_frames", 3)
    max_dur_ratio = config.get("interaction_max_duration_ratio", 0.8)

    n_frames_total = max(
        (len(t["bboxes"]) for t in tracks), default=0
    )

    fps_est = 6.0
    if timestamps and len(timestamps) >= 2:
        total_dur = timestamps[-1] - timestamps[0]
        if total_dur > 0:
            fps_est = (len(timestamps) - 1) / total_dur

    def _emit_interaction(t1, t2, ov_start, ov_end, iou_sum, total_n):
        duration = ov_end - ov_start + 1
        if duration < min_frames or duration / total_n > max_dur_ratio:
            return None
        mid = (ov_start + ov_end) // 2
        # Spatial: midpoint between the two objects' centers at mid-frame
        c1 = (bbox_center(t1["bboxes"][mid])
              if mid < len(t1["bboxes"]) else (0, 0))
        c2 = (bbox_center(t2["bboxes"][mid])
              if mid < len(t2["bboxes"]) else (0, 0))
        mid_bbox = [
            min(c1[0], c2[0]), min(c1[1], c2[1]),
            max(c1[0], c2[0]), max(c1[1], c2[1]),
        ]
        return {
            "type": "interaction",
            "obj_ids": [t1["obj_id"], t2["obj_id"]],
            "labels": [t1["label"], t2["label"]],
            "start_frame": ov_start,
            "end_frame": ov_end,
            "duration_frames": duration,
            "duration_category": classify_duration_category(
                duration, fps_est
            ),
            "temporal_position": classify_temporal_position(
                mid, n_frames_total
            ),
            "spatial_position": classify_spatial_position(
                mid_bbox, img_width, img_height
            ),
            "timestamp_start": frame_to_timestamp(ov_start, timestamps),
            "timestamp_end": frame_to_timestamp(ov_end, timestamps),
            "mean_iou": round(iou_sum / max(1, duration), 3),
        }

    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            t1, t2 = tracks[i], tracks[j]
            total_n = min(len(t1["bboxes"]), len(t2["bboxes"]))
            if total_n == 0:
                continue

            overlap_start = None
            iou_accum = 0.0
            for k in range(total_n):
                both_vis = t1["visible"][k] and t2["visible"][k]
                iou = (compute_bbox_iou(t1["bboxes"][k], t2["bboxes"][k])
                       if both_vis else 0.0)

                if iou > iou_thresh:
                    if overlap_start is None:
                        overlap_start = k
                        iou_accum = 0.0
                    iou_accum += iou
                else:
                    if overlap_start is not None:
                        ev = _emit_interaction(
                            t1, t2, overlap_start, k - 1,
                            iou_accum, total_n
                        )
                        if ev:
                            events.append(ev)
                        overlap_start = None

            # Handle segment at end of video
            if overlap_start is not None:
                ev = _emit_interaction(
                    t1, t2, overlap_start, total_n - 1,
                    iou_accum, total_n
                )
                if ev:
                    events.append(ev)

    return events


# ============================================================
# Per-Object Timeline Summary (for cross-event QA)
# ============================================================

def build_object_timelines(tracks, events):
    """
    For each tracked object, build a chronological timeline of its events
    plus basic visibility stats.  Used by Step 3 skeleton construction
    for cross-event and cross-object reasoning units.
    """
    timelines = {}
    for t in tracks:
        oid = t["obj_id"]
        vis_frames = sum(1 for v in t["visible"] if v)
        total = len(t["visible"])
        first_vis = next(
            (i for i, v in enumerate(t["visible"]) if v), None
        )
        last_vis = next(
            (total - 1 - i
             for i, v in enumerate(reversed(t["visible"])) if v),
            None
        )
        timelines[oid] = {
            "obj_id": oid,
            "label": t["label"],
            "total_frames": total,
            "visible_frames": vis_frames,
            "visibility_ratio": round(vis_frames / max(1, total), 3),
            "first_visible_frame": first_vis,
            "last_visible_frame": last_vis,
            "events": [],
        }

    # Attach events to their objects in chronological order
    for e in events:
        if "obj_id" in e:
            oid = e["obj_id"]
            if oid in timelines:
                timelines[oid]["events"].append(e)
        elif "obj_ids" in e:
            for oid in e["obj_ids"]:
                if oid in timelines:
                    timelines[oid]["events"].append(e)

    # Sort each timeline and compute event_type_set
    for tl in timelines.values():
        tl["events"].sort(
            key=lambda e: e.get("frame", e.get("start_frame", 0))
        )
        tl["event_type_set"] = sorted(set(
            e["type"] for e in tl["events"]
        ))
        tl["num_events"] = len(tl["events"])

    return timelines


# ============================================================
# Combine All Events
# ============================================================

def detect_all_events(track_data, config):
    """
    Run full event detection on a single video's tracking data.

    Returns a dict with:
      - events:              list of event dicts (skeleton-ready)
      - object_timelines:    per-object chronological event summaries
      - event_counts:        Counter of event types
      - distinct_event_types: sorted list of types present
    """
    tracks = track_data.get("objects", [])
    metadata = track_data.get("metadata", {})
    timestamps = track_data.get("timestamps", [])
    w = metadata.get("width", 640)
    h = metadata.get("height", 480)

    all_events = []

    # Per-object events (pass full tracks list for occluder detection)
    for track in tracks:
        all_events.extend(
            detect_object_events(
                track, config, w, h,
                timestamps=timestamps,
                tracks_all=tracks,
            )
        )

    # Pairwise events
    all_events.extend(
        detect_interactions(tracks, config, w, h, timestamps=timestamps)
    )

    # Sort by frame
    all_events.sort(
        key=lambda e: e.get("frame", e.get("start_frame", 0))
    )

    # Assign globally unique event_id (chronological order)
    for idx, e in enumerate(all_events):
        e["event_id"] = idx

    event_counts = Counter(e["type"] for e in all_events)
    distinct_types = sorted(set(e["type"] for e in all_events))
    distinct_meaningful = set(distinct_types) - {"state_change"}

    # Build per-object timelines
    object_timelines = build_object_timelines(tracks, all_events)

    result = {
        "video_id": track_data["video_id"],
        "source": track_data.get("source", "unknown"),
        "num_objects": len(tracks),
        "num_frames": len(timestamps) if timestamps else 0,
        "num_events": len(all_events),
        "event_counts": dict(event_counts),
        "distinct_event_types": distinct_types,
        "num_distinct_meaningful_types": len(distinct_meaningful),
        "events": all_events,
        "object_labels": [t["label"] for t in tracks],
        "object_timelines": {
            str(k): v for k, v in object_timelines.items()
        },
    }

    # --- v2 Augmentations (merged from step1c_augmentations.py) ---
    _apply_v2_augmentations(result, track_data)
    return result


# ============================================================
# v2 AUGMENTATION — label parsing, sibling groups, cross-object stats
# ============================================================
# These extend events.json with fields needed by the v2 DIM templates.
# Each section is a self-contained helper; _apply_v2_augmentations is the
# single entry point that orchestrates them.
# ============================================================

_DIGIT_SUFFIX_RE = re.compile(r"#\d+$")

# Trailing prepositional phrases that obscure the base noun.
# E.g. "cup on the left" — we want base=cup, not base=left.
# These are stripped BEFORE taking the last word as base_noun.
_TRAILING_PP_RE = re.compile(
    r"\s+(?:on|in|at|near|beside|next to|by|of|from|behind|in front of|"
    r"above|below|under|over)\s+the\s+\w+$",
    re.IGNORECASE,
)


def parse_label(raw_label: str) -> dict:
    """Parse a ref-expr label into base noun + modifier info.

    Heuristic: base noun = last content word AFTER stripping:
      - digit suffix: "cup #2" → "cup"
      - leading articles: "the red cup" → "red cup"
      - trailing PP: "cup on the left" → "cup"

    Returns a dict with:
        raw                     : original string
        normalized              : lowercased
        has_digit_suffix        : bool
        base_noun               : str (last content word)
        modifiers_text          : everything before base_noun
        has_spatial_modifier    : bool (leftmost/upper/...)
        has_temporal_modifier   : bool (first/earlier/...)
        has_count_modifier      : bool (twice/multiple/...)
    """
    label = (raw_label or "").strip().lower()
    info = {
        "raw": raw_label,
        "normalized": label,
        "has_digit_suffix": bool(_DIGIT_SUFFIX_RE.search(label)),
    }
    # 1. Strip digit suffix "cup #2" -> "cup"
    label_nosuffix = _DIGIT_SUFFIX_RE.sub("", label).strip()
    # 2. Strip leading articles
    for article in ("the ", "a ", "an "):
        if label_nosuffix.startswith(article):
            label_nosuffix = label_nosuffix[len(article):]
            break
    # 3. Strip trailing prepositional phrase (for base_noun extraction only;
    #    the original form is retained for modifier detection)
    label_stripped = _TRAILING_PP_RE.sub("", label_nosuffix).strip()
    words = label_stripped.split()
    if not words:
        info["base_noun"] = label_nosuffix
        info["modifiers_text"] = ""
    else:
        info["base_noun"] = words[-1]
        info["modifiers_text"] = " ".join(words[:-1])

    # Detect forbidden modifier categories on the FULL label (not stripped)
    for cat, markers in REF_EXPR_MODIFIERS.items():
        flag_name = f"has_{cat}_modifier"
        info[flag_name] = any(m in label for m in markers)

    return info


def _augment_object_timelines(event_data, track_data):
    """Add per-object fields: event_type_counts, event_type_unique,
    label_info, instance_confidence.
    """
    timelines = event_data.get("object_timelines", {})
    tracks_by_oid = {t["obj_id"]: t for t in track_data.get("objects", [])}

    for oid_str, tl in timelines.items():
        oid = tl.get("obj_id")

        # event_type_counts & event_type_unique (single-object events only)
        per_type_counts = Counter()
        for ev in tl.get("events", []):
            if ev.get("type") == "interaction":
                continue
            if ev.get("obj_id") == oid:
                per_type_counts[ev["type"]] += 1
        tl["event_type_counts"] = dict(per_type_counts)
        tl["event_type_unique"] = {
            etype: (cnt == 1) for etype, cnt in per_type_counts.items()
        }

        # label parsing
        tl["label_info"] = parse_label(tl.get("label", ""))

        # instance_confidence from postprocessed tracks.json
        track = tracks_by_oid.get(oid)
        tl["instance_confidence"] = (
            track.get("instance_confidence") if track else None
        )


def _compute_sibling_groups(event_data):
    """Group obj_ids that share a base noun (≥2 members per group).

    Feeds T2.3 reappear_identity: groups with 2+ siblings are candidates
    for identity-confusion questions.
    """
    timelines = event_data.get("object_timelines", {})
    groups_by_base = defaultdict(list)
    for oid_str, tl in timelines.items():
        base = tl.get("label_info", {}).get("base_noun", "")
        if not base:
            continue
        groups_by_base[base].append({
            "obj_id": tl.get("obj_id"),
            "label": tl.get("label"),
            "visibility_ratio": tl.get("visibility_ratio", 0),
            "instance_confidence": tl.get("instance_confidence"),
        })
    return [
        {"base_noun": base,
         "members": sorted(members, key=lambda x: x["obj_id"] or 0)}
        for base, members in groups_by_base.items()
        if len(members) >= 2
    ]


def _compute_pair_covisibility(track_data, min_covisible_frames=3):
    """For each pair (oid_a < oid_b), compute list of frames where both
    are visible. Feeds T3.1 conditional_state.

    Returns a list of pair summaries for pairs with ≥min_covisible_frames.
    """
    if not EVENT_STATS_CONFIG.get("compute_pair_covisibility", True):
        return []
    tracks = track_data.get("objects", [])
    n_frames = len(track_data.get("timestamps", []))
    if n_frames == 0 and tracks:
        n_frames = len(tracks[0].get("visible", []))

    pairs = []
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            ta, tb = tracks[i], tracks[j]
            va = ta.get("visible", [])
            vb = tb.get("visible", [])
            if not va or not vb:
                continue
            m = min(len(va), len(vb))
            covisible = [k for k in range(m) if va[k] and vb[k]]
            if len(covisible) < min_covisible_frames:
                continue
            pairs.append({
                "obj_id_a": ta["obj_id"],
                "obj_id_b": tb["obj_id"],
                "label_a": ta["label"],
                "label_b": tb["label"],
                "covisible_frames": covisible,
                "covisibility_ratio": round(
                    len(covisible) / max(1, n_frames), 3),
            })
    return pairs


def _compute_cross_event_pairs(event_data, min_frame_gap=2):
    """For each object, list chronologically ordered pairs of events with
    distinct types (and non-interaction, non-state_change). Feeds T2.2
    event_ordering and helps T3.2 cross_object_order.
    """
    pairs = []
    timelines = event_data.get("object_timelines", {})
    for oid_str, tl in timelines.items():
        oid = tl.get("obj_id")
        own_events = [
            e for e in tl.get("events", [])
            if e.get("type") not in ("interaction", "state_change")
            and e.get("obj_id") == oid
        ]
        own_events.sort(
            key=lambda e: e.get("frame", e.get("start_frame", 0)))
        for i in range(len(own_events)):
            for j in range(i + 1, len(own_events)):
                e_a, e_b = own_events[i], own_events[j]
                if e_a["type"] == e_b["type"]:
                    continue
                fa = e_a.get("frame", e_a.get("start_frame", 0))
                fb = e_b.get("frame", e_b.get("start_frame", 0))
                if fb - fa < min_frame_gap:
                    continue
                pairs.append({
                    "obj_id": oid,
                    "event_id_a": e_a.get("event_id"),
                    "event_id_b": e_b.get("event_id"),
                    "type_a": e_a["type"],
                    "type_b": e_b["type"],
                    "frame_a": fa,
                    "frame_b": fb,
                })
    return pairs


def _compute_reappear_identity_candidates(event_data, track_data):
    """Pre-compute (target, sibling) pairs that satisfy T2.3 preconditions.

    Preconditions (from config.EVENT_STATS_CONFIG.sibling_detection):
      - target has a reappear event preceded by full_occlusion or exit_frame
      - target's occlusion duration ≥ min_occlusion_duration_sec
      - sibling shares base_noun with target but differs in modifiers
      - sibling is visible during ≥min_covisibility_ratio of target's
        occlusion period
      - target's reappear position is ≥min_reappear_position_distance_ratio
        (of image diagonal) away from sibling's position at reappear frame
    """
    cfg = EVENT_STATS_CONFIG["sibling_detection"]
    min_covis = cfg["min_covisibility_ratio_during_target_occlusion"]
    min_pos_dist = cfg["min_reappear_position_distance_ratio"]
    min_occ_sec = cfg["min_occlusion_duration_sec"]

    metadata = track_data.get("metadata", {})
    width = metadata.get("width", 1280)
    height = metadata.get("height", 720)
    fps = metadata.get("fps", 3)
    img_diag = (width ** 2 + height ** 2) ** 0.5

    tracks_by_oid = {t["obj_id"]: t for t in track_data.get("objects", [])}
    timelines = event_data.get("object_timelines", {})

    def _center(box):
        if not box or box[2] <= box[0] or box[3] <= box[1]:
            return None
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    candidates = []
    for oid_str, tl in timelines.items():
        oid = tl.get("obj_id")
        target_base = tl.get("label_info", {}).get("base_noun", "")
        if not target_base:
            continue

        # Find reappear events for this target
        for e in tl.get("events", []):
            if e.get("type") != "reappear" or e.get("obj_id") != oid:
                continue
            reappear_frame = e.get("frame", e.get("start_frame"))
            if reappear_frame is None:
                continue

            # Find preceding full_occlusion or exit_frame on same object
            prev_occ = None
            for e2 in tl.get("events", []):
                if e2.get("type") not in ("full_occlusion", "exit_frame"):
                    continue
                if e2.get("obj_id") != oid:
                    continue
                end_f = e2.get("end_frame", e2.get("frame"))
                if end_f is not None and end_f <= reappear_frame:
                    prev_occ = e2  # take latest
            if prev_occ is None:
                continue
            occ_start = prev_occ.get("start_frame", prev_occ.get("frame"))
            occ_end = prev_occ.get("end_frame", prev_occ.get("frame"))
            if occ_start is None or occ_end is None:
                continue
            occ_duration_sec = (occ_end - occ_start) / max(1, fps)
            if occ_duration_sec < min_occ_sec:
                continue

            # Check each same-base-noun sibling
            for sib_oid_str, sib_tl in timelines.items():
                sib_oid = sib_tl.get("obj_id")
                if sib_oid == oid:
                    continue
                sib_info = sib_tl.get("label_info", {})
                if sib_info.get("base_noun", "") != target_base:
                    continue
                # Must differ in modifiers (else same exact label)
                if sib_info.get("modifiers_text") == \
                        tl.get("label_info", {}).get("modifiers_text"):
                    continue

                # Sibling visibility during target's occlusion window
                sib_track = tracks_by_oid.get(sib_oid)
                if not sib_track:
                    continue
                visible_arr = sib_track.get("visible", [])
                window = visible_arr[occ_start:occ_end + 1]
                if not window:
                    continue
                covis_ratio = sum(1 for v in window if v) / len(window)
                if covis_ratio < min_covis:
                    continue

                # Position distance at reappear frame
                target_track = tracks_by_oid.get(oid)
                if not target_track:
                    continue
                if reappear_frame >= len(target_track.get("bboxes", [])):
                    continue
                t_center = _center(target_track["bboxes"][reappear_frame])
                if t_center is None:
                    continue
                if reappear_frame >= len(sib_track.get("bboxes", [])):
                    continue
                s_center = _center(sib_track["bboxes"][reappear_frame])
                if s_center is None:
                    continue
                pos_dist = (
                    (t_center[0] - s_center[0]) ** 2
                    + (t_center[1] - s_center[1]) ** 2
                ) ** 0.5
                if pos_dist / img_diag < min_pos_dist:
                    continue

                candidates.append({
                    "target_obj_id": oid,
                    "target_label": tl.get("label"),
                    "sibling_obj_id": sib_oid,
                    "sibling_label": sib_tl.get("label"),
                    "reappear_event_id": e.get("event_id"),
                    "reappear_frame": reappear_frame,
                    "occlusion_duration_sec": round(occ_duration_sec, 2),
                    "sibling_covisibility": round(covis_ratio, 3),
                    "position_distance_ratio": round(pos_dist / img_diag, 3),
                })

    return candidates


def _apply_v2_augmentations(event_data, track_data):
    """Single entry point for v2 augmentations. Mutates event_data in place
    to add: per-object (event_type_counts, event_type_unique, label_info,
    instance_confidence), sibling_groups, pair_covisibility, cross_event_pairs,
    reappear_identity_candidates, v2_augmented.
    """
    _augment_object_timelines(event_data, track_data)
    event_data["sibling_groups"] = _compute_sibling_groups(event_data)
    event_data["pair_covisibility"] = _compute_pair_covisibility(track_data)
    event_data["cross_event_pairs"] = _compute_cross_event_pairs(event_data)
    event_data["reappear_identity_candidates"] = \
        _compute_reappear_identity_candidates(event_data, track_data)
    event_data["v2_augmented"] = True


# ============================================================
# Phenomenon Profile (for Step 2 video selection)
# ============================================================

def compute_phenomenon_profile(event_data):
    """
    Compute a boolean profile of which phenomenon slots this video
    satisfies.  Used by Step 2 video selection to drive phenomenon-first
    coverage.

    v2 adds slots that flag video-level feasibility for the new dims:
      - has_reappear_identity_candidate: at least one T2.3 candidate was
        precomputed (requires sibling + qualifying occlusion)
      - has_conditional_state_pair: at least one object pair with high
        covisibility and distinct base nouns (T3.1)
      - has_countable_event: at least one object has ≥2 of the same event
        type (T2.1 event_count)
      - has_high_confidence_objects: ≥2 objects with instance_confidence≥0.7
        (prerequisite for most Tier 2/3 dims)
      - has_cross_event_chain: ≥3 distinct event types on one object
        (T2.2 event_ordering)
    """
    ec = event_data.get("event_counts", {})
    distinct = set(event_data.get("distinct_event_types", []))

    has_full_occ = ec.get("full_occlusion", 0) > 0
    has_reappear = ec.get("reappear", 0) > 0
    has_partial = ec.get("partial_occlusion", 0) > 0
    has_exit = ec.get("exit_frame", 0) > 0
    has_enter = ec.get("enter_frame", 0) > 0
    has_appear = ec.get("appear", 0) > 0
    has_disappear = ec.get("disappear", 0) > 0
    has_state_change = ec.get("state_change", 0) > 0
    has_interaction = ec.get("interaction", 0) > 0

    meaningful_types = distinct - {"state_change"}

    # v2 profile computations
    timelines = event_data.get("object_timelines", {})
    reappear_candidates = event_data.get("reappear_identity_candidates", [])
    pair_covis = event_data.get("pair_covisibility", [])

    # T2.3 signal
    v2_has_reappear_identity = len(reappear_candidates) > 0

    # T3.1 signal: at least one pair with cross-base-noun and enough
    # co-visibility (0.3 threshold). Accept pair if the two objects have
    # DIFFERENT base nouns (since same-base-noun is T2.3 territory).
    v2_has_conditional_state_pair = False
    for pair in pair_covis:
        if pair.get("covisibility_ratio", 0) < 0.3:
            continue
        # Infer base nouns from timeline label_info (best-effort)
        tl_a = timelines.get(str(pair["obj_id_a"]), {})
        tl_b = timelines.get(str(pair["obj_id_b"]), {})
        base_a = tl_a.get("label_info", {}).get("base_noun", "")
        base_b = tl_b.get("label_info", {}).get("base_noun", "")
        if base_a and base_b and base_a != base_b:
            v2_has_conditional_state_pair = True
            break

    # T2.1 signal
    v2_has_countable_event = any(
        any(cnt >= 2 for cnt in tl.get("event_type_counts", {}).values())
        for tl in timelines.values()
    )

    # C-gate signal: how many high-confidence objects
    v2_num_hi_conf_objects = sum(
        1 for tl in timelines.values()
        if (tl.get("instance_confidence") or 0) >= 0.7
    )

    # T2.2 signal: any object with ≥3 distinct event types
    v2_has_cross_event_chain = any(
        len({e["type"] for e in tl.get("events", [])
             if e.get("type") not in ("interaction", "state_change")}) >= 3
        for tl in timelines.values()
    )

    return {
        # --- v1 slots (unchanged) ---
        "occlusion_reappear": has_full_occ and has_reappear,
        "partial_occlusion": has_partial,
        "exit_reenter": has_exit and (has_enter or has_reappear),
        "appear_disappear": has_appear or has_disappear,
        "state_change": has_state_change,
        "interaction": has_interaction,
        "multi_phenomenon": len(meaningful_types) >= 3,

        # --- v2 slots (new) ---
        "has_reappear_identity_candidate": v2_has_reappear_identity,
        "has_conditional_state_pair": v2_has_conditional_state_pair,
        "has_countable_event": v2_has_countable_event,
        "has_cross_event_chain": v2_has_cross_event_chain,
        "has_high_confidence_objects": v2_num_hi_conf_objects >= 2,

        # Raw counts for tie-breaking in Step 2
        "phenomenon_instance_counts": {
            "occlusion_reappear": (ec.get("full_occlusion", 0)
                                   + ec.get("reappear", 0)),
            "partial_occlusion": ec.get("partial_occlusion", 0),
            "exit_reenter": (ec.get("exit_frame", 0)
                             + ec.get("enter_frame", 0)),
            "appear_disappear": (ec.get("appear", 0)
                                 + ec.get("disappear", 0)),
            "state_change": ec.get("state_change", 0),
            "interaction": ec.get("interaction", 0),
            # v2 counts
            "reappear_identity_candidates": len(reappear_candidates),
            "conditional_state_pairs": sum(
                1 for p in pair_covis
                if p.get("covisibility_ratio", 0) >= 0.3),
            "hi_confidence_objects": v2_num_hi_conf_objects,
        },
    }


# ============================================================
# Video Filtering
# ============================================================

def filter_video(event_data, track_data, config):
    reasons = []
    passed = True

    n_objects = event_data["num_objects"]
    if n_objects < config["min_tracked_objects"]:
        reasons.append(f"too_few_objects ({n_objects})")
        passed = False

    ec = event_data.get("event_counts", {})

    # Count meaningful events (excluding state_change)
    n_meaningful = sum(v for k, v in ec.items() if k != "state_change")
    if n_meaningful < config["min_total_events"]:
        reasons.append(f"too_few_meaningful_events ({n_meaningful})")
        passed = False

    # --- Occlusion richness check ---
    has_full_occ = ec.get("full_occlusion", 0) > 0
    has_reappear = ec.get("reappear", 0) > 0
    has_partial = ec.get("partial_occlusion", 0) > 0
    has_exit = ec.get("exit_frame", 0) > 0
    has_enter = ec.get("enter_frame", 0) > 0
    n_interactions = ec.get("interaction", 0)

    has_full_occ_reappear = has_full_occ and has_reappear
    has_exit_reenter = has_exit and (has_enter or has_reappear)
    is_occlusion_interesting = (has_full_occ_reappear
                                or has_exit_reenter
                                or has_partial)

    if not is_occlusion_interesting and n_interactions < config["min_interactions"]:
        reasons.append("no_occlusion_events_and_few_interactions")
        passed = False

    # --- Duration check ---
    metadata = track_data.get("metadata", {})
    duration = metadata.get("duration", 0)
    if duration < config["min_video_duration_sec"]:
        reasons.append(f"too_short ({duration:.1f}s)")
        passed = False
    if duration > config["max_video_duration_sec"]:
        reasons.append(f"too_long ({duration:.1f}s)")
        passed = False

    # --- Multi-object involvement ---
    obj_ids_in_events = set()
    for e in event_data.get("events", []):
        if "obj_id" in e:
            obj_ids_in_events.add(e["obj_id"])
        if "obj_ids" in e:
            obj_ids_in_events.update(e["obj_ids"])

    if len(obj_ids_in_events) < 2:
        reasons.append(f"events_involve_{len(obj_ids_in_events)}_objects")
        passed = False

    # --- Quality score ---
    quality_score = (
        ec.get("full_occlusion", 0) * 5 +
        ec.get("reappear", 0) * 5 +
        ec.get("partial_occlusion", 0) * 3 +
        ec.get("exit_frame", 0) * 3 +
        ec.get("enter_frame", 0) * 3 +
        ec.get("appear", 0) * 2 +
        ec.get("disappear", 0) * 2 +
        ec.get("state_change", 0) * 1 +
        n_interactions * 3 +
        n_objects * 1 +
        (10 if has_full_occ_reappear else 0) +
        (5 if has_exit_reenter else 0)
    )

    # --- Phenomenon profile (for Step 2 selection) ---
    phenomenon_profile = compute_phenomenon_profile(event_data)

    return {
        "passed": passed,
        "reasons": reasons if not passed else ["all_criteria_met"],
        "quality_score": quality_score,
        "phenomenon_profile": phenomenon_profile,
        "stats": {
            "num_objects": n_objects,
            "num_events": event_data.get("num_events", 0),
            "num_distinct_meaningful_types":
                event_data.get("num_distinct_meaningful_types", 0),
            "has_full_occ_reappear": has_full_occ_reappear,
            "has_partial_occlusion": has_partial,
            "num_interactions": n_interactions,
            "duration": round(duration, 1),
            "objects_in_events": len(obj_ids_in_events),
            "event_counts": ec,
        },
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="TOC-Bench Step 1c: Event detection & video filtering"
    )
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    print("=" * 60)
    print("  TOC-Bench Step 1c: Event Detection & Video Filtering")
    print("  9 event types: appear, enter_frame, exit_frame,")
    print("  partial_occlusion, full_occlusion, reappear,")
    print("  disappear, state_change, interaction")
    print("=" * 60)

    track_files = sorted(TRACKS_DIR.glob("*.json"))
    track_files = [f for f in track_files if not f.name.startswith("_")]

    if args.source:
        track_files = [f for f in track_files if args.source in f.stem]

    print(f"  Found {len(track_files)} track files")

    if not track_files:
        print("  [ERROR] No track files found. Run step1b first.")
        sys.exit(1)

    all_filter_results = []

    for tf in tqdm(
        track_files,
        desc="Step1c detecting & filtering",
        total=len(track_files),
        dynamic_ncols=True,
    ):
        with open(tf) as f:
            track_data = json.load(f)

        event_data = detect_all_events(track_data, EVENT_CONFIG)

        event_path = EVENTS_DIR / tf.name
        with open(event_path, "w") as f:
            json.dump(event_data, f, indent=2)

        filter_result = filter_video(event_data, track_data, FILTER_CONFIG)
        filter_result["video_id"] = track_data["video_id"]
        filter_result["source"] = track_data.get("source", "unknown")
        all_filter_results.append(filter_result)

    passed = sorted(
        [r for r in all_filter_results if r["passed"]],
        key=lambda r: r["quality_score"], reverse=True,
    )
    failed = [r for r in all_filter_results if not r["passed"]]

    filtered_output = {
        "total_videos": len(all_filter_results),
        "passed": len(passed),
        "failed": len(failed),
        "pass_rate": round(
            len(passed) / max(1, len(all_filter_results)) * 100, 1
        ),
        "videos": passed,
    }
    filtered_path = FILTERED_DIR / "filtered_videos.json"
    with open(filtered_path, "w") as f:
        json.dump(filtered_output, f, indent=2)

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  Results")
    print(f"{'='*60}")
    print(f"  Total analyzed: {len(all_filter_results)}")
    print(f"  Passed:         {len(passed)} ({filtered_output['pass_rate']}%)")
    print(f"  Failed:         {len(failed)}")
    print(f"  Saved to:       {filtered_path}")

    # Per-source breakdown
    print(f"\n  Per-source:")
    source_stats = defaultdict(lambda: {"total": 0, "passed": 0})
    for r in all_filter_results:
        src = r["source"]
        source_stats[src]["total"] += 1
        if r["passed"]:
            source_stats[src]["passed"] += 1

    for src, s in sorted(source_stats.items()):
        rate = s["passed"] / max(1, s["total"]) * 100
        print(f"    {src:20s}  {s['passed']:4d} / {s['total']:4d}"
              f"  ({rate:.0f}%)")

    # Event distribution (passed videos)
    if passed:
        print(f"\n  Event distribution (passed videos):")
        total_ec = Counter()
        for r in passed:
            for etype, cnt in r["stats"]["event_counts"].items():
                total_ec[etype] += cnt
        for etype, cnt in total_ec.most_common():
            avg = cnt / len(passed)
            print(f"    {etype:25s}  total={cnt:5d}  avg/video={avg:.1f}")

    # Phenomenon slot coverage
    if passed:
        print(f"\n  Phenomenon slot coverage (passed videos):")
        slot_names = [
            "occlusion_reappear", "partial_occlusion", "exit_reenter",
            "appear_disappear", "state_change", "interaction",
            "multi_phenomenon",
        ]
        for slot in slot_names:
            count = sum(
                1 for r in passed
                if r.get("phenomenon_profile", {}).get(slot, False)
            )
            pct = count / len(passed) * 100 if passed else 0
            print(f"    {slot:25s}  {count:5d} / {len(passed)}"
                  f"  ({pct:.0f}%)")

    # Top-K
    if args.stats and passed:
        print(f"\n  Top-{args.top_k} by quality score:")
        for r in passed[:args.top_k]:
            ec = r["stats"]["event_counts"]
            n_types = r["stats"].get("num_distinct_meaningful_types", "?")
            print(f"    {r['video_id']:40s}  score={r['quality_score']:4d}"
                  f"  obj={r['stats']['num_objects']}"
                  f"  types={n_types}"
                  f"  full_occ={ec.get('full_occlusion', 0)}"
                  f"  partial={ec.get('partial_occlusion', 0)}"
                  f"  reappear={ec.get('reappear', 0)}"
                  f"  interact={r['stats']['num_interactions']}")

    # Failure reasons
    if failed:
        print(f"\n  Failure reasons:")
        reason_counts = Counter()
        for r in failed:
            for reason in r["reasons"]:
                reason_counts[reason.split("(")[0].strip()] += 1
        for reason, cnt in reason_counts.most_common():
            print(f"    {reason:40s}  {cnt}")


if __name__ == "__main__":
    main()