#!/usr/bin/env python3
"""
TOC-Bench Step 3a v2: Reasoning Unit Extraction for 11-Dim System
==================================================================
Per-video, reads v2-augmented events.json and produces reasoning units:
each unit corresponds to ONE candidate QA (one dim, one subject, one
concrete anchor fact). step3b later converts units to skeletons.

KEY DIFFERENCES FROM v1:
  - 11 dims (was 16), each with its own builder function
  - Hard preconditions enforced here (not delegated to step3b)
  - ref_expr modifier conflicts filtered at this layer
  - C-critical dims check instance_confidence >= 0.7
  - Multi-event uniqueness handled via "first/last" prefix (T1.1)
    or skip (T1.2 / SP.2 / others)

OUTPUT UNITS: each unit is self-contained and carries everything step3b
needs to build a skeleton without re-querying event_data. Units do NOT
yet have distractors — step3b's job.

Input:
    EVENTS_DIR/<video_id>.json   (v2-augmented from step1c)
    TRACKS_DIR/<video_id>.json   (postprocessed)
    FILTERED_DIR/selected_videos.json

Output:
    QA_DIR/reasoning_units/<video_id>.json
    QA_DIR/reasoning_units/_summary.json
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *a, **k): return x

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    ROOT, EVENTS_DIR, TRACKS_DIR, FILTERED_DIR, QA_DIR,
    DIMENSIONS, TIME_BUCKETS, DURATION_BUCKETS,
    SPATIAL_DIRECTION_OPTIONS, SPATIAL_CHANGE_THRESHOLDS,
    CONDITIONAL_STATE_OPTIONS,
    REF_EXPR_MODIFIERS, REF_EXPR_CONFIG, SAM3_POSTPROC_CONFIG,
    EVENT_STATS_CONFIG, EVENT_CONFIG,
    ref_expr_has_forbidden_modifier,
)

UNITS_DIR = QA_DIR / "reasoning_units"
MIN_REF_CONFIDENCE = REF_EXPR_CONFIG["min_confidence"]
C_DIM_MIN_CONFIDENCE = SAM3_POSTPROC_CONFIG["c_dim_min_confidence"]


# ============================================================================
# Event verb phrases — used across dims for natural question text
# ============================================================================
# These are anchor phrases; step3c may lightly rephrase them, but the
# semantic content is fixed here so answer construction stays consistent.
EVENT_VERB_PHRASES = {
    "full_occlusion":    {"verb_present": "gets fully hidden",
                          "verb_past":    "was fully hidden",
                          "state":        "fully hidden from view"},
    "partial_occlusion": {"verb_present": "gets partly hidden",
                          "verb_past":    "was partly hidden",
                          "state":        "partly hidden"},
    "reappear":          {"verb_present": "comes back into view",
                          "verb_past":    "came back into view",
                          "state":        "back in view"},
    "appear":            {"verb_present": "first shows up",
                          "verb_past":    "first showed up",
                          "state":        "visible for the first time"},
    "disappear":         {"verb_present": "disappears for good",
                          "verb_past":    "disappeared for good",
                          "state":        "gone"},
    "exit_frame":        {"verb_present": "leaves the frame",
                          "verb_past":    "left the frame",
                          "state":        "out of the frame"},
    "enter_frame":       {"verb_present": "enters the frame",
                          "verb_past":    "entered the frame",
                          "state":        "just entered"},
    "state_change":      {"verb_present": "changes abruptly",
                          "verb_past":    "changed abruptly",
                          "state":        "in a changed state"},
    "interaction":       {"verb_present": "interacts with another object",
                          "verb_past":    "interacted with another object",
                          "state":        "in contact"},
}


def event_phrase(etype, form="verb_present"):
    return EVENT_VERB_PHRASES.get(etype, {}).get(form, etype.replace("_", " "))


# ============================================================================
# Helpers
# ============================================================================

def event_center_frame(event):
    """Get the canonical frame for an event (point or ranged)."""
    if "frame" in event:
        return event["frame"]
    if "start_frame" in event and "end_frame" in event:
        return (event["start_frame"] + event["end_frame"]) // 2
    return event.get("start_frame", 0)


def event_start_frame(event):
    return event.get("start_frame", event.get("frame", 0))


def event_end_frame(event):
    return event.get("end_frame", event.get("frame", 0))


def event_duration_seconds(event, fps):
    """For ranged events: (end - start) / fps, else 0."""
    if "start_frame" in event and "end_frame" in event:
        return (event["end_frame"] - event["start_frame"]) / max(1, fps)
    return 0.0


def ratio_to_temporal_bucket(ratio):
    """Map [0,1] ratio → temporal bucket label from config."""
    ratio = max(0.0, min(0.9999, ratio))
    for name, (lo, hi) in TIME_BUCKETS.items():
        if lo <= ratio < hi:
            return name
    return list(TIME_BUCKETS.keys())[-1]


def seconds_to_duration_bucket(seconds):
    """Map seconds → duration bucket label from config."""
    for name, (lo, hi) in DURATION_BUCKETS.items():
        if lo <= seconds < hi:
            return name
    return list(DURATION_BUCKETS.keys())[-1]


def object_passes_c_gate(timeline):
    """True if object has instance_confidence >= threshold (for C-dim gate).

    Returns True when confidence is None AND the v2 config doesn't enforce,
    but emits a warning via return value (None). We treat None as "unknown",
    and default to PASS (backward compat during migration). Callers that
    want strictness should explicitly check `== True`.
    """
    conf = timeline.get("instance_confidence")
    if conf is None:
        return True  # conservative: pass during migration
    return conf >= C_DIM_MIN_CONFIDENCE


def ref_expr_ok_for_dim(label, dim_name):
    """Check if a label's ref expr is allowed as the SUBJECT of this dim."""
    dim_cfg = DIMENSIONS.get(dim_name, {})
    forbidden = dim_cfg.get("ref_expr_forbidden_modifiers", [])
    if not forbidden:
        return True
    return not ref_expr_has_forbidden_modifier(label, forbidden)


def make_unit(dim, video_id, subject, payload):
    """Wrap payload into a standard unit envelope."""
    return {
        "dim": dim,
        "video_id": video_id,
        "subject": subject,
        **payload,
    }


# ============================================================================
# TIER 1 — O+T baseline dims
# ============================================================================

def build_temporal_location_units(video_id, event_data, fps):
    """T1.1 temporal_location: when in the video does subject do event?

    Multi-event policy:
      count == 1 → emit 1 unit, question w/o prefix
      count == 2 → emit 2 units (one for first, one for last), prefix added
      count >= 3 → SKIP
    """
    units = []
    timelines = event_data.get("object_timelines", {})
    n_frames = event_data.get("num_frames", 0)
    if n_frames <= 1:
        return units

    for oid_str, tl in timelines.items():
        label = tl.get("label", "")
        if not ref_expr_ok_for_dim(label, "temporal_location"):
            continue

        # Group this object's events by type
        by_type = defaultdict(list)
        for ev in tl.get("events", []):
            if ev.get("obj_id") != tl.get("obj_id"):
                continue
            if ev.get("type") in ("state_change", "interaction"):
                continue
            by_type[ev["type"]].append(ev)

        for etype, evs in by_type.items():
            evs.sort(key=event_center_frame)
            n = len(evs)
            if n == 0:
                continue
            if n >= 3:
                continue  # ambiguous, skip
            if n == 1:
                bucket = ratio_to_temporal_bucket(
                    event_center_frame(evs[0]) / (n_frames - 1))
                units.append(make_unit(
                    "temporal_location", video_id,
                    {"obj_id": tl["obj_id"], "label": label},
                    {
                        "event_type": etype,
                        "event_id": evs[0].get("event_id"),
                        "prefix": None,
                        "correct_bucket": bucket,
                    }
                ))
            else:  # n == 2, emit first/last
                for prefix, ev in [("first", evs[0]), ("last", evs[-1])]:
                    bucket = ratio_to_temporal_bucket(
                        event_center_frame(ev) / (n_frames - 1))
                    units.append(make_unit(
                        "temporal_location", video_id,
                        {"obj_id": tl["obj_id"], "label": label},
                        {
                            "event_type": etype,
                            "event_id": ev.get("event_id"),
                            "prefix": prefix,
                            "correct_bucket": bucket,
                        }
                    ))
    return units


def build_duration_category_units(video_id, event_data, fps):
    """T1.2 duration_category: how long does subject stay in state X?

    Applies only to ranged events (full_occlusion, partial_occlusion,
    exit_frame, interaction). Multi-event uniqueness: if count >= 2 but
    all instances fall in the SAME bucket, allow; else skip.
    """
    units = []
    timelines = event_data.get("object_timelines", {})
    RANGED_TYPES = {"full_occlusion", "partial_occlusion",
                    "exit_frame", "interaction"}

    for oid_str, tl in timelines.items():
        label = tl.get("label", "")
        if not ref_expr_ok_for_dim(label, "duration_category"):
            continue

        by_type = defaultdict(list)
        for ev in tl.get("events", []):
            if ev.get("type") not in RANGED_TYPES:
                continue
            if ev.get("type") == "interaction":
                # interaction is pairwise — only include if tl owns one side
                if tl["obj_id"] not in ev.get("obj_ids", []):
                    continue
            else:
                if ev.get("obj_id") != tl.get("obj_id"):
                    continue
            dur = event_duration_seconds(ev, fps)
            if dur < 0.3:  # too short to be meaningful
                continue
            by_type[ev["type"]].append((ev, dur))

        for etype, items in by_type.items():
            buckets = [seconds_to_duration_bucket(d) for _, d in items]
            unique_buckets = set(buckets)
            if len(unique_buckets) > 1:
                continue  # multi-event, multi-bucket → skip
            # If all fall in same bucket, use first event as anchor
            anchor_ev, anchor_dur = items[0]
            units.append(make_unit(
                "duration_category", video_id,
                {"obj_id": tl["obj_id"], "label": label},
                {
                    "event_type": etype,
                    "event_id": anchor_ev.get("event_id"),
                    "correct_bucket": buckets[0],
                    "duration_sec": round(anchor_dur, 2),
                    "partner_label": (
                        # for interaction events, record the partner
                        next(
                            (l for oid, l in zip(anchor_ev.get("obj_ids", []),
                                                 anchor_ev.get("labels", []))
                             if oid != tl["obj_id"]), None)
                        if etype == "interaction" else None
                    ),
                }
            ))
    return units


def build_relative_spatial_change_units(video_id, event_data, track_data):
    """T1.3 relative_spatial_change: how does subject move across frame?

    Classification by net_displacement (first→last visible bbox center)
    relative to image diagonal. 3-way thresholds in config:
      < 10%          → "stays in roughly the same place"
      10-20%         → SKIP (ambiguous)
      > 20%          → decide main axis:
          |dx| > 1.5 * |dy|: horizontal
          |dy| > 1.5 * |dx|: vertical
          else:              SKIP (diagonal)
    """
    units = []
    metadata = track_data.get("metadata", {})
    width = metadata.get("width", 1280)
    height = metadata.get("height", 720)
    img_diag = (width ** 2 + height ** 2) ** 0.5
    if img_diag <= 0:
        return units

    thresholds = SPATIAL_CHANGE_THRESHOLDS
    static_max = thresholds["static_max_net_displacement"]
    ambig_max = thresholds["ambiguous_zone_max"]
    main_ratio = thresholds["main_direction_ratio"]

    tracks_by_oid = {t["obj_id"]: t for t in track_data.get("objects", [])}
    timelines = event_data.get("object_timelines", {})

    for oid_str, tl in timelines.items():
        label = tl.get("label", "")
        if not ref_expr_ok_for_dim(label, "relative_spatial_change"):
            continue
        first_vis = tl.get("first_visible_frame")
        last_vis = tl.get("last_visible_frame")
        if first_vis is None or last_vis is None or first_vis == last_vis:
            continue

        track = tracks_by_oid.get(tl["obj_id"])
        if not track:
            continue
        bboxes = track.get("bboxes", [])
        if first_vis >= len(bboxes) or last_vis >= len(bboxes):
            continue
        b_first = bboxes[first_vis]
        b_last = bboxes[last_vis]
        if not (b_first[2] > b_first[0] and b_last[2] > b_last[0]):
            continue
        c_first = ((b_first[0] + b_first[2]) / 2, (b_first[1] + b_first[3]) / 2)
        c_last = ((b_last[0] + b_last[2]) / 2, (b_last[1] + b_last[3]) / 2)
        dx = c_last[0] - c_first[0]
        dy = c_last[1] - c_first[1]
        net_disp = (dx ** 2 + dy ** 2) ** 0.5
        ratio = net_disp / img_diag

        if ratio < static_max:
            answer = "stays in roughly the same place"
        elif ratio <= ambig_max:
            continue  # skip ambiguous
        else:
            # Main axis
            if abs(dx) > abs(dy) * main_ratio:
                answer = ("moves from left to right" if dx > 0
                          else "moves from right to left")
            elif abs(dy) > abs(dx) * main_ratio:
                answer = ("moves from top to bottom" if dy > 0
                          else "moves from bottom to top")
            else:
                continue  # diagonal → skip

        units.append(make_unit(
            "relative_spatial_change", video_id,
            {"obj_id": tl["obj_id"], "label": label},
            {
                "correct_answer": answer,
                "displacement_ratio": round(ratio, 3),
            }
        ))
    return units


def build_event_existence_units(video_id, event_data):
    """SP.1 event_existence: does X ever Y?

    For each object, for each event type it actually has → emit "yes" unit.
    For each object, for event types the video has but this object lacks →
    emit "no" unit (with that event type).
    Answer distribution should be roughly 50/50; step3b balances.
    """
    units = []
    timelines = event_data.get("object_timelines", {})
    video_event_types = set(event_data.get("distinct_event_types", []))
    EXCLUDE = {"state_change", "interaction"}  # these don't naturally fit "does X ever ..."

    for oid_str, tl in timelines.items():
        label = tl.get("label", "")
        obj_types = set(tl.get("event_type_counts", {}).keys())

        # Yes units: one per event type this object has
        for etype in obj_types - EXCLUDE:
            units.append(make_unit(
                "event_existence", video_id,
                {"obj_id": tl["obj_id"], "label": label},
                {
                    "event_type": etype,
                    "polarity": "yes",
                }
            ))

        # No units: pick one event type present in video but not in this obj
        # (cap at 1 to avoid explosion — otherwise ratio skews heavily to "no")
        possible_no = (video_event_types - obj_types) - EXCLUDE
        if possible_no:
            # Deterministic choice for reproducibility
            chosen = sorted(possible_no)[0]
            units.append(make_unit(
                "event_existence", video_id,
                {"obj_id": tl["obj_id"], "label": label},
                {
                    "event_type": chosen,
                    "polarity": "no",
                }
            ))
    return units


def build_reappear_or_disappear_units(video_id, event_data):
    """SP.2 reappear_or_disappear: was X fully blocked or left the frame?

    Applies to objects with a reappear or disappear event whose cause can
    be determined (full_occlusion vs exit_frame). Uniqueness required.
    """
    units = []
    timelines = event_data.get("object_timelines", {})

    for oid_str, tl in timelines.items():
        label = tl.get("label", "")
        events = tl.get("events", [])

        # Handle reappear: determine preceding cause
        reappear_events = [e for e in events
                           if e.get("type") == "reappear"
                           and e.get("obj_id") == tl["obj_id"]]
        if len(reappear_events) == 1:
            e = reappear_events[0]
            # look for preceding full_occlusion or exit_frame on same object
            cause = None
            for e2 in events:
                if e2.get("type") not in ("full_occlusion", "exit_frame"):
                    continue
                if e2.get("obj_id") != tl["obj_id"]:
                    continue
                end_f = event_end_frame(e2)
                reap_f = event_center_frame(e)
                if end_f <= reap_f:
                    cause = e2["type"]  # take latest
            if cause:
                units.append(make_unit(
                    "reappear_or_disappear", video_id,
                    {"obj_id": tl["obj_id"], "label": label},
                    {
                        "scenario": "reappear",
                        "event_id": e.get("event_id"),
                        "mechanism": (
                            "fully blocked" if cause == "full_occlusion"
                            else "left the frame"),
                    }
                ))

        # Handle disappear: cause encoded on the event itself
        disappear_events = [e for e in events
                            if e.get("type") == "disappear"
                            and e.get("obj_id") == tl["obj_id"]]
        if len(disappear_events) == 1:
            e = disappear_events[0]
            cause = e.get("cause")
            if cause in ("full_occlusion", "exit_frame"):
                units.append(make_unit(
                    "reappear_or_disappear", video_id,
                    {"obj_id": tl["obj_id"], "label": label},
                    {
                        "scenario": "disappear",
                        "event_id": e.get("event_id"),
                        "mechanism": (
                            "fully blocked" if cause == "full_occlusion"
                            else "left the frame"),
                    }
                ))
    return units


# ============================================================================
# TIER 2 — O+T+C core dims
# ============================================================================

def build_event_count_units(video_id, event_data):
    """T2.1 event_count: how many times does X Y? (Numerical)

    Requires:
      - C-gate (instance_confidence >= 0.7)
      - count >= 2 (count == 1 trivially known, skip)
      - event type in countable list
    """
    units = []
    timelines = event_data.get("object_timelines", {})
    COUNTABLE = set(EVENT_STATS_CONFIG["count_event_types"])

    for oid_str, tl in timelines.items():
        label = tl.get("label", "")
        if not ref_expr_ok_for_dim(label, "event_count"):
            continue
        if not object_passes_c_gate(tl):
            continue

        counts = tl.get("event_type_counts", {})
        for etype, cnt in counts.items():
            if etype not in COUNTABLE:
                continue
            if cnt < 2:
                continue
            # Bucket: 2 / 3 / 4 / 5 or more (user decision Q3 in config_v2)
            if cnt == 2:
                answer = "2"
            elif cnt == 3:
                answer = "3"
            elif cnt == 4:
                answer = "4"
            else:
                answer = "5 or more"
            units.append(make_unit(
                "event_count", video_id,
                {"obj_id": tl["obj_id"], "label": label},
                {
                    "event_type": etype,
                    "raw_count": cnt,
                    "correct_answer": answer,
                }
            ))
    return units


def build_event_ordering_units(video_id, event_data, fps):
    """T2.2 event_ordering: chronologically sort 3 or 4 events for one subject.

    Requires:
      - C-gate
      - At least 3 distinct event types on this object
      - Frame gap between events >= 0.5 sec
    Preference: if ≥4 distinct events, emit Ordering_4 (only); else Ordering_3.
    Each subject contributes at most 1 ordering unit.
    """
    units = []
    timelines = event_data.get("object_timelines", {})
    min_gap_frames = max(1, int(0.5 * fps))

    for oid_str, tl in timelines.items():
        label = tl.get("label", "")
        if not ref_expr_ok_for_dim(label, "event_ordering"):
            continue
        if not object_passes_c_gate(tl):
            continue

        # Dedup by event_type (take earliest of each type)
        own_events = [e for e in tl.get("events", [])
                      if e.get("obj_id") == tl["obj_id"]
                      and e.get("type") not in ("state_change", "interaction")]
        own_events.sort(key=event_center_frame)

        by_type_earliest = {}
        for e in own_events:
            if e["type"] not in by_type_earliest:
                by_type_earliest[e["type"]] = e

        if len(by_type_earliest) < 3:
            continue

        # Sort selected events chronologically
        selected = sorted(by_type_earliest.values(), key=event_center_frame)

        # Check frame gaps
        ok = True
        for i in range(len(selected) - 1):
            if event_center_frame(selected[i + 1]) - event_center_frame(selected[i]) < min_gap_frames:
                ok = False
                break
        if not ok:
            continue

        # Prefer 4 events if available
        if len(selected) >= 4:
            k = 4
            chosen = selected[:4]
        else:
            k = 3
            chosen = selected[:3]

        units.append(make_unit(
            "event_ordering", video_id,
            {"obj_id": tl["obj_id"], "label": label},
            {
                "k": k,
                "events": [
                    {
                        "event_id": e.get("event_id"),
                        "event_type": e["type"],
                        "frame": event_center_frame(e),
                    } for e in chosen
                ],
                # correct order is simply 0, 1, 2, (3) in chronological index
                # step3b will map this to shuffled presentation
                "correct_order_chronological_indices": list(range(k)),
            }
        ))
    return units


def build_reappear_identity_units(video_id, event_data):
    """T2.3 reappear_identity: is the reappeared X the same one or different?

    Uses the reappear_identity_candidates precomputed in step1c.
    User decision Q2 (templates): Option α only — all answers will be
    'the same one' (since tracker says same obj_id). Label bias accepted.
    """
    units = []
    candidates = event_data.get("reappear_identity_candidates", [])
    timelines = event_data.get("object_timelines", {})

    for cand in candidates:
        tgt_oid = cand["target_obj_id"]
        tgt_tl = timelines.get(str(tgt_oid), {})
        sib_tl = timelines.get(str(cand["sibling_obj_id"]), {})

        # C-gate: both target and sibling must be confident
        if not object_passes_c_gate(tgt_tl):
            continue
        if not object_passes_c_gate(sib_tl):
            continue

        units.append(make_unit(
            "reappear_identity", video_id,
            {"obj_id": tgt_oid, "label": cand["target_label"]},
            {
                "sibling_obj_id": cand["sibling_obj_id"],
                "sibling_label": cand["sibling_label"],
                "reappear_event_id": cand["reappear_event_id"],
                "correct_answer": "the same one",  # Option α
                "occlusion_duration_sec": cand["occlusion_duration_sec"],
                "sibling_covisibility": cand["sibling_covisibility"],
                # base noun is shared — included for step3b option text
                "object_class": tgt_tl.get("label_info", {}).get("base_noun", "object"),
            }
        ))
    return units


def build_occluder_identity_units(video_id, event_data):
    """T2.4 occluder_identity: which object is blocking X (SP, top-1 vs top-2)?

    Precondition:
      - Object has full_occlusion event with occluder_candidates[≥2]
      - top1.score - top2.score >= 0.3 (clear winner)
      - Neither candidate has digit suffix
      - Target passes C-gate
    """
    units = []
    timelines = event_data.get("object_timelines", {})

    for oid_str, tl in timelines.items():
        if not object_passes_c_gate(tl):
            continue

        for ev in tl.get("events", []):
            if ev.get("type") != "full_occlusion":
                continue
            if ev.get("obj_id") != tl["obj_id"]:
                continue
            occs = ev.get("occluder_candidates", [])
            if len(occs) < 2:
                continue
            top1 = occs[0]
            top2 = occs[1]
            # Digit suffix filter
            from re import search as _re_search
            if _re_search(r"#\d+$", top1.get("label", "")):
                continue
            if _re_search(r"#\d+$", top2.get("label", "")):
                continue
            # Score gap
            s1 = top1.get("overlap_score", top1.get("score", 0))
            s2 = top2.get("overlap_score", top2.get("score", 0))
            if (s1 - s2) < 0.3:
                continue

            units.append(make_unit(
                "occluder_identity", video_id,
                {"obj_id": tl["obj_id"], "label": tl.get("label", "")},
                {
                    "event_id": ev.get("event_id"),
                    "correct_occluder_label": top1["label"],
                    "distractor_occluder_label": top2["label"],
                }
            ))
    return units


# ============================================================================
# TIER 3 — O+T+C×2 cross-object dims
# ============================================================================

def build_conditional_state_units(video_id, event_data, track_data):
    """T3.1 conditional_state: when A does X, what state is B in?

    Picks the highest-priority anchor event on A (appear/reappear/...)
    then reads B's state at that frame. Each (A, B) pair contributes
    at most 1 unit (see max_qa_per_subject in config).
    """
    units = []
    anchor_priority = DIMENSIONS["conditional_state"]["anchor_event_priority"]
    max_per_subj = DIMENSIONS["conditional_state"].get("max_qa_per_subject", 1)

    timelines = event_data.get("object_timelines", {})
    pair_covis = event_data.get("pair_covisibility", [])
    tracks_by_oid = {t["obj_id"]: t for t in track_data.get("objects", [])}

    # Build a quick lookup: for each pair (a, b), what frames are covisible?
    covis_map = {}  # (a, b) -> list of frames (sorted, a<b canonical)
    for pair in pair_covis:
        a, b = pair["obj_id_a"], pair["obj_id_b"]
        covis_map[(min(a, b), max(a, b))] = pair["covisible_frames"]

    def get_baseline_area(track_b, frame_idx, window=None):
        """Rolling-median baseline area from the last `window` visible frames
        BEFORE frame_idx. Mirrors step1c's baseline computation.

        Returns None if there are no prior visible frames (e.g. object hasn't
        appeared yet).
        """
        if window is None:
            window = EVENT_CONFIG.get("baseline_window", 10)
        mask_areas = track_b.get("mask_areas", [])
        visible = track_b.get("visible", [])
        prior_visible_areas = []
        for k in range(max(0, frame_idx - window), frame_idx):
            if k < len(visible) and visible[k] and k < len(mask_areas):
                area = mask_areas[k]
                if area > 0:
                    prior_visible_areas.append(area)
        if not prior_visible_areas:
            return None
        prior_visible_areas.sort()
        return prior_visible_areas[len(prior_visible_areas) // 2]

    def get_B_state_at(track_b, frame_idx):
        """Return one of CONDITIONAL_STATE_OPTIONS based on B's track.

        Decision tree:
          - B not tracked → None (skip)
          - B invisible at frame_idx:
              - if visible both before and after → "fully hidden"
              - else → "not yet appeared or already gone"
          - B visible at frame_idx:
              - area >= 50% baseline → "fully visible"
              - area < 50% baseline → "partly hidden"
              - no baseline yet (just appeared) → "fully visible"
        """
        if not track_b:
            return None
        visible = track_b.get("visible", [])
        if frame_idx >= len(visible):
            return None

        ever_before = any(visible[:frame_idx])
        ever_after = any(visible[frame_idx:])
        if not visible[frame_idx]:
            if ever_before and ever_after:
                return "fully hidden"
            else:
                return "not yet appeared or already gone"

        # B is visible at this frame. Check occlusion via mask area vs baseline.
        mask_areas = track_b.get("mask_areas", [])
        if frame_idx >= len(mask_areas) or mask_areas[frame_idx] <= 0:
            return "fully visible"  # fall back if no area info

        current_area = mask_areas[frame_idx]
        baseline = get_baseline_area(track_b, frame_idx)
        if baseline is None or baseline <= 0:
            # No prior frames to establish baseline — treat as fully visible
            return "fully visible"

        ratio = current_area / baseline
        partial_threshold = EVENT_CONFIG.get("partial_occlusion_ratio", 0.50)
        if ratio < partial_threshold:
            return "partly hidden"
        return "fully visible"

    subjects_seen = Counter()

    for oid_str, tl_a in timelines.items():
        if subjects_seen[tl_a["obj_id"]] >= max_per_subj:
            continue
        if not object_passes_c_gate(tl_a):
            continue

        # Pick highest-priority anchor event
        a_events = {e["type"]: e for e in tl_a.get("events", [])
                    if e.get("obj_id") == tl_a["obj_id"]
                    and e.get("type") in anchor_priority}
        anchor = None
        for etype in anchor_priority:
            if etype in a_events:
                anchor = a_events[etype]
                break
        if not anchor:
            continue
        anchor_frame = event_center_frame(anchor)

        # Find a B among covisible pairs (different base noun required)
        base_a = tl_a.get("label_info", {}).get("base_noun", "")
        best_b = None
        for oid_b_str, tl_b in timelines.items():
            if tl_b["obj_id"] == tl_a["obj_id"]:
                continue
            if not object_passes_c_gate(tl_b):
                continue
            base_b = tl_b.get("label_info", {}).get("base_noun", "")
            if base_b == base_a or not base_b:
                continue
            # Check covisible: B visible at anchor_frame OR near it
            track_b = tracks_by_oid.get(tl_b["obj_id"])
            if not track_b:
                continue
            # Only consider as B if the resulting state is deterministic,
            # i.e. not in an occlusion transition (best_effort; single-frame check)
            state = get_B_state_at(track_b, anchor_frame)
            if state is None:
                continue
            best_b = (tl_b, state)
            break  # first valid B is fine

        if not best_b:
            continue
        tl_b, state = best_b

        units.append(make_unit(
            "conditional_state", video_id,
            {"obj_id": tl_a["obj_id"], "label": tl_a["label"]},
            {
                "anchor_event_type": anchor["type"],
                "anchor_event_id": anchor.get("event_id"),
                "anchor_frame": anchor_frame,
                "partner": {"obj_id": tl_b["obj_id"], "label": tl_b["label"]},
                "correct_answer": state,  # one of CONDITIONAL_STATE_OPTIONS
            }
        ))
        subjects_seen[tl_a["obj_id"]] += 1
    return units


def build_cross_object_order_units(video_id, event_data, fps):
    """T3.2 cross_object_order: does A's X happen before B's Y (SP)?

    Requires:
      - A and B different base nouns
      - Both pass C-gate
      - A has event X, B has event Y; events must differ in frame by ≥1 sec
    """
    units = []
    timelines = event_data.get("object_timelines", {})
    min_gap_frames = max(1, int(1.0 * fps))

    for oid_a_str, tl_a in timelines.items():
        if not object_passes_c_gate(tl_a):
            continue
        label_a = tl_a.get("label", "")
        if not ref_expr_ok_for_dim(label_a, "cross_object_order"):
            continue
        base_a = tl_a.get("label_info", {}).get("base_noun", "")

        events_a = [e for e in tl_a.get("events", [])
                    if e.get("obj_id") == tl_a["obj_id"]
                    and e.get("type") not in ("state_change", "interaction")]

        for oid_b_str, tl_b in timelines.items():
            if tl_b["obj_id"] <= tl_a["obj_id"]:
                continue  # canonical ordering to avoid duplicates
            if not object_passes_c_gate(tl_b):
                continue
            label_b = tl_b.get("label", "")
            if not ref_expr_ok_for_dim(label_b, "cross_object_order"):
                continue
            base_b = tl_b.get("label_info", {}).get("base_noun", "")
            if base_a and base_b and base_a == base_b:
                continue  # same base noun — disallowed

            events_b = [e for e in tl_b.get("events", [])
                        if e.get("obj_id") == tl_b["obj_id"]
                        and e.get("type") not in ("state_change", "interaction")]

            # Cross-product pairs with frame gap
            for e_a in events_a:
                for e_b in events_b:
                    fa = event_center_frame(e_a)
                    fb = event_center_frame(e_b)
                    if abs(fa - fb) < min_gap_frames:
                        continue
                    units.append(make_unit(
                        "cross_object_order", video_id,
                        {"obj_id": tl_a["obj_id"], "label": label_a},
                        {
                            "obj_a": {"obj_id": tl_a["obj_id"],
                                      "label": label_a,
                                      "event_id": e_a.get("event_id"),
                                      "event_type": e_a["type"],
                                      "frame": fa},
                            "obj_b": {"obj_id": tl_b["obj_id"],
                                      "label": label_b,
                                      "event_id": e_b.get("event_id"),
                                      "event_type": e_b["type"],
                                      "frame": fb},
                            "correct_first": ("a" if fa < fb else "b"),
                        }
                    ))
                    # Cap: 1 pair per (A, B) to avoid explosion
                    break
                else:
                    continue
                break
    return units


# ============================================================================
# Master extraction
# ============================================================================

def extract_units_for_video(video_id, event_data, track_data):
    """Run all 11 dim builders on one video and return a flat unit list."""
    fps = event_data.get("metadata", {}).get("fps", 3)
    if fps <= 0:
        fps = track_data.get("metadata", {}).get("fps", 3)

    all_units = []
    # Tier 1
    all_units.extend(build_temporal_location_units(video_id, event_data, fps))
    all_units.extend(build_duration_category_units(video_id, event_data, fps))
    all_units.extend(build_relative_spatial_change_units(video_id, event_data, track_data))
    all_units.extend(build_event_existence_units(video_id, event_data))
    all_units.extend(build_reappear_or_disappear_units(video_id, event_data))
    # Tier 2
    all_units.extend(build_event_count_units(video_id, event_data))
    all_units.extend(build_event_ordering_units(video_id, event_data, fps))
    all_units.extend(build_reappear_identity_units(video_id, event_data))
    all_units.extend(build_occluder_identity_units(video_id, event_data))
    # Tier 3
    all_units.extend(build_conditional_state_units(video_id, event_data, track_data))
    all_units.extend(build_cross_object_order_units(video_id, event_data, fps))

    return all_units


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", type=str, default=None)
    ap.add_argument("--selected-only", action="store_true",
                    help="Only process videos in FILTERED_DIR/selected_videos.json")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    UNITS_DIR.mkdir(parents=True, exist_ok=True)

    # Collect video IDs
    if args.video_id:
        video_ids = [args.video_id]
    elif args.selected_only:
        sel_path = FILTERED_DIR / "selected_videos.json"
        if not sel_path.exists():
            print("[ERROR] selected_videos.json not found. Run step2 first.")
            sys.exit(1)
        with open(sel_path) as f:
            data = json.load(f)
        video_ids = [v["video_id"] for v in data.get("videos", [])]
    else:
        video_ids = sorted([f.stem for f in EVENTS_DIR.glob("*.json")
                            if not f.name.startswith("_")])

    agg_dim_counts = Counter()
    agg_tier_counts = Counter()
    agg_videos_with_units = 0
    agg_units_total = 0

    # Hallucination resources accumulators
    # per_video_existing: vid -> {"base_nouns": set, "full_labels": set}
    per_video_existing = {}
    # Global: base_noun -> set of modifier_prefixes ("red", "larger white", ...)
    global_modifier_pool = defaultdict(set)
    # Global: all full labels seen (for 20% cross-category hallucinations)
    global_label_pool = set()
    # Global: all base_nouns seen
    global_base_nouns = set()

    for vid in tqdm(video_ids):
        ef = EVENTS_DIR / f"{vid}.json"
        tf = TRACKS_DIR / f"{vid}.json"
        if not (ef.exists() and tf.exists()):
            continue

        # Load event_data unconditionally — we always need it for hallucination stats,
        # even if units.json already exists
        with open(ef) as f:
            event_data = json.load(f)

        # Accumulate hallucination resources from this video's labels
        v_base = set()
        v_full = set()
        for oid_str, tl in event_data.get("object_timelines", {}).items():
            label = tl.get("label", "")
            info = tl.get("label_info", {})
            base = info.get("base_noun", "")
            modifiers = info.get("modifiers_text", "")
            if label:
                v_full.add(label)
                global_label_pool.add(label)
            if base:
                v_base.add(base)
                global_base_nouns.add(base)
                if modifiers:
                    global_modifier_pool[base].add(modifiers)
        per_video_existing[vid] = {
            "base_nouns": sorted(v_base),
            "full_labels": sorted(v_full),
        }

        out = UNITS_DIR / f"{vid}.json"
        if out.exists() and not args.overwrite:
            # load existing to count stats
            with open(out) as f:
                existing = json.load(f)
            for u in existing.get("units", []):
                agg_dim_counts[u["dim"]] += 1
            agg_units_total += existing.get("num_units", 0)
            continue

        with open(tf) as f:
            track_data = json.load(f)

        units = extract_units_for_video(vid, event_data, track_data)
        for u in units:
            agg_dim_counts[u["dim"]] += 1
            tier = DIMENSIONS.get(u["dim"], {}).get("tier", "unknown")
            agg_tier_counts[tier] += 1
        if units:
            agg_videos_with_units += 1
        agg_units_total += len(units)

        out_data = {
            "video_id": vid,
            "num_units": len(units),
            "units": units,
        }
        with open(out, "w") as f:
            json.dump(out_data, f, indent=2)

    # Write summary
    summary = {
        "total_units": agg_units_total,
        "videos_with_units": agg_videos_with_units,
        "dim_counts": dict(agg_dim_counts),
        "tier_counts": dict(agg_tier_counts),
    }
    with open(UNITS_DIR / "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Write hallucination resources
    hallu_resources = {
        "per_video_existing": per_video_existing,
        "global_modifier_pool_by_base": {
            base: sorted(mods) for base, mods in global_modifier_pool.items()
        },
        "global_label_pool": sorted(global_label_pool),
        "global_base_nouns": sorted(global_base_nouns),
        "_stats": {
            "num_videos": len(per_video_existing),
            "num_distinct_labels": len(global_label_pool),
            "num_distinct_base_nouns": len(global_base_nouns),
            "avg_modifiers_per_base": (
                sum(len(m) for m in global_modifier_pool.values())
                / max(1, len(global_modifier_pool))
            ),
        },
    }
    with open(UNITS_DIR / "_hallucination_resources.json", "w") as f:
        json.dump(hallu_resources, f, indent=2)

    print("\n" + "=" * 60)
    print("  Unit Extraction Summary (v2, 11 dims)")
    print("=" * 60)
    print(f"  Total units:         {agg_units_total:,}")
    print(f"  Videos with units:   {agg_videos_with_units:,}")
    print("\n  By dim:")
    for dim in DIMENSIONS:
        cnt = agg_dim_counts.get(dim, 0)
        print(f"    {dim:<30s} {cnt:>6d}")
    print("\n  By tier:")
    for tier in ["tier1", "tier2", "tier3"]:
        print(f"    {tier:<10s} {agg_tier_counts.get(tier, 0):>6d}")

    print("\n  Hallucination resources:")
    print(f"    videos:                 {hallu_resources['_stats']['num_videos']:,}")
    print(f"    distinct labels:        {hallu_resources['_stats']['num_distinct_labels']:,}")
    print(f"    distinct base nouns:    {hallu_resources['_stats']['num_distinct_base_nouns']:,}")
    print(f"    avg modifiers per base: {hallu_resources['_stats']['avg_modifiers_per_base']:.1f}")
    print(f"    → {UNITS_DIR / '_hallucination_resources.json'}")


if __name__ == "__main__":
    main()