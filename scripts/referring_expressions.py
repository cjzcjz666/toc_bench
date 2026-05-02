#!/usr/bin/env python3
"""
TOC-Bench: Referring Expression Construction
=============================================
Builds unique, natural referring expressions for tracked objects
using only tracking data (no VLM needed).

Available disambiguation features (from tracking data):
  - Spatial position:  leftmost / rightmost / upper / lower / center
  - Relative size:     larger / smaller
  - Temporal identity: first to appear / last to disappear / appears later
  - Trajectory:        the one that reappears / the one that exits

For each object, we try to find a minimal set of features that uniquely
identifies it among all objects sharing the same base label.

If no unique description can be constructed, the object is marked with
ref_expr_confidence = 0, and downstream code should skip it for QA.

Usage:
    from referring_expressions import build_ref_exprs_for_video

    ref_map = build_ref_exprs_for_video(track_data, event_data)
    # ref_map[obj_id] = {
    #     "raw_label": "book #2",
    #     "base_label": "book",
    #     "canonical_ref_expr": "the book on the left",
    #     "ref_expr_confidence": 0.92,
    #     "disambiguation_features": ["spatial:left"],
    # }
"""

import re
from collections import defaultdict


# ============================================================
# Feature Extractors (from tracking data)
# ============================================================

def _extract_spatial_features(track, img_width, img_height):
    """
    Compute the object's dominant spatial position from its bbox history.
    Returns a dict of spatial features.
    """
    visible_bboxes = [
        bbox for bbox, vis in zip(track["bboxes"], track["visible"])
        if vis and bbox != [0, 0, 0, 0]
    ]
    if not visible_bboxes:
        return {}

    # Median center position
    cxs = [(b[0] + b[2]) / 2 for b in visible_bboxes]
    cys = [(b[1] + b[3]) / 2 for b in visible_bboxes]
    cx = sorted(cxs)[len(cxs) // 2]
    cy = sorted(cys)[len(cys) // 2]

    features = {}

    # Horizontal position
    if img_width > 0:
        rx = cx / img_width
        if rx < 0.33:
            features["spatial_h"] = "left"
        elif rx > 0.67:
            features["spatial_h"] = "right"
        else:
            features["spatial_h"] = "center"

    # Vertical position
    if img_height > 0:
        ry = cy / img_height
        if ry < 0.33:
            features["spatial_v"] = "upper"
        elif ry > 0.67:
            features["spatial_v"] = "lower"
        else:
            features["spatial_v"] = "middle"

    return features


def _extract_size_features(track):
    """
    Compute relative size from median visible mask area.
    Returns {"median_area": float}.
    """
    visible_areas = [
        a for a, vis in zip(track["mask_areas"], track["visible"])
        if vis and a > 0
    ]
    if not visible_areas:
        return {"median_area": 0}
    return {"median_area": sorted(visible_areas)[len(visible_areas) // 2]}


def _extract_temporal_features(track):
    """
    Extract temporal identity features: when does the object first/last appear.
    """
    n = len(track["visible"])
    first_vis = next((i for i in range(n) if track["visible"][i]), None)
    last_vis = next(
        (n - 1 - i for i in range(n) if track["visible"][n - 1 - i]), None
    )

    return {
        "first_visible_frame": first_vis,
        "last_visible_frame": last_vis,
        "visibility_ratio": (
            sum(1 for v in track["visible"] if v) / max(1, n)
        ),
    }


def _extract_event_features(obj_id, event_data):
    """
    Extract trajectory features from the event timeline.
    """
    timelines = event_data.get("object_timelines", {})
    tl = timelines.get(str(obj_id), timelines.get(obj_id, {}))
    event_types = set(tl.get("event_type_set", []))

    features = {}
    if "reappear" in event_types:
        features["has_reappear"] = True
    if "disappear" in event_types:
        features["has_disappear"] = True
    if "exit_frame" in event_types:
        features["has_exit"] = True
    if "enter_frame" in event_types:
        features["has_enter"] = True
    if "full_occlusion" in event_types:
        features["has_occlusion"] = True
    features["num_events"] = tl.get("num_events", 0)

    return features


# ============================================================
# Base Label Extraction
# ============================================================

def _strip_suffix(label):
    """
    Remove #N suffix from VLM-generated labels.
    "book #2" → "book", "white plate #1" → "white plate"
    """
    return re.sub(r'\s*#\d+$', '', label).strip()


def _find_duplicate_groups(tracks):
    """
    Group objects by their base label (without #N suffix).
    Returns dict: base_label → [track, track, ...]
    Only includes groups with ≥2 objects (the ambiguous ones).
    """
    groups = defaultdict(list)
    for t in tracks:
        base = _strip_suffix(t["label"])
        groups[base].append(t)

    # Only return groups with duplicates
    return {k: v for k, v in groups.items() if len(v) >= 2}


# ============================================================
# Disambiguation Logic
# ============================================================

def _build_ref_expr(track, feature_set, group_features, base_label,
                    img_width, img_height):
    """
    Try to build a unique referring expression for one object
    within its duplicate group.

    Strategy: try features in priority order, pick the first one
    that uniquely identifies this object.

    Priority:
      1. Spatial (left/right) — most natural and stable
      2. Size (larger/smaller) — needs clear difference
      3. Temporal (first to appear / appears later)
      4. Trajectory (reappears / exits / gets occluded)
      5. Compound (spatial + size or spatial + temporal)
    """
    oid = track["obj_id"]
    my_feat = feature_set[oid]
    other_feats = {
        k: v for k, v in feature_set.items() if k != oid
    }

    candidates = []

    # --- 1. Spatial horizontal ---
    my_h = my_feat.get("spatial_h")
    if my_h and my_h != "center":
        others_h = {of.get("spatial_h") for of in other_feats.values()}
        if my_h not in others_h:
            expr = f"the {base_label} on the {my_h}"
            candidates.append((expr, 0.90, ["spatial:" + my_h]))

    # Leftmost / rightmost (even among objects in the same horizontal third)
    if my_feat.get("spatial_h"):
        all_cxs = {}
        for k, f in feature_set.items():
            if "median_cx" in f:
                all_cxs[k] = f["median_cx"]
        if len(all_cxs) >= 2:
            sorted_by_x = sorted(all_cxs.items(), key=lambda x: x[1])
            # Require meaningful gap: at least 5% of image width
            min_gap = img_width * 0.05 if img_width > 0 else 30
            leftmost_id, leftmost_cx = sorted_by_x[0]
            rightmost_id, rightmost_cx = sorted_by_x[-1]
            if rightmost_cx - leftmost_cx >= min_gap:
                if leftmost_id == oid:
                    expr = f"the leftmost {base_label}"
                    candidates.append((expr, 0.88, ["spatial:leftmost"]))
                if rightmost_id == oid:
                    expr = f"the rightmost {base_label}"
                    candidates.append((expr, 0.88, ["spatial:rightmost"]))

    # --- 2. Size ---
    my_area = my_feat.get("median_area", 0)
    if my_area > 0:
        other_areas = [of.get("median_area", 0) for of in other_feats.values()]
        if other_areas:
            max_other = max(other_areas)
            min_other = min(other_areas)
            # Need ≥50% difference to be confident
            if my_area > max_other * 1.5:
                expr = f"the larger {base_label}"
                candidates.append((expr, 0.82, ["size:larger"]))
            elif my_area < min_other * 0.67:
                expr = f"the smaller {base_label}"
                candidates.append((expr, 0.82, ["size:smaller"]))

    # --- 3. Temporal identity ---
    my_first = my_feat.get("first_visible_frame")
    if my_first is not None:
        other_firsts = [
            of.get("first_visible_frame")
            for of in other_feats.values()
            if of.get("first_visible_frame") is not None
        ]
        if other_firsts:
            # Appears earliest
            if my_first < min(other_firsts):
                gap = min(other_firsts) - my_first
                if gap >= 3:  # at least 3 frames difference
                    expr = f"the {base_label} that appears first"
                    candidates.append((expr, 0.85, ["temporal:appears_first"]))
            # Appears latest
            elif my_first > max(other_firsts):
                gap = my_first - max(other_firsts)
                if gap >= 3:
                    expr = f"the {base_label} that appears later"
                    candidates.append((expr, 0.83, ["temporal:appears_later"]))

    # --- 4. Trajectory ---
    if my_feat.get("has_reappear"):
        others_reappear = any(of.get("has_reappear") for of in other_feats.values())
        if not others_reappear:
            expr = f"the {base_label} that reappears after being hidden"
            candidates.append((expr, 0.87, ["trajectory:reappears"]))

    if my_feat.get("has_exit"):
        others_exit = any(of.get("has_exit") for of in other_feats.values())
        if not others_exit:
            expr = f"the {base_label} that exits the frame"
            candidates.append((expr, 0.85, ["trajectory:exits"]))

    if my_feat.get("has_occlusion"):
        others_occ = any(of.get("has_occlusion") for of in other_feats.values())
        if not others_occ:
            expr = f"the {base_label} that gets occluded"
            candidates.append((expr, 0.86, ["trajectory:occluded"]))

    if my_feat.get("has_enter"):
        others_enter = any(of.get("has_enter") for of in other_feats.values())
        if not others_enter:
            expr = f"the {base_label} that enters the frame"
            candidates.append((expr, 0.84, ["trajectory:enters"]))

    # --- 5. Compound: spatial + temporal ---
    if not candidates:
        my_h = my_feat.get("spatial_h", "")
        my_first = my_feat.get("first_visible_frame")
        if my_h and my_first is not None:
            other_same_h = [
                of for of in other_feats.values()
                if of.get("spatial_h") == my_h
            ]
            if not other_same_h:
                expr = f"the {base_label} on the {my_h}"
                candidates.append((expr, 0.80, ["spatial:" + my_h]))
            else:
                # Same horizontal zone but different temporal
                other_firsts_same_h = [
                    of.get("first_visible_frame")
                    for of in other_same_h
                    if of.get("first_visible_frame") is not None
                ]
                if other_firsts_same_h and my_first < min(other_firsts_same_h):
                    expr = (f"the {base_label} on the {my_h} "
                            f"that appears first")
                    candidates.append((expr, 0.75, [
                        "spatial:" + my_h, "temporal:appears_first"
                    ]))

    # Pick best candidate
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0]

    return None


def _compute_median_cx(track):
    """Compute median horizontal center for spatial comparison."""
    visible_bboxes = [
        bbox for bbox, vis in zip(track["bboxes"], track["visible"])
        if vis and bbox != [0, 0, 0, 0]
    ]
    if not visible_bboxes:
        return None
    cxs = [(b[0] + b[2]) / 2 for b in visible_bboxes]
    return sorted(cxs)[len(cxs) // 2]


# ============================================================
# Main Entry Point
# ============================================================

def build_ref_exprs_for_video(track_data, event_data):
    """
    Build referring expressions for all objects in a video.

    For objects with unique base labels (no #N suffix, no duplicates),
    the raw label is used directly with high confidence.

    For objects in duplicate groups (book #1, book #2, ...),
    disambiguation features are used to construct unique descriptions.

    Returns:
        dict: obj_id → {
            "raw_label": str,
            "base_label": str,
            "canonical_ref_expr": str,
            "ref_expr_confidence": float,
            "disambiguation_features": list,
            "is_duplicate_group": bool,
        }
    """
    tracks = track_data.get("objects", [])
    metadata = track_data.get("metadata", {})
    img_w = metadata.get("width", 640)
    img_h = metadata.get("height", 480)

    result = {}

    # Find duplicate groups
    dup_groups = _find_duplicate_groups(tracks)
    dup_base_labels = set(dup_groups.keys())

    for t in tracks:
        oid = t["obj_id"]
        raw_label = t["label"]
        base_label = _strip_suffix(raw_label)

        if base_label not in dup_base_labels:
            # Unique label — use as-is
            result[oid] = {
                "raw_label": raw_label,
                "base_label": base_label,
                "canonical_ref_expr": f"the {base_label}",
                "ref_expr_confidence": 0.95,
                "disambiguation_features": [],
                "is_duplicate_group": False,
            }
        # Duplicates handled below

    # Handle duplicate groups
    for base_label, group_tracks in dup_groups.items():
        # Extract features for all objects in the group
        feature_set = {}
        for t in group_tracks:
            oid = t["obj_id"]
            feats = {}
            feats.update(_extract_spatial_features(t, img_w, img_h))
            feats.update(_extract_size_features(t))
            feats.update(_extract_temporal_features(t))
            feats.update(_extract_event_features(oid, event_data))
            # Add median_cx for leftmost/rightmost comparison
            mcx = _compute_median_cx(t)
            if mcx is not None:
                feats["median_cx"] = mcx
            feature_set[oid] = feats

        # Try to build unique ref expr for each member
        for t in group_tracks:
            oid = t["obj_id"]
            disambiguation = _build_ref_expr(
                t, feature_set, feature_set, base_label, img_w, img_h
            )

            if disambiguation is not None:
                expr, confidence, features = disambiguation
                result[oid] = {
                    "raw_label": t["label"],
                    "base_label": base_label,
                    "canonical_ref_expr": expr,
                    "ref_expr_confidence": confidence,
                    "disambiguation_features": features,
                    "is_duplicate_group": True,
                }
            else:
                # Cannot disambiguate — mark as non-questionable
                result[oid] = {
                    "raw_label": t["label"],
                    "base_label": base_label,
                    "canonical_ref_expr": t["label"],  # keep raw as fallback
                    "ref_expr_confidence": 0.0,
                    "disambiguation_features": [],
                    "is_duplicate_group": True,
                }

    return result


def filter_by_ref_confidence(units, ref_map, min_confidence=0.5):
    """
    Remove reasoning units whose subject(s) cannot be uniquely referred to.

    Args:
        units: list of reasoning unit dicts from step3a
        ref_map: dict from build_ref_exprs_for_video
        min_confidence: minimum ref_expr_confidence to keep

    Returns:
        (kept_units, dropped_count)
    """
    kept = []
    dropped = 0

    for unit in units:
        # Check all subjects
        subjects = []
        if "subject" in unit:
            subjects.append(unit["subject"])
        if "subjects" in unit:
            subjects.extend(unit["subjects"])

        all_ok = True
        for subj in subjects:
            oid = subj.get("obj_id")
            if oid is None:
                continue
            ref = ref_map.get(oid)
            if ref is None:
                continue
            if ref["ref_expr_confidence"] < min_confidence:
                all_ok = False
                break

        if all_ok:
            # Replace raw labels with canonical ref expressions
            for subj in subjects:
                oid = subj.get("obj_id")
                if oid is not None and oid in ref_map:
                    ref = ref_map[oid]
                    subj["label"] = ref["canonical_ref_expr"]
                    subj["raw_label"] = ref["raw_label"]
                    subj["ref_expr_confidence"] = ref["ref_expr_confidence"]
            kept.append(unit)
        else:
            dropped += 1

    return kept, dropped