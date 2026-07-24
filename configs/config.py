"""
TOC-Bench Configuration v2
=============================================================================
All paths, parameters, dim/format/bucket definitions, and data source specs.

Changes from v1:
  - NEW: 11 candidate dimension definitions; 10 retained in the released benchmark
  - NEW: 4 conceptual task formats (added Numerical; ordering has 3/4-event serializations)
  - NEW: 3-bucket temporal / duration / spatial-direction systems
  - NEW: SAM3 post-processing config (方案 C: ID repair + instance_confidence)
  - NEW: Per-dim precondition and answer-balancing rules
  - NEW: step3c do-not-paraphrase lockdown list
  - REMOVED: EVENT_CONFUSIONS table (each dim now has its own distractor rule)
  - REMOVED: old 16-dim schema in QA_CONFIG
"""

import os
from pathlib import Path

# =============================================================================
# 1. Directory Layout (unchanged from v1)
# =============================================================================
ROOT = Path(os.environ.get("TOC_BENCH_ROOT", "benchdata"))

VIDEOS_DIR       = ROOT / "videos"
FRAMES_DIR       = ROOT / "frames"
TRACKS_DIR       = ROOT / "tracks"
EVENTS_DIR       = ROOT / "events"
FILTERED_DIR     = ROOT / "filtered"
QA_DIR           = ROOT / "qa"
STRESS_DIR       = ROOT / "stress_videos"
EVAL_DIR         = ROOT / "eval"

for d in [VIDEOS_DIR, FRAMES_DIR, TRACKS_DIR, EVENTS_DIR,
          FILTERED_DIR, QA_DIR, STRESS_DIR, EVAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. Data Sources (unchanged from v1)
# =============================================================================
DATA_SOURCES = {
    "perception_test": {
        "description": "DeepMind Perception Test validation set – indoor hand-object interaction",
        "target_count": 400,
        "video_subdir": VIDEOS_DIR / "perception_test",
        "download_url": "https://storage.googleapis.com/dm-perception-test/zip_data/valid_videos.zip",
        "annotation_url": "https://storage.googleapis.com/dm-perception-test/zip_data/valid_annotations.zip",
        "auto_download": True,
        "video_ext": ".mp4",
        "license": "CC-BY-4.0",
    },
    "ovis": {
        "description": "OVIS – Occluded Video Instance Segmentation, outdoor crowded scenes",
        "target_count": 200,
        "video_subdir": VIDEOS_DIR / "ovis",
        "download_url": None,
        "instructions": (
            "1. Visit https://songbai.site/ovis/ and request access.\n"
            "2. Download the validation split frames.\n"
            "3. OVIS stores frames as JPEG sequences in per-video folders.\n"
            "   Place each video folder into: {video_subdir}\n"
            "   e.g. {video_subdir}/video_0601/00001.jpg\n"
        ),
        "auto_download": False,
        "video_ext": ".jpg_folder",
        "license": "Research use",
    },
    "mose": {
        "description": "MOSE – Complex VOS with disappearance, reappearance, small objects",
        "target_count": 200,
        "video_subdir": VIDEOS_DIR / "mose",
        "download_url": None,
        "instructions": (
            "1. Visit https://henghuiding.github.io/MOSE/ and request access.\n"
            "2. Download the validation split.\n"
            "3. MOSE stores frames as JPEG sequences in per-video folders.\n"
            "   Place each video folder into: {video_subdir}\n"
            "   The tracking script will auto-detect JPEG-folder videos.\n"
        ),
        "auto_download": False,
        "video_ext": ".jpg_folder",
        "license": "Research use",
    },
    "charades": {
        "description": "Charades-STA – daily household activities, multi-step temporal reasoning",
        "target_count": 150,
        "video_subdir": VIDEOS_DIR / "charades",
        "download_url": None,
        "instructions": (
            "1. Visit https://prior.allenai.org/projects/charades\n"
            "2. Download Charades videos (scaled to 480p recommended).\n"
            "3. Place .mp4 files into: {video_subdir}\n"
        ),
        "auto_download": False,
        "video_ext": ".mp4",
        "license": "Research use",
    },
    "star": {
        "description": "STAR – Situated Temporal Reasoning in real-world videos (shares Charades videos)",
        "target_count": 150,
        "video_subdir": VIDEOS_DIR / "charades",
        "download_url": None,
        "instructions": (
            "No separate download needed — STAR uses Charades videos.\n"
            "Just make sure Charades videos are in: {video_subdir}\n"
        ),
        "auto_download": False,
        "video_ext": ".mp4",
        "share_source": "charades",
        "license": "Research use",
    },
}


# =============================================================================
# 3. VLM-Guided Prompt Generation (Qwen3-VL-8B) — unchanged from v1
# =============================================================================
VLM_CONFIG = {
    "model_name": "/root/autodl-tmp/Qwen3-VL-8B-Instruct",
    "device": "cuda:0",
    "vlm_fps": 1.0,
    "vlm_max_frames": 10,
    "max_prompts_per_video": 15,
    "fallback_prompts": [
        "person", "hand", "cup", "bottle", "ball",
        "box", "bag", "animal", "vehicle", "food",
    ],
    "system_prompt": (
        "You are a visual object detector. Given video frames, list ALL distinct "
        "physical objects and people visible. Return ONLY a JSON list of short "
        "English noun phrases (2-5 words each). Rules:\n"
        "- Objects: be specific with color/material when distinctive, e.g. "
        "'red cup', 'wooden spoon', 'glass bottle'.\n"
        "- People: NEVER use generic 'person'. Instead, describe each person "
        "with distinguishing attributes: age, gender, clothing, or action. "
        "E.g. 'woman in red shirt', 'old man with hat', 'boy kicking ball', "
        "'running girl', 'man in black jacket'. If multiple similar people, "
        "differentiate them by clothing or position.\n"
        "- Body parts: use 'hand', 'arm', 'finger' ONLY when just that body "
        "part is visible without the rest of the body (e.g. a hand reaching "
        "into frame).\n"
        "- Do NOT include background elements like 'wall', 'floor', 'sky', "
        "'table surface', 'room'.\n"
        "Example output:\n"
        '["woman in red shirt", "boy in blue shorts", "red cup", '
        '"wooden spoon", "white plate", "glass bottle"]'
    ),
}


# =============================================================================
# 4. SAM 3.1 Tracking Parameters (v2: added post-processing config)
# =============================================================================
SAM3_CONFIG = {
    "confidence_threshold": 0.3,
    "min_mask_area": 100,
    "max_objects_per_video": 20,
    "device": "cuda:0",
    "batch_size": 1,
    "fps_for_tracking": 3,
}

# -----------------------------------------------------------------------------
# NEW in v2: 方案 C — SAM3 post-processing for instance stability
# -----------------------------------------------------------------------------
# After SAM3 tracking produces raw obj_id assignments, we run a post-processing
# pass that (a) detects ID splits/swaps, (b) merges spurious short fragments,
# and (c) assigns each obj_id an `instance_confidence` score [0, 1].
#
# Tier 2/3 C-dim templates use this confidence to gate question generation:
# only obj_ids with confidence >= threshold can serve as subjects in C-dim QAs.
# -----------------------------------------------------------------------------
SAM3_POSTPROC_CONFIG = {
    # --- Fragment cleanup ---
    "min_fragment_frames": 5,
    # Drop any obj_id visible for <5 frames total (likely tracker noise)

    "min_fragment_duration_ratio": 0.02,
    # Or <2% of video length, whichever is larger

    # --- ID split repair ---
    # If two obj_ids of the SAME label have non-overlapping time windows AND
    # their bbox trajectories are spatially continuous, merge into one.
    "split_repair_max_time_gap_sec": 2.0,
    # Time gap between the end of one fragment and start of next, max 2 seconds

    "split_repair_max_spatial_jump_ratio": 0.15,
    # Last bbox center of fragment A to first of fragment B must be
    # ≤15% of image diagonal

    "split_repair_require_same_label": True,
    # Only merge fragments that share the same raw label (do NOT cross labels)

    # --- ID swap detection ---
    # Within a single obj_id, detect sudden position jumps that suggest
    # the ID was reassigned to a different physical object.
    "swap_detect_position_jump_ratio": 0.35,
    # Sudden frame-to-frame bbox center jump >35% of diagonal = suspicious
    # (after excluding cases where object was invisible in between)

    "swap_detect_min_visible_gap": 3,
    # Position jumps are only flagged if the object was visible in both
    # the before-jump and after-jump frames (within min_visible_gap=3 frames
    # of each other). Jumps across occlusion periods are not flagged.

    # --- Instance confidence scoring ---
    # Final instance_confidence per obj_id is computed as:
    #   confidence = base_score * split_penalty * swap_penalty * coverage_bonus
    # where:
    #   base_score    = mean SAM3 confidence across visible frames [0, 1]
    #   split_penalty = 1.0 - 0.15 * num_merge_operations_applied
    #   swap_penalty  = 1.0 - 0.25 * num_unresolved_swaps_detected
    #   coverage_bonus= 1.0 if visible_ratio >= 0.3, else 0.8

    # --- C-dim gate threshold ---
    "c_dim_min_confidence": 0.7,
    # Tier 2/3 templates filter out obj_ids below this confidence

    # --- Logging ---
    "save_repair_log": True,
    # Save per-video repair operations to tracks_dir/repair_log/<video_id>.json
    # Useful for debugging and for reporting stats in the paper
}


# =============================================================================
# 5. Event Detection Parameters (unchanged from v1)
# =============================================================================
EVENT_CONFIG = {
    "baseline_window": 10,
    "partial_occlusion_ratio": 0.50,
    "min_partial_frames": 2,
    "full_occlusion_ratio": 0.10,
    "min_full_occlusion_frames": 2,
    "min_reappear_frames": 2,
    "border_margin": 0.05,
    "state_area_change_ratio": 0.8,
    "state_position_change_ratio": 0.4,
    "min_state_change_gap": 5,
    "interaction_iou_threshold": 0.05,
    "min_interaction_frames": 3,
    "interaction_max_duration_ratio": 0.8,
}


# -----------------------------------------------------------------------------
# NEW in v2: per-video statistics computed after event detection
# Used by step1c to emit additional fields for downstream QA generation
# -----------------------------------------------------------------------------
EVENT_STATS_CONFIG = {
    # For each object, count events per type — feeds T2.1 event_count
    "count_event_types": [
        "full_occlusion", "partial_occlusion", "reappear", "state_change",
    ],

    # Detect same-base-class sibling objects for T2.3 reappear_identity
    # (two objects sharing base noun but with distinct modifiers)
    "sibling_detection": {
        "require_same_base_noun": True,
        "min_covisibility_ratio_during_target_occlusion": 0.5,
        # Sibling must be visible during ≥50% of target's occlusion period

        "min_reappear_position_distance_ratio": 0.10,
        # Target's reappear position must be ≥10% diagonal from any sibling

        "min_occlusion_duration_sec": 1.0,
        # Target's occlusion must last ≥1 second for identity question
        # to be meaningful
    },

    # Co-visibility matrix: for each pair of obj_ids, track which frames
    # both are visible — feeds T3.1 conditional_state
    "compute_pair_covisibility": True,
}


# =============================================================================
# 6. Video Filtering Criteria (unchanged from v1)
# =============================================================================
FILTER_CONFIG = {
    "min_tracked_objects": 2,
    "min_total_events": 3,
    "min_interactions": 2,
    "min_video_duration_sec": 5,
    "max_video_duration_sec": 90,
}


# =============================================================================
# 7. QA Dimensions — v2 schema (REPLACES old hard_negative_types etc.)
# =============================================================================
#
# 11 candidate dimension definitions organized into tiers by what they test.
# The released benchmark retains 10; occluder_identity is construction-only.
#
#   Tier 1 (O+T):      baseline temporal reasoning about a single object
#   Tier 2 (O+T+C):    requires tracking the same object identity over time
#   Tier 3 (O+T+C×2):  requires tracking TWO objects and aligning events
#
#   SP-only:           dims that are naturally binary (yes/no, 2-way choice)
#
# Each dim declares:
#   tier           — "tier1", "tier2", "tier3"
#   format         — which of the 4 conceptual task formats to use
#   target_ratio   — target fraction of total QA (roughly 1/3 per tier)
#   c_critical     — if True, requires instance_confidence >= 0.7
#   notes          — design rationale for future reference
# =============================================================================

DIMENSIONS = {
    # ---------- Tier 1: O+T, baseline layer ----------
    "temporal_location": {
        "tier": "tier1",
        "format": "mcq_4",
        "target_ratio": 0.13,
        "c_critical": False,
        "allow_multi_event": "first_last_prefix",
        # Allows multi-event via "first X" / "last X" prefix; skip if ≥3 events
        "ref_expr_forbidden_modifiers": ["temporal"],
        # ref expr must not contain "first", "earlier", "later", etc.
        "notes": "When in the video does X Y? — asks which time-bucket (4 buckets)."
    },

    "duration_category": {
        "tier": "tier1",
        "format": "mcq_4",
        "target_ratio": 0.03,
        # Low target because this dim will be hit hard by step4 shuffled layer
        # User decision: 不豁免 shuffled. Accept low survival rate.
        "c_critical": False,
        "allow_multi_event": "bucket_consistent_only",
        # Multi-event allowed if all instances fall in the same duration bucket
        "notes": "How long does X stay in state Y? — 4-bucket duration (v1-style)."
    },

    "relative_spatial_change": {
        "tier": "tier1",
        "format": "mcq_4",
        "target_ratio": 0.13,
        "c_critical": False,
        "ref_expr_forbidden_modifiers": ["spatial"],
        # ref expr must not contain "leftmost", "upper", "on the left", etc.
        "notes": "How does X move across the frame? — 4 direction options + 1 stayed."
    },

    # ---------- Tier 2: O+T+C, core layer ----------
    "event_count": {
        "tier": "tier2",
        "format": "numerical",
        "target_ratio": 0.10,
        "c_critical": True,
        "ref_expr_forbidden_modifiers": ["temporal", "count"],
        # No "first", no "twice", etc.
        "notes": "How many times does X Y? — THE purest C test."
    },

    "event_ordering": {
        "tier": "tier2",
        "format": "ordering",
        "target_ratio": 0.12,
        "c_critical": True,
        "ref_expr_forbidden_modifiers": ["temporal"],
        "notes": "Sort events for X chronologically."
    },

    "reappear_identity": {
        "tier": "tier2",
        "format": "sp",
        "target_ratio": 0.08,
        "c_critical": True,
        "requires_sibling": True,
        # Requires at least one same-base-noun sibling per precondition in
        # EVENT_STATS_CONFIG["sibling_detection"]
        "notes": "Is the reappeared X the same one as before, or different?"
    },

    "occluder_identity": {
        "tier": "tier2",
        "format": "sp",
        "target_ratio": 0.10,
        "c_critical": True,
        "notes": "Which object is blocking X? — top-1 vs top-2 candidate."
    },

    # ---------- Tier 3: O+T+C×2, cross-object alignment ----------
    "conditional_state": {
        "tier": "tier3",
        "format": "mcq_4",
        "target_ratio": 0.13,
        "c_critical": True,
        "anchor_event_priority": [
            "appear", "reappear", "disappear", "exit_frame",
            "enter_frame", "full_occlusion",
        ],
        # Preference order for choosing the anchor event when object A
        # has multiple candidate events. partial_occlusion, interaction,
        # and state_change are NOT allowed as anchors.
        "max_qa_per_subject": 1,
        # Each subject A contributes at most 1 T3.1 question (to prevent
        # test-set homogeneity)
        "notes": "When A does X, what state is B in? — cross-object sync."
    },

    "cross_object_order": {
        "tier": "tier3",
        "format": "sp",
        "target_ratio": 0.12,
        "c_critical": True,
        "ref_expr_forbidden_modifiers": ["temporal"],
        "require_different_base_noun": True,
        # Subjects A and B must not share the same base noun
        # (avoids "the earlier cup appears before the later cup"-type ambiguity)
        "notes": "Does A's X happen before B's Y?"
    },

    # ---------- SP-only: naturally binary ----------
    "event_existence": {
        "tier": "tier1",
        "format": "sp",
        "target_ratio": 0.03,
        "c_critical": False,
        "notes": "Does X ever Y? — binary by design."
    },

    "reappear_or_disappear": {
        "tier": "tier2",
        "format": "sp",
        "target_ratio": 0.03,
        "c_critical": False,
        # Weakly touches C but not strongly required; leave in Tier 2 for
        # now, reclassify if analysis shows otherwise
        "notes": "Mechanism: fully blocked vs left the frame."
    },
}


# =============================================================================
# 8. Formats (NEW in v2) — declares what each format expects
# =============================================================================
FORMATS = {
    "mcq_4": {
        "description": "4-way multiple choice",
        "options_count": 4,
        "option_labels": ["A", "B", "C", "D"],
        "output_example": "C",
    },
    "sp": {
        "description": "Statement pair: choose which is supported",
        "options_count": 2,
        "option_labels": ["A", "B"],
        "output_example": "A",
    },
    "ordering": {
        "description": "Sort 3-4 events chronologically",
        "options_count": None,  # variable 3 or 4
        "option_labels": ["A", "B", "C", "D"],
        "output_example": "B,A,C",
        "variants": ["ordering_3", "ordering_4"],
    },
    "numerical": {
        "description": "Open numerical answer",
        "options_count": None,
        "output_example": "3",
        "allowed_values": ["2", "3", "4", "5 or more"],
        # User decision Q3: 扩到 4 档（2 / 3 / 4 / 5 or more）
        # Strict matching against these string values
        # (user decision: 严格匹配, report single accuracy)
    },
}


# =============================================================================
# 9. Buckets (NEW in v2) — replaces old TIME_BUCKETS, DURATION_BUCKETS
# =============================================================================

# Temporal buckets — used by temporal_location
# 4 buckets均分 (user decision Q1: 方案 ①)
# Event's center-frame ratio to total frames determines the bucket.
TIME_BUCKETS = {
    "beginning":  (0.00, 0.25),
    "early":      (0.25, 0.50),
    "late":       (0.50, 0.75),
    "end":        (0.75, 1.00),
}

# Duration buckets — used by duration_category
# 4 buckets (user decision Q2: v1 风格, 接受同义化风险)
# ⚠️ Known issue: "very briefly" vs "briefly" are near-synonyms in English.
# User has accepted this risk in favor of 4-option uniformity.
DURATION_BUCKETS = {
    "very briefly":             (0.0, 0.5),
    "briefly":                  (0.5, 2.0),
    "for several seconds":      (2.0, 5.0),
    "for an extended period":   (5.0, float("inf")),
}

# Spatial direction buckets — used by relative_spatial_change (NEW, replaces spatial_location)
# Determined by bbox center net displacement from first to last visible frame
SPATIAL_DIRECTION_OPTIONS = [
    "moves from left to right",
    "moves from right to left",
    "moves from top to bottom",
    "moves from bottom to top",
    "stays in roughly the same place",
]

# Relative-spatial-change thresholds (all expressed as ratio of image diagonal)
SPATIAL_CHANGE_THRESHOLDS = {
    "static_max_net_displacement": 0.10,
    # net_displacement < 10% → "stays in roughly the same place"

    "ambiguous_zone_max": 0.20,
    # 10% ≤ net_displacement ≤ 20% → SKIP the question (avoid ambiguity)

    "main_direction_ratio": 1.5,
    # |dx| > |dy| * 1.5 → primarily horizontal; otherwise primarily vertical.
    # If neither dominates (diagonal movement), SKIP.
}

# Cross-object state buckets — used by conditional_state
CONDITIONAL_STATE_OPTIONS = [
    "fully visible",
    "partly hidden",
    "fully hidden",
    "not yet appeared or already gone",
]


# =============================================================================
# 10. Referring Expression Constraints (NEW in v2)
# =============================================================================
# Modifiers we can detect in a ref expr and use to exclude certain dims.

REF_EXPR_MODIFIERS = {
    "spatial": [
        "leftmost", "rightmost", "upper", "lower", "top", "bottom",
        "left side", "right side", "on the left", "on the right",
        "upper-left", "upper-right", "lower-left", "lower-right",
        "on the top", "on the bottom",
    ],
    "temporal": [
        "first", "earlier", "later", "last", "appears first",
        "that appears first", "that shows up first", "that comes back",
    ],
    "count": [
        "only", "single", "twice", "thrice", "multiple", "many",
        # "#1", "#2", "#3" 等 digit suffixes handled separately
    ],
}

# Referring expression quality gates
REF_EXPR_CONFIG = {
    "min_confidence": 0.5,
    "exclude_digit_suffix_from_options": True,
    # ref exprs like "cup #2" may appear as subject but NEVER as distractor value
    "exclude_digit_suffix_from_subjects_for_c_dims": False,
    # For C-dim subjects we allow "cup #2" in the question stem because the
    # subject ref expr is already canonicalized via referring_expressions.py.
    # What matters is that options don't leak the ID.
}


# =============================================================================
# 11. step3c Surface Realization Constraints (NEW in v2)
# =============================================================================

# Tokens and phrases that step3c MUST NOT paraphrase — preserves schema consistency.
# Any LLM prompt for step3c should include this list explicitly.
DO_NOT_PARAPHRASE = [
    # Duration bucket labels (4 labels, user decision Q2)
    "very briefly",
    "briefly",
    "for several seconds",
    "for an extended period",

    # Temporal bucket labels (4 labels, user decision Q1)
    "beginning",
    "early",
    "late",
    "end",

    # Spatial direction labels
    "moves from left to right",
    "moves from right to left",
    "moves from top to bottom",
    "moves from bottom to top",
    "stays in roughly the same place",

    # Conditional state labels
    "fully visible",
    "partly hidden",
    "fully hidden",
    "not yet appeared or already gone",

    # Reappear_or_disappear mechanisms
    "fully blocked",
    "left the frame",

    # Reappear_identity polarity
    "the same one",
    "a different one",

    # Existence polarity tokens
    "does",
    "does not",
    "never",
    "ever",

    # Numerical answers (digits)
    # handled separately as regex "^[0-9]+$" or "5 or more"
]


# =============================================================================
# 12. step3b Skeleton Generation Config (NEW in v2)
# =============================================================================

FILTER_LAYERS = {
    "text_only": {
        "enabled_dims": "all",
        "exempt_dims": [
            "reappear_identity",  # 100% label bias (Option α): Qwen's world
                                  # prior answers "same one" without visual
                                  # input, killing all 37/37. Not real text
                                  # leakage — defer to human verification.
            "event_count",        # Qwen's text-only prior always "5 or more",
                                  # creating false fails for 5+ items. Not
                                  # real text leakage.
        ],
    },
    "single_frame": {
        "enabled_dims": "all",
        "exempt_dims": [
            "reappear_identity",  # Single-frame still hits the world prior;
                                  # 37/37 failed with n_correct=5 all "A"
                                  # predictions. Defer to human verification.
        ],
    },
    "shuffled": {
        "enabled_dims": "all",
        "exempt_dims": [
            "reappear_identity",  # Ordered and shuffled both hit world prior
                                  # (ordered_n_correct=3, shuffled_n_correct=3
                                  # all predicting "A"). Defer to human
                                  # verification; annotators will determine
                                  # whether these items are genuinely solvable.
        ],
        # User decision: 不豁免 duration_category, 接受低 survival
    },
}

# For numerical format answers, need regex to extract the number from
# free-form model output
NUMERICAL_ANSWER_EXTRACTION = {
    "regex_patterns": [
        r"\b(\d+)\s*or more\b",     # "5 or more"
        r"\b([0-9]+)\b",            # plain integer
    ],
    "fallback_value": None,
    # If no number extracted, count as incorrect
}


# =============================================================================
# 14. QA Generation Config (slimmed down from v1, legacy fields removed)
# =============================================================================
QA_CONFIG = {
    "model": "gpt-5",
    "max_frames_per_prompt": 12,
    "temperature": 0.7,
    # Note: no more "task_types" / "hard_negative_types" — each dim is now
    # self-contained in DIMENSIONS above.
}


# =============================================================================
# 15. Baseline Models (unchanged from v1)
# =============================================================================
BASELINE_MODELS = {
    "videochat2":        {"type": "classic",   "temporal": False},
    "video_llava":       {"type": "classic",   "temporal": False},
    "qwen3_vl_8b":       {"type": "sota",      "temporal": True},
    "llava_onevision_7b": {"type": "sota",     "temporal": False},
    "internvl2.5_8b":    {"type": "sota",      "temporal": False},
    "tarsier_7b":        {"type": "temporal",   "temporal": True},
    "llava_video_7b":    {"type": "temporal",   "temporal": True},
    "gpt5":              {"type": "closed",     "temporal": True},
    "gemini_2.5_pro":    {"type": "closed",     "temporal": True},
    "random":            {"type": "ablation",   "temporal": False},
    "text_only":         {"type": "ablation",   "temporal": False},
    "single_frame":      {"type": "ablation",   "temporal": False},
}


# =============================================================================
# 16. Sanity Checks (NEW in v2)
# =============================================================================
# Validate that target_ratios sum to approximately 1.0

_total_ratio = sum(d["target_ratio"] for d in DIMENSIONS.values())
assert 0.95 <= _total_ratio <= 1.05, (
    f"DIMENSIONS target_ratios sum to {_total_ratio:.3f}, should be ~1.0"
)

# Validate every dim has a valid format
for dim_name, dim_cfg in DIMENSIONS.items():
    fmt = dim_cfg["format"]
    assert fmt in FORMATS, f"Dim '{dim_name}' uses unknown format '{fmt}'"
    assert dim_cfg["tier"] in {"tier1", "tier2", "tier3"}, (
        f"Dim '{dim_name}' has invalid tier '{dim_cfg['tier']}'"
    )


# =============================================================================
# Helper accessors (NEW in v2)
# =============================================================================

def get_dims_by_tier(tier: str):
    """Return list of dim names belonging to a given tier."""
    return [name for name, cfg in DIMENSIONS.items() if cfg["tier"] == tier]


def get_dims_by_format(fmt: str):
    """Return list of dim names using a given format."""
    return [name for name, cfg in DIMENSIONS.items() if cfg["format"] == fmt]


def get_c_critical_dims():
    """Return list of dim names that require instance_confidence check."""
    return [name for name, cfg in DIMENSIONS.items() if cfg.get("c_critical", False)]


def ref_expr_has_forbidden_modifier(ref_expr: str, modifier_types: list) -> bool:
    """Check if a ref expr contains modifiers of any listed type.

    Args:
        ref_expr: the referring expression to check
        modifier_types: list of modifier category names, e.g. ["spatial", "temporal"]

    Returns:
        True if any forbidden modifier is present.
    """
    ref_lower = ref_expr.lower()
    for mod_type in modifier_types:
        for modifier in REF_EXPR_MODIFIERS.get(mod_type, []):
            if modifier in ref_lower:
                return True
    return False


if __name__ == "__main__":
    # Quick self-check: print dim summary
    print("=" * 70)
    print("TOC-Bench Configuration v2 — Dimension Summary")
    print("=" * 70)
    for tier in ["tier1", "tier2", "tier3"]:
        dims = get_dims_by_tier(tier)
        total_ratio = sum(DIMENSIONS[d]["target_ratio"] for d in dims)
        print(f"\n{tier.upper()} ({total_ratio:.0%} target):")
        for d in dims:
            cfg = DIMENSIONS[d]
            c_mark = "★" if cfg.get("c_critical") else " "
            print(f"  {c_mark} {d:<28s} {cfg['format']:<10s} {cfg['target_ratio']:.0%}")
    print(f"\nTotal: {_total_ratio:.1%}")
    print(f"C-critical dims: {len(get_c_critical_dims())} / {len(DIMENSIONS)}")
