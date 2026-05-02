#!/usr/bin/env python3
"""
TOC-Bench Step 3b v2: Skeleton Builder with Hallucination Injection
=====================================================================
Reads reasoning_units/<vid>.json and materializes them into QA skeletons
(one skeleton = one fully-specified question with options, correct answer,
and all metadata needed by step3c surface realization).

KEY FEATURES v2:
  - Per-dim dispatch to 5 format handlers (mcq_4, sp, ordering, numerical)
  - Per-dim distractor rules (no more EVENT_CONFUSIONS global table)
  - Hallucination injection for 4 MCQ dims (temporal_location,
    duration_category, relative_spatial_change, conditional_state):
      * 2.5% variant A (hallucinated subject)
      * 2.5% variant B (hallucinated event)
      * 10% "none-of-above"-style distractors injected into normal options
    User decisions Q1-Q4 already encoded in this module.
  - Answer-bias balancing via down-sampling
  - Per-video skeleton cap (from config.SKELETON_CONFIG)

Input:
    QA_DIR/reasoning_units/<video_id>.json
    QA_DIR/reasoning_units/_hallucination_resources.json

Output:
    QA_DIR/skeletons/<video_id>.json
    QA_DIR/skeletons/_summary.json
"""

import argparse
import json
import random
import re
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *a, **k): return x

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    QA_DIR, DIMENSIONS, FORMATS,
    TIME_BUCKETS, DURATION_BUCKETS,
    SPATIAL_DIRECTION_OPTIONS, CONDITIONAL_STATE_OPTIONS,
    SKELETON_CONFIG,
)

UNITS_DIR = QA_DIR / "reasoning_units"
SKELETONS_DIR = QA_DIR / "skeletons"
HALLU_RES_PATH = UNITS_DIR / "_hallucination_resources.json"


# ============================================================================
# Hallucination Injection Config
# ============================================================================
# Dims that accept hallucination injection. All four are MCQ_4 format.
HALLU_ELIGIBLE_DIMS = {
    "temporal_location", "duration_category",
    "relative_spatial_change", "conditional_state",
}

# Which dims allow variant B (hallucinated event). relative_spatial_change
# doesn't — "X never moves" conflicts with the valid "stays roughly in place"
# answer.
VARIANT_B_ELIGIBLE_DIMS = {
    "temporal_location", "duration_category", "conditional_state",
}

# Probabilities per user decision:
#   - hallu_question_rate = 0.05 (hallucinated question)
#   - hallu_distractor_rate = 0.10 (one distractor replaced by "none-of-above"-style)
HALLU_QUESTION_RATE = 0.05
HALLU_DISTRACTOR_RATE = 0.10

# Within the hallucination-question quota, split between variants.
# relative_spatial_change only does A, so all its 5% goes to A.
# Other three dims split 2.5% / 2.5% between A and B.
VARIANT_A_WEIGHT = 0.5  # for dims eligible for both A and B


# ============================================================================
# Event verb → natural phrase (mirrors step3a)
# ============================================================================
EVENT_VERB_PHRASES = {
    "full_occlusion":    "gets fully hidden",
    "partial_occlusion": "gets partly hidden",
    "reappear":          "comes back into view",
    "appear":            "first shows up",
    "disappear":         "disappears for good",
    "exit_frame":        "leaves the frame",
    "enter_frame":       "enters the frame",
    "state_change":      "changes abruptly",
    "interaction":       "interacts with another object",
}

EVENT_STATE_PHRASES = {  # for "how long does X stay in state Y"
    "full_occlusion":    "fully hidden from view",
    "partial_occlusion": "partly hidden",
    "exit_frame":        "out of the frame",
    "interaction":       "in contact with {partner}",
}


def event_verb(etype):
    return EVENT_VERB_PHRASES.get(etype, etype.replace("_", " "))


def event_state(etype, partner=None):
    s = EVENT_STATE_PHRASES.get(etype, etype.replace("_", " "))
    if "{partner}" in s:
        s = s.replace("{partner}", partner or "another object")
    return s


# ============================================================================
# Hallucination Subject Generation
# ============================================================================

# Palette of simple color/material modifiers that are natural for synthetic
# hallucinated subjects. Used when the global modifier pool for a base_noun
# is too sparse.
FALLBACK_MODIFIERS = [
    "red", "blue", "green", "yellow", "black", "white", "orange", "purple",
    "brown", "pink", "gray", "silver", "gold",
    "wooden", "metal", "plastic", "glass", "ceramic",
    "small", "large", "tiny",
]


def generate_hallucinated_subject(real_subject, video_existing_labels,
                                   hallu_resources, rng):
    """Produce a fake subject that PLAUSIBLY sounds like something that
    could be in the video but isn't actually there.

    Strategy (per user decision Q4, 80% / 20% mix):
      - 80% same-base-noun synthesis: "red cup" → "blue cup", "green cup"
      - 20% cross-category: sample from global_label_pool

    Returns a string like "the blue cup" or "the wooden spoon".
    """
    real_label = real_subject.get("label", "") if isinstance(real_subject, dict) else str(real_subject)
    existing_full = set(video_existing_labels.get("full_labels", []))
    existing_base = set(video_existing_labels.get("base_nouns", []))

    # Get real subject's base noun (best-effort parse)
    real_normalized = real_label.lower().strip()
    for art in ("the ", "a ", "an "):
        if real_normalized.startswith(art):
            real_normalized = real_normalized[len(art):]
            break
    real_words = real_normalized.split()
    real_base = real_words[-1] if real_words else ""

    strategy = "same_base" if rng.random() < 0.80 else "global"

    if strategy == "same_base" and real_base:
        # Same-base-noun synthesis: try modifiers from global pool for this base
        pool_by_base = hallu_resources.get("global_modifier_pool_by_base", {})
        modifiers_for_base = pool_by_base.get(real_base, [])
        if not modifiers_for_base:
            # Fallback to simple palette
            modifiers_for_base = FALLBACK_MODIFIERS

        rng.shuffle(list(modifiers_for_base))  # non-mutating
        candidates = list(modifiers_for_base)
        rng.shuffle(candidates)

        for mod in candidates:
            synth = f"the {mod} {real_base}"
            if synth.lower() in {l.lower() for l in existing_full}:
                continue
            # Also guard against matching the real label itself
            if synth.lower() == real_label.lower():
                continue
            return synth

        # If all synths match existing, fall through to global

    # Global sampling
    global_pool = hallu_resources.get("global_label_pool", [])
    # Candidates: global labels whose base_noun is NOT in this video's existing bases
    candidates = []
    for label in global_pool:
        low = label.lower()
        if low in {l.lower() for l in existing_full}:
            continue
        # Extract its base noun (last word after stripping article)
        tmp = low
        for art in ("the ", "a ", "an "):
            if tmp.startswith(art):
                tmp = tmp[len(art):]
                break
        words = tmp.split()
        if not words:
            continue
        label_base = words[-1]
        if label_base in existing_base:
            continue  # same base noun is already covered above path
        # Also skip labels with digit suffix or suspicious modifiers
        if "#" in label:
            continue
        candidates.append(label)

    if candidates:
        return rng.choice(candidates)

    # Last resort: synthesize from fallback palette + a generic base
    fallback_bases = ["object", "item", "thing"]
    return f"the {rng.choice(FALLBACK_MODIFIERS)} {rng.choice(fallback_bases)}"


# ============================================================================
# Option assembly helper (MCQ_4)
# ============================================================================

def shuffle_mcq_4_options(correct_text, distractors, rng):
    """Return a dict with options A/B/C/D filled and correct_answer letter."""
    assert len(distractors) == 3, f"expected 3 distractors, got {len(distractors)}"
    items = [("correct", correct_text)] + [("d", d) for d in distractors]
    rng.shuffle(items)
    out = {}
    answer = None
    for letter, (tag, text) in zip("ABCD", items):
        out[f"option_{letter}"] = text
        if tag == "correct":
            answer = letter
    out["correct_answer"] = answer
    return out


# ============================================================================
# Hallucination option-text templates
# ============================================================================

def hallucinated_subject_option_text(fake_subject):
    """Text shown when correct answer is 'there is no X'."""
    return f"there is no {_strip_leading_the(fake_subject)} in this video"


def hallucinated_event_option_text(real_subject_label, event_verb_phrase):
    """Text shown when correct answer is 'X never Ys'."""
    # "the red cup" + "gets fully hidden" → "the red cup never gets fully hidden"
    # Convert present-3rd-person to bare form: "gets" -> "gets" (keep; natural after "never")
    return f"{real_subject_label} never {event_verb_phrase}"


def none_of_above_distractor_text_subject(fake_subject):
    """Distractor version: used as one of 3 wrong options, not correct answer."""
    return f"there is no {_strip_leading_the(fake_subject)} in this video"


def none_of_above_distractor_text_event(real_subject_label, event_verb_phrase):
    return f"{real_subject_label} never {event_verb_phrase}"


def _strip_leading_the(s):
    """Remove a leading 'the ' prefix (case-insensitive). Does NOT treat
    'the ' as a character set — fixes str.lstrip('the ') bug that eats any
    combination of t/h/e/space.
    """
    s = (s or "").strip()
    if s.lower().startswith("the "):
        return s[4:].strip()
    return s


# ============================================================================
# Skeleton envelope builder
# ============================================================================

def make_skeleton_base(unit, fmt):
    """Standard fields every skeleton carries."""
    return {
        "skeleton_id": uuid.uuid4().hex[:12],
        "video_id": unit["video_id"],
        "dim": unit["dim"],
        "format": fmt,
        "tier": DIMENSIONS.get(unit["dim"], {}).get("tier"),
        "subject_label": (unit.get("subject") or {}).get("label"),
        "subject_obj_id": (unit.get("subject") or {}).get("obj_id"),
        "hallucination": None,  # becomes "variant_a" / "variant_b" when injected
    }


# ============================================================================
# TIER 1 BUILDERS
# ============================================================================

def build_temporal_location_skeleton(unit, rng,
                                      video_existing_labels=None,
                                      hallu_resources=None):
    """T1.1 temporal_location → MCQ_4 with 4 time buckets."""
    subject_label = unit["subject"]["label"]
    event_type = unit["event_type"]
    correct_bucket = unit["correct_bucket"]
    prefix = unit.get("prefix")

    # Build question text
    verb = event_verb(event_type)
    if prefix == "first":
        question = f"When in the video does {subject_label} first {verb}?"
    elif prefix == "last":
        question = f"The last time {subject_label} {verb}, when is it?"
    else:
        question = f"When in the video does {subject_label} {verb}?"

    # Correct answer from TIME_BUCKETS (4 keys)
    bucket_names = list(TIME_BUCKETS.keys())
    if correct_bucket not in bucket_names:
        return None  # safety guard
    correct_text = correct_bucket
    distractors = [b for b in bucket_names if b != correct_bucket]
    rng.shuffle(distractors)
    distractors = distractors[:3]

    sk = make_skeleton_base(unit, "mcq_4")
    sk.update({
        "question": question,
        "event_type": event_type,
        "correct_bucket": correct_bucket,
        **shuffle_mcq_4_options(correct_text, distractors, rng),
        "distractor_type_tag": "bucket_swap",
    })

    # Hallucination injection
    _maybe_inject_hallucination(
        sk, unit, rng, video_existing_labels, hallu_resources,
        event_verb_for_b=verb,
    )
    return sk


def build_duration_category_skeleton(unit, rng,
                                      video_existing_labels=None,
                                      hallu_resources=None):
    """T1.2 duration_category → MCQ_4 with 4 duration buckets."""
    subject_label = unit["subject"]["label"]
    event_type = unit["event_type"]
    correct_bucket = unit["correct_bucket"]
    partner_label = unit.get("partner_label")

    state_phrase = event_state(event_type, partner_label)
    question = f"How long does {subject_label} stay {state_phrase}?"

    bucket_names = list(DURATION_BUCKETS.keys())
    if correct_bucket not in bucket_names:
        return None
    correct_text = correct_bucket
    distractors = [b for b in bucket_names if b != correct_bucket]
    rng.shuffle(distractors)
    distractors = distractors[:3]

    sk = make_skeleton_base(unit, "mcq_4")
    sk.update({
        "question": question,
        "event_type": event_type,
        "correct_bucket": correct_bucket,
        "duration_sec": unit.get("duration_sec"),
        **shuffle_mcq_4_options(correct_text, distractors, rng),
        "distractor_type_tag": "duration_bucket_swap",
    })

    # Variant B for duration: "X is never {state_phrase}"
    # Build a naturalized verb for variant B "never" text:
    b_verb = f"stays {state_phrase}"
    _maybe_inject_hallucination(
        sk, unit, rng, video_existing_labels, hallu_resources,
        event_verb_for_b=b_verb,
    )
    return sk


def build_relative_spatial_change_skeleton(unit, rng,
                                             video_existing_labels=None,
                                             hallu_resources=None):
    """T1.3 relative_spatial_change → MCQ_4 with direction options."""
    subject_label = unit["subject"]["label"]
    correct_answer = unit["correct_answer"]

    question = f"How does {subject_label} move across the frame during the video?"

    # Distractors: 3 other direction options (5 total - 1 correct = 4 others,
    # pick 3 randomly)
    all_opts = list(SPATIAL_DIRECTION_OPTIONS)
    distractors = [o for o in all_opts if o != correct_answer]
    rng.shuffle(distractors)
    distractors = distractors[:3]

    sk = make_skeleton_base(unit, "mcq_4")
    sk.update({
        "question": question,
        "displacement_ratio": unit.get("displacement_ratio"),
        **shuffle_mcq_4_options(correct_answer, distractors, rng),
        "distractor_type_tag": "direction_swap",
    })

    # Hallucination: only variant A (per user decision Q2)
    _maybe_inject_hallucination(
        sk, unit, rng, video_existing_labels, hallu_resources,
        event_verb_for_b=None,  # variant B disabled for this dim
    )
    return sk


def build_event_existence_skeleton(unit, rng, **kwargs):
    """SP.1 event_existence → SP (statement pair)."""
    subject_label = unit["subject"]["label"]
    event_type = unit["event_type"]
    polarity = unit["polarity"]  # "yes" or "no"

    verb = event_verb(event_type)
    question = f"Does {subject_label} ever {verb} in the video?"

    stmt_yes = f"{subject_label} does {verb}"
    stmt_no = f"{subject_label} never {verb}"

    if polarity == "yes":
        correct_stmt, wrong_stmt = stmt_yes, stmt_no
    else:
        correct_stmt, wrong_stmt = stmt_no, stmt_yes

    # Randomize A/B placement
    if rng.random() < 0.5:
        stmt_a, stmt_b = correct_stmt, wrong_stmt
        answer = "A"
    else:
        stmt_a, stmt_b = wrong_stmt, correct_stmt
        answer = "B"

    sk = make_skeleton_base(unit, "sp")
    sk.update({
        "question": question,
        "event_type": event_type,
        "polarity": polarity,
        "statement_A": stmt_a,
        "statement_B": stmt_b,
        "correct_answer": answer,
    })
    return sk


def build_reappear_or_disappear_skeleton(unit, rng, **kwargs):
    """SP.2 reappear_or_disappear → SP (fully blocked vs left the frame)."""
    subject_label = unit["subject"]["label"]
    scenario = unit["scenario"]
    mechanism = unit["mechanism"]  # "fully blocked" or "left the frame"

    if scenario == "reappear":
        question = (f"Before {subject_label} comes back into view, "
                    f"was it fully blocked or did it leave the frame?")
        stmt_blocked = f"{subject_label} was fully blocked before coming back"
        stmt_left = f"{subject_label} left the frame before coming back"
    else:  # disappear
        question = (f"When {subject_label} disappears, does it get fully blocked "
                    f"or leave the frame?")
        stmt_blocked = f"{subject_label} got fully blocked"
        stmt_left = f"{subject_label} left the frame"

    correct_stmt = stmt_blocked if mechanism == "fully blocked" else stmt_left
    wrong_stmt = stmt_left if mechanism == "fully blocked" else stmt_blocked

    if rng.random() < 0.5:
        stmt_a, stmt_b = correct_stmt, wrong_stmt
        answer = "A"
    else:
        stmt_a, stmt_b = wrong_stmt, correct_stmt
        answer = "B"

    sk = make_skeleton_base(unit, "sp")
    sk.update({
        "question": question,
        "scenario": scenario,
        "mechanism": mechanism,
        "statement_A": stmt_a,
        "statement_B": stmt_b,
        "correct_answer": answer,
    })
    return sk


# ============================================================================
# TIER 2 BUILDERS
# ============================================================================

def build_event_count_skeleton(unit, rng, **kwargs):
    """T2.1 event_count → Numerical (answers: 2, 3, 4, 5 or more)."""
    subject_label = unit["subject"]["label"]
    event_type = unit["event_type"]
    correct_answer = unit["correct_answer"]

    verb_plural = event_verb(event_type)  # keep 3rd-person-singular form
    question = (f"How many times does {subject_label} {verb_plural} "
                f"in the video?")

    sk = make_skeleton_base(unit, "numerical")
    sk.update({
        "question": question,
        "event_type": event_type,
        "raw_count": unit.get("raw_count"),
        "correct_answer": correct_answer,  # string: "2" / "3" / "4" / "5 or more"
    })
    return sk


def build_event_ordering_skeleton(unit, rng, **kwargs):
    """T2.2 event_ordering → Ordering_3 or Ordering_4."""
    subject_label = unit["subject"]["label"]
    k = unit["k"]
    events = unit["events"]
    # Present events in shuffled order; chronological indices are [0..k-1]
    presented_indices = list(range(k))
    rng.shuffle(presented_indices)

    # Label each presented event A/B/C/D in presentation order
    labels = list("ABCD")[:k]
    presented = []
    for label, idx in zip(labels, presented_indices):
        e = events[idx]
        presented.append({
            "label": label,
            "event_text": f"it {event_verb(e['event_type'])}",
            "event_id": e.get("event_id"),
            "event_type": e["event_type"],
            "chronological_index": idx,
        })

    # Correct order: the presentation labels sorted by chronological_index
    correct_order = [p["label"] for p in sorted(
        presented, key=lambda p: p["chronological_index"])]

    question = (f"Put the following events for {subject_label} in order "
                f"from earliest to latest.")

    fmt_name = f"ordering_{k}"
    sk = make_skeleton_base(unit, "ordering")
    sk.update({
        "format": fmt_name,  # override for ordering variant
        "question": question,
        "k": k,
        "events": presented,
        "correct_order": correct_order,
    })
    return sk


def build_reappear_identity_skeleton(unit, rng, **kwargs):
    """T2.3 reappear_identity → SP ('same one' vs 'a different one').

    All answers are "same one" per Option α (user decision).
    """
    subject_label = unit["subject"]["label"]
    object_class = unit.get("object_class", "object")
    # correct_answer always "the same one" per Option α
    correct_stmt = (f"the {object_class} that comes back is the same one "
                    f"as before")
    wrong_stmt = (f"the {object_class} that comes back is a different one")

    question = (f"After {subject_label} comes back into view, is it the same "
                f"{object_class} as before, or a different one?")

    if rng.random() < 0.5:
        stmt_a, stmt_b = correct_stmt, wrong_stmt
        answer = "A"
    else:
        stmt_a, stmt_b = wrong_stmt, correct_stmt
        answer = "B"

    sk = make_skeleton_base(unit, "sp")
    sk.update({
        "question": question,
        "object_class": object_class,
        "sibling_label": unit.get("sibling_label"),
        "statement_A": stmt_a,
        "statement_B": stmt_b,
        "correct_answer": answer,
        "occlusion_duration_sec": unit.get("occlusion_duration_sec"),
    })
    return sk


def build_occluder_identity_skeleton(unit, rng, **kwargs):
    """T2.4 occluder_identity → SP (top-1 vs top-2 occluder)."""
    subject_label = unit["subject"]["label"]
    correct_occluder = unit["correct_occluder_label"]
    wrong_occluder = unit["distractor_occluder_label"]

    question = (f"What is likely blocking {subject_label} at the moment "
                f"it becomes fully hidden?")

    correct_stmt = f"the {_strip_leading_the(correct_occluder)} likely blocks it"
    wrong_stmt = f"the {_strip_leading_the(wrong_occluder)} likely blocks it"

    if rng.random() < 0.5:
        stmt_a, stmt_b = correct_stmt, wrong_stmt
        answer = "A"
    else:
        stmt_a, stmt_b = wrong_stmt, correct_stmt
        answer = "B"

    sk = make_skeleton_base(unit, "sp")
    sk.update({
        "question": question,
        "correct_occluder": correct_occluder,
        "distractor_occluder": wrong_occluder,
        "statement_A": stmt_a,
        "statement_B": stmt_b,
        "correct_answer": answer,
    })
    return sk


# ============================================================================
# TIER 3 BUILDERS
# ============================================================================

def build_conditional_state_skeleton(unit, rng,
                                       video_existing_labels=None,
                                       hallu_resources=None):
    """T3.1 conditional_state → MCQ_4 with 4 state options."""
    subject_a = unit["subject"]["label"]
    partner = unit["partner"]
    subject_b = partner["label"]
    anchor_type = unit["anchor_event_type"]
    correct_state = unit["correct_answer"]

    verb_a = event_verb(anchor_type)
    question = (f"At the moment {subject_a} {verb_a}, "
                f"what is the state of {subject_b}?")

    all_states = list(CONDITIONAL_STATE_OPTIONS)
    if correct_state not in all_states:
        return None
    distractors = [s for s in all_states if s != correct_state]
    rng.shuffle(distractors)
    distractors = distractors[:3]

    sk = make_skeleton_base(unit, "mcq_4")
    sk.update({
        "question": question,
        "subject_a_label": subject_a,
        "subject_b_label": subject_b,
        "anchor_event_type": anchor_type,
        "anchor_frame": unit.get("anchor_frame"),
        **shuffle_mcq_4_options(correct_state, distractors, rng),
        "distractor_type_tag": "state_swap",
    })

    # Variant B: "subject A never {anchor_verb}"
    # Variant A: hallucinated subject A or B
    _maybe_inject_hallucination(
        sk, unit, rng, video_existing_labels, hallu_resources,
        event_verb_for_b=verb_a,
    )
    return sk


def build_cross_object_order_skeleton(unit, rng, **kwargs):
    """T3.2 cross_object_order → SP (A's X before B's Y, vs reverse)."""
    obj_a = unit["obj_a"]
    obj_b = unit["obj_b"]
    verb_a = event_verb(obj_a["event_type"])
    verb_b = event_verb(obj_b["event_type"])
    correct_first = unit["correct_first"]  # "a" or "b"

    question = "Which event happens first in the video?"
    stmt_a_first = (f"{obj_a['label']} {verb_a} before "
                    f"{obj_b['label']} {verb_b}")
    stmt_b_first = (f"{obj_b['label']} {verb_b} before "
                    f"{obj_a['label']} {verb_a}")

    if correct_first == "a":
        correct_stmt, wrong_stmt = stmt_a_first, stmt_b_first
    else:
        correct_stmt, wrong_stmt = stmt_b_first, stmt_a_first

    if rng.random() < 0.5:
        sa, sb = correct_stmt, wrong_stmt
        answer = "A"
    else:
        sa, sb = wrong_stmt, correct_stmt
        answer = "B"

    sk = make_skeleton_base(unit, "sp")
    sk.update({
        "question": question,
        "obj_a_label": obj_a["label"],
        "obj_b_label": obj_b["label"],
        "event_a_type": obj_a["event_type"],
        "event_b_type": obj_b["event_type"],
        "statement_A": sa,
        "statement_B": sb,
        "correct_answer": answer,
    })
    return sk


# ============================================================================
# Hallucination Injection (applied to MCQ_4 skeletons only)
# ============================================================================

def _maybe_inject_hallucination(sk, unit, rng,
                                  video_existing_labels, hallu_resources,
                                  event_verb_for_b=None):
    """Roll the dice and possibly convert a normal MCQ_4 skeleton into a
    hallucination variant or inject a 'none-of-above'-style distractor.

    Modifies `sk` in place. Sets sk["hallucination"] to a tag if injected.

    Priority (mutually exclusive):
      1. HALLU_QUESTION_RATE (5%): replace correct answer with hallucination.
      2. HALLU_DISTRACTOR_RATE (10%): swap one wrong option for a
         hallucination-style distractor. Correct answer stays the same.
      3. Otherwise: normal skeleton, no change.
    """
    dim = sk["dim"]
    if dim not in HALLU_ELIGIBLE_DIMS:
        return
    if sk["format"] != "mcq_4":
        return
    if not hallu_resources or not video_existing_labels:
        return  # hallucination not available

    roll = rng.random()
    # Variant split: if eligible for B and event_verb_for_b is provided,
    # use 50/50 for A/B within the quota; else 100% A
    variant_b_enabled = (
        dim in VARIANT_B_ELIGIBLE_DIMS and event_verb_for_b is not None)

    if roll < HALLU_QUESTION_RATE:
        # Hallucination question — correct answer becomes a "no such X/event"
        if variant_b_enabled and rng.random() < 0.5:
            _apply_variant_b(sk, unit, rng, event_verb_for_b)
        else:
            _apply_variant_a(sk, unit, rng, video_existing_labels, hallu_resources)
    elif roll < HALLU_QUESTION_RATE + HALLU_DISTRACTOR_RATE:
        # Distractor injection — replace one wrong option with fake text
        _apply_distractor_injection(
            sk, unit, rng, video_existing_labels, hallu_resources,
            event_verb_for_b=event_verb_for_b,
            variant_b_enabled=variant_b_enabled)


def _apply_variant_a(sk, unit, rng, video_existing_labels, hallu_resources):
    """Variant A: hallucinated subject in the question.

    Rewrites the question to reference a fake subject, replaces the correct
    answer with 'there is no X in this video', keeps distractors unchanged
    (they'll look wrong because they reference the fake X as if it existed).
    """
    real_subject_label = unit["subject"]["label"]
    fake_subject = generate_hallucinated_subject(
        unit["subject"], video_existing_labels, hallu_resources, rng)

    # Rewrite question: replace real subject with fake subject
    sk["question"] = sk["question"].replace(real_subject_label, fake_subject, 1)
    # For conditional_state, the question has TWO subjects (A and B). We only
    # replace A. We could instead hallucinate B, but for simplicity (and to
    # keep answer coherent) we only do A.

    # Replace correct answer
    correct_text = hallucinated_subject_option_text(fake_subject)
    correct_letter = sk["correct_answer"]
    sk[f"option_{correct_letter}"] = correct_text

    sk["hallucination"] = "variant_a"
    sk["hallucinated_subject"] = fake_subject
    sk["distractor_type_tag"] = "hallucinated_question"


def _apply_variant_b(sk, unit, rng, event_verb_for_b):
    """Variant B: hallucinated event — "subject never Ys".

    The subject IS in the video, but the event never happens. Correct answer
    becomes 'X never Ys'. Question is unchanged (still asks when/how the
    event happened, testing whether model can recognize the event didn't).
    """
    real_subject_label = unit["subject"]["label"]
    correct_text = none_of_above_distractor_text_event(
        real_subject_label, event_verb_for_b)
    correct_letter = sk["correct_answer"]
    sk[f"option_{correct_letter}"] = correct_text

    sk["hallucination"] = "variant_b"
    sk["distractor_type_tag"] = "hallucinated_question"


def _apply_distractor_injection(sk, unit, rng,
                                  video_existing_labels, hallu_resources,
                                  event_verb_for_b=None,
                                  variant_b_enabled=True):
    """Replace one of 3 distractors with a hallucination-style statement.
    Correct answer unchanged.

    This trains/tests the model to notice that wrong-looking options aren't
    just about which bucket — they can also include "nonexistent X" type
    distractors. Prevents shortcut from "the weird option is always correct".
    """
    correct_letter = sk["correct_answer"]
    wrong_letters = [L for L in "ABCD" if L != correct_letter]
    victim_letter = rng.choice(wrong_letters)

    # Decide A-style or B-style distractor
    use_b = variant_b_enabled and rng.random() < 0.5
    if use_b:
        fake_text = none_of_above_distractor_text_event(
            unit["subject"]["label"], event_verb_for_b)
    else:
        fake_subj = generate_hallucinated_subject(
            unit["subject"], video_existing_labels, hallu_resources, rng)
        fake_text = none_of_above_distractor_text_subject(fake_subj)

    sk[f"option_{victim_letter}"] = fake_text
    sk["hallucination"] = None  # normal question, just with injected distractor
    sk["has_hallucination_distractor"] = True


# ============================================================================
# Dispatch table
# ============================================================================

BUILDERS = {
    "temporal_location":       build_temporal_location_skeleton,
    "duration_category":       build_duration_category_skeleton,
    "relative_spatial_change": build_relative_spatial_change_skeleton,
    "event_existence":         build_event_existence_skeleton,
    "reappear_or_disappear":   build_reappear_or_disappear_skeleton,
    "event_count":             build_event_count_skeleton,
    "event_ordering":          build_event_ordering_skeleton,
    "reappear_identity":       build_reappear_identity_skeleton,
    "occluder_identity":       build_occluder_identity_skeleton,
    "conditional_state":       build_conditional_state_skeleton,
    "cross_object_order":      build_cross_object_order_skeleton,
}


# ============================================================================
# Answer-bias balancing (down-sample per dim to equalize correct_answer distribution)
# ============================================================================

def _correct_answer_key(sk):
    """For balancing purposes, identify the 'correct_answer bucket' for this skeleton."""
    if sk.get("hallucination"):
        # Hallucination skeletons are a separate class — don't balance against
        # normal answers. Group them as "hallu" to keep their natural distribution.
        return f"hallu_{sk['hallucination']}"
    # For ordering, use the first correct letter as proxy
    if sk["format"].startswith("ordering"):
        return sk["correct_order"][0] if sk.get("correct_order") else "A"
    return sk.get("correct_answer", "?")


def balance_answers_per_dim(skeletons, exempt_dims=None):
    """Down-sample each dim's skeletons so that every answer-bucket has the
    same count. Hallucination subcategories are preserved as-is.

    exempt_dims: set of dim names that should NOT be balanced (e.g. event_count
    has natural distribution reflecting reality).
    """
    exempt = set(exempt_dims or [])
    balanced = []
    by_dim = defaultdict(list)
    for sk in skeletons:
        by_dim[sk["dim"]].append(sk)

    for dim, sks in by_dim.items():
        if dim in exempt:
            balanced.extend(sks)
            continue
        # Group by answer-key
        by_answer = defaultdict(list)
        for sk in sks:
            by_answer[_correct_answer_key(sk)].append(sk)
        if not by_answer:
            continue

        # Separate hallu from normal
        hallu_buckets = {k: v for k, v in by_answer.items() if k.startswith("hallu_")}
        normal_buckets = {k: v for k, v in by_answer.items() if not k.startswith("hallu_")}

        # Balance normal buckets
        if normal_buckets:
            min_count = min(len(v) for v in normal_buckets.values())
            for bucket, items in normal_buckets.items():
                random.shuffle(items)
                balanced.extend(items[:min_count])

        # Include all hallu items (their rate is already pre-configured)
        for items in hallu_buckets.values():
            balanced.extend(items)

    return balanced


def apply_per_video_cap(skeletons, max_per_video, rng):
    """Per-video cap with dim diversity via round-robin.

    Strategy for each video:
      1. First, always keep ALL hallucination skeletons (rare + valuable),
         even if that exceeds max_per_video.
      2. Then fill remaining slots via round-robin across dims: iterate dims
         in a shuffled order, taking one skeleton per dim per pass, until
         the cap is reached or candidates are exhausted.
      3. If hallucinations alone exceed max_per_video, keep them all
         (don't silently drop rare data for an arbitrary cap).
    """
    by_video = defaultdict(list)
    for sk in skeletons:
        by_video[sk["video_id"]].append(sk)

    kept_all = []
    for vid, sks in by_video.items():
        if len(sks) <= max_per_video:
            kept_all.extend(sks)
            continue

        hallu = [s for s in sks if s.get("hallucination")
                 or s.get("has_hallucination_distractor")]
        normal = [s for s in sks if not (s.get("hallucination")
                                          or s.get("has_hallucination_distractor"))]

        # Always keep all hallucination skeletons
        kept = list(hallu)

        # If hallu alone exceeds cap, that's fine — we take them all.
        # Otherwise fill with round-robin across dims.
        remaining = max_per_video - len(kept)
        if remaining > 0 and normal:
            # Group normal by dim
            by_dim = defaultdict(list)
            for s in normal:
                by_dim[s["dim"]].append(s)
            for v in by_dim.values():
                rng.shuffle(v)
            dims = list(by_dim.keys())
            rng.shuffle(dims)

            # Round-robin: one per dim per pass
            cursors = {d: 0 for d in dims}
            while remaining > 0:
                took_any = False
                for d in dims:
                    if remaining <= 0:
                        break
                    items = by_dim[d]
                    idx = cursors[d]
                    if idx < len(items):
                        kept.append(items[idx])
                        cursors[d] = idx + 1
                        remaining -= 1
                        took_any = True
                if not took_any:
                    break

        kept_all.extend(kept)

    return kept_all


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-per-video", type=int,
                    default=SKELETON_CONFIG["max_skeletons_per_video"])
    ap.add_argument("--skip-balance", action="store_true",
                    help="Skip answer-bias balancing (for debugging)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    SKELETONS_DIR.mkdir(parents=True, exist_ok=True)

    # Load hallucination resources
    if HALLU_RES_PATH.exists():
        with open(HALLU_RES_PATH) as f:
            hallu_resources = json.load(f)
    else:
        print("[WARN] hallucination_resources.json not found — "
              "hallucination injection disabled")
        hallu_resources = None

    # Collect video IDs
    if args.video_id:
        video_ids = [args.video_id]
    else:
        video_ids = sorted([f.stem for f in UNITS_DIR.glob("*.json")
                            if not f.name.startswith("_")])

    # Per-video generation
    all_skeletons = []
    dim_counts_raw = Counter()
    hallu_counts = Counter()

    for vid in tqdm(video_ids):
        uf = UNITS_DIR / f"{vid}.json"
        if not uf.exists():
            continue
        with open(uf) as f:
            units_data = json.load(f)
        units = units_data.get("units", [])

        # Per-video existing labels (for hallucination synthesis)
        if hallu_resources:
            v_existing = hallu_resources.get(
                "per_video_existing", {}).get(vid, {})
        else:
            v_existing = {"base_nouns": [], "full_labels": []}

        video_skeletons = []
        for unit in units:
            builder = BUILDERS.get(unit["dim"])
            if builder is None:
                continue
            try:
                sk = builder(
                    unit, rng,
                    video_existing_labels=v_existing,
                    hallu_resources=hallu_resources,
                )
            except Exception as e:
                # Skip broken units rather than crash the whole pipeline
                continue
            if sk is None:
                continue
            video_skeletons.append(sk)
            dim_counts_raw[sk["dim"]] += 1
            if sk.get("hallucination"):
                hallu_counts[f"{sk['dim']}/{sk['hallucination']}"] += 1
            elif sk.get("has_hallucination_distractor"):
                hallu_counts[f"{sk['dim']}/distractor"] += 1

        # NOTE: per-video cap is applied AFTER balance below, not here.
        # Applying cap first would shrink the pool before balance sees it,
        # distorting answer-bucket counts and over-deleting.
        all_skeletons.extend(video_skeletons)

    print(f"\nRaw skeletons: {len(all_skeletons):,}")
    print(f"Hallucination breakdown:")
    for k, v in sorted(hallu_counts.items()):
        print(f"  {k}: {v}")

    # Step 1: Balance answer distribution on the FULL raw pool
    if not args.skip_balance:
        exempt = set(SKELETON_CONFIG.get("answer_bias_exempt_dims", []))
        balanced = balance_answers_per_dim(all_skeletons, exempt_dims=exempt)
        print(f"After balancing: {len(balanced):,} "
              f"({len(all_skeletons) - len(balanced):,} dropped)")
    else:
        balanced = all_skeletons

    # Step 2: Apply per-video cap with dim diversity (round-robin)
    final = apply_per_video_cap(
        balanced, max_per_video=args.max_per_video, rng=rng)
    print(f"After per-video cap (max={args.max_per_video}): {len(final):,} "
          f"({len(balanced) - len(final):,} dropped)")

    # Write per-video skeleton files
    by_video = defaultdict(list)
    for sk in final:
        by_video[sk["video_id"]].append(sk)

    for vid, sks in by_video.items():
        with open(SKELETONS_DIR / f"{vid}.json", "w") as f:
            json.dump({"video_id": vid, "num_skeletons": len(sks),
                       "skeletons": sks}, f, indent=2)

    # Summary
    summary = {
        "total_skeletons": len(final),
        "raw_skeletons": len(all_skeletons),
        "after_balance": len(balanced),
        "videos_covered": len(by_video),
        "dim_counts": Counter(sk["dim"] for sk in final),
        "format_counts": Counter(sk["format"] for sk in final),
        "tier_counts": Counter(sk["tier"] for sk in final),
        "hallucination_counts": dict(hallu_counts),
    }
    summary["dim_counts"] = dict(summary["dim_counts"])
    summary["format_counts"] = dict(summary["format_counts"])
    summary["tier_counts"] = dict(summary["tier_counts"])

    with open(SKELETONS_DIR / "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print report
    print("\n" + "=" * 60)
    print("  Skeleton Generation Summary (v2)")
    print("=" * 60)
    print(f"  Total skeletons:   {summary['total_skeletons']:,}")
    print(f"  Videos covered:    {summary['videos_covered']:,}")
    print("\n  By dim:")
    for dim in DIMENSIONS:
        cnt = summary['dim_counts'].get(dim, 0)
        print(f"    {dim:<30s} {cnt:>6d}")
    print("\n  By format:")
    for fmt, cnt in sorted(summary['format_counts'].items()):
        print(f"    {fmt:<20s} {cnt:>6d}")
    print("\n  Hallucination injections:")
    for k in sorted(summary['hallucination_counts']):
        print(f"    {k:<40s} {summary['hallucination_counts'][k]:>4d}")


if __name__ == "__main__":
    main()