#!/usr/bin/env python3
"""
TOC-Bench Step 3c (v2): Dim-Aware Surface Realization
======================================================
Takes skeletons produced by step3b (v2 schema, 11 dims / 5 formats) and
rewrites them into natural, varied English using an LLM — WITHOUT changing
semantics.

Key v2 contracts (enforced here, not by prompt alone):
  - Bucket labels and polarity tokens listed in config.DO_NOT_PARAPHRASE
    must survive verbatim in the polished output. Any polished field that
    drops a required locked token is rolled back to the raw skeleton text.
  - Hallucination skeletons (variant_a / variant_b, and has_hallucination_
    distractor) are routed to dedicated prompts that preserve the
    "there is no X in this video" / "X never Y" / "no <fake>" patterns.
  - Each of the 11 dims gets its own system prompt built from a shared
    spine plus a dim-specific clause and a dim-specific lock list.
  - The correct_answer letter, correct_order list, raw_count, and
    numerical correct_answer string are NEVER routed through the LLM.

Output contract:
  - All original skeleton fields are preserved untouched.
  - Polished fields are added alongside:
      polished_question
      polished_option_A..D         (mcq_4)
      polished_statement_A/B       (sp)
      polished_events[i].event_text (ordering)
    numerical format gets only polished_question (answer string is locked).
  - verification_passed / verification_issues / realization_method are
    attached for downstream filtering.

Output location: QA_DIR/qa_items_natural/<video_id>.json
(aligned with what step4_temporal_filter.py expects)

Usage:
    python step3c_surface_realize.py --dry-run
    python step3c_surface_realize.py --model openai/gpt-5.4-mini
    python step3c_surface_realize.py --model openai/gpt-5.4-mini --concurrency 20
    python step3c_surface_realize.py --max-videos 10 --overwrite
"""

import argparse
import asyncio
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *a, **k):
        return x

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    QA_DIR,
    DIMENSIONS,
    TIME_BUCKETS,
    DURATION_BUCKETS,
    SPATIAL_DIRECTION_OPTIONS,
    CONDITIONAL_STATE_OPTIONS,
    DO_NOT_PARAPHRASE,
)

SKELETONS_DIR = QA_DIR / "skeletons"
QA_ITEMS_DIR = QA_DIR / "qa_items_natural"  # aligned with step4 input


# ============================================================================
# Dim-specific lock lists (tokens/phrases that must appear verbatim in output)
# ============================================================================
# These are drawn from DO_NOT_PARAPHRASE but scoped per-dim so each prompt
# only asks the LLM to preserve what is actually relevant to that dim.
# This keeps prompts shorter and the post-hoc verification targeted.

_TIME_BUCKET_LABELS = list(TIME_BUCKETS.keys())
_DURATION_BUCKET_LABELS = list(DURATION_BUCKETS.keys())
_SPATIAL_DIRECTION_LABELS = list(SPATIAL_DIRECTION_OPTIONS)
_CONDITIONAL_STATE_LABELS = list(CONDITIONAL_STATE_OPTIONS)

_MECHANISM_LABELS = ["fully blocked", "left the frame"]
_REAPPEAR_IDENTITY_LABELS = ["the same one", "a different one"]

# Hallucination-specific phrases that MUST survive rewriting verbatim
_HALLU_VARIANT_A_MARKER = "there is no"       # "there is no <fake> in this video"
_HALLU_VARIANT_B_MARKER = "never"             # "<subject> never <verb>"
_NUMERICAL_5_PLUS = "5 or more"

# Per-dim locks for the CORRECT option only. For mcq_4 dims, all 4 options
# are drawn from the same bucket vocabulary, so the "correct must contain"
# lock applies to all options — we express that via `locks_apply_to_all`.
DIM_LOCKS = {
    "temporal_location": {
        "labels": _TIME_BUCKET_LABELS,
        "locks_apply_to_all": True,
    },
    "duration_category": {
        "labels": _DURATION_BUCKET_LABELS,
        "locks_apply_to_all": True,
    },
    "relative_spatial_change": {
        "labels": _SPATIAL_DIRECTION_LABELS,
        "locks_apply_to_all": True,
    },
    "conditional_state": {
        "labels": _CONDITIONAL_STATE_LABELS,
        "locks_apply_to_all": True,
    },
    "reappear_or_disappear": {
        "labels": _MECHANISM_LABELS,
        "locks_apply_to_all": True,
    },
    # SP dims below: statements are freely composed, so no bucket locks,
    # but we still protect polarity/identity tokens per the spine prompt.
    "event_existence": {"labels": [], "locks_apply_to_all": False},
    "event_count": {"labels": [], "locks_apply_to_all": False},
    "event_ordering": {"labels": [], "locks_apply_to_all": False},
    "reappear_identity": {
        "labels": _REAPPEAR_IDENTITY_LABELS,
        "locks_apply_to_all": False,
    },
    "occluder_identity": {"labels": [], "locks_apply_to_all": False},
    "cross_object_order": {"labels": [], "locks_apply_to_all": False},
}


def _format_lock_list_for_prompt(labels):
    """Render a bullet list of locked phrases for injection into prompts."""
    if not labels:
        return ""
    return "\n".join(f'  - "{lbl}"' for lbl in labels)


# ============================================================================
# Prompt spine — shared across all dims
# ============================================================================

_SPINE = """\
You are polishing video-understanding benchmark questions. You rewrite them \
into more fluent English while preserving meaning EXACTLY. Absolute rules:

1. Do NOT change which answer is correct.
2. Do NOT introduce any fact not present in the input.
3. Do NOT change numbers, time buckets, duration buckets, direction phrases, \
state labels, or polarity words ("ever", "never", "does", "does not").
4. Any phrase listed in "PRESERVE VERBATIM" below must appear character-for- \
character in the output.
5. Keep options in their original register — do NOT expand short bucket \
labels into sentences, and do NOT compress long sentences into short labels.
6. Fix third-person-singular verb agreement after "does"/"doesn't" (e.g. \
"does X first gets" → "does X first get").
7. Return ONLY JSON, no markdown fences, no commentary.
"""


def build_dim_system_prompt(dim, fmt, hallucination_tag, has_distractor_hallu):
    """Build the per-dim system prompt. Combines spine + dim clause + locks
    + hallucination handling."""
    lock_cfg = DIM_LOCKS.get(dim, {"labels": [], "locks_apply_to_all": False})
    lock_lines = _format_lock_list_for_prompt(lock_cfg["labels"])

    dim_clause = _DIM_CLAUSES.get(dim, _GENERIC_DIM_CLAUSE)

    hallu_clause = ""
    if hallucination_tag == "variant_a":
        hallu_clause = (
            "\nCRITICAL — HALLUCINATION QUESTION (variant A): the question \n"
            "references an object that does NOT exist in the video. The \n"
            "correct option has the exact form "
            '"there is no <something> in this video". You MUST keep that \n'
            'phrase ("there is no ... in this video") intact in the correct \n'
            "option. Do NOT rephrase it into something that sounds like a \n"
            "normal answer (e.g. do not change it to \"the <something> is \n"
            "not present\" or \"no <something> appears\"). Distractor options \n"
            "describe the fake object AS IF it existed — leave that framing \n"
            "untouched; do not make them sound 'more correct' or 'less \n"
            "absurd'.\n"
        )
    elif hallucination_tag == "variant_b":
        hallu_clause = (
            "\nCRITICAL — HALLUCINATION QUESTION (variant B): the subject \n"
            "exists but the described event never actually happens. The \n"
            'correct option has the exact form "<subject> never <verb>". You \n'
            'MUST keep the word "never" in the correct option, and keep the \n'
            "same subject label as in the question. Do NOT rephrase \n"
            "\"never\" into \"doesn't\", \"does not\", \"fails to\", or any \n"
            "other form.\n"
        )
    elif has_distractor_hallu:
        hallu_clause = (
            "\nNOTE: exactly one of the wrong options is deliberately \n"
            'absurd — it either starts with "there is no" or uses the word \n'
            '"never". Keep that option\'s structure intact; do NOT rewrite \n'
            "it into a plausible-sounding alternative. It is supposed to \n"
            "look wrong.\n"
        )

    preserve_block = ""
    if lock_lines:
        preserve_block = (
            f"\nPRESERVE VERBATIM (these phrases must appear unchanged in \n"
            f"the output):\n{lock_lines}\n"
        )

    return f"{_SPINE}\nDIMENSION-SPECIFIC GUIDANCE:\n{dim_clause}\n{preserve_block}{hallu_clause}"


# ---------- dim-specific clauses ----------
_GENERIC_DIM_CLAUSE = (
    "Rewrite the question into natural English. You may vary connective "
    "words and sentence openings but keep all content words."
)

_DIM_CLAUSES = {
    "temporal_location": (
        "This item asks WHEN during the video an event happens. The four "
        "options are time-bucket labels (beginning / early / late / end). "
        "You may rewrite the question stem, but the four options are "
        "ALREADY the exact bucket names — keep each option as the single "
        "bucket word/phrase, optionally prefixed by a short connector like "
        '"at the" or "toward the". Do NOT paraphrase them into longer '
        'sentences like "during the first quarter".'
    ),
    "duration_category": (
        "This item asks HOW LONG something stays in a state. The four "
        "options are duration-bucket phrases (very briefly / briefly / "
        "for several seconds / for an extended period). Note that "
        '"very briefly" and "briefly" are close in meaning — this is '
        "intentional and they must both remain distinct options with "
        "their exact wording. Rewrite the question stem only; options "
        "stay as the exact bucket phrase."
    ),
    "relative_spatial_change": (
        "This item asks HOW an object moves across the frame. The options "
        "are direction phrases (moves from left to right / right to left / "
        "top to bottom / bottom to top / stays in roughly the same place). "
        "You MAY rewrite the question stem but the five direction phrases "
        "are fixed exactly as given."
    ),
    "event_count": (
        "This is an open-ended numerical question. Rewrite the question "
        "stem naturally — you may rephrase it (\"How many times does X Y\" "
        '→ "On how many occasions does X Y", "Count the times X Ys", etc.). '
        "Do NOT invent a multiple-choice list; the answer is a number. The "
        "answer string itself is locked externally and will not be shown to "
        "you."
    ),
    "event_ordering": (
        "This item asks the test-taker to sort 3 or 4 events chronologically. "
        "Rewrite each event description to read as a fluent standalone "
        "clause describing what happens to the subject, but DO NOT change "
        "the event's meaning or swap it with another event. Keep the "
        'pronoun "it" if the original uses it — the subject is shared across '
        "all events and is named in the question stem."
    ),
    "reappear_identity": (
        "This item asks whether a re-entering object is the same instance "
        "as before, or a different instance of the same type. Both "
        'statements use the phrases "the same one" or "a different one" — '
        "you may rewrite the surrounding clause but keep these polarity "
        "phrases intact."
    ),
    "occluder_identity": (
        "This item asks which of two candidate objects is more likely "
        "blocking the subject. Each statement names a specific occluder. "
        "Rewrite into fluent English but keep the specific occluder noun "
        "in each statement — do not swap which object each statement is "
        "about."
    ),
    "conditional_state": (
        "This item asks: at the moment subject A does X, what state is "
        "subject B in? The four options are visibility states (fully "
        "visible / partly hidden / fully hidden / not yet appeared or "
        "already gone). The question stem names TWO subjects (A and B) — "
        "keep both. Options stay as the exact state phrase."
    ),
    "cross_object_order": (
        "This item asks whether A's event precedes B's event. Each "
        'statement has the structure "<X> <event-X> before <Y> <event-Y>". '
        "Rewrite into fluent English but do NOT swap which subject comes "
        "first in each statement — that is the whole point of the test."
    ),
    "event_existence": (
        "This is a yes/no-style item presented as two statements — one "
        'asserts the event happens ("does"), the other denies it '
        '("never"). Keep these polarity words. Rewrite connective words '
        "only."
    ),
    "reappear_or_disappear": (
        "This item asks whether the subject was fully blocked or left the "
        "frame. Keep the exact phrases \"fully blocked\" and \"left the "
        "frame\" intact in both statements — they are the mechanism labels."
    ),
}


# ============================================================================
# User-message templates (one per format)
# ============================================================================

def build_user_msg(skeleton):
    """Construct the user-side message containing the raw skeleton content.
    Returns the user string; the JSON schema the LLM must return is embedded."""
    fmt = skeleton["format"]
    dim = skeleton["dim"]

    if fmt == "mcq_4":
        return (
            f"Dimension: {dim}\n"
            f"Question: {skeleton['question']}\n"
            f"A) {skeleton['option_A']}\n"
            f"B) {skeleton['option_B']}\n"
            f"C) {skeleton['option_C']}\n"
            f"D) {skeleton['option_D']}\n"
            f"\n"
            f'Return JSON with this exact schema:\n'
            f'{{"question": "...", "option_A": "...", "option_B": "...", '
            f'"option_C": "...", "option_D": "..."}}'
        )

    if fmt == "sp":
        return (
            f"Dimension: {dim}\n"
            f"Question: {skeleton['question']}\n"
            f"A) {skeleton['statement_A']}\n"
            f"B) {skeleton['statement_B']}\n"
            f"\n"
            f'Return JSON with this exact schema:\n'
            f'{{"question": "...", "statement_A": "...", "statement_B": "..."}}'
        )

    if fmt == "numerical":
        return (
            f"Dimension: {dim}\n"
            f"Question: {skeleton['question']}\n"
            f"\n"
            f"Rewrite ONLY the question into more natural English. Do not "
            f"produce options or an answer.\n"
            f'Return JSON: {{"question": "..."}}'
        )

    if fmt.startswith("ordering_"):
        events = skeleton["events"]
        event_lines = "\n".join(
            f"  {e['label']}) {e['event_text']}" for e in events
        )
        json_events = ", ".join(f'"event_{e["label"]}": "..."' for e in events)
        return (
            f"Dimension: {dim}\n"
            f"Question: {skeleton['question']}\n"
            f"Events (presentation order, not chronological):\n{event_lines}\n"
            f"\n"
            f'Return JSON: {{"question": "...", {json_events}}}'
        )

    # Unknown format — let caller handle
    return ""


# ============================================================================
# JSON parser
# ============================================================================

def _parse_json(text):
    """Extract a JSON object from LLM output, tolerating markdown fences
    and leading/trailing prose."""
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


# ============================================================================
# Async API caller
# ============================================================================

class AsyncAPICaller:
    """Async wrapper around OpenAI-compatible chat completion with retries
    and a concurrency semaphore."""

    def __init__(self, model, temperature=0.7, max_concurrency=15,
                 max_retries=3, base_delay=1.0):
        import openai
        self.client = openai.AsyncOpenAI()
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self._call_count = 0
        self._fail_count = 0

    async def call(self, system, user_msg):
        async with self.semaphore:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=self.temperature,
                        max_tokens=700,
                    )
                    self._call_count += 1
                    text = response.choices[0].message.content.strip()
                    result = _parse_json(text)
                    if result is not None:
                        return result
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.base_delay)
                        continue
                    self._fail_count += 1
                    return None
                except Exception as e:
                    if attempt < self.max_retries:
                        delay = self.base_delay * (2 ** attempt)
                        err_str = str(e).lower()
                        if "rate" in err_str or "429" in err_str:
                            delay = max(delay, 5.0)
                        await asyncio.sleep(delay)
                    else:
                        self._fail_count += 1
                        return None

    @property
    def stats(self):
        return {"calls": self._call_count, "failures": self._fail_count}


# ============================================================================
# Skeleton validation (v2 schema)
# ============================================================================

_REQUIRED_BASE = {"skeleton_id", "video_id", "dim", "format", "question"}


def _validate_skeleton(sk):
    """Return (ok, missing_fields) for a v2 skeleton."""
    missing = [k for k in _REQUIRED_BASE if not sk.get(k)]
    if missing:
        return False, missing

    fmt = sk["format"]

    if fmt == "mcq_4":
        for k in ["option_A", "option_B", "option_C", "option_D",
                  "correct_answer"]:
            val = sk.get(k)
            if val is None or (isinstance(val, str) and not val.strip()):
                missing.append(k)
        if sk.get("correct_answer") not in ("A", "B", "C", "D"):
            missing.append("correct_answer_invalid")

    elif fmt == "sp":
        for k in ["statement_A", "statement_B", "correct_answer"]:
            val = sk.get(k)
            if val is None or (isinstance(val, str) and not val.strip()):
                missing.append(k)
        if sk.get("correct_answer") not in ("A", "B"):
            missing.append("correct_answer_invalid")

    elif fmt == "numerical":
        ca = sk.get("correct_answer")
        if not isinstance(ca, str) or not ca.strip():
            missing.append("correct_answer")

    elif fmt.startswith("ordering_"):
        events = sk.get("events")
        if not events or not isinstance(events, list):
            missing.append("events")
        else:
            for i, e in enumerate(events):
                if not e.get("label") or not e.get("event_text"):
                    missing.append(f"events[{i}]_incomplete")
        if not sk.get("correct_order"):
            missing.append("correct_order")

    else:
        missing.append(f"unknown_format:{fmt}")

    return len(missing) == 0, missing


# ============================================================================
# Dry-run polishing (no API)
# ============================================================================

_QUESTION_START = (
    "when", "how", "what", "which", "where", "does", "do ", "is ", "are ",
    "did ", "was ", "were ", "why ", "who ",
)


def _light_polish_question(text):
    """Polish a question stem: capitalize first char, ensure terminal
    '?' if it looks like a question, else '.'. Only for question strings."""
    if not text:
        return text
    text = text.strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if text and text[-1] not in ".?!":
        text += "?" if text.lower().startswith(_QUESTION_START) else "."
    return text


def _light_polish_answer(text):
    """Polish an option / statement / event text. Unlike question polishing,
    this does NOT force capitalization or terminal punctuation — many
    skeleton options are bucket labels ('late', 'briefly') that must remain
    in their exact form to preserve DO_NOT_PARAPHRASE locks.
    """
    if not text:
        return text
    return text.strip()


def dry_polish_skeleton(sk):
    """Produce a polished-fields dict without calling an LLM."""
    fmt = sk["format"]
    out = {"polished_question": _light_polish_question(sk.get("question", ""))}
    if fmt == "mcq_4":
        for letter in "ABCD":
            out[f"polished_option_{letter}"] = _light_polish_answer(
                sk.get(f"option_{letter}", "")
            )
    elif fmt == "sp":
        out["polished_statement_A"] = _light_polish_answer(
            sk.get("statement_A", "")
        )
        out["polished_statement_B"] = _light_polish_answer(
            sk.get("statement_B", "")
        )
    elif fmt.startswith("ordering_"):
        out["polished_events"] = [
            {"label": e["label"],
             "event_text": _light_polish_answer(e["event_text"])}
            for e in sk.get("events", [])
        ]
    # numerical: only polished_question
    return out


# ============================================================================
# Lock-token verification helpers
# ============================================================================

def _contains(text, needle):
    """Case-insensitive substring match."""
    if not text or not needle:
        return False
    return needle.lower() in text.lower()


def _check_dim_locks(sk, polished):
    """Check that dim-specific locked phrases survive in the correct option
    (and in all options if locks_apply_to_all).
    Returns list of issue strings."""
    issues = []
    dim = sk["dim"]
    fmt = sk["format"]
    lock_cfg = DIM_LOCKS.get(dim, {"labels": [], "locks_apply_to_all": False})
    labels = lock_cfg["labels"]
    if not labels:
        return issues

    if fmt == "mcq_4" and lock_cfg["locks_apply_to_all"]:
        # Every option was drawn from the bucket vocabulary, so every
        # polished option must still contain one of the bucket phrases.
        for letter in "ABCD":
            orig = sk.get(f"option_{letter}", "")
            pol = polished.get(f"polished_option_{letter}", "")
            # Find which bucket label appears in original; the same one
            # must appear in polished.
            orig_label = next(
                (lbl for lbl in labels if _contains(orig, lbl)), None
            )
            if orig_label and not _contains(pol, orig_label):
                issues.append(f"lock_violation_option_{letter} ({orig_label})")
    elif fmt == "sp":
        # For dims with SP bucket locks (reappear_or_disappear,
        # reappear_identity), the relevant label should appear in the correct
        # statement at minimum. Check both statements.
        for suffix in ("A", "B"):
            orig = sk.get(f"statement_{suffix}", "")
            pol = polished.get(f"polished_statement_{suffix}", "")
            orig_label = next(
                (lbl for lbl in labels if _contains(orig, lbl)), None
            )
            if orig_label and not _contains(pol, orig_label):
                issues.append(f"lock_violation_statement_{suffix} ({orig_label})")
    return issues


def _check_hallucination_locks(sk, polished):
    """Hallucination-aware lock checks."""
    issues = []
    fmt = sk["format"]
    tag = sk.get("hallucination")
    has_distr = sk.get("has_hallucination_distractor", False)

    if fmt != "mcq_4":
        return issues

    correct_letter = sk.get("correct_answer")

    if tag == "variant_a":
        # Correct option must still contain "there is no"
        pol = polished.get(f"polished_option_{correct_letter}", "")
        if not _contains(pol, _HALLU_VARIANT_A_MARKER):
            issues.append("hallu_a_pattern_lost")
    elif tag == "variant_b":
        pol = polished.get(f"polished_option_{correct_letter}", "")
        if not _contains(pol, _HALLU_VARIANT_B_MARKER):
            issues.append("hallu_b_pattern_lost")
    elif has_distr:
        # One of the wrong options originally contained "there is no" or
        # "never". That pattern must still exist in at least one wrong option.
        wrong_letters = [L for L in "ABCD" if L != correct_letter]
        originals = [sk.get(f"option_{L}", "") for L in wrong_letters]
        polisheds = [polished.get(f"polished_option_{L}", "")
                     for L in wrong_letters]
        orig_has_a = any(_contains(o, _HALLU_VARIANT_A_MARKER) for o in originals)
        orig_has_b = any(_contains(o, _HALLU_VARIANT_B_MARKER) for o in originals)
        pol_has_a = any(_contains(p, _HALLU_VARIANT_A_MARKER) for p in polisheds)
        pol_has_b = any(_contains(p, _HALLU_VARIANT_B_MARKER) for p in polisheds)
        if orig_has_a and not pol_has_a:
            issues.append("hallu_distractor_a_pattern_lost")
        if orig_has_b and not pol_has_b:
            issues.append("hallu_distractor_b_pattern_lost")
    return issues


def _check_numerical_answer_locked(sk, polished):
    """numerical: answer string must be untouched externally. Only question
    is polished, so this is automatic — but we still sanity-check that the
    skeleton's correct_answer is in the allowed set."""
    ca = sk.get("correct_answer", "")
    if ca != _NUMERICAL_5_PLUS and not re.fullmatch(r"\d+", str(ca)):
        return ["numerical_answer_invalid"]
    return []


# ============================================================================
# Realize one skeleton
# ============================================================================

async def realize_skeleton(sk, api_caller, dry_run=False):
    """Return (polished_dict, method) or (None, None) for malformed.
    method ∈ {"llm", "template", "template_fallback", "rolled_back"}."""
    ok, _missing = _validate_skeleton(sk)
    if not ok:
        return None, None

    fmt = sk["format"]

    # Dry-run path: no API
    if dry_run:
        return dry_polish_skeleton(sk), "template"

    system = build_dim_system_prompt(
        dim=sk["dim"],
        fmt=fmt,
        hallucination_tag=sk.get("hallucination"),
        has_distractor_hallu=sk.get("has_hallucination_distractor", False),
    )
    user = build_user_msg(sk)
    if not user:
        return dry_polish_skeleton(sk), "template_fallback"

    result = await api_caller.call(system, user)
    if result is None:
        return dry_polish_skeleton(sk), "template_fallback"

    # Coerce result into polished_* schema
    polished = _coerce_llm_result(sk, result)

    # Post-hoc lock verification — roll back individual fields if needed
    polished, roll_issues = _roll_back_on_lock_violation(sk, polished)

    method = "rolled_back" if roll_issues else "llm"
    return polished, method


def _coerce_llm_result(sk, result):
    """Turn the LLM's JSON output into polished_* fields, preserving the
    skeleton's format schema."""
    fmt = sk["format"]
    out = {"polished_question": (result.get("question") or "").strip()}
    if fmt == "mcq_4":
        for letter in "ABCD":
            val = (result.get(f"option_{letter}") or "").strip()
            out[f"polished_option_{letter}"] = val
    elif fmt == "sp":
        out["polished_statement_A"] = (result.get("statement_A") or "").strip()
        out["polished_statement_B"] = (result.get("statement_B") or "").strip()
    elif fmt.startswith("ordering_"):
        polished_events = []
        for e in sk.get("events", []):
            key = f"event_{e['label']}"
            txt = (result.get(key) or "").strip()
            polished_events.append({"label": e["label"], "event_text": txt})
        out["polished_events"] = polished_events
    # numerical: only polished_question; answer is locked externally
    return out


def _roll_back_on_lock_violation(sk, polished):
    """If any polished field violates the dim locks or hallucination locks,
    roll that specific field back to its raw skeleton value. Returns
    (polished, rollback_issues)."""
    issues = []
    fmt = sk["format"]

    # Empty-field check: roll back any polished field that came back empty.
    if not polished.get("polished_question"):
        polished["polished_question"] = sk.get("question", "")
        issues.append("empty_question_rolled_back")

    if fmt == "mcq_4":
        for letter in "ABCD":
            key = f"polished_option_{letter}"
            if not polished.get(key):
                polished[key] = sk.get(f"option_{letter}", "")
                issues.append(f"empty_option_{letter}_rolled_back")
    elif fmt == "sp":
        for suffix in ("A", "B"):
            key = f"polished_statement_{suffix}"
            if not polished.get(key):
                polished[key] = sk.get(f"statement_{suffix}", "")
                issues.append(f"empty_statement_{suffix}_rolled_back")
    elif fmt.startswith("ordering_"):
        orig_events = {e["label"]: e["event_text"] for e in sk.get("events", [])}
        for pe in polished.get("polished_events", []):
            if not pe.get("event_text"):
                pe["event_text"] = orig_events.get(pe["label"], "")
                issues.append(f"empty_event_{pe['label']}_rolled_back")

    # Dim-specific bucket locks
    lock_issues = _check_dim_locks(sk, polished)
    if lock_issues:
        # Roll back affected fields to skeleton originals
        for issue in lock_issues:
            if "option_A" in issue:
                polished["polished_option_A"] = sk.get("option_A", "")
            elif "option_B" in issue:
                polished["polished_option_B"] = sk.get("option_B", "")
            elif "option_C" in issue:
                polished["polished_option_C"] = sk.get("option_C", "")
            elif "option_D" in issue:
                polished["polished_option_D"] = sk.get("option_D", "")
            elif "statement_A" in issue:
                polished["polished_statement_A"] = sk.get("statement_A", "")
            elif "statement_B" in issue:
                polished["polished_statement_B"] = sk.get("statement_B", "")
        issues.extend(lock_issues)

    # Hallucination locks
    hallu_issues = _check_hallucination_locks(sk, polished)
    if hallu_issues:
        # For hallucination violations, roll back ALL options to originals
        # (safest — we can't tell just from the pattern test which option
        # lost its marker)
        if fmt == "mcq_4":
            for letter in "ABCD":
                polished[f"polished_option_{letter}"] = sk.get(
                    f"option_{letter}", ""
                )
        issues.extend(hallu_issues)

    return polished, issues


# ============================================================================
# Verification (after realize)
# ============================================================================

def verify_item(sk, polished):
    """Return (passed, issues). Checks apply to the polished fields."""
    issues = []
    fmt = sk["format"]

    # Base: polished_question non-empty and reasonable length
    pq = polished.get("polished_question", "").strip()
    if not pq:
        issues.append("empty_polished_question")
    elif len(pq) < 5:
        issues.append("too_short_polished_question")

    if fmt == "mcq_4":
        _verify_mcq4(sk, polished, issues)
    elif fmt == "sp":
        _verify_sp(sk, polished, issues)
    elif fmt == "numerical":
        _verify_numerical(sk, polished, issues)
    elif fmt.startswith("ordering_"):
        _verify_ordering(sk, polished, issues)

    return len(issues) == 0, issues


def _verify_mcq4(sk, polished, issues):
    opts = [polished.get(f"polished_option_{L}", "") for L in "ABCD"]
    for L, opt in zip("ABCD", opts):
        if not opt.strip():
            issues.append(f"empty_polished_option_{L}")

    # Length balance — but skip for hallucination items where distractors
    # are bucket labels (1-3 words) while the correct or injected option
    # is a full hallucination sentence (~6-10 words). The imbalance is by
    # design, not a polish artifact.
    is_hallu = (
        sk.get("hallucination") in ("variant_a", "variant_b")
        or sk.get("has_hallucination_distractor")
    )
    if not is_hallu:
        lengths = [len(o.split()) for o in opts if o.strip()]
        if lengths:
            max_l, min_l = max(lengths), max(1, min(lengths))
            if max_l > min_l * 4:
                issues.append(
                    f"option_length_imbalance (max={max_l},min={min_l})"
                )

    # Duplicate options
    norm = [o.strip().lower() for o in opts]
    if len(set(n for n in norm if n)) < len([n for n in norm if n]):
        issues.append("duplicate_options")

    # Content preservation for the CORRECT option.
    # Exception: for hallucination variant_a/variant_b, content drift from
    # the original distractor-like correct text is expected — skip drift check.
    tag = sk.get("hallucination")
    if tag not in ("variant_a", "variant_b"):
        correct_letter = sk.get("correct_answer", "A")
        orig = sk.get(f"option_{correct_letter}", "")
        pol = polished.get(f"polished_option_{correct_letter}", "")
        if orig and pol:
            orig_words = set(re.findall(r"[a-z]+", orig.lower()))
            pol_words = set(re.findall(r"[a-z]+", pol.lower()))
            if orig_words:
                overlap = len(orig_words & pol_words) / len(orig_words)
                if overlap < 0.25:
                    issues.append(
                        f"content_drift_correct_option (overlap={overlap:.2f})"
                    )


def _verify_sp(sk, polished, issues):
    a = polished.get("polished_statement_A", "").strip()
    b = polished.get("polished_statement_B", "").strip()
    if not a:
        issues.append("empty_polished_statement_A")
    if not b:
        issues.append("empty_polished_statement_B")

    if a and b:
        if a.lower() == b.lower():
            issues.append("identical_statements")
        a_len, b_len = len(a.split()), len(b.split())
        ratio = max(a_len, b_len) / max(1, min(a_len, b_len))
        if ratio > 3.0:
            issues.append(f"statement_length_imbalance ({a_len} vs {b_len})")


def _verify_numerical(sk, polished, issues):
    # Numerical correct_answer never routed through LLM, but validate it.
    issues.extend(_check_numerical_answer_locked(sk, polished))


def _verify_ordering(sk, polished, issues):
    pe_list = polished.get("polished_events", [])
    orig_events = sk.get("events", [])
    if len(pe_list) != len(orig_events):
        issues.append(
            f"event_count_changed ({len(pe_list)} vs {len(orig_events)})"
        )
        return
    # Each event text non-trivial
    for pe in pe_list:
        txt = pe.get("event_text", "").strip()
        if not txt:
            issues.append(f"empty_event_{pe.get('label', '?')}")
        elif len(txt) < 5:
            issues.append(f"too_short_event_{pe.get('label', '?')}")

    # Labels preserved
    orig_labels = [e["label"] for e in orig_events]
    pol_labels = [pe.get("label") for pe in pe_list]
    if orig_labels != pol_labels:
        issues.append("event_labels_reordered")


# ============================================================================
# Assemble final QA item (preserves all skeleton fields + polished_*)
# ============================================================================

def assemble_qa_item(sk, polished, method, passed, issues):
    """Merge skeleton + polished + metadata into a single QA item dict.
    All original skeleton fields are preserved; polished_* added alongside."""
    item = dict(sk)  # shallow copy preserves every skeleton field
    item.update(polished)
    item["qa_id"] = sk["skeleton_id"]  # for step4 compatibility
    item["realization_method"] = method
    item["verification_passed"] = passed
    item["verification_issues"] = issues
    return item


# ============================================================================
# Per-video processing
# ============================================================================

async def realize_video_async(skeleton_data, api_caller, dry_run=False):
    skeletons = skeleton_data.get("skeletons", [])
    vid = skeleton_data.get("video_id", "?")
    if not skeletons:
        return []

    # Validate up front, skip malformed
    valid, malformed = [], 0
    for sk in skeletons:
        ok, _m = _validate_skeleton(sk)
        if ok:
            valid.append(sk)
        else:
            malformed += 1
    if malformed > 0:
        print(f"    [WARN] {vid}: {malformed}/{len(skeletons)} skeletons "
              f"malformed, skipped", flush=True)
    if not valid:
        return []

    tasks = [realize_skeleton(sk, api_caller, dry_run=dry_run) for sk in valid]
    results = await asyncio.gather(*tasks)

    items = []
    for (polished, method), sk in zip(results, valid):
        if polished is None:
            continue
        passed, issues = verify_item(sk, polished)
        items.append(assemble_qa_item(sk, polished, method, passed, issues))
    return items


def realize_video_sync(skeleton_data):
    """Dry-run synchronous path for one video."""
    skeletons = skeleton_data.get("skeletons", [])
    vid = skeleton_data.get("video_id", "?")
    items = []
    malformed = 0
    for sk in skeletons:
        ok, _m = _validate_skeleton(sk)
        if not ok:
            malformed += 1
            continue
        polished = dry_polish_skeleton(sk)
        passed, issues = verify_item(sk, polished)
        items.append(assemble_qa_item(sk, polished, "template", passed, issues))
    if malformed > 0:
        print(f"    [WARN] {vid}: {malformed} skeleton(s) malformed, skipped",
              flush=True)
    return items


# ============================================================================
# Batch orchestration
# ============================================================================

async def process_batch_async(skeleton_files, api_caller, dry_run, overwrite,
                              batch_size=50):
    """Yield (video_id, output_path, was_cached) for each file."""
    total = len(skeleton_files)
    for batch_start in range(0, total, batch_size):
        batch = skeleton_files[batch_start:batch_start + batch_size]
        coros, metas = [], []

        for sf in batch:
            vid = sf.stem
            output_path = QA_ITEMS_DIR / f"{vid}.json"
            if output_path.exists() and not overwrite:
                metas.append((vid, output_path, True))
                coros.append(None)
                continue
            with open(sf) as f:
                sk_data = json.load(f)
            metas.append((vid, output_path, False))
            coros.append(realize_video_async(sk_data, api_caller, dry_run))

        active = [c for c in coros if c is not None]
        active_results = await asyncio.gather(*active) if active else []

        idx = 0
        for vid, output_path, cached in metas:
            if cached:
                yield vid, output_path, True
            else:
                items = active_results[idx]
                idx += 1
                with open(output_path, "w") as f:
                    json.dump({
                        "video_id": vid,
                        "num_qa_items": len(items),
                        "qa_items": items,
                    }, f, indent=2)
                yield vid, output_path, False


# ============================================================================
# Main / CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TOC-Bench Step 3c (v2): dim-aware surface realization"
    )
    parser.add_argument("--model", type=str, default="openai/gpt-5.4-mini")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true",
                        help="Capitalize + punctuate only; no API calls")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("  TOC-Bench Step 3c v2: Dim-Aware Surface Realization")
    print("  Skeletons:", SKELETONS_DIR)
    print("  Output:   ", QA_ITEMS_DIR)
    if args.dry_run:
        print("  Mode: DRY-RUN (no API)")
    else:
        print(f"  Model: {args.model}   Concurrency: {args.concurrency}")
    print("=" * 70)

    QA_ITEMS_DIR.mkdir(parents=True, exist_ok=True)

    skeleton_files = sorted(
        f for f in SKELETONS_DIR.glob("*.json")
        if not f.name.startswith("_")
    )
    if args.max_videos:
        skeleton_files = skeleton_files[:args.max_videos]
    print(f"  Found {len(skeleton_files)} skeleton files")
    if not skeleton_files:
        print("  [ERROR] No skeletons. Run step3b first.")
        sys.exit(1)

    if args.dry_run:
        _run_dry(skeleton_files, args.overwrite)
        return

    try:
        import openai  # noqa: F401
    except ImportError:
        print("  [ERROR] openai package not installed. "
              "Use --dry-run or: pip install openai")
        sys.exit(1)

    api_caller = AsyncAPICaller(
        model=args.model, temperature=args.temperature,
        max_concurrency=args.concurrency,
    )
    asyncio.run(_run_async(skeleton_files, api_caller, args))


async def _run_async(skeleton_files, api_caller, args):
    all_items = []
    counters = _new_counters()
    skipped, new_processed = 0, 0
    t_start = time.time()

    with tqdm(total=len(skeleton_files), desc="  Realizing videos",
              unit="video") as pbar:
        async for vid, output_path, was_cached in process_batch_async(
            skeleton_files, api_caller, dry_run=False,
            overwrite=args.overwrite, batch_size=args.batch_size,
        ):
            pbar.update(1)
            with open(output_path) as f:
                items = json.load(f).get("qa_items", [])
            if was_cached:
                skipped += 1
            else:
                new_processed += 1
            all_items.extend(items)
            _update_counters(counters, items)
            pbar.set_postfix({
                "items": f"{counters['total']:,}",
                "passed": f"{counters['passed']:,}",
                "api": api_caller.stats["calls"],
            })

    _save_and_report(all_items, counters, skipped, time.time() - t_start,
                     args.model, api_caller.stats)


def _run_dry(skeleton_files, overwrite):
    all_items = []
    counters = _new_counters()
    skipped = 0
    t_start = time.time()

    for sf in tqdm(skeleton_files, desc="  Realizing videos (dry)",
                   unit="video"):
        vid = sf.stem
        output_path = QA_ITEMS_DIR / f"{vid}.json"
        if output_path.exists() and not overwrite:
            with open(output_path) as f:
                items = json.load(f).get("qa_items", [])
            skipped += 1
        else:
            with open(sf) as f:
                sk_data = json.load(f)
            items = realize_video_sync(sk_data)
            with open(output_path, "w") as f:
                json.dump({"video_id": vid,
                           "num_qa_items": len(items),
                           "qa_items": items}, f, indent=2)
        all_items.extend(items)
        _update_counters(counters, items)

    _save_and_report(all_items, counters, skipped, time.time() - t_start,
                     "dry_run", {})


def _new_counters():
    return {
        "total": 0, "passed": 0, "failed": 0,
        "by_format": Counter(),
        "by_dim": Counter(),
        "by_tier": Counter(),
        "by_method": Counter(),
        "by_hallu": Counter(),
        "issues": Counter(),
    }


def _update_counters(c, items):
    for it in items:
        c["total"] += 1
        c["by_format"][it.get("format", "?")] += 1
        c["by_dim"][it.get("dim", "?")] += 1
        c["by_tier"][it.get("tier", "?")] += 1
        c["by_method"][it.get("realization_method", "?")] += 1
        htag = it.get("hallucination") or (
            "distractor" if it.get("has_hallucination_distractor") else "none"
        )
        c["by_hallu"][htag] += 1
        if it.get("verification_passed"):
            c["passed"] += 1
        else:
            c["failed"] += 1
        for iss in it.get("verification_issues", []):
            base = iss.split(" (")[0]
            c["issues"][base] += 1


def _save_and_report(all_items, c, skipped, elapsed, model, api_stats):
    all_path = QA_ITEMS_DIR / "_all.json"
    with open(all_path, "w") as f:
        json.dump({"total_qa_items": len(all_items), "qa_items": all_items},
                  f, indent=2)

    print(f"\n{'='*70}\n  Results\n{'='*70}")
    print(f"  Total QA items:        {c['total']:,}")
    print(f"  Verification passed:   {c['passed']:,}")
    print(f"  Verification failed:   {c['failed']:,}")
    if c["total"] > 0:
        print(f"  Pass rate:             {c['passed']/c['total']*100:.1f}%")
    print(f"  Skipped (cached):      {skipped}")
    print(f"  Time elapsed:          {elapsed:.0f}s")
    if api_stats:
        print(f"  API calls:             {api_stats.get('calls', 0)}")
        print(f"  API failures:          {api_stats.get('failures', 0)}")

    print(f"\n  By tier:")
    for tier, cnt in sorted(c["by_tier"].items()):
        pct = cnt / max(1, c["total"]) * 100
        print(f"    {tier:<10s}  {cnt:>6,d}  ({pct:5.1f}%)")

    print(f"\n  By dim:")
    for dim, cnt in sorted(c["by_dim"].items(), key=lambda x: -x[1]):
        pct = cnt / max(1, c["total"]) * 100
        print(f"    {dim:<28s}  {cnt:>6,d}  ({pct:5.1f}%)")

    print(f"\n  By format:")
    for fmt, cnt in c["by_format"].most_common():
        pct = cnt / max(1, c["total"]) * 100
        print(f"    {fmt:<14s}  {cnt:>6,d}  ({pct:5.1f}%)")

    print(f"\n  By realization method:")
    for m, cnt in c["by_method"].most_common():
        pct = cnt / max(1, c["total"]) * 100
        print(f"    {m:<20s}  {cnt:>6,d}  ({pct:5.1f}%)")

    print(f"\n  By hallucination tag:")
    for tag, cnt in c["by_hallu"].most_common():
        pct = cnt / max(1, c["total"]) * 100
        print(f"    {tag:<12s}  {cnt:>6,d}  ({pct:5.1f}%)")

    if c["issues"]:
        print(f"\n  Verification issues (top 15):")
        for iss, cnt in c["issues"].most_common(15):
            print(f"    {iss:<42s}  {cnt:>6,d}")

    summary = {
        "total_qa_items": c["total"],
        "verification_passed": c["passed"],
        "verification_failed": c["failed"],
        "pass_rate": round(c["passed"] / max(1, c["total"]), 4),
        "by_tier": dict(c["by_tier"]),
        "by_dim": dict(c["by_dim"]),
        "by_format": dict(c["by_format"]),
        "by_method": dict(c["by_method"]),
        "by_hallu": dict(c["by_hallu"]),
        "issue_counts": dict(c["issues"]),
        "elapsed_seconds": round(elapsed, 1),
        "model": model,
        "api_stats": api_stats or {},
    }
    with open(QA_ITEMS_DIR / "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary: {QA_ITEMS_DIR / '_summary.json'}")
    print(f"  All QA:  {all_path}")


if __name__ == "__main__":
    main()