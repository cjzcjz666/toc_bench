#!/usr/bin/env python3
"""
TOC-Bench Step 4: Temporal Necessity Filter (Qwen3-VL-8B)
==========================================================
Three independent filter layers, each run separately and saved to its
own result file.  A QA item must pass ALL three layers to enter the
final benchmark.

Layer 1 — text_only:
    Qwen3-VL-8B receives ONLY the question + options (no images).
    If it answers correctly → text leaks the answer → mark FAIL.

Layer 2 — single_frame:
    Qwen3-VL-8B receives ONE randomly sampled frame + question.
    If it answers correctly → spatial info suffices → mark FAIL.

Layer 3 — shuffled:
    Qwen3-VL-8B receives frames in CORRECT order → record accuracy.
    Then receives frames in SHUFFLED order → record accuracy.
    If ordered correct but shuffled also correct → temporal order not
    needed → mark FAIL.

Supports four QA formats from step 3c (v2 naming):
    mcq_4          — 4-way multiple choice (A/B/C/D)
    sp             — contrastive statement pair (A/B)
    ordering_3/4   — event ordering (sort 3 or 4 events chronologically)
    numerical      — open numerical answer ("2" / "3" / "4" / "5 or more")

All prompts read the polished_* fields written by step 3c v2 when present,
falling back to the raw question/option/statement fields otherwise.

Per-layer dim exemption is supported via config.FILTER_LAYERS[layer]
.enabled_dims — items whose dim is not enabled for a layer are marked
passed for that layer without consuming a model inference.

Each layer writes its results independently:
    QA_DIR/filtered/layer1_text_only.json
    QA_DIR/filtered/layer2_single_frame.json
    QA_DIR/filtered/layer3_shuffled.json
    QA_DIR/filtered/_combined.json          (intersection of all passed)

Usage:
    python step4_temporal_filter.py --layer text_only
    python step4_temporal_filter.py --layer single_frame
    python step4_temporal_filter.py --layer shuffled
    python step4_temporal_filter.py --combine
    python step4_temporal_filter.py --dry-run --layer text_only
    python step4_temporal_filter.py --layer text_only --device cuda:0
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    QA_DIR, TRACKS_DIR, EVENTS_DIR, VLM_CONFIG,
    FILTER_LAYERS, DIMENSIONS,
)

QA_ITEMS_DIR = QA_DIR / "qa_items_natural"
FILTER_DIR = QA_DIR / "filtered_natural"

# ============================================================
# Configuration
# ============================================================

LAYER_CONFIGS = {
    "text_only": {
        "description": "Text-only: question + options, no visual input",
        "output_file": "layer1_text_only.json",
    },
    "single_frame": {
        "description": "Single random frame + question",
        "output_file": "layer2_single_frame.json",
    },
    "shuffled": {
        "description": "Ordered vs shuffled frame comparison",
        "output_file": "layer3_shuffled.json",
        "num_frames": 16,
        "delta_threshold": 0.05,
    },
}

# ============================================================
# Format-Specific Prompt Templates
# ============================================================

# --- MCQ-4 ---
MCQ_VISUAL_PROMPT = (
    "Answer the following multiple-choice question about the video. "
    "Respond with ONLY the letter A, B, C, or D.\n\n"
    "Question: {question}\n\n"
    "A) {option_A}\n"
    "B) {option_B}\n"
    "C) {option_C}\n"
    "D) {option_D}\n\n"
    "Answer:"
)

MCQ_TEXT_PROMPT = (
    "Answer the following multiple-choice question. "
    "You have NO visual information — answer based ONLY on the text. "
    "Respond with ONLY the letter A, B, C, or D.\n\n"
    "Question: {question}\n\n"
    "A) {option_A}\n"
    "B) {option_B}\n"
    "C) {option_C}\n"
    "D) {option_D}\n\n"
    "Answer:"
)

# --- Ordering (3 or 4 events) ---
# We present events and ask model to output the correct chronological
# ordering as a comma-separated sequence of labels.
ORDER_VISUAL_PROMPT = (
    "Watch the video and put the following events in chronological order "
    "(earliest to latest). Respond with ONLY the letters separated by "
    "commas (e.g. C, A, B).\n\n"
    "Question: {question}\n\n"
    "{event_list}\n\n"
    "Correct chronological order:"
)

ORDER_TEXT_PROMPT = (
    "Put the following events in chronological order "
    "(earliest to latest) based ONLY on common sense — you have NO visual "
    "information. Respond with ONLY the letters separated by commas "
    "(e.g. C, A, B).\n\n"
    "Question: {question}\n\n"
    "{event_list}\n\n"
    "Correct chronological order:"
)

# --- Statement Pair (binary A/B) ---
PAIR_VISUAL_PROMPT = (
    "Watch the video and decide which statement is better supported. "
    "Respond with ONLY the letter A or B.\n\n"
    "Question: {question}\n\n"
    "A) {statement_A}\n"
    "B) {statement_B}\n\n"
    "Answer:"
)

PAIR_TEXT_PROMPT = (
    "Decide which statement is more likely to be true. "
    "You have NO visual information — answer based ONLY on the text. "
    "Respond with ONLY the letter A or B.\n\n"
    "Question: {question}\n\n"
    "A) {statement_A}\n"
    "B) {statement_B}\n\n"
    "Answer:"
)


# --- Numerical (open-ended number) ---
# No options given to the model; the answer set ("2", "3", "4", "5 or more")
# is enforced at extraction time to avoid giving away the schema.
NUMERICAL_VISUAL_PROMPT = (
    "Watch the video and answer the following question with a single "
    "integer. If the count is five or more, respond with \"5 or more\".\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

NUMERICAL_TEXT_PROMPT = (
    "Answer the following question based ONLY on the text — you have NO "
    "visual information. Respond with a single integer, or \"5 or more\".\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


# ============================================================
# Polished-field readers (v2: step3c writes polished_* alongside originals)
# ============================================================

def _pq(item):
    """Prefer polished_question, fall back to question."""
    return (item.get("polished_question")
            or item.get("question") or "").strip()


def _po(item, letter):
    """Prefer polished_option_X, fall back to option_X (mcq_4)."""
    return (item.get(f"polished_option_{letter}")
            or item.get(f"option_{letter}") or "").strip()


def _ps(item, suffix):
    """Prefer polished_statement_A/B, fall back to statement_A/B (sp)."""
    return (item.get(f"polished_statement_{suffix}")
            or item.get(f"statement_{suffix}") or "").strip()


def _pe(item):
    """Return ordering events with polished event_text where available.
    Always returns a list of {'label', 'event_text'} dicts."""
    polished = item.get("polished_events")
    if polished:
        return [
            {"label": e["label"],
             "event_text": (e.get("event_text") or "").strip()}
            for e in polished
        ]
    # Fallback to raw events list
    return [
        {"label": e["label"],
         "event_text": (e.get("event_text") or "").strip()}
        for e in item.get("events", [])
    ]


# ============================================================
# Per-layer dim exemption
# ============================================================

def layer_applies_to_item(layer_name, item):
    """Check whether the given filter layer should be run on this item,
    based on config.FILTER_LAYERS[layer].enabled_dims.

    Rules:
      - "all" (default): always run
      - list of dim names: run only if item.dim is in the list
      - "exempt_dims": list of dim names to skip (opposite semantics)
    Returns True if the layer should run, False if the item should be
    auto-marked as passed for this layer.
    """
    cfg = FILTER_LAYERS.get(layer_name, {})
    enabled = cfg.get("enabled_dims", "all")
    exempt = cfg.get("exempt_dims", [])

    dim = item.get("dim", "")

    if exempt and dim in exempt:
        return False
    if enabled == "all" or enabled is None:
        return True
    if isinstance(enabled, (list, tuple, set)):
        return dim in enabled
    # Unknown enabled spec → be conservative and run
    return True




def build_prompt(item, text_only=False):
    """
    Build the appropriate prompt string for this item's format.
    Reads polished_* fields (from step3c v2) with fallback to raw fields.
    Returns (prompt_text, valid_answers).
      valid_answers: set of acceptable answer strings, or a sentinel
      "NUMERICAL" marking open-ended numeric output.
    """
    fmt = item.get("format", "mcq_4")
    question = _pq(item)

    if fmt == "mcq_4":
        template = MCQ_TEXT_PROMPT if text_only else MCQ_VISUAL_PROMPT
        prompt = template.format(
            question=question,
            option_A=_po(item, "A"),
            option_B=_po(item, "B"),
            option_C=_po(item, "C"),
            option_D=_po(item, "D"),
        )
        return prompt, {"A", "B", "C", "D"}

    if fmt.startswith("ordering_"):
        events = _pe(item)
        event_lines = "\n".join(
            f"{e['label']}) {e['event_text']}" for e in events
        )
        template = ORDER_TEXT_PROMPT if text_only else ORDER_VISUAL_PROMPT
        prompt = template.format(question=question, event_list=event_lines)
        labels = {e["label"] for e in events}
        return prompt, labels

    if fmt == "sp":
        template = PAIR_TEXT_PROMPT if text_only else PAIR_VISUAL_PROMPT
        prompt = template.format(
            question=question,
            statement_A=_ps(item, "A"),
            statement_B=_ps(item, "B"),
        )
        return prompt, {"A", "B"}

    if fmt == "numerical":
        template = NUMERICAL_TEXT_PROMPT if text_only else NUMERICAL_VISUAL_PROMPT
        prompt = template.format(question=question)
        return prompt, "NUMERICAL"

    return None, set()


def get_correct_answer(item):
    """
    Return the correct answer for any format.
      mcq_4, sp  → single letter string
      ordering_* → list of labels, e.g. ["C", "A", "B"]
      numerical  → string, one of ("2", "3", "4", "5 or more")
    """
    fmt = item.get("format", "mcq_4")
    if fmt.startswith("ordering_"):
        return item.get("correct_order", [])
    # mcq_4, sp, numerical all store their correct answer in correct_answer
    return item.get("correct_answer", "A")


def check_correct(prediction, correct, fmt):
    """
    Compare prediction to correct answer, format-aware.
    Returns bool.
    """
    if prediction is None:
        return False

    if fmt.startswith("ordering_"):
        if isinstance(prediction, list) and isinstance(correct, list):
            return prediction == correct
        return False

    if fmt == "numerical":
        # Strict string match against allowed set {"2","3","4","5 or more"}
        if not isinstance(prediction, str) or not isinstance(correct, str):
            return False
        return prediction.strip().lower() == correct.strip().lower()

    # mcq_4, sp
    return prediction == correct


# ============================================================
# Answer Extraction (format-aware)
# ============================================================

def extract_answer(text, fmt, item=None):
    """
    Parse model output into a structured answer.
    For mcq_4: returns single letter A/B/C/D or None.
    For sp:    returns A/B or None.
    For ordering_*: returns list of labels or None.
    For numerical:  returns string ("2"/"3"/"4"/"5 or more") or None.

    `item` is optional; for ordering it lets us derive k from correct_order
    instead of hardcoding against the format string.
    """
    if not text:
        return None
    text = text.strip()

    if fmt.startswith("ordering_"):
        return _extract_ordering(text, fmt, item=item)
    if fmt == "sp":
        return _extract_letter(text, valid={"A", "B"})
    if fmt == "numerical":
        return _extract_numerical(text)
    # mcq_4 default
    return _extract_letter(text, valid={"A", "B", "C", "D"})


def _extract_letter(text, valid):
    """Extract a single letter answer from response.

    Matching priority (high → low):
      1. Exact single-token match (e.g. "A")
      2. Letter followed by typical answer punctuation: "A.", "A)", "A:",
         "A,", "A]", "A\"", or an explicit 'Answer: A' pattern
      3. Standalone letter delimited by word boundaries AND surrounded by
         whitespace (not embedded in a word)

    We deliberately do NOT fall back to "first valid character in text",
    because that yields false positives like "cannot determine" → 'C'.
    If nothing convincingly looks like a letter answer, return None.
    """
    text_up = text.upper().strip()

    # 1. Exact single-token match
    if text_up in valid:
        return text_up

    # 2. "Answer: X" pattern (with optional punctuation after)
    pattern_alts = "|".join(sorted(valid))
    m = re.search(rf'ANSWER\s*[:\-]?\s*({pattern_alts})\b', text_up)
    if m:
        return m.group(1)

    # 3. Letter followed by typical answer punctuation
    m = re.search(rf'(?<![A-Z])({pattern_alts})\s*[.,):\]\"\']', text_up)
    if m:
        return m.group(1)

    # 4. Standalone letter: must be surrounded by whitespace or string
    #    boundaries, NOT embedded in another word.
    m = re.search(rf'(?:^|\s)({pattern_alts})(?=\s|$)', text_up)
    if m:
        return m.group(1)

    return None


def _extract_ordering(text, fmt, item=None):
    """
    Extract ordered label sequence from response.
    Accepts forms like "C, A, B", "C,A,B", "C A B", "CAB".

    Derives `k` (expected number of labels) from the item's correct_order
    when available, or from the format string suffix otherwise.
    """
    text_up = text.upper().strip()

    if item is not None and isinstance(item.get("correct_order"), list):
        k = len(item["correct_order"])
    elif fmt == "ordering_4":
        k = 4
    elif fmt == "ordering_3":
        k = 3
    else:
        # Fall back to parsing events if we can find them
        if item is not None:
            pe = _pe(item)
            k = len(pe) if pe else 3
        else:
            k = 3

    valid_labels = {chr(ord("A") + i) for i in range(k)}

    parts = re.findall(r"[A-Z]", text_up)
    seen = set()
    result = []
    for p in parts:
        if p in valid_labels and p not in seen:
            result.append(p)
            seen.add(p)

    if len(result) == k:
        return result
    return None


# Pre-compiled numerical extraction patterns (order matters: "5 or more"
# must be tried before plain integer so a "5 or more" response doesn't
# get parsed as "5").
_NUMERICAL_PATTERNS = [
    # "5 or more", "5+ or more", "five or more" (we spell-out digits just
    # in case the model writes "five or more"; map back to "5 or more")
    (re.compile(r"\b(\d+)\s*\+?\s*or\s+more\b", re.IGNORECASE), "or_more"),
    (re.compile(r"\b(five|5)\s*\+?\b", re.IGNORECASE), "five_plus"),
    # Plain integer
    (re.compile(r"\b(\d+)\b"), "int"),
]


def _extract_numerical(text):
    """Parse a numerical response into one of {"2","3","4","5 or more"}.

    Behaviour:
      - If the text contains an "or more" pattern → "5 or more".
      - If it contains an integer N:
          * N >= 5 → "5 or more"
          * N == 2/3/4 → str(N)
          * N in {0, 1} → None (not a valid benchmark answer)
      - Otherwise → None.
    """
    if not text:
        return None
    for pattern, kind in _NUMERICAL_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        if kind == "or_more":
            return "5 or more"
        if kind == "five_plus":
            return "5 or more"
        # int
        try:
            n = int(m.group(1))
        except (ValueError, IndexError):
            continue
        if n >= 5:
            return "5 or more"
        if n in (2, 3, 4):
            return str(n)
        return None
    return None


# ============================================================
# Qwen3-VL-8B Inference Engine
# ============================================================

class Qwen3VLEngine:
    """
    Wraps Qwen3-VL-8B for inference across all QA formats.
    Uses device_map="auto" to spread across available GPUs.
    """

    def __init__(self, model_name=None, device="auto", mock=False):
        self.mock = mock
        self.model = None
        self.processor = None
        self.tokenizer = None

        if mock:
            print("  [Qwen3-VL] Mock mode — random answers")
            return

        model_name = model_name or VLM_CONFIG.get(
            "model_name", "Qwen/Qwen3-VL-8B-Instruct"
        )

        import torch

        if device == "auto":
            device_map = "auto"
            n_gpus = torch.cuda.device_count()
            print(f"  Loading Qwen3-VL: {model_name} "
                  f"(device_map=auto, {n_gpus} GPUs)")
        else:
            device_map = {"": device}
            print(f"  Loading Qwen3-VL: {model_name} on {device}")

        from transformers import AutoProcessor, AutoTokenizer

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        try:
            from transformers import Qwen3VLForConditionalGeneration
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.bfloat16,
                device_map=device_map,
            )
        except ImportError:
            from transformers import Qwen2_5_VLForConditionalGeneration
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.bfloat16,
                device_map=device_map,
            )
        self.model.eval()
        if hasattr(self.model, "hf_device_map"):
            devices_used = set(str(v) for v in self.model.hf_device_map.values())
            print(f"  [OK] Qwen3-VL loaded across: {devices_used}")
        else:
            print(f"  [OK] Qwen3-VL loaded")

    def answer(self, item, images=None, text_only=False):
        """
        Answer a QA item of any format.

        Args:
            item: QA item dict (from step 3c output)
            images: list of PIL Images, or None
            text_only: if True, force text-only prompt

        Returns:
            predicted answer (letter, or list for ordering), or None
        """
        fmt = item.get("format", "mcq_4")
        prompt_text, valid = build_prompt(
            item, text_only=(text_only or images is None)
        )
        if prompt_text is None:
            return None

        if self.mock:
            return self._mock_answer(fmt, item=item)

        import torch
        from qwen_vl_utils import process_vision_info

        content = []
        if images and not text_only:
            for img in images:
                content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content}]

        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            # Ordering needs more tokens for the sequence. Numerical
            # also needs a few tokens ("5 or more" = 3 tokens).
            if fmt.startswith("ordering_"):
                max_tokens = 16
            elif fmt == "numerical":
                max_tokens = 8
            else:
                max_tokens = 8

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=False,
                )

            generated_ids = [
                out[len(inp):]
                for inp, out in zip(inputs.input_ids, output_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            # Free intermediate tensors
            del inputs, output_ids, generated_ids
            torch.cuda.empty_cache()

            return extract_answer(response, fmt, item=item)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    [WARN] OOM — clearing cache and retrying "
                      f"without images")
                torch.cuda.empty_cache()
                # Retry as text-only (better than returning None)
                if images and not text_only:
                    return self.answer(item, images=None, text_only=True)
            print(f"    [WARN] Qwen3-VL inference failed: {e}")
            return None
        except Exception as e:
            print(f"    [WARN] Qwen3-VL inference failed: {e}")
            return None

    def _mock_answer(self, fmt, item=None):
        """Generate a random answer for dry-run mode."""
        if fmt == "mcq_4":
            return random.choice(["A", "B", "C", "D"])
        if fmt == "sp":
            return random.choice(["A", "B"])
        if fmt == "numerical":
            return random.choice(["2", "3", "4", "5 or more"])
        if fmt.startswith("ordering_"):
            # Derive k from item if possible, else from fmt suffix
            if item is not None and isinstance(item.get("correct_order"),
                                                list):
                k = len(item["correct_order"])
            elif fmt == "ordering_4":
                k = 4
            else:
                k = 3
            labels = [chr(ord("A") + i) for i in range(k)]
            random.shuffle(labels)
            return labels
        return None


# ============================================================
# Frame Loading
# ============================================================

def load_video_frames_for_qa(video_id, n_frames=16, rng=None):
    """
    Load frames for a video from the tracking data's video path.
    Returns list of PIL Images, or empty list if unavailable.
    """
    from PIL import Image
    import cv2

    track_path = TRACKS_DIR / f"{video_id}.json"
    if not track_path.exists():
        return []

    with open(track_path) as f:
        td = json.load(f)

    video_path = td.get("video_path", "")
    if not video_path:
        return []

    path = Path(video_path)

    try:
        if path.is_dir():
            jpgs = sorted(path.glob("*.jpg"))
            if not jpgs:
                jpgs = sorted(path.glob("*.png"))
            if not jpgs:
                return []
            if len(jpgs) <= n_frames:
                selected = jpgs
            else:
                indices = np.linspace(0, len(jpgs) - 1, n_frames, dtype=int)
                selected = [jpgs[i] for i in indices]
            frames = []
            for fp in selected:
                img = Image.open(fp).convert("RGB")
                frames.append(img)
            return frames
        else:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                return []
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return []
            if total <= n_frames:
                indices = list(range(total))
            else:
                indices = np.linspace(0, total - 1, n_frames, dtype=int)
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb))
            cap.release()
            return frames
    except Exception as e:
        print(f"    [WARN] Frame load failed for {video_id}: {e}")
        return []


# ============================================================
# Checkpoint (atomic shard-file write)
# ============================================================

def _atomic_write_json(path, payload):
    """Write JSON atomically: write to .tmp then os.replace.
    Safe against concurrent reads by other processes."""
    import os as _os
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    _os.replace(tmp, path)


def make_checkpoint_fn(shard_file, layer, layer_desc, done_items,
                       prev_in_shard, result_key):
    """Return a callable that writes the current state of the shard to
    disk atomically. The callable takes one argument: the list of newly
    completed results from this run.

    Shard-file contents = done_items (from earlier runs / shards)
                        + prev_in_shard (pre-existing results in THIS shard file)
                        + current_new_results (this run, so far)

    Dedup is by qa_id with "new wins" semantics so that a re-computed
    item updates the shard record.
    """
    def checkpoint(current_new_results):
        # Union everything into one dict keyed by qa_id
        merged = {}
        # Done items from OTHER shards / previous runs shouldn't land in THIS
        # shard file (they live in their own files and will be de-duped at
        # _merge_all time). But writing them here doesn't hurt either — it
        # just makes the per-shard files denser. We intentionally do NOT
        # include done_items: keep each shard file representing the work
        # THIS shard did, for easier debugging.
        for it in prev_in_shard.values():
            merged[it.get("qa_id")] = it
        for it in current_new_results:
            qid = it.get("qa_id")
            if qid:
                merged[qid] = it
        items = list(merged.values())
        n_passed = sum(
            1 for it in items if it.get(result_key, {}).get("passed", True)
        )
        payload = {
            "layer": layer,
            "description": layer_desc,
            "total_items": len(items),
            "passed": n_passed,
            "failed": len(items) - n_passed,
            "pass_rate": round(n_passed / max(1, len(items)) * 100, 1),
            "checkpoint": True,
            "items": items,
        }
        _atomic_write_json(shard_file, payload)
    return checkpoint




# For sp (50% random baseline), a single trial is too noisy.
# We run 3 trials and require ALL correct to FAIL (unanimous).
# P(all 3 correct by chance) = 0.5^3 = 12.5% — much better than 50%.
# For mcq_4 (25%), ordering (<17%), and numerical (~25%), single trial suffices.
TEXTONLY_REPEAT_FORMATS = {"sp": 3}                # format → n_trials
TEXTONLY_MAJORITY = {"sp": 3}                       # format → min correct to FAIL


def run_text_only(items, engine, checkpoint_fn=None, checkpoint_every=100):
    """
    For each item, ask Qwen3-VL in text-only mode (no images).
    If model answers correctly → text leaks answer → FAIL.

    For statement_pair: runs 3 trials, requires majority (≥2/3) correct
    to FAIL, reducing false-positive rate from 50% to ~25%.

    If checkpoint_fn is provided, it is called every `checkpoint_every`
    items with the current partial `results` list. The callable is
    expected to atomically persist the shard state.
    """
    results = []
    n_failed = 0
    total_calls = 0
    fmt_fail = Counter()
    fmt_total = Counter()

    for idx, item in enumerate(items):
        fmt = item.get("format", "mcq_4")
        fmt_total[fmt] += 1
        correct = get_correct_answer(item)

        # Per-dim exemption: skip layer for items whose dim is not enabled
        if not layer_applies_to_item("text_only", item):
            item["layer1_result"] = {
                "predicted": None,
                "is_correct": False,
                "passed": True,
                "note": "layer_not_enabled_for_dim",
            }
            results.append(item)
            continue

        n_trials = TEXTONLY_REPEAT_FORMATS.get(fmt, 1)
        majority_k = TEXTONLY_MAJORITY.get(fmt, 1)

        predictions = []
        n_correct = 0

        for trial in range(n_trials):
            pred = engine.answer(item, images=None, text_only=True)
            is_correct = check_correct(pred, correct, fmt)
            predictions.append({"predicted": pred, "is_correct": is_correct})
            total_calls += 1
            if is_correct:
                n_correct += 1
            # Early stop: already reached majority
            if n_correct >= majority_k:
                break
            # Early stop: impossible to reach majority
            remaining = n_trials - (trial + 1)
            if n_correct + remaining < majority_k:
                break

        is_failed = n_correct >= majority_k  # model can solve it text-only
        passed = not is_failed

        if is_failed:
            n_failed += 1
            fmt_fail[fmt] += 1

        item["layer1_result"] = {
            "predicted": predictions[-1]["predicted"],  # last prediction
            "is_correct": is_failed,
            "n_trials": len(predictions),
            "n_correct": n_correct,
            "majority_threshold": majority_k,
            "passed": passed,
        }
        if n_trials > 1:
            item["layer1_result"]["all_predictions"] = predictions

        results.append(item)

        # Periodic checkpoint: persist partial results so Ctrl-C doesn't
        # lose progress. Uses configurable interval.
        if checkpoint_fn is not None and (idx + 1) % checkpoint_every == 0:
            try:
                checkpoint_fn(results)
            except Exception as e:
                print(f"    [WARN] checkpoint failed at idx={idx+1}: {e}",
                      flush=True)

        if (idx + 1) % 200 == 0:
            fail_rate = n_failed / (idx + 1) * 100
            print(f"    [{idx+1}/{len(items)}]  fail_rate={fail_rate:.1f}%"
                  f"  (pass rate={100-fail_rate:.1f}%)"
                  f"  total_calls={total_calls}", flush=True)

    # Final checkpoint to ensure the tail (items after last interval) is
    # persisted. Safe to call even if checkpoint_fn is None or was just
    # called on the exact last item.
    if checkpoint_fn is not None:
        try:
            checkpoint_fn(results)
        except Exception as e:
            print(f"    [WARN] final checkpoint failed: {e}", flush=True)

    n_passed = sum(1 for it in results if it["layer1_result"]["passed"])
    print(f"  Layer 1 done: passed={n_passed}/{len(items)}"
          f"  failed={n_failed}"
          f"  ({n_passed/max(1,len(items))*100:.1f}% pass rate)"
          f"  total_calls={total_calls}")

    for fmt in sorted(fmt_total):
        t = fmt_total[fmt]
        f = fmt_fail.get(fmt, 0)
        print(f"    {fmt:<20s}  failed={f}/{t}"
              f"  ({f/max(1,t)*100:.1f}%)"
              f"  passed={t-f}/{t}")

    return results


# ============================================================
# Layer 2: Progressive Single-Frame with Early Stopping
# ============================================================

def _dispersed_order(n):
    """
    Generate a frame index ordering that maximizes temporal spread.
    Uses binary-split insertion: first the middle, then quartiles,
    then eighths, etc. — so even if we stop after 3 frames, we've
    covered beginning, middle, and end.

    Example for n=16:
      [8, 4, 12, 2, 6, 10, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0]
    """
    if n <= 0:
        return []
    if n == 1:
        return [0]

    order = []
    # BFS over midpoints
    queue = [(0, n - 1)]
    while len(order) < n:
        next_queue = []
        for lo, hi in queue:
            if lo > hi:
                continue
            mid = (lo + hi) // 2
            if mid not in order:
                order.append(mid)
            if len(order) >= n:
                break
            next_queue.append((lo, mid - 1))
            next_queue.append((mid + 1, hi))
        if not next_queue:
            break
        queue = next_queue

    # Add any remaining indices
    for i in range(n):
        if i not in set(order):
            order.append(i)

    return order[:n]


# Fail thresholds per format: (max_probes, min_correct_to_fail)
# Calibrated against random baselines:
#   mcq_4:      25% random → P(≥3 in 8) ≈ 32% → k=3 has ~68% specificity
#   sp:         50% random → P(≥5 in 8) ≈ 36% → k=5 gives ~64% specificity
#   ordering_3: ~17% random → k=2 very unlikely by chance
#   ordering_4:  ~4% random → k=2 extremely unlikely by chance
#   numerical:  ~25% random (4 buckets) → k=3 treated like mcq_4
SINGLE_FRAME_THRESHOLDS = {
    "mcq_4":       (8, 3),
    "sp":          (8, 5),
    "ordering_3":  (6, 2),
    "ordering_4":  (6, 2),
    "numerical":   (8, 3),
}
SINGLE_FRAME_DEFAULT = (8, 3)


def run_single_frame(items, engine, rng, checkpoint_fn=None,
                     checkpoint_every=100):
    """
    Progressive single-frame filter with early stopping.

    For each item:
      1. Load 16 uniformly-sampled frames.
      2. Probe frames one at a time in dispersed temporal order
         (middle → quartiles → eighths...).
      3. After each probe, check: if correct_count >= threshold → FAIL
         (early stop). If remaining probes can't possibly reach threshold
         → PASS (early stop).
      4. After max_probes, decide based on total correct count.
    """
    results = []
    n_failed = 0
    n_no_frames = 0
    total_probes = 0
    fmt_fail = Counter()
    fmt_total = Counter()

    for idx, item in enumerate(items):
        fmt = item.get("format", "mcq_4")
        fmt_total[fmt] += 1
        video_id = item.get("video_id", "")

        # Per-dim exemption
        if not layer_applies_to_item("single_frame", item):
            item["layer2_result"] = {
                "n_correct": 0,
                "n_probes": 0,
                "passed": True,
                "note": "layer_not_enabled_for_dim",
            }
            results.append(item)
            continue

        frames = load_video_frames_for_qa(video_id, n_frames=16, rng=rng)

        if not frames:
            item["layer2_result"] = {
                "predicted": None,
                "n_correct": 0,
                "n_probes": 0,
                "passed": True,
                "note": "no_frames_available",
            }
            n_no_frames += 1
            results.append(item)
            continue

        max_probes, fail_k = SINGLE_FRAME_THRESHOLDS.get(fmt,
                                                          SINGLE_FRAME_DEFAULT)
        max_probes = min(max_probes, len(frames))

        # Determine dispersed probe order
        probe_order = _dispersed_order(len(frames))[:max_probes]

        correct_answer = get_correct_answer(item)
        n_correct = 0
        probes_done = 0
        predictions = []
        early_stop_reason = None

        for probe_idx in probe_order:
            frame = frames[probe_idx]
            pred = engine.answer(item, images=[frame])
            is_correct = check_correct(pred, correct_answer, fmt)
            predictions.append({
                "frame_idx": probe_idx,
                "predicted": pred,
                "is_correct": is_correct,
            })
            if is_correct:
                n_correct += 1
            probes_done += 1
            total_probes += 1

            # Early stop: already hit fail threshold
            if n_correct >= fail_k:
                early_stop_reason = "fail_threshold_reached"
                break

            # Early stop: impossible to reach fail threshold
            remaining = max_probes - probes_done
            if n_correct + remaining < fail_k:
                early_stop_reason = "cannot_reach_threshold"
                break

        passed = n_correct < fail_k
        if not passed:
            n_failed += 1
            fmt_fail[fmt] += 1

        item["layer2_result"] = {
            "n_correct": n_correct,
            "n_probes": probes_done,
            "max_probes": max_probes,
            "fail_threshold": fail_k,
            "early_stop": early_stop_reason,
            "predictions": predictions,
            "passed": passed,
        }
        results.append(item)

        # Periodic checkpoint
        if checkpoint_fn is not None and (idx + 1) % checkpoint_every == 0:
            try:
                checkpoint_fn(results)
            except Exception as e:
                print(f"    [WARN] checkpoint failed at idx={idx+1}: {e}",
                      flush=True)

        if (idx + 1) % 200 == 0:
            tested = idx + 1 - n_no_frames
            fail_rate = n_failed / max(1, tested) * 100
            avg_probes = total_probes / max(1, tested)
            print(f"    [{idx+1}/{len(items)}]"
                  f"  fail_rate={fail_rate:.1f}%"
                  f"  avg_probes={avg_probes:.1f}"
                  f"  no_frames={n_no_frames}", flush=True)

    # Final checkpoint
    if checkpoint_fn is not None:
        try:
            checkpoint_fn(results)
        except Exception as e:
            print(f"    [WARN] final checkpoint failed: {e}", flush=True)

    tested = len(items) - n_no_frames
    n_passed = sum(1 for it in results if it["layer2_result"]["passed"])
    avg_probes = total_probes / max(1, tested)
    print(f"  Layer 2 done: passed={n_passed}/{len(items)}"
          f"  failed={n_failed}"
          f"  avg_probes={avg_probes:.1f}/item"
          f"  total_inferences={total_probes:,}"
          f"  no_frames={n_no_frames}")

    for fmt in sorted(fmt_total):
        t = fmt_total[fmt]
        f = fmt_fail.get(fmt, 0)
        print(f"    {fmt:<20s}  failed={f}/{t}"
              f"  ({f/max(1,t)*100:.1f}%)"
              f"  passed={t-f}/{t}")

    # Early stop statistics
    early_stops = Counter()
    for it in results:
        r = it.get("layer2_result", {})
        es = r.get("early_stop")
        if es:
            early_stops[es] += 1
    if early_stops:
        print(f"\n  Early stop breakdown:")
        for reason, cnt in early_stops.most_common():
            print(f"    {reason:<30s}  {cnt:>6,d}")

    return results


# ============================================================
# Layer 3: Shuffled (ordered vs shuffled accuracy delta)
# ============================================================

def run_shuffled(items, engine, rng, n_frames=16, delta_threshold=0.05,
                 checkpoint_fn=None, checkpoint_every=100):
    """
    For each item:
      1. Feed frames in CORRECT order → get prediction(s)
      2. Feed frames in SHUFFLED order → get prediction(s)
      3. Decision:
         PASS if ordered correct AND shuffled wrong (needs temporal order)
         PASS if ordered wrong (model can't solve it either way — keep it)
         FAIL if both correct (temporal order not needed)
         FAIL if ordered wrong AND shuffled correct (anomaly)

    For statement_pair (50% random baseline): runs 3 trials each for
    ordered and shuffled, using majority vote to reduce noise.
    For mcq_4 and ordering: single trial suffices.
    """
    SHUFFLE_REPEAT_FORMATS = {"sp": 3}
    SHUFFLE_MAJORITY = {"sp": 3}  # unanimous: all 3 correct → FAIL

    results = []
    n_ordered_correct = 0
    n_shuffled_correct = 0
    n_no_frames = 0
    total_calls = 0

    for idx, item in enumerate(items):
        fmt = item.get("format", "mcq_4")
        video_id = item.get("video_id", "")

        # Per-dim exemption
        if not layer_applies_to_item("shuffled", item):
            item["layer3_result"] = {
                "ordered_pred": None,
                "shuffled_pred": None,
                "passed": True,
                "note": "layer_not_enabled_for_dim",
            }
            results.append(item)
            continue

        frames = load_video_frames_for_qa(video_id, n_frames=n_frames,
                                           rng=rng)

        if not frames:
            item["layer3_result"] = {
                "ordered_pred": None, "shuffled_pred": None,
                "passed": True, "note": "no_frames_available",
            }
            n_no_frames += 1
            results.append(item)
            continue

        correct = get_correct_answer(item)
        n_trials = SHUFFLE_REPEAT_FORMATS.get(fmt, 1)
        majority_k = SHUFFLE_MAJORITY.get(fmt, 1)

        # --- Ordered predictions ---
        ordered_n_correct = 0
        ordered_preds = []
        for trial in range(n_trials):
            pred = engine.answer(item, images=frames)
            is_correct = check_correct(pred, correct, fmt)
            ordered_preds.append({"predicted": pred, "is_correct": is_correct})
            total_calls += 1
            if is_correct:
                ordered_n_correct += 1
            # Early stop
            if ordered_n_correct >= majority_k:
                break
            remaining = n_trials - (trial + 1)
            if ordered_n_correct + remaining < majority_k:
                break

        ordered_correct = ordered_n_correct >= majority_k
        if ordered_correct:
            n_ordered_correct += 1

        # --- Shuffled predictions ---
        shuffled_n_correct = 0
        shuffled_preds = []
        for trial in range(n_trials):
            shuffled_frames = list(frames)
            rng.shuffle(shuffled_frames)
            pred = engine.answer(item, images=shuffled_frames)
            is_correct = check_correct(pred, correct, fmt)
            shuffled_preds.append({"predicted": pred, "is_correct": is_correct})
            total_calls += 1
            if is_correct:
                shuffled_n_correct += 1
            if shuffled_n_correct >= majority_k:
                break
            remaining = n_trials - (trial + 1)
            if shuffled_n_correct + remaining < majority_k:
                break

        shuffled_correct = shuffled_n_correct >= majority_k
        if shuffled_correct:
            n_shuffled_correct += 1

        # Decision
        if ordered_correct and not shuffled_correct:
            passed = True   # needs temporal order
        elif not ordered_correct:
            passed = True   # hard item, keep it
        else:
            passed = False  # both correct or shuffled-only correct

        item["layer3_result"] = {
            "ordered_pred": ordered_preds[-1]["predicted"],
            "ordered_correct": ordered_correct,
            "ordered_n_correct": ordered_n_correct,
            "shuffled_pred": shuffled_preds[-1]["predicted"],
            "shuffled_correct": shuffled_correct,
            "shuffled_n_correct": shuffled_n_correct,
            "n_trials": n_trials,
            "majority_threshold": majority_k,
            "passed": passed,
        }
        if n_trials > 1:
            item["layer3_result"]["ordered_preds"] = ordered_preds
            item["layer3_result"]["shuffled_preds"] = shuffled_preds

        results.append(item)

        # Periodic checkpoint
        if checkpoint_fn is not None and (idx + 1) % checkpoint_every == 0:
            try:
                checkpoint_fn(results)
            except Exception as e:
                print(f"    [WARN] checkpoint failed at idx={idx+1}: {e}",
                      flush=True)

        if (idx + 1) % 100 == 0:
            tested = idx + 1 - n_no_frames
            o_acc = n_ordered_correct / max(1, tested) * 100
            s_acc = n_shuffled_correct / max(1, tested) * 100
            print(f"    [{idx+1}/{len(items)}]  ordered={o_acc:.1f}%"
                  f"  shuffled={s_acc:.1f}%"
                  f"  delta={o_acc-s_acc:.1f}pp"
                  f"  calls={total_calls}", flush=True)

    # Final checkpoint
    if checkpoint_fn is not None:
        try:
            checkpoint_fn(results)
        except Exception as e:
            print(f"    [WARN] final checkpoint failed: {e}", flush=True)

    tested = len(items) - n_no_frames
    o_acc = n_ordered_correct / max(1, tested) * 100
    s_acc = n_shuffled_correct / max(1, tested) * 100
    n_passed = sum(1 for it in results if it["layer3_result"]["passed"])
    print(f"  Layer 3 done: ordered={o_acc:.1f}%  shuffled={s_acc:.1f}%"
          f"  delta={o_acc-s_acc:.1f}pp")
    print(f"  Passed: {n_passed}/{len(items)}"
          f"  total_calls={total_calls}"
          f"  no_frames={n_no_frames}")

    return results


# ============================================================
# Combine Layers
# ============================================================

def combine_layers(filter_dir):
    """
    Load all layer results and compute the intersection: items that
    passed ALL available layers.
    """
    layer_files = {
        "text_only": filter_dir / "layer1_text_only.json",
        "single_frame": filter_dir / "layer2_single_frame.json",
        "shuffled": filter_dir / "layer3_shuffled.json",
    }

    item_results = defaultdict(dict)
    all_items = {}
    layers_available = []

    for layer_name, path in layer_files.items():
        if not path.exists():
            print(f"  [SKIP] {path.name} not found")
            continue
        layers_available.append(layer_name)
        with open(path) as f:
            data = json.load(f)
        for item in data.get("items", []):
            qa_id = item["qa_id"]
            result_key = f"layer{list(layer_files.keys()).index(layer_name)+1}_result"
            passed = item.get(result_key, {}).get("passed", True)
            item_results[qa_id][layer_name] = passed
            all_items[qa_id] = item

    print(f"  Layers available: {layers_available}")
    print(f"  Total unique items: {len(item_results)}")

    combined_passed = []
    combined_failed = []
    fail_reasons = Counter()

    for qa_id, layer_passes in item_results.items():
        all_passed = all(layer_passes.get(l, True) for l in layers_available)
        item = all_items[qa_id]
        if all_passed:
            combined_passed.append(item)
        else:
            failed_layers = [
                l for l in layers_available if not layer_passes.get(l, True)
            ]
            for fl in failed_layers:
                fail_reasons[fl] += 1
            combined_failed.append(item)

    print(f"\n  Combined results:")
    print(f"    Passed all layers: {len(combined_passed)}")
    print(f"    Failed any layer:  {len(combined_failed)}")

    if fail_reasons:
        print(f"\n  Failures by layer:")
        for layer, cnt in fail_reasons.most_common():
            print(f"    {layer:<20s}  {cnt:>6,d}")

    # Per-tier survival (v2: tier1 = O+T, tier2 = O+T+C, tier3 = O+T+C×2)
    print(f"\n  Survival by tier:")
    all_tier = Counter(all_items[qid].get("tier", "?")
                       for qid in item_results)
    pass_tier = Counter(it.get("tier", "?") for it in combined_passed)
    for tier in ["tier1", "tier2", "tier3"]:
        inp = all_tier.get(tier, 0)
        out = pass_tier.get(tier, 0)
        rate = out / inp * 100 if inp else 0
        print(f"    {tier:<12s}  {out:>6,d} / {inp:>6,d}  ({rate:.1f}%)")

    # Per-dim survival
    print(f"\n  Survival by dim:")
    all_dim = Counter(all_items[qid].get("dim", "?")
                      for qid in item_results)
    pass_dim = Counter(it.get("dim", "?") for it in combined_passed)
    for dim in sorted(all_dim, key=lambda d: -all_dim[d]):
        inp = all_dim[dim]
        out = pass_dim.get(dim, 0)
        rate = out / inp * 100 if inp else 0
        c_mark = "★" if DIMENSIONS.get(dim, {}).get("c_critical") else " "
        print(f"    {c_mark} {dim:<28s}  {out:>6,d} / {inp:>6,d}  ({rate:.1f}%)")

    # Per-format survival
    print(f"\n  Survival by format:")
    all_fmt = Counter(all_items[qid].get("format", "?")
                      for qid in item_results)
    pass_fmt = Counter(it.get("format", "?") for it in combined_passed)
    for fmt in sorted(all_fmt):
        inp = all_fmt[fmt]
        out = pass_fmt.get(fmt, 0)
        rate = out / inp * 100 if inp else 0
        print(f"    {fmt:<20s}  {out:>6,d} / {inp:>6,d}  ({rate:.1f}%)")

    # Save
    output = {
        "layers_used": layers_available,
        "total_input": len(item_results),
        "total_passed": len(combined_passed),
        "total_failed": len(combined_failed),
        "pass_rate": round(
            len(combined_passed) / max(1, len(item_results)) * 100, 1
        ),
        "fail_reasons_by_layer": dict(fail_reasons),
        "qa_items": combined_passed,
    }
    out_path = filter_dir / "_combined.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}")

    return output


# ============================================================
# Main
# ============================================================

def _collect_existing_results(layer, filter_dir):
    """
    Scan the filter directory for ALL result files of this layer
    (both the final file and any shard files) and collect completed
    qa_id → item mappings. This means results from ANY previous
    shard configuration are recognized.
    """
    layer_cfg = LAYER_CONFIGS[layer]
    result_key = f"layer{list(LAYER_CONFIGS.keys()).index(layer)+1}_result"
    base_name = layer_cfg["output_file"].replace(".json", "")

    existing = {}  # qa_id → item

    # Scan: main file + any shard files
    candidates = list(filter_dir.glob(f"{base_name}*.json"))
    for path in candidates:
        try:
            with open(path) as f:
                data = json.load(f)
            for it in data.get("items", []):
                if result_key in it and it.get("qa_id"):
                    existing[it["qa_id"]] = it
        except Exception:
            continue

    return existing, result_key


def _merge_all(layer, filter_dir, all_items_ordered):
    """
    Merge ALL shard files + main file for this layer into the final
    output, ordered to match all_items_ordered (the full item list).
    """
    existing, result_key = _collect_existing_results(layer, filter_dir)
    layer_cfg = LAYER_CONFIGS[layer]

    # Build merged list in original order
    merged = []
    missing = 0
    for it in all_items_ordered:
        qid = it["qa_id"]
        if qid in existing:
            merged.append(existing[qid])
        else:
            missing += 1

    n_passed = sum(1 for it in merged
                   if it.get(result_key, {}).get("passed", True))
    n_failed = len(merged) - n_passed

    output_path = filter_dir / layer_cfg["output_file"]
    with open(output_path, "w") as f:
        json.dump({
            "layer": layer,
            "description": layer_cfg["description"],
            "total_items": len(merged),
            "passed": n_passed,
            "failed": n_failed,
            "pass_rate": round(n_passed / max(1, len(merged)) * 100, 1),
            "items": merged,
        }, f, indent=2)

    print(f"\n  Merged → {output_path}")
    print(f"  Total: {len(merged):,}  "
          f"Passed: {n_passed:,}  Failed: {n_failed:,}  "
          f"({n_passed/max(1,len(merged))*100:.1f}% pass rate)")
    if missing:
        print(f"  [WARN] {missing:,} items still missing — "
              f"run more shards to complete")

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="TOC-Bench Step 4: Temporal necessity filter"
    )
    parser.add_argument(
        "--layer", type=str, default=None,
        choices=["text_only", "single_frame", "shuffled"],
        help="Which layer to run"
    )
    parser.add_argument("--combine", action="store_true",
                        help="Combine existing layer results")
    parser.add_argument("--merge", action="store_true",
                        help="Merge all shard files for this layer into "
                             "the final layer file")
    parser.add_argument("--model", type=str, default=None,
                        help="Qwen3-VL model path (default: from config)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dry-run", action="store_true",
                        help="Random answers, no model loading")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="Ignore existing results, rerun everything")

    # Shard support: slice the REMAINING (not-yet-completed) items
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Split remaining TODO items into N shards")
    parser.add_argument("--shard-id", type=int, default=0,
                        help="Which shard of the TODO items to run (0-indexed)")

    # Checkpoint: persist partial shard state every N items so Ctrl-C
    # doesn't lose progress.
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="Write partial shard state every N items "
                             "(default: 100). Set to 0 to disable.")

    args = parser.parse_args()

    print("=" * 70)
    print("  TOC-Bench Step 4: Temporal Necessity Filter")
    print("=" * 70)

    FILTER_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # ---- Combine mode ----
    if args.combine:
        combine_layers(FILTER_DIR)
        return

    if args.layer is None:
        print("  [ERROR] Specify --layer or --combine")
        sys.exit(1)

    layer = args.layer
    layer_cfg = LAYER_CONFIGS[layer]
    print(f"  Layer: {layer} — {layer_cfg['description']}")

    # ---- Load ALL QA items ----
    input_path = Path(args.input) if args.input else QA_ITEMS_DIR / "_all.json"
    if not input_path.exists():
        print(f"  [ERROR] {input_path} not found. Run step3c first.")
        sys.exit(1)

    with open(input_path) as f:
        data = json.load(f)
    all_items = data.get("qa_items", [])
    print(f"  Loaded {len(all_items):,} QA items")

    before = len(all_items)
    all_items = [it for it in all_items if it.get("verification_passed", True)]
    if len(all_items) < before:
        print(f"  Skipped {before - len(all_items):,} verification-failed"
              f" → {len(all_items):,}")

    if args.max_items:
        all_items = all_items[:args.max_items]
        print(f"  Limited to {len(all_items):,}")

    # ---- Merge mode: just merge all existing shard files ----
    if args.merge:
        _merge_all(layer, FILTER_DIR, all_items)
        return

    # ---- Collect ALL existing results (from any previous shard config) ----
    if args.overwrite:
        existing_results = {}
        result_key = f"layer{list(LAYER_CONFIGS.keys()).index(layer)+1}_result"
        print(f"  --overwrite: ignoring all previous results")
    else:
        existing_results, result_key = _collect_existing_results(
            layer, FILTER_DIR)
        if existing_results:
            print(f"  Found {len(existing_results):,} completed items "
                  f"across all shard files")

    # ---- Split into done vs todo ----
    done_items = []
    todo_items = []
    for it in all_items:
        if it["qa_id"] in existing_results:
            done_items.append(existing_results[it["qa_id"]])
        else:
            todo_items.append(it)

    print(f"  Done: {len(done_items):,}  "
          f"TODO: {len(todo_items):,}  "
          f"Total: {len(all_items):,}")

    if not todo_items:
        print(f"  All items completed! Run --merge to produce final file, "
              f"or --overwrite to rerun.")
        return

    # ---- Shard the TODO items (not the full list!) ----
    if args.num_shards > 1:
        total_todo = len(todo_items)
        shard_size = (total_todo + args.num_shards - 1) // args.num_shards
        start = args.shard_id * shard_size
        end = min(start + shard_size, total_todo)
        todo_items = todo_items[start:end]
        print(f"  Shard {args.shard_id}/{args.num_shards}: "
              f"TODO[{start}:{end}] → {len(todo_items):,} items this run")

    if not todo_items:
        print(f"  This shard has no items to process.")
        return

    # Format breakdown
    fmt_counts = Counter(it.get("format", "?") for it in todo_items)
    print(f"  Format: "
          + ", ".join(f"{fmt}={cnt}" for fmt, cnt in fmt_counts.most_common()))

    # ---- Load engine ----
    engine = Qwen3VLEngine(
        model_name=args.model,
        device=args.device,
        mock=args.dry_run,
    )

    # ---- Compute shard file path BEFORE running so checkpoint can write ----
    if args.num_shards > 1:
        base_name = layer_cfg["output_file"].replace(".json", "")
        shard_file = FILTER_DIR / f"{base_name}_shard{args.shard_id}.json"
    else:
        shard_file = FILTER_DIR / layer_cfg["output_file"]

    # If shard file already has results from previous runs, load them so
    # checkpoint writes preserve them too
    prev_in_shard = {}
    if shard_file.exists() and not args.overwrite:
        try:
            with open(shard_file) as f:
                prev = json.load(f)
            for it in prev.get("items", []):
                if result_key in it:
                    prev_in_shard[it["qa_id"]] = it
        except Exception:
            pass

    # ---- Build checkpoint fn (writes prev_in_shard + new_results atomically
    # every N items) ----
    checkpoint_fn = None
    if args.checkpoint_every > 0:
        checkpoint_fn = make_checkpoint_fn(
            shard_file=shard_file,
            layer=layer,
            layer_desc=layer_cfg["description"],
            done_items=done_items,  # kept for signature compat; not used inside
            prev_in_shard=prev_in_shard,
            result_key=result_key,
        )
        if prev_in_shard:
            print(f"  Checkpoint: every {args.checkpoint_every} items "
                  f"(shard file has {len(prev_in_shard)} pre-existing items)")
        else:
            print(f"  Checkpoint: every {args.checkpoint_every} items")

    # ---- Run layer ----
    t_start = time.time()

    if layer == "text_only":
        new_results = run_text_only(
            todo_items, engine,
            checkpoint_fn=checkpoint_fn,
            checkpoint_every=args.checkpoint_every,
        )
    elif layer == "single_frame":
        new_results = run_single_frame(
            todo_items, engine, rng,
            checkpoint_fn=checkpoint_fn,
            checkpoint_every=args.checkpoint_every,
        )
    elif layer == "shuffled":
        n_frames = layer_cfg.get("num_frames", 16)
        delta = layer_cfg.get("delta_threshold", 0.05)
        new_results = run_shuffled(
            todo_items, engine, rng,
            n_frames=n_frames, delta_threshold=delta,
            checkpoint_fn=checkpoint_fn,
            checkpoint_every=args.checkpoint_every,
        )

    elapsed = time.time() - t_start

    # ---- Final (post-run) write of the shard file ----
    # This is a safety net — if the last checkpoint was <checkpoint_every items
    # ago, some results may not be on disk yet.

    # Merge: previous shard results + new results
    new_by_id = {it["qa_id"]: it for it in new_results}
    prev_in_shard.update(new_by_id)  # new overwrites old for same qa_id
    shard_items = list(prev_in_shard.values())

    n_passed = sum(1 for it in shard_items
                   if it.get(result_key, {}).get("passed", True))
    n_failed = len(shard_items) - n_passed

    _atomic_write_json(shard_file, {
        "layer": layer,
        "description": layer_cfg["description"],
        "total_items": len(shard_items),
        "passed": n_passed,
        "failed": n_failed,
        "pass_rate": round(n_passed / max(1, len(shard_items)) * 100, 1),
        "elapsed_seconds": round(elapsed, 1),
        "items": shard_items,
    })

    # ---- Report ----
    total_done_now = len(done_items) + len(new_results)
    total_all = len(all_items)

    print(f"\n{'='*70}")
    print(f"  Layer {layer} — This Run")
    print(f"{'='*70}")
    print(f"  Processed:  {len(new_results):,} items in {elapsed:.0f}s"
          f"  ({elapsed/max(1,len(new_results)):.2f}s/item)")
    print(f"  Saved to:   {shard_file.name}")
    print(f"  Progress:   {total_done_now:,} / {total_all:,}"
          f"  ({total_done_now/max(1,total_all)*100:.1f}%)")
    remaining = total_all - total_done_now
    if remaining > 0:
        print(f"  Remaining:  {remaining:,} — rerun same command to continue")
    else:
        print(f"  ALL DONE — run --merge to produce final layer file")


if __name__ == "__main__":
    main()