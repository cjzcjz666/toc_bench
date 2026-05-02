#!/usr/bin/env python3
"""
TOC-Bench Evaluation Runner
============================
Runs a single model over the benchmark and saves raw predictions.

Architecture:
  - All visual-language models go through an OpenAI-compatible client.
    Local models are served by vLLM (`vllm serve <model>`) and Anthropic-/
    OpenAI-/Google-hosted ones use their own base URLs.
  - Pure-logic baselines (random, frequency, always_a, text_only_*) bypass
    the HTTP layer.
  - Per-format prompts are fixed and identical across models — never
    model-tuned (TempCompass / TVBench protocol).

Input:
    --bench    path to toc_bench_clean_v2.json (from data_repair.py)
    --model    one of the keys in MODEL_REGISTRY (see below)
    --videos-root  root dir containing video files (we resolve
                   video_id → file path using a registry or convention)
    --frames    how many uniform frames to sample (default 32)

Output:
    --out      JSONL with one prediction per line:
               {qa_id, model, format, raw_response, extracted_answer,
                latency_s, error}

Usage examples:
    # Pure baseline (no GPU/API needed):
    python eval_runner.py --model random --bench .../toc_bench_v2.json \\
                          --out preds_random.jsonl

    # Local vLLM (start `vllm serve Qwen/Qwen2.5-VL-7B-Instruct` first):
    python eval_runner.py --model qwen2.5-vl-7b \\
        --bench .../toc_bench_v2.json --videos-root /data/videos \\
        --frames 32 --out preds_qwen7b.jsonl --concurrency 8

    # Closed-source API (set ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY):
    python eval_runner.py --model gpt-5 \\
        --bench .../toc_bench_v2.json --videos-root /data/videos \\
        --frames 32 --out preds_gpt5.jsonl --concurrency 4

The runner is RESUMABLE — it skips qa_ids already present in --out.

Hugging Face cache: set HF_HOME (e.g. export HF_HOME=/root/autodl-tmp/cache)
for both this script and `hf download` so weights land under HF_HOME/hub once.
"""

import argparse
import base64
import json
import os
import random
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# Local
sys.path.insert(0, str(Path(__file__).resolve().parent))
from answer_extractor import extract


def _normalize_hf_cache_env() -> None:
    """Single cache root: HF_HOME/hub for huggingface_hub and transformers.

    Shell templates often set TRANSFORMERS_CACHE to .../huggingface/hub while
    `hf download` uses another HF_HOME, which re-downloads weights. Drop legacy
    overrides so both use HF_HOME (default /root/autodl-tmp/cache on this bench).
    """
    default_home = "/root/autodl-tmp/cache"
    raw = os.environ.get("HF_HOME", default_home)
    home = os.path.expandvars(os.path.expanduser(str(raw).rstrip("/\\")))
    os.environ["HF_HOME"] = home
    for key in (
        "TRANSFORMERS_CACHE",
        "PYTORCH_TRANSFORMERS_CACHE",
        "PYTORCH_PRETRAINED_BERT_CACHE",
        "HUGGINGFACE_HUB_CACHE",
    ):
        os.environ.pop(key, None)


_normalize_hf_cache_env()


# =============================================================================
# Model Registry
# =============================================================================
# Each model gets a config dict. The KIND determines code path:
#   "openai_video"   — OpenAI-compatible chat API with video input. Used for
#                      vLLM-served local models AND OpenAI/Anthropic-hosted ones.
#                      Frames are sent as base64-encoded image_url parts.
#   "openai_text"    — same API, but skip frames (text-only baseline).
#   "google_video"   — Google Gemini API; uses google-genai SDK with native
#                      video upload. (Different SDK shape than OpenAI.)
#   "logic"          — Pure logic baseline; no network, no GPU.
#
# Local vLLM services are started SEPARATELY by the user. The runner just
# points base_url at the right endpoint.
# =============================================================================

MODEL_REGISTRY = {
       # =========================================================================
    # A. Closed-source / API models (9)
    # =========================================================================
    "gemini-3.1-pro-preview": {
        "kind": "openai_video",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model_id": "google/gemini-3.1-pro-preview",
        "frames_default": 32,
        "category": "closed_sota",
    },
    "gemini-3.1-flash-lite-preview": {
        "kind": "openai_video",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model_id": "google/gemini-3.1-flash-lite-preview",
        "frames_default": 32,
        "category": "closed_sota",
    },
    "gpt-5.5": {
        "kind": "openai_video",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model_id": "openai/gpt-5.5",
        "frames_default": 32,
        "category": "closed_sota",
    },
    "gpt-5.4-mini": {
        "kind": "openai_video",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model_id": "openai/gpt-5.4-mini",
        "frames_default": 32,
        "category": "closed_sota",
    },
    "claude-opus-4.7": {
        "kind": "openai_video",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model_id": "anthropic/claude-opus-4.7",
        "frames_default": 32,
        "category": "closed_sota",
    },
    "mimo-v2.5-pro": {
        "kind": "openai_video",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model_id": "xiaomi/mimo-v2.5-pro",
        "frames_default": 32,
        "category": "closed_sota",
    },
    "kimi-k2.6": {
        "kind": "openai_video",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model_id": "moonshotai/kimi-k2.6",
        "frames_default": 32,
        "category": "closed_sota",
        # See get_max_tokens: reasoning uses completion budget; tiny cap => null content.
        "split_reasoning_completion": True,
    },
    "glm-5v-turbo": {
        "kind": "openai_video",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model_id": "z-ai/glm-5v-turbo",
        "frames_default": 32,
        "category": "closed_sota",
    },
    "nemotron-3-nano-omni-30b-a3b-reasoning-free": {
        "kind": "openai_video",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model_id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "frames_default": 32,
        "category": "closed_sota",
        "is_thinking": True,
    },

    # =========================================================================
    # B. Open-source generalist (6)
    # =========================================================================
    "qwen2.5-vl-72b": {
        "kind": "openai_video",
        "base_url": "http://localhost:8001/v1",
        "api_key_env": None,
        "model_id": "Qwen/Qwen2.5-VL-72B-Instruct",
        "frames_default": 32,
        "category": "open_generalist",
    },
    "qwen2.5-vl-7b": {
        "kind": "openai_video",
        "base_url": "http://localhost:8002/v1",
        "api_key_env": None,
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "frames_default": 32,
        "category": "open_generalist",
    },
    "qwen3-vl-8b": {
        "kind": "hf_video",
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "frames_default": 32,
        "category": "open_generalist",
        "note": (
            "Local weights. Direct HF transformers (not vLLM): vLLM v0.20.0 "
            "has a known deepstack-token off-by-one bug with Qwen3-VL "
            "(`Requested more deepstack tokens than available in buffer`). "
            "Transformers inference avoids this. "
            "Cross-generation comparison vs Qwen2.5-VL-7B."
        ),
    },

    # =========================================================================
    # F. Reasoning / thinking models (3) — emit <think>...</think> CoT
    # before the final answer. Get reported separately in paper Section 5.X.
    # =========================================================================
    "qwen3-vl-8b-thinking": {
        "kind": "hf_video",
        "model_id": "Qwen/Qwen3-VL-8B-Thinking",
        "frames_default": 32,
        "category": "reasoning",
        "is_thinking": True,
        "note": (
            "Independent weights from Qwen3-VL-8B-Instruct (~16GB, separate "
            "download). Outputs <think>...</think> CoT followed by answer. "
            "We strip the thinking block before answer extraction. "
            "max_new_tokens auto-bumped to 4096 for reasoning trace."
        ),
    },
    "video-r1-7b": {
        "kind": "openai_video",
        "base_url": "http://localhost:8014/v1",
        "api_key_env": None,
        "model_id": "Video-R1/Video-R1-7B",
        "frames_default": 32,
        "category": "reasoning",
        "is_thinking": True,
        "note": (
            "First R1-style video reasoning model (Feng et al. 2025). "
            "Qwen2.5-VL-7B backbone + RL post-training with T-GRPO. "
            "vLLM-compatible (Qwen2.5-VL architecture). "
            "Start: vllm serve Video-R1/Video-R1-7B --port 8014, or serve a "
            "local snapshot with --served-model-name Video-R1/Video-R1-7B. "
            "If vLLM uses another name, set env VIDEO_R1_VLLM_MODEL_ID to match."
        ),
    },
    "videochat-r1.5-7b": {
        "kind": "openai_video",
        "base_url": "http://localhost:8015/v1",
        "api_key_env": None,
        "model_id": "OpenGVLab/VideoChat-R1_5-7B",
        "frames_default": 32,
        "category": "reasoning",
        "is_thinking": True,
        "note": (
            "VideoChat-R1.5 (Wang et al. 2025-09, NeurIPS 2025). "
            "Qwen2.5-VL-7B backbone + iterative perception + RL. "
            "vLLM-compatible. Start: vllm serve OpenGVLab/VideoChat-R1_5-7B --port 8015"
        ),
    },
    "internvl3-78b": {
        "kind": "hf_video",
        "model_id": "OpenGVLab/InternVL3-78B-hf",
        "frames_default": 32,
        "category": "open_generalist",
        "note": (
            "Use the -hf variant (HF-team-maintained transformers wrapper). "
            "vLLM serve breaks with the non-hf variant due to chat-template "
            "issues with multi-image content lists. The -hf variant uses "
            "standard AutoModelForImageTextToText."
        ),
    },
    "internvl3-8b": {
        "kind": "hf_video",
        "model_id": "OpenGVLab/InternVL3-8B",
        "frames_default": 32,
        "category": "open_generalist",
        "note": (
            "Native (non-hf) variant. Uses InternVL's custom model.chat() API. "
            "Auto-dispatched to _load_internvl3_native based on missing -hf "
            "suffix. If you want HF-standard transformers loading, switch "
            "model_id to 'OpenGVLab/InternVL3-8B'."
        ),
    },
    "minicpm-v-2.6": {
        "kind": "openai_video",
        "base_url": "http://localhost:8006/v1",
        "api_key_env": None,
        "model_id": "openbmb/MiniCPM-V-2_6",
        "frames_default": 32,
        "category": "open_generalist",
        "note": "Lightweight 8B; common in benchmarks for edge-deployment scenarios",
    },

    # =========================================================================
    # C. Open-source video specialists (4) — TOC-Bench's natural opponents
    # =========================================================================
    "llava-video-72b": {
        "kind": "hf_video",
        "model_id": "lmms-lab/LLaVA-Video-72B-Qwen2",
        "frames_default": 32,
        "category": "open_video_specialist",
        "note": (
            "Requires LLaVA-NeXT repo: "
            "pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git"
        ),
    },
    "llava-video-7b": {
        "kind": "hf_video",
        "model_id": "lmms-lab/LLaVA-Video-7B-Qwen2",
        "frames_default": 32,
        "category": "open_video_specialist",
        "note": (
            "Requires LLaVA-NeXT repo: "
            "pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git"
        ),
    },
    "videollama3-7b": {
        "kind": "hf_video",
        "model_id": "DAMO-NLP-SG/VideoLLaMA3-7B",
        "frames_default": 32,
        "category": "open_video_specialist",
        "note": (
            "DAMO-NLP-SG/VideoLLaMA3-7B. Uses custom modeling code "
            "(trust_remote_code=True). vLLM does not support its custom "
            "config / vision tokens. transformers-only."
        ),
    },

    # =========================================================================
    # D. Classic / historical reference (2)
    # =========================================================================
    "video-llava-7b": {
        "kind": "hf_video",
        "model_id": "LanguageBind/Video-LLaVA-7B-hf",
        "frames_default": 8,
        "category": "classic_baseline",
        "note": (
            "2023 baseline, 8-frame native sampling. Uses standard HF "
            "VideoLlavaForConditionalGeneration. vLLM does not support "
            "this old architecture; transformers-only."
        ),
    },
    "video-chatgpt-7b": {
        "kind": "hf_video",
        "model_id": "MBZUAI/Video-ChatGPT-7B",
        "frames_default": 100,  # Video-ChatGPT samples 100 frames natively
        "category": "classic_baseline",
        "note": (
            "2023 baseline. Requires the original repo: "
            "git clone https://github.com/mbzuai-oryx/Video-ChatGPT.git && "
            "cd Video-ChatGPT && pip install -e . "
            "Recommend a separate conda env (old dependency stack)."
        ),
    },

    # =========================================================================
    # E. Ablation baselines (4)
    # =========================================================================
    "llava-ov-7b-1frame": {
        "kind": "hf_video",
        "model_id": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "frames_default": 1,
        "category": "ablation",
        "note": (
            "1-frame image-only ablation — exposes spatial-only shortcut. "
            "Uses standard HF LlavaOnevisionForConditionalGeneration with "
            "the middle frame only. transformers-only."
        ),
    },
    "gpt-4o-text": {
        "kind": "openai_text",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model_id": "gpt-4o",
        "category": "ablation",
        "note": "Text-only — exposes language-prior shortcut.",
    },
    "random": {"kind": "logic", "logic_kind": "random", "category": "ablation"},
    "frequency": {"kind": "logic", "logic_kind": "frequency", "category": "ablation"},

    # Always-A is kept for diagnostic but not in the main paper table:
    "always_a": {"kind": "logic", "logic_kind": "always_a", "category": "diagnostic"},
}


# =============================================================================
# Frame sampling (uniform sub-sampling)
# =============================================================================

# Cached video registry (loaded once)
_VIDEO_REGISTRY_CACHE = {}


def load_video_registry(registry_path):
    """Load video_registry.json once and return dict {video_id → entry}.
    Each entry has fields: video_id, source, path, format ('video_file' or 'jpeg_folder')."""
    if registry_path in _VIDEO_REGISTRY_CACHE:
        return _VIDEO_REGISTRY_CACHE[registry_path]
    with open(registry_path) as f:
        data = json.load(f)
    if isinstance(data, list):
        d = {r['video_id']: r for r in data}
    else:
        d = data
    _VIDEO_REGISTRY_CACHE[registry_path] = d
    return d


def resolve_video_path(video_id, videos_root, registry=None):
    """Find the actual video file or jpeg-folder for a given video_id.

    Lookup order:
      1. If a registry dict is provided, use its `path` field (and remember
         whether it's a video file or jpeg folder via `format`).
      2. Fallback: try videos_root/<source>/<file>.{mp4,webm,mov} or as a
         directory of JPEG frames.

    Returns (Path, kind) where kind is 'video_file' or 'jpeg_folder', or
    (None, None) if not found.
    """
    # Registry path is canonical
    if registry is not None and video_id in registry:
        entry = registry[video_id]
        p = Path(entry['path'])
        if p.exists():
            return p, entry.get('format', 'video_file')
        # If absolute path doesn't exist, try resolving relative to videos_root
        if videos_root:
            relp = videos_root / entry['path'].lstrip('/')
            if relp.exists():
                return relp, entry.get('format', 'video_file')

    # Fallback: convention-based resolution
    if not videos_root:
        return None, None
    parts = video_id.split('_', 1)
    if len(parts) == 2:
        source, fname = parts
        for ext in ('.mp4', '.webm', '.mov'):
            cand = videos_root / source / f"{fname}{ext}"
            if cand.exists():
                return cand, 'video_file'
        cand = videos_root / source / fname
        if cand.exists() and cand.is_dir():
            return cand, 'jpeg_folder'
    for ext in ('.mp4', '.webm', '.mov'):
        cand = videos_root / f"{video_id}{ext}"
        if cand.exists():
            return cand, 'video_file'
    return None, None


def sample_frames_uniform(video_path, n_frames, kind='video_file'):
    """Sample n_frames uniformly from a video file or jpeg-folder.

    kind: 'video_file' (mp4/webm/mov decoded with cv2) or 'jpeg_folder'
          (folder of pre-extracted jpegs from MOSE/OVIS).
    Returns list of base64-encoded JPEG strings.
    """
    try:
        import cv2
    except ImportError:
        raise RuntimeError("OpenCV (cv2) required for video frame sampling")

    video_path = Path(video_path)

    if kind == 'jpeg_folder' or video_path.is_dir():
        # JPEG folder (MOSE/OVIS style)
        frames = sorted([f for f in video_path.iterdir()
                         if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
        if not frames:
            return []
        if len(frames) <= n_frames:
            picked_paths = frames
        else:
            indices = [int(i * len(frames) / n_frames) for i in range(n_frames)]
            picked_paths = [frames[i] for i in indices]
        b64s = []
        for p in picked_paths:
            with open(p, 'rb') as f:
                b64s.append(base64.b64encode(f.read()).decode('ascii'))
        return b64s

    # MP4 / WebM / MOV
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    if total <= n_frames:
        indices = list(range(total))
    else:
        indices = [int(i * total / n_frames) for i in range(n_frames)]

    b64s = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        # Resize so longer side ≤ 720
        h, w = frame.shape[:2]
        if max(h, w) > 720:
            scale = 720 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            continue
        b64s.append(base64.b64encode(buf.tobytes()).decode('ascii'))
    cap.release()
    return b64s


# =============================================================================
# Prompt builder (per format) — IDENTICAL across all models
# =============================================================================

def build_prompt(item):
    """Return (system_text, user_text) for the question. Frames added separately."""
    fmt = item["format"]
    q = item["question"]

    if fmt == "mcq_4":
        user = (
            "You will see a video and a multiple-choice question with four "
            "options. Choose the BEST option. Output ONLY the letter A, B, "
            "C, or D.\n\n"
            f"Question: {q}\n"
            f"A. {item['option_A']}\n"
            f"B. {item['option_B']}\n"
            f"C. {item['option_C']}\n"
            f"D. {item['option_D']}\n"
            "Answer:"
        )
    elif fmt == "sp":
        a = item.get("option_A") or item.get("statement_A")
        b = item.get("option_B") or item.get("statement_B")
        user = (
            "You will see a video and two statements. Choose the statement "
            "that is TRUE according to the video. Output ONLY the letter A "
            "or B.\n\n"
            f"Question: {q}\n"
            f"A. {a}\n"
            f"B. {b}\n"
            "Answer:"
        )
    elif fmt == "numerical":
        user = (
            "You will see a video and a counting question. Output ONLY a "
            "single integer (e.g., 2, 3, 4) or the phrase \"5 or more\". "
            "No other words.\n\n"
            f"Question: {q}\n"
            "Answer:"
        )
    elif fmt.startswith("ordering"):
        events = item["events"]
        opts = "\n".join(f"{e['label']}. {e['event_text']}" for e in events)
        user = (
            "You will see a video and a list of events. Output the events "
            "in chronological order from earliest to latest, as a comma-"
            "separated list of letters (e.g., \"B, A, C\" or \"B, A, C, D\")."
            " No other words.\n\n"
            f"Question: {q}\n"
            f"{opts}\n"
            "Answer:"
        )
    else:
        raise ValueError(f"Unknown format: {fmt}")

    system = "You are a video understanding model. Follow output format exactly."
    return system, user


def get_max_tokens(fmt, cfg=None):
    """Token budget per request.

    Standard models: 64 (mcq/sp/numerical) or 256 (ordering).
    Thinking models: 4096 to accommodate <think>...</think> reasoning trace
    plus the final answer. The reasoning trace can be 200-2000 tokens.

    OpenRouter / Kimi-style models bill reasoning in a separate stream (`reasoning` /
    `reasoning_details`); a small max_tokens can exhaust the budget before any
    `message.content` is emitted, leaving content null.
    """
    is_thinking = bool(cfg and cfg.get("is_thinking"))
    split_reasoning = bool(cfg and cfg.get("split_reasoning_completion"))
    if is_thinking or split_reasoning:
        return 18432
    return 256 if fmt.startswith("ordering") else 64


def _norm_text(s):
    """Lowercase + remove punctuation for fuzzy option matching."""
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_with_item_fallback(raw, item):
    """Primary extractor + context-aware fallback for verbose legacy outputs."""
    fmt = item["format"]
    ans = extract(raw, fmt)
    if ans is not None:
        return ans
    if not raw:
        return None

    raw_n = _norm_text(raw)

    # Fallback 1: numerical word forms ("three times", "twice", etc.)
    if fmt == "numerical":
        if re.search(r"\b(5\s*or\s*more|five\s+or\s+more|at\s+least\s+five)\b", raw_n):
            return "5 or more"
        word_to_num = {
            "two": 2, "twice": 2,
            "three": 3, "thrice": 3,
            "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        }
        for w, n in word_to_num.items():
            if re.search(rf"\b{w}\b", raw_n):
                if n in (2, 3, 4):
                    return str(n)
                if n >= 5:
                    return "5 or more"

    # Fallback 2: statement/option text matching for SP and MCQ.
    if fmt in ("sp", "mcq_4"):
        letters = ["A", "B"] if fmt == "sp" else ["A", "B", "C", "D"]
        matches = []
        for L in letters:
            opt = item.get(f"option_{L}") or item.get(f"statement_{L}")
            if not opt:
                continue
            opt_n = _norm_text(opt)
            if not opt_n:
                continue
            # Strong overlap in either direction.
            if opt_n in raw_n or raw_n in opt_n:
                matches.append(L)
                continue
            # Token overlap score for paraphrase-ish outputs.
            raw_toks = set(raw_n.split())
            opt_toks = set(opt_n.split())
            if not opt_toks:
                continue
            overlap = len(raw_toks & opt_toks) / len(opt_toks)
            if overlap >= 0.7:
                matches.append(L)
        if len(matches) == 1:
            return matches[0]

    return None


# =============================================================================
# Adapter: openai-compatible video chat (works for vLLM, GPT, Anthropic-OpenAI)
# =============================================================================

def _openai_assistant_visible_text(message) -> str:
    """Visible assistant text from OpenAI-style `message` (not chain-of-thought).

    Some providers put the user-facing answer only in `content`; others may use
    a list of content parts. We intentionally do not use `reasoning` here.
    """
    c = getattr(message, "content", None)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for block in c:
            if isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            else:
                t = getattr(block, "text", None)
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return ""


def call_openai_video(cfg, item, frames_b64, max_tokens):
    """Build OpenAI-style chat call with frames as image_url base64.

    If cfg['strict_numerical'] is True AND the item is numerical, attaches
    guided_choice constraint via vLLM extra_body to force the model to
    output exactly one of the legal answer strings.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("Install: pip install openai")

    api_key = (os.environ.get(cfg["api_key_env"])
               if cfg.get("api_key_env") else "EMPTY")
    client = OpenAI(api_key=api_key, base_url=cfg["base_url"])

    system_text, user_text = build_prompt(item)

    # Build content: frames followed by text
    content = []
    for b64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
    content.append({"type": "text", "text": user_text})

    # Optional guided decoding for numerical (vLLM-only feature)
    extra_body = {}
    if cfg.get("strict_numerical") and item["format"] == "numerical":
        extra_body["guided_choice"] = ["2", "3", "4", "5 or more"]

    # vLLM served name may differ from HF repo id (e.g. local snapshot path).
    openai_model = cfg["model_id"]
    if cfg.get("_name") == "video-r1-7b":
        openai_model = os.environ.get("VIDEO_R1_VLLM_MODEL_ID", openai_model)

    kwargs = dict(
        model=openai_model,
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": content},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
        seed=42,
    )
    if extra_body:
        kwargs["extra_body"] = extra_body

    resp = client.chat.completions.create(**kwargs)
    raw = _openai_assistant_visible_text(resp.choices[0].message)
    # Strip <think>...</think> for thinking models
    if cfg.get("is_thinking"):
        raw = _strip_thinking(raw)
    return raw


def call_openai_text(cfg, item, frames_b64, max_tokens):
    """Text-only baseline: don't include any frames."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("Install: pip install openai")

    api_key = (os.environ.get(cfg["api_key_env"])
               if cfg.get("api_key_env") else "EMPTY")
    client = OpenAI(api_key=api_key, base_url=cfg["base_url"])

    system_text, user_text = build_prompt(item)
    resp = client.chat.completions.create(
        model=cfg["model_id"],
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
        seed=42,
    )
    raw = _openai_assistant_visible_text(resp.choices[0].message)
    if cfg.get("is_thinking"):
        raw = _strip_thinking(raw)
    return raw


# =============================================================================
# Adapter: Google Gemini (different SDK)
# =============================================================================

def call_google_video(cfg, item, frames_b64, max_tokens):
    """Gemini accepts inline image bytes; we pass each frame as
    inline_data parts."""
    try:
        from google import genai
        from google.genai import types as gtypes
    except ImportError:
        raise RuntimeError("Install: pip install google-genai")

    api_key = os.environ.get(cfg["api_key_env"])
    if not api_key:
        raise RuntimeError(f"Set env var {cfg['api_key_env']}")
    client = genai.Client(api_key=api_key)

    system_text, user_text = build_prompt(item)

    parts = []
    for b64 in frames_b64:
        parts.append(gtypes.Part.from_bytes(
            data=base64.b64decode(b64), mime_type="image/jpeg"))
    parts.append(gtypes.Part.from_text(text=user_text))

    resp = client.models.generate_content(
        model=cfg["model_id"],
        contents=[gtypes.Content(role="user", parts=parts)],
        config=gtypes.GenerateContentConfig(
            system_instruction=system_text,
            temperature=0.0,
            max_output_tokens=max_tokens,
        ),
    )
    return resp.text or ""


# =============================================================================
# Adapter: Anthropic (Claude) — different SDK, similar shape to OpenAI
# =============================================================================

def call_anthropic_video(cfg, item, frames_b64, max_tokens):
    """Claude accepts up to ~100 images per request via image content blocks
    using base64 source format."""
    try:
        from anthropic import Anthropic
    except ImportError:
        raise RuntimeError("Install: pip install anthropic")

    api_key = os.environ.get(cfg["api_key_env"])
    if not api_key:
        raise RuntimeError(f"Set env var {cfg['api_key_env']}")
    client = Anthropic(api_key=api_key)

    system_text, user_text = build_prompt(item)

    content = []
    for b64 in frames_b64:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": b64,
            },
        })
    content.append({"type": "text", "text": user_text})

    resp = client.messages.create(
        model=cfg["model_id"],
        max_tokens=max_tokens,
        temperature=0.0,
        system=system_text,
        messages=[{"role": "user", "content": content}],
    )
    # resp.content is a list of content blocks; concat text blocks
    return "".join(
        block.text for block in resp.content if hasattr(block, "text")
    ) or ""


# =============================================================================
# Adapter: Hugging Face transformers fallback (model-specific)
# =============================================================================
# Used when vLLM doesn't support a model (Video-LLaVA / Video-ChatGPT).
# Each model has its own loading + inference idiom — so this is a dispatch
# table by model_id rather than a single function.
#
# IMPORTANT: HF adapters are LAZY-loaded — model weights are only loaded
# the first time call_hf_video runs. The loaded model is cached in
# `_HF_MODEL_CACHE` keyed by model_id.
# =============================================================================

_HF_MODEL_CACHE = {}
def call_hf_video(cfg, item, frames_b64, max_tokens):
    """Dispatch to the right HF inference function for this model_id."""
    model_id = cfg["model_id"]

    # Lazy-load model
    if model_id not in _HF_MODEL_CACHE:
        print(f"  [HF] Loading {model_id} (one-time)...")
        if "Qwen3-VL" in model_id:
            _HF_MODEL_CACHE[model_id] = _load_qwen3_vl(model_id)
        elif "InternVL3" in model_id or "InternVL2" in model_id:
            # Two flavors: -hf suffix uses standard HF API,
            # native (no -hf) uses model.chat() custom interface
            if model_id.endswith("-hf"):
                _HF_MODEL_CACHE[model_id] = _load_internvl3(model_id)
            else:
                _HF_MODEL_CACHE[model_id] = _load_internvl3_native(model_id)
        elif "LLaVA-Video" in model_id:
            _HF_MODEL_CACHE[model_id] = _load_llava_video(model_id)
        elif "llava-onevision" in model_id.lower() or "llava-ov" in model_id.lower():
            _HF_MODEL_CACHE[model_id] = _load_llava_ov(model_id)
        elif "VideoLLaMA3" in model_id:
            _HF_MODEL_CACHE[model_id] = _load_videollama3(model_id)
        elif "Video-LLaVA" in model_id:
            _HF_MODEL_CACHE[model_id] = _load_video_llava(model_id)
        elif "Video-ChatGPT" in model_id:
            _HF_MODEL_CACHE[model_id] = _load_video_chatgpt(model_id)
        else:
            raise RuntimeError(
                f"No HF adapter for model_id={model_id}. "
                f"Add a _load_<model> function in eval_runner.py."
            )
        print(f"  [HF] Loaded {model_id}")

    obj = _HF_MODEL_CACHE[model_id]
    return obj["infer_fn"](item, frames_b64, max_tokens)


# ---- Per-model HF loaders ----
# These are placeholder skeletons. Each is filled in by the user (or by
# Claude on demand) when vLLM serve for that model fails. The structure
# is the same: load model + processor, return dict with `infer_fn`.

def _load_qwen3_vl(model_id):
    """Loader for Qwen3-VL (any variant, including local paths).

    Uses HF transformers directly. Qwen3-VL has a custom processor that
    handles image patching + temporal alignment. We pass frames as PIL
    Images via the chat-template messages format.

    Why this is a fallback: vLLM v0.20.0 has known deepstack token
    off-by-one issues with Qwen3-VL (`Requested more deepstack tokens
    than available in buffer`). Transformers inference avoids this.
    """
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
    except ImportError:
        raise RuntimeError(
            "Install: pip install -U transformers torch accelerate")
    try:
        from PIL import Image
        import io
    except ImportError:
        raise RuntimeError("Install: pip install pillow")

    print(f"  [HF] Loading processor for {model_id}...")
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True)

    print(f"  [HF] Loading model {model_id} (this may take a minute)...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Capture the closure
    def infer_fn(item, frames_b64, max_tokens):
        # Decode base64 frames into PIL images
        pil_frames = []
        for b64 in frames_b64:
            img_bytes = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            pil_frames.append(img)

        # Build chat-template messages — Qwen3-VL expects 'video' content
        # block as a list of images (not a real video file)
        system_text, user_text = build_prompt(item)
        messages = [
            {"role": "system", "content": system_text},
            {
                "role": "user",
                "content": [
                    # Each frame is its own image content block
                    *[{"type": "image", "image": img} for img in pil_frames],
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        # Process inputs
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=pil_frames,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        # Generate
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=(processor.tokenizer.pad_token_id
                              or processor.tokenizer.eos_token_id),
            )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        gen_ids = out_ids[0][input_len:]
        text_out = processor.tokenizer.decode(
            gen_ids, skip_special_tokens=True).strip()
        # Strip <think>...</think> blocks (Qwen3-VL-Thinking variant emits
        # chain-of-thought reasoning; we only want the final answer)
        text_out = _strip_thinking(text_out)
        return text_out

    return {"model": model, "processor": processor, "infer_fn": infer_fn}


def _strip_thinking(text):
    """Remove <think>...</think> blocks from model output.

    Qwen3-VL-Thinking, Video-R1, VideoChat-R1.5 all wrap chain-of-thought
    reasoning in <think> tags. The actual answer follows. We strip the
    thinking content so the answer extractor doesn't get confused by
    letters/numbers inside the reasoning trace.
    """
    if not text:
        return text
    import re
    # Remove complete <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove orphan opening tag if model ran out of tokens mid-thought
    # (truncation case): if there's still <think> with no </think>, drop
    # everything from <think> onward (no final answer was emitted)
    if "<think>" in text and "</think>" not in text:
        text = text.split("<think>")[0]
    return text.strip()


def _load_internvl3(model_id):
    """Loader for OpenGVLab/InternVL3-*-hf (and InternVL2_5-*-hf).

    InternVL3-hf uses the standard HF ImageTextToText API — same shape as
    Qwen3-VL. We pass frames as image content blocks.

    Why this fallback exists:
      - The non-`-hf` variants (e.g. `OpenGVLab/InternVL3-8B`) ship custom
        modeling code; vLLM's chat template breaks on multi-image content
        lists, throwing 'can only concatenate str (not list) to str'.
      - The HF variants (`-hf` suffix) are HF-team-maintained and use
        `AutoModelForImageTextToText` correctly.
    """
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
    except ImportError:
        raise RuntimeError(
            "Install: pip install -U transformers torch accelerate")
    try:
        from PIL import Image
        import io
    except ImportError:
        raise RuntimeError("Install: pip install pillow")

    print(f"  [HF] Loading processor for {model_id}...")
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True)

    print(f"  [HF] Loading model {model_id} (this may take a minute)...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    def infer_fn(item, frames_b64, max_tokens):
        # Decode base64 frames into PIL images
        pil_frames = []
        for b64 in frames_b64:
            img_bytes = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            pil_frames.append(img)

        system_text, user_text = build_prompt(item)
        # Use robust two-stage processing:
        # 1) render chat template to plain text,
        # 2) pack text + images with processor(...).
        # This avoids message schema mismatches that can raise
        # "TypeError: string indices must be integers" on some releases.
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in pil_frames],
                    {"type": "text", "text": f"{system_text}\n\n{user_text}"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = processor(
            text=[prompt],
            images=pil_frames,
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=(processor.tokenizer.pad_token_id
                              or processor.tokenizer.eos_token_id),
            )

        # Decode only newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        gen_ids = out_ids[0][input_len:]
        text_out = processor.tokenizer.decode(
            gen_ids, skip_special_tokens=True).strip()
        return text_out

    return {"model": model, "processor": processor, "infer_fn": infer_fn}


# ---- InternVL3 NATIVE (non-hf variants) helpers ----
# These are stripped from InternVL's official inference example
# (https://huggingface.co/OpenGVLab/InternVL3-8B). Required for the
# original `OpenGVLab/InternVL3-{1B,8B,14B,38B,78B}` checkpoints, which
# ship a custom `model.chat()` API rather than the standard HF generate.

_INTERNVL_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_INTERNVL_IMAGENET_STD = (0.229, 0.224, 0.225)


def _internvl_build_transform(input_size=448):
    from torchvision.transforms.functional import InterpolationMode
    import torchvision.transforms as T
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=_INTERNVL_IMAGENET_MEAN, std=_INTERNVL_IMAGENET_STD),
    ])


def _internvl_find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for r in target_ratios:
        target_ar = r[0] / r[1]
        ratio_diff = abs(aspect_ratio - target_ar)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = r
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * r[0] * r[1]:
                best_ratio = r
    return best_ratio


def _internvl_dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Tile a PIL image into 1-12 sub-patches following InternVL's recipe."""
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    best = _internvl_find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_w, orig_h, image_size)
    target_w = image_size * best[0]
    target_h = image_size * best[1]
    blocks = best[0] * best[1]
    resized_img = image.resize((target_w, target_h))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % best[0]) * image_size,
            (i // best[0]) * image_size,
            ((i % best[0]) + 1) * image_size,
            ((i // best[0]) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        thumbnail = image.resize((image_size, image_size))
        processed_images.append(thumbnail)
    return processed_images


def _load_internvl3_native(model_id):
    """Loader for original (non-hf) InternVL3 checkpoints.

    Uses InternVL's custom `model.chat()` API. Multi-image inference
    requires building num_patches_list and inserting <image> tokens
    in the prompt.
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        raise RuntimeError("Install: pip install -U transformers torch accelerate torchvision")
    try:
        from PIL import Image
        import io
    except ImportError:
        raise RuntimeError("Install: pip install pillow")

    print(f"  [HF-native] Loading InternVL3 (native) from {model_id}...")
    print(f"  [HF-native] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, use_fast=False)

    print(f"  [HF-native] Loading model...")
    # Avoid `use_flash_attn=True` — known issue #1080 with transformers 4.52+
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
    ).eval()

    transform = _internvl_build_transform(input_size=448)

    def infer_fn(item, frames_b64, max_tokens):
        # Decode frames and tile each one (max_num=1 means no sub-tiling
        # for video frames, since we already have 32 frames)
        pixel_values_list = []
        num_patches_list = []
        for b64 in frames_b64:
            img_bytes = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            # For video frames, use 1 patch per frame (no tiling)
            tiles = _internvl_dynamic_preprocess(
                img, min_num=1, max_num=1, image_size=448, use_thumbnail=False)
            for t in tiles:
                pixel_values_list.append(transform(t))
            num_patches_list.append(len(tiles))

        pixel_values = torch.stack(pixel_values_list).to(
            torch.bfloat16).cuda()

        # Build prompt: <image> placeholders followed by question
        system_text, user_text = build_prompt(item)
        # InternVL prompt convention: each <image> placeholder consumes
        # one entry from num_patches_list
        image_tokens = "".join(f"Frame{i+1}: <image>\n" for i in range(len(frames_b64)))
        question = f"{system_text}\n\n{image_tokens}{user_text}"

        generation_config = dict(
            max_new_tokens=max_tokens,
            do_sample=False,
        )

        with torch.no_grad():
            response = model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=False,
            )

        return response.strip() if response else ""

    return {"model": model, "tokenizer": tokenizer, "infer_fn": infer_fn}


def _load_llava_video(model_id):
    """Loader for lmms-lab/LLaVA-Video-{7B,72B}-Qwen2.

    LLaVA-Video has no -hf variant. It requires the LLaVA-NeXT repo:
        pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git

    The repo provides:
      - load_pretrained_model() to instantiate model + tokenizer + processor
      - process_images() to preprocess image tensors
      - tokenizer_image_token() to tokenize prompts with <image> placeholders
      - conv_templates['qwen_1_5'] for Qwen2-based variants

    Inference signature:
      model.generate(input_ids, images=video_tensor, modalities=['video'], ...)
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError("Install: pip install -U torch")
    try:
        from PIL import Image
        import io
    except ImportError:
        raise RuntimeError("Install: pip install pillow")
    try:
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates
        import copy
    except ImportError:
        raise RuntimeError(
            "LLaVA-Video requires the LLaVA-NeXT package. Install with:\n"
            "  pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git\n"
            "Note: this can take 5-10 minutes and may require dev tools."
        )

    print(f"  [HF] Loading LLaVA-Video from {model_id}...")
    model_name = "llava_qwen"  # Required by LLaVA-NeXT for Qwen-based variants
    try:
        import flash_attn  # noqa: F401
        _llava_attn_impl = "flash_attention_2"
    except ImportError:
        _llava_attn_impl = "sdpa"
        print("  [HF] flash_attn not installed; LLaVA-Video using sdpa attention.")
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_id,
        None,           # model_base
        model_name,
        torch_dtype="bfloat16",
        device_map="auto",
        attn_implementation=_llava_attn_impl,
    )
    model.eval()

    def infer_fn(item, frames_b64, max_tokens):
        # Decode base64 frames into PIL images
        pil_frames = []
        for b64 in frames_b64:
            img_bytes = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            pil_frames.append(img)

        # prepare_inputs_labels_for_multimodal expects images=[(T,C,H,W)] (4D per
        # clip). With the real config, anyres/highres yields (T,P,C,H,W); the list
        # branch then unsqueeze(0)s to 6D and vision conv fails. Force pad/square
        # preprocessing for video frames only.
        _prep_cfg = copy.deepcopy(model.config)
        _prep_cfg.image_aspect_ratio = "pad"
        video_tensor = process_images(pil_frames, image_processor, _prep_cfg)
        if isinstance(video_tensor, list):
            video_tensor = torch.stack(video_tensor)
        if video_tensor.ndim == 5 and video_tensor.shape[0] == 1:
            video_tensor = video_tensor.squeeze(0)
        if video_tensor.ndim != 4:
            raise RuntimeError(
                "LLaVA-Video preprocess expected (T, C, H, W) after pad path; "
                f"got {tuple(video_tensor.shape)}"
            )
        video_tensor = video_tensor.to(model.device, dtype=torch.bfloat16)

        # Build prompt using qwen_1_5 conversation template
        # (LLaVA-Video-7B/72B both use Qwen2 backbone)
        system_text, user_text = build_prompt(item)
        # LLaVA-Video uses a single <image> placeholder for the entire video,
        # not one per frame
        time_instruction = (
            f"The video lasts for {len(pil_frames)} frames. "
            f"Please answer the following question about the video."
        )
        full_question = (
            f"{DEFAULT_IMAGE_TOKEN}\n{time_instruction}\n\n"
            f"{system_text}\n\n{user_text}"
        )

        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], full_question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                images=[video_tensor],
                modalities=["video"],
                do_sample=False,
                max_new_tokens=max_tokens,
            )

        # Decode only the new tokens
        text_out = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()
        return text_out

    return {
        "model": model,
        "tokenizer": tokenizer,
        "image_processor": image_processor,
        "infer_fn": infer_fn,
    }


def _load_videollama3(model_id):
    """Loader for DAMO-NLP-SG/VideoLLaMA3-7B.

    VideoLLaMA3 ships custom modeling code (trust_remote_code=True needed).
    Uses AutoModelForCausalLM + AutoProcessor; processor handles the
    chat template + multi-image tokenization.

    Why this is a fallback: vLLM does not support VideoLLaMA3's custom
    config / modeling layers as of v0.20.0.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        import transformers.image_utils as hf_image_utils
        from typing import Any
    except ImportError:
        raise RuntimeError("Install: pip install -U transformers torch accelerate")
    try:
        from PIL import Image
        import io
    except ImportError:
        raise RuntimeError("Install: pip install pillow")

    print(f"  [HF] Loading VideoLLaMA3 from {model_id}...")
    # Compatibility shim:
    # Some VideoLLaMA3 remote-code revisions import `VideoInput` from
    # transformers.image_utils, but certain transformers releases don't expose
    # this alias. It's only used for typing, so define a fallback when absent.
    if not hasattr(hf_image_utils, "VideoInput"):
        hf_image_utils.VideoInput = Any
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print(f"  [HF] Loading model (this may take a minute)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    def infer_fn(item, frames_b64, max_tokens):
        # Decode frames to PIL
        pil_frames = []
        for b64 in frames_b64:
            img_bytes = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            pil_frames.append(img)

        system_text, user_text = build_prompt(item)
        # VideoLLaMA3 chat template accepts list-of-frames as a video block.
        # The processor expects {'type': 'video', 'video': {'video_path': ...,
        # 'fps': N, 'max_frames': M}} OR a list of images. We use direct
        # image list to bypass video-file requirements (we already
        # pre-sampled frames).
        messages = [
            {"role": "system", "content": system_text},
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in pil_frames],
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        inputs = processor(
            conversation=messages,
            add_system_prompt=False,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) if hasattr(v, 'to') else v
                  for k, v in inputs.items()}
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        gen_ids = out_ids[0][input_len:]
        text_out = processor.batch_decode(
            [gen_ids], skip_special_tokens=True
        )[0].strip()
        if not text_out:
            # Some remote-code models return only newly generated tokens from
            # generate(); slicing by input_len then drops everything.
            text_out = processor.batch_decode(
                out_ids, skip_special_tokens=True
            )[0].strip()
        return text_out

    return {"model": model, "processor": processor, "infer_fn": infer_fn}


def _load_video_llava(model_id):
    """Loader for LanguageBind/Video-LLaVA-7B-hf.

    Uses standard HF VideoLlavaForConditionalGeneration. Trained on
    8 frames natively.
    """
    try:
        import torch
        from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
        import numpy as np
    except ImportError:
        raise RuntimeError("Install: pip install -U transformers torch numpy")
    try:
        from PIL import Image
        import io
    except ImportError:
        raise RuntimeError("Install: pip install pillow")

    print(f"  [HF] Loading Video-LLaVA from {model_id}...")
    processor = VideoLlavaProcessor.from_pretrained(model_id)
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    def infer_fn(item, frames_b64, max_tokens):
        # Video-LLaVA expects a single video tensor of shape (T, H, W, C)
        # built from numpy arrays (RGB uint8). It was trained on 8 frames.
        # If we have more frames, sub-sample to 8 uniformly.
        target_frames = 8
        if len(frames_b64) > target_frames:
            indices = [int(i * len(frames_b64) / target_frames)
                       for i in range(target_frames)]
            frames_b64 = [frames_b64[i] for i in indices]

        np_frames = []
        for b64 in frames_b64:
            img_bytes = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            np_frames.append(np.array(img))
        video = np.stack(np_frames)  # (T, H, W, C)

        system_text, user_text = build_prompt(item)
        # Video-LLaVA prompt convention: "USER: <video>\n{question} ASSISTANT:"
        prompt = f"USER: <video>\n{system_text}\n\n{user_text} ASSISTANT:"

        inputs = processor(
            text=prompt, videos=video, return_tensors="pt"
        ).to(model.device, torch.bfloat16)

        with torch.no_grad():
            out_ids = model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False,
            )
        input_len = inputs["input_ids"].shape[1]
        gen_ids = out_ids[0][input_len:]
        text_out = processor.batch_decode(
            [gen_ids], skip_special_tokens=True
        )[0].strip()
        return text_out

    return {"model": model, "processor": processor, "infer_fn": infer_fn}


def _load_video_chatgpt(model_id):
    """Loader for MBZUAI/Video-ChatGPT-7B.

    Important:
      - MBZUAI/Video-ChatGPT-7B is NOT a full HF model. It mainly provides
        the projection weight video_chatgpt-7B.bin.
      - VIDEO_CHATGPT_BASE_MODEL should point to the LLaVA-Lightening base model.
      - VIDEO_CHATGPT_REPO should point to a local clone of the original
        Video-ChatGPT repo.
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError("Install: pip install -U torch")

    try:
        from PIL import Image
        import io
    except ImportError:
        raise RuntimeError("Install: pip install pillow")

    # ---------------------------------------------------------------------
    # 1. Import original Video-ChatGPT repo
    # ---------------------------------------------------------------------
    # The original repo is not a normal pip package. We therefore inject
    # the repo root into sys.path.
    local_repo = os.environ.get("VIDEO_CHATGPT_REPO", "/tmp/Video-ChatGPT")
    if local_repo and os.path.isdir(local_repo):
        if local_repo not in sys.path:
            sys.path.insert(0, local_repo)

    try:
        from video_chatgpt.eval.model_utils import initialize_model
        from video_chatgpt.video_conversation import conv_templates
        from video_chatgpt.constants import DEFAULT_VIDEO_TOKEN
    except ImportError as e:
        raise RuntimeError(
            "Video-ChatGPT requires the original repo. Example:\n"
            "  git clone https://github.com/mbzuai-oryx/Video-ChatGPT.git /tmp/Video-ChatGPT\n"
            "  export VIDEO_CHATGPT_REPO=/tmp/Video-ChatGPT\n\n"
            f"Original import error: {repr(e)}"
        )

    # ---------------------------------------------------------------------
    # 2. Resolve base model and projection weight
    # ---------------------------------------------------------------------
    # Base model: should be a full LLaVA/LLaMA model with config.json.
    # Projection: video_chatgpt-7B.bin from MBZUAI/Video-ChatGPT-7B.
    base_model = os.environ.get(
        "VIDEO_CHATGPT_BASE_MODEL",
        "mmaaz60/LLaVA-7B-Lightening-v1-1",
    )

    projection_path = os.environ.get("VIDEO_CHATGPT_PROJECTION", "").strip()
    if not projection_path:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise RuntimeError(
                "Video-ChatGPT needs projection weights. Either install "
                "huggingface_hub or set VIDEO_CHATGPT_PROJECTION to a local "
                "video_chatgpt-7B.bin path."
            )

        projection_path = hf_hub_download(
            repo_id=model_id,
            filename="video_chatgpt-7B.bin",
        )

    print(f"  [HF] Loading Video-ChatGPT base model: {base_model}")
    print(f"  [HF] Using Video-ChatGPT projection: {projection_path}")

    # initialize_model returns:
    #   model, vision_tower, tokenizer, image_processor, video_token_len
    model, vision_tower, tokenizer, image_processor, video_token_len = (
        initialize_model(base_model, projection_path)
    )

    model.eval()
    vision_tower.eval()

    # Some old LLaMA tokenizers do not define pad_token.
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------------------------------------------------------------
    # 3. Helper: build Video-ChatGPT spatio-temporal features
    # ---------------------------------------------------------------------
    def get_video_spatio_temporal_features(frame_features):
        """Convert per-frame CLIP patch features into Video-ChatGPT features.

        frame_features shape:
            [T, N, D]
        where:
            T = number of frames, usually 100
            N = number of spatial patch tokens, e.g. 256 for CLIP-L/14 at 224
            D = vision hidden size

        Video-ChatGPT concatenates:
          - temporal-pooled spatial tokens: [N, D]
          - spatial-pooled frame tokens:    [T, D]
        Final shape:
            [N + T, D]
        """
        # Average over time: one feature per spatial patch.
        temporal_features = frame_features.mean(dim=0)  # [N, D]

        # Average over spatial patches: one feature per frame.
        spatial_features = frame_features.mean(dim=1)   # [T, D]

        # [N + T, D], e.g. [256 + 100, 1024] = [356, 1024]
        video_features = torch.cat(
            [temporal_features, spatial_features],
            dim=0,
        )

        return video_features

    # ---------------------------------------------------------------------
    # 4. Inference function
    # ---------------------------------------------------------------------
    def infer_fn(item, frames_b64, max_tokens):
        if not frames_b64:
            return ""

        target_frames = 100

        # Decode base64 frames to PIL.
        pil_frames = []
        for b64 in frames_b64:
            img_bytes = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            pil_frames.append(img)

        # Uniformly repeat / downsample to exactly 100 frames.
        if len(pil_frames) < target_frames:
            indices = [
                int(i * len(pil_frames) / target_frames)
                for i in range(target_frames)
            ]
            pil_frames = [pil_frames[i] for i in indices]
        elif len(pil_frames) > target_frames:
            indices = [
                int(i * len(pil_frames) / target_frames)
                for i in range(target_frames)
            ]
            pil_frames = [pil_frames[i] for i in indices]

        # Force same RGB size before batching.
        # This avoids:
        #   ValueError: Unable to create tensor, you should probably activate padding...
        # in old CLIP/LLaVA image processors.
        pil_frames = [
            img.convert("RGB").resize((224, 224), Image.BICUBIC)
            for img in pil_frames
        ]

        # Build benchmark prompt.
        system_text, user_text = build_prompt(item)

        # Preprocess frames for CLIP vision tower.
        image_tensor = image_processor.preprocess(
            pil_frames,
            return_tensors="pt",
        )["pixel_values"]

        image_tensor = image_tensor.half().to(model.device)

        # Extract CLIP frame features.
        # Video-ChatGPT uses CLIP-L/14 features, then performs spatial/temporal pooling.
        with torch.no_grad():
            image_forward_outs = vision_tower(
                image_tensor,
                output_hidden_states=True,
            )

            # Use the penultimate hidden state and remove CLS token.
            # Shape: [T, N, D]
            frame_features = image_forward_outs.hidden_states[-2][:, 1:]

            video_spatio_temporal_features = get_video_spatio_temporal_features(
                frame_features
            )

        # Build Video-ChatGPT conversation prompt.
        conv = conv_templates["video-chatgpt_v1"].copy()
        # Override Video-ChatGPT's default "explain in detail" system prompt
        # so it matches our benchmark-wide strict output format.
        conv.system = system_text
        conv.append_message(
            conv.roles[0],
            DEFAULT_VIDEO_TOKEN + "\n" + user_text,
        )
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        tok = tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        input_ids = tok.input_ids.to(model.device)

        # Generate.
        # Some Video-ChatGPT code versions expect [1, N, D],
        # while others accept [N, D]. Try the batched form first.
        with torch.no_grad():
            try:
                output_ids = model.generate(
                    input_ids,
                    video_spatio_temporal_features=video_spatio_temporal_features.unsqueeze(0),
                    do_sample=False,
                    max_new_tokens=max_tokens,
                )
            except Exception:
                output_ids = model.generate(
                    input_ids,
                    video_spatio_temporal_features=video_spatio_temporal_features,
                    do_sample=False,
                    max_new_tokens=max_tokens,
                )

        text_out = tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0].strip()

        return text_out

    return {
        "model": model,
        "vision_tower": vision_tower,
        "tokenizer": tokenizer,
        "image_processor": image_processor,
        "video_token_len": video_token_len,
        "infer_fn": infer_fn,
    }


def _load_llava_ov(model_id):
    """Loader for lmms-lab/llava-onevision-qwen2-7b-ov.
 
    LLaVA-OneVision is image-only (no video native support). We use it as
    a single-frame ablation — pass only the middle frame.
 
    Uses standard HF LlavaOnevisionForConditionalGeneration.
    """
    try:
        import torch
        from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
    except ImportError:
        raise RuntimeError("Install: pip install -U transformers torch accelerate")
    try:
        from PIL import Image
        import io
    except ImportError:
        raise RuntimeError("Install: pip install pillow")
 
    print(f"  [HF] Loading LLaVA-OneVision from {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
 
    def infer_fn(item, frames_b64, max_tokens):
        # Single-frame ablation: take just the middle frame
        if not frames_b64:
            return ""
        middle_idx = len(frames_b64) // 2
        b64 = frames_b64[middle_idx]
        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
 
        system_text, user_text = build_prompt(item)
        # Use the robust HF pattern for LLaVA-OneVision:
        # 1) render chat template to plain text with image placeholders,
        # 2) feed text + PIL image to processor.
        # This avoids schema mismatches that can raise
        # "TypeError: string indices must be integers" in some versions.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{system_text}\n\n{user_text}"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = processor(
            text=[prompt],
            images=[img],
            return_tensors="pt",
            padding=True,
        ).to(model.device, torch.bfloat16)
 
        with torch.no_grad():
            out_ids = model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False,
            )
        input_len = inputs["input_ids"].shape[1]
        gen_ids = out_ids[0][input_len:]
        text_out = processor.batch_decode(
            [gen_ids], skip_special_tokens=True
        )[0].strip()
        return text_out
 
    return {"model": model, "processor": processor, "infer_fn": infer_fn}

# =============================================================================
# Adapter: pure logic baselines
# =============================================================================

def call_logic(cfg, item, _frames, _max_tokens, rng_state):
    """Local random/frequency/always_a — no network. rng_state is a Random()
    instance owned by the caller for reproducibility."""
    kind = cfg["logic_kind"]
    fmt = item["format"]

    if kind == "random":
        if fmt == "mcq_4":
            return rng_state.choice(["A", "B", "C", "D"])
        if fmt == "sp":
            return rng_state.choice(["A", "B"])
        if fmt == "numerical":
            return rng_state.choice(["2", "3", "4", "5 or more"])
        if fmt.startswith("ordering"):
            k = 3 if fmt == "ordering_3" else 4
            letters = list("ABCD")[:k]
            rng_state.shuffle(letters)
            return ", ".join(letters)

    if kind == "always_a":
        if fmt == "mcq_4" or fmt == "sp":
            return "A"
        if fmt == "numerical":
            return "2"  # most common bucket
        if fmt.startswith("ordering"):
            k = 3 if fmt == "ordering_3" else 4
            return ", ".join(list("ABCD")[:k])  # presentation order

    if kind == "frequency":
        # Most common per-format answer, computed on the fly elsewhere
        # See `frequency_prediction` below for how this is wired.
        # The runner pre-computes the frequency map and stuffs it into cfg.
        freq = cfg.get("_freq_map", {})
        ans = freq.get(fmt)
        if ans is None:
            return "A"  # fallback
        if fmt.startswith("ordering"):
            return ", ".join(ans)
        return ans

    raise ValueError(f"Unknown logic kind: {kind}")


def compute_frequency_map(items):
    """Most common answer per format from the actual benchmark data."""
    from collections import Counter
    by_fmt = {}
    for fmt in ("mcq_4", "sp", "numerical", "ordering_3", "ordering_4"):
        sub = [it for it in items if it["format"] == fmt]
        if not sub:
            continue
        if fmt.startswith("ordering"):
            answers = [tuple(it["correct_order"]) for it in sub]
        else:
            answers = [it["correct_answer"] for it in sub]
        most = Counter(answers).most_common(1)[0][0]
        by_fmt[fmt] = list(most) if isinstance(most, tuple) else most
    return by_fmt


# =============================================================================
# Per-item runner
# =============================================================================

def process_item(item, cfg, n_frames, videos_root, rng_state, registry=None):
    fmt = item["format"]
    qa_id = item["qa_id"]
    max_tokens = get_max_tokens(fmt, cfg)
    t0 = time.time()

    try:
        # Frame sampling (skip for logic and text-only)
        if cfg["kind"] in ("logic", "openai_text"):
            frames_b64 = []
        else:
            video_path, kind = resolve_video_path(
                item["video_id"], videos_root, registry=registry)
            if not video_path:
                return {
                    "qa_id": qa_id, "model": cfg.get("_name"),
                    "format": fmt, "raw_response": None,
                    "extracted_answer": None,
                    "latency_s": 0.0,
                    "error": f"video not found: {item['video_id']}",
                }
            frames_b64 = sample_frames_uniform(video_path, n_frames, kind=kind)
            if not frames_b64:
                return {
                    "qa_id": qa_id, "model": cfg.get("_name"),
                    "format": fmt, "raw_response": None,
                    "extracted_answer": None,
                    "latency_s": 0.0,
                    "error": f"no frames extracted: {video_path} (kind={kind})",
                }

        # Dispatch
        if cfg["kind"] == "openai_video":
            raw = call_openai_video(cfg, item, frames_b64, max_tokens)
        elif cfg["kind"] == "openai_text":
            raw = call_openai_text(cfg, item, frames_b64, max_tokens)
        elif cfg["kind"] == "google_video":
            raw = call_google_video(cfg, item, frames_b64, max_tokens)
        elif cfg["kind"] == "anthropic_video":
            raw = call_anthropic_video(cfg, item, frames_b64, max_tokens)
        elif cfg["kind"] == "hf_video":
            raw = call_hf_video(cfg, item, frames_b64, max_tokens)
        elif cfg["kind"] == "logic":
            raw = call_logic(cfg, item, frames_b64, max_tokens, rng_state)
        else:
            raise ValueError(f"Unknown kind: {cfg['kind']}")

        # Keep benchmark behavior unchanged for all models except Video-ChatGPT.
        if cfg.get("_name") == "video-chatgpt-7b":
            extracted = _extract_with_item_fallback(raw, item)
        else:
            extracted = extract(raw, fmt)
        return {
            "qa_id": qa_id, "model": cfg.get("_name"),
            "format": fmt, "raw_response": raw,
            "extracted_answer": extracted,
            "latency_s": round(time.time() - t0, 3),
            "error": None,
        }

    except Exception as e:
        return {
            "qa_id": qa_id, "model": cfg.get("_name"),
            "format": fmt, "raw_response": None,
            "extracted_answer": None,
            "latency_s": round(time.time() - t0, 3),
            "error": f"{type(e).__name__}: {e}",
        }


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, help="path to toc_bench_clean_v2.json")
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--videos-root", default=None,
                    help="Root dir for video files (required for non-logic models)")
    ap.add_argument("--video-registry", default=None,
                    help="Optional path to video_registry.json. If provided, "
                         "video paths are looked up by video_id (recommended; "
                         "handles jpeg_folder MOSE/OVIS correctly).")
    ap.add_argument("--frames", type=int, default=None,
                    help="Override frames_default for this model")
    ap.add_argument("--out", required=True, help="JSONL output path")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Parallel workers for HTTP-bound models")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only first N items (for testing)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strict-numerical", action="store_true",
                    help="Force numerical answers to one of [2,3,4,'5 or more'] "
                         "via vLLM guided_choice. Use only for the appendix run; "
                         "main run should leave this OFF to expose model's true "
                         "out-of-space failure mode.")
    args = ap.parse_args()

    cfg = dict(MODEL_REGISTRY[args.model])
    cfg["_name"] = args.model
    if args.strict_numerical:
        cfg["strict_numerical"] = True

    n_frames = (args.frames if args.frames is not None
                else cfg.get("frames_default", 32))

    # Need videos-root for any model that takes video input
    if cfg["kind"] not in ("logic", "openai_text") and not args.videos_root:
        print("[ERROR] --videos-root required for video-input models",
              file=sys.stderr)
        sys.exit(1)
    videos_root = Path(args.videos_root) if args.videos_root else None

    # Load video registry (optional but recommended)
    registry = None
    if args.video_registry:
        registry = load_video_registry(args.video_registry)
        print(f"Loaded video_registry with {len(registry)} entries from "
              f"{args.video_registry}")

    # Load benchmark
    with open(args.bench) as f:
        bench = json.load(f)
    items = bench["items"]
    if args.limit:
        items = items[:args.limit]

    # Wire frequency map for `frequency` baseline
    if cfg.get("logic_kind") == "frequency":
        cfg["_freq_map"] = compute_frequency_map(bench["items"])
        print(f"Frequency map: {cfg['_freq_map']}")

    # Resume support: load existing predictions
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_qa_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("error") is None:
                        done_qa_ids.add(rec["qa_id"])
                except Exception:
                    continue
        print(f"Resuming: {len(done_qa_ids)} items already done; "
              f"{len(items) - len(done_qa_ids)} remaining")

    items_todo = [it for it in items if it["qa_id"] not in done_qa_ids]

    # Per-thread RNG (logic adapters use it; one shared RNG is fine since
    # we run logic in 1 thread — but we keep it thread-local anyway)
    rng_state = random.Random(args.seed)

    # Open output in append mode
    out_f = open(out_path, "a", buffering=1)  # line-buffered

    print(f"\nModel: {args.model}")
    print(f"Kind: {cfg['kind']}")
    print(f"Frames: {n_frames}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Total items: {len(items)}, todo: {len(items_todo)}")
    print()

    # Logic baselines: serial (super fast)
    if cfg["kind"] == "logic":
        for i, it in enumerate(items_todo):
            rec = process_item(it, cfg, n_frames, videos_root, rng_state,
                               registry=registry)
            out_f.write(json.dumps(rec) + "\n")
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(items_todo)}")
        out_f.close()
        print(f"\nDone. Output: {out_path}")
        return

    # HTTP-bound: thread pool
    success = failed = 0
    t_start = time.time()

    if args.concurrency <= 1:
        # Serial
        for i, it in enumerate(items_todo):
            rec = process_item(it, cfg, n_frames, videos_root, rng_state,
                               registry=registry)
            out_f.write(json.dumps(rec) + "\n")
            if rec["error"]:
                failed += 1
            else:
                success += 1
            if (i + 1) % 50 == 0 or (i + 1) == len(items_todo):
                elapsed = time.time() - t_start
                rate = (i + 1) / max(elapsed, 1)
                eta = (len(items_todo) - i - 1) / max(rate, 0.001)
                print(f"  [{i+1}/{len(items_todo)}] OK={success} FAIL={failed} "
                      f"rate={rate:.2f}/s ETA={eta/60:.1f}min")
    else:
        # Parallel
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futs = {
                executor.submit(process_item, it, cfg, n_frames,
                                videos_root, rng_state, registry): it
                for it in items_todo
            }
            for j, fut in enumerate(as_completed(futs)):
                rec = fut.result()
                out_f.write(json.dumps(rec) + "\n")
                if rec["error"]:
                    failed += 1
                else:
                    success += 1
                if (j + 1) % 50 == 0 or (j + 1) == len(items_todo):
                    elapsed = time.time() - t_start
                    rate = (j + 1) / max(elapsed, 1)
                    eta = (len(items_todo) - j - 1) / max(rate, 0.001)
                    print(f"  [{j+1}/{len(items_todo)}] OK={success} "
                          f"FAIL={failed} rate={rate:.2f}/s "
                          f"ETA={eta/60:.1f}min")

    out_f.close()
    print(f"\nDone. Output: {out_path}")
    print(f"  Success: {success}  Failed: {failed}")
    if failed > 0:
        print(f"  Inspect failures with:")
        print(f"    grep '\"error\":\"' {out_path} | head")


if __name__ == "__main__":
    main()