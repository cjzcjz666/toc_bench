#!/usr/bin/env python3
"""
TOC-Bench Step 1b: Object Tracking with VLM-Guided Prompts + SAM 3.1
=====================================================================
Two-phase pipeline:
  Phase 1: Qwen3-VL-8B scans sampled frames → per-video object noun phrases
           Then VLM is unloaded to free GPU memory.
  Phase 2: SAM 3.1 uses those phrases as text prompts → track objects.

Usage:
    python step1b_track_objects.py [--video-id <id>] [--max-videos 10] [--device cuda:0]
    python step1b_track_objects.py --mock        # test without GPU
    python step1b_track_objects.py --no-vlm      # skip VLM, use fallback prompts

Requirements:
    pip install transformers qwen-vl-utils   # for Qwen3-VL
    pip install sam3                          # for SAM 3.1
    GPU with >=24GB VRAM recommended
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import ROOT, TRACKS_DIR, SAM3_CONFIG, VLM_CONFIG


# ============================================================
# Video I/O
# ============================================================

def load_video_frames(video_path: str, target_fps: float = 6.0,
                      max_frames: int = 300) -> tuple:
    path = Path(video_path)

    if path.is_dir():
        # JPEG folder (MOSE / OVIS style)
        jpg_files = sorted(path.glob("*.jpg"))
        if not jpg_files:
            jpg_files = sorted(path.glob("*.png"))
        if not jpg_files:
            raise ValueError(f"No image files found in {path}")

        total_frames = len(jpg_files)

        # Estimate original fps: OVIS/MOSE typically store every frame at ~30fps
        # or already subsampled at ~5-6fps. Heuristic: if >100 frames for a
        # short clip, it's likely high fps; otherwise assume frames are already sparse.
        # We use a conservative estimate and let target_fps handle the subsampling.
        if total_frames > 200:
            original_fps = 30.0  # likely full frame rate
        elif total_frames > 50:
            original_fps = 15.0  # moderate
        else:
            original_fps = 5.0   # already sparse, treat each frame as ~0.2s apart

        step = max(1, int(original_fps / target_fps))
        candidates = jpg_files[::step]

        # If still too many, uniformly resample to max_frames covering entire video
        if len(candidates) > max_frames:
            indices = np.linspace(0, len(candidates) - 1, max_frames, dtype=int)
            candidates = [candidates[i] for i in indices]

        frames, timestamps = [], []
        for f in candidates:
            img = cv2.imread(str(f))
            if img is not None:
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                orig_idx = jpg_files.index(f)
                timestamps.append(orig_idx / original_fps)

        h, w = frames[0].shape[:2] if frames else (0, 0)
        metadata = {
            "original_fps": original_fps, "duration": total_frames / original_fps,
            "width": w, "height": h,
            "total_frames": total_frames, "extracted_frames": len(frames),
        }
        return frames, timestamps, metadata
    else:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / original_fps if original_fps > 0 else 0

        step = max(1, int(original_fps / target_fps))
        frame_indices = list(range(0, total_frames, step))

        # If still too many, uniformly resample to max_frames covering entire video
        if len(frame_indices) > max_frames:
            idx_arr = np.linspace(0, len(frame_indices) - 1, max_frames, dtype=int)
            frame_indices = [frame_indices[i] for i in idx_arr]

        frames, timestamps = [], []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                timestamps.append(idx / original_fps)
        cap.release()

        metadata = {
            "original_fps": original_fps, "duration": duration,
            "width": width, "height": height,
            "total_frames": total_frames, "extracted_frames": len(frames),
        }
        return frames, timestamps, metadata


# ============================================================
# VLM Prompt Generator (Qwen3-VL-8B-Instruct)
# ============================================================

class VLMPromptGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.processor = None
        self.use_mock = False

        try:
            self._load_model()
        except Exception as e:
            print(f"  [WARN] Qwen3-VL not loaded: {e}")
            print(f"  [WARN] Will use fallback prompts.")
            self.use_mock = True

    def _load_model(self):
        import torch
        from transformers import AutoProcessor, AutoTokenizer

        model_name = self.config.get("model_name", "Qwen/Qwen3-VL-8B-Instruct")
        print(f"  Loading VLM: {model_name} ...")

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Try Qwen3-VL first, fall back to Qwen2.5-VL class
        try:
            from transformers import Qwen3VLForConditionalGeneration
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="auto",
            )
        except ImportError:
            from transformers import Qwen2_5_VLForConditionalGeneration
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="auto",
            )

        print(f"  [OK] VLM loaded")

    def generate_prompts(self, frames: list, timestamps: list) -> list:
        if self.use_mock:
            return self.config.get("fallback_prompts", ["person", "object"])
        return self._vlm_generate(frames, timestamps)

    def _vlm_generate(self, frames, timestamps) -> list:
        """
        Run VLM on the provided frames (already sampled by caller) to get object list.
        No secondary sampling — uses all frames given.
        """
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        # Convert all provided frames to PIL (caller already sampled at vlm_fps)
        pil_images = [Image.fromarray(f) for f in frames]

        # Build multimodal message — images as PIL objects
        image_content = []
        for img in pil_images:
            image_content.append({"type": "image", "image": img})
        image_content.append({
            "type": "text",
            "text": self.config.get("system_prompt",
                "List all distinct physical objects visible in these frames as a JSON list.")
        })

        messages = [{"role": "user", "content": image_content}]

        try:
            # Step 1: Use tokenizer (not processor) to apply chat template
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Step 2: Use qwen_vl_utils to extract image inputs
            image_inputs, video_inputs = process_vision_info(messages)

            # Step 3: Process text + images together through processor
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, max_new_tokens=512, do_sample=False,
                )

            # Trim input tokens to get only generated text
            generated_ids = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, output_ids)
            ]
            response_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

        except Exception as e:
            print(f"    [WARN] VLM generation failed: {e}")
            return self.config.get("fallback_prompts", ["person", "object"])

        prompts = self._parse_object_list(response_text)
        if not prompts:
            print(f"    [WARN] VLM returned empty, using fallback")
            return self.config.get("fallback_prompts", ["person", "object"])

        max_prompts = self.config.get("max_prompts_per_video", 15)
        return prompts[:max_prompts]

    def _parse_object_list(self, response_text: str) -> list:
        # Try JSON array
        try:
            match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if match:
                items = json.loads(match.group())
                if isinstance(items, list):
                    cleaned, seen = [], set()
                    for item in items:
                        if isinstance(item, str):
                            item = item.strip().lower()
                            if item and item not in seen and len(item) < 50:
                                cleaned.append(item)
                                seen.add(item)
                    return cleaned
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: split by separators
        items = re.split(r'[\n,;•\-\d.]+', response_text)
        cleaned, seen = [], set()
        skip_words = {'wall', 'floor', 'ceiling', 'sky', 'background', 'room', 'light'}
        for item in items:
            item = item.strip().strip('"\'').lower()
            if (item and len(item) > 1 and len(item) < 50
                and item not in seen
                and not any(w in item for w in skip_words)):
                cleaned.append(item)
                seen.add(item)
        return cleaned

    def unload(self):
        if self.model is not None:
            import torch
            del self.model
            del self.processor
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
            self.model = None
            self.processor = None
            self.tokenizer = None
            torch.cuda.empty_cache()
            print("  [OK] VLM unloaded, GPU memory freed for SAM3")


# ============================================================
# SAM 3.1 Tracker
# ============================================================

class SAM3Tracker:
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.use_mock = False
        try:
            self._load_model()
        except ImportError:
            print("  [WARN] SAM 3.1 not installed. Using mock tracker.")
            self.use_mock = True

    def _load_model(self):
        from sam3.model_builder import build_sam3_video_predictor
        self.model = build_sam3_video_predictor()
        print(f"  [OK] SAM 3.1 loaded")

    def track_video(self, frames, timestamps, text_prompts, metadata):
        if self.use_mock:
            return self._mock_track(frames, timestamps, text_prompts, metadata)
        return self._sam3_track(frames, timestamps, text_prompts, metadata)

    def _sam3_track(self, frames, timestamps, text_prompts, metadata):
        video_predictor = self.model
        all_tracks = []

        import tempfile
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp_dir:
            for i, frame in enumerate(frames):
                Image.fromarray(frame).save(os.path.join(tmp_dir, f"{i:06d}.jpg"))

            response = video_predictor.handle_request(
                request=dict(type="start_session", resource_path=tmp_dir)
            )
            session_id = response["session_id"]

            for prompt_text in text_prompts:
                try:
                    video_predictor.handle_request(
                        request=dict(type="reset_session", session_id=session_id)
                    )

                    response = video_predictor.handle_request(
                        request=dict(
                            type="add_prompt", session_id=session_id,
                            frame_index=0, text=prompt_text,
                        )
                    )

                    out = response.get("outputs", {})
                    if not isinstance(out, dict) or len(out.get("out_boxes_xywh", [])) == 0:
                        continue
                    print(f"      SAM3 '{prompt_text}': {len(out['out_boxes_xywh'])} det")

                    outputs_per_frame = {}
                    for prop_resp in video_predictor.handle_stream_request(
                        request=dict(type="propagate_in_video", session_id=session_id)
                    ):
                        outputs_per_frame[prop_resp["frame_index"]] = prop_resp["outputs"]

                    n_frames = len(frames)
                    min_area = self.config.get("min_mask_area", 100)

                    per_obj = {}
                    for fi, fo in outputs_per_frame.items():
                        if not isinstance(fo, dict):
                            continue
                        # SAM3 actual keys: out_obj_ids, out_binary_masks, out_probs, out_boxes_xywh
                        obj_ids = fo.get("out_obj_ids", np.array([]))
                        masks = fo.get("out_binary_masks", np.array([]))
                        probs = fo.get("out_probs", np.array([]))

                        for k in range(len(obj_ids)):
                            oid_int = int(obj_ids[k])
                            if oid_int not in per_obj:
                                per_obj[oid_int] = {}
                            mask = masks[k] if k < len(masks) else None
                            score = float(probs[k]) if k < len(probs) else 0.0
                            if mask is not None:
                                m = mask.cpu().numpy() if hasattr(mask, 'cpu') else np.array(mask)
                                while m.ndim > 2:
                                    m = m[0]
                                per_obj[oid_int][fi] = (m, score)

                    for oid_int, fdata in per_obj.items():
                        track = {
                            "obj_id": len(all_tracks), "label": prompt_text,
                            "frames": [], "bboxes": [], "visible": [],
                            "confidence": [], "mask_areas": [],
                        }
                        for fi in range(n_frames):
                            if fi in fdata:
                                m, sc = fdata[fi]
                                if m.any():
                                    ys, xs = np.where(m > 0.5)
                                    if len(ys) > 0:
                                        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                                        area = int((m > 0.5).sum())
                                        vis = area >= min_area
                                    else:
                                        bbox, area, vis, sc = [0,0,0,0], 0, False, 0.0
                                else:
                                    bbox, area, vis, sc = [0,0,0,0], 0, False, 0.0
                            else:
                                bbox, area, vis, sc = [0,0,0,0], 0, False, 0.0

                            track["frames"].append(fi)
                            track["bboxes"].append(bbox)
                            track["visible"].append(vis)
                            track["confidence"].append(round(float(sc), 4))
                            track["mask_areas"].append(area)

                        if sum(track["visible"]) / max(1, n_frames) > 0.1:
                            all_tracks.append(track)

                except Exception as e:
                    print(f"      [WARN] Prompt '{prompt_text}' failed: {e}")
                    continue

            video_predictor.handle_request(
                request=dict(type="close_session", session_id=session_id)
            )

        return all_tracks[:self.config.get("max_objects_per_video", 20)]

    def _mock_track(self, frames, timestamps, text_prompts, metadata):
        n_frames = len(frames)
        if n_frames == 0:
            return []
        h, w = frames[0].shape[:2]
        rng = np.random.RandomState(hash(str(timestamps[0])) % 2**31)
        n_objects = rng.randint(2, min(6, len(text_prompts) + 1))
        tracks = []

        for obj_id in range(n_objects):
            label = text_prompts[obj_id % len(text_prompts)]
            cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
            bw, bh = rng.uniform(0.05, 0.2) * w, rng.uniform(0.05, 0.2) * h
            vx, vy = rng.uniform(-3, 3), rng.uniform(-2, 2)
            occ_s = rng.randint(n_frames // 4, n_frames * 3 // 4)
            occ_e = min(occ_s + rng.randint(2, max(3, n_frames // 6)), n_frames - 1)
            app_f = rng.choice([0, 0, 0, rng.randint(1, max(2, n_frames // 4))])
            dis_f = rng.choice([n_frames] * 3 + [rng.randint(n_frames * 3 // 4, n_frames)])

            track = {"obj_id": obj_id, "label": label, "frames": [], "bboxes": [],
                     "visible": [], "confidence": [], "mask_areas": []}
            for i in range(n_frames):
                cx += vx + rng.normal(0, 0.5)
                cy += vy + rng.normal(0, 0.3)
                cx, cy = np.clip(cx, bw/2, w-bw/2), np.clip(cy, bh/2, h-bh/2)
                x1, y1 = int(max(0, cx-bw/2)), int(max(0, cy-bh/2))
                x2, y2 = int(min(w, cx+bw/2)), int(min(h, cy+bh/2))
                area = (x2-x1) * (y2-y1)
                vis = not (occ_s <= i < occ_e or i < app_f or i >= dis_f)
                conf = rng.uniform(0.85, 0.98) if vis else rng.uniform(0.05, 0.25)
                track["frames"].append(i)
                track["bboxes"].append([x1, y1, x2, y2])
                track["visible"].append(vis)
                track["confidence"].append(round(float(conf), 4))
                track["mask_areas"].append(area if vis else 0)

            if sum(track["visible"]) / n_frames > 0.2:
                tracks.append(track)
        return tracks


# ============================================================
# Main: Two-Phase Pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="TOC-Bench Step 1b: VLM-guided SAM 3.1 object tracking"
    )
    parser.add_argument("--video-id", type=str, default=None)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Split (to_process) into this many contiguous chunks")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Which shard to run: [0, num-shards)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mock", action="store_true",
                        help="Mock mode: no GPU needed")
    parser.add_argument("--no-vlm", action="store_true",
                        help="Skip VLM execution. When used with --vlm-prompts-cache, "
                             "loads prompts from cache; otherwise falls back to "
                             "_vlm_prompts_cache.json if it exists.")
    parser.add_argument(
        "--vlm-prompts-cache",
        type=str,
        default=None,
        help="Path to a VLM prompts cache JSON. Used only when --no-vlm."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  TOC-Bench Step 1b: VLM-Guided SAM 3.1 Tracking")
    print("=" * 60)

    registry_path = ROOT / "video_registry.json"
    if not registry_path.exists():
        print(f"[ERROR] Registry not found: {registry_path}")
        sys.exit(1)

    with open(registry_path) as f:
        registry = json.load(f)

    if args.video_id:
        registry = [v for v in registry if v["video_id"] == args.video_id]
    if args.source:
        registry = [v for v in registry if v["source"] == args.source]
    if args.max_videos:
        registry = registry[:args.max_videos]

    # Filter out already processed (unless overwrite)
    to_process = []
    skipped_existing = 0
    for entry in registry:
        output_path = TRACKS_DIR / f"{entry['video_id']}.json"
        if output_path.exists() and not args.overwrite:
            skipped_existing += 1
        else:
            to_process.append(entry)

    print(f"  Registry: {len(registry)} videos")
    print(f"  Already done: {skipped_existing}")
    print(f"  To process: {len(to_process)}")

    if args.num_shards > 1 and not args.no_vlm:
        if not (0 <= args.shard_index < args.num_shards):
            print(f"[ERROR] shard-index must be in [0, num-shards), got {args.shard_index}")
            sys.exit(1)
        n = len(to_process)
        chunk = (n + args.num_shards - 1) // args.num_shards
        start = args.shard_index * chunk
        end = min(start + chunk, n)
        to_process = to_process[start:end]
        print(f"  Shard {args.shard_index}/{args.num_shards}: to_process [{start}, {end}) => {len(to_process)} videos")

    if not to_process:
        print("  Nothing to do!")
        return

    vlm_config = VLM_CONFIG.copy()
    sam_config = SAM3_CONFIG.copy()
    sam_config["device"] = args.device

    # ========== Phase 1: VLM prompt generation ==========
    print(f"\n{'='*60}")
    print(f"  Phase 1: VLM Prompt Generation ({len(to_process)} videos)")
    print(f"{'='*60}")

    prompts_cache = {}
    vlm_fps = vlm_config.get("vlm_fps", 1.0)
    vlm_max_frames = vlm_config.get("vlm_max_frames", 10)

    if args.no_vlm:
        cache_path = (
            Path(args.vlm_prompts_cache)
            if args.vlm_prompts_cache
            else (TRACKS_DIR / "_vlm_prompts_cache.json")
        )
        if not cache_path.exists():
            print(f"[ERROR] prompts cache not found: {cache_path}")
            print("Run Phase 1 first (without --no-vlm), or pass --vlm-prompts-cache.")
            sys.exit(1)
        with open(cache_path) as f:
            prompts_cache = json.load(f)
        cache_keys = set(prompts_cache.keys())
        before = len(to_process)
        to_process = [e for e in to_process if e.get("video_id") in cache_keys]
        print(
            f"\n  Loaded prompts cache from {cache_path} (keys={len(prompts_cache)})"
            f"\n  Filtered to_process by cache keys: {before} -> {len(to_process)}"
        )
        # In --no-vlm mode, shard should be applied on the cache-filtered set,
        # so shards correspond to "already computed VLM prompts", not the full registry.
        if args.num_shards > 1:
            if not (0 <= args.shard_index < args.num_shards):
                print(f"[ERROR] shard-index must be in [0, num-shards), got {args.shard_index}")
                sys.exit(1)
            n = len(to_process)
            chunk = (n + args.num_shards - 1) // args.num_shards
            start = args.shard_index * chunk
            end = min(start + chunk, n)
            to_process = to_process[start:end]
            print(
                f"  No-vlm shard {args.shard_index}/{args.num_shards}: "
                f"to_process [{start}, {end}) => {len(to_process)} videos"
            )
        if not to_process:
            print("  Nothing to do after cache-key filtering!")
            return
    else:
        if args.mock:
            vlm = VLMPromptGenerator(vlm_config)
            vlm.use_mock = True
        else:
            vlm = VLMPromptGenerator(vlm_config)

        for i, entry in enumerate(to_process):
            vid = entry["video_id"]
            print(f"  [{i+1}/{len(to_process)}] {vid} ... ", end="", flush=True)
            try:
                frames, timestamps, _ = load_video_frames(
                    entry["path"], target_fps=vlm_fps, max_frames=vlm_max_frames
                )
                prompts = vlm.generate_prompts(frames, timestamps)
                prompts_cache[vid] = prompts
                print(f"{prompts}")
            except Exception as e:
                print(f"[ERROR] {e}")
                prompts_cache[vid] = vlm_config.get(
                    "fallback_prompts", ["person", "object"]
                )

        # Save prompts cache for reproducibility
        prompts_cache_path = TRACKS_DIR / "_vlm_prompts_cache.json"
        with open(prompts_cache_path, "w") as f:
            json.dump(prompts_cache, f, indent=2)
        print(f"\n  Prompts cache saved to {prompts_cache_path}")

        # Free VLM GPU memory
        vlm.unload()

    # ========== Phase 2: SAM3 tracking ==========
    print(f"\n{'='*60}")
    print(f"  Phase 2: SAM3 Object Tracking ({len(to_process)} videos)")
    print(f"{'='*60}")

    if args.mock:
        tracker = SAM3Tracker(sam_config)
        tracker.use_mock = True
    else:
        tracker = SAM3Tracker(sam_config)

    success, failed = 0, 0
    for i, entry in enumerate(to_process):
        vid = entry["video_id"]
        output_path = TRACKS_DIR / f"{vid}.json"
        print(f"\n[{i+1}/{len(to_process)}] {vid}")

        t0 = time.time()
        try:
            frames, timestamps, metadata = load_video_frames(
                entry["path"], target_fps=sam_config.get("fps_for_tracking", 6.0),
            )
        except Exception as e:
            print(f"    [ERROR] Load: {e}")
            failed += 1
            continue

        if len(frames) < 4:
            print(f"    [SKIP] Too few frames ({len(frames)})")
            failed += 1
            continue

        text_prompts = prompts_cache.get(vid, ["person", "object"])
        print(f"    {len(frames)} frames, {len(text_prompts)} prompts")

        try:
            tracks = tracker.track_video(frames, timestamps, text_prompts, metadata)
        except Exception as e:
            print(f"    [ERROR] Track: {e}")
            failed += 1
            continue

        result = {
            "video_id": vid,
            "source": entry["source"],
            "video_path": entry["path"],
            "metadata": metadata,
            "timestamps": [round(t, 3) for t in timestamps],
            "vlm_prompts": text_prompts,
            "num_objects": len(tracks),
            "objects": tracks,
            "processing_time": round(time.time() - t0, 2),
        }

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"    {len(tracks)} objects tracked ({time.time()-t0:.1f}s)")
        success += 1

    print(f"\n{'='*60}")
    print(f"  Done! Success: {success}  Failed: {failed}  Skipped: {skipped_existing}")
    print(f"  Tracks: {TRACKS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()