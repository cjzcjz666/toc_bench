#!/usr/bin/env python3
"""
Run a safe subset test for step3b taskwise pipeline.

What it does:
1) Scan reasoning_units and find files containing target dims.
2) Pick N files (default 3).
3) Run step3b taskwise candidate building + balanced sampling only on them.
4) Save everything under qa/test_step3b_subset to avoid polluting normal outputs.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import QA_DIR
import scripts.step3b_build_skeletons_taskwise as step3b


TARGET_DIMS = {
    "which_appears_first",
    "which_visible_longer",
    "which_more_events",
    "interaction_existence",
    "interaction_partner",
    "contrastive_partner",
}


def find_matching_files(units_dir: Path):
    matched = []
    for fp in sorted(units_dir.glob("*.json")):
        if fp.name.startswith("_"):
            continue
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue

        hit_dims = Counter()
        for unit in data.get("units", []):
            for ask in unit.get("askable", []):
                dim = ask.get("dim")
                if dim in TARGET_DIMS:
                    hit_dims[dim] += 1
        if hit_dims:
            matched.append((fp, hit_dims))
    return matched


def main():
    parser = argparse.ArgumentParser(description="Subset test for step3b taskwise.")
    parser.add_argument("--units-dir", type=str, default=None,
                        help="Input reasoning_units dir (default: QA_DIR/reasoning_units)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output base dir (default: QA_DIR/test_step3b_subset)")
    parser.add_argument("--pick", type=int, default=3,
                        help="How many matching unit files to pick")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", type=int, default=200,
                        help="Sampling target for subset run")
    parser.add_argument("--max-per-video", type=int, default=6)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    units_dir = Path(args.units_dir) if args.units_dir else (QA_DIR / "reasoning_units")
    out_base = Path(args.out_dir) if args.out_dir else (QA_DIR / "test_step3b_subset")
    subset_units_dir = out_base / "reasoning_units_subset"
    subset_skeletons_dir = out_base / "skeletons_subset"
    out_base.mkdir(parents=True, exist_ok=True)
    subset_units_dir.mkdir(parents=True, exist_ok=True)
    subset_skeletons_dir.mkdir(parents=True, exist_ok=True)

    matches = find_matching_files(units_dir)
    if not matches:
        print(f"[ERROR] No files in {units_dir} contain target dims.")
        return

    print(f"Found matching files: {len(matches)}")
    rng.shuffle(matches)
    selected = matches[:max(1, args.pick)]
    print(f"Picked files: {len(selected)}")

    # Save selected source files into subset input folder for reproducibility.
    selected_meta = []
    for fp, hit_dims in selected:
        dst = subset_units_dir / fp.name
        dst.write_text(fp.read_text())
        selected_meta.append({
            "file": fp.name,
            "video_id": fp.stem,
            "hit_dims": dict(hit_dims),
        })

    # Build a global label pool only from selected files.
    global_label_pool = set()
    loaded_data = []
    for item in selected_meta:
        p = subset_units_dir / item["file"]
        data = json.loads(p.read_text())
        loaded_data.append(data)
        for unit in data.get("units", []):
            if "subject" in unit:
                global_label_pool.add(unit["subject"]["label"])
            if "subjects" in unit:
                for s in unit["subjects"]:
                    global_label_pool.add(s["label"])
    global_label_pool = sorted(global_label_pool)

    (subset_skeletons_dir / "_global_label_pool.json").write_text(
        json.dumps(global_label_pool, indent=2)
    )

    all_candidates = []
    for data in loaded_data:
        vid = data.get("video_id", "unknown_video")
        video_labels = step3b.get_video_labels(data)
        per_video = step3b.build_candidates_for_video(
            vid, data, video_labels, global_label_pool, rng
        )
        all_candidates.extend(per_video)

    selected_candidates = step3b.balanced_sample(
        all_candidates, args.target, args.max_per_video, seed=args.seed
    )

    skeletons_by_video = defaultdict(list)
    for sk in selected_candidates:
        vid = sk["video_id"]
        sk.pop("weight", None)
        skeletons_by_video[vid].append(sk)

    for vid, skels in skeletons_by_video.items():
        out = {
            "video_id": vid,
            "num_skeletons": len(skels),
            "skeletons": skels,
        }
        (subset_skeletons_dir / f"{vid}.json").write_text(json.dumps(out, indent=2))

    # Summary
    all_sk = [s for arr in skeletons_by_video.values() for s in arr]
    summary = {
        "selected_files": selected_meta,
        "target_dims": sorted(TARGET_DIMS),
        "matched_files_total": len(matches),
        "picked_files": len(selected_meta),
        "raw_candidates": len(all_candidates),
        "selected_skeletons": len(all_sk),
        "target": args.target,
        "max_per_video": args.max_per_video,
        "task_type_counts": dict(Counter(s.get("task_type", "?") for s in all_sk)),
        "format_counts": dict(Counter(s.get("format", "?") for s in all_sk)),
        "dimension_counts": dict(Counter(s.get("dimension", "?") for s in all_sk)),
    }
    (subset_skeletons_dir / "_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nDone.")
    print(f"Subset units:      {subset_units_dir}")
    print(f"Subset skeletons:  {subset_skeletons_dir}")
    print(f"Raw candidates:    {len(all_candidates)}")
    print(f"Selected skeleton: {len(all_sk)}")


if __name__ == "__main__":
    main()

