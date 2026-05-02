#!/usr/bin/env python3
"""
Fix duplicate labels in existing track JSONs.
Adds #N suffix when multiple objects share the same label.

Usage:
    python fix_duplicate_labels.py
"""

import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import TRACKS_DIR


def fix_labels(track_path):
    with open(track_path) as f:
        data = json.load(f)

    objects = data.get("objects", [])
    if not objects:
        return False

    # Count how many times each base label appears
    label_counts = Counter(obj["label"] for obj in objects)

    # Only fix if there are duplicates
    has_dupes = any(c > 1 for c in label_counts.values())
    if not has_dupes:
        return False

    # Assign #N to duplicated labels
    label_counter = {}
    for obj in objects:
        base_label = obj["label"]
        if label_counts[base_label] > 1:
            label_counter[base_label] = label_counter.get(base_label, 0) + 1
            obj["label"] = f"{base_label} #{label_counter[base_label]}"

    # Update object_labels if present in vlm_prompts context
    data["objects"] = objects

    with open(track_path, "w") as f:
        json.dump(data, f, indent=2)

    return True


def main():
    track_files = sorted(TRACKS_DIR.glob("*.json"))
    track_files = [f for f in track_files if not f.name.startswith("_")]

    print(f"Scanning {len(track_files)} track files...")

    fixed = 0
    for tf in track_files:
        if fix_labels(tf):
            fixed += 1

    print(f"Done. Fixed {fixed} / {len(track_files)} files.")


if __name__ == "__main__":
    main()