#!/usr/bin/env python3
"""
TOC-Bench Step 1a: Multi-source Video Download
===============================================
Downloads videos from 5 data sources:
  - Perception Test (auto)
  - OVIS (manual)
  - MOSE (manual)
  - Charades-STA (manual)
  - STAR (manual)

Usage:
    python download_videos.py [--source perception_test] [--dry-run]

Set TOC_BENCH_ROOT env var to control the base directory (default: /data/toc_bench).
"""

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import DATA_SOURCES, VIDEOS_DIR, ROOT


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file using wget with progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [SKIP] Already exists: {dest}")
        return True

    print(f"  [DOWNLOAD] {desc or url}")
    print(f"  -> {dest}")
    try:
        subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(dest), url],
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] Download failed: {e}")
        return False
    except FileNotFoundError:
        # wget not available, try curl
        try:
            subprocess.run(
                ["curl", "-L", "-o", str(dest), "--progress-bar", url],
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"  [ERROR] Neither wget nor curl available: {e}")
            return False


def unzip_file(zip_path: Path, dest_dir: Path) -> bool:
    """Unzip a file to destination directory."""
    print(f"  [UNZIP] {zip_path.name} -> {dest_dir}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dest_dir)
        return True
    except Exception as e:
        print(f"  [ERROR] Unzip failed: {e}")
        return False


def download_perception_test(config: dict, dry_run: bool = False) -> dict:
    """
    Download Perception Test validation videos.
    Returns stats dict.
    """
    print("\n" + "=" * 60)
    print("📦 Perception Test (validation set)")
    print("=" * 60)

    video_dir = config["video_subdir"]
    video_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing = list(video_dir.glob("*.mp4"))
    if len(existing) >= 100:  # validation set has ~1500 videos
        print(f"  [OK] Found {len(existing)} videos already in {video_dir}")
        return {"source": "perception_test", "status": "exists", "count": len(existing)}

    if dry_run:
        print(f"  [DRY-RUN] Would download from: {config['download_url']}")
        return {"source": "perception_test", "status": "dry_run", "count": 0}

    # Download zip
    zip_path = VIDEOS_DIR / "perception_test_valid_videos.zip"
    if not download_file(config["download_url"], zip_path, "Perception Test valid videos"):
        return {"source": "perception_test", "status": "download_failed", "count": 0}

    # Unzip
    if not unzip_file(zip_path, video_dir):
        return {"source": "perception_test", "status": "unzip_failed", "count": 0}

    # Count videos (may be in subfolders)
    videos = list(video_dir.rglob("*.mp4"))
    # If videos are in a subfolder, flatten
    if not list(video_dir.glob("*.mp4")) and videos:
        print(f"  [INFO] Moving videos from subfolders to {video_dir}")
        for v in videos:
            dest = video_dir / v.name
            if not dest.exists():
                v.rename(dest)

    final_count = len(list(video_dir.rglob("*.mp4")))
    print(f"  [OK] {final_count} videos ready in {video_dir}")

    # Optionally clean up zip
    # zip_path.unlink()

    return {"source": "perception_test", "status": "ok", "count": final_count}


def print_manual_instructions(source_name: str, config: dict):
    """Print manual download instructions for a data source."""
    print(f"\n" + "=" * 60)
    print(f"📋 {source_name.upper()} (manual download required)")
    print("=" * 60)
    print(f"  Description: {config['description']}")
    print(f"  Target count: {config['target_count']} videos")
    print(f"  Destination: {config['video_subdir']}")
    print()

    instructions = config.get("instructions", "No instructions available.")
    # Format instructions with actual paths
    instructions = instructions.format(video_subdir=config["video_subdir"])
    for line in instructions.strip().split("\n"):
        print(f"  {line.strip()}")
    print()


def check_source_status(source_name: str, config: dict) -> dict:
    """Check how many videos are already present for a source."""
    video_dir = config["video_subdir"]
    if not video_dir.exists():
        return {"source": source_name, "status": "missing", "count": 0}

    ext = config["video_ext"]
    if ext == ".jpg_folder":
        # MOSE: count subdirectories that contain JPEG files
        count = sum(1 for d in video_dir.iterdir()
                    if d.is_dir() and list(d.glob("*.jpg")))
    elif isinstance(ext, list):
        count = sum(len(list(video_dir.glob(f"*{e}"))) for e in ext)
    else:
        count = len(list(video_dir.glob(f"*{ext}")))

    status = "ok" if count >= config["target_count"] * 0.5 else "insufficient"
    return {"source": source_name, "status": status, "count": count}


def create_video_registry(sources: dict) -> Path:
    """
    Scan all video directories and create a unified registry JSON.
    This is the input to the tracking pipeline.

    Sources with 'share_source' (e.g. STAR shares Charades) will NOT
    re-scan the directory. Instead, their videos are a subset of the
    parent source — we just tag them differently. During tracking,
    each physical video is processed only once; the source label is
    used later for per-source analysis.
    """
    registry = []
    registered_paths = set()  # avoid duplicates when dirs overlap

    for source_name, config in sources.items():
        # Skip shared sources — they'll be handled after the parent is scanned
        if config.get("share_source"):
            continue

        video_dir = config["video_subdir"]
        if not video_dir.exists():
            continue

        ext = config["video_ext"]
        if ext == ".jpg_folder":
            for d in sorted(video_dir.iterdir()):
                if d.is_dir() and list(d.glob("*.jpg")):
                    path_str = str(d)
                    if path_str not in registered_paths:
                        registry.append({
                            "video_id": f"{source_name}_{d.name}",
                            "source": source_name,
                            "path": path_str,
                            "format": "jpeg_folder",
                        })
                        registered_paths.add(path_str)
        else:
            exts = ext if isinstance(ext, list) else [ext]
            for e in exts:
                for f in sorted(video_dir.glob(f"*{e}")):
                    path_str = str(f)
                    if path_str not in registered_paths:
                        registry.append({
                            "video_id": f"{source_name}_{f.stem}",
                            "source": source_name,
                            "path": path_str,
                            "format": "video_file",
                        })
                        registered_paths.add(path_str)

    # Save registry
    registry_path = ROOT / "video_registry.json"
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"\n📝 Video registry: {registry_path}")
    print(f"   Total videos registered: {len(registry)}")
    for source_name in sources:
        if sources[source_name].get("share_source"):
            parent = sources[source_name]["share_source"]
            parent_count = sum(1 for r in registry if r["source"] == parent)
            print(f"   - {source_name}: (shares {parent_count} videos with {parent})")
        else:
            count = sum(1 for r in registry if r["source"] == source_name)
            print(f"   - {source_name}: {count}")

    return registry_path


def main():
    parser = argparse.ArgumentParser(description="TOC-Bench: Download videos from multiple sources")
    parser.add_argument("--source", type=str, default=None,
                        help="Download only this source (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded without downloading")
    parser.add_argument("--status", action="store_true",
                        help="Just show download status for all sources")
    parser.add_argument("--registry", action="store_true",
                        help="Only create/update the video registry from existing files")
    args = parser.parse_args()

    print("=" * 60)
    print("  TOC-Bench Step 1a: Multi-Source Video Download")
    print(f"  Project root: {ROOT}")
    print("=" * 60)

    # --- Status check ---
    if args.status:
        print("\n📊 Current download status:\n")
        for name, config in DATA_SOURCES.items():
            stat = check_source_status(name, config)
            icon = "✅" if stat["status"] == "ok" else "⚠️" if stat["count"] > 0 else "❌"
            target = config["target_count"]
            print(f"  {icon} {name:20s}  {stat['count']:5d} / {target}  ({stat['status']})")
        print()
        create_video_registry(DATA_SOURCES)
        return

    # --- Registry only ---
    if args.registry:
        create_video_registry(DATA_SOURCES)
        return

    # --- Download ---
    sources_to_process = (
        {args.source: DATA_SOURCES[args.source]}
        if args.source and args.source in DATA_SOURCES
        else DATA_SOURCES
    )

    results = []
    for name, config in sources_to_process.items():
        config["video_subdir"].mkdir(parents=True, exist_ok=True)

        if config.get("auto_download"):
            if name == "perception_test":
                result = download_perception_test(config, args.dry_run)
                results.append(result)
            # Add other auto-downloadable sources here
        else:
            print_manual_instructions(name, config)
            stat = check_source_status(name, config)
            results.append(stat)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("📊 Download Summary")
    print("=" * 60)
    total = 0
    for r in results:
        icon = "✅" if r["status"] in ("ok", "exists") else "⚠️" if r["count"] > 0 else "❌"
        print(f"  {icon} {r['source']:20s}  {r['count']} videos  ({r['status']})")
        total += r["count"]
    print(f"\n  Total: {total} videos")

    # Create registry from whatever we have
    create_video_registry(DATA_SOURCES)

    if total == 0:
        print("\n⚠️  No videos found yet. Please download the datasets following")
        print("   the instructions above, then run:")
        print(f"   python {__file__} --status")


if __name__ == "__main__":
    main()