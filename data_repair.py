#!/usr/bin/env python3
"""
TOC-Bench Data Repair
=====================
Two fixes for toc_bench_clean.json:

  A1. correct_answer='d' (lowercase) on 1 item → uppercase to 'D'
  A2. SP imbalance: 121 A vs 66 B (always-A baseline = 64.7%)
      → randomly flip A↔B on excess A-correct items so distribution is ~50/50.
        Flipping is purely a label swap (statement_A ↔ statement_B,
        correct_answer toggled). Question text and option semantics are
        identical — only display order changes.

Output: toc_bench_clean_v2.json (same schema, with metadata.repair_log
added per item that was modified).

Usage:
    python data_repair.py \
        --in /mnt/user-data/uploads/toc_bench_clean.json \
        --out /mnt/user-data/outputs/toc_bench_clean_v2.json \
        --seed 42
"""

import argparse
import json
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path


def fix_lowercase_answer(items):
    """A1: uppercase any stray 'a'/'b'/'c'/'d' in correct_answer."""
    fixed = 0
    for it in items:
        ans = it.get("correct_answer")
        if ans and ans in ("a", "b", "c", "d"):
            it["correct_answer"] = ans.upper()
            it.setdefault("metadata", {}).setdefault("repair_log", []).append(
                f"A1: lowercased {ans!r} → {ans.upper()!r}"
            )
            fixed += 1
    return fixed


def rebalance_sp_answers(items, rng):
    """A2: SP A/B distribution → ~50/50 by flipping excess.

    Strategy:
      - Find SP items: format == 'sp'
      - Count A vs B
      - Determine how many A→B flips needed: (n_A - n_B) // 2
      - Randomly pick that many A-correct items and swap statement_A↔B
      - After swap, correct_answer becomes 'B'
    Question text is preserved; only the option labels are exchanged.
    """
    sp_items = [it for it in items if it.get("format") == "sp"]
    n_a = sum(1 for it in sp_items if it["correct_answer"] == "A")
    n_b = sum(1 for it in sp_items if it["correct_answer"] == "B")

    n_flip = max(0, (n_a - n_b) // 2)
    if n_flip == 0:
        return 0, n_a, n_b, n_a, n_b

    a_items = [it for it in sp_items if it["correct_answer"] == "A"]
    rng.shuffle(a_items)
    flipped = 0
    for it in a_items[:n_flip]:
        # Swap statement_A and statement_B
        sa = it.get("statement_A")
        sb = it.get("statement_B")
        if sa is None or sb is None:
            # SP items in this dataset use option_A/option_B keys?
            # Check both naming conventions to be safe.
            sa = it.get("option_A")
            sb = it.get("option_B")
            if sa is None or sb is None:
                continue
            it["option_A"], it["option_B"] = sb, sa
        else:
            it["statement_A"], it["statement_B"] = sb, sa
        it["correct_answer"] = "B"
        it.setdefault("metadata", {}).setdefault("repair_log", []).append(
            "A2: SP A↔B flipped to balance"
        )
        flipped += 1

    # Recount after flip
    n_a2 = sum(1 for it in sp_items if it["correct_answer"] == "A")
    n_b2 = sum(1 for it in sp_items if it["correct_answer"] == "B")
    return flipped, n_a, n_b, n_a2, n_b2


def rebalance_mcq4_answers(items, rng):
    """A4: MCQ_4 A/B/C/D distribution → ~25%/25%/25%/25%.

    Strategy: greedy label rotation.
      - Compute fair share = total/4
      - For letters in surplus (count > fair_share), swap items with
        letters in deficit (count < fair_share)
      - Each swap exchanges the entire option_X / option_Y text and
        updates correct_answer accordingly. Question and option semantics
        are preserved; only the label assigned to each option changes.

    Note: this preserves hallucination-option scrambling — if an option
    text was "there is no X in this video", it just moves to a different
    letter. The hallucination structure is intact.
    """
    mcq_items = [it for it in items if it.get("format") == "mcq_4"]
    n = len(mcq_items)
    if n == 0:
        return 0, {}, {}

    # Count per letter
    def count_by_letter(target_items):
        from collections import Counter
        return Counter(it["correct_answer"] for it in target_items)

    before = dict(count_by_letter(mcq_items))
    fair = n / 4.0

    # Build per-letter surplus/deficit
    swaps_done = 0
    rng.shuffle(mcq_items)  # randomize iteration order

    # Iterative greedy: while there's any letter with surplus > 1, find a
    # surplus item and swap with a deficit letter
    max_iter = n * 2  # safety bound
    for _ in range(max_iter):
        cur_counts = count_by_letter(mcq_items)
        # Surplus letters (count > fair, with tolerance > 1)
        surplus = sorted(
            [(L, c - fair) for L, c in cur_counts.items() if c - fair > 0.5],
            key=lambda x: -x[1],
        )
        deficit = sorted(
            [(L, fair - c) for L, c in cur_counts.items() if fair - c > 0.5],
            key=lambda x: -x[1],
        )
        if not surplus or not deficit:
            break

        # Take the most-surplus letter and most-deficit letter
        surplus_letter = surplus[0][0]
        deficit_letter = deficit[0][0]

        # Find an item where correct_answer == surplus_letter
        # and swap it with deficit_letter
        for it in mcq_items:
            if it["correct_answer"] != surplus_letter:
                continue
            # Swap option_<surplus> and option_<deficit>
            key_s = f"option_{surplus_letter}"
            key_d = f"option_{deficit_letter}"
            if key_s in it and key_d in it:
                it[key_s], it[key_d] = it[key_d], it[key_s]
                it["correct_answer"] = deficit_letter
                it.setdefault("metadata", {}).setdefault("repair_log", []).append(
                    f"A4: MCQ_4 {surplus_letter}↔{deficit_letter} swap"
                )
                swaps_done += 1
                break
        else:
            # No item found — shouldn't happen but break to avoid infinite loop
            break

    after = dict(count_by_letter(mcq_items))
    return swaps_done, before, after


KEEP_INVALID_ORDERING_QA_IDS = {
    "7d73a701e7a6",
    "e258822dd00a",
}


def drop_invalid_ordering(items):
    """A3: drop ordering items whose correct_order has duplicate letters or
    wrong length (these are ground-truth bugs; can't fix without watching
    the video).
    """
    kept = []
    dropped = []
    kept_invalid = []
    for it in items:
        fmt = it.get("format", "")
        if not fmt.startswith("ordering"):
            kept.append(it)
            continue
        co = it.get("correct_order") or []
        expected_k = 3 if fmt == "ordering_3" else 4
        is_valid = (
            len(co) == expected_k and
            len(set(co)) == expected_k and
            all(c in "ABCD" for c in co)
        )
        qa_id = it.get("qa_id")
        if is_valid:
            kept.append(it)
        elif qa_id in KEEP_INVALID_ORDERING_QA_IDS:
            kept.append(it)
            kept_invalid.append({
                "qa_id": qa_id,
                "format": fmt,
                "correct_order": co,
                "reason": (
                    "duplicate letters" if len(set(co)) != len(co)
                    else "wrong length" if len(co) != expected_k
                    else "non-letter values"
                ),
            })
        else:
            dropped.append({
                "qa_id": qa_id,
                "format": fmt,
                "correct_order": co,
                "reason": (
                    "duplicate letters" if len(set(co)) != len(co)
                    else "wrong length" if len(co) != expected_k
                    else "non-letter values"
                ),
            })
    return kept, dropped, kept_invalid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    with open(args.in_path) as f:
        data = json.load(f)
    data = deepcopy(data)
    items = data["items"]

    print(f"Loaded {len(items)} items from {args.in_path}")
    print()

    # === Fix A1 ===
    n_fixed_a1 = fix_lowercase_answer(items)
    print(f"[A1] Lowercase correct_answer fixed: {n_fixed_a1}")

    # === Fix A2 ===
    n_flipped, before_a, before_b, after_a, after_b = rebalance_sp_answers(items, rng)
    print(f"[A2] SP A/B before: {before_a}/{before_b} "
          f"(always-A baseline = {before_a/(before_a+before_b)*100:.1f}%)")
    print(f"     SP A/B after:  {after_a}/{after_b} "
          f"(always-A baseline = {after_a/(after_a+after_b)*100:.1f}%)")
    print(f"     Items flipped: {n_flipped}")

    # === Fix A3 ===
    items, dropped, kept_invalid = drop_invalid_ordering(items)
    data["items"] = items
    data["total_items"] = len(items)
    print(f"[A3] Invalid ordering items dropped: {len(dropped)}")
    for d in dropped:
        print(f"     - {d['qa_id']} [{d['format']}] {d['correct_order']} ({d['reason']})")
    print(f"[A3] Invalid ordering items kept by whitelist: {len(kept_invalid)}")
    for d in kept_invalid:
        print(f"     + {d['qa_id']} [{d['format']}] {d['correct_order']} ({d['reason']})")

    # === Fix A4 ===
    n_swaps, before_dist, after_dist = rebalance_mcq4_answers(items, rng)
    print(f"[A4] MCQ_4 A/B/C/D before: {before_dist}")
    print(f"     MCQ_4 A/B/C/D after:  {after_dist}")
    print(f"     Items swapped: {n_swaps}")
    if before_dist:
        max_letter_before = max(before_dist.values())
        max_letter_after = max(after_dist.values())
        total = sum(after_dist.values())
        print(f"     Always-most-common baseline: "
              f"{max_letter_before/total*100:.1f}% → {max_letter_after/total*100:.1f}%")
    print()

    # Update top-level version + add repair note
    data["version"] = data.get("version", "") + "+repaired"
    data["repair_summary"] = {
        "a1_lowercase_fixed": n_fixed_a1,
        "a2_sp_flipped": n_flipped,
        "a2_sp_before": {"A": before_a, "B": before_b},
        "a2_sp_after": {"A": after_a, "B": after_b},
        "a3_ordering_dropped": len(dropped),
        "a3_dropped_qa_ids": [d["qa_id"] for d in dropped],
        "a3_kept_invalid_qa_ids": [d["qa_id"] for d in kept_invalid],
        "a4_mcq4_swaps": n_swaps,
        "a4_mcq4_before": before_dist,
        "a4_mcq4_after": after_dist,
        "seed": args.seed,
    }

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {args.out_path}")

    # Final sanity dump
    print("\nFinal answer distribution:")
    by_fmt = Counter()
    for it in items:
        fmt = it["format"]
        ans = it.get("correct_answer") or ",".join(it.get("correct_order", []))
        by_fmt[(fmt, ans)] += 1
    for (fmt, ans), n in sorted(by_fmt.items()):
        print(f"  {fmt:<12s} {ans!r:<15s} {n}")


if __name__ == "__main__":
    main()