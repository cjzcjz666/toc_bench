#!/usr/bin/env python3
"""
TOC-Bench Filter Diagnostic
============================
Reads all step4 layer results + step3c output and produces a single
report that answers:

  1. Where did each dim die? Which layer contributed how many fails?
  2. Did the tier distribution collapse in step3b, step3c, or step4?
  3. For dims with suspicious survival (very low or very high), what does
     Qwen actually predict?
  4. For numerical: distribution of Qwen's predictions by layer, broken
     down by correct_answer, so we can tell if e.g. all "5 or more" items
     are being killed by text-only prior.
  5. Per-dim breakdown of layer-by-layer pass rates (like a funnel).

Usage:
    python scripts/diagnose_filter_results.py
    python scripts/diagnose_filter_results.py --verbose  # per-item prints
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import QA_DIR, DIMENSIONS

FILTER_DIR = QA_DIR / "filtered_natural"
QA_ITEMS_DIR = QA_DIR / "qa_items_natural"


def _load_json(path):
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [WARN] Could not load {path}: {e}")
        return None


def _items_from_layer_file(path):
    """Return a dict of qa_id → item from a layer result file. Handles
    shard file name too."""
    data = _load_json(path)
    if data is None:
        return {}
    out = {}
    for it in data.get("items", []):
        qid = it.get("qa_id")
        if qid:
            out[qid] = it
    return out


def _items_from_step3c():
    """Collect all QA items from step3c output."""
    all_items = {}
    all_json = QA_ITEMS_DIR / "_all.json"
    if all_json.exists():
        data = _load_json(all_json)
        if data:
            for it in data.get("qa_items", []):
                qid = it.get("qa_id")
                if qid:
                    all_items[qid] = it
    else:
        # Fallback: scan per-video files
        for f in QA_ITEMS_DIR.glob("*.json"):
            if f.name.startswith("_"):
                continue
            data = _load_json(f)
            if data:
                for it in data.get("qa_items", []):
                    qid = it.get("qa_id")
                    if qid:
                        all_items[qid] = it
    return all_items


def _ok(v):
    """Extract layer `passed` from a layer_result dict."""
    if not isinstance(v, dict):
        return True
    return bool(v.get("passed", True))


# ============================================================
# Diagnostic sections
# ============================================================

def section_tier_pipeline(items_3c, layer_items):
    """Track tier distribution at each stage of the pipeline."""
    print("\n" + "=" * 72)
    print("  SECTION 1: Tier distribution across pipeline stages")
    print("=" * 72)
    print("  Reveals where the tier2 imbalance originated.\n")

    # Stage A: step3c output (pre-filter)
    c_3c_tier = Counter(it.get("tier", "?") for it in items_3c.values())
    c_3c_dim = Counter(it.get("dim", "?") for it in items_3c.values())

    # Stage B: layer1 pass
    layer1 = layer_items.get("text_only", {})
    l1_pass_ids = {qid for qid, it in layer1.items()
                   if _ok(it.get("layer1_result"))}

    # Stage C: layer2 pass
    layer2 = layer_items.get("single_frame", {})
    l2_pass_ids = {qid for qid, it in layer2.items()
                   if _ok(it.get("layer2_result"))}

    # Stage D: layer3 pass
    layer3 = layer_items.get("shuffled", {})
    l3_pass_ids = {qid for qid, it in layer3.items()
                   if _ok(it.get("layer3_result"))}

    # Stage E: combined (all layers pass)
    combined_ids = l1_pass_ids & l2_pass_ids & l3_pass_ids

    tier_counts_by_stage = {
        "step3c out":   Counter(),
        "+ layer1":     Counter(),
        "+ layer2":     Counter(),
        "+ layer3":     Counter(),
        "combined":     Counter(),
    }

    for qid, it in items_3c.items():
        tier = it.get("tier", "?")
        tier_counts_by_stage["step3c out"][tier] += 1
        if qid in l1_pass_ids:
            tier_counts_by_stage["+ layer1"][tier] += 1
        if qid in l2_pass_ids:
            tier_counts_by_stage["+ layer2"][tier] += 1
        if qid in l3_pass_ids:
            tier_counts_by_stage["+ layer3"][tier] += 1
        if qid in combined_ids:
            tier_counts_by_stage["combined"][tier] += 1

    print(f"  {'Stage':<14s} {'tier1':>8s} {'tier2':>8s} {'tier3':>8s} "
          f"{'total':>8s} | {'t1%':>5s} {'t2%':>5s} {'t3%':>5s}")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*8} | "
          f"{'-'*5} {'-'*5} {'-'*5}")
    for stage, cnts in tier_counts_by_stage.items():
        t1, t2, t3 = cnts.get("tier1", 0), cnts.get("tier2", 0), cnts.get("tier3", 0)
        total = t1 + t2 + t3
        if total > 0:
            p1, p2, p3 = [100 * x / total for x in (t1, t2, t3)]
        else:
            p1 = p2 = p3 = 0
        print(f"  {stage:<14s} {t1:>8,d} {t2:>8,d} {t3:>8,d} {total:>8,d} | "
              f"{p1:>4.1f}% {p2:>4.1f}% {p3:>4.1f}%")

    print("\n  Target per design (step3b): tier1=32% / tier2=43% / tier3=25%")
    # Diagnosis: which stage diverged most?
    t2_at_step3c_pct = 100 * tier_counts_by_stage["step3c out"]["tier2"] / max(
        1, sum(tier_counts_by_stage["step3c out"].values())
    )
    t2_final_pct = 100 * tier_counts_by_stage["combined"]["tier2"] / max(
        1, sum(tier_counts_by_stage["combined"].values())
    )
    print(f"\n  Diagnosis:")
    if t2_at_step3c_pct < 20:
        print(f"    → tier2 was ALREADY under-represented at step3c exit "
              f"({t2_at_step3c_pct:.1f}% vs 43% target)")
        print(f"    → Root cause is upstream of step4 (step3a/3b unit generation).")
    else:
        print(f"    → tier2 was OK at step3c exit ({t2_at_step3c_pct:.1f}%) "
              f"but lost in filter layers ({t2_final_pct:.1f}% combined)")


def section_layer_kill_attribution(layer_items, items_3c):
    """For each dim, how many items did each layer kill?"""
    print("\n" + "=" * 72)
    print("  SECTION 2: Per-dim layer kill attribution")
    print("=" * 72)
    print("  For each dim: how many items did each layer fail (independently)?")
    print("  Sum may exceed total fails because layers are not disjoint.\n")

    layer1 = layer_items.get("text_only", {})
    layer2 = layer_items.get("single_frame", {})
    layer3 = layer_items.get("shuffled", {})

    # For each qa_id, check each layer's pass
    dim_fail_by_layer = defaultdict(lambda: {"l1": 0, "l2": 0, "l3": 0,
                                               "total": 0})
    for qid, it in items_3c.items():
        dim = it.get("dim", "?")
        dim_fail_by_layer[dim]["total"] += 1
        l1_it = layer1.get(qid)
        l2_it = layer2.get(qid)
        l3_it = layer3.get(qid)
        if l1_it and not _ok(l1_it.get("layer1_result")):
            dim_fail_by_layer[dim]["l1"] += 1
        if l2_it and not _ok(l2_it.get("layer2_result")):
            dim_fail_by_layer[dim]["l2"] += 1
        if l3_it and not _ok(l3_it.get("layer3_result")):
            dim_fail_by_layer[dim]["l3"] += 1

    print(f"  {'dim':<28s} {'total':>7s}  "
          f"{'L1 fail':>10s} {'L2 fail':>10s} {'L3 fail':>10s}")
    print(f"  {'-'*28} {'-'*7}  {'-'*10} {'-'*10} {'-'*10}")
    for dim in sorted(dim_fail_by_layer.keys(),
                       key=lambda d: -dim_fail_by_layer[d]["total"]):
        d = dim_fail_by_layer[dim]
        tot = d["total"]
        c_mark = "★" if DIMENSIONS.get(dim, {}).get("c_critical") else " "
        print(f"  {c_mark} {dim:<26s} {tot:>7,d}  "
              f"{d['l1']:>6,d} ({100*d['l1']/max(1,tot):>4.1f}%)  "
              f"{d['l2']:>6,d} ({100*d['l2']/max(1,tot):>4.1f}%)  "
              f"{d['l3']:>6,d} ({100*d['l3']/max(1,tot):>4.1f}%)")


def section_numerical_prediction_diagnosis(layer_items, items_3c):
    """For event_count items: what does Qwen actually predict at each layer?"""
    print("\n" + "=" * 72)
    print("  SECTION 3: Numerical (event_count) prediction diagnosis")
    print("=" * 72)
    print("  Breaks down Qwen's actual predictions per layer per correct_answer.")
    print("  A dim with systematic prediction bias (e.g. always '5 or more')")
    print("  is likely being filtered by world-prior rather than real leakage.\n")

    layer1 = layer_items.get("text_only", {})
    layer2 = layer_items.get("single_frame", {})
    layer3 = layer_items.get("shuffled", {})

    # Group event_count items by correct_answer
    ec_by_ans = defaultdict(list)
    for qid, it in items_3c.items():
        if it.get("dim") != "event_count":
            continue
        ans = it.get("correct_answer", "?")
        ec_by_ans[ans].append(qid)

    for ans in sorted(ec_by_ans.keys()):
        qids = ec_by_ans[ans]
        print(f"  Correct answer = {ans!r}   ({len(qids)} items)")

        # Layer 1 predictions
        l1_preds = Counter()
        l1_correct = 0
        for qid in qids:
            it = layer1.get(qid)
            if not it:
                continue
            r = it.get("layer1_result", {})
            pred = r.get("predicted")
            l1_preds[str(pred)] += 1
            if r.get("is_correct"):
                l1_correct += 1
        print(f"    Layer1 (text_only) Qwen predicted: {dict(l1_preds)}")
        print(f"    Layer1 correct (→ failed filter): {l1_correct}/{len(qids)}")

        # Layer 2
        l2_preds = Counter()
        l2_fails = 0
        for qid in qids:
            it = layer2.get(qid)
            if not it:
                continue
            r = it.get("layer2_result", {})
            for p in r.get("predictions", []):
                l2_preds[str(p.get("predicted"))] += 1
            if not r.get("passed", True):
                l2_fails += 1
        if l2_preds:
            top = l2_preds.most_common(5)
            print(f"    Layer2 (single_frame) prediction counts (top 5): {top}")
        print(f"    Layer2 failed filter: {l2_fails}/{len(qids)}")

        # Layer 3
        l3_pass = 0
        l3_total = 0
        for qid in qids:
            it = layer3.get(qid)
            if not it:
                continue
            r = it.get("layer3_result", {})
            l3_total += 1
            if r.get("passed", True):
                l3_pass += 1
        print(f"    Layer3 (shuffled) passed: {l3_pass}/{l3_total}")
        print()


def section_reappear_identity_diagnosis(layer_items, items_3c):
    """Why did reappear_identity go 0/37?"""
    print("\n" + "=" * 72)
    print("  SECTION 4: reappear_identity diagnosis (0/37 survival)")
    print("=" * 72)

    rid_ids = [qid for qid, it in items_3c.items()
               if it.get("dim") == "reappear_identity"]
    if not rid_ids:
        print("  No reappear_identity items found in step3c output.")
        return

    print(f"  Found {len(rid_ids)} reappear_identity items.\n")

    # correct_answer distribution (should all be "same one"-style under Option α)
    ans_dist = Counter()
    for qid in rid_ids:
        ans_dist[items_3c[qid].get("correct_answer", "?")] += 1
    print(f"  correct_answer distribution: {dict(ans_dist)}")
    print(f"  (Expected: all 'A' or all 'B' depending on statement shuffle.)\n")

    layer1 = layer_items.get("text_only", {})
    layer2 = layer_items.get("single_frame", {})
    layer3 = layer_items.get("shuffled", {})

    l1_fail = sum(1 for qid in rid_ids
                  if qid in layer1 and not _ok(layer1[qid].get("layer1_result")))
    l2_fail = sum(1 for qid in rid_ids
                  if qid in layer2 and not _ok(layer2[qid].get("layer2_result")))
    l3_fail = sum(1 for qid in rid_ids
                  if qid in layer3 and not _ok(layer3[qid].get("layer3_result")))

    print(f"  Layer1 (text_only) failures:  {l1_fail}/{len(rid_ids)} "
          f"({100*l1_fail/max(1,len(rid_ids)):.1f}%)")
    print(f"  Layer2 (single_frame) fails:  {l2_fail}/{len(rid_ids)} "
          f"({100*l2_fail/max(1,len(rid_ids)):.1f}%)")
    print(f"  Layer3 (shuffled) failures:   {l3_fail}/{len(rid_ids)} "
          f"({100*l3_fail/max(1,len(rid_ids)):.1f}%)")

    # Layer1 Qwen predictions (since this is the main culprit under Option α)
    l1_preds = Counter()
    for qid in rid_ids:
        it = layer1.get(qid)
        if it:
            pred = it.get("layer1_result", {}).get("predicted")
            l1_preds[str(pred)] += 1
    print(f"\n  Layer1 Qwen prediction distribution: {dict(l1_preds)}")


def section_relative_spatial_change_diagnosis(layer_items, items_3c):
    """Why is relative_spatial_change at 10.6%?"""
    print("\n" + "=" * 72)
    print("  SECTION 5: relative_spatial_change diagnosis (10.6% survival)")
    print("=" * 72)

    rsc_ids = [qid for qid, it in items_3c.items()
               if it.get("dim") == "relative_spatial_change"]
    if not rsc_ids:
        print("  No items found.")
        return

    print(f"  Found {len(rsc_ids)} items.\n")

    layer1 = layer_items.get("text_only", {})
    layer2 = layer_items.get("single_frame", {})
    layer3 = layer_items.get("shuffled", {})

    l1_fail = sum(1 for qid in rsc_ids
                  if qid in layer1 and not _ok(layer1[qid].get("layer1_result")))
    l2_fail = sum(1 for qid in rsc_ids
                  if qid in layer2 and not _ok(layer2[qid].get("layer2_result")))
    l3_fail = sum(1 for qid in rsc_ids
                  if qid in layer3 and not _ok(layer3[qid].get("layer3_result")))

    print(f"  Layer1 fails: {l1_fail}/{len(rsc_ids)} "
          f"({100*l1_fail/max(1,len(rsc_ids)):.1f}%)")
    print(f"  Layer2 fails: {l2_fail}/{len(rsc_ids)} "
          f"({100*l2_fail/max(1,len(rsc_ids)):.1f}%)")
    print(f"  Layer3 fails: {l3_fail}/{len(rsc_ids)} "
          f"({100*l3_fail/max(1,len(rsc_ids)):.1f}%)")

    # By correct_answer
    ans_dist = Counter()
    ans_pass = Counter()
    combined_pass_ids = (
        {qid for qid, it in layer1.items() if _ok(it.get("layer1_result"))} &
        {qid for qid, it in layer2.items() if _ok(it.get("layer2_result"))} &
        {qid for qid, it in layer3.items() if _ok(it.get("layer3_result"))}
    )
    for qid in rsc_ids:
        it = items_3c[qid]
        # In mcq_4 the correct_answer is a letter; we want the ACTUAL direction
        # string. Find it from the option with that letter.
        letter = it.get("correct_answer", "?")
        direction = it.get(f"option_{letter}", "?")
        ans_dist[direction] += 1
        if qid in combined_pass_ids:
            ans_pass[direction] += 1

    print(f"\n  By correct direction (total → passed):")
    for direction in sorted(ans_dist.keys()):
        tot = ans_dist[direction]
        p = ans_pass.get(direction, 0)
        rate = 100 * p / max(1, tot)
        print(f"    {direction:<48s}  {p:>4,d}/{tot:>4,d}  ({rate:.1f}%)")


def section_hallucination_survival(items_3c, combined_ids):
    """How did hallucination items fare?"""
    print("\n" + "=" * 72)
    print("  SECTION 6: Hallucination item survival")
    print("=" * 72)
    print("  Are hallucination items (variant_a / variant_b) over- or")
    print("  under-represented in the final benchmark?\n")

    hallu_total = Counter()
    hallu_pass = Counter()
    for qid, it in items_3c.items():
        tag = it.get("hallucination")
        if it.get("has_hallucination_distractor"):
            tag = tag or "distractor_only"
        key = tag or "none"
        hallu_total[key] += 1
        if qid in combined_ids:
            hallu_pass[key] += 1

    print(f"  {'hallucination type':<24s} {'total':>8s} {'passed':>8s} {'rate':>8s}")
    print(f"  {'-'*24} {'-'*8} {'-'*8} {'-'*8}")
    for key in sorted(hallu_total.keys()):
        tot = hallu_total[key]
        pas = hallu_pass.get(key, 0)
        rate = 100 * pas / max(1, tot)
        print(f"  {key:<24s} {tot:>8,d} {pas:>8,d} {rate:>7.1f}%")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print("  TOC-Bench Filter Result Diagnostic")
    print("=" * 72)

    # Load step3c output
    print("\n  Loading step3c output...")
    items_3c = _items_from_step3c()
    print(f"  Loaded {len(items_3c):,} items from step3c output")

    # Load layer results
    print("  Loading layer results...")
    layer_items = {
        "text_only": _items_from_layer_file(
            FILTER_DIR / "layer1_text_only.json"),
        "single_frame": _items_from_layer_file(
            FILTER_DIR / "layer2_single_frame.json"),
        "shuffled": _items_from_layer_file(
            FILTER_DIR / "layer3_shuffled.json"),
    }
    for name, d in layer_items.items():
        print(f"    {name}: {len(d):,} items")

    # Combined pass set
    combined_ids = set(items_3c.keys())
    for layer_key, result_key in [("text_only", "layer1_result"),
                                    ("single_frame", "layer2_result"),
                                    ("shuffled", "layer3_result")]:
        layer = layer_items.get(layer_key, {})
        if layer:
            passed_ids = {qid for qid, it in layer.items()
                          if _ok(it.get(result_key))}
            combined_ids &= passed_ids

    print(f"\n  Combined pass: {len(combined_ids):,} items")

    # Run all diagnostic sections
    section_tier_pipeline(items_3c, layer_items)
    section_layer_kill_attribution(layer_items, items_3c)
    section_numerical_prediction_diagnosis(layer_items, items_3c)
    section_reappear_identity_diagnosis(layer_items, items_3c)
    section_relative_spatial_change_diagnosis(layer_items, items_3c)
    section_hallucination_survival(items_3c, combined_ids)

    print("\n" + "=" * 72)
    print("  Diagnostic complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()