#!/usr/bin/env python3
"""
identify_invalid.py
-------------------
Reads both LLM judge output files and produces a union list of every
problem / variant pair flagged INVALID by at least one judge.

Union logic: a variant is flagged if Judge A says INVALID  OR  Judge B says INVALID.
(Use --intersection to switch to AND logic instead.)

Outputs
-------
  data/invalid_variants.json   – machine-readable list
  data/invalid_variants.txt    – human-readable summary
  (stdout)                     – printed summary table

Usage
-----
  python3 code/identify_invalid.py
  python3 code/identify_invalid.py --intersection
  python3 code/identify_invalid.py --min-confidence 3
  python3 code/identify_invalid.py --judge-a data/llm_judge_deepseek-r1-32b.json \
                                   --judge-b data/llm_judge_qwen3-32b.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── defaults ──────────────────────────────────────────────────────────────────
DATA_DIR      = Path(__file__).parent.parent / "data"
DEFAULT_JUDGE_A = DATA_DIR / "llm_judge_deepseek-r1-32b.json"
DEFAULT_JUDGE_B = DATA_DIR / "llm_judge_qwen3-32b.json"
OUT_JSON      = DATA_DIR / "invalid_variants.json"
OUT_TXT       = DATA_DIR / "invalid_variants.txt"


def load_judge(path: Path) -> dict:
    if not path.exists():
        print(f"  WARNING: judge file not found: {path}")
        return {}
    with open(path) as f:
        return json.load(f)


def is_invalid(entry: dict, min_confidence: int) -> bool:
    """Return True if the entry is a confident INVALID verdict."""
    if not entry:
        return False
    verdict    = entry.get("verdict", "UNKNOWN").upper()
    confidence = entry.get("confidence", 0)
    if verdict == "INVALID" and confidence >= min_confidence:
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Identify invalid paraphrases from LLM judges.")
    parser.add_argument("--judge-a",        default=str(DEFAULT_JUDGE_A),
                        help="Path to Judge A JSON file")
    parser.add_argument("--judge-b",        default=str(DEFAULT_JUDGE_B),
                        help="Path to Judge B JSON file")
    parser.add_argument("--intersection",   action="store_true",
                        help="Use AND logic (both judges must agree) instead of OR")
    parser.add_argument("--min-confidence", type=int, default=1,
                        help="Minimum confidence score to count a verdict (default: 1, i.e. any verdict)")
    args = parser.parse_args()

    logic_label = "INTERSECTION (both judges agree)" if args.intersection else "UNION (at least one judge)"
    print("=" * 70)
    print("  ARCG — Identify Invalid Paraphrases from LLM Judges")
    print(f"  Logic: {logic_label}")
    print(f"  Min confidence: {args.min_confidence}")
    print("=" * 70)

    judge_a = load_judge(Path(args.judge_a))
    judge_b = load_judge(Path(args.judge_b))

    if not judge_a and not judge_b:
        print("\nERROR: No judge files found. Run script 3 first.")
        sys.exit(1)

    # Collect all problem IDs from both files
    all_problem_ids = sorted(set(list(judge_a.keys()) + list(judge_b.keys())))

    invalid_records   = []   # full detail records
    invalid_problem_ids = set()  # problems where ANY variant is invalid

    variant_stats = {"total": 0, "invalid_union": 0, "invalid_both": 0,
                     "invalid_a_only": 0, "invalid_b_only": 0, "unknown": 0}

    for prob_id in all_problem_ids:
        a_verdicts = judge_a.get(prob_id, {})
        b_verdicts = judge_b.get(prob_id, {})

        # All variant keys across both judges (skip P0 = original)
        all_variants = sorted(set(list(a_verdicts.keys()) + list(b_verdicts.keys())))
        all_variants = [v for v in all_variants if v != "P0"]

        for variant in all_variants:
            variant_stats["total"] += 1

            a_entry = a_verdicts.get(variant, {})
            b_entry = b_verdicts.get(variant, {})

            a_invalid = is_invalid(a_entry, args.min_confidence)
            b_invalid = is_invalid(b_entry, args.min_confidence)

            a_unknown = a_entry.get("verdict", "UNKNOWN").upper() in ("UNKNOWN", "")
            b_unknown = b_entry.get("verdict", "UNKNOWN").upper() in ("UNKNOWN", "")

            if a_unknown or b_unknown:
                variant_stats["unknown"] += 1

            both_invalid  = a_invalid and b_invalid
            union_invalid = a_invalid or b_invalid

            if union_invalid:
                variant_stats["invalid_union"] += 1
            if both_invalid:
                variant_stats["invalid_both"] += 1
            if a_invalid and not b_invalid:
                variant_stats["invalid_a_only"] += 1
            if b_invalid and not a_invalid:
                variant_stats["invalid_b_only"] += 1

            # Apply the chosen logic
            flagged = both_invalid if args.intersection else union_invalid

            if flagged:
                invalid_problem_ids.add(prob_id)
                record = {
                    "problem_id": prob_id,
                    "variant":    variant,
                    "judge_a": {
                        "verdict":    a_entry.get("verdict", "N/A"),
                        "confidence": a_entry.get("confidence", 0),
                        "reason":     a_entry.get("reason", ""),
                    },
                    "judge_b": {
                        "verdict":    b_entry.get("verdict", "N/A"),
                        "confidence": b_entry.get("confidence", 0),
                        "reason":     b_entry.get("reason", ""),
                    },
                }
                invalid_records.append(record)

    # ── Print summary ──────────────────────────────────────────────────────────
    flagged_count = len(invalid_records)
    print(f"\n{'─'*70}")
    print(f"  Problems evaluated : {len(all_problem_ids)}")
    print(f"  Variants evaluated : {variant_stats['total']}")
    print(f"  UNKNOWN verdicts   : {variant_stats['unknown']}")
    print(f"  Invalid (union)    : {variant_stats['invalid_union']}")
    print(f"  Invalid (both)     : {variant_stats['invalid_both']}")
    print(f"  Invalid (A only)   : {variant_stats['invalid_a_only']}")
    print(f"  Invalid (B only)   : {variant_stats['invalid_b_only']}")
    print(f"{'─'*70}")
    print(f"  Flagged by [{logic_label}]: {flagged_count} variants")
    print(f"  Affected problems  : {len(invalid_problem_ids)}")
    print(f"{'─'*70}\n")

    if not invalid_records:
        print("  No invalid variants found. All paraphrases passed.")
    else:
        # Print table
        col_w = [40, 8, 10, 10, 50]
        header = f"{'Problem ID':<{col_w[0]}} {'Variant':<{col_w[1]}} {'Judge A':<{col_w[2]}} {'Judge B':<{col_w[3]}} {'Reason (A | B)'}"
        print(header)
        print("─" * (sum(col_w) + 4))
        for r in invalid_records:
            reason_a = r["judge_a"]["reason"][:40] if r["judge_a"]["reason"] else "—"
            reason_b = r["judge_b"]["reason"][:40] if r["judge_b"]["reason"] else "—"
            reason   = f"{reason_a} | {reason_b}"
            print(
                f"{r['problem_id']:<{col_w[0]}} "
                f"{r['variant']:<{col_w[1]}} "
                f"{r['judge_a']['verdict']:<{col_w[2]}} "
                f"{r['judge_b']['verdict']:<{col_w[3]}} "
                f"{reason}"
            )

    # ── Write outputs ──────────────────────────────────────────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "logic":              "intersection" if args.intersection else "union",
        "min_confidence":     args.min_confidence,
        "flagged_count":      flagged_count,
        "affected_problems":  sorted(invalid_problem_ids),
        "invalid_variants":   invalid_records,
        "stats":              variant_stats,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {OUT_JSON}")

    with open(OUT_TXT, "w") as f:
        f.write(f"ARCG Invalid Paraphrases — {logic_label}\n")
        f.write(f"Min confidence: {args.min_confidence}\n")
        f.write("=" * 70 + "\n\n")
        if not invalid_records:
            f.write("No invalid variants found.\n")
        else:
            for r in invalid_records:
                f.write(f"Problem : {r['problem_id']}\n")
                f.write(f"Variant : {r['variant']}\n")
                f.write(f"Judge A : {r['judge_a']['verdict']} (conf={r['judge_a']['confidence']}) — {r['judge_a']['reason']}\n")
                f.write(f"Judge B : {r['judge_b']['verdict']} (conf={r['judge_b']['confidence']}) — {r['judge_b']['reason']}\n")
                f.write("\n")
        f.write(f"\nAffected problem IDs ({len(invalid_problem_ids)}):\n")
        for pid in sorted(invalid_problem_ids):
            f.write(f"  {pid}\n")
    print(f"  Saved: {OUT_TXT}")

    # ── Convenience: print just the problem IDs for easy copy-paste ───────────
    if invalid_problem_ids:
        print(f"\n  Affected problem IDs (for use with remove_problems.py):")
        print("  " + " ".join(sorted(invalid_problem_ids)))


if __name__ == "__main__":
    main()
