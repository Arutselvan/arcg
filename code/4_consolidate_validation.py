"""
ARCG Experiment  --  Step 4: Consolidate Validation Reports
============================================================
Merges the four validation sources:
  - Human Annotator 1:  data/human_validation_annotator1.xlsx  (filled)
  - Human Annotator 2:  data/human_validation_annotator2.xlsx  (filled)
  - LLM Judge A:        data/llm_judge_deepseek-r1-70b.json
  - LLM Judge B:        data/llm_judge_qwen3-32b.json

For each paraphrase, computes:
  - Individual verdicts from all 4 sources
  - Majority vote (3 of 4 or 4 of 4 required to accept)
  - Pairwise Cohen's Kappa for all 6 annotator pairs
  - Overall inter-annotator agreement statistics

Outputs:
  data/validated_paraphrases.json   -- accepted paraphrases only
  data/validation_report.json       -- full statistics and per-item details
  data/validation_summary.txt       -- human-readable summary for the paper

Acceptance policy:
  A paraphrase is ACCEPTED if at least 3 of the 4 sources mark it VALID.
  Problems where fewer than 3 paraphrases are accepted are flagged.

Requirements
------------
  pip install openpyxl scipy

Usage
-----
  python 4_consolidate_validation.py
"""

import json
import os
import sys
from collections import defaultdict

import openpyxl

DATA_DIR         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
PARAPHRASE_FILE  = os.path.join(DATA_DIR, "paraphrases.json")

HUMAN_FILES = [
    os.path.join(DATA_DIR, "human_validation_annotator1.xlsx"),
    os.path.join(DATA_DIR, "human_validation_annotator2.xlsx"),
]

LLM_JUDGE_FILES = [
    os.path.join(DATA_DIR, "llm_judge_deepseek-r1-70b.json"),
    os.path.join(DATA_DIR, "llm_judge_qwen3-32b.json"),
]

OUTPUT_VALIDATED  = os.path.join(DATA_DIR, "validated_paraphrases.json")
OUTPUT_REPORT     = os.path.join(DATA_DIR, "validation_report.json")
OUTPUT_SUMMARY    = os.path.join(DATA_DIR, "validation_summary.txt")

ACCEPTANCE_THRESHOLD = 3   # out of 4 sources must agree VALID

# ---------------------------------------------------------------------------
# Cohen's Kappa
# ---------------------------------------------------------------------------

def cohens_kappa(labels_a: list[int], labels_b: list[int]) -> float:
    """
    Compute Cohen's Kappa for two binary label sequences (1=VALID, 0=INVALID).
    Returns kappa in [-1, 1].
    """
    assert len(labels_a) == len(labels_b), "Label lists must be the same length."
    n = len(labels_a)
    if n == 0:
        return 0.0

    # Observed agreement
    p_o = sum(a == b for a, b in zip(labels_a, labels_b)) / n

    # Expected agreement
    p_a1 = sum(labels_a) / n
    p_b1 = sum(labels_b) / n
    p_e  = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)

    if p_e == 1.0:
        return 1.0

    return (p_o - p_e) / (1 - p_e)


def kappa_interpretation(k: float) -> str:
    if k < 0:
        return "poor (less than chance)"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"

# ---------------------------------------------------------------------------
# Load human annotations from Excel
# ---------------------------------------------------------------------------

def load_human_excel(path: str) -> dict:
    """
    Returns dict: {problem_id: {paraphrase_id: "YES"/"NO"/"" }}
    """
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found. Skipping.")
        return {}

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["Annotations"]

    results = defaultdict(dict)
    header  = None

    for row in ws.iter_rows(values_only=True):
        if header is None:
            header = [str(c).strip().lower() if c else "" for c in row]
            continue
        if not any(row):
            continue

        row_dict = dict(zip(header, row))
        pid  = str(row_dict.get("problem_id", "")).strip()
        vid  = str(row_dict.get("paraphrase_id", "")).strip()
        val  = str(row_dict.get("valid (yes/no)", "") or "").strip().upper()

        if pid and vid:
            results[pid][vid] = val

    return dict(results)


# ---------------------------------------------------------------------------
# Load LLM judge JSON
# ---------------------------------------------------------------------------

def load_llm_judge(path: str) -> dict:
    """
    Returns dict: {problem_id: {paraphrase_id: "VALID"/"INVALID"}}
    """
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found. Skipping.")
        return {}

    with open(path) as f:
        raw = json.load(f)

    results = {}
    for pid, variants in raw.items():
        results[pid] = {}
        for vid, info in variants.items():
            verdict = info.get("verdict", "UNKNOWN").upper()
            # Normalise to YES/NO
            results[pid][vid] = "YES" if verdict == "VALID" else "NO"

    return results


# ---------------------------------------------------------------------------
# Main consolidation
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("ARCG Step 4: Consolidate Validation Reports")
    print("=" * 60)

    # Load original paraphrases
    if not os.path.exists(PARAPHRASE_FILE):
        print(f"ERROR: {PARAPHRASE_FILE} not found.")
        sys.exit(1)

    with open(PARAPHRASE_FILE) as f:
        problems = json.load(f)

    print(f"Loaded {len(problems)} problems.")

    # Load all four sources
    source_labels = [
        "Human-1", "Human-2", "LLM-DeepSeek-R1-70B", "LLM-Qwen3-32B"
    ]
    sources = []

    print("\nLoading human annotations...")
    for path in HUMAN_FILES:
        sources.append(load_human_excel(path))
        print(f"  {os.path.basename(path)}: {sum(len(v) for v in sources[-1].values())} entries")

    print("Loading LLM judge results...")
    for path in LLM_JUDGE_FILES:
        sources.append(load_llm_judge(path))
        print(f"  {os.path.basename(path)}: {sum(len(v) for v in sources[-1].values())} entries")

    # Check how many sources are available
    available_sources = [
        (label, src) for label, src in zip(source_labels, sources) if src
    ]
    n_sources = len(available_sources)
    print(f"\nAvailable sources: {n_sources} / 4")
    if n_sources < 2:
        print("WARNING: Fewer than 2 sources available. "
              "Proceeding with available sources only.")

    # Build per-paraphrase verdict table
    # Structure: {pid: {vid: {source_label: "YES"/"NO"}}}
    verdict_table = defaultdict(lambda: defaultdict(dict))

    for label, src in available_sources:
        for pid, variants in src.items():
            for vid, verdict in variants.items():
                verdict_table[pid][vid][label] = verdict

    # Compute majority vote and acceptance
    validated_problems = []
    report_items       = []
    rejected_count     = 0

    # Collect binary labels per source pair for kappa
    # kappa_data[label] = list of 0/1 for each non-P0 paraphrase
    kappa_data = {label: [] for label, _ in available_sources}

    for problem in problems:
        pid = problem["id"]
        accepted_variants = [
            next(v for v in problem["paraphrases"] if v["id"] == "P0")
        ]
        problem_report = {"id": pid, "domain": problem["domain"],
                          "difficulty": problem["difficulty"], "variants": {}}

        for variant in problem["paraphrases"]:
            vid = variant["id"]
            if vid == "P0":
                problem_report["variants"][vid] = {
                    "verdicts": {l: "YES" for l, _ in available_sources},
                    "yes_count": n_sources,
                    "accepted": True,
                    "majority_vote": "YES",
                }
                continue

            verdicts = verdict_table[pid].get(vid, {})
            yes_count = sum(
                1 for label, _ in available_sources
                if verdicts.get(label, "NO").upper() in ("YES", "VALID")
            )
            accepted = yes_count >= ACCEPTANCE_THRESHOLD

            # Collect for kappa
            for label, _ in available_sources:
                v = verdicts.get(label, "NO").upper()
                kappa_data[label].append(1 if v in ("YES", "VALID") else 0)

            item_report = {
                "verdicts":      verdicts,
                "yes_count":     yes_count,
                "accepted":      accepted,
                "majority_vote": "YES" if accepted else "NO",
            }
            problem_report["variants"][vid] = item_report

            if accepted:
                accepted_variants.append(variant)
            else:
                rejected_count += 1

        # Build validated problem entry
        validated_entry = {k: v for k, v in problem.items() if k != "paraphrases"}
        validated_entry["paraphrases"] = accepted_variants
        validated_entry["n_accepted"]  = len(accepted_variants) - 1  # exclude P0
        validated_problems.append(validated_entry)
        report_items.append(problem_report)

    # Compute pairwise Cohen's Kappa
    kappa_results = {}
    labels_list   = [label for label, _ in available_sources]
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            la, lb = labels_list[i], labels_list[j]
            if len(kappa_data[la]) == len(kappa_data[lb]) and kappa_data[la]:
                k = cohens_kappa(kappa_data[la], kappa_data[lb])
                kappa_results[f"{la} vs {lb}"] = round(k, 4)

    # Overall acceptance stats
    total_paraphrases = sum(
        len(p["paraphrases"]) - 1  # exclude P0
        for p in problems
    )
    total_accepted = sum(p["n_accepted"] for p in validated_problems)
    acceptance_rate = total_accepted / max(total_paraphrases, 1)

    # Save outputs
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(OUTPUT_VALIDATED, "w") as f:
        json.dump(validated_problems, f, indent=2, ensure_ascii=False)

    report = {
        "n_problems":          len(problems),
        "n_sources":           n_sources,
        "source_labels":       labels_list,
        "acceptance_threshold": ACCEPTANCE_THRESHOLD,
        "total_paraphrases":   total_paraphrases,
        "total_accepted":      total_accepted,
        "total_rejected":      rejected_count,
        "acceptance_rate":     round(acceptance_rate, 4),
        "cohens_kappa":        kappa_results,
        "items":               report_items,
    }

    with open(OUTPUT_REPORT, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Human-readable summary
    summary_lines = [
        "ARCG Paraphrase Validation Summary",
        "=" * 50,
        f"Problems:              {len(problems)}",
        f"Sources used:          {n_sources} ({', '.join(labels_list)})",
        f"Acceptance threshold:  {ACCEPTANCE_THRESHOLD} of {n_sources} sources",
        f"Total paraphrases:     {total_paraphrases}",
        f"Accepted:              {total_accepted} ({100*acceptance_rate:.1f}%)",
        f"Rejected:              {rejected_count} ({100*(1-acceptance_rate):.1f}%)",
        "",
        "Inter-Annotator Agreement (Cohen's Kappa)",
        "-" * 50,
    ]
    for pair, k in kappa_results.items():
        interp = kappa_interpretation(k)
        summary_lines.append(f"  {pair:<45} kappa = {k:.4f}  ({interp})")

    summary_lines += [
        "",
        "Problems with fewer than 3 accepted paraphrases:",
        "-" * 50,
    ]
    flagged = [p for p in validated_problems if p["n_accepted"] < 3]
    if flagged:
        for p in flagged:
            summary_lines.append(
                f"  {p['id']}  ({p['domain']}, {p['difficulty']})  "
                f"-- {p['n_accepted']} paraphrases accepted"
            )
    else:
        summary_lines.append("  None -- all problems have at least 3 accepted paraphrases.")

    summary_text = "\n".join(summary_lines)
    with open(OUTPUT_SUMMARY, "w") as f:
        f.write(summary_text)

    print("\n" + summary_text)
    print(f"\nOutputs saved:")
    print(f"  {OUTPUT_VALIDATED}")
    print(f"  {OUTPUT_REPORT}")
    print(f"  {OUTPUT_SUMMARY}")
    print("\nNext step: run 5_run_experiment.py")


if __name__ == "__main__":
    main()
