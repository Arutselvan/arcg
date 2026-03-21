"""
convert_jsonl_to_paraphrases.py  --  Convert Claude/Gemini JSONL output to paraphrases.json
============================================================================================
After you paste the model's JSONL output into  data/paraphrases_raw.jsonl,
run this script to validate it and convert it to the  data/paraphrases.json
format expected by all downstream scripts (2_generate_validation_template.py
through 6_analyze_and_plot.py).

Usage
-----
  1. Run:  python code/extract_questions.py
           This writes  data/questions.json

  2. Paste the contents of  data/questions.json  into the chat prompt in
     code/prompt_for_claude.txt  and send it to Claude Opus or Gemini 2.5 Pro.

  3. Copy the model's raw JSONL output (one JSON object per line) into:
         data/paraphrases_raw.jsonl

  4. Run:  python code/convert_jsonl_to_paraphrases.py

  5. If validation passes, data/paraphrases.json is ready.
     Proceed to:  python code/2_generate_validation_template.py

Validation checks performed
----------------------------
  - Every problem in questions.json has a corresponding JSONL line
  - Every line has all 5 paraphrase keys (P1-P5)
  - No paraphrase is empty or identical to the original question
  - No paraphrase contains solution markers (step-by-step, therefore, = , etc.)
  - No paraphrase is identical to another paraphrase for the same problem
  - For multiple-choice problems: all answer choice labels are still present
"""

import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "..", "data")
QUESTIONS   = os.path.join(DATA_DIR, "questions.json")
RAW_JSONL   = os.path.join(DATA_DIR, "paraphrases_raw.jsonl")
OUTPUT      = os.path.join(DATA_DIR, "paraphrases.json")

PARAPHRASE_KEYS = ["P1", "P2", "P3", "P4", "P5"]

# Phrases that indicate the model solved the problem instead of paraphrasing it
SOLUTION_MARKERS = [
    r"\bstep[\s\-]?\d",          # "step 1", "step-by-step"
    r"\btherefore\b",
    r"\bthe answer is\b",
    r"\bthe correct answer is\b",
    r"\bthus\b",
    r"\bhence\b",
    r"\bexplanation\b",
    r"\bsolution\b",
    r"\bsolving\b",
    r"\bwe (get|find|have|calculate|compute)\b",
    r"\bfirst,\s",
    r"\bsecond,\s",
    r"\bfinally,\s",
    r"=\s*\d",                    # "= 17"
    r"\d+\s*\+\s*\d+",           # "15 + 2"
    r"\d+\s*×\s*\d+",            # "3 × 5"
    r"\d+\s*\*\s*\d+",           # "3 * 5"
]
SOLUTION_RE = re.compile("|".join(SOLUTION_MARKERS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_questions() -> dict:
    """Returns a dict keyed by problem ID."""
    if not os.path.exists(QUESTIONS):
        sys.exit(f"ERROR: {QUESTIONS} not found.\nRun:  python code/extract_questions.py  first.")
    with open(QUESTIONS, encoding="utf-8") as f:
        problems = json.load(f)
    return {p["id"]: p for p in problems}


def load_raw_jsonl() -> dict:
    """Returns a dict keyed by problem ID from the raw JSONL file."""
    if not os.path.exists(RAW_JSONL):
        sys.exit(
            f"ERROR: {RAW_JSONL} not found.\n"
            "Paste the model's JSONL output into that file and re-run."
        )
    rows = {}
    errors = []
    with open(RAW_JSONL, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            # Strip accidental markdown fences
            if line.startswith("```"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"  Line {lineno}: JSON parse error: {e}")
                continue
            pid = obj.get("id")
            if not pid:
                errors.append(f"  Line {lineno}: missing 'id' field")
                continue
            rows[pid] = obj
    if errors:
        print(f"JSONL parse errors ({len(errors)}):")
        for e in errors:
            print(e)
    return rows


def validate(questions: dict, rows: dict) -> list[dict]:
    """Validates all paraphrases and returns the merged list. Exits on fatal errors."""
    warnings = []
    fatal    = []
    output   = []

    for pid, problem in questions.items():
        if pid not in rows:
            fatal.append(f"  MISSING: no JSONL row for problem '{pid}'")
            continue

        row = rows[pid]
        paraphrases = {}

        for key in PARAPHRASE_KEYS:
            text = row.get(key, "").strip()

            # Check presence
            if not text:
                warnings.append(f"  [{pid}] {key}: empty — will be skipped")
                continue

            # Check not identical to original
            if text.lower() == problem["question"].lower():
                warnings.append(f"  [{pid}] {key}: identical to original — will be skipped")
                continue

            # Check for solution markers
            if SOLUTION_RE.search(text):
                warnings.append(f"  [{pid}] {key}: contains solution markers — will be skipped\n"
                                 f"    Text: {text[:120]}")
                continue

            # Check ends with question mark (warning only)
            if not text.rstrip().endswith("?"):
                warnings.append(f"  [{pid}] {key}: does not end with '?' — kept but flagged\n"
                                 f"    Text: {text[:120]}")

            # For multiple-choice: check answer choice labels are preserved
            if problem.get("answer_type") == "multiple_choice" and "choices" in problem:
                for label in problem["choices"]:
                    pattern = rf"\({re.escape(label)}\)"
                    if not re.search(pattern, text):
                        warnings.append(
                            f"  [{pid}] {key}: missing answer choice ({label}) — kept but flagged"
                        )

            paraphrases[key] = text

        # Check for duplicate paraphrases within the same problem
        seen_texts = {}
        for key, text in list(paraphrases.items()):
            norm = text.lower().strip()
            if norm in seen_texts:
                warnings.append(
                    f"  [{pid}] {key}: identical to {seen_texts[norm]} — will be skipped"
                )
                del paraphrases[key]
            else:
                seen_texts[norm] = key

        # Build output record (same schema as script 1's paraphrases.json)
        record = {
            "id":          problem["id"],
            "domain":      problem["domain"],
            "difficulty":  problem["difficulty"],
            "source":      problem["source"],
            "answer":      problem["answer"],
            "answer_type": problem.get("answer_type", "numeric"),
            "variants": {
                "P0": problem["question"],   # original
                **paraphrases,
            },
        }
        if problem.get("choices"):
            record["choices"] = problem["choices"]

        output.append(record)

    # Print summary
    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(w)

    if fatal:
        print(f"\nFATAL ERRORS ({len(fatal)}):")
        for e in fatal:
            print(e)
        sys.exit("\nFix the errors above and re-run.")

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading questions...")
    questions = load_questions()
    print(f"  {len(questions)} problems loaded from {QUESTIONS}")

    print("\nLoading raw JSONL...")
    rows = load_raw_jsonl()
    print(f"  {len(rows)} JSONL rows loaded from {RAW_JSONL}")

    print("\nValidating paraphrases...")
    output = validate(questions, rows)

    # Summary stats
    total_variants = sum(len(r["variants"]) - 1 for r in output)  # exclude P0
    print(f"\nValidation complete:")
    print(f"  Problems:          {len(output)} / {len(questions)}")
    print(f"  Paraphrases kept:  {total_variants} / {len(output) * 5} possible")
    skipped = len(output) * 5 - total_variants
    if skipped:
        print(f"  Paraphrases skipped (empty/duplicate/solved): {skipped}")

    # Write output
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nWrote: {OUTPUT}")
    print("\nNext step:  python code/2_generate_validation_template.py")


if __name__ == "__main__":
    main()
