"""
extract_questions.py  --  Inspect the 75 benchmark questions
=============================================================
Loads the same 75 problems used by the ARCG experiment
(50 GSM8K math + 25 ARC-Challenge logic) and writes them to:

  data/questions.txt   -- human-readable numbered list
  data/questions.json  -- machine-readable list

Also prints a summary table to stdout.

Usage
-----
  python code/extract_questions.py

No Ollama required. Only needs:  pip install datasets
"""

import json
import os
import random
import sys
import textwrap

# ---------------------------------------------------------------------------
# Same config as script 1 -- must stay in sync
# ---------------------------------------------------------------------------
N_MATH    = 50
N_LOGIC   = 25
SEED      = 42
DATA_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
TXT_OUT   = os.path.join(DATA_DIR, "questions.txt")
JSON_OUT  = os.path.join(DATA_DIR, "questions.json")
DIFF_EASY = 150   # GSM8K: characters in solution text
DIFF_MED  = 300

os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset loaders (identical logic to script 1)
# ---------------------------------------------------------------------------

def load_gsm8k(n: int, seed: int) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("ERROR: run  pip install datasets  first.")
    print(f"Loading GSM8K (n={n})...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    easy, medium, hard = [], [], []
    for item in ds:
        q = item["question"].strip()
        a = item["answer"].strip()
        sol_len = len(a)
        diff = "easy" if sol_len < DIFF_EASY else ("medium" if sol_len < DIFF_MED else "hard")
        # Build a short slug for the ID
        words = [w for w in q.split() if w.isalpha()][:4]
        slug  = "_".join(words)
        record = {
            "id":          f"GSM_{slug}",
            "domain":      "math",
            "difficulty":  diff,
            "question":    q,
            "answer":      a.split("####")[-1].strip() if "####" in a else a,
            "answer_type": "numeric",
            "source":      "GSM8K",
        }
        if diff == "easy":
            easy.append(record)
        elif diff == "medium":
            medium.append(record)
        else:
            hard.append(record)

    rng = random.Random(seed)
    per_band = n // 3
    selected = (
        rng.sample(easy,   min(per_band, len(easy)))   +
        rng.sample(medium, min(per_band, len(medium))) +
        rng.sample(hard,   min(n - 2 * per_band, len(hard)))
    )
    # Deduplicate IDs
    seen, unique = set(), []
    for i, r in enumerate(selected):
        base_id = r["id"]
        uid = base_id if base_id not in seen else f"{base_id}_{i}"
        r["id"] = uid
        seen.add(uid)
        unique.append(r)
    return unique[:n]


def load_arc_challenge(n: int, seed: int) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("ERROR: run  pip install datasets  first.")
    print(f"Loading ARC-Challenge (n={n})...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    easy, medium, hard = [], [], []
    for item in ds:
        choices = item["choices"]
        labels  = choices["label"]
        texts   = choices["text"]
        options = {l: t for l, t in zip(labels, texts)}
        opts_str = "  ".join(f"({l}) {t}" for l, t in options.items())
        full_q   = f"{item['question'].strip()}\n{opts_str}"
        qlen = len(item["question"])
        diff = "easy" if qlen < 80 else ("medium" if qlen < 140 else "hard")
        record = {
            "id":          f"ARC_{item['id']}",
            "domain":      "logic",
            "difficulty":  diff,
            "question":    full_q,
            "answer":      item["answerKey"].strip(),
            "answer_type": "multiple_choice",
            "source":      "ARC-Challenge",
            "choices":     options,
        }
        if diff == "easy":
            easy.append(record)
        elif diff == "medium":
            medium.append(record)
        else:
            hard.append(record)

    rng = random.Random(seed)
    per_band = n // 3
    selected = (
        rng.sample(easy,   min(per_band, len(easy)))   +
        rng.sample(medium, min(per_band, len(medium))) +
        rng.sample(hard,   min(n - 2 * per_band, len(hard)))
    )
    return selected[:n]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    problems = load_gsm8k(N_MATH, SEED) + load_arc_challenge(N_LOGIC, SEED)
    print(f"\nTotal problems loaded: {len(problems)}")

    # ---- Summary table ----
    print("\n" + "=" * 72)
    print(f"{'#':>3}  {'ID':<35}  {'Domain':<6}  {'Diff':<6}  {'Source'}")
    print("-" * 72)
    for i, p in enumerate(problems, 1):
        print(f"{i:>3}  {p['id']:<35}  {p['domain']:<6}  {p['difficulty']:<6}  {p['source']}")
    print("=" * 72)

    # ---- Counts ----
    from collections import Counter
    by_domain = Counter(p["domain"] for p in problems)
    by_diff   = Counter(p["difficulty"] for p in problems)
    print(f"\nDomain breakdown: {dict(by_domain)}")
    print(f"Difficulty breakdown: {dict(by_diff)}")

    # ---- Write human-readable TXT ----
    with open(TXT_OUT, "w", encoding="utf-8") as f:
        f.write(f"ARCG Benchmark  --  {len(problems)} Questions\n")
        f.write("=" * 72 + "\n\n")
        for i, p in enumerate(problems, 1):
            f.write(f"[{i:>3}]  {p['id']}\n")
            f.write(f"       Source: {p['source']}  |  Domain: {p['domain']}  |  Difficulty: {p['difficulty']}\n")
            f.write(f"       Answer: {p['answer']}\n")
            f.write("\n       QUESTION:\n")
            # Wrap long lines for readability
            for line in p["question"].splitlines():
                wrapped = textwrap.fill(line, width=68, initial_indent="       ",
                                        subsequent_indent="       ")
                f.write(wrapped + "\n")
            f.write("\n" + "-" * 72 + "\n\n")

    # ---- Write JSON ----
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(problems, f, indent=2, ensure_ascii=False)

    print(f"\nFiles written:")
    print(f"  {TXT_OUT}   (human-readable)")
    print(f"  {JSON_OUT}  (machine-readable)")
    print("\nOpen questions.txt to inspect all 75 questions.")


if __name__ == "__main__":
    main()
