"""
ARCG Experiment  --  Step 1: Build Benchmark and Generate Paraphrases
======================================================================
Loads 75 real problems (50 from GSM8K, 25 from ARC-Challenge), generates
5 semantically-equivalent paraphrases per problem using deepseek-r1:70b,
and saves the shared paraphrase dataset to data/paraphrases.json.

This file is the single source of truth for all downstream scripts.
Every evaluation model in Step 2 sees the EXACT same paraphrase set.

Paraphrase strategies (one per variant P1-P5):
  P1 -- Formal restatement:    mathematical/logical language, no narrative
  P2 -- Informal restatement:  everyday conversational language
  P3 -- Passive restructuring: passive voice, altered sentence order
  P4 -- Decomposed:            problem split into numbered sub-steps
  P5 -- Analogical:            same structure, different real-world context

The original problem is stored as P0 (no paraphrase).

Requirements
------------
  pip install datasets requests tqdm

  Ollama must be installed (https://ollama.com).
  This script starts the Ollama server automatically if it is not running
  and pulls deepseek-r1:70b if it is not already available.

Usage
-----
  python 1_build_and_paraphrase.py

Output
------
  data/paraphrases.json   -- 75 problems x 6 variants (P0-P5)
"""

import json
import os
import re
import subprocess
import sys
import time

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL        = "http://localhost:11434"
PARAPHRASE_MODEL  = "deepseek-r1:70b"   # strong reasoning model for paraphrasing
N_MATH            = 50                  # problems from GSM8K
N_LOGIC           = 25                  # problems from ARC-Challenge
RANDOM_SEED       = 42
DATA_DIR          = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUTPUT_FILE       = os.path.join(DATA_DIR, "paraphrases.json")
REQUEST_TIMEOUT   = 300                 # seconds; 70B model can be slow

# Difficulty bands for GSM8K (approximate by solution length)
DIFFICULTY_THRESHOLDS = {"easy": 150, "medium": 300}  # characters in solution

# ---------------------------------------------------------------------------
# Ollama server and model management
# ---------------------------------------------------------------------------

def is_ollama_running() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def start_ollama_server():
    """Start the Ollama server as a background process."""
    print("Ollama server is not running. Starting it now...")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(30):
        time.sleep(2)
        if is_ollama_running():
            print("Ollama server started successfully.")
            return
    print("ERROR: Could not start Ollama server after 60 seconds.")
    sys.exit(1)


def get_available_models() -> list[str]:
    r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    return [m["name"] for m in r.json().get("models", [])]


def ensure_model(model: str):
    """Pull the model if it is not already available locally."""
    available = get_available_models()
    # Match by prefix to handle tag variants (e.g. deepseek-r1:70b)
    if any(m.startswith(model.split(":")[0]) and model.split(":")[-1] in m
           for m in available):
        print(f"  Model {model} is already available.")
        return
    print(f"  Pulling {model} (this may take a while for large models)...")
    result = subprocess.run(["ollama", "pull", model], check=False)
    if result.returncode != 0:
        print(f"ERROR: Failed to pull {model}. Check your internet connection.")
        sys.exit(1)
    print(f"  {model} pulled successfully.")


def ensure_ollama_ready(model: str):
    if not is_ollama_running():
        start_ollama_server()
    ensure_model(model)

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def classify_difficulty_gsm8k(solution: str) -> str:
    n = len(solution)
    if n <= DIFFICULTY_THRESHOLDS["easy"]:
        return "easy"
    if n <= DIFFICULTY_THRESHOLDS["medium"]:
        return "medium"
    return "hard"


def extract_gsm8k_answer(solution: str) -> str:
    """GSM8K solutions end with '#### <number>'."""
    m = re.search(r"####\s*([\-\d,\.]+)", solution)
    if m:
        return m.group(1).replace(",", "").strip()
    # Fallback: last number in solution
    nums = re.findall(r"[\-\d,\.]+", solution)
    return nums[-1].replace(",", "") if nums else ""


def load_gsm8k(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset
    print(f"Loading GSM8K (n={n})...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    # Stratified sample: roughly equal easy/medium/hard
    easy, medium, hard = [], [], []
    for item in ds:
        diff = classify_difficulty_gsm8k(item["answer"])
        record = {
            "id":          f"GSM_{item['question'][:20].replace(' ','_').replace('/','')[:15]}",
            "domain":      "math",
            "difficulty":  diff,
            "question":    item["question"].strip(),
            "answer":      extract_gsm8k_answer(item["answer"]),
            "answer_type": "numeric",
            "source":      "GSM8K",
        }
        if diff == "easy":
            easy.append(record)
        elif diff == "medium":
            medium.append(record)
        else:
            hard.append(record)

    import random
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
    from datasets import load_dataset
    print(f"Loading ARC-Challenge (n={n})...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    easy, medium, hard = [], [], []
    for item in ds:
        choices = item["choices"]
        labels  = choices["label"]
        texts   = choices["text"]
        options = {l: t for l, t in zip(labels, texts)}
        # Format question with labelled options
        opts_str = "  ".join(f"({l}) {t}" for l, t in options.items())
        full_q   = f"{item['question'].strip()}\n{opts_str}"
        # Rough difficulty by question length
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

    import random
    rng = random.Random(seed)
    per_band = n // 3
    selected = (
        rng.sample(easy,   min(per_band, len(easy)))   +
        rng.sample(medium, min(per_band, len(medium))) +
        rng.sample(hard,   min(n - 2 * per_band, len(hard)))
    )
    return selected[:n]

# ---------------------------------------------------------------------------
# Paraphrase generation
# ---------------------------------------------------------------------------

PARAPHRASE_STRATEGIES = {
    "P1": "Restate the problem using precise formal mathematical or logical language. "
          "Do not use narrative or story framing. Preserve all numerical values and constraints exactly.",
    "P2": "Restate the problem in casual, everyday conversational language as if explaining "
          "it to a friend. Keep all numbers and logical constraints identical.",
    "P3": "Restate the problem using passive voice and a different sentence order. "
          "The meaning, numbers, and constraints must remain identical.",
    "P4": "Break the problem into clearly numbered sub-steps or sub-questions that together "
          "ask for the same final answer. Do not solve it.",
    "P5": "Restate the problem using a completely different real-world context or analogy "
          "(e.g., change a shopping scenario to a farming scenario) while preserving the "
          "identical mathematical or logical structure and the same correct answer.",
}


def build_paraphrase_prompt(problem: dict, strategy_key: str, strategy_desc: str) -> str:
    domain_hint = (
        "This is a mathematical word problem. The correct numerical answer must be preserved."
        if problem["domain"] == "math"
        else "This is a multiple-choice logical reasoning question. "
             "The answer choices and correct answer letter must be preserved exactly."
    )
    return (
        f"You are a careful linguistic expert. Your task is to paraphrase the following problem.\n\n"
        f"PARAPHRASE STRATEGY ({strategy_key}): {strategy_desc}\n\n"
        f"IMPORTANT RULES:\n"
        f"1. The paraphrase must be semantically equivalent -- the correct answer must not change.\n"
        f"2. {domain_hint}\n"
        f"3. Do NOT solve the problem.\n"
        f"4. Do NOT add hints or extra information.\n"
        f"5. Output ONLY the paraphrased problem text. No explanation, no preamble.\n\n"
        f"ORIGINAL PROBLEM:\n{problem['question']}\n\n"
        f"PARAPHRASED VERSION:"
    )


def call_ollama(prompt: str, model: str) -> str:
    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,   # slight creativity for paraphrasing
            "num_predict": 512,
        },
    }
    for attempt in range(3):
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            r.raise_for_status()
            text = r.json()["response"].strip()
            # Strip <think>...</think> blocks (DeepSeek R1 reasoning traces)
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            return text
        except Exception as exc:
            wait = 2 ** attempt
            print(f"    [attempt {attempt+1}/3] error: {exc}. Retrying in {wait}s...")
            time.sleep(wait)
    return ""


def generate_paraphrases(problem: dict) -> list[dict]:
    variants = [
        {
            "id":       "P0",
            "strategy": "original",
            "text":     problem["question"],
        }
    ]
    for key, desc in PARAPHRASE_STRATEGIES.items():
        prompt = build_paraphrase_prompt(problem, key, desc)
        text   = call_ollama(prompt, PARAPHRASE_MODEL)
        if not text:
            print(f"    WARNING: Empty paraphrase for {problem['id']} {key}. Using original.")
            text = problem["question"]
        variants.append({"id": key, "strategy": key, "text": text})
    return variants

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint() -> list[dict]:
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            return json.load(f)
    return []


def save_checkpoint(data: list[dict]):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("ARCG Step 1: Build Benchmark and Generate Paraphrases")
    print("=" * 60)

    # Ensure Ollama is running and model is available
    ensure_ollama_ready(PARAPHRASE_MODEL)

    # Load datasets
    gsm_problems = load_gsm8k(N_MATH, RANDOM_SEED)
    arc_problems = load_arc_challenge(N_LOGIC, RANDOM_SEED)
    all_problems = gsm_problems + arc_problems
    print(f"\nTotal problems: {len(all_problems)} "
          f"(GSM8K: {len(gsm_problems)}, ARC-Challenge: {len(arc_problems)})")

    # Load checkpoint
    done_data = load_checkpoint()
    done_ids  = {p["id"] for p in done_data}
    remaining = [p for p in all_problems if p["id"] not in done_ids]
    print(f"Already paraphrased: {len(done_data)}/{len(all_problems)}. "
          f"Remaining: {len(remaining)}")

    if not remaining:
        print("All paraphrases already generated.")
        print(f"Output: {OUTPUT_FILE}")
        return

    # Generate paraphrases
    results = list(done_data)
    for problem in tqdm(remaining, desc="Paraphrasing"):
        variants = generate_paraphrases(problem)
        entry = {
            "id":          problem["id"],
            "domain":      problem["domain"],
            "difficulty":  problem["difficulty"],
            "answer":      problem["answer"],
            "answer_type": problem["answer_type"],
            "source":      problem["source"],
            "paraphrases": variants,
        }
        if problem.get("choices"):
            entry["choices"] = problem["choices"]
        results.append(entry)
        save_checkpoint(results)

    print(f"\nDone. {len(results)} problems saved to {OUTPUT_FILE}")
    print("Next step: run 2_generate_validation_template.py")


if __name__ == "__main__":
    main()
