"""
ARCG Experiment  --  Step 1: Build Benchmark and Generate Paraphrases
======================================================================
Loads 75 real problems (50 from GSM8K, 25 from ARC-Challenge), generates
5 semantically-equivalent paraphrases per problem using deepseek-r1:32b,
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
  and pulls deepseek-r1:32b if it is not already available.

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
PARAPHRASE_MODEL  = "deepseek-r1:32b"   # strong reasoning model for paraphrasing
N_MATH            = 50                  # problems from GSM8K
N_LOGIC           = 25                  # problems from ARC-Challenge
RANDOM_SEED       = 42
DATA_DIR          = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUTPUT_FILE       = os.path.join(DATA_DIR, "paraphrases.json")
REQUEST_TIMEOUT   = 300                 # seconds per request
MAX_RETRIES       = 8                   # retries per paraphrase call
WARMUP_TIMEOUT    = 300                 # seconds to wait for model warmup

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


# Environment variables that fix Ollama's RAM check on systems where
# buff/cache inflates the "used" column.  OLLAMA_MAX_LOADED_MODELS=1
# prevents Ollama from trying to keep multiple models resident, and
# OLLAMA_SCHED_SPREAD=0 keeps the model on a single GPU.
OLLAMA_ENV = {
    **os.environ,
    "OLLAMA_MAX_LOADED_MODELS": "1",
    "OLLAMA_SCHED_SPREAD":      "0",
    "OLLAMA_KEEP_ALIVE":        "10m",
}


def restart_ollama_server():
    """Kill any running Ollama process and restart with the correct env vars."""
    print("Restarting Ollama server with GPU-optimised settings...")
    # Kill existing Ollama processes
    subprocess.run(["pkill", "-x", "ollama"], check=False)
    time.sleep(3)
    subprocess.Popen(
        ["ollama", "serve"],
        env=OLLAMA_ENV,
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


def start_ollama_server():
    """Start the Ollama server as a background process."""
    print("Ollama server is not running. Starting it now...")
    subprocess.Popen(
        ["ollama", "serve"],
        env=OLLAMA_ENV,
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
    # Match by prefix to handle tag variants (e.g. deepseek-r1:32b)
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


def unload_all_models():
    """Unload all currently loaded models from GPU VRAM before switching."""
    print("  Unloading all models from GPU memory...")
    try:
        r = requests.get(f"{OLLAMA_URL}/api/ps", timeout=5)
        if r.status_code == 200:
            loaded = r.json().get("models", [])
            for m in loaded:
                name = m.get("name", "")
                if name:
                    try:
                        requests.post(
                            f"{OLLAMA_URL}/api/generate",
                            json={"model": name, "keep_alive": 0},
                            timeout=30,
                        )
                        print(f"    Unloaded: {name}")
                    except Exception:
                        pass
    except Exception:
        pass
    time.sleep(3)
    print("  GPU memory cleared.")


def warmup_model(model: str):
    """Send a trivial prompt and wait until the model responds successfully.
    This ensures the model is fully loaded into VRAM before real work begins.
    A 32B model can take 20-60 seconds to load on first use.
    """
    print(f"  Warming up {model} (waiting for first successful response)...")
    payload = {
        "model":  model,
        "prompt": "Reply with the single word: ready",
        "stream": False,
        "options": {"num_predict": 8, "temperature": 0.0},
    }
    deadline = time.time() + WARMUP_TIMEOUT
    attempt  = 0
    while time.time() < deadline:
        attempt += 1
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json=payload,
                timeout=120,
            )
            if r.status_code == 200:
                print(f"  Model {model} is warm and ready (attempt {attempt}).")
                return
            # 500 = model still loading; anything else is unexpected
            print(f"  Warmup attempt {attempt}: HTTP {r.status_code}. Retrying in 10s...")
        except Exception as exc:
            print(f"  Warmup attempt {attempt}: {exc}. Retrying in 10s...")
        time.sleep(10)
    print(f"ERROR: {model} did not become ready within {WARMUP_TIMEOUT}s.")
    sys.exit(1)


def ensure_ollama_ready(model: str):
    # Unload any resident model first to free VRAM, then restart with
    # correct env vars to ensure GPU is used without RAM check issues.
    unload_all_models()
    restart_ollama_server()
    ensure_model(model)
    warmup_model(model)

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
    "P4": "Rewrite the problem by splitting the scenario into clearly numbered sub-questions "
          "that together lead to the same final answer. Each sub-question should ask for an "
          "intermediate quantity needed to reach the answer. Do NOT compute any values, do NOT "
          "show any arithmetic, and do NOT reveal the answer. The output must still be a question.",
    "P5": "Restate the problem using a completely different real-world context or analogy "
          "(e.g., change a shopping scenario to a farming scenario) while preserving the "
          "identical mathematical or logical structure and the same correct answer.",
}


def build_paraphrase_prompt(problem: dict, strategy_key: str, strategy_desc: str) -> str:
    if problem["domain"] == "math":
        domain_rules = (
            "This is a mathematical word problem. "
            "The correct numerical answer MUST remain identical to the original. "
            "Do NOT compute, reveal, or hint at the answer anywhere in your output."
        )
    else:
        domain_rules = (
            "This is a multiple-choice logical reasoning question. "
            "All answer choices (A, B, C, D, E) and the correct answer letter MUST be preserved exactly. "
            "Do NOT compute, reveal, or hint at the correct choice anywhere in your output."
        )
    return (
        f"You are a linguistic rewriting expert. "
        f"Your ONLY job is to rewrite the problem statement using the strategy below. "
        f"You must NOT solve the problem, show working, or reveal the answer in any form.\n\n"
        f"REWRITING STRATEGY ({strategy_key}): {strategy_desc}\n\n"
        f"STRICT RULES -- violating any rule makes your output invalid:\n"
        f"1. Output ONLY the rewritten problem text. No preamble, no explanation, no commentary.\n"
        f"2. Do NOT solve the problem. Do NOT show any calculations or reasoning steps.\n"
        f"3. Do NOT include the answer, the solution, or any numerical result in your output.\n"
        f"4. The rewritten problem must be answerable and must have the SAME correct answer as the original.\n"
        f"5. {domain_rules}\n"
        f"6. Keep all quantities, names, and relationships intact -- only the wording changes.\n"
        f"7. The output must end with a question mark (it is still a question).\n\n"
        f"ORIGINAL PROBLEM:\n{problem['question']}\n\n"
        f"REWRITTEN PROBLEM (question only, no solution):"
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
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            if r.status_code == 500:
                # Model may still be loading or temporarily busy
                wait = min(10 * (attempt + 1), 60)
                print(f"    [attempt {attempt+1}/{MAX_RETRIES}] HTTP 500 (model loading?). "
                      f"Retrying in {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            text = r.json()["response"].strip()
            # Strip <think>...</think> blocks (DeepSeek R1 reasoning traces)
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            # Reject responses that look like solved answers rather than questions.
            # A valid paraphrase must end with a question mark and must not contain
            # step-by-step solution markers.
            solution_markers = [
                "step-by-step explanation", "step-by-step",
                "step 1:", "step 2:", "step 3:",
                "therefore,", "therefore the",
                "the answer is", "the final answer",
                "thus,", "thus the",
                "so the answer", "so johnny", "so mike",
                "solution:", "answer:",
                "explanation:", "working:",
                "calculate:", "compute:",
                "= (",             # e.g. "= (3 x 5)"
            ]
            lower = text.lower()
            is_solution = any(m in lower for m in solution_markers)
            if is_solution:
                print(f"    [attempt {attempt+1}/{MAX_RETRIES}] Response looks like a solution, not a paraphrase. Retrying...")
                time.sleep(5)
                continue
            return text
        except requests.exceptions.Timeout:
            wait = 15
            print(f"    [attempt {attempt+1}/{MAX_RETRIES}] Timeout. Retrying in {wait}s...")
            time.sleep(wait)
        except Exception as exc:
            wait = min(10 * (attempt + 1), 60)
            print(f"    [attempt {attempt+1}/{MAX_RETRIES}] error: {exc}. Retrying in {wait}s...")
            time.sleep(wait)
    return ""


PARAPHRASE_OUTER_RETRIES = 3   # outer retries per paraphrase slot before skipping


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
        text   = ""
        for outer in range(1, PARAPHRASE_OUTER_RETRIES + 1):
            text = call_ollama(prompt, PARAPHRASE_MODEL)
            if text:
                break
            print(f"    WARNING: Empty paraphrase for {problem['id']} {key} "
                  f"(outer attempt {outer}/{PARAPHRASE_OUTER_RETRIES}). Retrying...")
            time.sleep(5)
        if not text:
            # Do NOT fall back to original — skip this slot to keep the dataset clean.
            print(f"    SKIP: Could not generate paraphrase for {problem['id']} {key} "
                  f"after {PARAPHRASE_OUTER_RETRIES} outer retries. Slot omitted.")
            continue
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
