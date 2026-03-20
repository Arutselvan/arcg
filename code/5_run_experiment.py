"""
ARCG Experiment  --  Step 5: Run Evaluation Across 10 Reasoning Models
=======================================================================
Reads data/validated_paraphrases.json (output of Step 4) and runs all
10 reasoning models on every (problem, paraphrase) pair using
chain-of-thought prompting.

Models evaluated (all reasoning models, all fit on H100 80GB at Q4):
  1.  deepseek-r1:7b          DeepSeek R1 distill (Qwen base)
  2.  deepseek-r1:8b          DeepSeek R1 distill (Llama base)
  3.  deepseek-r1:14b         DeepSeek R1 distill
  4.  deepseek-r1:32b         DeepSeek R1 distill (largest in DeepSeek R1 series)
  5.  llama3.3:70b             Meta Llama 3.3 70B (replaces deepseek-r1:70b)
  6.  qwen3:8b                Qwen 3 8B (hybrid thinking)
  7.  qwen3:32b               Qwen 3 32B (hybrid thinking)
  8.  magistral:24b           Mistral reasoning model
  9.  phi4-reasoning:14b      Microsoft Phi-4 Reasoning
  10. glm-4.7-flash           Zhipu AI GLM-4.7-Flash

Answer extraction strategy (per model family):
  - All models: greedy decoding (temperature=0), structured CoT prompt
  - Extraction priority:
      1. Explicit "ANSWER: <value>" tag in response
      2. "The answer is <value>" / "= <value>" pattern
      3. Last number in response (math) / last letter A-E (logic)
      4. Fallback: empty string (counted as wrong)

Output:
  data/experiment_results.json

Checkpointing: saves after every (model, problem) pair.
Resume: re-run the script; already-completed pairs are skipped.

Requirements
------------
  pip install requests tqdm

  Ollama must be installed. Models are pulled automatically if missing.

Usage
-----
  python 5_run_experiment.py
  python 5_run_experiment.py --models deepseek-r1:7b deepseek-r1:8b deepseek-r1:32b
  python 5_run_experiment.py --skip-pull   # if models already downloaded
"""

import argparse
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

OLLAMA_URL      = "http://localhost:11434"
DATA_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
INPUT_FILE      = os.path.join(DATA_DIR, "validated_paraphrases.json")
OUTPUT_FILE     = os.path.join(DATA_DIR, "experiment_results.json")
REQUEST_TIMEOUT = 480   # seconds per request

EVAL_MODELS = [
    "deepseek-r1:7b",
    "deepseek-r1:8b",
    "deepseek-r1:14b",
    "deepseek-r1:32b",
    "qwen3:8b",
    "qwen3:32b",
    "magistral:24b",
    "phi4-reasoning:14b",
    "glm-4.7-flash",
    "llama3.3:70b",
]

# Model family tags for extraction routing
MODEL_FAMILIES = {
    "deepseek-r1": "deepseek",
    "qwen3":       "qwen",
    "magistral":   "mistral",
    "phi4":        "phi",
    "glm":         "glm",
    "llama3.3":    "llama",
}

# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def is_ollama_running() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


OLLAMA_ENV = {
    **os.environ,
    "OLLAMA_MAX_LOADED_MODELS": "1",
    "OLLAMA_SCHED_SPREAD":      "0",
    "OLLAMA_KEEP_ALIVE":        "10m",
}


def restart_ollama_server():
    """Kill any running Ollama process and restart with GPU-optimised env vars."""
    print("Restarting Ollama server with GPU-optimised settings...")
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
            print("Ollama server started.")
            return
    print("ERROR: Could not start Ollama server.")
    sys.exit(1)


def start_ollama_server():
    print("Starting Ollama server...")
    subprocess.Popen(
        ["ollama", "serve"],
        env=OLLAMA_ENV,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(30):
        time.sleep(2)
        if is_ollama_running():
            print("Ollama server started.")
            return
    print("ERROR: Could not start Ollama server.")
    sys.exit(1)


def get_available_models() -> list[str]:
    r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    return [m["name"] for m in r.json().get("models", [])]


def ensure_model(model: str):
    available = get_available_models()
    tag  = model.split(":")[-1]
    base = model.split(":")[0]
    if any(base in m and tag in m for m in available):
        return
    print(f"  Pulling {model} (large models may take 10-30 min)...")
    result = subprocess.run(["ollama", "pull", model], check=False)
    if result.returncode != 0:
        print(f"ERROR: Failed to pull {model}.")
        sys.exit(1)
    print(f"  {model} ready.")


def unload_all_models():
    """Ask Ollama to unload all currently loaded models to free GPU VRAM.
    Uses the /api/generate endpoint with keep_alive=0 for each loaded model.
    Then restarts the server to ensure a clean slate.
    """
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
    time.sleep(3)  # Give Ollama time to release VRAM
    print("  GPU memory cleared.")


def ensure_ollama_ready(model: str, skip_pull: bool = False):
    unload_all_models()
    restart_ollama_server()
    if not skip_pull:
        ensure_model(model)

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

MATH_SYSTEM = (
    "You are an expert mathematician. Solve the problem step by step. "
    "At the end of your solution, write your final answer on its own line "
    "in the format:\nANSWER: <number>\n"
    "Use only digits and a decimal point if needed. No units, no commas."
)

LOGIC_SYSTEM = (
    "You are an expert at logical reasoning. Analyse the problem step by step. "
    "At the end of your analysis, write your final answer on its own line "
    "in the format:\nANSWER: <letter>\n"
    "The letter must be one of the answer choice labels (e.g. A, B, C, D)."
)


def build_eval_prompt(problem: dict, paraphrase_text: str) -> str:
    system = MATH_SYSTEM if problem["domain"] == "math" else LOGIC_SYSTEM
    return f"{system}\n\nPROBLEM:\n{paraphrase_text}\n\nSOLUTION:"


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def get_model_family(model: str) -> str:
    for prefix, family in MODEL_FAMILIES.items():
        if model.startswith(prefix):
            return family
    return "generic"


def strip_thinking_traces(text: str) -> str:
    """Remove <think>...</think> blocks used by DeepSeek R1 and some Qwen models."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Also strip /think or similar variants
    text = re.sub(r"</?think[^>]*>", "", text)
    return text.strip()


def extract_answer_math(text: str) -> str:
    """
    Multi-strategy numeric answer extraction for math problems.
    Priority:
      1. Explicit ANSWER: tag
      2. "the answer is X" / "= X" at end of sentence
      3. Boxed LaTeX: \\boxed{X}
      4. Last number in the cleaned text
    """
    text = strip_thinking_traces(text)

    # Strategy 1: ANSWER: tag
    m = re.search(r"ANSWER\s*:\s*([\-\d,\.\s/]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "").strip().split()[0]

    # Strategy 2: "the answer is X" or "equals X"
    m = re.search(
        r"(?:the answer is|equals|=)\s*([\-\d,\.]+)",
        text, re.IGNORECASE
    )
    if m:
        return m.group(1).replace(",", "").strip()

    # Strategy 3: LaTeX boxed
    m = re.search(r"\\boxed\{([\-\d,\.]+)\}", text)
    if m:
        return m.group(1).replace(",", "").strip()

    # Strategy 4: Last number in text
    nums = re.findall(r"[\-]?\d+(?:\.\d+)?", text)
    if nums:
        return nums[-1]

    return ""


def extract_answer_logic(text: str) -> str:
    """
    Multi-strategy letter answer extraction for multiple-choice logic problems.
    Priority:
      1. Explicit ANSWER: tag
      2. "the answer is (X)" or "answer: X"
      3. Last standalone letter A-E in the text
    """
    text = strip_thinking_traces(text)

    # Strategy 1: ANSWER: tag
    m = re.search(r"ANSWER\s*:\s*\(?([A-Ea-e])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Strategy 2: "the answer is (X)" or "answer is X"
    m = re.search(
        r"(?:the answer is|answer is|answer:)\s*\(?([A-Ea-e])\)?",
        text, re.IGNORECASE
    )
    if m:
        return m.group(1).upper()

    # Strategy 3: "(X)" pattern near the end of the text
    candidates = re.findall(r"\(([A-Ea-e])\)", text)
    if candidates:
        return candidates[-1].upper()

    # Strategy 4: Last standalone letter A-E
    candidates = re.findall(r"\b([A-Ea-e])\b", text)
    if candidates:
        return candidates[-1].upper()

    return ""


def extract_answer(text: str, domain: str) -> str:
    if domain == "math":
        return extract_answer_math(text)
    return extract_answer_logic(text)


def is_correct(extracted: str, ground_truth: str, domain: str) -> bool:
    if not extracted:
        return False
    if domain == "math":
        # Numeric comparison with tolerance
        try:
            return abs(float(extracted) - float(ground_truth.replace(",", ""))) < 1e-4
        except ValueError:
            return extracted.strip() == ground_truth.strip()
    else:
        return extracted.strip().upper() == ground_truth.strip().upper()

# ---------------------------------------------------------------------------
# Ollama inference
# ---------------------------------------------------------------------------

def call_ollama(prompt: str, model: str) -> tuple[str, float]:
    """Returns (response_text, elapsed_seconds)."""
    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,       # greedy decoding for reproducibility
            "num_predict": 1024,    # enough for CoT + answer
            "seed":        42,
        },
    }
    t0 = time.time()
    for attempt in range(3):
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            r.raise_for_status()
            elapsed = time.time() - t0
            return r.json()["response"].strip(), elapsed
        except Exception as exc:
            wait = 2 ** attempt
            print(f"    [attempt {attempt+1}/3] {exc}. Retry in {wait}s...")
            time.sleep(wait)
    return "", time.time() - t0

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint() -> dict:
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            return json.load(f)
    return {}


def save_checkpoint(data: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ARCG evaluation runner")
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Subset of models to run (default: all 10)"
    )
    parser.add_argument(
        "--skip-pull", action="store_true",
        help="Skip ollama pull (assume models already downloaded)"
    )
    args = parser.parse_args()

    models_to_run = args.models if args.models else EVAL_MODELS

    print("=" * 60)
    print("ARCG Step 5: Run Evaluation")
    print(f"Models: {models_to_run}")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        # Fall back to unvalidated paraphrases if validation not yet done
        fallback = os.path.join(DATA_DIR, "paraphrases.json")
        if os.path.exists(fallback):
            print(f"WARNING: {INPUT_FILE} not found.")
            print(f"Using {fallback} (unvalidated paraphrases).")
            input_path = fallback
        else:
            print(f"ERROR: Neither {INPUT_FILE} nor {fallback} found.")
            print("Run 1_build_and_paraphrase.py first.")
            sys.exit(1)
    else:
        input_path = INPUT_FILE

    with open(input_path) as f:
        problems = json.load(f)

    print(f"Loaded {len(problems)} problems.")

    results = load_checkpoint()
    print(f"Checkpoint: {len(results)} (model, problem) pairs already done.")

    if not is_ollama_running():
        start_ollama_server()

    for model in models_to_run:
        print(f"\n{'='*50}")
        print(f"Model: {model}")
        print(f"{'='*50}")

        ensure_ollama_ready(model, skip_pull=args.skip_pull)

        if model not in results:
            results[model] = {}

        done_pids = set(results[model].keys())
        remaining = [p for p in problems if p["id"] not in done_pids]
        print(f"  Remaining: {len(remaining)} / {len(problems)} problems")

        for problem in tqdm(remaining, desc=model):
            pid = problem["id"]
            problem_result = {
                "domain":     problem["domain"],
                "difficulty": problem["difficulty"],
                "answer":     problem["answer"],
                "variants":   {},
            }

            for variant in problem["paraphrases"]:
                vid    = variant["id"]
                prompt = build_eval_prompt(problem, variant["text"])
                raw, elapsed = call_ollama(prompt, model)

                extracted = extract_answer(raw, problem["domain"])
                correct   = is_correct(extracted, problem["answer"], problem["domain"])

                problem_result["variants"][vid] = {
                    "paraphrase_text":  variant["text"],
                    "raw_response":     raw,
                    "extracted_answer": extracted,
                    "correct":          correct,
                    "elapsed_sec":      round(elapsed, 2),
                }

            results[model][pid] = problem_result
            save_checkpoint(results)

        # Per-model accuracy summary
        total, correct_count = 0, 0
        for pid, pr in results[model].items():
            for vid, vr in pr["variants"].items():
                if vid == "P0":
                    total += 1
                    if vr["correct"]:
                        correct_count += 1

        acc = correct_count / max(total, 1)
        print(f"  Accuracy on P0 (original): {correct_count}/{total} = {acc:.1%}")

    print(f"\nAll models complete. Results saved to {OUTPUT_FILE}")
    print("Next step: run 6_analyze_and_plot.py")


if __name__ == "__main__":
    main()
