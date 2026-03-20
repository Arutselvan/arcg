"""
ARCG Experiment  --  Step 3: LLM-as-Judge Paraphrase Evaluation
================================================================
Uses two strong reasoning models to independently judge whether each
generated paraphrase is semantically equivalent to the original problem.

Judge models:
  Judge A: deepseek-r1:32b
  Judge B: qwen3:32b

Each judge produces a structured verdict (VALID/INVALID) with a
confidence score (1-5) and a brief justification for each paraphrase.

Output files:
  data/llm_judge_deepseek-r1-32b.json
  data/llm_judge_qwen3-32b.json

These files are consumed by 4_consolidate_validation.py together with
the two human annotation Excel files.

Requirements
------------
  pip install requests tqdm

  Ollama must be installed and the two judge models must be available.
  This script pulls them automatically if they are not present.

Usage
-----
  python 3_llm_judge.py
  python 3_llm_judge.py --judge deepseek-r1:32b   # run only one judge
  python 3_llm_judge.py --judge qwen3:32b
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

OLLAMA_URL     = "http://localhost:11434"
DATA_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
PARAPHRASE_FILE = os.path.join(DATA_DIR, "paraphrases.json")

JUDGE_MODELS = [
    "deepseek-r1:32b",
    "qwen3:32b",
]

REQUEST_TIMEOUT = 300

# ---------------------------------------------------------------------------
# Ollama helpers (same pattern as script 1)
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
    tag = model.split(":")[-1]
    base = model.split(":")[0]
    if any(base in m and tag in m for m in available):
        print(f"  {model} already available.")
        return
    print(f"  Pulling {model}...")
    result = subprocess.run(["ollama", "pull", model], check=False)
    if result.returncode != 0:
        print(f"ERROR: Failed to pull {model}.")
        sys.exit(1)
    print(f"  {model} ready.")


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


def ensure_ollama_ready(model: str):
    unload_all_models()
    restart_ollama_server()
    ensure_model(model)

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You are a rigorous linguistic and logical expert evaluating paraphrase quality. "
    "Your judgments must be precise, consistent, and grounded in the text."
)


def build_judge_prompt(problem: dict, original: str, paraphrase: str,
                        strategy_desc: str) -> str:
    domain_hint = (
        "This is a mathematical word problem. The correct numerical answer must be identical."
        if problem["domain"] == "math"
        else (
            "This is a multiple-choice logical reasoning question. "
            "The answer choices and correct answer letter must be identical."
        )
    )
    return (
        f"{JUDGE_SYSTEM}\n\n"
        f"TASK: Determine whether the PARAPHRASE is semantically equivalent to the ORIGINAL.\n\n"
        f"DOMAIN: {problem['domain'].upper()}  |  DIFFICULTY: {problem['difficulty']}\n"
        f"PARAPHRASE STRATEGY: {strategy_desc}\n"
        f"{domain_hint}\n\n"
        f"ORIGINAL:\n{original}\n\n"
        f"PARAPHRASE:\n{paraphrase}\n\n"
        f"CORRECT ANSWER (for reference): {problem['answer']}\n\n"
        f"Evaluate the paraphrase on these criteria:\n"
        f"  1. Semantic equivalence: Does it ask for exactly the same thing?\n"
        f"  2. Numerical/choice preservation: Are all values and options identical?\n"
        f"  3. No information added or removed that would change difficulty or answer.\n"
        f"  4. Grammatical coherence: Is it a well-formed question?\n\n"
        f"Respond in this EXACT format (no other text):\n"
        f"VERDICT: <VALID or INVALID>\n"
        f"CONFIDENCE: <integer 1-5, where 5 is most confident>\n"
        f"REASON: <one sentence explaining your judgment>\n"
    )


STRATEGY_DESCRIPTIONS = {
    "P1": "Formal restatement -- precise mathematical/logical language",
    "P2": "Informal restatement -- casual everyday language",
    "P3": "Passive restructuring -- passive voice, different sentence order",
    "P4": "Decomposed -- problem split into numbered sub-steps",
    "P5": "Analogical -- same structure, different real-world context",
}

# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------

def call_ollama(prompt: str, model: str) -> str:
    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,   # deterministic judgments
            "num_predict": 256,
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
            # Strip DeepSeek R1 thinking traces
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            return text
        except Exception as exc:
            wait = 2 ** attempt
            print(f"    [attempt {attempt+1}/3] {exc}. Retry in {wait}s...")
            time.sleep(wait)
    return ""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_verdict(response: str) -> dict:
    verdict    = "UNKNOWN"
    confidence = 0
    reason     = ""

    v_match = re.search(r"VERDICT\s*:\s*(VALID|INVALID)", response, re.IGNORECASE)
    c_match = re.search(r"CONFIDENCE\s*:\s*([1-5])", response)
    r_match = re.search(r"REASON\s*:\s*(.+)", response, re.IGNORECASE)

    if v_match:
        verdict = v_match.group(1).upper()
    if c_match:
        confidence = int(c_match.group(1))
    if r_match:
        reason = r_match.group(1).strip()

    return {"verdict": verdict, "confidence": confidence, "reason": reason}

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def output_path(model: str) -> str:
    safe_name = model.replace(":", "-").replace("/", "-")
    return os.path.join(DATA_DIR, f"llm_judge_{safe_name}.json")


def load_checkpoint(model: str) -> dict:
    path = output_path(model)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_checkpoint(model: str, data: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(output_path(model), "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_judge(model: str, problems: list[dict]):
    print(f"\n{'='*60}")
    print(f"Running judge: {model}")
    print(f"{'='*60}")

    ensure_ollama_ready(model)

    results = load_checkpoint(model)
    print(f"  Checkpoint: {len(results)} problems already judged.")

    for problem in tqdm(problems, desc=f"Judging [{model}]"):
        pid = problem["id"]
        if pid in results:
            continue

        p0_text = next(
            (v["text"] for v in problem["paraphrases"] if v["id"] == "P0"),
            "",
        )
        problem_results = {}

        for variant in problem["paraphrases"]:
            vid = variant["id"]
            if vid == "P0":
                # Original is always valid by definition
                problem_results[vid] = {
                    "verdict":    "VALID",
                    "confidence": 5,
                    "reason":     "Original problem -- no judgment needed.",
                }
                continue

            strategy_desc = STRATEGY_DESCRIPTIONS.get(vid, vid)
            prompt        = build_judge_prompt(
                problem, p0_text, variant["text"], strategy_desc
            )
            response      = call_ollama(prompt, model)
            parsed        = parse_verdict(response)
            parsed["raw_response"] = response
            problem_results[vid]   = parsed

        results[pid] = problem_results
        save_checkpoint(model, results)

    print(f"  Judge {model} complete. Results saved to {output_path(model)}")

    # Print summary
    total, valid, invalid, unknown = 0, 0, 0, 0
    for pid, variants in results.items():
        for vid, r in variants.items():
            if vid == "P0":
                continue
            total += 1
            v = r.get("verdict", "UNKNOWN")
            if v == "VALID":
                valid += 1
            elif v == "INVALID":
                invalid += 1
            else:
                unknown += 1

    print(f"\n  Summary for {model}:")
    print(f"    Total paraphrases judged : {total}")
    print(f"    VALID                    : {valid}  ({100*valid/max(total,1):.1f}%)")
    print(f"    INVALID                  : {invalid}  ({100*invalid/max(total,1):.1f}%)")
    print(f"    UNKNOWN (parse error)    : {unknown}")


def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge paraphrase evaluation")
    parser.add_argument(
        "--judge",
        type=str,
        default=None,
        help="Run only this judge model (e.g. deepseek-r1:32b). Default: run all.",
    )
    args = parser.parse_args()

    if not os.path.exists(PARAPHRASE_FILE):
        print(f"ERROR: {PARAPHRASE_FILE} not found. Run 1_build_and_paraphrase.py first.")
        sys.exit(1)

    with open(PARAPHRASE_FILE) as f:
        problems = json.load(f)

    print(f"Loaded {len(problems)} problems.")

    models_to_run = [args.judge] if args.judge else JUDGE_MODELS

    for model in models_to_run:
        run_judge(model, problems)

    print("\nAll judges complete.")
    print("Next step: run 4_consolidate_validation.py")


if __name__ == "__main__":
    main()
