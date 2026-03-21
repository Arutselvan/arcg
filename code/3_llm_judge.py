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
    # Force OS to reclaim page cache so Ollama sees enough MemFree
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from free_ram_cache import ensure_free_ram_gb
        ensure_free_ram_gb(40.0)
    except Exception as exc:
        print(f"  [RAM] free_ram_cache skipped: {exc}")
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


def gpu_cleanup():
    """Aggressively free all GPU VRAM and CUDA contexts before loading a new model."""
    print("  [GPU cleanup] Starting full GPU/CUDA memory cleanup...")

    # Step 1: Ask Ollama to unload all models gracefully
    try:
        r = requests.get(f"{OLLAMA_URL}/api/ps", timeout=5)
        if r.status_code == 200:
            for m in r.json().get("models", []):
                name = m.get("name", "")
                if name:
                    try:
                        requests.post(
                            f"{OLLAMA_URL}/api/generate",
                            json={"model": name, "keep_alive": 0},
                            timeout=30,
                        )
                        print(f"    [GPU cleanup] Unloaded model: {name}")
                    except Exception:
                        pass
    except Exception:
        pass
    time.sleep(2)

    # Step 2: Kill all Ollama processes with SIGKILL
    for sig in ["-TERM", "-KILL"]:
        subprocess.run(["pkill", sig, "-f", "ollama"], check=False)
    time.sleep(3)

    # Step 3: Kill stray processes by name
    for proc_name in ["ollama runner", "ollama_llama_server"]:
        subprocess.run(["pkill", "-9", "-f", proc_name], check=False)
    time.sleep(2)

    # Step 4: Use fuser to kill any process holding /dev/nvidia* device files
    print("    [GPU cleanup] Scanning /dev/nvidia* for processes holding GPU device files...")
    gpu_devices = ["/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm",
                   "/dev/nvidia-modeset", "/dev/nvidia-uvm-tools"]
    killed_pids = set()
    for dev in gpu_devices:
        try:
            result = subprocess.run(["fuser", dev], capture_output=True, text=True, check=False)
            for pid in result.stdout.strip().split():
                pid = pid.strip()
                if pid and pid not in killed_pids:
                    subprocess.run(["kill", "-9", pid], check=False)
                    print(f"    [GPU cleanup] Killed PID {pid} holding {dev}")
                    killed_pids.add(pid)
        except Exception:
            pass
    if killed_pids:
        time.sleep(3)
    else:
        print("    [GPU cleanup] No processes found holding GPU device files.")

    # Step 5a: Python torch CUDA cache clear (if torch is installed)
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    torch.cuda.synchronize()
            print(f"    [GPU cleanup] torch.cuda.empty_cache() cleared on {torch.cuda.device_count()} device(s).")
    except ImportError:
        pass
    except Exception as exc:
        print(f"    [GPU cleanup] torch CUDA clear skipped: {exc}")

    # Step 5b: Reset CUDA primary contexts via ctypes
    try:
        import ctypes
        libcuda = None
        for path in ["/usr/lib/x86_64-linux-gnu/libcuda.so.1",
                     "/usr/local/cuda/lib64/libcuda.so", "libcuda.so.1"]:
            try:
                libcuda = ctypes.CDLL(path); break
            except OSError:
                continue
        if libcuda:
            libcuda.cuInit(0)
            device_count = ctypes.c_int(0)
            libcuda.cuDeviceGetCount(ctypes.byref(device_count))
            for i in range(device_count.value):
                device = ctypes.c_int(0)
                libcuda.cuDeviceGet(ctypes.byref(device), i)
                ret = libcuda.cuDevicePrimaryCtxReset(device)
                print(f"    [GPU cleanup] cuDevicePrimaryCtxReset(GPU {i}) = {ret} (0=success, 201=not owner)")
        else:
            print("    [GPU cleanup] libcuda.so not found; skipping ctypes reset.")
    except Exception as exc:
        print(f"    [GPU cleanup] ctypes CUDA reset skipped: {exc}")

    # Step 6: Try nvidia-smi --gpu-reset (bare metal only)
    try:
        result = subprocess.run(["nvidia-smi", "--gpu-reset", "-i", "0"],
                                capture_output=True, text=True, timeout=30, check=False)
        if result.returncode == 0:
            print("    [GPU cleanup] nvidia-smi --gpu-reset succeeded.")
        else:
            print(f"    [GPU cleanup] nvidia-smi --gpu-reset: {result.stderr.strip()[:80]}")
    except Exception:
        pass

    # Step 7: Try unloading nvidia_uvm kernel module
    try:
        r = subprocess.run(["sudo", "rmmod", "nvidia_uvm"],
                           capture_output=True, text=True, timeout=15, check=False)
        if r.returncode == 0:
            print("    [GPU cleanup] nvidia_uvm unloaded.")
            time.sleep(2)
            subprocess.run(["sudo", "modprobe", "nvidia_uvm"], check=False)
            print("    [GPU cleanup] nvidia_uvm reloaded.")
        else:
            print(f"    [GPU cleanup] rmmod nvidia_uvm: {r.stderr.strip()[:80]}")
    except Exception:
        pass

    # Step 8: Verify VRAM
    time.sleep(5)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False)
        if result.returncode == 0:
            free_mb, total_mb = [int(x.strip()) for x in result.stdout.strip().split(",")]
            used_mb = total_mb - free_mb
            print(f"    [GPU cleanup] VRAM: {used_mb} MiB used / {total_mb} MiB total ({free_mb} MiB free)")
            if used_mb > 2000:
                print("    [GPU cleanup] WARNING: VRAM not fully cleared. A reboot may be needed.")
    except Exception:
        pass
    print("  [GPU cleanup] Done.")


def ensure_ollama_ready(model: str):
    gpu_cleanup()
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
        f"You MUST respond using ONLY these three lines, with no preamble or explanation:\n"
        f"VERDICT: VALID\n"
        f"CONFIDENCE: 5\n"
        f"REASON: one sentence here\n\n"
        f"Replace VALID with INVALID if the paraphrase fails any criterion. "
        f"Replace 5 with your actual confidence (1=very unsure, 5=certain). "
        f"Do not add any other text before or after these three lines."
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
    max_attempts = 8
    for attempt in range(max_attempts):
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            if r.status_code == 500:
                # Log the actual error body so we can see what is happening
                try:
                    err_body = r.json()
                except Exception:
                    err_body = r.text[:200]
                wait = min(30, 5 * (attempt + 1))
                print(f"    [attempt {attempt+1}/{max_attempts}] HTTP 500: {err_body}. "
                      f"Retry in {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            text = r.json().get("response", "").strip()
            # Strip DeepSeek R1 / Qwen3 thinking traces
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            if text:
                return text
            # Empty response -- retry
            wait = 5
            print(f"    [attempt {attempt+1}/{max_attempts}] Empty response. Retry in {wait}s...")
            time.sleep(wait)
        except requests.exceptions.Timeout:
            wait = 15
            print(f"    [attempt {attempt+1}/{max_attempts}] Timeout. Retry in {wait}s...")
            time.sleep(wait)
        except Exception as exc:
            wait = min(30, 5 * (attempt + 1))
            print(f"    [attempt {attempt+1}/{max_attempts}] {type(exc).__name__}: {exc}. "
                  f"Retry in {wait}s...")
            time.sleep(wait)
    print(f"    WARNING: All {max_attempts} attempts failed. Returning empty response.")
    return ""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_verdict(response: str, raw: str = "") -> dict:
    """Parse a judge verdict from the model response.

    Handles three response styles:
      1. Structured:  VERDICT: VALID / CONFIDENCE: 4 / REASON: ...
      2. Natural language containing 'valid' or 'invalid' keyword
      3. Sentiment fallback: positive language -> VALID, negative -> INVALID
    """
    verdict    = "UNKNOWN"
    confidence = 0
    reason     = ""

    if not response:
        return {"verdict": verdict, "confidence": confidence,
                "reason": reason, "raw_response": raw}

    # --- Style 1: structured key:value format ---
    v_match = re.search(r"VERDICT\s*[:\-]\s*(VALID|INVALID)", response, re.IGNORECASE)
    c_match = re.search(r"CONFIDENCE\s*[:\-]\s*([1-5])", response)
    r_match = re.search(r"REASON\s*[:\-]\s*(.+)", response, re.IGNORECASE)

    if v_match:
        verdict    = v_match.group(1).upper()
        confidence = int(c_match.group(1)) if c_match else 3
        reason     = r_match.group(1).strip() if r_match else ""
        return {"verdict": verdict, "confidence": confidence,
                "reason": reason, "raw_response": raw}

    # --- Style 2: natural language with explicit VALID/INVALID word ---
    # e.g. "The paraphrase is VALID" or "This is an invalid paraphrase"
    nl_match = re.search(r"\b(VALID|INVALID)\b", response, re.IGNORECASE)
    if nl_match:
        verdict = nl_match.group(1).upper()
        # Try to extract a confidence number anywhere in the response
        any_conf = re.search(r"\b([1-5])\s*(?:/\s*5|out of 5)?\b", response)
        confidence = int(any_conf.group(1)) if any_conf else 3
        # Use the whole response as the reason (truncated)
        reason = response.strip()[:300]
        return {"verdict": verdict, "confidence": confidence,
                "reason": reason, "raw_response": raw}

    # --- Style 3: sentiment fallback ---
    # If the model says things like "maintains the same structure", "equivalent",
    # "preserves", "correctly", treat as VALID; "changes", "alters", "incorrect" -> INVALID
    pos_words = re.compile(
        r"\b(equivalent|preserves|maintains|correct|same|identical|valid|accurate|appropriate)\b",
        re.IGNORECASE)
    neg_words = re.compile(
        r"\b(invalid|incorrect|changes|alters|different|loses|adds|removes|misleading)\b",
        re.IGNORECASE)
    pos_hits = len(pos_words.findall(response))
    neg_hits = len(neg_words.findall(response))
    if pos_hits > neg_hits:
        verdict    = "VALID"
        confidence = 2   # low confidence since we used fallback
        reason     = f"[fallback] {response.strip()[:200]}"
    elif neg_hits > pos_hits:
        verdict    = "INVALID"
        confidence = 2
        reason     = f"[fallback] {response.strip()[:200]}"
    else:
        reason = response.strip()[:200]

    return {"verdict": verdict, "confidence": confidence,
            "reason": reason, "raw_response": raw}

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
            parsed        = parse_verdict(response, raw=response)
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
