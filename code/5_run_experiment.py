"""
ARCG Experiment  --  Step 5: Run Evaluation Across 10 Reasoning Models
=======================================================================
Reads data/paraphrases.json (validated paraphrases) and runs all
10 reasoning models on every (problem, paraphrase) pair using
chain-of-thought prompting.

Models evaluated (all reasoning models, all fit on H100 80GB at Q4):
  1.  deepseek-r1:7b          DeepSeek R1 distill (Qwen base)
  2.  deepseek-r1:8b          DeepSeek R1 distill (Llama base)
  3.  deepseek-r1:14b         DeepSeek R1 distill
  4.  deepseek-r1:32b         DeepSeek R1 distill (largest in DeepSeek R1 series)
  5.  qwen3:30b               Qwen 3 30B (dense reasoning, replaces deepseek-r1:70b)
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
INPUT_FILE      = os.path.join(DATA_DIR, "paraphrases.json")
OUTPUT_FILE     = os.path.join(DATA_DIR, "experiment_results.json")
REQUEST_TIMEOUT = 480   # seconds per request

EVAL_MODELS = [
    "deepseek-r1:7b",
    "deepseek-r1:8b",
    "deepseek-r1:14b",
    "deepseek-r1:32b",
    "qwen3:30b",
    "qwen3:8b",
    "qwen3:32b",
    "magistral:24b",
    "phi4-reasoning:14b",
    "glm-4.7-flash",
]

# Model family tags for extraction routing
MODEL_FAMILIES = {
    "deepseek-r1": "deepseek",
    "qwen3":       "qwen",
    "magistral":   "mistral",
    "phi4":        "phi",
    "glm":         "glm",
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


def ensure_ollama_ready(model: str, skip_pull: bool = False):
    gpu_cleanup()
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
        parts = m.group(1).replace(",", "").strip().split()
        if parts:
            return parts[0]

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
            "num_predict": 4096,    # allow full reasoning chain + answer
            "num_ctx":     8192,    # override Ollama's VRAM-based auto-ctx (262K)
            "seed":        42,
        },
    }
    t0 = time.time()
    max_attempts = 8
    for attempt in range(max_attempts):
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            if r.status_code == 500:
                err_msg = ""
                try:
                    err_msg = r.json().get("error", r.text[:120])
                except Exception:
                    err_msg = r.text[:120]
                wait = min(60, 5 * (attempt + 1))
                print(f"    [attempt {attempt+1}/{max_attempts}] HTTP 500: {err_msg}. Retry in {wait}s...")
                if attempt == max_attempts // 2:
                    print("    [attempt] Half retries exhausted — restarting Ollama server...")
                    restart_ollama_server()
                    ensure_model(model)
                time.sleep(wait)
                continue
            r.raise_for_status()
            resp_text = r.json().get("response", "").strip()
            if not resp_text:
                wait = 5
                print(f"    [attempt {attempt+1}/{max_attempts}] Empty response from model. Retry in {wait}s...")
                time.sleep(wait)
                continue
            elapsed = time.time() - t0
            return resp_text, elapsed
        except requests.exceptions.Timeout:
            wait = 15
            print(f"    [attempt {attempt+1}/{max_attempts}] Request timed out. Retry in {wait}s...")
            time.sleep(wait)
        except Exception as exc:
            wait = min(30, 5 * (attempt + 1))
            print(f"    [attempt {attempt+1}/{max_attempts}] {type(exc).__name__}: {exc}. Retry in {wait}s...")
            time.sleep(wait)
    print(f"    ERROR: All {max_attempts} attempts failed for model={model}. Returning empty response.")
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
