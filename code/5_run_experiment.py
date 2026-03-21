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


# CUDA library paths — needed when Ollama is launched as a subprocess
# (the user's shell may have these set but subprocess may not inherit them)
_LD_PATHS = [
    "/usr/local/cuda/lib64",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/local/lib",
]
_existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
_extra_ld = ":".join(p for p in _LD_PATHS if os.path.isdir(p))
_ld_library_path = ":".join(filter(None, [_extra_ld, _existing_ld]))

OLLAMA_ENV = {
    **os.environ,
    "OLLAMA_MAX_LOADED_MODELS": "1",
    "OLLAMA_SCHED_SPREAD":      "0",
    "OLLAMA_KEEP_ALIVE":        "10m",
    "LD_LIBRARY_PATH":          _ld_library_path,
    "CUDA_VISIBLE_DEVICES":     os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
}

OLLAMA_LOG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "ollama_server.log"
)


def restart_ollama_server():
    """Kill ALL Ollama processes (parent + runner children) and restart.

    Key fixes vs previous version:
    - Uses `pkill -f ollama` (not `-x`) to also kill `ollama runner` children
      that hold the GPU context open
    - Logs Ollama stdout/stderr to data/ollama_server.log instead of DEVNULL
      so GPU init errors are visible
    - Passes explicit LD_LIBRARY_PATH and CUDA_VISIBLE_DEVICES so the
      subprocess can find libcuda even if the shell environment differs
    """
    print("Restarting Ollama server with GPU-optimised settings...")

    # Kill parent + all children (runner, llama_server, etc.)
    for sig in ["-TERM", "-KILL"]:
        subprocess.run(["pkill", sig, "-f", "ollama"], check=False)
    time.sleep(5)  # give kernel time to reclaim GPU context

    # Force OS to reclaim page cache so Ollama sees enough MemFree
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from free_ram_cache import ensure_free_ram_gb
        ensure_free_ram_gb(40.0)
    except Exception as exc:
        print(f"  [RAM] free_ram_cache skipped: {exc}")

    os.makedirs(os.path.dirname(OLLAMA_LOG_FILE), exist_ok=True)
    log_fh = open(OLLAMA_LOG_FILE, "a")
    log_fh.write(f"\n\n=== ollama serve restart at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log_fh.flush()

    subprocess.Popen(
        ["ollama", "serve"],
        env=OLLAMA_ENV,
        stdout=log_fh,
        stderr=log_fh,
    )
    for _ in range(30):
        time.sleep(2)
        if is_ollama_running():
            print(f"Ollama server started. Logs: {OLLAMA_LOG_FILE}")
            return
    print(f"ERROR: Could not start Ollama server. Check logs: {OLLAMA_LOG_FILE}")
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
    """Prepare Ollama for the next model.

    Always runs GPU cleanup (unloads previous model from VRAM).
    Only restarts the Ollama server if it is not already running.
    Never kills a working Ollama process — restarting it breaks the
    CUDA environment that was set up when the user launched it manually.
    """
    # Step 1: Gracefully unload all currently-loaded models to free VRAM
    if is_ollama_running():
        print("  [setup] Unloading previous models from VRAM...")
        try:
            r = requests.get(f"{OLLAMA_URL}/api/ps", timeout=5)
            if r.status_code == 200:
                for m in r.json().get("models", []):
                    name = m.get("name", "")
                    if name:
                        requests.post(
                            f"{OLLAMA_URL}/api/generate",
                            json={"model": name, "keep_alive": 0},
                            timeout=30,
                        )
                        print(f"    Unloaded: {name}")
        except Exception as exc:
            print(f"    [setup] Could not unload models: {exc}")
        time.sleep(3)

    # Step 2: GPU cleanup (free VRAM without touching Ollama process)
    gpu_cleanup()

    # Step 3: Start Ollama only if it went down during cleanup
    if not is_ollama_running():
        print("  [setup] Ollama stopped during cleanup — restarting...")
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
    return f"{system}\n\nPROBLEM:\n{paraphrase_text}\n\nPlease solve the problem above and write your final answer in the required format."


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def get_model_family(model: str) -> str:
    for prefix, family in MODEL_FAMILIES.items():
        if model.startswith(prefix):
            return family
    return "generic"


def strip_thinking_traces(text: str) -> str:
    """Remove <think>...</think> blocks used by DeepSeek R1 and some Qwen models.

    Preserves the text AFTER the closing </think> tag, which is the actual answer.
    If the response is only a <think> block with nothing after it, returns the
    content inside the think block as a fallback so answer extraction can still
    find numbers/letters within the reasoning trace.
    """
    # Extract the part after </think> first
    after_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    after_think = re.sub(r"</?think[^>]*>", "", after_think).strip()

    if after_think:
        return after_think

    # Fallback: nothing after </think> — extract content inside <think> block
    inside = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if inside:
        return inside.group(1).strip()

    return text.strip()


def extract_answer_math(text: str) -> str:
    """
    Multi-strategy numeric answer extraction for math problems.
    Priority:
      1. Explicit ANSWER: tag (last occurrence)
      2. LaTeX boxed: \\boxed{X} (last occurrence) -- high confidence
      3. "the answer is X" / "= X" (last occurrence)
      4. Bold markdown answer: **X** near end
      5. Last number in the cleaned text

    All strategies use the LAST match to avoid picking up intermediate
    calculation steps (e.g. "= 2" in the middle of working) instead of
    the final answer at the end of the response.
    """
    text = strip_thinking_traces(text)

    # Strategy 1: Explicit ANSWER: tag (last occurrence wins)
    matches = re.findall(r"ANSWER\s*:\s*([\-\d,\.\s/]+)", text, re.IGNORECASE)
    if matches:
        parts = matches[-1].replace(",", "").strip().split()
        if parts:
            return parts[0]

    # Strategy 2: LaTeX boxed (last occurrence) -- very reliable signal
    # Match both \boxed{X} (1 backslash) and \\boxed{X} (2 backslashes, from
    # double-escaped JSON storage) so this works regardless of how the string
    # was loaded.
    matches = re.findall(r"\\{1,2}boxed\{([\-\d,\.]+)\}", text)
    if matches:
        return matches[-1].replace(",", "").strip()

    # Strategy 3: "the answer is X" or "equals X" (last occurrence)
    matches = re.findall(
        r"(?:the answer is|equals)\s*([\-\d,\.]+)",
        text, re.IGNORECASE
    )
    if matches:
        return matches[-1].replace(",", "").strip()

    # Strategy 4: Bold markdown **X** near end of text
    matches = re.findall(r"\*\*([\-\d,\.]+)\*\*", text)
    if matches:
        return matches[-1].replace(",", "").strip()

    # Strategy 5: Last number in text
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

def _call_ollama_stream(prompt: str, model: str, options: dict) -> str:
    """Call Ollama using streaming mode and assemble the full response.

    Newer Ollama versions (>=0.6) split the stream into two fields:
      - 'thinking': tokens inside the <think> block
      - 'response': tokens after </think> (the actual answer)

    Older versions put everything (including <think> tags) into 'response'.

    We collect BOTH fields so we always get the full text regardless of
    which Ollama version is running.  The thinking block is wrapped back
    into <think>...</think> tags so strip_thinking_traces() can parse it.
    """
    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": True,
        "options": options,
    }
    thinking_chunks: list[str] = []
    response_chunks: list[str] = []
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        stream=True,
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    for raw_line in r.iter_lines():
        if not raw_line:
            continue
        try:
            chunk = json.loads(raw_line)
        except Exception:
            continue
        thinking_token = chunk.get("thinking", "")
        response_token = chunk.get("response", "")
        if thinking_token:
            thinking_chunks.append(thinking_token)
        if response_token:
            response_chunks.append(response_token)
        if chunk.get("done"):
            break

    thinking_text = "".join(thinking_chunks).strip()
    response_text = "".join(response_chunks).strip()

    # Reconstruct full text so strip_thinking_traces() works correctly
    if thinking_text and response_text:
        return f"<think>{thinking_text}</think>\n{response_text}"
    elif thinking_text:
        # Model only produced thinking tokens (no answer after </think>)
        # Wrap so strip_thinking_traces fallback can search inside
        return f"<think>{thinking_text}</think>"
    else:
        return response_text


def call_ollama(prompt: str, model: str) -> tuple[str, float]:
    """Returns (response_text, elapsed_seconds).

    Uses non-streaming mode (confirmed working).  Collects BOTH the
    'response' field (answer after </think>) and the 'thinking' field
    (tokens inside <think>) so the full CoT trace is preserved.

    num_predict=8192 gives deepseek-r1:7b enough budget to finish its
    thinking chain AND write the ANSWER line (4096 was too short).
    """
    options = {
        "temperature": 0,        # greedy decoding for reproducibility
        "num_predict": -1,       # unlimited: let model finish thinking + answer
        "num_ctx":     32768,    # 32k context window (fits long reasoning chains)
        "seed":        42,
    }
    t0 = time.time()
    max_attempts = 8
    for attempt in range(max_attempts):
        try:
            payload = {
                "model":  model,
                "prompt": prompt,
                "stream": False,
                "options": options,
            }
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            if r.status_code == 500:
                err = r.text[:200]
                wait = min(60, 5 * (attempt + 1))
                print(f"    [attempt {attempt+1}/{max_attempts}] HTTP 500: {err}. Retry in {wait}s...")
                if attempt == max_attempts // 2:
                    restart_ollama_server()
                    ensure_model(model)
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()

            # Ollama >=0.6 splits thinking models into two fields:
            #   'thinking' = tokens inside <think> block
            #   'response' = tokens after </think> (the actual answer)
            # Older versions put everything into 'response'.
            # We collect both so the full CoT trace is preserved.
            thinking = data.get("thinking", "").strip()
            response = data.get("response", "").strip()

            if thinking and response:
                resp_text = f"<think>{thinking}</think>\n{response}"
            elif response:
                resp_text = response
            elif thinking:
                # Model ran out of num_predict inside <think> — no answer yet.
                # Return thinking so strip_thinking_traces fallback can search it.
                resp_text = f"<think>{thinking}</think>"
            else:
                resp_text = ""

            if resp_text:
                elapsed = time.time() - t0
                return resp_text, elapsed

            wait = 5
            print(f"    [attempt {attempt+1}/{max_attempts}] Empty response. "
                  f"eval_count={data.get('eval_count')} done_reason={data.get('done_reason')}. "
                  f"Retry in {wait}s...")
            if attempt == max_attempts // 2:
                print("    [attempt] Half retries exhausted — restarting Ollama server...")
                restart_ollama_server()
                ensure_model(model)
            time.sleep(wait)

        except requests.exceptions.Timeout:
            wait = 15
            print(f"    [attempt {attempt+1}/{max_attempts}] Request timed out. Retry in {wait}s...")
            time.sleep(wait)
        except Exception as exc:
            wait = min(30, 5 * (attempt + 1))
            print(f"    [attempt {attempt+1}/{max_attempts}] {type(exc).__name__}: {exc}. Retry in {wait}s...")
            if "500" in str(exc) and attempt == max_attempts // 2:
                restart_ollama_server()
                ensure_model(model)
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
