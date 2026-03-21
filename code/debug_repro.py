"""
debug_repro.py -- Reproduce exactly what script 5 does and print raw results.
Run this on the server to see what is actually returned.

Usage:
    python3 code/debug_repro.py
"""
import json, re, requests, time

OLLAMA_URL = "http://localhost:11434"

MATH_SYSTEM = (
    "You are an expert mathematician. Solve the problem step by step. "
    "At the end of your solution, write your final answer on its own line "
    "in the format:\nANSWER: <number>\n"
    "Use only digits and a decimal point if needed. No units, no commas."
)

import os, sys
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
with open(os.path.join(DATA_DIR, "paraphrases.json")) as f:
    problems = json.load(f)

p = problems[0]
v = p["paraphrases"][0]
prompt = f"{MATH_SYSTEM}\n\nPROBLEM:\n{v['text']}\n\nPlease solve the problem above and write your final answer in the required format."

model = sys.argv[1] if len(sys.argv) > 1 else "deepseek-r1:7b"

print(f"Model: {model}")
print(f"Prompt length: {len(prompt)} chars")
print(f"Prompt preview: {prompt[:200]}")
print()

# Exactly what script 5 sends
payload = {
    "model": model,
    "prompt": prompt,
    "stream": False,
    "options": {
        "temperature": 0,
        "num_predict": 4096,
        "num_ctx": 8192,
        "seed": 42,
    },
}

print("Sending request (stream=False, num_ctx=8192, num_predict=4096)...")
t0 = time.time()
r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=480)
elapsed = time.time() - t0
print(f"HTTP {r.status_code} in {elapsed:.1f}s")
print(f"Raw body length: {len(r.text)}")

try:
    d = r.json()
except Exception as e:
    print(f"JSON parse error: {e}")
    print(f"Raw body: {r.text[:500]}")
    sys.exit(1)

print(f"Keys: {list(d.keys())}")
print(f"eval_count: {d.get('eval_count')}")
print(f"prompt_eval_count: {d.get('prompt_eval_count')}")
print(f"done: {d.get('done')}")
print(f"done_reason: {d.get('done_reason')}")
print(f"response length: {len(d.get('response', ''))}")
print(f"thinking length: {len(d.get('thinking', ''))}")
print()
print("=== response (first 300) ===")
print(repr(d.get("response", "")[:300]))
print()
print("=== thinking (first 300) ===")
print(repr(d.get("thinking", "")[:300]))
