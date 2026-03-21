"""
debug_ollama.py  --  Diagnose empty-response issues with Ollama
================================================================
Run this FIRST before 5_run_experiment.py to understand exactly
what Ollama is returning.

Usage:
    python3 code/debug_ollama.py
    python3 code/debug_ollama.py --model deepseek-r1:7b
    python3 code/debug_ollama.py --stream   # test streaming endpoint
"""

import argparse
import json
import sys
import time

import requests

OLLAMA_URL = "http://localhost:11434"
TEST_PROMPT = (
    "You are an expert mathematician. Solve the problem step by step. "
    "At the end of your solution, write your final answer on its own line "
    "in the format:\nANSWER: <number>\n"
    "\nPROBLEM:\nJanet's ducks lay 16 eggs per day. She eats 3 for breakfast "
    "every morning and bakes muffins for her friends every day with 4. She "
    "sells the remainder at the farmers' market daily for $2 per fresh duck "
    "egg. How much in dollars does she make every day at the farmers' market?\n"
    "\nPlease solve the problem above and write your final answer in the "
    "required format."
)


def check_server():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"[OK] Ollama is running. Available models: {models}")
        return models
    except Exception as e:
        print(f"[FAIL] Ollama not reachable: {e}")
        sys.exit(1)


def test_generate_non_stream(model: str):
    print(f"\n--- Non-streaming /api/generate with model={model} ---")
    payload = {
        "model": model,
        "prompt": TEST_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 512,
            "num_ctx": 4096,
            "seed": 42,
        },
    }
    print(f"Payload (options): {payload['options']}")
    t0 = time.time()
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=300,
        )
        elapsed = time.time() - t0
        print(f"HTTP status: {r.status_code}  ({elapsed:.1f}s)")
        print(f"Response headers: {dict(r.headers)}")
        raw_text = r.text
        print(f"Raw response body length: {len(raw_text)} chars")
        print(f"First 500 chars of raw body:\n{raw_text[:500]}")
        try:
            data = r.json()
        except Exception:
            print("[FAIL] Could not parse JSON response body.")
            return
        print(f"\nJSON keys: {list(data.keys())}")
        response_field = data.get("response", "<KEY MISSING>")
        print(f"data['response'] repr (first 300): {repr(response_field[:300])}")
        print(f"data['response'] length: {len(response_field)}")
        if data.get("done") is False:
            print("[WARN] done=False — model may have been cut off (num_predict too low?)")
        print(f"done: {data.get('done')}")
        print(f"done_reason: {data.get('done_reason')}")
        print(f"eval_count (tokens generated): {data.get('eval_count')}")
        print(f"prompt_eval_count (tokens in prompt): {data.get('prompt_eval_count')}")
    except requests.exceptions.Timeout:
        print("[FAIL] Request timed out after 300s")
    except Exception as exc:
        print(f"[FAIL] Exception: {type(exc).__name__}: {exc}")


def test_generate_stream(model: str):
    print(f"\n--- Streaming /api/generate with model={model} ---")
    payload = {
        "model": model,
        "prompt": TEST_PROMPT,
        "stream": True,
        "options": {
            "temperature": 0,
            "num_predict": 512,
            "num_ctx": 4096,
            "seed": 42,
        },
    }
    t0 = time.time()
    chunks = []
    try:
        with requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            stream=True,
            timeout=300,
        ) as r:
            print(f"HTTP status: {r.status_code}")
            for line in r.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        chunks.append(token)
                        if chunk.get("done"):
                            print(f"\n[done=True] eval_count={chunk.get('eval_count')} done_reason={chunk.get('done_reason')}")
                            break
                    except Exception:
                        pass
        elapsed = time.time() - t0
        full_text = "".join(chunks)
        print(f"Total tokens streamed: {len(chunks)}")
        print(f"Total text length: {len(full_text)}")
        print(f"Elapsed: {elapsed:.1f}s")
        print(f"First 500 chars:\n{full_text[:500]}")
        print(f"Last 200 chars:\n{full_text[-200:]}")
    except Exception as exc:
        print(f"[FAIL] Exception: {type(exc).__name__}: {exc}")


def test_chat_endpoint(model: str):
    print(f"\n--- /api/chat with model={model} ---")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 512,
            "num_ctx": 4096,
            "seed": 42,
        },
    }
    t0 = time.time()
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=300,
        )
        elapsed = time.time() - t0
        print(f"HTTP status: {r.status_code}  ({elapsed:.1f}s)")
        try:
            data = r.json()
        except Exception:
            print(f"[FAIL] Non-JSON body: {r.text[:200]}")
            return
        msg = data.get("message", {})
        content = msg.get("content", "<MISSING>")
        print(f"message.content length: {len(content)}")
        print(f"First 500 chars:\n{content[:500]}")
        print(f"done: {data.get('done')}  done_reason: {data.get('done_reason')}")
        print(f"eval_count: {data.get('eval_count')}")
    except Exception as exc:
        print(f"[FAIL] Exception: {type(exc).__name__}: {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-r1:7b")
    parser.add_argument("--stream", action="store_true", help="Also test streaming endpoint")
    parser.add_argument("--chat", action="store_true", help="Also test /api/chat endpoint")
    args = parser.parse_args()

    available = check_server()

    # Auto-pick a model if the requested one is not available
    model = args.model
    if not any(model in m for m in available):
        if available:
            model = available[0]
            print(f"[WARN] {args.model} not found. Using {model} instead.")
        else:
            print("[FAIL] No models available. Pull one first: ollama pull deepseek-r1:7b")
            sys.exit(1)

    test_generate_non_stream(model)

    if args.stream:
        test_generate_stream(model)

    if args.chat:
        test_chat_endpoint(model)

    print("\n=== DIAGNOSIS SUMMARY ===")
    print("If data['response'] is empty in non-stream but non-empty in stream,")
    print("  => switch call_ollama to use streaming mode.")
    print("If eval_count=0 or done_reason='load', model did not generate tokens.")
    print("  => model failed to load; check VRAM with: nvidia-smi")
    print("If HTTP 500, check Ollama logs: journalctl -u ollama -n 50")
    print("If timeout, increase REQUEST_TIMEOUT or reduce num_ctx/num_predict.")


if __name__ == "__main__":
    main()
