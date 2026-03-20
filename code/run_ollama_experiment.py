#!/usr/bin/env python3
"""
ARCG Experiment Runner — Ollama Edition
========================================
Runs the Answer-Reasoning Consistency Gap (ARCG) experiment
across 5 local Ollama models on an RTX 4080 (16GB VRAM).

SETUP (run once):
    pip install requests sentence-transformers numpy scipy tqdm

MODELS (pull once before running):
    ollama pull llama3.1:8b
    ollama pull qwen2.5:14b
    ollama pull mistral:7b
    ollama pull gemma2:9b
    ollama pull phi4:14b

RUN:
    python run_ollama_experiment.py

OUTPUT:
    arcg_ollama_results.json  — hand this file back for analysis and publication
"""

import json
import re
import time
import hashlib
import os
import sys
from datetime import datetime
from itertools import combinations

import requests
import numpy as np
from scipy import stats
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
OUTPUT_FILE = "arcg_ollama_results.json"
CHECKPOINT_FILE = "arcg_ollama_checkpoint.json"

# 5 models that fit entirely in 16GB VRAM on an RTX 4080
# VRAM usage (approx): llama3.1:8b ~5GB, qwen2.5:14b ~9GB, mistral:7b ~5GB,
#                      gemma2:9b ~6GB, phi4:14b ~9GB
MODELS = [
    "llama3.1:8b",    # Meta Llama 3.1 8B — strong general reasoning baseline
    "qwen2.5:14b",    # Alibaba Qwen 2.5 14B — top-tier reasoning at 14B scale
    "mistral:7b",     # Mistral 7B v0.3 — efficient, widely studied
    "gemma2:9b",      # Google Gemma 2 9B — strong instruction following
    "phi4:14b",       # Microsoft Phi-4 14B — strong math/logic reasoning
]

# Number of paraphrase variants per problem (P0 = original + P1..P5 = paraphrases)
N_PARAPHRASES = 5   # P0 (original) + 5 paraphrases = 6 total per problem

# Timeout per Ollama request (seconds)
REQUEST_TIMEOUT = 120

# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK — 30 problems (15 math + 15 logic), stratified by difficulty
# These are fixed, human-curated problems with known ground-truth answers.
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARK = [
    # ── MATH: EASY ────────────────────────────────────────────────────────────
    {
        "id": "M_E1", "domain": "math", "difficulty": "easy",
        "question": "A baker made 48 cookies. She sold 30 of them and gave 6 to her neighbor. How many cookies does she have left?",
        "answer": "12",
    },
    {
        "id": "M_E2", "domain": "math", "difficulty": "easy",
        "question": "Tom has $45. He buys a book for $12 and a pen for $3. How much money does Tom have left?",
        "answer": "30",
    },
    {
        "id": "M_E3", "domain": "math", "difficulty": "easy",
        "question": "A train travels at 60 miles per hour. How far does it travel in 2.5 hours?",
        "answer": "150",
    },
    {
        "id": "M_E4", "domain": "math", "difficulty": "easy",
        "question": "There are 5 shelves in a library. Each shelf holds 24 books. How many books can the library hold in total?",
        "answer": "120",
    },
    {
        "id": "M_E5", "domain": "math", "difficulty": "easy",
        "question": "Maria earns $15 per hour. She worked 8 hours on Monday and 6 hours on Tuesday. How much did she earn in total?",
        "answer": "210",
    },
    # ── MATH: MEDIUM ──────────────────────────────────────────────────────────
    {
        "id": "M_M1", "domain": "math", "difficulty": "medium",
        "question": "A store sells apples for $0.50 each and oranges for $0.75 each. If John buys 8 apples and 6 oranges, and pays with a $10 bill, how much change does he receive?",
        "answer": "1.50",
    },
    {
        "id": "M_M2", "domain": "math", "difficulty": "medium",
        "question": "A rectangular garden is 12 meters long and 8 meters wide. A path 1 meter wide runs along the inside of the entire perimeter. What is the area of the path?",
        "answer": "76",
    },
    {
        "id": "M_M3", "domain": "math", "difficulty": "medium",
        "question": "A car depreciates by 15% of its value each year. If a car is worth $20,000 today, what will it be worth after 2 years? Round to the nearest dollar.",
        "answer": "14450",
    },
    {
        "id": "M_M4", "domain": "math", "difficulty": "medium",
        "question": "Three friends split a restaurant bill equally. The food cost $72 and they left a 20% tip. How much did each person pay?",
        "answer": "28.80",
    },
    {
        "id": "M_M5", "domain": "math", "difficulty": "medium",
        "question": "A tank is 40% full. After adding 30 liters, it becomes 70% full. What is the total capacity of the tank in liters?",
        "answer": "100",
    },
    # ── MATH: HARD ────────────────────────────────────────────────────────────
    {
        "id": "M_H1", "domain": "math", "difficulty": "hard",
        "question": "A factory produces widgets at a rate of 120 per hour for the first 4 hours, then 90 per hour for the next 3 hours. If each widget sells for $2.50 and the factory's operating cost is $800 per day, what is the net profit for this 7-hour period?",
        "answer": "1075",
    },
    {
        "id": "M_H2", "domain": "math", "difficulty": "hard",
        "question": "A train leaves City A at 8:00 AM traveling at 80 km/h. Another train leaves City B (400 km away) at 9:00 AM traveling toward City A at 100 km/h. At what time do they meet?",
        "answer": "11:20 AM",
    },
    {
        "id": "M_H3", "domain": "math", "difficulty": "hard",
        "question": "A store offers a 25% discount on all items. A customer also has a coupon for an additional 10% off the discounted price. If the original price of an item is $80, how much does the customer pay?",
        "answer": "54",
    },
    {
        "id": "M_H4", "domain": "math", "difficulty": "hard",
        "question": "A swimming pool can be filled by pipe A in 6 hours and by pipe B in 4 hours. Pipe C can drain the full pool in 12 hours. If all three pipes are open simultaneously, how many hours will it take to fill the pool?",
        "answer": "4",
    },
    {
        "id": "M_H5", "domain": "math", "difficulty": "hard",
        "question": "A company's revenue grew by 20% in year 1, declined by 10% in year 2, and grew by 15% in year 3. If the initial revenue was $500,000, what was the revenue at the end of year 3?",
        "answer": "621000",
    },
    # ── LOGIC: EASY ───────────────────────────────────────────────────────────
    {
        "id": "L_E1", "domain": "logic", "difficulty": "easy",
        "question": "All mammals are warm-blooded. Dolphins are mammals. Are dolphins warm-blooded? Answer Yes or No.",
        "answer": "Yes",
    },
    {
        "id": "L_E2", "domain": "logic", "difficulty": "easy",
        "question": "If it rains, the ground gets wet. The ground is not wet. Did it rain? Answer Yes or No.",
        "answer": "No",
    },
    {
        "id": "L_E3", "domain": "logic", "difficulty": "easy",
        "question": "Alice is taller than Bob. Bob is taller than Carol. Is Alice taller than Carol? Answer Yes or No.",
        "answer": "Yes",
    },
    {
        "id": "L_E4", "domain": "logic", "difficulty": "easy",
        "question": "A number is even if it is divisible by 2. The number 14 is divisible by 2. Is 14 even? Answer Yes or No.",
        "answer": "Yes",
    },
    {
        "id": "L_E5", "domain": "logic", "difficulty": "easy",
        "question": "All squares are rectangles. All rectangles have four right angles. Do all squares have four right angles? Answer Yes or No.",
        "answer": "Yes",
    },
    # ── LOGIC: MEDIUM ─────────────────────────────────────────────────────────
    {
        "id": "L_M1", "domain": "logic", "difficulty": "medium",
        "question": "In a race, Amy finished before Ben. Ben finished before Carol. Dan finished after Carol. Who finished last? Answer with just the name.",
        "answer": "Dan",
    },
    {
        "id": "L_M2", "domain": "logic", "difficulty": "medium",
        "question": "Some birds can fly. Penguins are birds. Can we conclude that penguins can fly? Answer Yes or No and explain why.",
        "answer": "No",
    },
    {
        "id": "L_M3", "domain": "logic", "difficulty": "medium",
        "question": "If all A are B, and all B are C, and X is not C, can X be A? Answer Yes or No.",
        "answer": "No",
    },
    {
        "id": "L_M4", "domain": "logic", "difficulty": "medium",
        "question": "Three boxes are labeled 'Apples', 'Oranges', and 'Mixed'. All labels are wrong. You pick one fruit from the 'Mixed' box and it is an apple. What is in the 'Oranges' box? Answer with just the contents.",
        "answer": "Mixed",
    },
    {
        "id": "L_M5", "domain": "logic", "difficulty": "medium",
        "question": "A is the father of B. B is the sister of C. C is the mother of D. What is A's relationship to D? Answer with the relationship.",
        "answer": "Grandfather",
    },
    # ── LOGIC: HARD ───────────────────────────────────────────────────────────
    {
        "id": "L_H1", "domain": "logic", "difficulty": "hard",
        "question": "Five people (A, B, C, D, E) sit in a row. A is not next to B. C is between D and E. B is at one end. D is not at either end. What position (1-5) is C in? Answer with just the number.",
        "answer": "3",
    },
    {
        "id": "L_H2", "domain": "logic", "difficulty": "hard",
        "question": "A knight always tells the truth and a knave always lies. Person X says 'I am a knave.' Is this statement possible? Answer Yes or No and explain.",
        "answer": "No",
    },
    {
        "id": "L_H3", "domain": "logic", "difficulty": "hard",
        "question": "If P implies Q, and Q implies R, and R is false, what can we conclude about P? Answer: P is true, P is false, or P could be either.",
        "answer": "P is false",
    },
    {
        "id": "L_H4", "domain": "logic", "difficulty": "hard",
        "question": "100 prisoners are each assigned a unique number 1-100. Each prisoner must guess their own number. They can discuss a strategy beforehand. What is the maximum guaranteed fraction of prisoners that can correctly guess their number using the optimal strategy?",
        "answer": "approximately 0.31",
    },
    {
        "id": "L_H5", "domain": "logic", "difficulty": "hard",
        "question": "You have 12 balls, one of which is either heavier or lighter than the others. Using a balance scale exactly 3 times, can you always identify the odd ball AND determine if it is heavier or lighter? Answer Yes or No.",
        "answer": "Yes",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# PARAPHRASE STRATEGIES (5 systematic transformations)
# ─────────────────────────────────────────────────────────────────────────────

PARAPHRASE_STRATEGIES = [
    {
        "id": "P1",
        "name": "formal",
        "instruction": "Restate the following problem in formal, academic language. Preserve all numerical values and logical relationships exactly. Do not change the problem. Output only the reworded problem, nothing else."
    },
    {
        "id": "P2",
        "name": "informal",
        "instruction": "Restate the following problem in casual, conversational language as if explaining to a friend. Preserve all numerical values and logical relationships exactly. Do not change the problem. Output only the reworded problem, nothing else."
    },
    {
        "id": "P3",
        "name": "restructured",
        "instruction": "Restate the following problem by changing the sentence structure and word order (e.g., use passive voice, reverse the order of given information). Preserve all numerical values and logical relationships exactly. Do not change the problem. Output only the reworded problem, nothing else."
    },
    {
        "id": "P4",
        "name": "decomposed",
        "instruction": "Restate the following problem by breaking it into explicit numbered sub-steps or sub-questions that together form the same problem. Preserve all numerical values and logical relationships exactly. Do not change the problem. Output only the reworded problem, nothing else."
    },
    {
        "id": "P5",
        "name": "analogical",
        "instruction": "Restate the following problem by substituting the real-world entities with different but analogous ones (e.g., replace 'baker' with 'farmer', 'cookies' with 'apples', etc.) while keeping the mathematical or logical structure identical. Preserve all numerical values. Output only the reworded problem, nothing else."
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def check_ollama():
    """Verify Ollama is running and list available models."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        r.raise_for_status()
        available = [m["name"] for m in r.json().get("models", [])]
        return available
    except Exception as e:
        print(f"\n[ERROR] Cannot connect to Ollama at {OLLAMA_BASE_URL}")
        print(f"  Make sure Ollama is running: `ollama serve`")
        print(f"  Error: {e}")
        sys.exit(1)


def ollama_generate(model: str, prompt: str, temperature: float = 0.0) -> str:
    """Call Ollama /api/generate and return the response text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 1024,
            "top_p": 1.0,
        }
    }
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=REQUEST_TIMEOUT
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.exceptions.Timeout:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[ERROR: {e}]"


def extract_answer(response: str, problem: dict) -> str:
    """
    Extract the final answer from a CoT response.
    Looks for common answer patterns: 'answer is X', '= X', boxed answers, etc.
    """
    text = response.strip()

    # Pattern 1: "The answer is X" / "Answer: X"
    m = re.search(r'(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+([^\n.]+)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip('.')

    # Pattern 2: "= X" at end of line
    m = re.search(r'=\s*\$?([0-9,\.]+)\s*$', text, re.MULTILINE)
    if m:
        return m.group(1).replace(',', '').strip()

    # Pattern 3: "Therefore, X" / "So, X"
    m = re.search(r'(?:therefore|thus|so|hence)[,:\s]+([^\n.]+)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip('.')

    # Pattern 4: Last number in text (for numeric problems)
    if problem["domain"] == "math":
        nums = re.findall(r'\$?([0-9,]+(?:\.[0-9]+)?)', text)
        if nums:
            return nums[-1].replace(',', '')

    # Pattern 5: Yes/No extraction for logic
    if problem["domain"] == "logic":
        m = re.search(r'\b(Yes|No)\b', text, re.IGNORECASE)
        if m:
            return m.group(1).capitalize()

    # Fallback: last non-empty line
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return lines[-1] if lines else text[:100]


def answers_match(extracted: str, ground_truth: str) -> bool:
    """Fuzzy match between extracted answer and ground truth."""
    def normalize(s):
        s = s.lower().strip().rstrip('.')
        s = re.sub(r'[\$,]', '', s)
        s = re.sub(r'\s+', ' ', s)
        return s

    ext = normalize(extracted)
    gt = normalize(ground_truth)

    if ext == gt:
        return True

    # Try numeric comparison
    try:
        return abs(float(ext) - float(gt)) < 0.01
    except ValueError:
        pass

    # Partial match for long answers
    if len(gt) > 5 and gt in ext:
        return True

    return False


def compute_pairwise_fac(answers: list) -> float:
    """Compute Final Answer Consistency (FAC) as mean pairwise agreement."""
    if len(answers) < 2:
        return 1.0
    pairs = list(combinations(range(len(answers)), 2))
    matches = sum(1 for i, j in pairs if answers_match(answers[i], answers[j]))
    return matches / len(pairs)


def compute_pairwise_rsc(chains: list, embedder) -> float:
    """Compute Reasoning Step Consistency (RSC) as mean pairwise cosine similarity."""
    if len(chains) < 2:
        return 1.0
    embeddings = embedder.encode(chains, convert_to_numpy=True, show_progress_bar=False)
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings = embeddings / norms
    pairs = list(combinations(range(len(chains)), 2))
    sims = [float(np.dot(embeddings[i], embeddings[j])) for i, j in pairs]
    return float(np.mean(sims))


def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint: dict):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: GENERATE PARAPHRASES
# Uses the first available model (llama3.1:8b) to generate paraphrases.
# ─────────────────────────────────────────────────────────────────────────────

def generate_paraphrases(paraphrase_model: str, checkpoint: dict) -> dict:
    """Generate 5 paraphrases for each benchmark problem."""
    print(f"\n{'='*60}")
    print(f"PHASE 1: Generating paraphrases using {paraphrase_model}")
    print(f"{'='*60}")

    paraphrases = checkpoint.get("paraphrases", {})
    problems_to_do = [p for p in BENCHMARK if p["id"] not in paraphrases]

    if not problems_to_do:
        print("  All paraphrases already generated (loaded from checkpoint).")
        return paraphrases

    for problem in tqdm(problems_to_do, desc="Generating paraphrases"):
        pid = problem["id"]
        paraphrases[pid] = {"P0": problem["question"]}

        for strategy in PARAPHRASE_STRATEGIES:
            prompt = (
                f"{strategy['instruction']}\n\n"
                f"Problem: {problem['question']}"
            )
            response = ollama_generate(paraphrase_model, prompt, temperature=0.3)
            # Clean up: remove any preamble like "Here is the reworded problem:"
            response = re.sub(r'^[^:]+:\s*', '', response, count=1).strip()
            paraphrases[pid][strategy["id"]] = response if response and "[ERROR" not in response else problem["question"]

        checkpoint["paraphrases"] = paraphrases
        save_checkpoint(checkpoint)

    print(f"  Generated paraphrases for {len(paraphrases)} problems.")
    return paraphrases


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: RUN INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

COT_SYSTEM_PROMPT = """You are a precise reasoning assistant. For each problem:
1. Think step by step, showing your work clearly.
2. At the end, state your final answer on a new line starting with "The answer is: "
Be concise but thorough."""

def run_inference(models: list, paraphrases: dict, checkpoint: dict) -> list:
    """Run CoT inference for all models × problems × paraphrases."""
    print(f"\n{'='*60}")
    print(f"PHASE 2: Running CoT inference")
    print(f"  Models: {models}")
    print(f"  Problems: {len(BENCHMARK)}")
    print(f"  Paraphrases per problem: {N_PARAPHRASES + 1} (P0–P5)")
    total = len(models) * len(BENCHMARK) * (N_PARAPHRASES + 1)
    print(f"  Total API calls: {total}")
    print(f"{'='*60}")

    results = checkpoint.get("inference_results", [])
    done_keys = {(r["model"], r["problem_id"], r["paraphrase_id"]) for r in results}

    para_ids = ["P0"] + [s["id"] for s in PARAPHRASE_STRATEGIES]

    with tqdm(total=total, desc="Inference") as pbar:
        pbar.update(len(done_keys))
        for model in models:
            for problem in BENCHMARK:
                pid = problem["id"]
                problem_paras = paraphrases.get(pid, {"P0": problem["question"]})
                for para_id in para_ids:
                    key = (model, pid, para_id)
                    if key in done_keys:
                        continue

                    question = problem_paras.get(para_id, problem["question"])
                    prompt = f"{COT_SYSTEM_PROMPT}\n\nProblem: {question}"

                    start = time.time()
                    response = ollama_generate(model, prompt, temperature=0.0)
                    elapsed = time.time() - start

                    extracted = extract_answer(response, problem)
                    correct = answers_match(extracted, problem["answer"])

                    results.append({
                        "model": model,
                        "problem_id": pid,
                        "domain": problem["domain"],
                        "difficulty": problem["difficulty"],
                        "paraphrase_id": para_id,
                        "question": question,
                        "response": response,
                        "reasoning_chain": response,
                        "extracted_answer": extracted,
                        "ground_truth": problem["answer"],
                        "is_correct": correct,
                        "latency_sec": round(elapsed, 2),
                    })

                    checkpoint["inference_results"] = results
                    save_checkpoint(checkpoint)
                    done_keys.add(key)
                    pbar.update(1)

    print(f"  Collected {len(results)} responses.")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: COMPUTE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(results: list) -> list:
    """Compute FAC, RSC, ARCG, and ARC for each (problem, model) pair."""
    print(f"\n{'='*60}")
    print(f"PHASE 3: Computing ARCG metrics")
    print(f"{'='*60}")

    # Lazy-load sentence embedder
    try:
        from sentence_transformers import SentenceTransformer
        print("  Loading sentence embedding model (all-MiniLM-L6-v2)...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        print("  [WARNING] sentence-transformers not installed. RSC will be set to 0.")
        embedder = None

    # Group results by (problem_id, model)
    groups = {}
    for r in results:
        key = (r["problem_id"], r["model"])
        groups.setdefault(key, []).append(r)

    metrics = []
    for (pid, model), group in tqdm(groups.items(), desc="Computing metrics"):
        answers = [r["extracted_answer"] for r in group]
        chains = [r["reasoning_chain"] for r in group]
        correctness = [r["is_correct"] for r in group]

        fac = compute_pairwise_fac(answers)
        rsc = compute_pairwise_rsc(chains, embedder) if embedder else 0.0
        arcg = fac - rsc
        accuracy = float(np.mean(correctness))

        # ARC: fraction of responses where reasoning is coherent with answer
        # Approximated as: if the extracted answer appears in the response text
        arc_scores = []
        for r in group:
            ans = r["extracted_answer"].lower().strip()
            resp = r["response"].lower()
            # Coherent if the answer string appears in the response
            coherent = ans in resp if len(ans) > 0 else False
            arc_scores.append(float(coherent))
        arc = float(np.mean(arc_scores))

        problem_meta = next(p for p in BENCHMARK if p["id"] == pid)
        metrics.append({
            "problem_id": pid,
            "model": model,
            "domain": problem_meta["domain"],
            "difficulty": problem_meta["difficulty"],
            "FAC": round(fac, 4),
            "RSC": round(rsc, 4),
            "ARCG": round(arcg, 4),
            "ARC": round(arc, 4),
            "accuracy": round(accuracy, 4),
            "n_responses": len(group),
        })

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4: COMPUTE SUMMARY STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_summary(metrics: list, models: list) -> dict:
    """Compute per-model and per-domain summary statistics."""
    print(f"\n{'='*60}")
    print(f"PHASE 4: Computing summary statistics")
    print(f"{'='*60}")

    summary = {}

    for model in models:
        model_metrics = [m for m in metrics if m["model"] == model]
        if not model_metrics:
            continue

        fac_vals = np.array([m["FAC"] for m in model_metrics])
        rsc_vals = np.array([m["RSC"] for m in model_metrics])
        arcg_vals = np.array([m["ARCG"] for m in model_metrics])
        arc_vals = np.array([m["ARC"] for m in model_metrics])
        acc_vals = np.array([m["accuracy"] for m in model_metrics])

        # Statistical tests
        t_arcg, p_arcg = stats.ttest_1samp(arcg_vals, 0)
        t_fac_rsc, p_fac_rsc = stats.ttest_rel(fac_vals, rsc_vals)
        r_arcg_acc, p_r = stats.pearsonr(arcg_vals, acc_vals)

        model_summary = {
            "n_problems": len(model_metrics),
            "overall": {
                "FAC_mean": round(float(fac_vals.mean()), 4),
                "FAC_std": round(float(fac_vals.std()), 4),
                "RSC_mean": round(float(rsc_vals.mean()), 4),
                "RSC_std": round(float(rsc_vals.std()), 4),
                "ARCG_mean": round(float(arcg_vals.mean()), 4),
                "ARCG_std": round(float(arcg_vals.std()), 4),
                "ARC_mean": round(float(arc_vals.mean()), 4),
                "ARC_std": round(float(arc_vals.std()), 4),
                "accuracy_mean": round(float(acc_vals.mean()), 4),
                "accuracy_std": round(float(acc_vals.std()), 4),
            },
            "statistical_tests": {
                "ttest_ARCG_vs_zero": {"t": round(float(t_arcg), 4), "p": round(float(p_arcg), 4)},
                "paired_ttest_FAC_vs_RSC": {"t": round(float(t_fac_rsc), 4), "p": round(float(p_fac_rsc), 4)},
                "pearson_ARCG_accuracy": {"r": round(float(r_arcg_acc), 4), "p": round(float(p_r), 4)},
            },
            "by_domain": {},
            "by_difficulty": {},
        }

        for domain in ["math", "logic"]:
            dm = [m for m in model_metrics if m["domain"] == domain]
            if dm:
                model_summary["by_domain"][domain] = {
                    "FAC_mean": round(float(np.mean([m["FAC"] for m in dm])), 4),
                    "RSC_mean": round(float(np.mean([m["RSC"] for m in dm])), 4),
                    "ARCG_mean": round(float(np.mean([m["ARCG"] for m in dm])), 4),
                    "accuracy_mean": round(float(np.mean([m["accuracy"] for m in dm])), 4),
                }

        for diff in ["easy", "medium", "hard"]:
            dm = [m for m in model_metrics if m["difficulty"] == diff]
            if dm:
                model_summary["by_difficulty"][diff] = {
                    "FAC_mean": round(float(np.mean([m["FAC"] for m in dm])), 4),
                    "RSC_mean": round(float(np.mean([m["RSC"] for m in dm])), 4),
                    "ARCG_mean": round(float(np.mean([m["ARCG"] for m in dm])), 4),
                    "accuracy_mean": round(float(np.mean([m["accuracy"] for m in dm])), 4),
                }

        summary[model] = model_summary

        # Print summary
        print(f"\n  Model: {model}")
        print(f"    Accuracy:  {model_summary['overall']['accuracy_mean']:.3f} ± {model_summary['overall']['accuracy_std']:.3f}")
        print(f"    FAC:       {model_summary['overall']['FAC_mean']:.3f} ± {model_summary['overall']['FAC_std']:.3f}")
        print(f"    RSC:       {model_summary['overall']['RSC_mean']:.3f} ± {model_summary['overall']['RSC_std']:.3f}")
        print(f"    ARCG:      {model_summary['overall']['ARCG_mean']:.3f} ± {model_summary['overall']['ARCG_std']:.3f}  (p={model_summary['statistical_tests']['ttest_ARCG_vs_zero']['p']:.4f})")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  ARCG EXPERIMENT — Ollama Edition")
    print("  Answer-Reasoning Consistency Gap Study")
    print("  Author: Arut Selvan Dhanasekaran")
    print("="*60)

    # 1. Check Ollama is running
    available_models = check_ollama()
    print(f"\n[OK] Ollama is running. Available models: {available_models}")

    # 2. Check required models are pulled
    missing = [m for m in MODELS if m not in available_models]
    if missing:
        print(f"\n[WARNING] The following models are not pulled yet:")
        for m in missing:
            print(f"  ollama pull {m}")
        print("\nPulling missing models now...")
        for m in missing:
            print(f"  Pulling {m}...")
            os.system(f"ollama pull {m}")
        # Re-check
        available_models = check_ollama()
        still_missing = [m for m in MODELS if m not in available_models]
        if still_missing:
            print(f"\n[ERROR] Could not pull: {still_missing}")
            print("  Please pull them manually and re-run the script.")
            sys.exit(1)

    print(f"\n[OK] All {len(MODELS)} models available.")

    # 3. Load checkpoint
    checkpoint = load_checkpoint()
    print(f"[OK] Checkpoint loaded ({len(checkpoint.get('inference_results', []))} results cached).")

    # 4. Generate paraphrases
    paraphrase_model = MODELS[0]  # Use first model for paraphrase generation
    paraphrases = generate_paraphrases(paraphrase_model, checkpoint)

    # 5. Run inference
    results = run_inference(MODELS, paraphrases, checkpoint)

    # 6. Compute metrics
    metrics = compute_metrics(results)

    # 7. Compute summary statistics
    summary = compute_summary(metrics, MODELS)

    # 8. Save full results
    output = {
        "experiment": "ARCG — Answer-Reasoning Consistency Gap",
        "description": "Measures the gap between final answer consistency (FAC) and reasoning chain consistency (RSC) across 5 semantically equivalent paraphrases of 30 reasoning problems.",
        "author": "Arut Selvan Dhanasekaran",
        "timestamp": datetime.now().isoformat(),
        "hardware": "RTX 4080 16GB VRAM",
        "models": MODELS,
        "n_problems": len(BENCHMARK),
        "n_paraphrases_per_problem": N_PARAPHRASES + 1,
        "total_responses": len(results),
        "benchmark": BENCHMARK,
        "paraphrases": paraphrases,
        "inference_results": results,
        "metrics": metrics,
        "summary": summary,
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    # Clean up checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"  Results saved to: {OUTPUT_FILE}")
    print(f"  Total responses: {len(results)}")
    print(f"  File size: {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")
    print(f"{'='*60}")
    print(f"\n  Send '{OUTPUT_FILE}' back for analysis and publication.")
    print(f"  The file contains all raw data needed to generate figures,")
    print(f"  tables, and the complete arXiv paper.\n")


if __name__ == "__main__":
    main()
