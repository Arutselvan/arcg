# ARCG: Stable Reasoning, Fragile Answers

**Stable Reasoning, Fragile Answers: How LLMs Fail at the Final Execution Step Under Semantic Variation**

*Arut Selvan Dhanasekaran — Independent Researcher — arutselvan710@gmail.com*

Preprint submitted to COLM 2026.

---

## Overview

This repository contains the complete experimental pipeline, data, figures, and LaTeX source for the paper introducing the **Answer-Reasoning Consistency Gap (ARCG)** metric.

ARCG measures the divergence between how consistently a language model produces the same final answer versus how consistently it produces the same reasoning chain, when the same problem is presented in semantically equivalent paraphrases:

```
ARCG = FAC - RSC
```

where FAC is Final Answer Consistency and RSC is Reasoning Step Consistency. A negative ARCG indicates that reasoning chains are more stable than final answers, a phenomenon we term *execution fragility*.

The experiment runs 10 open-source reasoning models on 75 real problems from GSM8K and ARC-Challenge, each presented in 5 semantically equivalent paraphrases. Paraphrase quality is verified by two independent human annotators and two LLM judges, with inter-annotator agreement measured via Cohen's Kappa.

---

## Repository Structure

```
arcg/
├── README.md
├── code/
│   ├── 1_build_and_paraphrase.py         Build benchmark + generate paraphrases
│   ├── 2_generate_validation_template.py  Generate human annotation Excel templates
│   ├── 3_llm_judge.py                    LLM-as-judge paraphrase evaluation
│   ├── 4_consolidate_validation.py       Merge human + LLM reports, compute Kappa
│   ├── 5_run_experiment.py               10-model evaluation runner with checkpointing
│   └── 6_analyze_and_plot.py             Metrics, statistical tests, vector PDF figures
├── data/
│   ├── paraphrases.json                  Output of Step 1
│   ├── human_validation_annotator1.xlsx  Fill and return (Step 2)
│   ├── human_validation_annotator2.xlsx  Fill and return (Step 2)
│   ├── llm_judge_deepseek-r1-70b.json    Output of Step 3
│   ├── llm_judge_qwen3-32b.json          Output of Step 3
│   ├── validated_paraphrases.json        Output of Step 4
│   ├── experiment_results.json           Output of Step 5
│   └── metrics.json                      Output of Step 6
├── figures/
│   ├── fig1_fac_rsc_comparison.pdf
│   ├── fig2_arcg_by_domain.pdf
│   ├── fig3_arcg_by_difficulty.pdf
│   ├── fig4_fac_accuracy_scatter.pdf
│   ├── fig5_arcg_violin.pdf
│   ├── fig6_model_scaling.pdf
│   ├── fig7_rsc_heatmap.pdf
│   └── tables.tex
└── paper/
    ├── main.tex
    ├── main.pdf
    ├── references.bib
    └── colm2026_conference.sty
```

---

## Hardware Requirements

All scripts are designed to run on a single NVIDIA H100 80GB GPU. Models are run one at a time (not simultaneously). The largest single model is `deepseek-r1:70b` at approximately 40 GB at Q4 quantization.

For smaller GPUs (16 GB), you can run the 7B and 8B models only by passing `--models deepseek-r1:7b deepseek-r1:8b qwen3:8b phi4-mini-reasoning:3.8b` to script 5.

---

## Software Requirements

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Install Python dependencies
pip install datasets openpyxl scipy sentence-transformers matplotlib numpy requests tqdm
```

All scripts start the Ollama server automatically if it is not running, and pull any required models if they are not already downloaded.

---

## Running the Experiment

Run the six scripts in order. Each script saves its output to the `data/` directory and can be safely resumed if interrupted.

---

### Script 1: Build Benchmark and Generate Paraphrases

```bash
python code/1_build_and_paraphrase.py
```

This script loads 75 real problems from HuggingFace:

- 40 problems from GSM8K (grade school math), stratified as 15 easy, 15 medium, 10 hard
- 35 problems from ARC-Challenge (science logic), stratified as 15 easy, 12 medium, 8 hard

For each problem, `deepseek-r1:70b` generates 5 paraphrases using distinct linguistic strategies:

| ID | Strategy | Description |
|----|----------|-------------|
| P1 | Formal restatement | Precise mathematical or logical language |
| P2 | Informal restatement | Casual everyday language |
| P3 | Passive restructuring | Passive voice, different sentence order |
| P4 | Decomposed | Problem split into numbered sub-steps |
| P5 | Analogical | Same logical structure, different real-world context |

The model is prompted to reason through semantic equivalence before generating each paraphrase, and to confirm the ground-truth answer is preserved.

**Output:** `data/paraphrases.json`

**Model pulled automatically:** `deepseek-r1:70b` (~40 GB)

**Estimated runtime:** 2-3 hours on H100

---

### Script 2: Generate Human Annotation Templates

```bash
python code/2_generate_validation_template.py
```

Creates two Excel workbooks for independent human annotation. Each workbook contains one row per paraphrase with the original problem, the paraphrase text, the paraphrase strategy label, and a YES/NO column for the annotator to fill in.

Annotators should work independently without communicating. A paraphrase should be marked YES if and only if it asks the same question as the original and preserves the correct answer.

**Output:** `data/human_validation_annotator1.xlsx`, `data/human_validation_annotator2.xlsx`

**Time required per annotator:** approximately 90 minutes for 375 paraphrases

---

### Script 3: LLM-as-Judge Evaluation

```bash
# Run both judges (recommended)
python code/3_llm_judge.py

# Run a single judge
python code/3_llm_judge.py --judge deepseek-r1:70b
python code/3_llm_judge.py --judge qwen3:32b
```

Two reasoning models independently evaluate each paraphrase. Each judge is given the original problem, the paraphrase, and the ground-truth answer, and asked to reason through whether the paraphrase is semantically equivalent. The judge outputs a binary VALID/INVALID verdict and a confidence score from 1 to 5.

**Output:** `data/llm_judge_deepseek-r1-70b.json`, `data/llm_judge_qwen3-32b.json`

**Models pulled automatically:** `deepseek-r1:70b`, `qwen3:32b`

**Estimated runtime:** 1-2 hours per judge on H100

---

### Script 4: Consolidate Validation Reports

```bash
python code/4_consolidate_validation.py
```

Merges all four validation sources (two human annotators, two LLM judges) and computes:

- Pairwise Cohen's Kappa for all 6 annotator pairs (human-human, LLM-LLM, human-LLM)
- Overall Fleiss' Kappa across all four sources
- Per-paraphrase majority vote (accept if at least 3 of 4 sources mark VALID)
- Per-strategy and per-domain acceptance rates

A paraphrase is included in the final validated set only if it receives at least 3 VALID votes out of 4.

**Input:** Filled Excel files from Step 2 and JSON files from Step 3

**Output:**
- `data/validated_paraphrases.json` — accepted paraphrases only
- `data/validation_report.json` — full statistics
- `data/validation_summary.txt` — human-readable summary for the paper

**Note:** If the human annotation files have not been filled yet, the script runs with only the two LLM judges and prints a clear warning. You can re-run it after human annotation is complete to produce the final validated set.

---

### Script 5: Run the Evaluation

```bash
# Run all 10 models (recommended)
python code/5_run_experiment.py

# Run a subset of models
python code/5_run_experiment.py --models deepseek-r1:7b deepseek-r1:8b qwen3:8b

# Skip model pulling if already downloaded
python code/5_run_experiment.py --skip-pull
```

Runs all 10 reasoning models on every validated (problem, paraphrase) pair. For each response, the script:

1. Sends the problem with a structured CoT prompt
2. Extracts the reasoning chain (everything before the final answer token)
3. Extracts the final answer using a per-model extraction strategy
4. Verifies the answer against the ground truth
5. Saves the result immediately to checkpoint

The answer extraction logic is tailored to each model's output format. All models are prompted with the same system message requesting a reasoning chain followed by a boxed final answer.

**Output:** `data/experiment_results.json`

**Models evaluated:**

| Model | Family | VRAM (Q4) |
|-------|--------|-----------|
| `deepseek-r1:7b` | DeepSeek R1 | ~5 GB |
| `deepseek-r1:8b` | DeepSeek R1 (Llama base) | ~5 GB |
| `deepseek-r1:14b` | DeepSeek R1 | ~9 GB |
| `deepseek-r1:32b` | DeepSeek R1 | ~20 GB |
| `deepseek-r1:70b` | DeepSeek R1 | ~40 GB |
| `qwen3:8b` | Qwen 3 | ~5 GB |
| `qwen3:32b` | Qwen 3 | ~20 GB |
| `magistral:24b` | Mistral | ~15 GB |
| `phi4-reasoning:14b` | Phi-4 | ~9 GB |
| `glm-4.7-flash` | GLM | ~19 GB |

**Estimated runtime:** 4-8 hours for all 10 models on H100

---

### Script 6: Analyze and Generate Figures

```bash
python code/6_analyze_and_plot.py

# Skip sentence embedding computation (RSC will be NaN)
python code/6_analyze_and_plot.py --no-embed
```

Computes all ARCG metrics and generates 7 publication-quality vector PDF figures and a LaTeX tables file.

**Metrics computed:**

| Metric | Definition |
|--------|-----------|
| FAC | Fraction of paraphrases that yield the same extracted answer as the majority vote |
| RSC | Mean pairwise cosine similarity of CoT reasoning chains via `all-MiniLM-L6-v2` |
| ARCG | FAC minus RSC |
| ARC | Fraction of responses where the reasoning chain logically supports the extracted answer |

**Statistical tests run:**

- Paired t-test: FAC vs RSC per model (tests whether ARCG is significantly different from zero)
- Kruskal-Wallis test: ARCG across models (tests whether models differ significantly)
- Pearson correlation: FAC vs accuracy, RSC vs accuracy
- Mann-Whitney U: Math vs Logic domain ARCG comparison
- Spearman correlation: Model size (parameters) vs ARCG

**Output:**
- `data/metrics.json`
- `figures/fig1_fac_rsc_comparison.pdf` through `figures/fig7_rsc_heatmap.pdf`
- `figures/tables.tex`

---

## Compiling the Paper

```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## LLM Usage Disclosure

In accordance with COLM 2026 policy, the authors disclose that a large language model was used to assist in generating the Python code for the experimental pipeline, producing the data visualization plots, and drafting and refining the prose of the paper. The core research question, experimental design, metric formulation (ARCG), model selection, and scientific interpretation of results were conducted independently by the human authors.

---

## Citation

```bibtex
@article{dhanasekaran2026arcg,
  title   = {Stable Reasoning, Fragile Answers: How {LLMs} Fail at the
             Final Execution Step Under Semantic Variation},
  author  = {Dhanasekaran, Arut Selvan},
  journal = {arXiv preprint},
  year    = {2026}
}
```
