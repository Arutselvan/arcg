# Stable Reasoning, Fragile Answers

**Stable Reasoning, Fragile Answers: How LLMs Fail at the Final Execution Step Under Semantic Variation**

*Arut Selvan Dhanasekaran — Independent Researcher — arutselvan710@gmail.com*

Preprint submitted to COLM 2026.

---

## Overview

This repository contains all code, data, figures, and the LaTeX source for the paper introducing the **Answer-Reasoning Consistency Gap (ARCG)** metric. The paper evaluates five open-source LLMs on a curated benchmark of mathematical and logical reasoning tasks, using systematic paraphrasing to probe the decoupling between reasoning stability and answer stability.

**Key Finding:** ARCG is universally negative across all five tested models. Reasoning chains are significantly more stable across paraphrases than final answers, revealing an "execution gap" rather than the post-hoc rationalization predicted by prior work.

---

## Repository Structure

```
arcg-paper/
├── README.md
├── code/
│   ├── run_ollama_experiment.py     # Full experiment pipeline (run on your hardware)
│   └── generate_figures_pdf.py     # Regenerates all vector PDF figures from results
├── data/
│   └── arcg_ollama_results.json    # Raw experiment results (900 CoT responses)
├── figures/
│   ├── fig1_fac_rsc.pdf            # FAC vs RSC comparison across models
│   ├── fig2_arcg_domain.pdf        # ARCG by domain (Math vs Logic)
│   ├── fig3_arcg_difficulty.pdf    # ARCG by difficulty level
│   ├── fig4_fac_accuracy.pdf       # FAC/RSC vs accuracy scatter
│   └── fig5_arcg_violin.pdf        # ARCG distribution violin plot
└── paper/
    ├── main.tex                    # LaTeX source (COLM 2026 preprint)
    ├── main.pdf                    # Compiled paper
    ├── references.bib              # BibTeX bibliography (35 entries)
    ├── colm2026_conference.sty     # Official COLM 2026 style file
    ├── colm2026_conference.bst     # Official COLM 2026 bibliography style
    ├── fancyhdr.sty
    └── natbib.sty
```

---

## Reproducing the Experiment

### Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- RTX 4080 (16GB VRAM) or equivalent

### Step 1: Install Python dependencies

```bash
pip install requests sentence-transformers numpy scipy tqdm
```

### Step 2: Pull the five models

```bash
ollama pull llama3.1:8b
ollama pull qwen2.5:14b
ollama pull mistral:7b
ollama pull gemma2:9b
ollama pull phi4:14b
```

### Step 3: Run the experiment

```bash
python code/run_ollama_experiment.py
```

The script saves a checkpoint after each problem and resumes automatically if interrupted. Output is saved to `data/arcg_ollama_results.json`.

### Step 4: Regenerate figures

```bash
python code/generate_figures_pdf.py
```

---

## Compiling the Paper

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## LLM Usage Disclosure

In accordance with COLM 2026 policy, the author discloses that an LLM was used to assist in generating the Python code for the experimental pipeline, producing the data visualization plots, and drafting and refining the prose of the paper. The core research ideation, experimental design, and metric formulation (ARCG) were conducted independently by the human author.

---

## Citation

If you use this work, please cite:

```bibtex
@article{dhanasekaran2026arcg,
  title={Stable Reasoning, Fragile Answers: How {LLMs} Fail at the Final Execution Step Under Semantic Variation},
  author={Dhanasekaran, Arut Selvan},
  journal={arXiv preprint},
  year={2026}
}
```
