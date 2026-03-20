#!/usr/bin/env bash
# =============================================================================
# setup.sh  --  ARCG Experiment Environment Setup
# =============================================================================
# Installs all dependencies required to run the ARCG experiment pipeline:
#   1. Python 3 (checks existing, installs if missing)
#   2. pip (via official get-pip.py curl method)
#   3. All Python packages used across scripts 1-6
#   4. Ollama (via official install script)
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# Tested on Ubuntu 22.04 with NVIDIA H100 80GB.
# =============================================================================

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Colour

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

echo ""
echo "=============================================="
echo "  ARCG Experiment  --  Environment Setup"
echo "=============================================="
echo ""

# ── 1. Check for curl ─────────────────────────────────────────────────────────
info "Checking for curl..."
if ! command -v curl &>/dev/null; then
    info "curl not found. Installing via apt..."
    sudo apt-get update -qq && sudo apt-get install -y curl
fi
success "curl is available."

# ── 2. Check for Python 3 ────────────────────────────────────────────────────
info "Checking for Python 3..."
if command -v python3 &>/dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    success "Found $PYTHON_VERSION"
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    success "Found $PYTHON_VERSION"
    PYTHON=python
else
    info "Python 3 not found. Installing via apt..."
    sudo apt-get update -qq && sudo apt-get install -y python3 python3-distutils
    PYTHON=python3
    success "Python 3 installed."
fi

# ── 3. Install pip via curl (get-pip.py) ─────────────────────────────────────
info "Installing pip via get-pip.py (curl)..."
if $PYTHON -m pip --version &>/dev/null 2>&1; then
    success "pip is already available: $($PYTHON -m pip --version)"
else
    info "Downloading get-pip.py..."
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    $PYTHON /tmp/get-pip.py --user
    rm -f /tmp/get-pip.py
    # Ensure pip is on PATH when installed with --user
    export PATH="$HOME/.local/bin:$PATH"
    success "pip installed: $($PYTHON -m pip --version)"
fi

# Upgrade pip to latest
info "Upgrading pip..."
$PYTHON -m pip install --upgrade pip --quiet
success "pip upgraded."

# ── 4. Install Python packages ───────────────────────────────────────────────
info "Installing Python dependencies..."

PACKAGES=(
    # Data loading
    "datasets"                    # HuggingFace datasets (GSM8K, ARC-Challenge)

    # HTTP / Ollama API
    "requests"                    # Ollama REST API calls

    # Progress bars
    "tqdm"                        # Progress bars in scripts 1, 3, 5

    # Excel annotation templates
    "openpyxl"                    # Read/write .xlsx files (scripts 2, 4)

    # Statistics
    "scipy"                       # Cohen's Kappa, t-tests, Mann-Whitney U (scripts 4, 6)
    "numpy"                       # Numerical operations

    # Sentence embeddings for RSC
    "sentence-transformers"       # all-MiniLM-L6-v2 for reasoning chain similarity (scripts 5, 6)

    # Plotting
    "matplotlib"                  # All 7 vector PDF figures (script 6)
)

for pkg in "${PACKAGES[@]}"; do
    # Strip inline comments
    pkg_name=$(echo "$pkg" | awk '{print $1}')
    info "  Installing $pkg_name..."
    $PYTHON -m pip install "$pkg_name" --quiet
done

success "All Python packages installed."

# ── 5. Verify key imports ─────────────────────────────────────────────────────
info "Verifying imports..."
$PYTHON - <<'EOF'
import sys
failed = []
packages = [
    ("datasets",             "datasets"),
    ("requests",             "requests"),
    ("tqdm",                 "tqdm"),
    ("openpyxl",             "openpyxl"),
    ("scipy",                "scipy"),
    ("numpy",                "numpy"),
    ("sentence_transformers","sentence_transformers"),
    ("matplotlib",           "matplotlib"),
]
for display, mod in packages:
    try:
        __import__(mod)
        print(f"  [OK]  {display}")
    except ImportError as e:
        print(f"  [FAIL] {display}: {e}")
        failed.append(display)
if failed:
    print(f"\nFailed imports: {failed}")
    sys.exit(1)
else:
    print("\nAll imports verified successfully.")
EOF
success "Import verification passed."

# ── 6. Install Ollama ─────────────────────────────────────────────────────────
info "Checking for Ollama..."
if command -v ollama &>/dev/null; then
    OLLAMA_VERSION=$(ollama --version 2>&1 || echo "unknown version")
    success "Ollama already installed: $OLLAMA_VERSION"
else
    info "Installing Ollama via official install script..."
    curl -fsSL https://ollama.com/install.sh | sh
    success "Ollama installed."
fi

# ── 7. Start Ollama server (background) ───────────────────────────────────────
info "Starting Ollama server in the background..."
if curl -s http://localhost:11434/api/tags &>/dev/null; then
    success "Ollama server is already running."
else
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    OLLAMA_PID=$!
    info "Waiting for Ollama server to become ready (PID $OLLAMA_PID)..."
    for i in $(seq 1 30); do
        if curl -s http://localhost:11434/api/tags &>/dev/null; then
            success "Ollama server is ready."
            break
        fi
        sleep 1
        if [ "$i" -eq 30 ]; then
            warn "Ollama server did not respond within 30 seconds."
            warn "Check /tmp/ollama.log for details."
        fi
    done
fi

# ── 8. GPU check ──────────────────────────────────────────────────────────────
echo ""
info "GPU status:"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    warn "nvidia-smi not found. Ensure CUDA drivers are installed for GPU inference."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo -e "  ${GREEN}Setup complete.${NC}"
echo "=============================================="
echo ""
echo "Run the experiment pipeline in order:"
echo ""
echo "  python3 code/1_build_and_paraphrase.py"
echo "  python3 code/2_generate_validation_template.py"
echo "  python3 code/3_llm_judge.py"
echo "  # Fill in data/human_validation_annotator{1,2}.xlsx"
echo "  python3 code/4_consolidate_validation.py"
echo "  python3 code/5_run_experiment.py"
echo "  python3 code/6_analyze_and_plot.py"
echo ""
