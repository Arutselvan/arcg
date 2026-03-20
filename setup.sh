#!/usr/bin/env bash
# =============================================================================
# setup.sh  --  ARCG Experiment Environment Setup
# =============================================================================
# Installs all dependencies required to run the ARCG experiment pipeline:
#   1. Python 3 (checks existing, installs if missing)
#   2. pip (via official get-pip.py curl method)
#   3. All Python packages used across scripts 1-6
#   4. Ollama (via official install script)
#   5. CUDA library symlinks so Ollama can detect the H100 GPU
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# Tested on Ubuntu 22.04/24.04 with NVIDIA H100 80GB.
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
    "datasets"                    # HuggingFace datasets (GSM8K, ARC-Challenge)
    "requests"                    # Ollama REST API calls
    "tqdm"                        # Progress bars in scripts 1, 3, 5
    "openpyxl"                    # Read/write .xlsx files (scripts 2, 4)
    "scipy"                       # Cohen's Kappa, t-tests, Mann-Whitney U (scripts 4, 6)
    "numpy"                       # Numerical operations
    "sentence-transformers"       # all-MiniLM-L6-v2 for reasoning chain similarity (scripts 5, 6)
    "matplotlib"                  # All 7 vector PDF figures (script 6)
)

for pkg in "${PACKAGES[@]}"; do
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

# ── 7. Fix CUDA library path so Ollama can detect the GPU ────────────────────
#
# On many cloud/HPC Ubuntu installs the NVIDIA driver ships libcuda.so under
# /usr/lib/x86_64-linux-gnu/ but Ollama's GPU discovery (gpu.go) looks for
# it via the dynamic linker.  If ldconfig has not indexed that path, Ollama
# falls back to CPU-only mode and every model call returns HTTP 500 because
# there is not enough system RAM to load a 70B model.
#
# Fix: ensure /usr/local/cuda/lib64 and the driver lib path are in
# /etc/ld.so.conf.d/ and run ldconfig, then set LD_LIBRARY_PATH for the
# current session.
#
info "Fixing CUDA library paths for Ollama GPU detection..."

CUDA_CONF="/etc/ld.so.conf.d/cuda-arcg.conf"

# Find the actual cuda lib64 directory
CUDA_LIB=""
for candidate in /usr/local/cuda/lib64 /usr/local/cuda-*/lib64; do
    if [ -d "$candidate" ]; then
        CUDA_LIB="$candidate"
        break
    fi
done

# Find the nvidia driver lib directory
DRIVER_LIB=""
for candidate in /usr/lib/x86_64-linux-gnu /usr/lib64; do
    if ls "$candidate"/libcuda.so* &>/dev/null 2>&1; then
        DRIVER_LIB="$candidate"
        break
    fi
done

if [ -n "$CUDA_LIB" ] || [ -n "$DRIVER_LIB" ]; then
    {
        echo "# Added by ARCG setup.sh for Ollama GPU detection"
        [ -n "$CUDA_LIB" ]   && echo "$CUDA_LIB"
        [ -n "$DRIVER_LIB" ] && echo "$DRIVER_LIB"
    } | sudo tee "$CUDA_CONF" > /dev/null
    sudo ldconfig
    success "ldconfig updated with CUDA paths."

    # Also export for the current shell session (Ollama server inherits this)
    [ -n "$CUDA_LIB" ]   && export LD_LIBRARY_PATH="${CUDA_LIB}:${LD_LIBRARY_PATH:-}"
    [ -n "$DRIVER_LIB" ] && export LD_LIBRARY_PATH="${DRIVER_LIB}:${LD_LIBRARY_PATH:-}"
    info "LD_LIBRARY_PATH set to: $LD_LIBRARY_PATH"
else
    warn "Could not find CUDA or driver lib directories."
    warn "If Ollama still falls back to CPU, set LD_LIBRARY_PATH manually:"
    warn "  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH"
fi

# ── 8. Add LD_LIBRARY_PATH to /etc/environment for persistence ───────────────
# This ensures the Ollama systemd service also picks up the CUDA libs.
if [ -n "$CUDA_LIB" ] || [ -n "$DRIVER_LIB" ]; then
    info "Persisting LD_LIBRARY_PATH in /etc/environment..."
    COMBINED_PATH="${CUDA_LIB:-}${CUDA_LIB:+:}${DRIVER_LIB:-}"
    # Only add if not already present
    if ! grep -q "LD_LIBRARY_PATH" /etc/environment 2>/dev/null; then
        echo "LD_LIBRARY_PATH=${COMBINED_PATH}" | sudo tee -a /etc/environment > /dev/null
        success "LD_LIBRARY_PATH added to /etc/environment."
    else
        warn "LD_LIBRARY_PATH already in /etc/environment. Skipping."
    fi
fi

# ── 9. Restart Ollama service so it picks up the new library paths ────────────
info "Restarting Ollama service to apply GPU library fix..."
if systemctl is-active --quiet ollama 2>/dev/null; then
    sudo systemctl restart ollama
    sleep 5
    success "Ollama service restarted."
elif pgrep -x ollama &>/dev/null; then
    # Ollama running as a plain process (not systemd)
    pkill -x ollama 2>/dev/null || true
    sleep 2
    nohup env LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" ollama serve \
        > /tmp/ollama.log 2>&1 &
    sleep 5
    success "Ollama restarted as background process."
else
    # Not running yet — start it fresh
    nohup env LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" ollama serve \
        > /tmp/ollama.log 2>&1 &
    sleep 5
    success "Ollama started."
fi

# Wait for server to be ready
info "Waiting for Ollama server to become ready..."
for i in $(seq 1 30); do
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        success "Ollama server is ready."
        break
    fi
    sleep 2
    if [ "$i" -eq 30 ]; then
        warn "Ollama server did not respond within 60 seconds."
        warn "Check /tmp/ollama.log for details."
    fi
done

# ── 10. Verify Ollama can see the GPU ─────────────────────────────────────────
info "Verifying Ollama GPU detection..."
OLLAMA_LOG_CHECK=$(curl -s http://localhost:11434/api/tags 2>/dev/null || echo "")
if journalctl -u ollama -n 20 --no-pager 2>/dev/null | grep -q "inference compute.*cuda"; then
    success "Ollama is using CUDA (GPU detected)."
elif [ -f /tmp/ollama.log ] && grep -q "inference compute.*cuda" /tmp/ollama.log 2>/dev/null; then
    success "Ollama is using CUDA (GPU detected)."
else
    warn "Could not confirm GPU detection from logs."
    warn "Run: OLLAMA_DEBUG=1 ollama run deepseek-r1:70b 'hi' 2>&1 | head -20"
    warn "Look for 'inference compute ... library=cuda' in the output."
fi

# ── 11. GPU status ────────────────────────────────────────────────────────────
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
