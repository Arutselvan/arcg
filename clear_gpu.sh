#!/usr/bin/env bash
# =============================================================================
# clear_gpu.sh  --  Aggressively free zombie VRAM on NVIDIA GPUs
#
# Run this when nvidia-smi shows VRAM used but no processes listed.
# Usage:  bash clear_gpu.sh
# =============================================================================

set -euo pipefail

echo "============================================================"
echo "  GPU VRAM Cleanup Script"
echo "============================================================"

# --- Step 1: Show current VRAM state ---
echo ""
echo "[1] Current VRAM state:"
nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total \
           --format=csv,noheader,nounits | \
  awk -F',' '{printf "    GPU %s: %s MiB used / %s MiB total (%s MiB free)\n", $1, $2, $4, $3}'

# --- Step 2: Kill all Ollama processes ---
echo ""
echo "[2] Killing all Ollama processes..."
pkill -TERM -f ollama 2>/dev/null && echo "    SIGTERM sent" || echo "    No ollama processes found"
sleep 2
pkill -KILL -f ollama 2>/dev/null && echo "    SIGKILL sent" || echo "    No ollama processes remaining"
sleep 3

# --- Step 3: Find and kill any process holding GPU device files ---
echo ""
echo "[3] Finding processes holding /dev/nvidia* device files..."
GPU_PIDS=""
for dev in /dev/nvidia* /dev/nvidiactl /dev/nvidia-uvm; do
  if [ -e "$dev" ]; then
    PIDS=$(fuser "$dev" 2>/dev/null || true)
    if [ -n "$PIDS" ]; then
      echo "    $dev held by PIDs: $PIDS"
      GPU_PIDS="$GPU_PIDS $PIDS"
    fi
  fi
done

if [ -n "$GPU_PIDS" ]; then
  echo "    Killing GPU-holding PIDs:$GPU_PIDS"
  for pid in $GPU_PIDS; do
    kill -9 "$pid" 2>/dev/null && echo "    Killed PID $pid" || echo "    PID $pid already gone"
  done
  sleep 3
else
  echo "    No processes found holding GPU device files."
fi

# --- Step 4: Try CUDA context reset via Python ctypes ---
echo ""
echo "[4] Attempting CUDA primary context reset via ctypes..."
python3 - <<'PYEOF'
import ctypes, sys

libcuda_paths = [
    "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
    "/usr/local/cuda/lib64/libcuda.so",
    "libcuda.so.1",
]
libcuda = None
for path in libcuda_paths:
    try:
        libcuda = ctypes.CDLL(path)
        print(f"    Loaded: {path}")
        break
    except OSError:
        continue

if not libcuda:
    print("    libcuda.so not found. Skipping.")
    sys.exit(0)

ret = libcuda.cuInit(0)
print(f"    cuInit(0) = {ret}")

device_count = ctypes.c_int(0)
libcuda.cuDeviceGetCount(ctypes.byref(device_count))
print(f"    Devices found: {device_count.value}")

for i in range(device_count.value):
    device = ctypes.c_int(0)
    libcuda.cuDeviceGet(ctypes.byref(device), i)
    ret = libcuda.cuDevicePrimaryCtxReset(device)
    print(f"    cuDevicePrimaryCtxReset(GPU {i}) = {ret}  (0=success, 201=not owner)")
PYEOF

# --- Step 5: Try nvidia-smi --gpu-reset (bare metal only) ---
echo ""
echo "[5] Attempting nvidia-smi --gpu-reset..."
if sudo nvidia-smi --gpu-reset -i 0 2>/dev/null; then
  echo "    nvidia-smi --gpu-reset succeeded!"
else
  echo "    nvidia-smi --gpu-reset not available (container restriction)."
fi

# --- Step 6: Try unloading/reloading nvidia_uvm kernel module ---
echo ""
echo "[6] Attempting nvidia_uvm kernel module reload..."
if sudo rmmod nvidia_uvm 2>/dev/null; then
  echo "    nvidia_uvm unloaded."
  sleep 2
  if sudo modprobe nvidia_uvm 2>/dev/null; then
    echo "    nvidia_uvm reloaded."
  else
    echo "    WARNING: nvidia_uvm could not be reloaded. Run: sudo modprobe nvidia_uvm"
  fi
else
  echo "    Could not unload nvidia_uvm (still in use or no permission)."
fi

# --- Step 7: Final VRAM state ---
echo ""
echo "[7] VRAM state after cleanup:"
nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total \
           --format=csv,noheader,nounits | \
  awk -F',' '{printf "    GPU %s: %s MiB used / %s MiB total (%s MiB free)\n", $1, $2, $4, $3}'

echo ""
echo "============================================================"
echo "  Done. If VRAM is still not free, a full reboot is needed."
echo "  Run: sudo reboot"
echo "============================================================"
