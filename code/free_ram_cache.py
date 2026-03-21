"""
free_ram_cache.py
=================
Forces the Linux OS to reclaim page cache by allocating a large block of
memory and immediately releasing it. This causes the kernel to evict cached
pages to satisfy the allocation, converting buff/cache into free memory.

Ollama reads /proc/meminfo's MemFree field instead of MemAvailable, so even
when 93 GiB is "available" it will refuse to load a 36 GiB model if MemFree
is only 29 GiB. Running this before starting Ollama forces MemFree up.

Usage (standalone):
    python3 code/free_ram_cache.py [target_gb]
    python3 code/free_ram_cache.py 40   # ensure at least 40 GB free

Usage (from another script):
    from free_ram_cache import ensure_free_ram_gb
    ensure_free_ram_gb(40)
"""

import ctypes
import os
import subprocess
import sys
import time


def _get_meminfo() -> dict:
    """Parse /proc/meminfo and return values in MiB."""
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    val = int(parts[1])  # kB
                    info[key] = val // 1024  # convert to MiB
    except Exception:
        pass
    return info


def _free_mb() -> int:
    return _get_meminfo().get("MemFree", 0)


def _available_mb() -> int:
    return _get_meminfo().get("MemAvailable", 0)


def ensure_free_ram_gb(target_gb: float = 40.0) -> bool:
    """
    Ensure at least `target_gb` GiB of FREE (not just available) RAM.

    Strategy:
    1. Check if MemFree is already sufficient.
    2. If not, allocate a bytearray of (target - current_free + 2 GB buffer)
       to force the OS to evict page cache, then immediately del it.
    3. Call malloc_trim(0) via libc to return freed heap memory to the OS.
    4. Verify MemFree increased.

    Returns True if target was met, False otherwise.
    """
    target_mb = int(target_gb * 1024)
    free_before = _free_mb()
    avail_before = _available_mb()

    print(f"  [RAM] MemFree={free_before} MiB  MemAvailable={avail_before} MiB  "
          f"Target={target_mb} MiB free")

    if free_before >= target_mb:
        print(f"  [RAM] Already have {free_before} MiB free. No action needed.")
        return True

    if avail_before < target_mb:
        print(f"  [RAM] WARNING: Only {avail_before} MiB available — not enough "
              f"to meet {target_mb} MiB target even after cache eviction.")
        return False

    # How much do we need to allocate to push MemFree up?
    alloc_mb = min(target_mb - free_before + 2048, avail_before - 1024)
    alloc_mb = max(alloc_mb, 0)

    if alloc_mb <= 0:
        return True

    print(f"  [RAM] Allocating {alloc_mb} MiB to force page-cache eviction...")
    try:
        # Allocate and touch every page to force actual physical allocation
        chunk = bytearray(alloc_mb * 1024 * 1024)
        # Touch every 4096th byte (one per page) to ensure pages are faulted in
        for i in range(0, len(chunk), 4096):
            chunk[i] = 0
        del chunk
    except MemoryError:
        print("  [RAM] MemoryError during allocation — system may be truly low on memory.")

    # Call malloc_trim to return freed memory to OS immediately
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass

    time.sleep(1)

    free_after = _free_mb()
    print(f"  [RAM] After eviction: MemFree={free_after} MiB "
          f"(gained {free_after - free_before} MiB)")

    if free_after >= target_mb:
        print(f"  [RAM] Target met. Ollama should now be able to load the model.")
        return True
    else:
        print(f"  [RAM] Still only {free_after} MiB free. "
              f"Ollama may still fail if it reads MemFree.")
        return False


if __name__ == "__main__":
    target = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
    ok = ensure_free_ram_gb(target)
    sys.exit(0 if ok else 1)
