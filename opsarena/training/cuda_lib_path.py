"""Preload CUDA and PyTorch shared libraries so vLLM can import cleanly.

TRL 1.0.0 imports ``VLLMGeneration`` at module level inside ``grpo_trainer.py``,
which triggers ``vllm._C`` to load.  ``vllm._C`` links against:

  - ``libcudart.so.12``  — CUDA 12 runtime
  - ``libtorch_cuda.so`` / ``libtorch.so`` / ``libc10*.so``  — PyTorch CUDA libs

On DGX Spark the system CUDA is 13.x, so ``libcudart.so.12`` is only available
inside Ollama's bundled CUDA 12 directory.  PyTorch's libs are in the venv's
torch package dir.  Neither is on ``LD_LIBRARY_PATH`` by default.

We use ``ctypes.CDLL(..., mode=RTLD_GLOBAL)`` to preload all required libraries
before any TRL/vLLM import occurs.  RTLD_GLOBAL is critical: it exports the
symbols globally so that subsequently opened shared objects can find them.
"""

from __future__ import annotations

import ctypes
import os


# Directories to search for libcudart.so.12, in priority order.
_CUDART_SEARCH_DIRS = [
    # DGX Spark: CUDA 13 system-wide, but Ollama ships CUDA 12 runtime here
    "/usr/local/lib/ollama/cuda_v12",
    # Standard CUDA 12 toolkit installations
    "/usr/local/cuda/lib64",
    "/usr/local/cuda-12/lib64",
    "/usr/local/cuda-12.8/lib64",
    "/usr/local/cuda-12.6/lib64",
    "/usr/local/cuda-12.5/lib64",
    "/usr/local/cuda-12.4/lib64",
    "/usr/local/cuda-12.3/lib64",
    "/usr/local/cuda-12.2/lib64",
    "/usr/local/cuda-12.1/lib64",
    "/usr/local/cuda-12.0/lib64",
    # ARM64 (sbsa-linux) paths
    "/usr/local/cuda-12/targets/sbsa-linux/lib",
    "/usr/local/cuda-12.8/targets/sbsa-linux/lib",
    # RHEL / Debian system paths
    "/usr/lib64",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu",
]

# PyTorch internal libs that vLLM._C links against, in load order.
_TORCH_PRELOAD_LIBS = [
    "libgomp.so.1",       # OpenMP (may already be loaded; ignore if missing)
    "libtorch.so",
    "libtorch_cpu.so",
    "libc10.so",
    "libcudart.so.12",    # may also live in torch/lib for pip-wheel installs
    "libc10_cuda.so",
    "libtorch_cuda.so",
]


def _prepend_ld(lib_dir: str) -> None:
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in current.split(os.pathsep) if p]
    if lib_dir not in parts:
        os.environ["LD_LIBRARY_PATH"] = lib_dir if not current else lib_dir + os.pathsep + current


def _try_load(path: str) -> bool:
    """Attempt RTLD_GLOBAL load; return True on success."""
    try:
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        return True
    except OSError:
        return False


def prepend_nvidia_cuda_runtime_lib_path() -> None:
    """Preload libcudart.so.12 and all PyTorch CUDA libs with RTLD_GLOBAL.

    Must be called before ``from trl import GRPOTrainer``.  Safe to call
    multiple times (ctypes caches loaded libs internally).
    """

    # ── Step 1: locate and load libcudart.so.12 ──────────────────────────────
    cudart_loaded = False
    for lib_dir in _CUDART_SEARCH_DIRS:
        for name in ("libcudart.so.12", "libcudart.so"):
            candidate = os.path.join(lib_dir, name)
            if os.path.isfile(candidate) and _try_load(candidate):
                _prepend_ld(lib_dir)
                cudart_loaded = True
                break
        if cudart_loaded:
            break

    # ── Step 2: locate torch lib/ and preload its CUDA libraries ─────────────
    torch_lib: str | None = None
    try:
        import torch as _torch
        _torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
        if os.path.isdir(_torch_lib):
            torch_lib = _torch_lib
            _prepend_ld(torch_lib)
    except ImportError:
        pass

    if torch_lib:
        for lib_name in _TORCH_PRELOAD_LIBS:
            lib_path = os.path.join(torch_lib, lib_name)
            if os.path.isfile(lib_path):
                _try_load(lib_path)

    # ── Step 3: nvidia pip-package path (pip-wheel torch installs) ───────────
    try:
        import nvidia.cuda_runtime as _ncr  # type: ignore[import]
        ncr_file = getattr(_ncr, "__file__", None)
        if ncr_file:
            ncr_lib = os.path.join(os.path.dirname(ncr_file), "lib")
            if os.path.isdir(ncr_lib):
                _prepend_ld(ncr_lib)
                for name in ("libcudart.so.12", "libcudart.so"):
                    candidate = os.path.join(ncr_lib, name)
                    if os.path.isfile(candidate):
                        _try_load(candidate)
                        break
    except ImportError:
        pass
