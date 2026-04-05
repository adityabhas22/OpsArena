"""Ensure CUDA and PyTorch shared libraries are loadable before vLLM import.

vLLM's ``_C`` extension links against both ``libcudart.so.12`` (CUDA runtime) and
``libtorch_cuda.so`` (PyTorch CUDA).  On systems where CUDA is installed in a
non-standard prefix (e.g. DGX Spark with CUDA 13 system-wide and only Ollama's
bundled CUDA 12 libs available), neither library is on ``LD_LIBRARY_PATH`` by default.

We unconditionally prepend:
1. PyTorch's own ``lib/`` directory  — provides ``libtorch_cuda.so`` and often
   ``libcudart.so`` too (pip-wheel installs).
2. The first system directory that contains ``libcudart.so*``  — Ollama's bundled
   CUDA 12 path, standard CUDA toolkit prefixes, or distro lib dirs.

Both must be on ``LD_LIBRARY_PATH`` before ``import vllm`` runs.
"""

from __future__ import annotations

import os


# System directories to probe for libcudart.so*, in priority order.
_CUDART_SEARCH_DIRS = [
    # Ollama bundles CUDA 12 runtime on DGX Spark (system CUDA is 13)
    "/usr/local/lib/ollama/cuda_v12",
    # Standard CUDA toolkit installations
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
    # ARM64 (sbsa-linux) CUDA toolkit paths
    "/usr/local/cuda-12/targets/sbsa-linux/lib",
    "/usr/local/cuda-12.8/targets/sbsa-linux/lib",
    # RHEL-style
    "/usr/lib64",
    # Debian/Ubuntu
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu",
]


def _has_libcudart(lib_dir: str) -> bool:
    try:
        return any(f.startswith("libcudart") for f in os.listdir(lib_dir))
    except (FileNotFoundError, PermissionError):
        return False


def _prepend(lib_dir: str) -> None:
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in current.split(os.pathsep) if p]
    if lib_dir not in parts:
        os.environ["LD_LIBRARY_PATH"] = lib_dir if not current else lib_dir + os.pathsep + current


def prepend_nvidia_cuda_runtime_lib_path() -> None:
    """Prepend PyTorch lib/ and a libcudart directory to ``LD_LIBRARY_PATH``.

    Must be called before ``import vllm`` (or ``from trl import GRPOTrainer``
    when ``use_vllm=True``).  Safe to call multiple times.
    """

    # Always add torch's lib/ first — vLLM's _C extension links against
    # libtorch_cuda.so which lives there regardless of CUDA install location.
    try:
        import torch as _torch
        torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
        if os.path.isdir(torch_lib):
            _prepend(torch_lib)
    except ImportError:
        pass

    # Add nvidia pip package lib/ if present (pip-wheel torch installs).
    try:
        import nvidia.cuda_runtime as _ncr  # type: ignore[import]
        ncr_file = getattr(_ncr, "__file__", None)
        if ncr_file:
            ncr_lib = os.path.join(os.path.dirname(ncr_file), "lib")
            if os.path.isdir(ncr_lib):
                _prepend(ncr_lib)
    except ImportError:
        pass

    # Find a directory that actually contains libcudart.so* and add it.
    for lib_dir in _CUDART_SEARCH_DIRS:
        if _has_libcudart(lib_dir):
            _prepend(lib_dir)
            return
