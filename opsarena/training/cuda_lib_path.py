"""Ensure CUDA runtime library (libcudart.so.12) is loadable before vLLM import.

vLLM loads ``libcudart.so.12`` at import time via its C extension.  On machines where
CUDA is system-installed (e.g. DGX Spark) or pip-installed (PyTorch nvidia-cuda-runtime
wheels), the library directory may not be on ``LD_LIBRARY_PATH``.

Search order:
1. PyTorch's bundled lib directory (fastest for pip-wheel torch installations).
2. ``nvidia.cuda_runtime`` pip package (also ships ``libcudart.so.12``).
3. Common system CUDA installation prefixes (``/usr/local/cuda*``).
4. Debian/RHEL system lib dirs.

We prepend the *first* matching directory to ``LD_LIBRARY_PATH`` so that vLLM's
``dlopen()`` can find the library during the subsequent ``import vllm``.
"""

from __future__ import annotations

import os


# System CUDA prefixes to probe, ordered from most specific to least.
_SYSTEM_CUDA_LIB64_DIRS = [
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
    # RHEL-style paths
    "/usr/lib64",
    # Debian/Ubuntu paths
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu",
]


def _has_libcudart(lib_dir: str) -> bool:
    """Return True if lib_dir exists and contains any libcudart.so* file."""
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
    """Find libcudart.so.* and prepend its directory to ``LD_LIBRARY_PATH``.

    Runs before vLLM is imported so that vLLM's C extension can dlopen the
    library.  Safe to call multiple times (idempotent).
    """

    candidates: list[str] = []

    # 1. PyTorch bundles libcudart.so in its own lib/ for pip-installed wheels.
    try:
        import torch as _torch
        candidates.append(os.path.join(os.path.dirname(_torch.__file__), "lib"))
    except ImportError:
        pass

    # 2. nvidia-cuda-runtime-cu12 pip package (installed alongside torch cu12 wheels).
    try:
        import nvidia.cuda_runtime as _ncr  # type: ignore[import]
        ncr_file = getattr(_ncr, "__file__", None)
        if ncr_file:
            candidates.append(os.path.join(os.path.dirname(ncr_file), "lib"))
    except ImportError:
        pass

    # 3. System CUDA installation (DGX Spark, bare-metal servers, conda envs).
    candidates.extend(_SYSTEM_CUDA_LIB64_DIRS)

    for lib_dir in candidates:
        if _has_libcudart(lib_dir):
            _prepend(lib_dir)
            return
