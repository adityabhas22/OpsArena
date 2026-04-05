"""Ensure CUDA / PyTorch shared libraries are discoverable by vLLM.

TRL 1.0.0 imports ``VLLMGeneration`` at module level inside ``grpo_trainer.py``,
which triggers ``vllm._C`` to load.  ``vllm._C`` links against:

  - ``libcudart.so.12``  — CUDA 12 runtime
  - ``libtorch_cuda.so`` / ``libtorch.so`` / ``libc10*.so``  — PyTorch CUDA libs

On DGX Spark the system CUDA is 13.x, so ``libcudart.so.12`` is only available
inside Ollama's bundled CUDA 12 directory.  PyTorch's libs are in the venv's
torch package dir.  Neither is on ``LD_LIBRARY_PATH`` by default.

Strategy:
  1. Collect every lib directory that matters (torch/lib, cudart, nvidia pip).
  2. If ``LD_LIBRARY_PATH`` already contains them, do nothing.
  3. Otherwise set ``LD_LIBRARY_PATH`` and **re-exec the current process** so
     the dynamic linker picks up the new paths from the very start.  This is
     the only fully reliable approach — ``ctypes.CDLL`` preloading works for
     *symbols* but not for ``DT_NEEDED`` resolution in subsequently loaded
     shared objects.
"""

from __future__ import annotations

import os
import sys


_CUDART_SEARCH_DIRS = [
    "/usr/local/lib/ollama/cuda_v12",
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
    "/usr/local/cuda-12/targets/sbsa-linux/lib",
    "/usr/local/cuda-12.8/targets/sbsa-linux/lib",
    "/usr/lib64",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu",
]

_ENV_MARKER = "_OPSARENA_LD_REEXEC"


def _find_lib_dirs() -> list[str]:
    """Return library directories that need to be on LD_LIBRARY_PATH."""
    dirs: list[str] = []

    # PyTorch lib/
    try:
        import torch as _torch
        torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
        if os.path.isdir(torch_lib):
            dirs.append(torch_lib)
    except ImportError:
        pass

    # nvidia-cuda-runtime pip package
    try:
        import nvidia.cuda_runtime as _ncr  # type: ignore[import]
        ncr_file = getattr(_ncr, "__file__", None)
        if ncr_file:
            ncr_lib = os.path.join(os.path.dirname(ncr_file), "lib")
            if os.path.isdir(ncr_lib):
                dirs.append(ncr_lib)
    except ImportError:
        pass

    # System / Ollama cudart dirs
    for d in _CUDART_SEARCH_DIRS:
        if os.path.isdir(d):
            for name in ("libcudart.so.12", "libcudart.so"):
                if os.path.isfile(os.path.join(d, name)):
                    dirs.append(d)
                    break

    return dirs


def prepend_nvidia_cuda_runtime_lib_path() -> None:
    """Ensure torch/lib and cudart paths are on LD_LIBRARY_PATH.

    If they aren't already present, this function sets them and **re-execs
    the current process** so the dynamic linker sees the paths from startup.
    A marker env var prevents infinite re-exec loops.

    Must be called before ``from trl import GRPOTrainer``.
    """
    if os.environ.get(_ENV_MARKER):
        return

    needed = _find_lib_dirs()
    if not needed:
        return

    current = os.environ.get("LD_LIBRARY_PATH", "")
    current_parts = set(p for p in current.split(os.pathsep) if p)
    missing = [d for d in needed if d not in current_parts]

    if not missing:
        return

    new_ld = os.pathsep.join(missing)
    if current:
        new_ld = new_ld + os.pathsep + current
    os.environ["LD_LIBRARY_PATH"] = new_ld
    os.environ[_ENV_MARKER] = "1"

    print(f"[cuda_lib_path] LD_LIBRARY_PATH updated, re-execing: +{missing}")
    os.execvp(sys.executable, [sys.executable] + sys.argv)
