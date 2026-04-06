# Refund GRPO Training

OpsArena now includes a narrow refund-only GRPO scaffold:

- `opsarena/training/refund_grpo_env.py`
- `scripts/train_refund_grpo.py`

This is the recommended starting point for RL experiments.

## Why a narrow wrapper

The base environment exposes a large typed action space across refunds, AP, KYC, and queue operations. That is too broad for an initial small-model GRPO run.

`RefundExceptionToolEnv` narrows training to:

- one workflow: `refund_exception`
- a smaller public tool surface
- compact text observations with:
  - available actions
  - current queue and case state
  - notification and QA requirements
  - pending case events such as dispute outcomes

## Reward function

`refund_terminal_benchmark_reward` uses bounded shaping plus a terminal bonus:

- valid tool calls add a small shaping reward
- intermediate `benchmark_score` progress adds bounded shaping even before the episode is done
- completed episodes receive a larger terminal bonus from final `benchmark_score`
- repeated invalid actions apply a small penalty

This keeps the policy anchored to the final grader while avoiding a dead gradient when the base
model cannot yet complete refund episodes reliably.

## Usage

Install training dependencies:

```bash
pip install -e ".[dev,train]"
```

If you want `--use-vllm`, also install the vLLM extra and verify that the active PyTorch
build includes CUDA libraries. On Linux `aarch64`, the default PyPI `torch` wheel can be
CPU-only, which causes `vllm` imports to fail with `libtorch_cuda.so: cannot open shared
object file`. Replace it with the official CUDA 13.0 wheel in the repo venv:

```bash
uv pip install --python .venv/bin/python -e ".[dev,train,train-vllm]"
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu130 \
  --reinstall torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
```

Run a first training job:

```bash
python scripts/train_refund_grpo.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir artifacts/refund-grpo
```

## Current recommendation

Prefer:

- refund-only training first
- non-vLLM generation first, or vLLM colocate mode only after smoke checks

Do not assume vLLM server mode is safe for multi-step agent rollouts yet. Upstream TRL has an open issue on per-rollout prefix handling for multi-step OpenEnv trajectories, which matters for environments like OpsArena.

## What is still missing

This scaffold is a starting point, not the final training stack.

Still needed before a broader run:

1. task-specific wrappers for invoice and KYC
2. stronger queue-triage scenario entropy
3. family-level train/dev/test/OOD splits
4. a custom rollout path if dynamic tool masking becomes necessary
