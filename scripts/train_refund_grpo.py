from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import fields
from functools import partial

from opsarena.training import (
    RefundExceptionToolEnv,
    build_refund_grpo_prompt_dataset,
    refund_terminal_benchmark_reward,
)
from opsarena.training.cuda_lib_path import prepend_nvidia_cuda_runtime_lib_path
from opsarena.training.json_logger import JSONMetricsLogger
from opsarena.training.trl_tokenizer import prepare_tokenizer_for_grpo


def _validate_vllm_runtime(args: argparse.Namespace) -> None:
    """Fail fast when ``--use-vllm`` is requested from a CPU-only PyTorch install."""

    if not args.use_vllm:
        return

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only in training environments
        raise SystemExit(
            "--use-vllm requires PyTorch to be installed in the active environment before TRL imports."
        ) from exc

    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    has_libtorch_cuda = os.path.isfile(os.path.join(torch_lib, "libtorch_cuda.so"))
    torch_cuda = getattr(torch.version, "cuda", None)

    if torch_cuda and has_libtorch_cuda and torch.cuda.is_available():
        return

    raise SystemExit(
        "--use-vllm requires a CUDA-enabled PyTorch build in the active interpreter.\n"
        f"Detected torch {torch.__version__} at {torch.__file__}\n"
        f"torch.version.cuda={torch_cuda!r}, libtorch_cuda.so present={has_libtorch_cuda}, "
        f"torch.cuda.is_available()={torch.cuda.is_available()}.\n"
        "For this repo's default .venv on Linux/aarch64, install the matching CUDA wheels with:\n"
        "  uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu130 "
        "--reinstall torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0"
    )


def _build_grpo_config(args: argparse.Namespace, GRPOConfig: type) -> object:
    """Instantiate GRPOConfig with only kwargs supported by the installed TRL version.

    TRL 1.x removed ``max_prompt_length`` from ``GRPOConfig``; use ``max_completion_length`` and,
    when vLLM is enabled, ``vllm_max_model_length`` (prompt + completion headroom).
    """

    candidate: dict = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "num_generations": args.num_generations,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seed": args.seed,
        "report_to": [],
        "max_completion_length": args.max_completion_length,
        "use_vllm": args.use_vllm,
        "log_completions": True,
        "logging_steps": 1,
        "save_steps": 50,
        # Qwen3 defaults to <think> mode which burns the entire token budget on
        # chain-of-thought before emitting a tool call.  Disable it so the model
        # spends its tokens on actual tool-calling turns.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if args.use_vllm:
        candidate["vllm_max_model_length"] = args.max_prompt_length + args.max_completion_length + 256

    # TRL requires generation_batch_size >= num_generations and divisible by it.
    # Default is per_device_train_batch_size * gradient_accumulation_steps which
    # may be too small when num_generations is large.
    default_gen_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
    if default_gen_batch < args.num_generations:
        candidate["generation_batch_size"] = args.num_generations

    valid = {f.name for f in fields(GRPOConfig)}
    filtered = {k: v for k, v in candidate.items() if k in valid}
    dropped = set(candidate) - set(filtered)
    if dropped:
        warnings.warn(
            "GRPOConfig in this TRL build does not accept these kwargs (they were ignored): "
            + ", ".join(sorted(dropped))
            + ". Training will use TRL defaults for those settings — check trl.__version__.",
            stacklevel=2,
        )
    config = GRPOConfig(**filtered)
    # One place to read what actually runs (helps debug “stuck” metrics / odd completion lengths).
    mcl = getattr(config, "max_completion_length", None)
    print(
        "[train_refund_grpo] trl GRPOConfig: "
        f"max_completion_length={mcl!r}, num_generations={getattr(config, 'num_generations', None)!r}, "
        f"use_vllm={getattr(config, 'use_vllm', None)!r}"
    )
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training scaffold for OpsArena refund_exception.")
    parser.add_argument("--model", required=True, help="Base chat model, for example Qwen/Qwen3-4B-Instruct-2507.")
    parser.add_argument("--output-dir", default="artifacts/refund-grpo")
    parser.add_argument("--num-examples", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help=(
            "Enable vLLM generation (requires vLLM: pip install -e '.[train-vllm]'). "
            "If import fails with libcudart.so, run without this flag or uninstall vLLM. "
            "In zsh, quote extras. Prefer colocate mode; avoid server mode for multi-step runs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # So vLLM / TRL can find libcudart.so.* from PyTorch's nvidia-cuda-runtime wheels
    prepend_nvidia_cuda_runtime_lib_path()
    _validate_vllm_runtime(args)
    try:
        from datasets import Dataset
        from peft import LoraConfig
        import trl
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:  # pragma: no cover - exercised only in training environments
        raise SystemExit(
            "Training dependencies are missing. Install them with `pip install -e '.[train]'` "
            "and use a CUDA/ROCm/PyTorch setup appropriate for your machine."
        ) from exc

    train_dataset = Dataset.from_list(build_refund_grpo_prompt_dataset(args.num_examples))

    print(f"[train_refund_grpo] trl.__version__={getattr(trl, '__version__', '?')}")
    config = _build_grpo_config(args, GRPOConfig)

    processing_class = prepare_tokenizer_for_grpo(args.model)

    trainer = GRPOTrainer(
        model=args.model,
        processing_class=processing_class,
        reward_funcs=refund_terminal_benchmark_reward,
        train_dataset=train_dataset,
        peft_config=LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
        args=config,
        environment_factory=partial(RefundExceptionToolEnv, random_seed=args.seed),
    )
    trainer.add_callback(JSONMetricsLogger(output_dir=args.output_dir))
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
