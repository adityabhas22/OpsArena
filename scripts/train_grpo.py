"""Unified GRPO training script for all OpsArena tasks.

Usage:
    python scripts/train_grpo.py --task-id refund_exception --model Qwen/Qwen3-4B-Instruct-2507
    python scripts/train_grpo.py --task-id invoice_plus_kyc --model Qwen/Qwen3-4B-Instruct-2507
    python scripts/train_grpo.py --task-id queue_triage --model Qwen/Qwen3-4B-Instruct-2507 --max-completion-length 4096
    python scripts/train_grpo.py --task-id ap_payment_run --model Qwen/Qwen3-4B-Instruct-2507
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from functools import partial
from typing import Any

from opsarena.training import (
    APPaymentRunToolEnv,
    InvoiceKYCToolEnv,
    QueueTriageToolEnv,
    RefundExceptionToolEnv,
    ap_payment_terminal_benchmark_reward,
    build_ap_payment_prompt_dataset,
    build_invoice_kyc_prompt_dataset,
    build_queue_triage_prompt_dataset,
    build_refund_grpo_prompt_dataset,
    invoice_kyc_terminal_benchmark_reward,
    queue_triage_terminal_benchmark_reward,
    refund_terminal_benchmark_reward,
)
from opsarena.training.cuda_lib_path import prepend_nvidia_cuda_runtime_lib_path
from opsarena.training.json_logger import JSONMetricsLogger
from opsarena.training.trl_tokenizer import prepare_tokenizer_for_grpo

# Per-task defaults: (env_class, reward_fn, dataset_builder, default_max_completion, default_max_steps)
TASK_REGISTRY: dict[str, dict[str, Any]] = {
    "refund_exception": {
        "env_cls": RefundExceptionToolEnv,
        "reward_fn": refund_terminal_benchmark_reward,
        "dataset_fn": build_refund_grpo_prompt_dataset,
        "default_max_completion": 2048,
        "default_max_steps": 600,
    },
    "invoice_plus_kyc": {
        "env_cls": InvoiceKYCToolEnv,
        "reward_fn": invoice_kyc_terminal_benchmark_reward,
        "dataset_fn": build_invoice_kyc_prompt_dataset,
        "default_max_completion": 3072,
        "default_max_steps": 600,
    },
    "queue_triage": {
        "env_cls": QueueTriageToolEnv,
        "reward_fn": queue_triage_terminal_benchmark_reward,
        "dataset_fn": build_queue_triage_prompt_dataset,
        "default_max_completion": 4096,
        "default_max_steps": 800,
    },
    "ap_payment_run": {
        "env_cls": APPaymentRunToolEnv,
        "reward_fn": ap_payment_terminal_benchmark_reward,
        "dataset_fn": build_ap_payment_prompt_dataset,
        "default_max_completion": 3072,
        "default_max_steps": 600,
    },
}


def _build_grpo_config(args: argparse.Namespace, GRPOConfig: type) -> object:
    """Instantiate GRPOConfig with only kwargs supported by the installed TRL version."""
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

    default_gen_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
    if default_gen_batch < args.num_generations:
        candidate["generation_batch_size"] = args.num_generations

    valid = {f.name for f in fields(GRPOConfig)}
    filtered = {k: v for k, v in candidate.items() if k in valid}
    dropped = set(candidate) - set(filtered)
    if dropped:
        print(f"[train_grpo] Dropped unsupported GRPOConfig keys: {dropped}")
    return GRPOConfig(**filtered)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified GRPO training for all OpsArena tasks.")
    parser.add_argument("--task-id", required=True, choices=list(TASK_REGISTRY.keys()),
                        help="Which task to train on.")
    parser.add_argument("--model", required=True, help="Base model, e.g. Qwen/Qwen3-4B-Instruct-2507.")
    parser.add_argument("--output-dir", default=None,
                        help="Output dir (default: artifacts/<task-id>-grpo).")
    parser.add_argument("--num-examples", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Training steps (default: per-task).")
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=None,
                        help="Max completion tokens (default: per-task).")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--use-vllm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_cfg = TASK_REGISTRY[args.task_id]

    # Apply per-task defaults
    if args.output_dir is None:
        args.output_dir = f"artifacts/{args.task_id}-grpo"
    if args.max_completion_length is None:
        args.max_completion_length = task_cfg["default_max_completion"]
    if args.max_steps is None:
        args.max_steps = task_cfg["default_max_steps"]

    prepend_nvidia_cuda_runtime_lib_path()
    try:
        from datasets import Dataset
        from peft import LoraConfig
        import trl
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Install with `pip install -e '.[train]'`."
        ) from exc

    print(f"[train_grpo] task={args.task_id}, trl={getattr(trl, '__version__', '?')}, "
          f"max_completion={args.max_completion_length}, max_steps={args.max_steps}")

    train_dataset = Dataset.from_list(task_cfg["dataset_fn"](args.num_examples))
    config = _build_grpo_config(args, GRPOConfig)
    processing_class = prepare_tokenizer_for_grpo(args.model)

    env_cls = task_cfg["env_cls"]
    trainer = GRPOTrainer(
        model=args.model,
        processing_class=processing_class,
        reward_funcs=task_cfg["reward_fn"],
        train_dataset=train_dataset,
        peft_config=LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
        args=config,
        environment_factory=partial(env_cls, random_seed=args.seed),
    )
    trainer.add_callback(JSONMetricsLogger(output_dir=args.output_dir))
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
