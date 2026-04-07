"""SFT warm-start: teach the model tool-call format before GRPO.

Trains LoRA on oracle trajectories for 1-2 epochs so the model knows how to
emit tool calls.  Then merges LoRA into the base model so GRPO can load it
directly with --model.

Usage:
    # Step 1: Generate oracle trajectories
    python scripts/generate_sft_warmstart.py --num-seeds 100

    # Step 2: SFT warm-start (1-2 epochs, ~5 minutes on a single GPU)
    python scripts/sft_warmstart.py --model Qwen/Qwen3-4B-Instruct-2507

    # Step 3: GRPO from the warm-started model
    python scripts/train_refund_grpo.py \
        --model artifacts/sft-warmstart-merged \
        --output-dir artifacts/refund-grpo-v4
"""
from __future__ import annotations

import argparse
import json
from dataclasses import fields

from opsarena.training.cuda_lib_path import prepend_nvidia_cuda_runtime_lib_path


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT warm-start for tool-calling")
    parser.add_argument("--model", required=True, help="Base model")
    parser.add_argument("--data", default="artifacts/sft-warmstart.jsonl")
    parser.add_argument("--output-dir", default="artifacts/sft-warmstart-merged")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    args = parser.parse_args()

    prepend_nvidia_cuda_runtime_lib_path()

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    # Load data
    with open(args.data) as f:
        examples = [json.loads(line) for line in f]
    print(f"[sft_warmstart] Loaded {len(examples)} oracle trajectories from {args.data}")

    dataset = Dataset.from_list(examples)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build SFTConfig with only supported kwargs (TRL version compat)
    candidate = {
        "output_dir": args.output_dir + "-lora",
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": 5,
        "save_strategy": "no",
        "report_to": [],
        "max_length": args.max_seq_length,
        "max_seq_length": args.max_seq_length,
    }
    valid = {f.name for f in fields(SFTConfig)}
    filtered = {k: v for k, v in candidate.items() if k in valid}
    config = SFTConfig(**filtered)

    trainer = SFTTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    trainer.train()

    # Merge LoRA into base model so GRPO can load it as --model
    print("[sft_warmstart] Merging LoRA into base model...")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[sft_warmstart] Saved merged model to {args.output_dir}")


if __name__ == "__main__":
    main()
