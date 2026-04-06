"""Benchmark a GRPO checkpoint using the same protocol as training.

Uses the RefundExceptionToolEnv directly (same as TRL's environment_factory),
with the Qwen3 chat template + tool schemas, so the model sees the exact
same format it was trained on.

Usage:
    # Benchmark GRPO checkpoint
    python scripts/benchmark_checkpoint.py \
        --model artifacts/sft-warmstart-merged \
        --adapter artifacts/refund-grpo-v4/checkpoint-150 \
        --seeds 1,2,3,4,5,6,7,8,9,10,11,12

    # Benchmark SFT warm-start (no adapter)
    python scripts/benchmark_checkpoint.py \
        --model artifacts/sft-warmstart-merged \
        --seeds 1,2,3,4,5,6,7,8,9,10,11,12

    # Benchmark base model
    python scripts/benchmark_checkpoint.py \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --seeds 1,2,3,4,5,6,7,8,9,10,11,12
"""
from __future__ import annotations

import argparse
import inspect
import json
import re
from typing import Any

from opsarena.training.refund_grpo_env import REFUND_GRPO_SYSTEM_PROMPT, RefundExceptionToolEnv
from opsarena.training.cuda_lib_path import prepend_nvidia_cuda_runtime_lib_path


def _get_tool_schemas(env_cls: type) -> list[dict]:
    """Build JSON tool schemas from env class public methods (same as TRL does)."""
    schemas = []
    for name, method in inspect.getmembers(env_cls, predicate=inspect.isfunction):
        if name.startswith("_") or name == "reset":
            continue
        sig = inspect.signature(method)
        params = {}
        required = []
        for pname, p in sig.parameters.items():
            if pname == "self":
                continue
            ptype = "string"
            if p.annotation in (int, "int"):
                ptype = "integer"
            elif p.annotation in (float, "float"):
                ptype = "number"
            elif p.annotation in (bool, "bool"):
                ptype = "boolean"
            elif hasattr(p.annotation, "__args__"):  # list[str] etc
                ptype = "array"
            params[pname] = {"type": ptype}
            if p.default is inspect.Parameter.empty:
                required.append(pname)
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": (method.__doc__ or name).split("\n")[0],
                "parameters": {
                    "type": "object",
                    "properties": params,
                    "required": required,
                },
            },
        })
    return schemas


def _parse_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    """Parse tool call from model output — handles multiple formats."""
    # Qwen3 native: <tool_call>{"name": ..., "arguments": ...}</tool_call>
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(1))
            name = obj.get("name") or obj.get("tool")
            args = obj.get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)
            if isinstance(name, str) and isinstance(args, dict):
                return name, args
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: {"tool": ..., "arguments": ...} or {"name": ..., "arguments": ...}
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            name = obj.get("tool") or obj.get("name")
            args = obj.get("arguments") or obj.get("parameters") or {}
            if isinstance(args, str):
                args = json.loads(args)
            if isinstance(name, str) and isinstance(args, dict):
                return name, args
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def run_episode(model, tokenizer, *, seed: int, max_turns: int = 30, tools: list[dict]) -> dict:
    """Run one episode using the training-identical chat template with tools."""
    import torch

    env = RefundExceptionToolEnv(seed_sequence=[seed], random_seed=0)
    observation = env.reset()

    messages = [
        {"role": "system", "content": REFUND_GRPO_SYSTEM_PROMPT},
        {"role": "user", "content": (
            "Resolve the refund case by calling tools step by step: "
            "open the case, query policy, make a decision, send notifications "
            "if required, complete QA if required, then close the case. "
            "Do not stop until the case is closed.\n\n"
            f"Environment:\n{observation}"
        )},
    ]

    device = next(model.parameters()).device
    parse_failures = 0
    tool_calls = 0

    for turn in range(max_turns):
        if env.done:
            break

        # Apply chat template WITH tools (same as TRL does during training)
        prompt = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        reply = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
        # Clean up special tokens for display but keep tool_call tags for parsing
        reply_clean = reply.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

        parsed = _parse_tool_call(reply_clean)
        if parsed is None:
            parse_failures += 1
            messages.append({"role": "assistant", "content": reply_clean})
            messages.append({"role": "user", "content": (
                "Call a tool. Use the format: "
                '<tool_call>{"name": "<tool_name>", "arguments": {...}}</tool_call>'
            )})
            continue

        tool_name, arguments = parsed
        tool_calls += 1

        # Add as proper tool_call message
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {
                "name": tool_name,
                "arguments": json.dumps(arguments),
            }}],
        })

        # Execute tool
        fn = getattr(env, tool_name, None)
        if fn is None or not callable(fn) or tool_name.startswith("_"):
            tool_result = f"error: unknown tool {tool_name!r}"
        else:
            try:
                sig = inspect.signature(fn)
                kwargs = {k: v for k, v in arguments.items() if k in sig.parameters}
                tool_result = fn(**kwargs)
            except Exception as exc:
                tool_result = f"error: {exc}"

        messages.append({"role": "tool", "content": tool_result, "name": tool_name})

    return {
        "seed": seed,
        "benchmark_score": round(env.benchmark_score, 4),
        "done": env.done,
        "turns": turn + 1 if not env.done else turn,
        "tool_calls": tool_calls,
        "parse_failures": parse_failures,
        "invalid_actions": env.invalid_action_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None, help="LoRA adapter path")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer model ID (defaults to --model)")
    parser.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10,11,12")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    prepend_nvidia_cuda_runtime_lib_path()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    seeds = [int(s) for s in args.seeds.split(",")]
    tokenizer_id = args.tokenizer or args.model

    print(f"[benchmark] Loading tokenizer from {tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[benchmark] Loading model from {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.adapter:
        print(f"[benchmark] Loading adapter from {args.adapter}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()

    model.eval()
    tools = _get_tool_schemas(RefundExceptionToolEnv)

    results = []
    for seed in seeds:
        r = run_episode(model, tokenizer, seed=seed, max_turns=args.max_turns, tools=tools)
        results.append(r)
        print(f"  seed={r['seed']}  score={r['benchmark_score']:.4f}  done={r['done']}  "
              f"tools={r['tool_calls']}  parse_fail={r['parse_failures']}  invalid={r['invalid_actions']}")

    mean_score = sum(r["benchmark_score"] for r in results) / len(results)
    done_rate = sum(1 for r in results if r["done"]) / len(results)
    print(f"\n--- Results ---")
    print(f"mean_benchmark_score: {mean_score:.4f}")
    print(f"done_rate: {done_rate:.4f}")
    print(f"seeds evaluated: {len(results)}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"results": results, "mean_score": mean_score, "done_rate": done_rate}, f, indent=2)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
