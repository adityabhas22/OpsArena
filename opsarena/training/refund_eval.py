"""Local model evaluation for refund GRPO: same env + seeds for base vs LoRA."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from opsarena.training.refund_grpo_env import REFUND_GRPO_SYSTEM_PROMPT, RefundExceptionToolEnv


def _refund_tool_names() -> list[str]:
    names: list[str] = []
    for name in sorted(dir(RefundExceptionToolEnv)):
        if name.startswith("_"):
            continue
        attr = getattr(RefundExceptionToolEnv, name)
        if not callable(attr) or name == "reset":
            continue
        names.append(name)
    return names


def _invoke_tool(env: RefundExceptionToolEnv, tool: str, arguments: dict[str, Any]) -> str:
    fn = getattr(env, tool, None)
    if fn is None or not callable(fn) or tool.startswith("_"):
        return f"error: unknown tool {tool!r}"
    import inspect

    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    kwargs: dict[str, Any] = {}
    for p in params:
        if p.name in arguments:
            kwargs[p.name] = arguments[p.name]
        elif p.default is inspect.Parameter.empty:
            return f"error: missing required argument {p.name!r} for tool {tool!r}"
    try:
        return fn(**kwargs)
    except Exception as exc:  # noqa: BLE001 — surface to model as observation
        return f"error: {exc}"


def _parse_tool_json(text: str) -> tuple[str, dict[str, Any]] | None:
    """Extract a tool call from model output.

    Supports three formats:
    1. Eval JSON:     {"tool": "open_case", "arguments": {"case_id": "c1"}}
    2. Qwen3 native:  <tool_call>{"name": "open_case", "arguments": {"case_id": "c1"}}</tool_call>
    3. Generic JSON:  {"name": "open_case", "arguments": {"case_id": "c1"}}
    """
    import re

    def _try_load(s: str) -> tuple[str, dict[str, Any]] | None:
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None
        # Accept both "tool" and "name" keys
        t = obj.get("tool") or obj.get("name")
        if not isinstance(t, str):
            return None
        args = obj.get("arguments") or obj.get("parameters") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return None
        if not isinstance(args, dict):
            return None
        return t, args

    raw = text.strip()

    # Try extracting from <tool_call>...</tool_call> blocks (Qwen3 native format)
    tool_call_match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", raw, re.DOTALL)
    if tool_call_match:
        hit = _try_load(tool_call_match.group(1))
        if hit:
            return hit

    # Try the full text as JSON
    hit = _try_load(raw)
    if hit:
        return hit

    # Try extracting first JSON object
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        hit = _try_load(raw[start : end + 1])
        if hit:
            return hit
    return None


def _build_system_prompt() -> str:
    tools = ", ".join(_refund_tool_names())
    return (
        REFUND_GRPO_SYSTEM_PROMPT
        + "\n\nRespond with a tool call each turn. Accepted formats:\n"
        '{"tool": "<name>", "arguments": { ... }}\n'
        "or\n"
        '<tool_call>{"name": "<name>", "arguments": { ... }}</tool_call>\n'
        f"Valid tool names: {tools}\n"
        "Use empty arguments {{}} when there are no parameters."
    )


@dataclass
class RefundEpisodeResult:
    seed: int
    benchmark_score: float
    done: bool
    turns: int
    parse_failures: int
    invalid_actions: int
    messages: list[dict[str, str]] = field(default_factory=list)


def run_refund_episode_local(
    model: Any,
    tokenizer: Any,
    *,
    seed: int,
    max_turns: int = 80,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    device: str | torch.device | None = None,
) -> RefundEpisodeResult:
    """Run one refund episode with JSON tool protocol (fair for base vs fine-tuned)."""

    env = RefundExceptionToolEnv(seed_sequence=[seed], random_seed=0)
    observation = env.reset()
    system = _build_system_prompt()
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": observation},
    ]
    parse_failures = 0
    turns = 0
    dev = device or next(model.parameters()).device

    for _ in range(max_turns):
        if env.done:
            break
        turns += 1
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(dev)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if temperature and temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        messages.append({"role": "assistant", "content": reply})

        parsed = _parse_tool_json(reply)
        if parsed is None:
            parse_failures += 1
            feedback = (
                "Invalid response. Reply with a tool call. Format: "
                '{"tool": "<name>", "arguments": { ... }} '
                'or <tool_call>{"name": "<name>", "arguments": { ... }}</tool_call>'
            )
            messages.append({"role": "user", "content": feedback})
            continue

        tool_name, arguments = parsed
        tool_result = _invoke_tool(env, tool_name, arguments)
        messages.append({"role": "user", "content": f"Tool result:\n{tool_result}"})

    return RefundEpisodeResult(
        seed=seed,
        benchmark_score=float(env.benchmark_score),
        done=bool(env.done),
        turns=turns,
        parse_failures=parse_failures,
        invalid_actions=int(env.invalid_action_count),
        messages=messages,
    )


def load_policy_model(
    base_model_id: str,
    *,
    adapter_path: str | None = None,
    torch_dtype: str = "bfloat16",
) -> tuple[Any, Any]:
    """Load causal LM + tokenizer; optionally merge LoRA from ``adapter_path``."""

    dtype = getattr(torch, torch_dtype, torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer
