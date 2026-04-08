from __future__ import annotations

"""OpsArena benchmark inference entrypoint.

This script is intentionally designed as a strict, evaluator-friendly harness for
OpenEnv benchmark execution.

Operational goals:
1) Use environment-provided runtime configuration (HF_TOKEN, API_BASE_URL,
    MODEL_NAME) so no provider secrets are hardcoded.
2) Emit structured, flush-safe stdout markers so the evaluator can parse progress
    online:
    - [START] once per task
    - [STEP] once per attempted or successful action
    - [END] once per task with summary fields
3) Drive the environment through tool-calling with dynamic tool masking using
    observation.available_actions to reduce hallucinations.
4) Keep a persistent ReAct-style conversation trace across steps so the model
    remembers prior actions and tool outcomes.
5) Fail soft on malformed model tool calls by feeding error feedback back into
    the conversation, allowing self-correction instead of immediate crashes.

Design notes:
- The evaluator is sensitive to output shape and timing. Every marker uses
  flush=True so partial progress is visible even if the process exits early.
- This is an inference harness, not a training loop. We optimize for robust,
  deterministic benchmark execution and parseable logs.
"""

import argparse
import json
import os
from typing import Any

from openai import OpenAI

from opsarena.action_docs import render_action_catalog_as_json
from opsarena.client import OpsArenaEnv
from opsarena.models import RawOpsAction

SYSTEM_PROMPT = """You are operating an ecommerce operations dashboard.
Resolve cases correctly, follow policy, avoid compliance violations, and use as few unnecessary actions as possible."""

# Canonical enum values for approve.decision_code.
# Some models produce semantically similar but invalid strings; aliases map these
# to valid values prior to Pydantic validation.
VALID_DECISION_CODES = {"standard_approval", "exception_approval", "partial_approval"}
DECISION_CODE_ALIASES = {
    "refund_approved": "standard_approval",
    "approved": "standard_approval",
    "approve": "standard_approval",
    "exception": "exception_approval",
    "partial": "partial_approval",
}

# Upper bound for consecutive malformed/hallucinated tool calls before ending a
# task loop. This prevents infinite retry loops when a model drifts.
MAX_HALLUCINATIONS_PER_TASK = 5


def _render_observation(observation) -> str:
    """Render the environment observation into a stable JSON string.

    The model receives this text as conversational context. We use
    model_dump(mode="json") for Pydantic compatibility and default=str to safely
    serialize any non-JSON-native values.
    """
    return json.dumps(observation.model_dump(mode="json"), indent=2, default=str)


def _normalize_tool_arguments(action_name: str, arguments: dict) -> dict:
    """Normalize model-produced tool arguments before schema validation.

    Why this exists:
    - Models may echo action_type in arguments despite tool_call.function.name
      already carrying the authoritative action identifier.
    - Models may produce near-miss enum tokens (for example, "refund_approved"
      for approve.decision_code) that would otherwise fail validation.

    Parameters
    ----------
    action_name:
        The requested tool/function name from the model tool call.
    arguments:
        Parsed JSON argument object from tool_call.function.arguments.

    Returns
    -------
    dict
        A sanitized copy suitable for RawOpsAction construction.
    """
    normalized = dict(arguments)
    # Some models echo action_type inside arguments; rely on function name as source of truth.
    normalized.pop("action_type", None)

    if action_name == "approve":
        decision_code = normalized.get("decision_code")
        if isinstance(decision_code, str):
            candidate = DECISION_CODE_ALIASES.get(decision_code.strip().lower(), decision_code)
            if candidate not in VALID_DECISION_CODES:
                candidate = "standard_approval"
            normalized["decision_code"] = candidate

    return normalized


def _build_tool_lookup() -> dict[str, dict[str, Any]]:
    """Build a name -> tool schema lookup from the full action catalog.

    The OpenAI-compatible tool payload is a list. Lookup by name allows cheap,
    per-step filtering against observation.available_actions.
    """
    return {
        tool["function"]["name"]: tool
        for tool in render_action_catalog_as_json()
        if isinstance(tool, dict) and "function" in tool and isinstance(tool["function"], dict)
    }


def _tools_for_observation(
    tool_lookup: dict[str, dict[str, Any]], available_actions: list[str] | None
) -> list[dict[str, Any]]:
    """Select tools that are valid for the current observation.

    Dynamic masking narrows the model decision surface and substantially reduces
    invalid tool picks in large multi-workflow catalogs.

    Fallback behavior:
    - If available_actions is empty/missing, return all tools.
    - If none of the listed actions match known tools, return all tools.
      This avoids deadlock if server/client schemas briefly diverge.
    """
    if not available_actions:
        return list(tool_lookup.values())
    filtered = [tool_lookup[name] for name in available_actions if name in tool_lookup]
    return filtered or list(tool_lookup.values())


def run_task(base_url: str, task_id: str, client: OpenAI, model_name: str, seed: int) -> dict:
    """Execute one benchmark task until terminal state or guard-triggered stop.

    Flow summary:
    1) Reset environment for (task_id, seed)
    2) Repeatedly request exactly one tool call from the model
    3) Validate and execute that action against OpsArena
    4) Append tool success/error feedback into message history (ReAct trace)
    5) Emit streaming [STEP] markers after each attempt
    6) Emit [END] marker with task-level rewards and success flag

    Returns
    -------
    dict
        Final environment state via env.state().model_dump().
    """
    tool_lookup = _build_tool_lookup()
    env = OpsArenaEnv(base_url=base_url).sync()

    with env:
        result = env.reset(task_id=task_id, seed=seed)
        step_count = 0
        hallucination_count = 0
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _render_observation(result.observation)},
        ]

        # Required stream marker consumed by evaluator regex parser.
        print(f"[START] env=benchmark task={task_id} model={model_name}", flush=True)

        while not result.done:
            available_actions = result.observation.available_actions
            tools = _tools_for_observation(tool_lookup, available_actions)

            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=tools,
                    tool_choice="required",
                )
            except Exception:
                # Transport/provider failure. Emit parseable step and stop task.
                print(
                    f"[STEP] step={step_count} action=llm_call reward=0.0000 done={result.done} error=llm_error",
                    flush=True,
                )
                break

            message = completion.choices[0].message
            tool_calls = message.tool_calls or []
            if not tool_calls:
                hallucination_count += 1
                # Model response without a tool call is invalid under tool_choice.
                print(
                    f"[STEP] step={step_count} action=no_tool_call reward=0.0000 done={result.done} error=hallucination",
                    flush=True,
                )
                messages.append({"role": "assistant", "content": message.content or ""})
                messages.append(
                    {
                        "role": "user",
                        "content": "You must call exactly one valid tool from available_actions with valid JSON arguments.",
                    }
                )
                if hallucination_count >= MAX_HALLUCINATIONS_PER_TASK:
                    break
                continue

            tool_call = tool_calls[0]
            tool_call_id = getattr(tool_call, "id", None) or f"call_{step_count}"
            raw_arguments = tool_call.function.arguments or "{}"
            action_name = tool_call.function.name
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": action_name,
                                "arguments": raw_arguments,
                            },
                        }
                    ],
                }
            )

            try:
                arguments = json.loads(raw_arguments)
            except Exception as exc:
                hallucination_count += 1
                # Invalid JSON arguments: record failure in logs and feedback loop.
                print(
                    f"[STEP] step={step_count} action={action_name} reward=0.0000 done={result.done} error=hallucination",
                    flush=True,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": action_name,
                        "content": json.dumps(
                            {
                                "ok": False,
                                "error": "invalid_json_arguments",
                                "details": str(exc),
                            }
                        ),
                    }
                )
                if hallucination_count >= MAX_HALLUCINATIONS_PER_TASK:
                    break
                continue

            action_name = tool_call.function.name
            if not isinstance(arguments, dict):
                arguments = {}
            arguments = _normalize_tool_arguments(action_name, arguments)

            try:
                action = RawOpsAction(action_type=action_name, **arguments)
            except Exception as exc:
                hallucination_count += 1
                # Schema validation failed (wrong enum/shape/etc). Keep going with
                # explicit tool feedback so the model can self-correct next turn.
                print(
                    f"[STEP] step={step_count} action={action_name} reward=0.0000 done={result.done} error=hallucination",
                    flush=True,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": action_name,
                        "content": json.dumps(
                            {
                                "ok": False,
                                "error": "validation_error",
                                "details": str(exc),
                            }
                        ),
                    }
                )
                if hallucination_count >= MAX_HALLUCINATIONS_PER_TASK:
                    break
                continue

            result = env.step(action)
            hallucination_count = 0
            step_count += 1
            reward = float(result.reward or 0.0)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": action_name,
                    "content": json.dumps(
                        {
                            "ok": True,
                            "reward": reward,
                            "done": bool(result.done),
                            "observation": result.observation.model_dump(mode="json"),
                        },
                        default=str,
                    ),
                }
            )
            print(
                f"[STEP] step={step_count} action={action_name} reward={reward:.4f} done={result.done} error=none",
                flush=True,
            )

        # END marker includes task summary fields for evaluator post-processing.
        final_state = env.state().model_dump()
        benchmark_reward = float(final_state.get("benchmark_score", 0.0))
        objective_reward = float(final_state.get("objective_score", 0.0))
        train_reward = float(final_state.get("train_score", 0.0))
        cases_resolved = int(final_state.get("cases_resolved", 0) or 0)
        cases_total = int(final_state.get("cases_total", 0) or 0)
        success = bool(result.done and cases_total > 0 and cases_resolved >= cases_total)
        print(
            f"[END] task={task_id} steps={step_count} success={success} "
            f"objective_reward={objective_reward:.4f} train_reward={train_reward:.4f} benchmark_reward={benchmark_reward:.4f}",
            flush=True,
        )
        return final_state


def main() -> None:
    """Entrypoint for benchmark execution across the standard task set."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    # The validator injects these env vars at runtime.
    hf_token = os.environ.get("HF_TOKEN")
    api_base = os.environ.get("API_BASE_URL")
    model_name = os.environ.get("MODEL_NAME") or args.model

    if not hf_token:
        raise RuntimeError("Missing HF_TOKEN environment variable")
    if not api_base:
        raise RuntimeError("Missing API_BASE_URL environment variable")
    if not model_name:
        raise RuntimeError("Missing MODEL_NAME environment variable or --model")

    client = OpenAI(api_key=hf_token, base_url=api_base)

    for task_id in ("refund_exception", "invoice_plus_kyc", "queue_triage", "ap_payment_run"):
        run_task(args.base_url, task_id, client, model_name, args.seed)


if __name__ == "__main__":
    main()