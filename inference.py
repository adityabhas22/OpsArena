from __future__ import annotations

import argparse
import json
import os

from openai import OpenAI

from opsarena.action_docs import render_action_catalog_as_json
from opsarena.client import OpsArenaEnv
from opsarena.models import RawOpsAction

SYSTEM_PROMPT = """You are operating an ecommerce operations dashboard.
Resolve cases correctly, follow policy, avoid compliance violations, and use as few unnecessary actions as possible."""

VALID_DECISION_CODES = {"standard_approval", "exception_approval", "partial_approval"}
DECISION_CODE_ALIASES = {
    "refund_approved": "standard_approval",
    "approved": "standard_approval",
    "approve": "standard_approval",
    "exception": "exception_approval",
    "partial": "partial_approval",
}


def _render_observation(observation) -> str:
    return json.dumps(observation.model_dump(mode="json"), indent=2, default=str)


def _normalize_tool_arguments(action_name: str, arguments: dict) -> dict:
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


def run_task(base_url: str, task_id: str, client: OpenAI, model_name: str, seed: int) -> dict:
    tools = render_action_catalog_as_json()
    env = OpsArenaEnv(base_url=base_url).sync()

    with env:
        result = env.reset(task_id=task_id, seed=seed)
        step_count = 0
        print(f"[START] task={task_id} model={model_name}", flush=True)

        while not result.done:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _render_observation(result.observation)},
                ],
                tools=tools,
                tool_choice="required",
            )
            tool_call = completion.choices[0].message.tool_calls[0]
            arguments = json.loads(tool_call.function.arguments or "{}")
            action_name = tool_call.function.name
            if not isinstance(arguments, dict):
                arguments = {}
            arguments = _normalize_tool_arguments(action_name, arguments)
            try:
                action = RawOpsAction(action_type=action_name, **arguments)
            except Exception:
                print(
                    f"[STEP] step={step_count} action={action_name} error=hallucination done=True",
                    flush=True,
                )
                break

            result = env.step(action)
            step_count += 1
            reward = float(result.reward or 0.0)
            print(
                f"[STEP] step={step_count} action={action_name} reward={reward:.4f} done={result.done}",
                flush=True,
            )

        final_state = env.state().model_dump()
        final_score = float(final_state.get("benchmark_score", 0.0))
        print(f"[END] steps={step_count} score={final_score:.4f}", flush=True)
        return final_state


def main() -> None:
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
        state = run_task(args.base_url, task_id, client, model_name, args.seed)
        print(json.dumps({"task_id": task_id, "state": state}, indent=2), flush=True)


if __name__ == "__main__":
    main()