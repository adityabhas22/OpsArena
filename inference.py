from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from opsarena.action_docs import render_action_catalog_as_json
from opsarena.client import OpsArenaEnv
from opsarena.models import RawOpsAction

SYSTEM_PROMPT = (
    "You are operating an ecommerce operations dashboard. "
    "Resolve cases correctly, follow policy, avoid compliance violations, and "
    "use as few unnecessary actions as possible."
)

TASK_IDS = (
    "refund_exception",
    "invoice_plus_kyc",
    "queue_triage",
    "ap_payment_run",
)

# Required pre-submission environment variable contract.
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional in docker-image workflow variants.
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


def _build_tool_map() -> dict[str, dict[str, Any]]:
    tools = render_action_catalog_as_json()
    return {tool["function"]["name"]: tool for tool in tools}


def _tools_for_observation(
    tool_map: dict[str, dict[str, Any]], available_actions: list[str] | None
) -> list[dict[str, Any]]:
    if not available_actions:
        return list(tool_map.values())
    selected: list[dict[str, Any]] = []
    for action_name in available_actions:
        tool = tool_map.get(action_name)
        if tool is not None:
            selected.append(tool)
    return selected or list(tool_map.values())


def _render_observation(observation: Any) -> str:
    return json.dumps(observation.model_dump(mode="json"), separators=(",", ":"), default=str)


def run_task(task_id: str, seed: int = 7) -> dict[str, Any]:
    if not HF_TOKEN:
        raise RuntimeError("Missing required environment variable: HF_TOKEN")

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    tool_map = _build_tool_map()

    env = OpsArenaEnv(base_url="http://localhost:8000").sync()
    with env:
        print(f"[START] Task: {task_id} | Seed: {seed}", flush=True)
        result = env.reset(task_id=task_id, seed=seed)

        step_count = 0
        while not result.done:
            step_count += 1
            obs = result.observation
            tools = _tools_for_observation(tool_map, getattr(obs, "available_actions", None))

            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _render_observation(obs)},
                ],
                tools=tools,
                tool_choice="required",
            )

            message = completion.choices[0].message
            if not message.tool_calls:
                print(
                    f"[STEP] {step_count} | Action: list_queue | Args: {{}}",
                    flush=True,
                )
                action = RawOpsAction(action_type="list_queue")
            else:
                tool_call = message.tool_calls[0]
                action_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")
                print(
                    f"[STEP] {step_count} | Action: {action_name} | Args: {args}",
                    flush=True,
                )
                action = RawOpsAction(action_type=action_name, **args)

            result = env.step(action)

        final_state = env.state().model_dump(mode="json")
        final_score = final_state.get("benchmark_score", 0.0)
        print(f"[END] Task: {task_id} | Final Score: {final_score}", flush=True)
        return final_state


def main() -> None:
    seed = int(os.environ.get("SEED", "7"))
    for task_id in TASK_IDS:
        run_task(task_id=task_id, seed=seed)


if __name__ == "__main__":
    main()
