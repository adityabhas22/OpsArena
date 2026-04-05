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


def _render_observation(observation) -> str:
    return json.dumps(observation.model_dump(mode="json"), indent=2, default=str)


def run_task(base_url: str, task_id: str, model: str, seed: int) -> dict:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    tools = render_action_catalog_as_json()
    env = OpsArenaEnv(base_url=base_url).sync()
    with env:
        result = env.reset(task_id=task_id, seed=seed)
        while not result.done:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _render_observation(result.observation)},
                ],
                tools=tools,
                tool_choice="required",
            )
            tool_call = completion.choices[0].message.tool_calls[0]
            arguments = json.loads(tool_call.function.arguments or "{}")
            action = RawOpsAction(action_type=tool_call.function.name, **arguments)
            result = env.step(action)
        return env.state().model_dump()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    for task_id in ("refund_exception", "invoice_plus_kyc", "queue_triage"):
        state = run_task(args.base_url, task_id, args.model, args.seed)
        print(json.dumps({"task_id": task_id, "state": state}, indent=2))


if __name__ == "__main__":
    main()
