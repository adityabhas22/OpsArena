"""
OpsArena Inference Script
=========================

MANDATORY
- Environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root
  directory of the project.
- Participants must use OpenAI Client for all LLM calls using above variables.

STDOUT FORMAT
- [START] task=<task_name> env=<benchmark> model=<model_name>
- [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
- [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = Any  # type: ignore[misc,assignment]

from opsarena.action_docs import render_action_catalog_as_json
from opsarena.client import OpsArenaEnv
from opsarena.models import OpsArenaObservation, RawOpsAction


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _load_repo_env() -> None:
    env_path = Path(__file__).resolve().with_name(".env")
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_repo_env()

BENCHMARK = os.getenv("OPSARENA_BENCHMARK", "opsarena")
ENV_BASE_URL = os.getenv("OPSARENA_ENV_URL", "http://localhost:8000")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
TASK_IDS = tuple(
    t.strip()
    for t in os.getenv(
        "OPSARENA_TASKS",
        "refund_exception,invoice_plus_kyc,queue_triage,ap_payment_run",
    ).split(",")
    if t.strip()
)
SEED = int(os.getenv("OPSARENA_SEED", "7"))
MAX_STEPS = int(os.getenv("OPSARENA_MAX_STEPS", "80"))
TEMPERATURE = float(os.getenv("OPSARENA_TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("OPSARENA_MAX_TOKENS", "512"))
SUCCESS_THRESHOLD = float(os.getenv("OPSARENA_SUCCESS_THRESHOLD", "0.1"))
MAX_CONTEXT_MESSAGES = int(os.getenv("OPSARENA_MAX_CONTEXT", "60"))


SYSTEM_PROMPT = """\
You are an AI operations agent managing an e-commerce operations queue. \
You process exception cases: refund/dispute handling, invoice reconciliation, \
AP payment runs, and KYC compliance reviews.

WORKFLOW PATTERN (general — adapt to case type):
1. list_queue or open_case → pick a case to work on
2. query_policy → review the relevant policy before decisions
3. view_record → examine linked records (orders, invoices, KYC docs, disputes)
4. Make a decision (approve / reject / escalate) based on evidence gathered
5. send_message → notify the customer of the outcome (see SLOTS below)
6. Complete QA if required (send_to_qa → approve_qa)
7. close_case → finalize with resolution_code="completed"

KEY RULES:
- Always query_policy before approve/reject decisions.
- Always view_record for linked records before deciding.
- Check "close_blockers" in the observation — they list prerequisites for closing.
- Check "recommended_action_categories" — they suggest the category of next action.
- When "waiting_on" has entries, use advance_clock or switch to another case.
- send_message before closing if customer notification is a close blocker.
- For KYC: run_sanctions_screen, verify documents, check beneficial owners.
- For invoices: check for duplicates (record_three_way_match), review PO/receipt.
- For refunds: review dispute evidence, handle chargeback lifecycle.
- For queue_triage: process multiple cases, prioritize by SLA urgency.

PARAMETER HINTS:
- policy_id: "refund_policy", "invoice_policy", or "kyc_policy" (match case_type)
- record_type + record_id: use values from "linked_records" in the observation
- decision_code for approve: "standard_approval" | "exception_approval" | "partial_approval"
- resolution_code for close_case: "completed"
- reason_code for reject: descriptive code e.g. "duplicate_match", "sanctions_match"
- assignee_type for send_to_qa / approve_qa: "qa_reviewer"
- verification_decision for review_kyc: "approve" | "request_resubmission" | "reject"

SEND_MESSAGE SLOTS — send_message requires a "slots" dict with template variables:
- template_id="refund_approved" → slots: {"amount": "<dollar amount>", "order_id": "<order id>"}
- template_id="case_closed" → slots: {"case_id": "<case id>", "resolution": "<resolution summary>"}
- template_id="info_request" → slots: {"fields": "<comma-separated fields needed>"}

If an action fails, read the error message carefully and adjust parameters. \
Do NOT repeat the exact same call — change something.
"""


# ---------------------------------------------------------------------------
# Logging helpers — mandatory [START] / [STEP] / [END] format
# ---------------------------------------------------------------------------

def _format_action(action: RawOpsAction) -> str:
    args = action.model_dump(exclude_none=True)
    args.pop("action_type", None)
    args.pop("metadata", None)
    if not args:
        return action.action_type
    rendered = ",".join(
        f"{k}={json.dumps(v, separators=(',', ':'))}"
        for k, v in sorted(args.items())
    )
    return f"{action.action_type}({rendered})"


def _flat_error(err: str | None) -> str:
    return " ".join(str(err).split()) if err else "null"


def format_start_line(task: str, env: str, model: str) -> str:
    return f"[START] task={task} env={env} model={model}"


def format_step_line(
    step: int, action: RawOpsAction, reward: float, done: bool, error: str | None,
) -> str:
    return (
        f"[STEP] step={step} action={_format_action(action)} reward={reward:.2f} "
        f"done={str(done).lower()} error={_flat_error(error)}"
    )


def format_end_line(
    success: bool, steps: int, score: float, rewards: list[float],
) -> str:
    rstr = ",".join(f"{r:.2f}" for r in rewards)
    return f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rstr}"


def _log(line: str) -> None:
    print(line, flush=True)


# ---------------------------------------------------------------------------
# Tool catalog
# ---------------------------------------------------------------------------

def _build_tool_map() -> dict[str, dict[str, Any]]:
    return {t["function"]["name"]: t for t in render_action_catalog_as_json()}


def _tools_for_available_actions(
    tool_map: dict[str, dict[str, Any]], available: list[str],
) -> list[dict[str, Any]]:
    selected = [tool_map[n] for n in available if n in tool_map]
    return selected or list(tool_map.values())


# ---------------------------------------------------------------------------
# Observation rendering
# ---------------------------------------------------------------------------

def _render_observation(
    task_id: str, obs: OpsArenaObservation,
) -> str:
    lines: list[str] = [
        f"task: {task_id}",
        f"clock: {obs.clock}",
        f"available_actions: {', '.join(obs.available_actions or [])}",
    ]

    if obs.queue_view:
        lines.append("queue:")
        for item in (obs.queue_view or [])[:8]:
            lines.append(
                f"  {item.case_id} | {item.case_type} | {item.priority} | "
                f"sla_rem={item.sla_remaining_minutes}m | {item.status} | "
                f"{item.summary}"
            )

    d = obs.case_detail
    if d:
        lines.append("")
        lines.append("=== OPEN CASE ===")
        lines.append(f"case_id: {d.case_id}")
        lines.append(f"case_type: {d.case_type}")
        lines.append(f"status: {d.status}")
        lines.append(f"case_phase: {d.case_phase}")
        lines.append(f"amount: {d.amount}")
        lines.append(f"required_checks: {', '.join(d.required_checks)}")
        lines.append(f"checks_completed: {', '.join(d.checks_completed)}")
        lines.append(
            f"requested_info_fields: "
            f"{', '.join(d.requested_info_fields) or 'none'}"
        )
        lines.append(
            f"close_blockers: {', '.join(d.close_blockers) or 'none'}"
        )
        lines.append(f"waiting_on: {', '.join(d.waiting_on) or 'none'}")
        lines.append(f"next_due_minutes: {d.next_due_minutes}")
        lines.append(
            f"recommended_action_categories: "
            f"{', '.join(d.recommended_action_categories)}"
        )
        lines.append(
            f"visible_flags: {', '.join(d.visible_flags) or 'none'}"
        )
        lines.append(
            "linked_records: "
            + ", ".join(
                f"{r.record_type}:{r.record_id}" for r in d.linked_records
            )
        )
        lines.append(
            "workflow_metadata: "
            + json.dumps(
                d.workflow_metadata, separators=(",", ":"), default=str,
            )
        )
        if d.communication_log:
            lines.append(
                "communication_log: "
                + ", ".join(
                    f"{e.timestamp}:{e.template_id or e.subject}"
                    for e in d.communication_log[-3:]
                )
            )

    if obs.policy_result:
        lines.append("")
        lines.append(
            "policy_result: "
            + json.dumps(
                obs.policy_result.model_dump(mode="json"),
                separators=(",", ":"),
                default=str,
            )
        )
    if obs.record_view:
        lines.append("")
        lines.append(
            "record_view: "
            + json.dumps(obs.record_view, separators=(",", ":"), default=str)
        )

    if obs.system_message:
        lines.append(f"system_message: {obs.system_message}")
    if obs.error:
        lines.append(f"error: {obs.error}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Conversation management
# ---------------------------------------------------------------------------

def _trim_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Trim conversation keeping system msg + recent assistant/tool pairs intact."""
    if len(messages) <= MAX_CONTEXT_MESSAGES:
        return messages

    system = messages[0]
    rest = messages[1:]

    keep = rest[-MAX_CONTEXT_MESSAGES:]

    while keep and keep[0]["role"] == "tool":
        keep = keep[1:]

    return [system] + keep


def _make_assistant_msg(msg: Any) -> dict[str, Any]:
    """Convert an OpenAI ChatCompletionMessage to a dict for the messages list."""
    out: dict[str, Any] = {"role": "assistant"}
    if msg.content:
        out["content"] = msg.content
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return out


def _add_tool_results(
    messages: list[dict[str, Any]],
    assistant_msg: dict[str, Any],
    first_result: str,
) -> None:
    """Add tool result messages for ALL tool_calls in assistant_msg."""
    for i, tc in enumerate(assistant_msg.get("tool_calls", [])):
        messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": first_result if i == 0 else "Ignored — one action per step.",
        })


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_llm(
    client: OpenAI,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> Any:
    kwargs: dict[str, Any] = dict(
        model=MODEL_NAME,
        messages=_trim_messages(messages),
        tools=tools,
        tool_choice="required",
        parallel_tool_calls=False,
        temperature=TEMPERATURE,
        stream=False,
    )
    if MODEL_NAME.startswith(("gpt-5", "o3", "o4")):
        kwargs["max_completion_tokens"] = MAX_TOKENS
    else:
        kwargs["max_tokens"] = MAX_TOKENS
    return client.chat.completions.create(**kwargs)


def _parse_tool_call(
    msg: Any, obs: OpsArenaObservation,
) -> tuple[RawOpsAction | None, str | None]:
    """Extract the first tool call from the LLM response and build a RawOpsAction."""
    if not msg.tool_calls:
        return None, "no_tool_call"

    tc = msg.tool_calls[0]
    action_name = tc.function.name
    available = set(obs.available_actions or [])
    if action_name not in available:
        return None, f"unavailable:{action_name}"

    try:
        arguments = json.loads(tc.function.arguments or "{}")
    except json.JSONDecodeError as exc:
        return None, f"bad_json:{exc}"
    if not isinstance(arguments, dict):
        return None, "bad_args"

    arguments.pop("action_type", None)

    _QUEUE_LEVEL = {
        "list_queue", "search_cases", "batch_reorder",
        "bulk_assign", "bulk_route", "rebalance_queue", "advance_clock",
    }
    if (
        obs.case_detail
        and "case_id" not in arguments
        and action_name not in _QUEUE_LEVEL
    ):
        arguments["case_id"] = obs.case_detail.case_id

    try:
        return RawOpsAction(action_type=action_name, **arguments), None
    except Exception as exc:
        return None, f"validation:{exc}"


# ---------------------------------------------------------------------------
# Fallback — deliberately minimal, no domain knowledge
# ---------------------------------------------------------------------------

def _fallback_action(obs: OpsArenaObservation) -> RawOpsAction:
    avail = set(obs.available_actions or [])
    if obs.case_detail is None:
        if "open_case" in avail and obs.queue_view:
            return RawOpsAction(
                action_type="open_case", case_id=obs.queue_view[0].case_id,
            )
        if "list_queue" in avail:
            return RawOpsAction(
                action_type="list_queue", limit=10, sort_by="priority",
            )
    if "advance_clock" in avail:
        return RawOpsAction(action_type="advance_clock", minutes=5)
    return RawOpsAction(action_type="list_queue", limit=10, sort_by="priority")


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(
    task_id: str,
    client: OpenAI,
    tool_map: dict[str, dict[str, Any]],
) -> float:
    env = OpsArenaEnv(base_url=ENV_BASE_URL).sync()
    rewards: list[float] = []
    steps_taken = 0
    final_score = 0.0
    success = False

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    _log(format_start_line(task_id, BENCHMARK, MODEL_NAME))

    with env:
        try:
            result = env.reset(task_id=task_id, seed=SEED)

            while not result.done and steps_taken < MAX_STEPS:
                obs = result.observation
                obs_text = _render_observation(task_id, obs)
                tools = _tools_for_available_actions(
                    tool_map, obs.available_actions or [],
                )
                user_msg = {"role": "user", "content": obs_text}

                # --- LLM call ---
                action = None
                assistant_msg = None
                used_fallback = False
                try:
                    resp = _call_llm(
                        client, messages + [user_msg], tools,
                    )
                    raw_msg = resp.choices[0].message
                    assistant_msg = _make_assistant_msg(raw_msg)
                    action, parse_err = _parse_tool_call(raw_msg, obs)
                except Exception as exc:
                    parse_err = f"llm_error:{exc}"

                if action is None:
                    action = _fallback_action(obs)
                    used_fallback = True

                # --- Execute action ---
                try:
                    result = env.step(action)
                except Exception as step_exc:
                    steps_taken += 1
                    rewards.append(0.0)
                    err_msg = " ".join(str(step_exc).split())[:120]
                    _log(format_step_line(
                        steps_taken, action, 0.0, False, err_msg,
                    ))
                    if assistant_msg and "tool_calls" in assistant_msg:
                        messages.append(user_msg)
                        messages.append(assistant_msg)
                        _add_tool_results(
                            messages, assistant_msg,
                            f"STEP FAILED: {err_msg}. Fix parameters or try a different action.",
                        )
                    continue

                steps_taken += 1
                reward = float(result.reward or 0.0)
                rewards.append(reward)
                error = result.observation.error

                _log(format_step_line(
                    steps_taken, action, reward, result.done, error,
                ))

                # --- Commit to conversation history ---
                if assistant_msg and "tool_calls" in assistant_msg:
                    new_obs = _render_observation(task_id, result.observation)
                    if used_fallback:
                        tool_content = (
                            f"ERROR: {parse_err}. Fallback action was used.\n\n"
                            f"Result after fallback ({_format_action(action)}):\n{new_obs}"
                        )
                    elif error:
                        tool_content = f"ACTION ERROR: {error}\n\n{new_obs}"
                    else:
                        tool_content = new_obs

                    messages.append(user_msg)
                    messages.append(assistant_msg)
                    _add_tool_results(messages, assistant_msg, tool_content)

            try:
                final = env.state().model_dump(mode="json")
                final_score = float(final.get("benchmark_score", 0.0))
            except Exception:
                final_score = 0.0
            success = final_score >= SUCCESS_THRESHOLD
            return final_score

        finally:
            _log(format_end_line(success, steps_taken, final_score, rewards))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not HF_TOKEN:
        raise RuntimeError(
            "Set HF_TOKEN, OPENAI_API_KEY, or API_KEY environment variable"
        )
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    tool_map = _build_tool_map()

    scores: dict[str, float] = {}
    for task_id in TASK_IDS:
        scores[task_id] = run_task(task_id, client, tool_map)

    print("\n--- SUMMARY ---", flush=True)
    for tid, sc in scores.items():
        print(f"  {tid}: {sc:.3f}", flush=True)
    avg = sum(scores.values()) / max(1, len(scores))
    print(f"  average: {avg:.3f}", flush=True)


if __name__ == "__main__":
    main()
