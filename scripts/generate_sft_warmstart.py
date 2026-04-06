"""Generate SFT warm-start data from oracle trajectories.

Runs the oracle across many seeds, captures each step as a chat message with
tool calls, and saves as a JSONL dataset suitable for SFT before GRPO.

This breaks the GRPO cold-start problem: the model learns tool-call format
from supervised examples, then GRPO fine-tunes the policy decisions.

Usage:
    python scripts/generate_sft_warmstart.py --task-id refund_exception --num-seeds 100
    python scripts/generate_sft_warmstart.py --task-id refund_exception --num-seeds 100 --output artifacts/sft-warmstart.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from typing import Any

from opsarena.training.refund_grpo_env import REFUND_GRPO_SYSTEM_PROMPT, RefundExceptionToolEnv


def _call_tool(env: RefundExceptionToolEnv, messages: list, tool_name: str, tool_args: dict) -> str | None:
    """Execute one tool call, append assistant + tool messages, return obs or None on error."""
    messages.append({
        "role": "assistant",
        "tool_calls": [{"type": "function", "function": {
            "name": tool_name,
            "arguments": json.dumps(tool_args),
        }}],
        "content": "",
    })
    try:
        obs_text = getattr(env, tool_name)(**tool_args)
    except Exception:
        messages.pop()  # Remove failed assistant message
        return None
    messages.append({"role": "tool", "content": obs_text, "name": tool_name})
    return obs_text


def _run_oracle_trajectory(seed: int) -> list[dict[str, Any]] | None:
    """Run oracle on a seed and return the chat trajectory, or None on failure."""
    from opsarena.domain.workflows.refund import DisputeStage, MonitoringProgramStatus, RefundWorkflowState

    env = RefundExceptionToolEnv(seed_sequence=[seed])
    initial_obs = env.reset()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": REFUND_GRPO_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Resolve the refund case by calling tools step by step: "
                "open the case, query policy, make a decision, send notifications "
                "if required, complete QA if required, then close the case. "
                "Do not stop until the case is closed.\n\n"
                f"Environment:\n{initial_obs}"
            ),
        },
    ]

    state = env._env._state
    if state is None:
        return None

    case_id = next(iter(state.cases.keys()), None)
    if case_id is None:
        return None

    # 1. Open case
    _call_tool(env, messages, "open_case", {"case_id": case_id})

    # 2. Query policy
    _call_tool(env, messages, "query_policy", {"policy_id": "refund_policy"})

    # 3. Risk mitigations
    case = env._env._state.cases[case_id]
    workflow = case.workflow
    if isinstance(workflow, RefundWorkflowState):
        if workflow.monitoring_program_status == MonitoringProgramStatus.BREACHED and not workflow.payout_frozen:
            _call_tool(env, messages, "freeze_payouts", {"case_id": case_id, "reason_code": "threshold_exceeded"})

        # 4. Decision
        if case.resolution.value == "pending":
            if workflow.dispute_stage == DisputeStage.INQUIRY:
                if case.hidden.true_dispute_should_accept or case.amount <= 75:
                    _call_tool(env, messages, "refund_pre_dispute_alert", {"case_id": case_id})
                else:
                    _call_tool(env, messages, "challenge_dispute", {"case_id": case_id})
            elif workflow.dispute_stage == DisputeStage.CHARGEBACK_OPEN:
                if case.hidden.true_dispute_should_accept:
                    _call_tool(env, messages, "accept_dispute", {"case_id": case_id})
                elif not workflow.dispute_evidence_fields:
                    _call_tool(env, messages, "submit_dispute_evidence", {
                        "case_id": case_id,
                        "evidence_fields": ["customer_communication", "tracking_number", "delivery_confirmation"],
                    })
                else:
                    _call_tool(env, messages, "challenge_dispute", {"case_id": case_id})
            elif workflow.dispute_stage == DisputeStage.PRE_ARBITRATION:
                decision = "accept" if case.hidden.true_dispute_should_accept else "contest"
                _call_tool(env, messages, "resolve_prearbitration", {
                    "case_id": case_id, "prearbitration_decision": decision,
                })
            elif workflow.dispute_stage == DisputeStage.LOST:
                _call_tool(env, messages, "accept_dispute", {"case_id": case_id})
            elif case.hidden.true_fraud_risk > 0.7:
                _call_tool(env, messages, "reject", {"case_id": case_id, "reason_code": "suspicious_pattern"})
            else:
                _call_tool(env, messages, "approve", {"case_id": case_id})

    # 5. Advance clock for pending events (dispute outcomes, etc.)
    for _ in range(3):  # Up to 3 clock advances
        if not env._env._state or not env._env._state.scheduled_events:
            break
        case_events = [e for e in env._env._state.scheduled_events if e.case_id == case_id]
        if not case_events:
            break
        next_time = min(e.at_time for e in case_events)
        minutes = max(1, next_time - env._env._state.current_time)
        _call_tool(env, messages, "advance_clock", {"minutes": minutes})

    # 6. Notification
    c = env._env._state.cases[case_id]
    if c.requires_customer_notification and not c.customer_notified and c.resolution.value != "pending":
        template = "refund_approved" if c.resolution.value == "approved" else "case_closed"
        _call_tool(env, messages, "send_message", {"case_id": case_id, "template_id": template})

    # 7. QA
    c = env._env._state.cases[case_id]
    if c.qa_required and getattr(c.qa_status, "value", c.qa_status) != "passed":
        _call_tool(env, messages, "send_to_qa", {"case_id": case_id})
        _call_tool(env, messages, "approve_qa", {"case_id": case_id})

    # 8. Close case
    c = env._env._state.cases[case_id]
    if c.resolution.value != "pending":
        _call_tool(env, messages, "close_case", {"case_id": case_id})

    if not env.done:
        return None

    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", default="refund_exception")
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--output", default="artifacts/sft-warmstart.jsonl")
    args = parser.parse_args()

    rng = random.Random(42)
    seeds = [rng.randint(0, 1_000_000) for _ in range(args.num_seeds)]

    successes = 0
    with open(args.output, "w") as f:
        for seed in seeds:
            try:
                messages = _run_oracle_trajectory(seed)
            except Exception as e:
                print(f"  seed={seed}: error={e}")
                continue
            if messages is None:
                continue
            f.write(json.dumps({"messages": messages}) + "\n")
            successes += 1

    print(f"Generated {successes}/{len(seeds)} oracle trajectories → {args.output}")


if __name__ == "__main__":
    main()
