from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.oracle import run_oracle
from server.environment import OpsArenaEnvironment
from opsarena.models import (
    AdvanceClockAction,
    ApproveAction,
    CloseCaseAction,
    EscalateAction,
    OpenCaseAction,
    RejectAction,
    RequestInfoAction,
    SendMessageAction,
)

TASK_IDS = ("refund_exception", "invoice_plus_kyc", "queue_triage", "ap_payment_run")
EVAL_SEEDS = tuple(range(1, 9))


def _run_policy(task_id: str, policy_name: str, seed: int = 7) -> float:
    env = OpsArenaEnvironment()
    env.reset(task_id=task_id, seed=seed)
    max_steps = env._state.metadata.get("max_steps", 20)
    for _ in range(max_steps):
        if env._is_done():
            break
        queue = [case for case in env._state.cases.values() if case.status != "closed"]
        if not queue:
            break
        case = sorted(queue, key=lambda item: (item.priority, item.sla_deadline))[0]
        env.step(OpenCaseAction(case_id=case.case_id))
        if policy_name == "always_approve":
            env.step(ApproveAction(case_id=case.case_id))
            if case.requires_customer_notification:
                env.step(SendMessageAction(case_id=case.case_id, template_id="refund_approved", slots={"amount": str(case.amount), "order_id": case.case_id}))
            env.step(CloseCaseAction(case_id=case.case_id, resolution_code="approve"))
        elif policy_name == "always_escalate":
            target = case.allowed_escalation_queues[0].value
            env.step(EscalateAction(case_id=case.case_id, target_queue=target, reason_code="policy_ambiguity"))
            if case.requires_customer_notification:
                env.step(SendMessageAction(case_id=case.case_id, template_id="refund_approved", slots={"amount": str(case.amount), "order_id": case.case_id}))
            env.step(CloseCaseAction(case_id=case.case_id, resolution_code="escalate"))
        elif policy_name == "always_request_info":
            field_name = case.pending_info_fields[0] if case.pending_info_fields else "goods_receipt"
            env.step(RequestInfoAction(case_id=case.case_id, field_name=field_name))
            env.step(AdvanceClockAction(minutes=15))
        else:
            env.step(ApproveAction(case_id=case.case_id))
            if case.requires_customer_notification:
                env.step(SendMessageAction(case_id=case.case_id, template_id="refund_approved", slots={"amount": str(case.amount), "order_id": case.case_id}))
            env.step(CloseCaseAction(case_id=case.case_id, resolution_code="greedy"))
    if not env._state.metadata.get("episode_scored") and env._is_done():
        env._state.grader_breakdown = env._state.grader_breakdown or {}
    return env.state.benchmark_score or env.state.objective_score


def main() -> None:
    policies = ["always_approve", "always_escalate", "always_request_info", "greedy_closer"]
    report: dict[str, dict] = {}
    for task_id in TASK_IDS:
        oracle_scores = [run_oracle(task_id, seed=seed)["benchmark_score"] for seed in EVAL_SEEDS]
        oracle_mean = sum(oracle_scores) / len(oracle_scores)
        scores = {
            name: [_run_policy(task_id, name, seed=seed) for seed in EVAL_SEEDS]
            for name in policies
        }
        policy_means = {
            name: sum(values) / len(values)
            for name, values in scores.items()
        }
        threshold = oracle_mean * 0.8
        gaps = {name: round(policy_means[name] - oracle_mean, 4) for name in policies}
        report[task_id] = {
            "eval_seeds": list(EVAL_SEEDS),
            "oracle_scores": oracle_scores,
            "oracle_mean": round(oracle_mean, 6),
            "policy_scores": scores,
            "policy_means": {name: round(value, 6) for name, value in policy_means.items()},
            "guardrail_threshold": round(threshold, 6),
            "gap_to_oracle": gaps,
            "passes_guardrail": all(mean < threshold for mean in policy_means.values()),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
