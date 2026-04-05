from __future__ import annotations

import json

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


def _run_policy(task_id: str, policy_name: str, seed: int = 7) -> float:
    env = OpsArenaEnvironment()
    env.reset(task_id=task_id, seed=seed)
    max_steps = env._state.metadata.get("max_steps", 20)
    for _ in range(max_steps):
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
    return env.state.objective_score


def main() -> None:
    policies = ["always_approve", "always_escalate", "always_request_info", "greedy_closer"]
    report: dict[str, dict] = {}
    for task_id in ("refund_exception", "invoice_plus_kyc", "queue_triage"):
        oracle = run_oracle(task_id)["objective_score"]
        scores = {name: _run_policy(task_id, name) for name in policies}
        threshold = oracle * 0.5
        gaps = {name: round(score - oracle, 4) for name, score in scores.items()}
        report[task_id] = {
            "oracle_score": oracle,
            "policy_scores": scores,
            "guardrail_threshold": threshold,
            "gap_to_oracle": gaps,
            "passes_guardrail": all(score < threshold for score in scores.values()),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
