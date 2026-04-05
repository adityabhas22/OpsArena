from __future__ import annotations

from math import comb

from opsarena.engine.state import WorldState
from opsarena.enums import CaseType, Resolution, TaskId


def _hard_gate(state: WorldState) -> float | None:
    if state.metrics.compliance_violations > 0:
        return 0.0
    if state.metrics.data_breach_count > 0:
        return 0.0
    return None


def _case_outcome_score(case) -> float:
    if case.case_type == CaseType.REFUND:
        if case.workflow_data.get("dispute_resolution") == "accepted":
            return 1.0 if case.workflow_data.get("dispute_should_accept") and case.status == "closed" else 0.25
        expected = Resolution.REJECTED if case.true_fraud_risk > 0.7 else Resolution.APPROVED
    elif case.case_type == CaseType.INVOICE:
        if case.workflow_data.get("credit_memo_status") in {"received", "applied"}:
            return 1.0 if case.resolution == Resolution.APPROVED and case.status == "closed" else 0.0
        expected = Resolution.REJECTED if case.true_is_duplicate else Resolution.APPROVED
    else:
        if not case.true_doc_valid:
            expected = Resolution.REJECTED
        elif not case.kyc_complete:
            expected = Resolution.DEFERRED
        else:
            expected = Resolution.APPROVED
    return 1.0 if case.resolution == expected and case.status == "closed" else 0.0


def grade_outcome(state: WorldState) -> float:
    cases = list(state.cases.values())
    if not cases:
        return 0.0
    return sum(_case_outcome_score(case) for case in cases) / len(cases)


def grade_trajectory(state: WorldState) -> float:
    audit = state.audit_log
    checks = {
        "policy_before_approve": 1.0,
        "kyc_doc_before_approve": 1.0,
        "notify_before_close": 1.0,
        "open_before_action": 1.0,
        "request_info_limit": 1.0,
        "secondary_approval_before_approve": 1.0,
        "follow_up_discipline": 1.0,
    }
    for case in state.cases.values():
        events = [entry.action_type for entry in audit if entry.case_id == case.case_id]
        if case.resolution == Resolution.APPROVED and "query_policy" not in events:
            checks["policy_before_approve"] = 0.0
        if case.case_type == CaseType.KYC and case.resolution == Resolution.APPROVED and "view_record" not in events:
            checks["kyc_doc_before_approve"] = 0.0
        if case.status == "closed" and case.requires_customer_notification and "send_message" not in events:
            checks["notify_before_close"] = 0.0
        if any(evt in {"approve", "reject", "escalate", "close_case"} for evt in events) and "open_case" not in events:
            checks["open_before_action"] = 0.0
        if case.requested_info_fields and len(case.requested_info_fields) > 2:
            checks["request_info_limit"] = 0.0
        if (
            case.workflow_data.get("secondary_approval_required")
            and case.resolution == Resolution.APPROVED
            and "send_for_secondary_approval" not in events
        ):
            checks["secondary_approval_before_approve"] = 0.0
        if case.follow_up_overdue or state.metrics.follow_ups_overdue > 0:
            checks["follow_up_discipline"] = 0.0
    return sum(checks.values()) / len(checks)


def _kendall_tau(order: list[str], optimal_order: list[str]) -> float:
    if len(order) < 2 or len(order) != len(optimal_order):
        return 1.0
    concordant = 0
    discordant = 0
    index = {case_id: idx for idx, case_id in enumerate(optimal_order)}
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            concordant += int(index[order[i]] < index[order[j]])
            discordant += int(index[order[i]] > index[order[j]])
    total = comb(len(order), 2)
    return (concordant - discordant) / total if total else 1.0


def grade_efficiency(state: WorldState) -> float:
    cases = list(state.cases.values())
    if not cases:
        return 0.0
    sla_score = sum(1 for case in cases if case.status == "closed" and state.current_time <= case.sla_deadline) / len(cases)
    step_score = max(0.0, 1.0 - max(0, state.step_count - len(cases) * 8) / max(1, len(cases) * 20))
    follow_up_score = max(0.0, 1.0 - state.metrics.follow_ups_overdue / max(1, len(cases)))
    if state.task_id == TaskId.QUEUE_TRIAGE:
        optimal_order = [case.case_id for case in sorted(cases, key=lambda item: (item.priority, item.sla_deadline))]
        tau = max(0.0, _kendall_tau([entry.case_id for entry in state.audit_log if entry.case_id], optimal_order))
        return (sla_score + step_score + tau + follow_up_score) / 4
    return (sla_score + step_score + follow_up_score) / 3


def grade_episode(state: WorldState) -> dict:
    hard_gate = _hard_gate(state)
    if hard_gate is not None:
        return {"score": hard_gate, "outcome": 0.0, "process": 0.0, "efficiency": 0.0}
    outcome = grade_outcome(state)
    process = grade_trajectory(state)
    efficiency = grade_efficiency(state)
    score = 0.5 * outcome + 0.3 * process + 0.2 * efficiency
    return {
        "score": round(score, 6),
        "outcome": round(outcome, 6),
        "process": round(process, 6),
        "efficiency": round(efficiency, 6),
    }
