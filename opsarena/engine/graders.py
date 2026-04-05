from __future__ import annotations

from math import comb

from opsarena.domain.workflows.invoice import CreditMemoStatus, InvoiceWorkflowState, PaymentBatchStatus
from opsarena.domain.workflows.kyc import KYCWorkflowState, OFACReportStatus, SanctionsStatus
from opsarena.domain.workflows.refund import DisputeResolution, DisputeStage, RefundWorkflowState
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
        workflow = case.workflow
        assert isinstance(workflow, RefundWorkflowState)
        if workflow.pre_dispute_type.value != "none" or workflow.dispute_stage != DisputeStage.CHARGEBACK_OPEN:
            if case.hidden.true_dispute_should_accept:
                return 1.0 if case.resolution == Resolution.APPROVED and case.status == "closed" else 0.0
            if workflow.dispute_resolution == DisputeResolution.WON or case.resolution == Resolution.REJECTED:
                return 1.0 if case.status == "closed" else 0.5
        if workflow.dispute_resolution == DisputeResolution.ACCEPTED:
            return 1.0 if case.hidden.true_dispute_should_accept and case.status == "closed" else 0.25
        expected = Resolution.REJECTED if case.hidden.true_fraud_risk > 0.7 else Resolution.APPROVED
    elif case.case_type == CaseType.INVOICE:
        workflow = case.workflow
        assert isinstance(workflow, InvoiceWorkflowState)
        if workflow.credit_memo_status in {CreditMemoStatus.RECEIVED, CreditMemoStatus.APPLIED}:
            return 1.0 if case.resolution == Resolution.APPROVED and case.status == "closed" else 0.0
        expected = Resolution.REJECTED if case.hidden.true_is_duplicate else Resolution.APPROVED
    else:
        workflow = case.workflow
        assert isinstance(workflow, KYCWorkflowState)
        if case.hidden.true_sanctions_match:
            expected = Resolution.REJECTED
        elif not case.hidden.true_doc_valid:
            expected = Resolution.REJECTED
        elif not workflow.kyc_complete:
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
        "qa_before_close": 1.0,
        "qa_rework_discipline": 1.0,
        "refund_risk_controls": 1.0,
        "sanctions_before_approve": 1.0,
        "ofac_report_before_close": 1.0,
        "freeze_on_confirmed_match": 1.0,
        "stop_payment_discipline": 1.0,
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
        info_requests = [f for f in case.requested_info_fields if not f.startswith("owners.")]
        if info_requests and len(info_requests) > 2:
            checks["request_info_limit"] = 0.0
        if (
            isinstance(case.workflow, InvoiceWorkflowState)
            and case.workflow.secondary_approval_required
            and case.resolution == Resolution.APPROVED
            and "send_for_secondary_approval" not in events
        ):
            checks["secondary_approval_before_approve"] = 0.0
        if case.follow_up_overdue or state.metrics.follow_ups_overdue > 0:
            checks["follow_up_discipline"] = 0.0
        if case.qa_required and case.status == "closed" and "approve_qa" not in events:
            checks["qa_before_close"] = 0.0
        if case.qa_rework_overdue or state.metrics.qa_rework_overdue > 0:
            checks["qa_rework_discipline"] = 0.0
        if (
            isinstance(case.workflow, RefundWorkflowState)
            and case.workflow.monitoring_program_status.value == "breached"
            and case.status == "closed"
            and not any(evt in {"freeze_payouts", "set_reserve_percent", "set_payout_delay_days"} for evt in events)
        ):
            checks["refund_risk_controls"] = 0.0
        if (
            isinstance(case.workflow, KYCWorkflowState)
            and case.resolution == Resolution.APPROVED
            and case.workflow.sanctions_status != SanctionsStatus.CLEAR
        ):
            checks["sanctions_before_approve"] = 0.0
        if (
            isinstance(case.workflow, KYCWorkflowState)
            and case.hidden.true_ofac_report_required
            and case.resolution == Resolution.REJECTED
            and "file_ofac_report" not in events
        ):
            checks["ofac_report_before_close"] = 0.0
        if (
            isinstance(case.workflow, KYCWorkflowState)
            and case.hidden.true_sanctions_match
            and case.resolution == Resolution.REJECTED
            and "freeze_payments" not in events
        ):
            checks["freeze_on_confirmed_match"] = 0.0
        if (
            isinstance(case.workflow, InvoiceWorkflowState)
            and case.workflow.payment_batch_status == PaymentBatchStatus.COMPLETED
            and case.hidden.true_is_duplicate
            and not any(evt in {"stop_payment", "remove_from_payment_batch"} for evt in events)
        ):
            checks["stop_payment_discipline"] = 0.0
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
    qa_score = max(0.0, 1.0 - (state.metrics.qa_reviews_failed + state.metrics.qa_rework_overdue) / max(1, len(cases) * 2))
    assignment_score = max(0.0, 1.0 - state.queue_state().unassigned_count / max(1, len(cases)))
    if state.task_id == TaskId.QUEUE_TRIAGE:
        optimal_order = [case.case_id for case in sorted(cases, key=lambda item: (item.priority, item.sla_deadline))]
        tau = max(0.0, _kendall_tau([entry.case_id for entry in state.audit_log if entry.case_id], optimal_order))
        return (sla_score + step_score + tau + follow_up_score + qa_score + assignment_score) / 6
    return (sla_score + step_score + follow_up_score + qa_score + assignment_score) / 5


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
