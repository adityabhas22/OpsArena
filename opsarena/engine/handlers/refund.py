from __future__ import annotations

from opsarena.domain.events import DisputeOutcomeEvent
from opsarena.domain.workflows.refund import DisputeResolution, DisputeStage, RefundExecutionState
from opsarena.engine.handlers.common import linked_record_id, require_case, require_refund_workflow
from opsarena.engine.handlers.result import TransitionResult
from opsarena.engine.scheduler import schedule_event
from opsarena.engine.state import WorldState
from opsarena.enums import CaseType, RecordType, Resolution
from opsarena.models import AcceptDisputeAction, ExecuteRefundAction, SubmitDisputeEvidenceAction


def handle_execute_refund(state: WorldState, action: ExecuteRefundAction):
    case = require_case(state, action.case_id)
    if case.case_type != CaseType.REFUND:
        raise ValueError("refund_execution_not_supported")
    if case.resolution != Resolution.APPROVED:
        raise ValueError("refund_not_approved")
    payment_id = linked_record_id(case, RecordType.PAYMENT)
    if payment_id is None:
        raise ValueError("payment_record_missing")
    payment = state.records.payments[payment_id]
    approved_amount = action.approved_amount if action.approved_amount is not None else case.amount
    workflow = require_refund_workflow(case)
    payment.status = "refunded" if approved_amount >= case.amount else "partially_refunded"
    workflow.refund_execution_state = (
        RefundExecutionState.REFUNDED if approved_amount >= case.amount else RefundExecutionState.PARTIALLY_REFUNDED
    )
    workflow.refunded_amount = approved_amount
    return TransitionResult(True, f"Refund executed for {case.case_id}"), case


def handle_accept_dispute(state: WorldState, action: AcceptDisputeAction):
    case = require_case(state, action.case_id)
    if case.case_type != CaseType.REFUND:
        raise ValueError("dispute_workflow_not_supported")
    dispute_id = linked_record_id(case, RecordType.DISPUTE)
    if dispute_id is None:
        raise ValueError("dispute_record_missing")
    state.records.disputes[dispute_id].status = "lost"
    workflow = require_refund_workflow(case)
    workflow.dispute_stage = DisputeStage.FINALIZED
    workflow.dispute_workflow_status = "accepted"
    workflow.dispute_resolution = DisputeResolution.ACCEPTED
    case.resolution = Resolution.APPROVED
    case.status = "resolved"
    state.metrics.disputes_accepted += 1
    return TransitionResult(True, f"Accepted dispute for {case.case_id}"), case


def handle_submit_dispute_evidence(state: WorldState, action: SubmitDisputeEvidenceAction):
    case = require_case(state, action.case_id)
    if case.case_type != CaseType.REFUND:
        raise ValueError("dispute_workflow_not_supported")
    dispute_id = linked_record_id(case, RecordType.DISPUTE)
    if dispute_id is None:
        raise ValueError("dispute_record_missing")
    dispute = state.records.disputes[dispute_id]
    if state.current_time > dispute.evidence_due_by:
        raise ValueError("dispute_deadline_passed")
    workflow = require_refund_workflow(case)
    for field in action.evidence_fields:
        if field not in workflow.dispute_evidence_fields:
            workflow.dispute_evidence_fields.append(field)
    dispute.submission_count += 1
    dispute.status = "under_review"
    previous_stage = workflow.dispute_stage
    workflow.dispute_stage = DisputeStage.EVIDENCE_SUBMITTED
    workflow.dispute_workflow_status = "submitted"
    case.gather_evidence("dispute_package")
    case.mark_check("review_dispute")

    outcome = "won"
    if case.hidden.true_fraud_risk > 0.7:
        outcome = "lost"
    elif previous_stage == DisputeStage.PRE_ARBITRATION:
        outcome = "won" if len(workflow.dispute_evidence_fields) >= 3 else "lost"
    elif len(workflow.dispute_evidence_fields) < 2:
        outcome = "pre_arbitration"
    schedule_event(
        state,
        DisputeOutcomeEvent(at_time=state.current_time + 20, case_id=case.case_id, outcome=outcome),
    )
    return TransitionResult(True, f"Dispute evidence submitted for {case.case_id}"), case
