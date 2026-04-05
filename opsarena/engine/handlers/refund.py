from __future__ import annotations

from opsarena.domain.events import (
    DisputeOutcomeEvent,
    InquiryEscalatesToChargebackEvent,
    MonitoringThresholdBreachedEvent,
    PrearbitrationReceivedEvent,
    ReserveReleaseDueEvent,
)
from opsarena.domain.workflows.refund import (
    DisputeResolution,
    DisputeStage,
    MonitoringProgramStatus,
    PreDisputeType,
    PrearbitrationDecision,
    RefundExecutionState,
)
from opsarena.engine.handlers.common import linked_record_id, require_case, require_refund_workflow, sync_refund_risk_flags
from opsarena.engine.handlers.result import TransitionResult
from opsarena.engine.scheduler import schedule_event
from opsarena.engine.state import WorldState
from opsarena.enums import CaseType, RecordType, Resolution
from opsarena.models import (
    AcceptDisputeAction,
    ChallengeDisputeAction,
    ClearReserveAction,
    ExecuteRefundAction,
    FreezePayoutsAction,
    RefundPreDisputeAlertAction,
    ResolvePrearbitrationAction,
    SetPayoutDelayDaysAction,
    SetReservePercentAction,
    SubmitDisputeEvidenceAction,
    UnfreezePayoutsAction,
)


def _strong_dispute_packet(case, workflow) -> bool:
    return len(workflow.dispute_evidence_fields) >= 2 or case.evidence_items_gathered >= 3


def _schedule_monitoring_breach_if_needed(state: WorldState, case) -> None:
    workflow = require_refund_workflow(case)
    if workflow.monitoring_program_status != MonitoringProgramStatus.BREACHED:
        return
    if any(
        event.event_type == "monitoring_threshold_breached" and event.case_id == case.case_id
        for event in state.scheduled_events
    ):
        return
    schedule_event(
        state,
        MonitoringThresholdBreachedEvent(at_time=state.current_time + 5, case_id=case.case_id),
    )


def _record_dispute_loss(state: WorldState, case) -> None:
    workflow = require_refund_workflow(case)
    workflow.merchant_dispute_ratio_30d = round(workflow.merchant_dispute_ratio_30d + 0.003, 4)
    if case.hidden.true_fraud_risk > 0.7:
        workflow.merchant_fraud_ratio_30d = round(workflow.merchant_fraud_ratio_30d + 0.002, 4)
    sync_refund_risk_flags(case)
    _schedule_monitoring_breach_if_needed(state, case)


def _record_dispute_win(case) -> None:
    workflow = require_refund_workflow(case)
    workflow.merchant_dispute_ratio_30d = max(0.0, round(workflow.merchant_dispute_ratio_30d - 0.0005, 4))
    sync_refund_risk_flags(case)


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
    sync_refund_risk_flags(case)
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
    workflow.pre_dispute_due_at = None
    workflow.prearbitration_due_at = None
    workflow.representment_due_at = None
    case.resolution = Resolution.APPROVED
    case.status = "resolved"
    state.metrics.disputes_accepted += 1
    _record_dispute_loss(state, case)
    return TransitionResult(True, f"Accepted dispute for {case.case_id}"), case


def handle_refund_pre_dispute_alert(state: WorldState, action: RefundPreDisputeAlertAction):
    case = require_case(state, action.case_id)
    if case.case_type != CaseType.REFUND:
        raise ValueError("dispute_workflow_not_supported")
    workflow = require_refund_workflow(case)
    if workflow.pre_dispute_type == PreDisputeType.NONE or workflow.dispute_stage != DisputeStage.INQUIRY:
        raise ValueError("pre_dispute_alert_not_available")
    payment_id = linked_record_id(case, RecordType.PAYMENT)
    if payment_id is None:
        raise ValueError("payment_record_missing")
    approved_amount = action.approved_amount if action.approved_amount is not None else case.amount
    payment = state.records.payments[payment_id]
    payment.status = "refunded" if approved_amount >= case.amount else "partially_refunded"
    workflow.refund_execution_state = (
        RefundExecutionState.REFUNDED if approved_amount >= case.amount else RefundExecutionState.PARTIALLY_REFUNDED
    )
    workflow.refunded_amount = approved_amount
    workflow.dispute_stage = DisputeStage.FINALIZED
    workflow.dispute_resolution = DisputeResolution.ACCEPTED
    workflow.dispute_workflow_status = "pre_dispute_refunded"
    workflow.pre_dispute_due_at = None
    case.resolution = Resolution.APPROVED
    case.status = "resolved"
    sync_refund_risk_flags(case)
    return TransitionResult(True, f"Resolved pre-dispute alert for {case.case_id}"), case


def handle_challenge_dispute(state: WorldState, action: ChallengeDisputeAction):
    case = require_case(state, action.case_id)
    if case.case_type != CaseType.REFUND:
        raise ValueError("dispute_workflow_not_supported")
    workflow = require_refund_workflow(case)
    if workflow.dispute_stage == DisputeStage.PRE_ARBITRATION:
        raise ValueError("prearbitration_requires_resolution")
    if workflow.dispute_stage == DisputeStage.INQUIRY:
        if workflow.pre_dispute_type == PreDisputeType.NONE:
            raise ValueError("pre_dispute_alert_not_available")
        if case.hidden.true_dispute_should_accept:
            raise ValueError("pre_dispute_should_be_refunded")
        if _strong_dispute_packet(case, workflow):
            workflow.dispute_stage = DisputeStage.FINALIZED
            workflow.dispute_resolution = DisputeResolution.WON
            workflow.dispute_workflow_status = "inquiry_resolved"
            workflow.pre_dispute_due_at = None
            case.resolution = Resolution.REJECTED
            case.status = "resolved"
            _record_dispute_win(case)
            return TransitionResult(True, f"Resolved inquiry without chargeback for {case.case_id}"), case
        schedule_event(
            state,
            InquiryEscalatesToChargebackEvent(
                at_time=workflow.pre_dispute_due_at or state.current_time + 5,
                case_id=case.case_id,
            ),
        )
        workflow.dispute_workflow_status = "inquiry_contested"
        return TransitionResult(True, f"Challenged inquiry for {case.case_id}"), case

    if workflow.dispute_stage != DisputeStage.CHARGEBACK_OPEN:
        raise ValueError("chargeback_not_open")
    if not workflow.dispute_evidence_fields and not _strong_dispute_packet(case, workflow):
        raise ValueError("no_evidence_submitted")
    workflow.dispute_workflow_status = "challenged"
    if case.hidden.true_dispute_should_accept:
        schedule_event(
            state,
            DisputeOutcomeEvent(at_time=state.current_time + 12, case_id=case.case_id, outcome="lost"),
        )
    elif len(workflow.dispute_evidence_fields) >= 3 or case.evidence_items_gathered >= 4:
        schedule_event(
            state,
            DisputeOutcomeEvent(at_time=state.current_time + 12, case_id=case.case_id, outcome="won"),
        )
    else:
        schedule_event(
            state,
            PrearbitrationReceivedEvent(at_time=state.current_time + 15, case_id=case.case_id),
        )
    return TransitionResult(True, f"Challenged dispute for {case.case_id}"), case


def handle_resolve_prearbitration(state: WorldState, action: ResolvePrearbitrationAction):
    case = require_case(state, action.case_id)
    if case.case_type != CaseType.REFUND:
        raise ValueError("dispute_workflow_not_supported")
    workflow = require_refund_workflow(case)
    if workflow.dispute_stage != DisputeStage.PRE_ARBITRATION:
        raise ValueError("prearbitration_not_open")
    workflow.prearbitration_due_at = None
    if action.prearbitration_decision == PrearbitrationDecision.ACCEPT:
        workflow.dispute_stage = DisputeStage.FINALIZED
        workflow.dispute_resolution = DisputeResolution.ACCEPTED
        workflow.dispute_workflow_status = "prearbitration_accepted"
        case.resolution = Resolution.APPROVED
        case.status = "resolved"
        _record_dispute_loss(state, case)
        return TransitionResult(True, f"Accepted pre-arbitration for {case.case_id}"), case

    if not _strong_dispute_packet(case, workflow):
        raise ValueError("insufficient_evidence_for_prearbitration")
    workflow.dispute_workflow_status = "prearbitration_contested"
    outcome = "won" if not case.hidden.true_dispute_should_accept else "lost"
    schedule_event(
        state,
        DisputeOutcomeEvent(at_time=state.current_time + 10, case_id=case.case_id, outcome=outcome),
    )
    return TransitionResult(True, f"Contested pre-arbitration for {case.case_id}"), case


def handle_freeze_payouts(state: WorldState, action: FreezePayoutsAction):
    case = require_case(state, action.case_id)
    workflow = require_refund_workflow(case)
    workflow.payout_frozen = True
    workflow.payout_freeze_reason = action.reason_code.value
    sync_refund_risk_flags(case)
    return TransitionResult(True, f"Froze payouts for {case.case_id}"), case


def handle_unfreeze_payouts(state: WorldState, action: UnfreezePayoutsAction):
    case = require_case(state, action.case_id)
    workflow = require_refund_workflow(case)
    if not workflow.payout_frozen:
        raise ValueError("payouts_not_frozen")
    if workflow.monitoring_program_status == MonitoringProgramStatus.BREACHED:
        raise ValueError("monitoring_breach_still_active")
    workflow.payout_frozen = False
    workflow.payout_freeze_reason = None
    sync_refund_risk_flags(case)
    return TransitionResult(True, f"Unfroze payouts for {case.case_id}"), case


def handle_set_reserve_percent(state: WorldState, action: SetReservePercentAction):
    case = require_case(state, action.case_id)
    workflow = require_refund_workflow(case)
    workflow.reserve_percent = action.reserve_percent
    workflow.reserve_release_due_at = state.current_time + (action.release_after_minutes or 30)
    schedule_event(
        state,
        ReserveReleaseDueEvent(at_time=workflow.reserve_release_due_at, case_id=case.case_id),
    )
    sync_refund_risk_flags(case)
    return TransitionResult(True, f"Set reserve to {action.reserve_percent:.1f}% for {case.case_id}"), case


def handle_clear_reserve(state: WorldState, action: ClearReserveAction):
    case = require_case(state, action.case_id)
    workflow = require_refund_workflow(case)
    if workflow.reserve_percent <= 0:
        raise ValueError("reserve_not_active")
    if workflow.monitoring_program_status == MonitoringProgramStatus.BREACHED:
        raise ValueError("reserve_cannot_clear_during_breach")
    workflow.reserve_percent = 0.0
    workflow.reserve_release_due_at = None
    case.visible_flags = [flag for flag in case.visible_flags if flag != "reserve_release_due"]
    sync_refund_risk_flags(case)
    return TransitionResult(True, f"Cleared reserve for {case.case_id}"), case


def handle_set_payout_delay_days(state: WorldState, action: SetPayoutDelayDaysAction):
    case = require_case(state, action.case_id)
    workflow = require_refund_workflow(case)
    workflow.payout_delay_days = action.payout_delay_days
    sync_refund_risk_flags(case)
    return TransitionResult(True, f"Set payout delay to {action.payout_delay_days} days for {case.case_id}"), case


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
    if outcome == "pre_arbitration":
        schedule_event(
            state,
            PrearbitrationReceivedEvent(at_time=state.current_time + 20, case_id=case.case_id),
        )
    else:
        schedule_event(
            state,
            DisputeOutcomeEvent(at_time=state.current_time + 20, case_id=case.case_id, outcome=outcome),
        )
    return TransitionResult(True, f"Dispute evidence submitted for {case.case_id}"), case
