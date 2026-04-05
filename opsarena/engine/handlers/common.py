from __future__ import annotations

from opsarena.domain.case import CaseState
from opsarena.domain.core import QAStatus
from opsarena.domain.workflows.invoice import InvoiceWorkflowState
from opsarena.domain.workflows.kyc import (
    BeneficialOwnerStatus,
    EDDStatus,
    KYCWorkflowState,
    OFACReportStatus,
    SanctionsStatus,
)
from opsarena.domain.workflows.refund import DisputeResolution, MonitoringProgramStatus, PreDisputeType, RefundWorkflowState
from opsarena.engine.state import WorldState
from opsarena.enums import CaseType, RecordType, Resolution, SortField
from opsarena.models import AdvanceClockAction, OpsAction


# Realistic time costs per action type (in simulated minutes).
# Most quick actions take 1 minute; investigation/compliance actions take longer.
_ACTION_TIME_COSTS: dict[str, int] = {
    "run_sanctions_screen": 2,
    "start_edd_review": 2,
    "review_beneficial_owner": 2,
    "file_ofac_report": 2,
    "review_kyc": 2,
    "record_three_way_match": 2,
    "submit_dispute_evidence": 2,
}


def require_case(state: WorldState, case_id: str | None) -> CaseState:
    if case_id is None or case_id not in state.cases:
        raise ValueError("unknown_case")
    return state.cases[case_id]


def tool_time_cost(action: OpsAction) -> int:
    if isinstance(action, AdvanceClockAction):
        return action.minutes
    return _ACTION_TIME_COSTS.get(action.action_type, 1)


def sort_key(case: CaseState, sort_by: SortField) -> tuple:
    if sort_by == SortField.SLA_REMAINING:
        return (case.sla_deadline, case.priority)
    if sort_by == SortField.CREATED_AT:
        return (case.created_at, case.priority)
    if sort_by == SortField.AMOUNT:
        return (-case.amount, case.priority)
    return (case.priority, case.sla_deadline)


def record_evidence(case: CaseState, record_type: RecordType) -> None:
    evidence_map = {
        RecordType.ORDER: ("order_history", "review_order"),
        RecordType.CUSTOMER: ("customer_profile", "review_customer"),
        RecordType.SHIPPING: ("shipping_status", "review_shipping"),
        RecordType.PAYMENT: ("payment_history", "review_payment"),
        RecordType.DISPUTE: ("fraud_signals", "review_dispute"),
        RecordType.INVOICE: ("invoice_data", "review_invoice"),
        RecordType.PURCHASE_ORDER: ("purchase_order", "review_po"),
        RecordType.RECEIPT: ("goods_receipt", "review_receipt"),
        RecordType.KYC_DOCUMENT: ("kyc_documents", "review_document"),
    }
    evidence = evidence_map.get(record_type)
    if evidence is None:
        return
    evidence_type, check_name = evidence
    case.gather_evidence(evidence_type)
    if check_name in case.required_check_names:
        case.mark_check(check_name)


def linked_record_id(case: CaseState, record_type: RecordType) -> str | None:
    for record in case.linked_records:
        if record.record_type == record_type:
            return record.record_id
    return None


def touch_case(case: CaseState, current_time: int) -> None:
    case.last_touched_at = current_time
    if case.follow_up_overdue:
        case.follow_up_overdue = False
        case.visible_flags = [flag for flag in case.visible_flags if flag != "follow_up_overdue"]
    if case.qa_rework_overdue:
        case.qa_rework_overdue = False
        case.rework_due_at = None
        case.visible_flags = [flag for flag in case.visible_flags if flag != "qa_rework_overdue"]
    if case.next_touch_at is not None and current_time >= case.next_touch_at:
        case.next_touch_at = None
        case.waiting_reason = None
        if case.status in {"waiting_follow_up", "waiting_external"}:
            case.status = "in_progress"


def claimed_case_count(state: WorldState) -> int:
    return sum(1 for case in state.cases.values() if case.status != "closed" and case.claimed_by is not None)


def expected_resolution(case: CaseState) -> Resolution:
    if case.case_type == CaseType.REFUND:
        workflow = require_refund_workflow(case)
        if workflow.pre_dispute_type != PreDisputeType.NONE:
            return Resolution.APPROVED if case.hidden.true_dispute_should_accept else Resolution.REJECTED
        return Resolution.REJECTED if case.hidden.true_fraud_risk > 0.7 else Resolution.APPROVED
    if case.case_type == CaseType.INVOICE:
        return Resolution.REJECTED if case.hidden.true_is_duplicate else Resolution.APPROVED
    if case.case_type == CaseType.KYC:
        workflow = require_kyc_workflow(case)
        if case.hidden.true_sanctions_match:
            return Resolution.REJECTED
        if not case.hidden.true_doc_valid:
            return Resolution.REJECTED
        if not workflow.kyc_complete or not workflow.approval_ready():
            return Resolution.DEFERRED
        return Resolution.APPROVED
    return Resolution.PENDING


def can_close(case: CaseState) -> tuple[bool, str]:
    if case.resolution == Resolution.PENDING:
        return False, "case has no resolution"
    if case.next_touch_at is not None:
        return False, "follow up still scheduled"
    if case.follow_up_overdue:
        return False, "follow up overdue"
    if case.qa_status == QAStatus.PENDING:
        return False, "qa review pending"
    if case.qa_required and case.qa_status != QAStatus.PASSED:
        return False, "qa approval missing"
    if case.requires_customer_notification and not case.customer_notified:
        return False, "customer notification missing"
    if case.case_type == CaseType.KYC:
        workflow = require_kyc_workflow(case)
        if case.hidden.true_ofac_report_required and workflow.ofac_report_status != OFACReportStatus.FILED:
            return False, "ofac report pending"
        if workflow.sanctions_status == SanctionsStatus.POTENTIAL_MATCH:
            return False, "sanctions review pending"
        if workflow.edd_status in {EDDStatus.IN_PROGRESS, EDDStatus.AWAITING_RESPONSE}:
            return False, "edd review pending"
        if workflow.beneficial_owner_status in {BeneficialOwnerStatus.PENDING_REVIEW, BeneficialOwnerStatus.NEEDS_CORRECTION}:
            return False, "beneficial owner review pending"
        if case.resolution == Resolution.APPROVED and not workflow.kyc_complete:
            return False, "kyc incomplete"
        if case.resolution == Resolution.APPROVED and not workflow.approval_ready():
            return False, "compliance review incomplete"
        if case.resolution == Resolution.REJECTED and case.hidden.true_sanctions_match and not workflow.payments_frozen:
            return False, "payments not frozen"
    if case.resolution == Resolution.APPROVED:
        secondary_required = False
        approval_status = None
        if isinstance(case.workflow, RefundWorkflowState):
            secondary_required = case.workflow.secondary_approval_required
            approval_status = case.workflow.approval_status
            if (
                case.workflow.monitoring_program_status == MonitoringProgramStatus.BREACHED
                and not case.workflow.payout_frozen
                and case.workflow.reserve_percent <= 0
                and case.workflow.payout_delay_days <= 0
            ):
                return False, "risk controls missing"
            if case.workflow.pre_dispute_type != PreDisputeType.NONE and case.workflow.dispute_resolution == DisputeResolution.PENDING:
                return False, "pre dispute unresolved"
        elif isinstance(case.workflow, InvoiceWorkflowState):
            secondary_required = case.workflow.secondary_approval_required
            approval_status = case.workflow.approval_status.value
        if secondary_required and approval_status not in {"approved", "not_requested"}:
            return False, "secondary approval incomplete"
    return True, ""


def require_refund_workflow(case: CaseState) -> RefundWorkflowState:
    if not isinstance(case.workflow, RefundWorkflowState):
        raise ValueError("refund_workflow_not_available")
    return case.workflow


def sync_refund_risk_flags(case: CaseState) -> None:
    if not isinstance(case.workflow, RefundWorkflowState):
        return
    workflow = case.workflow
    workflow.recompute_risk_state()

    flags = [flag for flag in case.visible_flags if flag not in {"payout_frozen", "reserve_active", "payout_delay", "monitoring_breach"}]
    if workflow.payout_frozen:
        flags.append("payout_frozen")
    if workflow.reserve_percent > 0:
        flags.append("reserve_active")
    if workflow.payout_delay_days > 0:
        flags.append("payout_delay")
    if workflow.monitoring_program_status == MonitoringProgramStatus.BREACHED:
        flags.append("monitoring_breach")
    case.visible_flags = list(dict.fromkeys(flags))


def require_invoice_workflow(case: CaseState) -> InvoiceWorkflowState:
    if not isinstance(case.workflow, InvoiceWorkflowState):
        raise ValueError("invoice_workflow_not_available")
    return case.workflow


def require_kyc_workflow(case: CaseState) -> KYCWorkflowState:
    if not isinstance(case.workflow, KYCWorkflowState):
        raise ValueError("kyc_workflow_not_available")
    return case.workflow


def sync_kyc_flags(case: CaseState) -> None:
    if not isinstance(case.workflow, KYCWorkflowState):
        return
    workflow = case.workflow
    workflow.payout_hold = (
        not workflow.kyc_complete
        or workflow.sanctions_status != SanctionsStatus.CLEAR
        or workflow.edd_status in {EDDStatus.IN_PROGRESS, EDDStatus.AWAITING_RESPONSE}
        or workflow.beneficial_owner_status in {BeneficialOwnerStatus.PENDING_REVIEW, BeneficialOwnerStatus.NEEDS_CORRECTION}
        or workflow.payments_frozen
        or workflow.ofac_report_status in {OFACReportStatus.PENDING, OFACReportStatus.MISSED}
    )

    flags = [
        flag
        for flag in case.visible_flags
        if flag
        not in {
            "payout_hold",
            "payments_frozen",
            "sanctions_review",
            "sanctions_match",
            "edd_pending",
            "edd_response_due",
            "ofac_report_due",
            "report_overdue",
        }
    ]
    if workflow.payout_hold:
        flags.append("payout_hold")
    if workflow.payments_frozen:
        flags.append("payments_frozen")
    if workflow.sanctions_status == SanctionsStatus.POTENTIAL_MATCH:
        flags.append("sanctions_review")
    if workflow.sanctions_status == SanctionsStatus.CONFIRMED_MATCH:
        flags.append("sanctions_match")
    if workflow.edd_status in {EDDStatus.IN_PROGRESS, EDDStatus.AWAITING_RESPONSE}:
        flags.append("edd_pending")
    if workflow.report_due_at is not None and workflow.ofac_report_status != OFACReportStatus.FILED:
        flags.append("ofac_report_due")
    if workflow.ofac_report_status == OFACReportStatus.MISSED:
        flags.append("report_overdue")
    case.visible_flags = list(dict.fromkeys(flags))
