from __future__ import annotations

from opsarena.domain.case import CaseState
from opsarena.domain.workflows.invoice import InvoiceWorkflowState
from opsarena.domain.workflows.kyc import KYCWorkflowState
from opsarena.domain.workflows.refund import RefundWorkflowState
from opsarena.engine.state import WorldState
from opsarena.enums import CaseType, RecordType, Resolution, SortField
from opsarena.models import AdvanceClockAction, OpsAction


def require_case(state: WorldState, case_id: str | None) -> CaseState:
    if case_id is None or case_id not in state.cases:
        raise ValueError("unknown_case")
    return state.cases[case_id]


def tool_time_cost(action: OpsAction) -> int:
    if isinstance(action, AdvanceClockAction):
        return action.minutes
    return 1


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
    if case.next_touch_at is not None and current_time >= case.next_touch_at:
        case.next_touch_at = None
        case.waiting_reason = None
        if case.status in {"waiting_follow_up", "waiting_external"}:
            case.status = "in_progress"


def claimed_case_count(state: WorldState) -> int:
    return sum(1 for case in state.cases.values() if case.status != "closed" and case.claimed_by is not None)


def expected_resolution(case: CaseState) -> Resolution:
    if case.case_type == CaseType.REFUND:
        return Resolution.REJECTED if case.hidden.true_fraud_risk > 0.7 else Resolution.APPROVED
    if case.case_type == CaseType.INVOICE:
        return Resolution.REJECTED if case.hidden.true_is_duplicate else Resolution.APPROVED
    if case.case_type == CaseType.KYC:
        workflow = require_kyc_workflow(case)
        if not case.hidden.true_doc_valid:
            return Resolution.REJECTED
        if not workflow.kyc_complete:
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
    if case.requires_customer_notification and not case.customer_notified:
        return False, "customer notification missing"
    if case.case_type == CaseType.KYC and case.resolution == Resolution.APPROVED and not require_kyc_workflow(case).kyc_complete:
        return False, "kyc incomplete"
    if case.resolution == Resolution.APPROVED:
        secondary_required = False
        approval_status = None
        if isinstance(case.workflow, RefundWorkflowState):
            secondary_required = case.workflow.secondary_approval_required
            approval_status = case.workflow.approval_status
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


def require_invoice_workflow(case: CaseState) -> InvoiceWorkflowState:
    if not isinstance(case.workflow, InvoiceWorkflowState):
        raise ValueError("invoice_workflow_not_available")
    return case.workflow


def require_kyc_workflow(case: CaseState) -> KYCWorkflowState:
    if not isinstance(case.workflow, KYCWorkflowState):
        raise ValueError("kyc_workflow_not_available")
    return case.workflow
