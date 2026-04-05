from __future__ import annotations

from opsarena.domain.core import ApprovalHistoryEntry
from opsarena.domain.events import (
    POChangeApprovedEvent,
    StopPaymentConfirmedEvent,
    VendorCreditMemoReceivedEvent,
    VendorRevisedInvoiceEvent,
)
from opsarena.domain.workflows.invoice import (
    ApprovalStatus,
    CreditMemoStatus,
    DuplicateStatus,
    PaymentBatchStatus,
    POChangeStatus,
    RecoveryStatus,
    VendorResponseStatus,
)
from opsarena.engine.handlers.common import require_case, require_invoice_workflow
from opsarena.engine.handlers.result import TransitionResult
from opsarena.engine.scheduler import schedule_event
from opsarena.engine.state import WorldState
from opsarena.enums import CaseType, Resolution
from opsarena.models import (
    ApplyCreditMemoAction,
    PlacePaymentHoldAction,
    RecordThreeWayMatchAction,
    RecordVendorRefundAction,
    ReleasePaymentHoldAction,
    RemoveFromPaymentBatchAction,
    RequestCreditMemoAction,
    RequestPOChangeAction,
    RequestRevisedInvoiceAction,
    StopPaymentAction,
    WriteOffSmallBalanceAction,
)


def handle_record_three_way_match(state: WorldState, action: RecordThreeWayMatchAction):
    case = require_case(state, action.case_id)
    if case.case_type != CaseType.INVOICE:
        raise ValueError("three_way_match_not_supported")
    workflow = require_invoice_workflow(case)
    workflow.match_status = action.match_status
    if action.match_status.value == "duplicate":
        workflow.duplicate_status = DuplicateStatus.CONFIRMED
    elif action.match_status.value == "matched":
        workflow.duplicate_status = DuplicateStatus.FALSE_POSITIVE
    workflow.variance_amount = action.variance_amount
    case.mark_check("review_invoice")
    case.mark_check("review_po")
    if action.match_status.value == "matched":
        case.mark_check("review_receipt")
    if action.match_status.value == "missing_receipt":
        if "goods_receipt" not in case.pending_info_fields:
            case.pending_info_fields.append("goods_receipt")
        case.status = "pending_info"
    return TransitionResult(True, f"Three-way match recorded for {case.case_id}"), case


def handle_place_payment_hold(state: WorldState, action: PlacePaymentHoldAction):
    case = require_case(state, action.case_id)
    if case.case_type != CaseType.INVOICE:
        raise ValueError("payment_hold_not_supported")
    workflow = require_invoice_workflow(case)
    workflow.payment_hold = True
    workflow.payment_hold_reason = action.reason_code.value
    workflow.payment_hold_set_at = state.current_time
    if "payment_hold" not in case.visible_flags:
        case.visible_flags.append("payment_hold")
    return TransitionResult(True, f"Payment hold placed for {case.case_id}"), case


def handle_release_payment_hold(state: WorldState, action: ReleasePaymentHoldAction):
    case = require_case(state, action.case_id)
    if case.case_type != CaseType.INVOICE:
        raise ValueError("payment_hold_not_supported")
    workflow = require_invoice_workflow(case)
    if not workflow.payment_hold:
        raise ValueError("payment_hold_not_active")
    if workflow.credit_memo_status == CreditMemoStatus.REQUESTED:
        raise ValueError("credit_memo_pending")
    workflow.payment_hold = False
    workflow.payment_hold_released_at = state.current_time
    workflow.payment_hold_reason = None
    case.visible_flags = [flag for flag in case.visible_flags if flag != "payment_hold"]
    return TransitionResult(True, f"Payment hold released for {case.case_id}"), case


def handle_request_credit_memo(state: WorldState, action: RequestCreditMemoAction):
    case = require_case(state, action.case_id)
    if case.case_type != CaseType.INVOICE:
        raise ValueError("credit_memo_not_supported")
    workflow = require_invoice_workflow(case)
    if workflow.credit_memo_status == CreditMemoStatus.REQUESTED:
        raise ValueError("credit_memo_already_requested")
    requested_amount = action.approved_amount or workflow.variance_amount or workflow.credit_memo_amount or max(
        1.0, round(case.amount * 0.2, 2)
    )
    workflow.credit_memo_status = CreditMemoStatus.REQUESTED
    workflow.credit_memo_amount = requested_amount
    workflow.payment_hold = True
    workflow.payment_hold_reason = "credit_memo_pending"
    if "payment_hold" not in case.visible_flags:
        case.visible_flags.append("payment_hold")
    state.metrics.credit_memos_requested += 1
    schedule_event(
        state,
        VendorCreditMemoReceivedEvent(
            at_time=state.current_time + (case.hidden.hidden_response_latency_minutes or 30),
            case_id=case.case_id,
            amount=requested_amount,
        ),
    )
    return TransitionResult(True, f"Requested credit memo for {case.case_id}"), case


def apply_secondary_approval_decision(case, state_time: int, outcome: str) -> None:
    workflow = require_invoice_workflow(case)
    workflow.approval_status = ApprovalStatus(outcome)
    workflow.approval_chain.append(
        ApprovalHistoryEntry(
            at_time=state_time,
            status=workflow.approval_status.value,
            owner=workflow.approval_assignee or "manager_review",
        )
    )


def handle_request_revised_invoice(state: WorldState, action: RequestRevisedInvoiceAction):
    case = require_case(state, action.case_id)
    if case.case_type != CaseType.INVOICE:
        raise ValueError("revised_invoice_not_supported")
    workflow = require_invoice_workflow(case)
    if workflow.vendor_response_status == VendorResponseStatus.AWAITING:
        raise ValueError("vendor_response_already_pending")
    workflow.vendor_response_status = VendorResponseStatus.AWAITING
    latency = case.hidden.true_vendor_response_minutes or 30
    revised_amount = int(round((case.amount - (workflow.variance_amount or 0)) * 100))
    schedule_event(
        state,
        VendorRevisedInvoiceEvent(
            at_time=state.current_time + latency,
            case_id=case.case_id,
            revised_amount=revised_amount,
        ),
    )
    return TransitionResult(True, f"Requested revised invoice for {case.case_id}"), case


def handle_request_po_change(state: WorldState, action: RequestPOChangeAction):
    case = require_case(state, action.case_id)
    if case.case_type != CaseType.INVOICE:
        raise ValueError("po_change_not_supported")
    workflow = require_invoice_workflow(case)
    if workflow.po_change_status != POChangeStatus.NOT_REQUESTED:
        raise ValueError("po_change_already_requested")
    workflow.po_change_status = POChangeStatus.PENDING_APPROVAL
    schedule_event(
        state,
        POChangeApprovedEvent(
            at_time=state.current_time + 15,
            case_id=case.case_id,
            approved=case.hidden.true_po_change_approved,
        ),
    )
    return TransitionResult(True, f"Requested PO change for {case.case_id}"), case


def handle_remove_from_payment_batch(state: WorldState, action: RemoveFromPaymentBatchAction):
    case = require_case(state, action.case_id)
    if case.case_type != CaseType.INVOICE:
        raise ValueError("payment_batch_not_supported")
    workflow = require_invoice_workflow(case)
    if workflow.payment_batch_status not in {PaymentBatchStatus.SCHEDULED, PaymentBatchStatus.IN_PROGRESS}:
        raise ValueError("payment_batch_not_removable")
    workflow.payment_batch_status = PaymentBatchStatus.NOT_SCHEDULED
    workflow.payment_batch_id = None
    return TransitionResult(True, f"Removed {case.case_id} from payment batch"), case


def handle_stop_payment(state: WorldState, action: StopPaymentAction):
    case = require_case(state, action.case_id)
    if case.case_type != CaseType.INVOICE:
        raise ValueError("stop_payment_not_supported")
    workflow = require_invoice_workflow(case)
    if workflow.payment_batch_status != PaymentBatchStatus.IN_PROGRESS:
        raise ValueError("payment_batch_not_in_progress")
    if workflow.stop_payment_window_until is not None and state.current_time > workflow.stop_payment_window_until:
        raise ValueError("stop_payment_window_expired")
    schedule_event(
        state,
        StopPaymentConfirmedEvent(
            at_time=state.current_time + 5,
            case_id=case.case_id,
            success=case.hidden.true_stop_payment_success,
        ),
    )
    return TransitionResult(True, f"Stop payment requested for {case.case_id}"), case


def handle_record_vendor_refund(state: WorldState, action: RecordVendorRefundAction):
    case = require_case(state, action.case_id)
    if case.case_type != CaseType.INVOICE:
        raise ValueError("vendor_refund_not_supported")
    workflow = require_invoice_workflow(case)
    refund = action.refund_amount
    recoverable = case.hidden.true_recoverable_amount
    if workflow.recovery_status == RecoveryStatus.NOT_NEEDED:
        workflow.recovery_status = RecoveryStatus.IN_PROGRESS
    if refund >= recoverable and recoverable > 0:
        workflow.recovery_status = RecoveryStatus.COMPLETE
    elif refund > 0:
        workflow.recovery_status = RecoveryStatus.PARTIAL
    return TransitionResult(True, f"Recorded vendor refund of {refund} for {case.case_id}"), case


def handle_apply_credit_memo(state: WorldState, action: ApplyCreditMemoAction):
    case = require_case(state, action.case_id)
    if case.case_type != CaseType.INVOICE:
        raise ValueError("credit_memo_not_supported")
    workflow = require_invoice_workflow(case)
    if workflow.credit_memo_status != CreditMemoStatus.RECEIVED:
        raise ValueError("credit_memo_not_received")
    memo = state.records.credit_memos.get(action.credit_memo_id)
    if memo is None:
        raise ValueError("credit_memo_not_found")
    workflow.credit_memo_status = CreditMemoStatus.APPLIED
    if workflow.recovery_status in {RecoveryStatus.NOT_NEEDED, RecoveryStatus.IN_PROGRESS}:
        workflow.recovery_status = RecoveryStatus.COMPLETE
    return TransitionResult(True, f"Applied credit memo {action.credit_memo_id} for {case.case_id}"), case


def handle_write_off_small_balance(state: WorldState, action: WriteOffSmallBalanceAction):
    case = require_case(state, action.case_id)
    if case.case_type != CaseType.INVOICE:
        raise ValueError("write_off_not_supported")
    workflow = require_invoice_workflow(case)
    remaining = abs(workflow.variance_amount or case.amount)
    if remaining > workflow.write_off_threshold:
        raise ValueError("balance_exceeds_write_off_threshold")
    case.resolution = Resolution.APPROVED
    case.status = "resolved"
    workflow.recovery_status = RecoveryStatus.COMPLETE
    return TransitionResult(True, f"Wrote off small balance for {case.case_id}"), case
