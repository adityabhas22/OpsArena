from __future__ import annotations

from opsarena.domain.core import ApprovalHistoryEntry
from opsarena.domain.events import VendorCreditMemoReceivedEvent
from opsarena.domain.workflows.invoice import ApprovalStatus, CreditMemoStatus, DuplicateStatus
from opsarena.engine.handlers.common import require_case, require_invoice_workflow
from opsarena.engine.handlers.result import TransitionResult
from opsarena.engine.scheduler import schedule_event
from opsarena.engine.state import WorldState
from opsarena.enums import CaseType
from opsarena.models import (
    PlacePaymentHoldAction,
    RecordThreeWayMatchAction,
    ReleasePaymentHoldAction,
    RequestCreditMemoAction,
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
