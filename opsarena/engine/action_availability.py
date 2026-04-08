from __future__ import annotations

from opsarena.domain.core import QAStatus
from opsarena.domain.workflows.invoice import (
    CreditMemoStatus,
    InvoiceWorkflowState,
    POChangeStatus,
    PaymentBatchStatus,
    VendorResponseStatus,
)
from opsarena.domain.workflows.kyc import BeneficialOwnerStatus, EDDStatus, OFACReportStatus, SanctionsStatus
from opsarena.domain.workflows.refund import DisputeStage, PreDisputeType, RefundWorkflowState
from opsarena.engine.state import CaseState, WorldState


QUEUE_ACTIONS = [
    "list_queue",
    "search_cases",
    "batch_reorder",
    "bulk_assign",
    "bulk_route",
    "rebalance_queue",
    "advance_clock",
]
COMMON_CASE_ACTIONS = [
    "inspect_audit",
    "log_internal_note",
    "prioritize",
    "schedule_follow_up",
]


def available_actions_for_case(case: CaseState) -> list[str]:
    actions = set(COMMON_CASE_ACTIONS)
    if case.claimed_by is None:
        actions.add("claim_case")
    else:
        actions.add("return_to_queue")
    if case.sla_paused_at is None:
        actions.add("pause_sla")
    else:
        actions.add("resume_sla")
    actions.update({"open_case", "view_record", "query_policy", "assign", "route_case", "request_info", "send_message"})
    if case.qa_status == QAStatus.PENDING:
        actions.update({"approve_qa", "fail_qa"})
    elif case.resolution.value == "pending":
        actions.update({"approve", "reject", "escalate", "defer"})
    else:
        if case.qa_required and case.qa_status != QAStatus.PASSED:
            actions.add("send_to_qa")
        else:
            actions.add("close_case")

    if case.qa_status != QAStatus.PENDING:
        if isinstance(case.workflow, RefundWorkflowState):
            actions.add("execute_refund")
            if case.workflow.pre_dispute_type != PreDisputeType.NONE and case.workflow.dispute_stage == DisputeStage.INQUIRY:
                actions.update({"refund_pre_dispute_alert", "challenge_dispute"})
            if case.workflow.dispute_stage in {DisputeStage.CHARGEBACK_OPEN, DisputeStage.PRE_ARBITRATION}:
                actions.update({"accept_dispute", "submit_dispute_evidence"})
            if case.workflow.dispute_stage == DisputeStage.CHARGEBACK_OPEN:
                actions.add("challenge_dispute")
            if case.workflow.dispute_stage == DisputeStage.PRE_ARBITRATION:
                actions.add("resolve_prearbitration")
            if case.workflow.payout_frozen:
                actions.add("unfreeze_payouts")
            else:
                actions.add("freeze_payouts")
            actions.add("set_reserve_percent")
            actions.add("set_payout_delay_days")
            if case.workflow.reserve_percent > 0:
                actions.add("clear_reserve")
        elif isinstance(case.workflow, InvoiceWorkflowState):
            actions.add("record_three_way_match")
            if case.workflow.payment_hold:
                actions.add("release_payment_hold")
            else:
                actions.add("place_payment_hold")
            if case.workflow.credit_memo_status != CreditMemoStatus.RECEIVED:
                actions.add("request_credit_memo")
            if case.workflow.secondary_approval_required and case.workflow.approval_status.value != "pending_secondary":
                actions.add("send_for_secondary_approval")
            if case.workflow.vendor_response_status != VendorResponseStatus.AWAITING:
                actions.add("request_revised_invoice")
            if case.workflow.po_change_status == POChangeStatus.NOT_REQUESTED:
                actions.add("request_po_change")
            if case.workflow.payment_batch_status in {PaymentBatchStatus.SCHEDULED, PaymentBatchStatus.IN_PROGRESS}:
                actions.add("remove_from_payment_batch")
            if (
                case.workflow.payment_batch_status == PaymentBatchStatus.IN_PROGRESS
                and case.workflow.stop_payment_window_until is not None
            ):
                actions.add("stop_payment")
            actions.add("record_vendor_refund")
            if case.workflow.credit_memo_status == CreditMemoStatus.RECEIVED:
                actions.add("apply_credit_memo")
            remaining = abs(case.workflow.variance_amount or case.amount)
            if remaining <= case.workflow.write_off_threshold:
                actions.add("write_off_small_balance")
        else:
            actions.update({"review_kyc", "trigger_reverification"})
            if case.workflow.sanctions_status != SanctionsStatus.CONFIRMED_MATCH:
                actions.add("run_sanctions_screen")
            if case.workflow.sanctions_status in {SanctionsStatus.POTENTIAL_MATCH, SanctionsStatus.CONFIRMED_MATCH}:
                actions.add("freeze_payments")
            if case.workflow.edd_status in {EDDStatus.NOT_STARTED, EDDStatus.AWAITING_RESPONSE, EDDStatus.IN_PROGRESS}:
                actions.add("start_edd_review")
            if case.workflow.beneficial_owner_status in {
                BeneficialOwnerStatus.PENDING_REVIEW,
                BeneficialOwnerStatus.NEEDS_CORRECTION,
            }:
                actions.add("review_beneficial_owner")
            if case.workflow.correction_fields:
                actions.add("request_field_correction")
            if case.workflow.ofac_report_status in {OFACReportStatus.PENDING, OFACReportStatus.MISSED}:
                actions.add("file_ofac_report")

    return sorted(actions)


def available_actions(state: WorldState) -> list[str]:
    if state.current_case_id and state.current_case_id in state.cases:
        current_case = state.cases[state.current_case_id]
        if current_case.status != "closed":
            return sorted(set(QUEUE_ACTIONS + available_actions_for_case(current_case)))
    return QUEUE_ACTIONS + ["open_case"]
