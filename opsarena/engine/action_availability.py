from __future__ import annotations

from opsarena.domain.core import QAStatus
from opsarena.domain.workflows.invoice import CreditMemoStatus, InvoiceWorkflowState
from opsarena.domain.workflows.refund import DisputeStage, RefundWorkflowState
from opsarena.engine.state import CaseState, WorldState


QUEUE_ACTIONS = ["list_queue", "search_cases", "batch_reorder", "advance_clock"]
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
            if case.workflow.dispute_stage in {DisputeStage.INQUIRY, DisputeStage.CHARGEBACK_OPEN, DisputeStage.PRE_ARBITRATION}:
                actions.update({"accept_dispute", "submit_dispute_evidence"})
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
        else:
            actions.update({"review_kyc", "trigger_reverification"})

    return sorted(actions)


def available_actions(state: WorldState) -> list[str]:
    if state.current_case_id and state.current_case_id in state.cases:
        return sorted(set(QUEUE_ACTIONS + available_actions_for_case(state.cases[state.current_case_id])))
    return QUEUE_ACTIONS + ["open_case"]
