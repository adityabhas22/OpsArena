from __future__ import annotations

from collections.abc import Callable
from typing import Any

from opsarena.engine.handlers.invoice import (
    handle_place_payment_hold,
    handle_record_three_way_match,
    handle_release_payment_hold,
    handle_request_credit_memo,
)
from opsarena.engine.handlers.kyc import handle_review_kyc, handle_trigger_reverification
from opsarena.engine.handlers.refund import (
    handle_accept_dispute,
    handle_execute_refund,
    handle_submit_dispute_evidence,
)
from opsarena.engine.handlers.result import TransitionResult
from opsarena.engine.handlers.shared import (
    handle_advance_clock,
    handle_approve,
    handle_assign,
    handle_batch_reorder,
    handle_claim_case,
    handle_close_case,
    handle_defer,
    handle_escalate,
    handle_inspect_audit,
    handle_list_queue,
    handle_log_internal_note,
    handle_open_case,
    handle_pause_sla,
    handle_prioritize,
    handle_query_policy,
    handle_reject,
    handle_reopen_case,
    handle_request_info,
    handle_resume_sla,
    handle_return_to_queue,
    handle_route_case,
    handle_schedule_follow_up,
    handle_search_cases,
    handle_send_for_secondary_approval,
    handle_send_message,
    handle_view_record,
)
from opsarena.engine.state import WorldState
from opsarena.models import (
    AcceptDisputeAction,
    AdvanceClockAction,
    ApproveAction,
    AssignAction,
    BatchReorderAction,
    ClaimCaseAction,
    CloseCaseAction,
    DeferAction,
    EscalateAction,
    ExecuteRefundAction,
    InspectAuditAction,
    ListQueueAction,
    LogInternalNoteAction,
    OpenCaseAction,
    OpsAction,
    PauseSLAAction,
    PlacePaymentHoldAction,
    PrioritizeAction,
    QueryPolicyAction,
    RecordThreeWayMatchAction,
    RejectAction,
    ReleasePaymentHoldAction,
    ReopenCaseAction,
    RequestCreditMemoAction,
    RequestInfoAction,
    ResumeSLAAction,
    ReviewKYCAction,
    ReturnToQueueAction,
    RouteCaseAction,
    ScheduleFollowUpAction,
    SearchCasesAction,
    SendForSecondaryApprovalAction,
    SendMessageAction,
    SubmitDisputeEvidenceAction,
    TriggerReverificationAction,
    ViewRecordAction,
)

ActionHandler = Callable[[WorldState, Any], tuple[TransitionResult, Any]]

ACTION_HANDLERS: dict[type, ActionHandler] = {
    ListQueueAction: handle_list_queue,
    SearchCasesAction: handle_search_cases,
    OpenCaseAction: handle_open_case,
    ViewRecordAction: handle_view_record,
    QueryPolicyAction: handle_query_policy,
    InspectAuditAction: handle_inspect_audit,
    LogInternalNoteAction: handle_log_internal_note,
    SendMessageAction: handle_send_message,
    RequestInfoAction: handle_request_info,
    AssignAction: handle_assign,
    ClaimCaseAction: handle_claim_case,
    ReturnToQueueAction: handle_return_to_queue,
    RouteCaseAction: handle_route_case,
    PrioritizeAction: handle_prioritize,
    BatchReorderAction: handle_batch_reorder,
    ScheduleFollowUpAction: handle_schedule_follow_up,
    PauseSLAAction: handle_pause_sla,
    ResumeSLAAction: handle_resume_sla,
    EscalateAction: handle_escalate,
    DeferAction: handle_defer,
    ExecuteRefundAction: handle_execute_refund,
    AcceptDisputeAction: handle_accept_dispute,
    SubmitDisputeEvidenceAction: handle_submit_dispute_evidence,
    RecordThreeWayMatchAction: handle_record_three_way_match,
    PlacePaymentHoldAction: handle_place_payment_hold,
    ReleasePaymentHoldAction: handle_release_payment_hold,
    RequestCreditMemoAction: handle_request_credit_memo,
    SendForSecondaryApprovalAction: handle_send_for_secondary_approval,
    ReviewKYCAction: handle_review_kyc,
    TriggerReverificationAction: handle_trigger_reverification,
    ApproveAction: handle_approve,
    RejectAction: handle_reject,
    CloseCaseAction: handle_close_case,
    ReopenCaseAction: handle_reopen_case,
    AdvanceClockAction: handle_advance_clock,
}


def dispatch_action(state: WorldState, action: OpsAction) -> tuple[TransitionResult, Any]:
    handler = ACTION_HANDLERS.get(type(action))
    if handler is None:
        raise ValueError("unsupported_action")
    return handler(state, action)
