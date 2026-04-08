from __future__ import annotations

from typing import Annotated, Any, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field, TypeAdapter, model_validator

from .domain.workflows.refund import PrearbitrationDecision
from .enums import (
    DecisionCode,
    MatchStatus,
    Priority,
    ReasonCode,
    RecordType,
    SortField,
    TargetQueue,
    VerificationDecision,
)


class QueueItem(BaseModel):
    case_id: str
    case_type: str
    priority: str
    sla_remaining_minutes: int
    summary: str
    status: str
    current_owner: str | None = None
    active_queue: str | None = None
    amount: float | None = None
    customer_name: str | None = None
    flags: list[str] = Field(default_factory=list)


class LinkedRecordView(BaseModel):
    record_type: str
    record_id: str
    title: str


class MessageSummary(BaseModel):
    timestamp: int
    channel: str
    subject: str
    template_id: str | None = None


class CaseDetail(BaseModel):
    case_id: str
    case_type: str
    priority: str
    status: str
    sla_deadline: int
    created_at: int
    customer_id: str | None = None
    vendor_id: str | None = None
    amount: float | None = None
    currency: str = "USD"
    visible_summary: str
    visible_flags: list[str] = Field(default_factory=list)
    linked_records: list[LinkedRecordView] = Field(default_factory=list)
    required_checks: list[str] = Field(default_factory=list)
    checks_completed: list[str] = Field(default_factory=list)
    communication_log: list[MessageSummary] = Field(default_factory=list)
    internal_notes: list[str] = Field(default_factory=list)
    requested_info_fields: list[str] = Field(default_factory=list)
    current_owner: str = ""
    case_phase: str = ""
    close_blockers: list[str] = Field(default_factory=list)
    waiting_on: list[str] = Field(default_factory=list)
    next_due_minutes: int | None = None
    recommended_action_categories: list[str] = Field(default_factory=list)
    workflow_metadata: dict[str, Any] = Field(default_factory=dict)


class PolicyResult(BaseModel):
    policy_id: str
    clause_id: str | None = None
    title: str
    description: str
    matched_actions: list[str] = Field(default_factory=list)


class ActionResult(BaseModel):
    success: bool = True
    action_type: str = ""
    message: str = ""
    error_code: str | None = None


class AuditEntryView(BaseModel):
    timestamp: int
    case_id: str | None = None
    action_type: str
    message: str


class OpsArenaObservation(Observation):
    queue_view: list[QueueItem] | None = None
    case_detail: CaseDetail | None = None
    record_view: dict[str, Any] | None = None
    policy_result: PolicyResult | None = None
    audit_trail: list[AuditEntryView] | None = None
    clock: int = 0
    escalation_queue_load: str = "light"
    escalation_slots_remaining: int = 0
    system_message: str = ""
    error: str | None = None
    available_actions: list[str] = Field(default_factory=list)


class OpsArenaState(State):
    task_id: str = ""
    scenario_seed: int = 0
    simulated_time: int = 0
    cases_resolved: int = 0
    cases_total: int = 0
    objective_score: float = 0.0
    train_score: float = 0.0
    benchmark_score: float = 0.0
    current_case_id: str | None = None
    grader_breakdown: dict[str, Any] = Field(default_factory=dict)


class RawOpsAction(Action):
    action_type: str
    case_id: str | None = None
    case_ids: list[str] | None = None
    sort_by: SortField | None = None
    filters: dict[str, Any] | None = None
    limit: int | None = None
    record_type: RecordType | None = None
    record_id: str | None = None
    policy_id: str | None = None
    clause_id: str | None = None
    decision_code: DecisionCode | None = None
    approved_amount: float | None = None
    notes: str | None = None
    reason_code: ReasonCode | None = None
    target_queue: TargetQueue | None = None
    priority_override: Priority | None = None
    until_time: int | None = None
    field_name: str | None = None
    template_id: str | None = None
    assignee_type: str | None = None
    slots: dict[str, str] | None = None
    note_code: str | None = None
    note_metadata: dict[str, Any] | None = None
    new_priority: Priority | None = None
    ordering_rule: str | None = None
    rebalance_strategy: str | None = None
    follow_up_at: int | None = None
    minutes: int | None = None
    resolution_code: str | None = None
    evidence_fields: list[str] | None = None
    match_status: MatchStatus | None = None
    variance_amount: float | None = None
    verification_decision: VerificationDecision | None = None
    requirements: list[str] | None = None
    assignee_pool: list[str] | None = None
    max_cases: int | None = None
    reserve_percent: float | None = None
    payout_delay_days: int | None = None
    release_after_minutes: int | None = None
    prearbitration_decision: PrearbitrationDecision | None = None
    change_description: str | None = None
    refund_amount: float | None = None
    credit_memo_id: str | None = None


class ListQueueAction(Action):
    action_type: Literal["list_queue"] = "list_queue"
    sort_by: SortField = SortField.PRIORITY
    filters: dict[str, Any] = Field(default_factory=dict)
    limit: int = Field(default=50, ge=1, le=100)


class OpenCaseAction(Action):
    action_type: Literal["open_case"] = "open_case"
    case_id: str


class ViewRecordAction(Action):
    action_type: Literal["view_record"] = "view_record"
    record_type: RecordType
    record_id: str


class QueryPolicyAction(Action):
    action_type: Literal["query_policy"] = "query_policy"
    policy_id: str
    clause_id: str | None = None


class SearchCasesAction(Action):
    action_type: Literal["search_cases"] = "search_cases"
    filters: dict[str, Any] = Field(default_factory=dict)


class InspectAuditAction(Action):
    action_type: Literal["inspect_audit"] = "inspect_audit"
    case_id: str


class ApproveAction(Action):
    """Approve a case. Requires prior policy query and record review."""
    action_type: Literal["approve"] = "approve"
    case_id: str
    decision_code: DecisionCode = Field(default=DecisionCode.STANDARD_APPROVAL, description="'standard_approval', 'exception_approval', or 'partial_approval'")
    approved_amount: float | None = Field(default=None, description="Required for partial_approval; the approved dollar amount")
    notes: str | None = Field(default=None, description="Justification for the approval decision")

    @model_validator(mode="after")
    def validate_partial(self) -> "ApproveAction":
        if (
            self.decision_code == DecisionCode.PARTIAL_APPROVAL
            and self.approved_amount is None
        ):
            raise ValueError("partial_approval requires approved_amount")
        return self


class RejectAction(Action):
    action_type: Literal["reject"] = "reject"
    case_id: str
    reason_code: ReasonCode
    notes: str | None = None


class EscalateAction(Action):
    action_type: Literal["escalate"] = "escalate"
    case_id: str
    target_queue: TargetQueue
    reason_code: ReasonCode
    priority_override: Priority | None = None


class DeferAction(Action):
    action_type: Literal["defer"] = "defer"
    case_id: str
    until_time: int = Field(..., gt=0)
    reason_code: ReasonCode


class RequestInfoAction(Action):
    action_type: Literal["request_info"] = "request_info"
    case_id: str
    field_name: str
    template_id: str | None = None


class AssignAction(Action):
    action_type: Literal["assign"] = "assign"
    case_id: str
    assignee_type: str


class ClaimCaseAction(Action):
    action_type: Literal["claim_case"] = "claim_case"
    case_id: str
    assignee_type: str = "ops_agent"


class ReturnToQueueAction(Action):
    action_type: Literal["return_to_queue"] = "return_to_queue"
    case_id: str
    reason_code: ReasonCode


class RouteCaseAction(Action):
    action_type: Literal["route_case"] = "route_case"
    case_id: str
    target_queue: TargetQueue
    reason_code: ReasonCode
    assignee_type: str | None = None


class SendMessageAction(Action):
    """Send a templated message to the customer."""
    action_type: Literal["send_message"] = "send_message"
    case_id: str
    template_id: str = Field(description="Template to use: 'refund_approved', 'case_closed', or 'info_request'")
    slots: dict[str, str] = Field(
        default_factory=dict,
        description="Optional template variable overrides. Auto-filled from case data if omitted.",
    )


class LogInternalNoteAction(Action):
    action_type: Literal["log_internal_note"] = "log_internal_note"
    case_id: str
    note_code: str
    note_metadata: dict[str, Any] = Field(default_factory=dict)


class PrioritizeAction(Action):
    action_type: Literal["prioritize"] = "prioritize"
    case_id: str
    new_priority: Priority


class BatchReorderAction(Action):
    action_type: Literal["batch_reorder"] = "batch_reorder"
    ordering_rule: str


class ScheduleFollowUpAction(Action):
    action_type: Literal["schedule_follow_up"] = "schedule_follow_up"
    case_id: str
    follow_up_at: int = Field(..., gt=0)
    reason_code: ReasonCode


class PauseSLAAction(Action):
    action_type: Literal["pause_sla"] = "pause_sla"
    case_id: str
    reason_code: ReasonCode


class ResumeSLAAction(Action):
    action_type: Literal["resume_sla"] = "resume_sla"
    case_id: str


class ExecuteRefundAction(Action):
    """Execute the approved refund payment back to the customer."""
    action_type: Literal["execute_refund"] = "execute_refund"
    case_id: str
    approved_amount: float | None = Field(default=None, gt=0, description="Refund amount in dollars")
    notes: str | None = None


class AcceptDisputeAction(Action):
    action_type: Literal["accept_dispute"] = "accept_dispute"
    case_id: str
    notes: str | None = None


class SubmitDisputeEvidenceAction(Action):
    action_type: Literal["submit_dispute_evidence"] = "submit_dispute_evidence"
    case_id: str
    evidence_fields: list[str] = Field(min_length=1)
    notes: str | None = None


class RecordThreeWayMatchAction(Action):
    action_type: Literal["record_three_way_match"] = "record_three_way_match"
    case_id: str
    match_status: MatchStatus
    variance_amount: float | None = Field(default=None, ge=0)
    notes: str | None = None

    @model_validator(mode="after")
    def validate_variance(self) -> "RecordThreeWayMatchAction":
        if self.match_status == MatchStatus.VARIANCE and self.variance_amount is None:
            raise ValueError("variance requires variance_amount")
        return self


class PlacePaymentHoldAction(Action):
    action_type: Literal["place_payment_hold"] = "place_payment_hold"
    case_id: str
    reason_code: ReasonCode
    notes: str | None = None


class ReleasePaymentHoldAction(Action):
    action_type: Literal["release_payment_hold"] = "release_payment_hold"
    case_id: str
    notes: str | None = None


class RequestCreditMemoAction(Action):
    action_type: Literal["request_credit_memo"] = "request_credit_memo"
    case_id: str
    approved_amount: float | None = Field(default=None, gt=0)
    notes: str | None = None


class SendForSecondaryApprovalAction(Action):
    action_type: Literal["send_for_secondary_approval"] = "send_for_secondary_approval"
    case_id: str
    reason_code: ReasonCode
    assignee_type: str | None = None
    notes: str | None = None


class ReviewKYCAction(Action):
    """Submit KYC verification decision after reviewing documents and running checks."""
    action_type: Literal["review_kyc"] = "review_kyc"
    case_id: str
    verification_decision: VerificationDecision = Field(description="'approve', 'request_resubmission', or 'reject'")
    notes: str | None = Field(default=None, description="Justification for the KYC decision")


class TriggerReverificationAction(Action):
    action_type: Literal["trigger_reverification"] = "trigger_reverification"
    case_id: str
    requirements: list[str] = Field(default_factory=list)
    notes: str | None = None


class RunSanctionsScreenAction(Action):
    action_type: Literal["run_sanctions_screen"] = "run_sanctions_screen"
    case_id: str
    notes: str | None = None


class StartEDDReviewAction(Action):
    action_type: Literal["start_edd_review"] = "start_edd_review"
    case_id: str
    notes: str | None = None


class ReviewBeneficialOwnerAction(Action):
    action_type: Literal["review_beneficial_owner"] = "review_beneficial_owner"
    case_id: str
    verification_decision: VerificationDecision
    notes: str | None = None


class RequestFieldCorrectionAction(Action):
    action_type: Literal["request_field_correction"] = "request_field_correction"
    case_id: str
    field_name: str
    notes: str | None = None


class FileOFACReportAction(Action):
    action_type: Literal["file_ofac_report"] = "file_ofac_report"
    case_id: str
    notes: str | None = None


class FreezePaymentsAction(Action):
    action_type: Literal["freeze_payments"] = "freeze_payments"
    case_id: str
    reason_code: ReasonCode
    notes: str | None = None


class ChallengeDisputeAction(Action):
    action_type: Literal["challenge_dispute"] = "challenge_dispute"
    case_id: str
    notes: str | None = None


class RefundPreDisputeAlertAction(Action):
    action_type: Literal["refund_pre_dispute_alert"] = "refund_pre_dispute_alert"
    case_id: str
    approved_amount: float | None = Field(default=None, gt=0)
    notes: str | None = None


class ResolvePrearbitrationAction(Action):
    action_type: Literal["resolve_prearbitration"] = "resolve_prearbitration"
    case_id: str
    prearbitration_decision: PrearbitrationDecision
    notes: str | None = None


class FreezePayoutsAction(Action):
    action_type: Literal["freeze_payouts"] = "freeze_payouts"
    case_id: str
    reason_code: ReasonCode
    notes: str | None = None


class UnfreezePayoutsAction(Action):
    action_type: Literal["unfreeze_payouts"] = "unfreeze_payouts"
    case_id: str
    notes: str | None = None


class SetReservePercentAction(Action):
    action_type: Literal["set_reserve_percent"] = "set_reserve_percent"
    case_id: str
    reserve_percent: float = Field(..., ge=0, le=100)
    release_after_minutes: int | None = Field(default=None, ge=1, le=1000)
    notes: str | None = None


class ClearReserveAction(Action):
    action_type: Literal["clear_reserve"] = "clear_reserve"
    case_id: str
    notes: str | None = None


class SetPayoutDelayDaysAction(Action):
    action_type: Literal["set_payout_delay_days"] = "set_payout_delay_days"
    case_id: str
    payout_delay_days: int = Field(..., ge=0, le=30)
    notes: str | None = None


class BulkAssignAction(Action):
    action_type: Literal["bulk_assign"] = "bulk_assign"
    case_ids: list[str] = Field(min_length=1, max_length=10)
    assignee_type: str


class BulkRouteAction(Action):
    action_type: Literal["bulk_route"] = "bulk_route"
    case_ids: list[str] = Field(min_length=1, max_length=10)
    target_queue: TargetQueue
    reason_code: ReasonCode
    assignee_type: str | None = None


class RebalanceQueueAction(Action):
    action_type: Literal["rebalance_queue"] = "rebalance_queue"
    assignee_pool: list[str] = Field(min_length=1, max_length=10)
    max_cases: int = Field(default=3, ge=1, le=10)
    rebalance_strategy: str = "sla_priority"

    @model_validator(mode="after")
    def validate_strategy(self) -> "RebalanceQueueAction":
        if self.rebalance_strategy not in {"sla_priority", "oldest", "amount"}:
            raise ValueError("invalid_rebalance_strategy")
        return self


class SendToQAAction(Action):
    action_type: Literal["send_to_qa"] = "send_to_qa"
    case_id: str
    assignee_type: str | None = None
    notes: str | None = None


class ApproveQAAction(Action):
    action_type: Literal["approve_qa"] = "approve_qa"
    case_id: str
    assignee_type: str | None = None
    notes: str | None = None


class FailQAAction(Action):
    action_type: Literal["fail_qa"] = "fail_qa"
    case_id: str
    reason_code: ReasonCode
    assignee_type: str | None = None
    notes: str | None = None


class RequestRevisedInvoiceAction(Action):
    action_type: Literal["request_revised_invoice"] = "request_revised_invoice"
    case_id: str
    reason_code: ReasonCode
    notes: str | None = None


class RequestPOChangeAction(Action):
    action_type: Literal["request_po_change"] = "request_po_change"
    case_id: str
    change_description: str
    notes: str | None = None


class RemoveFromPaymentBatchAction(Action):
    action_type: Literal["remove_from_payment_batch"] = "remove_from_payment_batch"
    case_id: str
    notes: str | None = None


class StopPaymentAction(Action):
    action_type: Literal["stop_payment"] = "stop_payment"
    case_id: str
    notes: str | None = None


class RecordVendorRefundAction(Action):
    action_type: Literal["record_vendor_refund"] = "record_vendor_refund"
    case_id: str
    refund_amount: float = Field(..., gt=0)
    notes: str | None = None


class ApplyCreditMemoAction(Action):
    action_type: Literal["apply_credit_memo"] = "apply_credit_memo"
    case_id: str
    credit_memo_id: str
    notes: str | None = None


class WriteOffSmallBalanceAction(Action):
    action_type: Literal["write_off_small_balance"] = "write_off_small_balance"
    case_id: str
    reason_code: ReasonCode
    notes: str | None = None


class AdvanceClockAction(Action):
    action_type: Literal["advance_clock"] = "advance_clock"
    minutes: int = Field(..., ge=1, le=480)


class CloseCaseAction(Action):
    """Close a case. All close_blockers must be resolved first."""
    action_type: Literal["close_case"] = "close_case"
    case_id: str
    resolution_code: str = Field(description="Usually 'completed'")


class ReopenCaseAction(Action):
    action_type: Literal["reopen_case"] = "reopen_case"
    case_id: str
    reason_code: ReasonCode


OpsAction = Annotated[
    (
        ListQueueAction
        | OpenCaseAction
        | ViewRecordAction
        | QueryPolicyAction
        | SearchCasesAction
        | InspectAuditAction
        | ApproveAction
        | RejectAction
        | EscalateAction
        | DeferAction
        | RequestInfoAction
        | AssignAction
        | BulkAssignAction
        | BulkRouteAction
        | ClaimCaseAction
        | ReturnToQueueAction
        | RouteCaseAction
        | SendMessageAction
        | LogInternalNoteAction
        | PrioritizeAction
        | BatchReorderAction
        | ScheduleFollowUpAction
        | PauseSLAAction
        | ResumeSLAAction
        | ExecuteRefundAction
        | AcceptDisputeAction
        | ChallengeDisputeAction
        | RefundPreDisputeAlertAction
        | ResolvePrearbitrationAction
        | SubmitDisputeEvidenceAction
        | FreezePayoutsAction
        | UnfreezePayoutsAction
        | SetReservePercentAction
        | ClearReserveAction
        | SetPayoutDelayDaysAction
        | RecordThreeWayMatchAction
        | PlacePaymentHoldAction
        | ReleasePaymentHoldAction
        | RequestCreditMemoAction
        | SendForSecondaryApprovalAction
        | ReviewKYCAction
        | TriggerReverificationAction
        | RunSanctionsScreenAction
        | StartEDDReviewAction
        | ReviewBeneficialOwnerAction
        | RequestFieldCorrectionAction
        | FileOFACReportAction
        | FreezePaymentsAction
        | RebalanceQueueAction
        | SendToQAAction
        | ApproveQAAction
        | FailQAAction
        | RequestRevisedInvoiceAction
        | RequestPOChangeAction
        | RemoveFromPaymentBatchAction
        | StopPaymentAction
        | RecordVendorRefundAction
        | ApplyCreditMemoAction
        | WriteOffSmallBalanceAction
        | AdvanceClockAction
        | CloseCaseAction
        | ReopenCaseAction
    ),
    Field(discriminator="action_type"),
]

OPS_ACTION_ADAPTER = TypeAdapter(OpsAction)


def _build_action_field_map() -> dict[str, set[str]]:
    from opsarena.action_docs import iter_action_models
    return {
        model.model_fields["action_type"].default: set(model.model_fields.keys())
        for model in iter_action_models()
    }


_ACTION_ACCEPTED_FIELDS = _build_action_field_map()


def validate_ops_action(action: Action | dict[str, Any]) -> OpsAction:
    payload = action.model_dump(exclude_none=True) if isinstance(action, Action) else action
    action_type = payload.get("action_type")
    accepted = _ACTION_ACCEPTED_FIELDS.get(action_type)
    if accepted:
        payload = {k: v for k, v in payload.items() if k in accepted}
    return OPS_ACTION_ADAPTER.validate_python(payload)
