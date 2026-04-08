from __future__ import annotations

from opsarena.documents import GoodsReceipt, ReceiptLineItem
from opsarena.domain.core import ApprovalHistoryEntry, MessageLogEntry, QAReviewEntry, QAStatus, RouteHistoryEntry
from opsarena.domain.events import (
    ChargebackEvent,
    FollowUpDueEvent,
    InfoResponseEvent,
    QASampleSelectedEvent,
    ReworkDueEvent,
    ReopenEvent,
    SecondaryApprovalDecisionEvent,
)
from opsarena.engine.handlers.common import (
    can_close,
    claimed_case_count,
    record_evidence,
    require_case,
    require_invoice_workflow,
    require_refund_workflow,
    require_kyc_workflow,
    sort_key,
    sync_kyc_flags,
)
from opsarena.engine.handlers.result import TransitionResult
from opsarena.engine.policies import query_policy
from opsarena.engine.scheduler import process_due_events, schedule_event
from opsarena.engine.state import WorldState
from opsarena.enums import CaseType, RecordType, Resolution, SortField
from opsarena.models import (
    AdvanceClockAction,
    ApproveAction,
    AssignAction,
    BatchReorderAction,
    ClaimCaseAction,
    CloseCaseAction,
    DeferAction,
    EscalateAction,
    InspectAuditAction,
    ListQueueAction,
    LogInternalNoteAction,
    OpenCaseAction,
    PauseSLAAction,
    PrioritizeAction,
    QueryPolicyAction,
    RejectAction,
    ReopenCaseAction,
    RequestInfoAction,
    ResumeSLAAction,
    ReturnToQueueAction,
    RouteCaseAction,
    ScheduleFollowUpAction,
    SearchCasesAction,
    SendToQAAction,
    ApproveQAAction,
    FailQAAction,
    SendMessageAction,
    SendForSecondaryApprovalAction,
    ViewRecordAction,
)


def _clear_qa_rework(case) -> None:
    case.rework_due_at = None
    case.qa_rework_overdue = False
    case.visible_flags = [flag for flag in case.visible_flags if flag != "qa_rework_overdue"]


def handle_list_queue(state: WorldState, action: ListQueueAction) -> tuple[TransitionResult, None]:
    state.queue_order.sort(key=lambda case_id: sort_key(state.cases[case_id], action.sort_by))
    return TransitionResult(True, "Queue listed"), None


def handle_search_cases(state: WorldState, action: SearchCasesAction) -> tuple[TransitionResult, None]:
    return TransitionResult(True, f"Search returned {len(state.open_cases())} visible cases"), None


def handle_open_case(state: WorldState, action: OpenCaseAction):
    case = require_case(state, action.case_id)
    state.current_case_id = case.case_id
    case.status = "in_progress" if case.status in {"open", "reopened", "rework", "assigned", "routed"} else case.status
    return TransitionResult(True, f"Opened {case.case_id}"), case


def handle_view_record(state: WorldState, action: ViewRecordAction):
    case = state.cases.get(state.current_case_id) if state.current_case_id else None
    if case and action.record_type in case.hidden.forbidden_record_types:
        state.metrics.data_breach_count += 1
        raise ValueError("forbidden_record_access")
    state.current_record_type = action.record_type
    state.current_record_id = action.record_id
    if case:
        record_evidence(case, action.record_type)
    return TransitionResult(True, f"Viewed {action.record_type.value}:{action.record_id}"), case


def handle_query_policy(state: WorldState, action: QueryPolicyAction):
    case = state.cases.get(state.current_case_id) if state.current_case_id else None
    policy = state.records.policies.get(action.policy_id)
    if policy is None:
        raise ValueError(f"unknown_policy:{action.policy_id}")
    context = {
        "refund_amount": case.amount if case else 0,
        "invoice_total": case.amount if case else 0,
        "priority": case.priority if case else 3,
    }
    clauses = query_policy(policy, context=context, clause_id=action.clause_id)
    if case:
        case.policy_checked = True
        case.mark_check("review_policy")
        case.gather_evidence("refund_policy" if case.case_type == CaseType.REFUND else "compliance_policy")
    state.current_policy_id = action.policy_id
    state.current_clause_id = clauses[0].clause_id if clauses else None
    return TransitionResult(True, f"Policy {action.policy_id} queried"), case


def handle_inspect_audit(state: WorldState, action: InspectAuditAction):
    case = require_case(state, action.case_id)
    return TransitionResult(True, f"Audit loaded for {action.case_id}"), case


def handle_log_internal_note(state: WorldState, action: LogInternalNoteAction):
    case = require_case(state, action.case_id)
    case.internal_notes.append(action.note_code)
    return TransitionResult(True, f"Internal note logged for {case.case_id}"), case


def _auto_fill_slots(case, template_id: str, provided: dict[str, str]) -> dict[str, str]:
    """Fill missing template slots from case data when possible."""
    slots = dict(provided)
    defaults: dict[str, str] = {
        "case_id": case.case_id,
        "amount": str(case.amount) if case.amount else "0",
        "resolution": case.resolution.value if hasattr(case.resolution, "value") else str(case.resolution),
    }
    if case.linked_records:
        for rec in case.linked_records:
            if rec.record_type in ("order",):
                defaults["order_id"] = rec.record_id
    if case.requested_info_fields:
        defaults["fields"] = ", ".join(case.requested_info_fields)
    for key, val in defaults.items():
        if key not in slots:
            slots[key] = val
    return slots


def handle_send_message(state: WorldState, action: SendMessageAction):
    case = require_case(state, action.case_id)
    template = state.records.message_templates.get(action.template_id)
    if template is None:
        raise ValueError(f"unknown_template:{action.template_id}")
    slots = _auto_fill_slots(case, action.template_id, action.slots)
    body = template.body_template.format(**slots)
    case.communication_log.append(
        MessageLogEntry(
            timestamp=state.current_time,
            channel=template.channel,
            subject=template.subject_template,
            body=body,
            template_id=template.template_id,
        )
    )
    case.notifications_sent += 1
    case.customer_notified = True
    return TransitionResult(True, f"Message sent via {template.channel}"), case


def handle_request_info(state: WorldState, action: RequestInfoAction):
    case = require_case(state, action.case_id)
    if case.requested_info_fields.count(action.field_name) >= 2:
        raise ValueError("duplicate_info_request")
    case.requested_info_fields.append(action.field_name)
    case.notifications_sent += 1
    latency = case.hidden.hidden_response_latency_minutes or 30
    if action.field_name == "goods_receipt":
        receipt = GoodsReceipt(
            receipt_id=f"rcpt_{case.case_id}",
            receipt_type="grn",
            status="received",
            po_reference="PO-2001",
            vendor_id=case.vendor_id or "vendor_1",
            receiving_warehouse="WH-1",
            received_by="receiver_1",
            receipt_date=state.current_time + latency,
            carrier="UPS",
            tracking_numbers=["1Z999"],
            shipment_status="delivered",
            line_items=[
                ReceiptLineItem(
                    po_line_number=1,
                    sku="pk-tape",
                    description="Packaging tape",
                    ordered_quantity=100,
                    shipped_quantity=100,
                    received_quantity=100,
                    accepted_quantity=100,
                    rejected_quantity=0,
                    condition="good",
                )
            ],
        )
        schedule_event(
            state,
            InfoResponseEvent(
                at_time=state.current_time + latency,
                case_id=case.case_id,
                field_name=action.field_name,
                receipt_record=receipt,
            ),
        )
    elif action.field_name == "individual.verification.document":
        schedule_event(
            state,
            InfoResponseEvent(
                at_time=state.current_time + latency,
                case_id=case.case_id,
                field_name=action.field_name,
                record_id=case.linked_records[-1].record_id,
            ),
        )
    else:
        schedule_event(
            state,
            InfoResponseEvent(
                at_time=state.current_time + latency,
                case_id=case.case_id,
                field_name=action.field_name,
            ),
        )
    return TransitionResult(True, f"Requested {action.field_name}"), case


def handle_assign(state: WorldState, action: AssignAction):
    case = require_case(state, action.case_id)
    case.current_owner = action.assignee_type
    return TransitionResult(True, f"Assigned {case.case_id} to {action.assignee_type}"), case


def handle_claim_case(state: WorldState, action: ClaimCaseAction):
    case = require_case(state, action.case_id)
    if case.claimed_by is not None and case.claimed_by != action.assignee_type:
        raise ValueError("case_already_claimed")
    claim_capacity = state.metadata.get("claim_capacity", 2)
    if case.claimed_by is None and claimed_case_count(state) >= claim_capacity:
        state.metrics.claim_overflow_attempts += 1
        raise ValueError("claim_capacity_reached")
    case.claimed_by = action.assignee_type
    case.claimed_at = state.current_time
    case.current_owner = action.assignee_type
    case.status = "claimed" if case.status == "open" else case.status
    state.current_case_id = case.case_id
    state.metrics.cases_claimed += 1
    return TransitionResult(True, f"Claimed {case.case_id}"), case


def handle_return_to_queue(state: WorldState, action: ReturnToQueueAction):
    case = require_case(state, action.case_id)
    case.claimed_by = None
    case.claimed_at = None
    case.current_owner = "queue"
    case.status = "open" if case.status not in {"closed", "resolved"} else case.status
    case.route_reason = action.reason_code.value
    return TransitionResult(True, f"Returned {case.case_id} to queue"), case


def handle_route_case(state: WorldState, action: RouteCaseAction):
    case = require_case(state, action.case_id)
    if action.target_queue not in case.allowed_escalation_queues:
        raise ValueError("invalid_route_target")
    case.current_owner = action.assignee_type or action.target_queue.value
    case.status = "routed"
    case.active_queue = action.target_queue.value
    case.route_reason = action.reason_code.value
    case.route_history.append(
        RouteHistoryEntry(
            at_time=state.current_time,
            queue=action.target_queue.value,
            owner=case.current_owner,
            reason=action.reason_code.value,
        )
    )
    return TransitionResult(True, f"Routed {case.case_id} to {action.target_queue.value}"), case


def handle_prioritize(state: WorldState, action: PrioritizeAction):
    case = require_case(state, action.case_id)
    case.priority = int(action.new_priority)
    state.queue_order.sort(key=lambda cid: sort_key(state.cases[cid], SortField.PRIORITY))
    return TransitionResult(True, f"Priority updated for {case.case_id}"), case


def handle_batch_reorder(state: WorldState, action: BatchReorderAction):
    if action.ordering_rule == "sla":
        state.queue_order.sort(key=lambda cid: sort_key(state.cases[cid], SortField.SLA_REMAINING))
    elif action.ordering_rule == "amount":
        state.queue_order.sort(key=lambda cid: sort_key(state.cases[cid], SortField.AMOUNT))
    else:
        state.queue_order.sort(key=lambda cid: sort_key(state.cases[cid], SortField.PRIORITY))
    return TransitionResult(True, f"Queue reordered by {action.ordering_rule}"), None


def handle_schedule_follow_up(state: WorldState, action: ScheduleFollowUpAction):
    case = require_case(state, action.case_id)
    if action.follow_up_at <= state.current_time:
        raise ValueError("follow_up_must_be_future")
    case.next_touch_at = action.follow_up_at
    case.waiting_reason = action.reason_code.value
    case.follow_up_overdue = False
    case.visible_flags = [flag for flag in case.visible_flags if flag != "follow_up_overdue"]
    case.status = "waiting_follow_up"
    schedule_event(
        state,
        FollowUpDueEvent(
            at_time=action.follow_up_at,
            case_id=case.case_id,
            scheduled_for=action.follow_up_at,
            reason_code=action.reason_code,
        ),
    )
    return TransitionResult(True, f"Follow-up scheduled for {case.case_id}"), case


def handle_pause_sla(state: WorldState, action: PauseSLAAction):
    case = require_case(state, action.case_id)
    if case.sla_paused_at is not None:
        raise ValueError("sla_already_paused")
    case.sla_paused_at = state.current_time
    case.sla_pause_reason = action.reason_code.value
    case.pre_pause_status = case.status
    case.status = "waiting_external"
    return TransitionResult(True, f"SLA paused for {case.case_id}"), case


def handle_resume_sla(state: WorldState, action: ResumeSLAAction):
    case = require_case(state, action.case_id)
    paused_at = case.sla_paused_at
    if paused_at is None:
        raise ValueError("sla_not_paused")
    paused_minutes = max(0, state.current_time - paused_at)
    case.sla_deadline += paused_minutes
    previous_status = case.pre_pause_status or "open"
    case.sla_pause_reason = None
    case.sla_paused_at = None
    case.pre_pause_status = None
    case.status = "in_progress" if previous_status in {"open", "waiting_external"} else previous_status
    return TransitionResult(True, f"SLA resumed for {case.case_id}"), case


def handle_send_to_qa(state: WorldState, action: SendToQAAction):
    case = require_case(state, action.case_id)
    if case.resolution == Resolution.PENDING:
        raise ValueError("case_not_ready_for_qa")
    if case.status == "closed":
        raise ValueError("closed_case_cannot_enter_qa")
    if case.qa_status == QAStatus.PENDING:
        raise ValueError("qa_already_pending")
    case.qa_status = QAStatus.PENDING
    case.qa_owner = action.assignee_type or "qa_queue"
    case.status = "pending_qa"
    _clear_qa_rework(case)
    case.qa_history.append(
        QAReviewEntry(
            at_time=state.current_time,
            status=QAStatus.PENDING,
            owner=case.qa_owner,
            notes=action.notes,
        )
    )
    state.metrics.qa_reviews_requested += 1
    return TransitionResult(True, f"Sent {case.case_id} to QA"), case


def handle_approve_qa(state: WorldState, action: ApproveQAAction):
    case = require_case(state, action.case_id)
    if case.qa_status != QAStatus.PENDING:
        raise ValueError("qa_not_pending")
    owner = action.assignee_type or case.qa_owner or "qa_queue"
    case.qa_status = QAStatus.PASSED
    case.qa_owner = owner
    case.status = "resolved"
    _clear_qa_rework(case)
    case.qa_history.append(
        QAReviewEntry(
            at_time=state.current_time,
            status=QAStatus.PASSED,
            owner=owner,
            notes=action.notes,
        )
    )
    state.metrics.qa_reviews_passed += 1
    return TransitionResult(True, f"QA approved {case.case_id}"), case


def handle_fail_qa(state: WorldState, action: FailQAAction):
    case = require_case(state, action.case_id)
    if case.qa_status != QAStatus.PENDING:
        raise ValueError("qa_not_pending")
    owner = action.assignee_type or case.qa_owner or "qa_queue"
    case.qa_status = QAStatus.FAILED
    case.qa_owner = owner
    case.status = "rework"
    case.resolution = Resolution.PENDING
    case.rework_due_at = state.current_time + (case.hidden.hidden_follow_up_latency_minutes or 30)
    case.qa_rework_overdue = False
    case.visible_flags = list(dict.fromkeys(case.visible_flags + ["qa_rework"]))
    case.qa_history.append(
        QAReviewEntry(
            at_time=state.current_time,
            status=QAStatus.FAILED,
            owner=owner,
            reason=action.reason_code.value,
            notes=action.notes,
        )
    )
    schedule_event(
        state,
        ReworkDueEvent(
            at_time=case.rework_due_at,
            case_id=case.case_id,
            scheduled_for=case.rework_due_at,
        ),
    )
    state.metrics.qa_reviews_failed += 1
    return TransitionResult(True, f"QA failed {case.case_id}"), case


def handle_escalate(state: WorldState, action: EscalateAction):
    case = require_case(state, action.case_id)
    if action.target_queue not in case.allowed_escalation_queues:
        raise ValueError("invalid_escalation_target")
    _clear_qa_rework(case)
    case.resolution = Resolution.ESCALATED
    case.status = "escalated"
    case.escalation_justified = True
    case.active_queue = action.target_queue.value
    state.metadata["escalation_queue_load"] = state.metadata.get("escalation_queue_load", 0) + 1
    state.metrics.escalation_count += 1
    return TransitionResult(True, f"Escalated {case.case_id} to {action.target_queue.value}"), case


def handle_defer(state: WorldState, action: DeferAction):
    case = require_case(state, action.case_id)
    _clear_qa_rework(case)
    case.resolution = Resolution.DEFERRED
    case.status = "pending_info"
    return TransitionResult(True, f"Deferred {case.case_id}"), case


def handle_send_for_secondary_approval(state: WorldState, action: SendForSecondaryApprovalAction):
    case = require_case(state, action.case_id)
    if case.case_type == CaseType.REFUND:
        workflow = require_refund_workflow(case)
        if workflow.approval_status == "pending_secondary":
            raise ValueError("secondary_approval_pending")
        workflow.approval_status = "pending_secondary"
        expected = "denied" if case.hidden.true_fraud_risk > 0.7 else "approved"
    else:
        workflow = require_invoice_workflow(case)
        if workflow.approval_status.value == "pending_secondary":
            raise ValueError("secondary_approval_pending")
        workflow.approval_status = workflow.approval_status.PENDING_SECONDARY
        workflow.approval_reason = action.reason_code.value
        workflow.approval_assignee = action.assignee_type or "manager_review"
        workflow.approval_chain.append(
            ApprovalHistoryEntry(
                at_time=state.current_time,
                status="pending_secondary",
                owner=workflow.approval_assignee,
                reason=action.reason_code.value,
            )
        )
        expected = workflow.approval_expected_outcome.value
    state.metadata["approval_queue_load"] = state.metadata.get("approval_queue_load", 0) + 1
    case.status = "pending_approval"
    state.metrics.secondary_approvals_requested += 1
    schedule_event(
        state,
        SecondaryApprovalDecisionEvent(
            at_time=state.current_time + 15 + max(0, state.metadata.get("approval_queue_load", 1) - 1) * 10,
            case_id=case.case_id,
            outcome=expected,
        ),
    )
    return TransitionResult(True, f"Secondary approval requested for {case.case_id}"), case


def handle_approve(state: WorldState, action: ApproveAction):
    case = require_case(state, action.case_id)
    if case.case_type == CaseType.REFUND:
        workflow = require_refund_workflow(case)
        if workflow.secondary_approval_required and workflow.approval_status not in {"approved", "not_requested"}:
            raise ValueError("secondary_approval_required")
    if case.case_type == CaseType.INVOICE:
        workflow = require_invoice_workflow(case)
        if workflow.secondary_approval_required and workflow.approval_status.value not in {"approved", "not_requested"}:
            raise ValueError("secondary_approval_required")
        if workflow.credit_memo_status.value == "requested":
            raise ValueError("credit_memo_pending")
    if case.case_type == CaseType.KYC:
        workflow = require_kyc_workflow(case)
        if not workflow.kyc_complete:
            raise ValueError("kyc_incomplete")
        if not workflow.approval_ready():
            raise ValueError("compliance_review_incomplete")
    _clear_qa_rework(case)
    case.resolution = Resolution.APPROVED
    case.status = "resolved"
    if case.case_type == CaseType.REFUND and case.hidden.true_fraud_risk > 0.7:
        schedule_event(state, ChargebackEvent(at_time=state.current_time + 30, case_id=case.case_id))
    if case.case_type == CaseType.INVOICE and case.hidden.true_is_duplicate:
        state.metrics.duplicate_payments += 1
    if case.case_type == CaseType.KYC:
        sync_kyc_flags(case)
    return TransitionResult(True, f"Approved {case.case_id}"), case


def handle_reject(state: WorldState, action: RejectAction):
    case = require_case(state, action.case_id)
    _clear_qa_rework(case)
    case.resolution = Resolution.REJECTED
    case.status = "resolved"
    if case.case_type == CaseType.KYC:
        sync_kyc_flags(case)
    return TransitionResult(True, f"Rejected {case.case_id}"), case


def handle_close_case(state: WorldState, action: CloseCaseAction):
    case = require_case(state, action.case_id)
    allowed, reason = can_close(case)
    if not allowed:
        raise ValueError(reason.replace(" ", "_"))
    case.status = "closed"
    case.terminal_reason = action.resolution_code
    state.metrics.cases_resolved += 1
    state.queue_order = [cid for cid in state.queue_order if state.cases[cid].status != "closed"] + [
        cid for cid in state.queue_order if state.cases[cid].status == "closed"
    ]
    if case.hidden.qa_sample_on_close and case.qa_status == QAStatus.NOT_REQUESTED:
        schedule_event(
            state,
            QASampleSelectedEvent(
                at_time=state.current_time + (case.hidden.qa_sample_delay_minutes or 10),
                case_id=case.case_id,
            ),
        )
    if case.requires_customer_notification and not case.customer_notified:
        schedule_event(state, ReopenEvent(at_time=state.current_time + 15, case_id=case.case_id))
    return TransitionResult(True, f"Closed {case.case_id}"), case


def handle_reopen_case(state: WorldState, action: ReopenCaseAction):
    case = require_case(state, action.case_id)
    case.status = "reopened"
    case.resolution = Resolution.PENDING
    return TransitionResult(True, f"Reopened {case.case_id}"), case


def handle_advance_clock(state: WorldState, action: AdvanceClockAction):
    state.current_time += action.minutes
    state.metrics.simulated_minutes += action.minutes
    messages = process_due_events(state)
    return TransitionResult(True, "; ".join(messages) if messages else f"Advanced clock by {action.minutes}m"), None
