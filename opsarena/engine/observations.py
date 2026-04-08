from __future__ import annotations

from typing import Any

from opsarena.domain.core import QAStatus
from opsarena.domain.workflows.invoice import ApprovalStatus, CreditMemoStatus, InvoiceWorkflowState, VendorResponseStatus
from opsarena.domain.workflows.kyc import EDDStatus, OFACReportStatus
from opsarena.engine.action_availability import available_actions
from opsarena.engine.state import AuditEntry, CaseState, WorldState
from opsarena.enums import RecordType, Resolution
from opsarena.models import (
    ActionResult,
    AuditEntryView,
    CaseDetail,
    LinkedRecordView,
    MessageSummary,
    OpsArenaObservation,
    PolicyResult,
    QueueItem,
)


def escalation_load_label(load: int, capacity: int) -> str:
    ratio = load / max(1, capacity)
    if ratio >= 1.0:
        return "full"
    if ratio >= 0.75:
        return "heavy"
    if ratio >= 0.4:
        return "medium"
    return "light"


def _priority_label(priority: int) -> str:
    return f"P{priority}"


def _workflow_metadata(case: CaseState) -> dict[str, Any]:
    data = {
        "active_queue": case.active_queue,
        "route_reason": case.route_reason,
        "route_history": [entry.model_dump() for entry in case.route_history],
        "sla_paused_at": case.sla_paused_at,
        "sla_pause_reason": case.sla_pause_reason,
        "qa_status": case.qa_status.value,
        "qa_owner": case.qa_owner,
        "qa_required": case.qa_required,
        "qa_history": [
            {
                "at_time": entry.at_time,
                "status": entry.status.value,
                "owner": entry.owner,
                "reason": entry.reason,
                "notes": entry.notes,
            }
            for entry in case.qa_history
        ],
        "rework_due_at": case.rework_due_at,
        "qa_rework_overdue": case.qa_rework_overdue,
        **case.workflow.public_metadata(),
    }
    if case.pending_info_fields:
        data["pending_info_fields"] = case.pending_info_fields
    if case.requested_info_fields:
        data["requested_info_fields"] = case.requested_info_fields
    if case.claimed_by:
        data["claimed_by"] = case.claimed_by
    if case.claimed_at is not None:
        data["claimed_at"] = case.claimed_at
    if case.next_touch_at is not None:
        data["next_touch_at"] = case.next_touch_at
    if case.waiting_reason:
        data["waiting_reason"] = case.waiting_reason
    if case.follow_up_overdue:
        data["follow_up_overdue"] = True
    return data


def _next_due_minutes(state: WorldState, case: CaseState, clock: int) -> int | None:
    workflow_meta = case.workflow.public_metadata()
    due_fields = [
        "pre_dispute_due_at",
        "representment_due_at",
        "prearbitration_due_at",
        "report_due_at",
        "edd_due_at",
        "next_touch_at",
        "stop_payment_window_until",
        "rework_due_at",
    ]
    candidates = [
        value
        for field in due_fields
        if isinstance((value := workflow_meta.get(field, getattr(case, field, None))), int) and value > clock
    ]
    event_due = [
        event.at_time
        for event in state.scheduled_events
        if event.case_id == case.case_id and event.at_time > clock
    ]
    candidates.extend(event_due)
    if not candidates:
        return None
    return max(1, min(candidates) - clock)


def _close_blockers(case: CaseState) -> list[str]:
    blockers: list[str] = []
    if case.resolution == Resolution.PENDING:
        blockers.append("resolution_missing")
    if case.next_touch_at is not None:
        blockers.append("follow_up_scheduled")
    if case.follow_up_overdue:
        blockers.append("follow_up_overdue")
    if case.qa_status == QAStatus.PENDING:
        blockers.append("qa_review_pending")
    elif case.qa_required and case.qa_status != QAStatus.PASSED:
        blockers.append("qa_approval_missing")
    if case.requires_customer_notification and not case.customer_notified:
        blockers.append("customer_notification_missing")
    if case.resolution == Resolution.PENDING and case.pending_info_fields:
        blockers.append("external_info_pending")
    if isinstance(case.workflow, InvoiceWorkflowState):
        if case.workflow.secondary_approval_required and case.workflow.approval_status == ApprovalStatus.PENDING_SECONDARY:
            blockers.append("secondary_approval_pending")
        if case.workflow.credit_memo_status == CreditMemoStatus.REQUESTED:
            blockers.append("credit_memo_pending")
    else:
        workflow_meta = case.workflow.public_metadata()
        if workflow_meta.get("approval_status") == "pending_secondary":
            blockers.append("secondary_approval_pending")
    if hasattr(case.workflow, "correction_fields") and getattr(case.workflow, "correction_fields"):
        blockers.append("correction_required")
    if hasattr(case.workflow, "ofac_report_status"):
        if case.hidden.true_ofac_report_required and case.workflow.ofac_report_status != OFACReportStatus.FILED:
            blockers.append("ofac_report_pending")
    if isinstance(case.workflow, InvoiceWorkflowState):
        if case.resolution == Resolution.PENDING and case.workflow.vendor_response_status == VendorResponseStatus.AWAITING:
            blockers.append("vendor_response_pending")
    return blockers


def _waiting_on(case: CaseState) -> list[str]:
    waiting: list[str] = []
    if case.waiting_reason:
        waiting.append(f"follow_up:{case.waiting_reason}")
    waiting.extend(
        f"info_response:{field}"
        for field in case.pending_info_fields
        if field in case.requested_info_fields
    )
    if isinstance(case.workflow, InvoiceWorkflowState):
        if case.workflow.approval_status == ApprovalStatus.PENDING_SECONDARY:
            waiting.append("secondary_approval")
        if case.workflow.vendor_response_status == VendorResponseStatus.AWAITING:
            waiting.append("vendor_response")
        if case.workflow.credit_memo_status == CreditMemoStatus.REQUESTED:
            waiting.append("credit_memo_response")
    if hasattr(case.workflow, "dispute_workflow_status"):
        status = getattr(case.workflow, "dispute_workflow_status")
        if status in {"submitted", "challenged", "inquiry_contested", "prearbitration_contested"}:
            waiting.append(f"dispute:{status}")
    if hasattr(case.workflow, "edd_status"):
        if case.workflow.edd_status in {EDDStatus.AWAITING_RESPONSE, EDDStatus.IN_PROGRESS}:
            waiting.append(f"edd:{case.workflow.edd_status.value}")
    return waiting


def _case_phase(case: CaseState, close_blockers: list[str], waiting_on: list[str]) -> str:
    if case.status == "closed":
        return "closed"
    if case.qa_status == QAStatus.PENDING:
        return "ready_for_qa"
    if case.resolution != Resolution.PENDING:
        if close_blockers:
            return "ready_for_closeout"
        return "ready_to_close"
    if waiting_on or case.status in {"waiting_external", "pending_info", "pending_approval"}:
        return "waiting_external"
    if case.completed_check_names != case.required_check_names or not case.policy_checked:
        return "investigate"
    return "ready_to_decide"


def _recommended_action_categories(case: CaseState, close_blockers: list[str], waiting_on: list[str]) -> list[str]:
    categories: list[str] = []
    if case.qa_status == QAStatus.PENDING:
        categories.append("qa")
        return categories
    if case.resolution != Resolution.PENDING:
        if "ofac_report_pending" in close_blockers:
            categories.append("resolution")
        if "customer_notification_missing" in close_blockers:
            categories.append("notify")
        if "qa_approval_missing" in close_blockers:
            categories.append("qa")
        if any(blocker in close_blockers for blocker in {"follow_up_scheduled", "external_info_pending", "secondary_approval_pending", "vendor_response_pending"}):
            categories.append("wait")
        if not close_blockers:
            categories.append("close")
        return categories

    if not case.policy_checked:
        categories.append("policy_review")
    if case.completed_check_names != case.required_check_names:
        categories.append("record_review")
    pending_unrequested = [
        field
        for field in case.pending_info_fields
        if field not in case.requested_info_fields
    ]
    if pending_unrequested:
        categories.append("request_missing_info")
    if waiting_on:
        categories.append("wait")
    categories.append("resolution")
    return list(dict.fromkeys(categories))


def render_queue_view(state: WorldState) -> list[QueueItem]:
    items: list[QueueItem] = []
    for case_id in state.queue_order:
        case = state.cases.get(case_id)
        if case is None:
            continue
        if case.status == "closed":
            continue
        customer_name = None
        if case.customer_id and case.customer_id in state.records.customers:
            customer = state.records.customers[case.customer_id]
            customer_name = f"{customer.first_name} {customer.last_name}"
        items.append(
            QueueItem(
                case_id=case.case_id,
                case_type=case.case_type.value,
                priority=_priority_label(case.priority),
                sla_remaining_minutes=case.sla_deadline - state.current_time,
                summary=case.visible_summary,
                status=case.status,
                current_owner=case.current_owner,
                active_queue=case.active_queue,
                amount=case.amount or None,
                customer_name=customer_name,
                flags=case.visible_flags,
            )
        )
    return items


def render_case_detail(state: WorldState, case: CaseState, clock: int) -> CaseDetail:
    close_blockers = _close_blockers(case)
    waiting_on = _waiting_on(case)
    return CaseDetail(
        case_id=case.case_id,
        case_type=case.case_type.value,
        priority=_priority_label(case.priority),
        status=case.status,
        sla_deadline=case.sla_deadline,
        created_at=case.created_at,
        customer_id=case.customer_id,
        vendor_id=case.vendor_id,
        amount=case.amount,
        currency=case.currency,
        visible_summary=case.visible_summary,
        visible_flags=case.visible_flags,
        linked_records=[
            LinkedRecordView(
                record_type=record.record_type.value,
                record_id=record.record_id,
                title=record.title,
            )
            for record in case.linked_records
        ],
        required_checks=case.required_check_names,
        checks_completed=case.completed_check_names,
        communication_log=[
            MessageSummary(
                timestamp=entry.timestamp,
                channel=entry.channel,
                subject=entry.subject,
                template_id=entry.template_id,
            )
            for entry in case.communication_log
        ],
        internal_notes=case.internal_notes,
        requested_info_fields=case.requested_info_fields,
        current_owner=case.current_owner,
        case_phase=_case_phase(case, close_blockers, waiting_on),
        close_blockers=close_blockers,
        waiting_on=waiting_on,
        next_due_minutes=_next_due_minutes(state, case, clock),
        recommended_action_categories=_recommended_action_categories(case, close_blockers, waiting_on),
        workflow_metadata=_workflow_metadata(case),
    )


def render_record_view(state: WorldState) -> dict | None:
    record_type = state.current_record_type
    record_id = state.current_record_id
    if not record_type or not record_id:
        return None
    _STORE_MAP = {
        RecordType.ORDER: "orders",
        RecordType.CUSTOMER: "customers",
        RecordType.SHIPPING: "shipping",
        RecordType.PAYMENT: "payments",
        RecordType.DISPUTE: "disputes",
        RecordType.INVOICE: "invoices",
        RecordType.CREDIT_MEMO: "credit_memos",
        RecordType.PURCHASE_ORDER: "purchase_orders",
        RecordType.RECEIPT: "receipts",
        RecordType.KYC_DOCUMENT: "kyc_verifications",
    }
    store_attr = _STORE_MAP.get(record_type)
    if store_attr is None:
        return None
    store = getattr(state.records, store_attr, {})
    record = store.get(record_id)
    if record is None:
        state.current_record_type = None
        state.current_record_id = None
        return None
    return record.model_dump()


def render_policy_result(state: WorldState) -> PolicyResult | None:
    if not state.current_policy_id:
        return None
    policy = state.records.policies.get(state.current_policy_id)
    if policy is None:
        state.current_policy_id = None
        return None
    if state.current_clause_id is None:
        description = "\n".join(clause.description for clause in policy.clauses[:3])
        actions = [action.action_type for clause in policy.clauses for action in clause.actions]
        return PolicyResult(
            policy_id=policy.policy_id,
            title=policy.title,
            description=description,
            matched_actions=actions[:5],
        )
    clause = next((c for c in policy.clauses if c.clause_id == state.current_clause_id), None)
    if clause is None:
        state.current_clause_id = None
        return PolicyResult(
            policy_id=policy.policy_id,
            title=policy.title,
            description="\n".join(c.description for c in policy.clauses[:3]),
            matched_actions=[a.action_type for c in policy.clauses for a in c.actions][:5],
        )
    return PolicyResult(
        policy_id=policy.policy_id,
        clause_id=clause.clause_id,
        title=policy.title,
        description=clause.description,
        matched_actions=[action.action_type for action in clause.actions],
    )


def render_observation(
    state: WorldState,
    success: bool = True,
    action_type: str = "",
    message: str = "",
    error_code: str | None = None,
    reward: float | None = None,
    done: bool = False,
) -> OpsArenaObservation:
    current_case = state.cases.get(state.current_case_id) if state.current_case_id else None
    if current_case is not None and current_case.status == "closed":
        current_case = None
    queue_view = render_queue_view(state)
    esc_load = state.metadata.get("escalation_queue_load", 0)
    esc_capacity = state.metadata.get("escalation_queue_capacity", 2)
    return OpsArenaObservation(
        done=done,
        reward=reward,
        queue_view=queue_view,
        case_detail=render_case_detail(state, current_case, state.current_time) if current_case else None,
        record_view=render_record_view(state),
        policy_result=render_policy_result(state),
        audit_trail=[
            AuditEntryView(
                timestamp=entry.timestamp,
                case_id=entry.case_id,
                action_type=entry.action_type,
                message=entry.message,
            )
            for entry in state.audit_log[-5:]
        ],
        clock=state.current_time,
        escalation_queue_load=escalation_load_label(esc_load, esc_capacity),
        escalation_slots_remaining=max(0, esc_capacity - esc_load),
        system_message=message,
        error=None if success else message,
        available_actions=available_actions(state),
        metadata={
            "action_result": ActionResult(
                success=success,
                action_type=action_type,
                message=message,
                error_code=error_code,
            ).model_dump(),
            "queue_metrics": state.queue_state().model_dump(
                include={
                    "oldest_open_case_age_minutes",
                    "queue_backlog_age_minutes",
                    "overdue_follow_ups",
                    "claimed_case_count",
                    "unassigned_count",
                    "exception_queue_size",
                    "agent_capacity",
                    "total_cases_resolved",
                    "total_sla_breaches",
                }
            ),
            "approval_queue": {
                "load": state.metadata.get("approval_queue_load", 0),
                "capacity": state.metadata.get("approval_queue_capacity", 1),
            },
            "agent_loads": {
                owner: sum(
                    1
                    for case in state.cases.values()
                    if case.status != "closed" and case.current_owner == owner
                )
                for owner in sorted(
                    {
                        case.current_owner
                        for case in state.cases.values()
                        if case.current_owner not in {"queue", "ops_agent"}
                    }
                )
            },
        },
    )
