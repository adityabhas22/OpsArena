from __future__ import annotations

from typing import Any

from opsarena.engine.state import AuditEntry, CaseState, WorldState
from opsarena.enums import Priority, RecordType
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
    visible_keys = {
        "active_queue",
        "route_reason",
        "route_history",
        "sla_paused_at",
        "sla_pause_reason",
        "refund_execution_state",
        "refunded_amount",
        "dispute_stage",
        "dispute_resolution",
        "dispute_fee",
        "dispute_workflow_status",
        "dispute_evidence_fields",
        "match_status",
        "duplicate_status",
        "variance_amount",
        "payment_hold",
        "payment_hold_reason",
        "credit_memo_status",
        "credit_memo_amount",
        "approval_status",
        "approval_reason",
        "approval_assignee",
        "approval_chain",
        "verification_status",
        "requirements_due",
        "payout_hold",
        "kyc_stage",
    }
    data = {key: value for key, value in case.workflow_data.items() if key in visible_keys}
    if case.pending_info_fields:
        data["pending_info_fields"] = case.pending_info_fields
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


def render_queue_view(state: WorldState) -> list[QueueItem]:
    items: list[QueueItem] = []
    for case_id in state.queue_order:
        case = state.cases[case_id]
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
                amount=case.amount or None,
                customer_name=customer_name,
                flags=case.visible_flags,
            )
        )
    return items


def render_case_detail(case: CaseState) -> CaseDetail:
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
        current_owner=case.current_owner,
        workflow_metadata=_workflow_metadata(case),
    )


def render_record_view(state: WorldState) -> dict | None:
    record_type = state.current_record_type
    record_id = state.current_record_id
    if not record_type or not record_id:
        return None
    if record_type == RecordType.ORDER:
        return state.records.orders[record_id].model_dump()
    if record_type == RecordType.CUSTOMER:
        return state.records.customers[record_id].model_dump()
    if record_type == RecordType.SHIPPING:
        return state.records.shipping[record_id].model_dump()
    if record_type == RecordType.PAYMENT:
        return state.records.payments[record_id].model_dump()
    if record_type == RecordType.DISPUTE:
        return state.records.disputes[record_id].model_dump()
    if record_type == RecordType.INVOICE:
        return state.records.invoices[record_id].model_dump()
    if record_type == RecordType.CREDIT_MEMO:
        return state.records.credit_memos[record_id].model_dump()
    if record_type == RecordType.PURCHASE_ORDER:
        return state.records.purchase_orders[record_id].model_dump()
    if record_type == RecordType.RECEIPT:
        return state.records.receipts[record_id].model_dump()
    if record_type == RecordType.KYC_DOCUMENT:
        return state.records.kyc_verifications[record_id].model_dump()
    return None


def render_policy_result(state: WorldState) -> PolicyResult | None:
    if not state.current_policy_id:
        return None
    policy = state.records.policies[state.current_policy_id]
    if state.current_clause_id is None:
        description = "\n".join(clause.description for clause in policy.clauses[:3])
        actions = [action.action_type for clause in policy.clauses for action in clause.actions]
        return PolicyResult(
            policy_id=policy.policy_id,
            title=policy.title,
            description=description,
            matched_actions=actions[:5],
        )
    clause = next(clause for clause in policy.clauses if clause.clause_id == state.current_clause_id)
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
    queue_view = render_queue_view(state)
    esc_load = state.metadata.get("escalation_queue_load", 0)
    esc_capacity = state.metadata.get("escalation_queue_capacity", 2)
    return OpsArenaObservation(
        done=done,
        reward=reward,
        queue_view=queue_view,
        case_detail=render_case_detail(current_case) if current_case else None,
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
        available_actions=[
            "list_queue",
            "open_case",
            "view_record",
            "query_policy",
            "search_cases",
            "inspect_audit",
            "approve",
            "reject",
            "escalate",
            "defer",
            "request_info",
            "assign",
            "claim_case",
            "return_to_queue",
            "route_case",
            "send_message",
            "log_internal_note",
            "prioritize",
            "batch_reorder",
            "schedule_follow_up",
            "pause_sla",
            "resume_sla",
            "execute_refund",
            "accept_dispute",
            "submit_dispute_evidence",
            "record_three_way_match",
            "place_payment_hold",
            "release_payment_hold",
            "request_credit_memo",
            "send_for_secondary_approval",
            "review_kyc",
            "trigger_reverification",
            "advance_clock",
            "close_case",
            "reopen_case",
        ],
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
                    "overdue_follow_ups",
                    "claimed_case_count",
                    "total_cases_resolved",
                    "total_sla_breaches",
                }
            ),
            "approval_queue": {
                "load": state.metadata.get("approval_queue_load", 0),
                "capacity": state.metadata.get("approval_queue_capacity", 1),
            },
        },
    )
