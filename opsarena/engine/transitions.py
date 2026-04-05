from __future__ import annotations

from dataclasses import dataclass

from opsarena.documents import CreditMemoRecord, GoodsReceipt, ReceiptLineItem
from opsarena.engine.policies import query_policy
from opsarena.engine.scheduler import process_due_events, schedule_event
from opsarena.engine.state import AuditEntry, CaseState, MessageLogEntry, WorldState
from opsarena.enums import CaseType, Priority, ReasonCode, RecordType, Resolution, SortField, TargetQueue, TaskId
from opsarena.models import (
    AdvanceClockAction,
    ApproveAction,
    AssignAction,
    BatchReorderAction,
    ClaimCaseAction,
    CloseCaseAction,
    DeferAction,
    EscalateAction,
    ExecuteRefundAction,
    AcceptDisputeAction,
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
    RequestInfoAction,
    RequestCreditMemoAction,
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
from opsarena.rewards import (
    RewardBreakdown,
    compute_queue_reward,
    compute_shaping_reward,
    compute_step_reward,
)


@dataclass
class TransitionResult:
    success: bool
    message: str
    objective_reward: float = 0.0
    train_reward: float = 0.0
    error_code: str | None = None


def _require_case(state: WorldState, case_id: str | None) -> CaseState:
    if case_id is None or case_id not in state.cases:
        raise ValueError("unknown_case")
    return state.cases[case_id]


def _append_audit(state: WorldState, case_id: str | None, action_type: str, message: str, success: bool = True) -> None:
    state.audit_log.append(
        AuditEntry(
            timestamp=state.current_time,
            case_id=case_id,
            action_type=action_type,
            message=message,
            success=success,
        )
    )


def _tool_time_cost(action: OpsAction) -> int:
    if isinstance(action, AdvanceClockAction):
        return action.minutes
    return 1


def _sort_key(case: CaseState, sort_by: SortField) -> tuple:
    if sort_by == SortField.SLA_REMAINING:
        return (case.sla_deadline, case.priority)
    if sort_by == SortField.CREATED_AT:
        return (case.created_at, case.priority)
    if sort_by == SortField.AMOUNT:
        return (-case.amount, case.priority)
    return (case.priority, case.sla_deadline)


def _record_evidence(state: WorldState, case: CaseState, record_type: RecordType) -> None:
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
    if not evidence:
        return
    evidence_type, check_name = evidence
    case.gather_evidence(evidence_type)
    if check_name in case.required_check_names:
        case.mark_check(check_name)


def _linked_record_id(case: CaseState, record_type: RecordType) -> str | None:
    for record in case.linked_records:
        if record.record_type == record_type:
            return record.record_id
    return None


def _touch_case(case: CaseState, current_time: int) -> None:
    case.last_touched_at = current_time
    if case.follow_up_overdue:
        case.follow_up_overdue = False
        case.visible_flags = [flag for flag in case.visible_flags if flag != "follow_up_overdue"]
    if case.next_touch_at is not None and current_time >= case.next_touch_at:
        case.next_touch_at = None
        case.waiting_reason = None
        if case.status in {"waiting_follow_up", "waiting_external"}:
            case.status = "in_progress"


def _claimed_case_count(state: WorldState) -> int:
    return sum(1 for case in state.cases.values() if case.status != "closed" and case.claimed_by is not None)


def _expected_resolution(case: CaseState) -> Resolution:
    if case.case_type == CaseType.REFUND:
        return Resolution.REJECTED if case.true_fraud_risk > 0.7 else Resolution.APPROVED
    if case.case_type == CaseType.INVOICE:
        return Resolution.REJECTED if case.true_is_duplicate else Resolution.APPROVED
    if case.case_type == CaseType.KYC:
        if not case.true_doc_valid:
            return Resolution.REJECTED
        if not case.kyc_complete:
            return Resolution.DEFERRED
        return Resolution.APPROVED
    return Resolution.PENDING


def _can_close(case: CaseState) -> tuple[bool, str]:
    if case.resolution == Resolution.PENDING:
        return False, "case has no resolution"
    if case.next_touch_at is not None:
        return False, "follow up still scheduled"
    if case.follow_up_overdue:
        return False, "follow up overdue"
    if case.requires_customer_notification and not case.customer_notified:
        return False, "customer notification missing"
    if case.case_type == CaseType.KYC and case.resolution == Resolution.APPROVED and not case.kyc_complete:
        return False, "kyc incomplete"
    if (
        case.workflow_data.get("secondary_approval_required")
        and case.resolution == Resolution.APPROVED
        and case.workflow_data.get("approval_status") not in {"approved", "not_requested"}
    ):
        return False, "secondary approval incomplete"
    return True, ""


def _apply_action(state: WorldState, action: OpsAction) -> tuple[TransitionResult, CaseState | None, CaseState | None, WorldState]:
    prev_queue = state.queue_state().model_copy(deep=True)
    target_case = None
    prev_case = None
    if hasattr(action, "case_id"):
        action_case_id = getattr(action, "case_id", None)
        if action_case_id and action_case_id in state.cases:
            target_case = state.cases[action_case_id]
            prev_case = target_case.model_copy(deep=True)
            _touch_case(target_case, state.current_time)

    if isinstance(action, ListQueueAction):
        state.queue_order.sort(key=lambda case_id: _sort_key(state.cases[case_id], action.sort_by))
        return TransitionResult(True, "Queue listed"), None, None, prev_queue

    if isinstance(action, SearchCasesAction):
        return TransitionResult(True, f"Search returned {len(state.open_cases())} visible cases"), None, None, prev_queue

    if isinstance(action, OpenCaseAction):
        case = _require_case(state, action.case_id)
        state.current_case_id = case.case_id
        case.status = "in_progress" if case.status == "open" else case.status
        return TransitionResult(True, f"Opened {case.case_id}"), case, prev_case, prev_queue

    if isinstance(action, ViewRecordAction):
        case = target_case or (state.cases[state.current_case_id] if state.current_case_id else None)
        if case and action.record_type in case.forbidden_record_types:
            state.metrics.data_breach_count += 1
            raise ValueError("forbidden_record_access")
        state.current_record_type = action.record_type
        state.current_record_id = action.record_id
        if case:
            _record_evidence(state, case, action.record_type)
        return TransitionResult(True, f"Viewed {action.record_type.value}:{action.record_id}"), case, prev_case, prev_queue

    if isinstance(action, QueryPolicyAction):
        case = target_case or (state.cases[state.current_case_id] if state.current_case_id else None)
        policy = state.records.policies[action.policy_id]
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
        return TransitionResult(True, f"Policy {action.policy_id} queried"), case, prev_case, prev_queue

    if isinstance(action, InspectAuditAction):
        _require_case(state, action.case_id)
        return TransitionResult(True, f"Audit loaded for {action.case_id}"), target_case, prev_case, prev_queue

    if isinstance(action, LogInternalNoteAction):
        case = _require_case(state, action.case_id)
        case.internal_notes.append(action.note_code)
        return TransitionResult(True, f"Internal note logged for {case.case_id}"), case, prev_case, prev_queue

    if isinstance(action, SendMessageAction):
        case = _require_case(state, action.case_id)
        template = state.records.message_templates[action.template_id]
        body = template.body_template.format(**action.slots)
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
        return TransitionResult(True, f"Message sent via {template.channel}"), case, prev_case, prev_queue

    if isinstance(action, RequestInfoAction):
        case = _require_case(state, action.case_id)
        if case.requested_info_fields.count(action.field_name) >= 2:
            raise ValueError("duplicate_info_request")
        case.requested_info_fields.append(action.field_name)
        case.notifications_sent += 1
        latency = case.hidden_response_latency_minutes or 30
        payload = {"field_name": action.field_name, "record_id": None}
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
            payload["record"] = receipt
        if action.field_name == "individual.verification.document":
            payload["record_id"] = case.linked_records[-1].record_id
        schedule_event(state, state.current_time + latency, "info_response", case.case_id, payload)
        return TransitionResult(True, f"Requested {action.field_name}"), case, prev_case, prev_queue

    if isinstance(action, AssignAction):
        case = _require_case(state, action.case_id)
        case.current_owner = action.assignee_type
        return TransitionResult(True, f"Assigned {case.case_id} to {action.assignee_type}"), case, prev_case, prev_queue

    if isinstance(action, ClaimCaseAction):
        case = _require_case(state, action.case_id)
        if case.claimed_by is not None and case.claimed_by != action.assignee_type:
            raise ValueError("case_already_claimed")
        claim_capacity = state.metadata.get("claim_capacity", 2)
        if case.claimed_by is None and _claimed_case_count(state) >= claim_capacity:
            state.metrics.claim_overflow_attempts += 1
            raise ValueError("claim_capacity_reached")
        case.claimed_by = action.assignee_type
        case.claimed_at = state.current_time
        case.current_owner = action.assignee_type
        case.status = "claimed" if case.status == "open" else case.status
        state.current_case_id = case.case_id
        state.metrics.cases_claimed += 1
        return TransitionResult(True, f"Claimed {case.case_id}"), case, prev_case, prev_queue

    if isinstance(action, ReturnToQueueAction):
        case = _require_case(state, action.case_id)
        case.claimed_by = None
        case.claimed_at = None
        case.current_owner = "queue"
        case.status = "open" if case.status not in {"closed", "resolved"} else case.status
        case.workflow_data["return_reason"] = action.reason_code.value
        return TransitionResult(True, f"Returned {case.case_id} to queue"), case, prev_case, prev_queue

    if isinstance(action, RouteCaseAction):
        case = _require_case(state, action.case_id)
        if action.target_queue not in case.allowed_escalation_queues:
            raise ValueError("invalid_route_target")
        case.current_owner = action.assignee_type or action.target_queue.value
        case.status = "routed"
        case.workflow_data["active_queue"] = action.target_queue.value
        case.workflow_data["route_reason"] = action.reason_code.value
        route_history = case.workflow_data.setdefault("route_history", [])
        route_history.append(
            {
                "at_time": state.current_time,
                "queue": action.target_queue.value,
                "owner": case.current_owner,
                "reason": action.reason_code.value,
            }
        )
        return TransitionResult(True, f"Routed {case.case_id} to {action.target_queue.value}"), case, prev_case, prev_queue

    if isinstance(action, PrioritizeAction):
        case = _require_case(state, action.case_id)
        case.priority = int(action.new_priority)
        state.queue_order.sort(key=lambda cid: _sort_key(state.cases[cid], SortField.PRIORITY))
        return TransitionResult(True, f"Priority updated for {case.case_id}"), case, prev_case, prev_queue

    if isinstance(action, BatchReorderAction):
        if action.ordering_rule == "sla":
            state.queue_order.sort(key=lambda cid: _sort_key(state.cases[cid], SortField.SLA_REMAINING))
        elif action.ordering_rule == "amount":
            state.queue_order.sort(key=lambda cid: _sort_key(state.cases[cid], SortField.AMOUNT))
        else:
            state.queue_order.sort(key=lambda cid: _sort_key(state.cases[cid], SortField.PRIORITY))
        return TransitionResult(True, f"Queue reordered by {action.ordering_rule}"), None, None, prev_queue

    if isinstance(action, ScheduleFollowUpAction):
        case = _require_case(state, action.case_id)
        if action.follow_up_at <= state.current_time:
            raise ValueError("follow_up_must_be_future")
        case.next_touch_at = action.follow_up_at
        case.waiting_reason = action.reason_code.value
        case.follow_up_overdue = False
        case.visible_flags = [flag for flag in case.visible_flags if flag != "follow_up_overdue"]
        case.status = "waiting_follow_up"
        schedule_event(
            state,
            action.follow_up_at,
            "follow_up_due",
            case.case_id,
            {"scheduled_for": action.follow_up_at, "reason_code": action.reason_code.value},
        )
        return TransitionResult(True, f"Follow-up scheduled for {case.case_id}"), case, prev_case, prev_queue

    if isinstance(action, PauseSLAAction):
        case = _require_case(state, action.case_id)
        if case.workflow_data.get("sla_paused_at") is not None:
            raise ValueError("sla_already_paused")
        case.workflow_data["sla_paused_at"] = state.current_time
        case.workflow_data["sla_pause_reason"] = action.reason_code.value
        case.workflow_data["pre_pause_status"] = case.status
        case.status = "waiting_external"
        return TransitionResult(True, f"SLA paused for {case.case_id}"), case, prev_case, prev_queue

    if isinstance(action, ResumeSLAAction):
        case = _require_case(state, action.case_id)
        paused_at = case.workflow_data.get("sla_paused_at")
        if paused_at is None:
            raise ValueError("sla_not_paused")
        paused_minutes = max(0, state.current_time - paused_at)
        case.sla_deadline += paused_minutes
        previous_status = case.workflow_data.pop("pre_pause_status", "open")
        case.workflow_data.pop("sla_pause_reason", None)
        case.workflow_data.pop("sla_paused_at", None)
        case.status = "in_progress" if previous_status in {"open", "waiting_external"} else previous_status
        return TransitionResult(True, f"SLA resumed for {case.case_id}"), case, prev_case, prev_queue

    if isinstance(action, EscalateAction):
        case = _require_case(state, action.case_id)
        if action.target_queue not in case.allowed_escalation_queues:
            raise ValueError("invalid_escalation_target")
        case.resolution = Resolution.ESCALATED
        case.status = "escalated"
        case.escalation_justified = action.reason_code in {ReasonCode.THRESHOLD_EXCEEDED, ReasonCode.POLICY_AMBIGUITY}
        state.metadata["escalation_queue_load"] = state.metadata.get("escalation_queue_load", 0) + 1
        state.metrics.escalation_count += 1
        return TransitionResult(True, f"Escalated {case.case_id} to {action.target_queue.value}"), case, prev_case, prev_queue

    if isinstance(action, DeferAction):
        case = _require_case(state, action.case_id)
        case.resolution = Resolution.DEFERRED
        case.status = "pending_info"
        return TransitionResult(True, f"Deferred {case.case_id}"), case, prev_case, prev_queue

    if isinstance(action, ExecuteRefundAction):
        case = _require_case(state, action.case_id)
        if case.case_type != CaseType.REFUND:
            raise ValueError("refund_execution_not_supported")
        if case.resolution != Resolution.APPROVED:
            raise ValueError("refund_not_approved")
        payment_id = _linked_record_id(case, RecordType.PAYMENT)
        if payment_id is None:
            raise ValueError("payment_record_missing")
        payment = state.records.payments[payment_id]
        approved_amount = action.approved_amount if action.approved_amount is not None else case.amount
        payment.status = "refunded" if approved_amount >= case.amount else "partially_refunded"
        case.workflow_data["refund_execution_state"] = payment.status
        case.workflow_data["refunded_amount"] = approved_amount
        return TransitionResult(True, f"Refund executed for {case.case_id}"), case, prev_case, prev_queue

    if isinstance(action, AcceptDisputeAction):
        case = _require_case(state, action.case_id)
        if case.case_type != CaseType.REFUND:
            raise ValueError("dispute_workflow_not_supported")
        dispute_id = _linked_record_id(case, RecordType.DISPUTE)
        if dispute_id is None:
            raise ValueError("dispute_record_missing")
        dispute = state.records.disputes[dispute_id]
        dispute.status = "lost"
        case.workflow_data["dispute_stage"] = "finalized"
        case.workflow_data["dispute_workflow_status"] = "accepted"
        case.workflow_data["dispute_resolution"] = "accepted"
        case.resolution = Resolution.APPROVED
        case.status = "resolved"
        state.metrics.disputes_accepted += 1
        return TransitionResult(True, f"Accepted dispute for {case.case_id}"), case, prev_case, prev_queue

    if isinstance(action, SubmitDisputeEvidenceAction):
        case = _require_case(state, action.case_id)
        if case.case_type != CaseType.REFUND:
            raise ValueError("dispute_workflow_not_supported")
        dispute_id = _linked_record_id(case, RecordType.DISPUTE)
        if dispute_id is None:
            raise ValueError("dispute_record_missing")
        dispute = state.records.disputes[dispute_id]
        if state.current_time > dispute.evidence_due_by:
            raise ValueError("dispute_deadline_passed")
        submitted = case.workflow_data.setdefault("dispute_evidence_fields", [])
        for field in action.evidence_fields:
            if field not in submitted:
                submitted.append(field)
        dispute.submission_count += 1
        dispute.status = "under_review"
        stage = case.workflow_data.get("dispute_stage", "chargeback_open")
        case.workflow_data["dispute_stage"] = "evidence_submitted"
        case.workflow_data["dispute_workflow_status"] = "submitted"
        case.gather_evidence("dispute_package")
        case.mark_check("review_dispute")
        outcome = "won"
        if case.true_fraud_risk > 0.7:
            outcome = "lost"
        elif stage == "pre_arbitration":
            outcome = "won" if len(submitted) >= 3 else "lost"
        elif len(submitted) < 2:
            outcome = "pre_arbitration"
        schedule_event(
            state,
            state.current_time + 20,
            "dispute_outcome",
            case.case_id,
            {"outcome": outcome},
        )
        return TransitionResult(True, f"Dispute evidence submitted for {case.case_id}"), case, prev_case, prev_queue

    if isinstance(action, RecordThreeWayMatchAction):
        case = _require_case(state, action.case_id)
        if case.case_type != CaseType.INVOICE:
            raise ValueError("three_way_match_not_supported")
        case.workflow_data["match_status"] = action.match_status.value
        if action.match_status.value == "duplicate":
            case.workflow_data["duplicate_status"] = "confirmed"
        elif action.match_status.value == "matched":
            case.workflow_data["duplicate_status"] = "false_positive"
        if action.variance_amount is not None:
            case.workflow_data["variance_amount"] = action.variance_amount
        elif "variance_amount" in case.workflow_data:
            case.workflow_data.pop("variance_amount")
        case.mark_check("review_invoice")
        case.mark_check("review_po")
        if action.match_status.value == "matched":
            case.mark_check("review_receipt")
        if action.match_status.value == "missing_receipt":
            if "goods_receipt" not in case.pending_info_fields:
                case.pending_info_fields.append("goods_receipt")
            case.status = "pending_info"
        return TransitionResult(True, f"Three-way match recorded for {case.case_id}"), case, prev_case, prev_queue

    if isinstance(action, PlacePaymentHoldAction):
        case = _require_case(state, action.case_id)
        if case.case_type != CaseType.INVOICE:
            raise ValueError("payment_hold_not_supported")
        case.workflow_data["payment_hold"] = True
        case.workflow_data["payment_hold_reason"] = action.reason_code.value
        case.workflow_data["payment_hold_set_at"] = state.current_time
        if "payment_hold" not in case.visible_flags:
            case.visible_flags.append("payment_hold")
        return TransitionResult(True, f"Payment hold placed for {case.case_id}"), case, prev_case, prev_queue

    if isinstance(action, ReleasePaymentHoldAction):
        case = _require_case(state, action.case_id)
        if case.case_type != CaseType.INVOICE:
            raise ValueError("payment_hold_not_supported")
        if not case.workflow_data.get("payment_hold"):
            raise ValueError("payment_hold_not_active")
        if case.workflow_data.get("credit_memo_status") == "requested":
            raise ValueError("credit_memo_pending")
        case.workflow_data["payment_hold"] = False
        case.workflow_data["payment_hold_released_at"] = state.current_time
        case.workflow_data.pop("payment_hold_reason", None)
        case.visible_flags = [flag for flag in case.visible_flags if flag != "payment_hold"]
        return TransitionResult(True, f"Payment hold released for {case.case_id}"), case, prev_case, prev_queue

    if isinstance(action, RequestCreditMemoAction):
        case = _require_case(state, action.case_id)
        if case.case_type != CaseType.INVOICE:
            raise ValueError("credit_memo_not_supported")
        if case.workflow_data.get("credit_memo_status") == "requested":
            raise ValueError("credit_memo_already_requested")
        requested_amount = action.approved_amount or case.workflow_data.get("variance_amount") or case.workflow_data.get(
            "credit_memo_amount",
            max(1.0, round(case.amount * 0.2, 2)),
        )
        case.workflow_data["credit_memo_status"] = "requested"
        case.workflow_data["credit_memo_amount"] = requested_amount
        case.workflow_data["payment_hold"] = True
        case.workflow_data["payment_hold_reason"] = "credit_memo_pending"
        if "payment_hold" not in case.visible_flags:
            case.visible_flags.append("payment_hold")
        state.metrics.credit_memos_requested += 1
        schedule_event(
            state,
            state.current_time + (case.hidden_response_latency_minutes or 30),
            "vendor_credit_memo_received",
            case.case_id,
            {"amount": requested_amount},
        )
        return TransitionResult(True, f"Requested credit memo for {case.case_id}"), case, prev_case, prev_queue

    if isinstance(action, SendForSecondaryApprovalAction):
        case = _require_case(state, action.case_id)
        if case.workflow_data.get("approval_status") == "pending_secondary":
            raise ValueError("secondary_approval_pending")
        state.metadata["approval_queue_load"] = state.metadata.get("approval_queue_load", 0) + 1
        case.workflow_data["approval_status"] = "pending_secondary"
        case.workflow_data["approval_reason"] = action.reason_code.value
        case.workflow_data["approval_assignee"] = action.assignee_type or "manager_review"
        case.workflow_data.setdefault("approval_chain", []).append(
            {
                "at_time": state.current_time,
                "status": "pending_secondary",
                "owner": action.assignee_type or "manager_review",
                "reason": action.reason_code.value,
            }
        )
        case.status = "pending_approval"
        state.metrics.secondary_approvals_requested += 1
        expected = case.workflow_data.get("approval_expected_outcome")
        if expected is None:
            if case.case_type == CaseType.REFUND:
                expected = "denied" if case.true_fraud_risk > 0.7 else "approved"
            elif case.case_type == CaseType.INVOICE:
                expected = "denied" if case.true_is_duplicate else "approved"
            else:
                expected = "approved"
        schedule_event(
            state,
            state.current_time + 15 + max(0, state.metadata.get("approval_queue_load", 1) - 1) * 10,
            "secondary_approval_decision",
            case.case_id,
            {"outcome": expected},
        )
        return TransitionResult(True, f"Secondary approval requested for {case.case_id}"), case, prev_case, prev_queue

    if isinstance(action, ReviewKYCAction):
        case = _require_case(state, action.case_id)
        if case.case_type != CaseType.KYC:
            raise ValueError("kyc_review_not_supported")
        verification_id = _linked_record_id(case, RecordType.KYC_DOCUMENT)
        if verification_id is None:
            raise ValueError("kyc_record_missing")
        verification = state.records.kyc_verifications[verification_id]
        case.mark_check("review_kyc_profile")
        case.mark_check("review_document")
        if action.verification_decision.value == "approve":
            verification.status = "verified"
            verification.requirements_currently_due = []
            case.kyc_complete = True
            if not case.true_doc_valid:
                state.metrics.compliance_violations += 1
        elif action.verification_decision.value == "request_resubmission":
            verification.status = "requires_input"
            verification.requirements_currently_due = (
                verification.requirements_currently_due
                or case.hidden_required_documents
                or ["individual.verification.document"]
            )
            case.pending_info_fields = list(dict.fromkeys(case.pending_info_fields + verification.requirements_currently_due))
            case.kyc_complete = False
        else:
            verification.status = "rejected"
            case.kyc_complete = False
        case.workflow_data["verification_status"] = verification.status
        case.workflow_data["requirements_due"] = verification.requirements_currently_due.copy()
        case.workflow_data["payout_hold"] = not case.kyc_complete
        return TransitionResult(True, f"KYC review completed for {case.case_id}"), case, prev_case, prev_queue

    if isinstance(action, TriggerReverificationAction):
        case = _require_case(state, action.case_id)
        if case.case_type != CaseType.KYC:
            raise ValueError("kyc_reverification_not_supported")
        verification_id = _linked_record_id(case, RecordType.KYC_DOCUMENT)
        if verification_id is None:
            raise ValueError("kyc_record_missing")
        verification = state.records.kyc_verifications[verification_id]
        requirements = action.requirements or case.hidden_required_documents or ["individual.verification.document"]
        verification.status = "requires_input"
        verification.requirements_currently_due = list(dict.fromkeys(requirements))
        case.pending_info_fields = list(dict.fromkeys(case.pending_info_fields + verification.requirements_currently_due))
        case.kyc_complete = False
        case.workflow_data["verification_status"] = verification.status
        case.workflow_data["requirements_due"] = verification.requirements_currently_due.copy()
        case.workflow_data["payout_hold"] = True
        return TransitionResult(True, f"Reverification triggered for {case.case_id}"), case, prev_case, prev_queue

    if isinstance(action, ApproveAction):
        case = _require_case(state, action.case_id)
        if (
            case.workflow_data.get("secondary_approval_required")
            and case.workflow_data.get("approval_status") not in {"approved", "not_requested"}
        ):
            raise ValueError("secondary_approval_required")
        if case.workflow_data.get("credit_memo_status") == "requested":
            raise ValueError("credit_memo_pending")
        case.resolution = Resolution.APPROVED
        case.status = "resolved"
        if case.case_type == CaseType.KYC and not case.kyc_complete:
            state.metrics.compliance_violations += 1
        if case.case_type == CaseType.REFUND and case.true_fraud_risk > 0.7:
            schedule_event(state, state.current_time + 30, "chargeback", case.case_id)
        if case.case_type == CaseType.INVOICE and case.true_is_duplicate:
            state.metrics.duplicate_payments += 1
        return TransitionResult(True, f"Approved {case.case_id}"), case, prev_case, prev_queue

    if isinstance(action, RejectAction):
        case = _require_case(state, action.case_id)
        case.resolution = Resolution.REJECTED
        case.status = "resolved"
        return TransitionResult(True, f"Rejected {case.case_id}"), case, prev_case, prev_queue

    if isinstance(action, CloseCaseAction):
        case = _require_case(state, action.case_id)
        allowed, reason = _can_close(case)
        if not allowed:
            raise ValueError(reason.replace(" ", "_"))
        case.status = "closed"
        case.terminal_reason = action.resolution_code
        state.metrics.cases_resolved += 1
        state.queue_order = [cid for cid in state.queue_order if state.cases[cid].status != "closed"] + [
            cid for cid in state.queue_order if state.cases[cid].status == "closed"
        ]
        if case.requires_customer_notification and not case.customer_notified:
            schedule_event(state, state.current_time + 15, "reopen", case.case_id)
        return TransitionResult(True, f"Closed {case.case_id}"), case, prev_case, prev_queue

    if isinstance(action, ReopenCaseAction):
        case = _require_case(state, action.case_id)
        case.status = "reopened"
        case.resolution = Resolution.PENDING
        return TransitionResult(True, f"Reopened {case.case_id}"), case, prev_case, prev_queue

    if isinstance(action, AdvanceClockAction):
        state.current_time += action.minutes
        state.metrics.simulated_minutes += action.minutes
        messages = process_due_events(state)
        return TransitionResult(True, "; ".join(messages) if messages else f"Advanced clock by {action.minutes}m"), target_case, prev_case, prev_queue

    raise ValueError("unsupported_action")


def apply_action(state: WorldState, action: OpsAction) -> TransitionResult:
    state.step_count += 1
    state.metrics.tool_calls += 1

    try:
        result, case, prev_case, prev_queue = _apply_action(state, action)
        if not isinstance(action, AdvanceClockAction):
            state.current_time += _tool_time_cost(action)
            state.metrics.simulated_minutes += _tool_time_cost(action)
            process_due_events(state)
        reward_breakdown = RewardBreakdown(case_id=case.case_id if case else "queue", case_type=(case.case_type if case else CaseType.TRIAGE))
        if case:
            t_total = case.workflow_data.get("sla_total", max(1, case.sla_deadline - case.created_at))
            t_remaining = case.sla_deadline - state.current_time
            reward_breakdown = compute_step_reward(
                case=case,
                queue=state.queue_state(),
                action_type=action.action_type,
                episode_metrics=state.metrics,
                t_remaining=t_remaining,
                t_total=t_total,
                evidence_type=case.evidence_types_gathered[-1] if case.evidence_types_gathered else None,
                evidence_items_before=prev_case.evidence_items_gathered if prev_case else 0,
                evidence_items_after=case.evidence_items_gathered,
                evidence_time_cost=_tool_time_cost(action),
                evidence_already_gathered=set(prev_case.evidence_types_gathered) if prev_case else None,
            )
            shaping_reward = compute_shaping_reward(
                case=case,
                queue=state.queue_state(),
                prev_case=prev_case,
                prev_queue=prev_queue,
            )
        else:
            shaping_reward = 0.0
        objective_reward = reward_breakdown.objective_total
        if state.task_id == TaskId.QUEUE_TRIAGE and all(case.status == "closed" for case in state.cases.values()):
            queue_reward = compute_queue_reward(
                queue=state.queue_state(),
                initial_case_count=len(state.cases),
                initial_backlog=len(state.cases),
                resolved_cases=list(state.cases.values()),
                remaining_cases=[],
            )
            objective_reward += queue_reward["total"]
        train_reward = objective_reward + shaping_reward
        state.objective_score += objective_reward
        state.train_score += train_reward
        state.last_action_result = result.message
        _append_audit(state, case.case_id if case else None, action.action_type, result.message)
        result.objective_reward = objective_reward
        result.train_reward = train_reward
        return result
    except Exception as exc:
        state.metrics.invalid_actions += 1
        message = str(exc)
        _append_audit(state, getattr(action, "case_id", None), action.action_type, message, success=False)
        state.objective_score += -2.0
        state.train_score += -2.0
        state.last_action_result = message
        return TransitionResult(
            success=False,
            message=message.replace("_", " "),
            objective_reward=-2.0,
            train_reward=-2.0,
            error_code="invalid_action",
        )
