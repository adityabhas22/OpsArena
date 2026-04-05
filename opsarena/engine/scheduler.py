from __future__ import annotations

from uuid import uuid4

from opsarena.documents import CreditMemoRecord
from opsarena.engine.state import AuditEntry, ScheduledEvent, WorldState
from opsarena.enums import RecordType, Resolution


def schedule_event(state: WorldState, at_time: int, event_type: str, case_id: str, payload: dict | None = None) -> None:
    state.scheduled_events.append(
        ScheduledEvent(
            event_id=f"evt-{uuid4().hex[:8]}",
            at_time=at_time,
            event_type=event_type,
            case_id=case_id,
            payload=payload or {},
        )
    )
    state.scheduled_events.sort(key=lambda event: event.at_time)


def process_due_events(state: WorldState) -> list[str]:
    messages: list[str] = []
    due = [event for event in state.scheduled_events if event.at_time <= state.current_time]
    state.scheduled_events = [event for event in state.scheduled_events if event.at_time > state.current_time]
    for event in due:
        case = state.cases[event.case_id]
        if event.event_type == "info_response":
            field_name = event.payload["field_name"]
            if field_name in case.pending_info_fields:
                case.pending_info_fields.remove(field_name)
            if field_name == "goods_receipt":
                receipt = event.payload["record"]
                state.records.receipts[receipt.receipt_id] = receipt
                case.linked_records.append(
                    type(case.linked_records[0])(
                        record_type=RecordType.RECEIPT,
                        record_id=receipt.receipt_id,
                        title=f"Receipt {receipt.receipt_id}",
                    )
                )
                case.evidence_types_available.append("goods_receipt")
                case.evidence_items_available = len(case.evidence_types_available)
            if field_name == "individual.verification.document":
                verification = state.records.kyc_verifications[event.payload["record_id"]]
                verification.status = "verified"
                verification.requirements_currently_due = []
                case.kyc_complete = True
                case.workflow_data["verification_status"] = verification.status
                case.workflow_data["requirements_due"] = []
                case.workflow_data["kyc_stage"] = "pending_review"
            messages.append(f"{field_name} received for {case.case_id}")
        elif event.event_type == "chargeback":
            case.status = "reopened"
            state.metrics.chargebacks += 1
            state.metrics.reopens += 1
            case.workflow_data["dispute_stage"] = "chargeback_open"
            messages.append(f"Chargeback triggered on {case.case_id}")
        elif event.event_type == "dispute_outcome":
            outcome = event.payload["outcome"]
            case.workflow_data["dispute_stage"] = outcome
            case.workflow_data["dispute_workflow_status"] = outcome
            if outcome == "won":
                case.workflow_data["dispute_resolution"] = "won"
                if case.resolution == Resolution.PENDING:
                    case.resolution = Resolution.APPROVED
                    case.status = "resolved"
                messages.append(f"Dispute won for {case.case_id}")
            elif outcome == "pre_arbitration":
                case.status = "reopened"
                case.workflow_data["dispute_stage"] = "pre_arbitration"
                case.workflow_data["dispute_workflow_status"] = "pre_arbitration"
                state.metrics.reopens += 1
                messages.append(f"Pre-arbitration received for {case.case_id}")
            else:
                case.status = "reopened"
                case.workflow_data["dispute_resolution"] = "lost"
                state.metrics.chargebacks += 1
                state.metrics.reopens += 1
                messages.append(f"Dispute lost for {case.case_id}")
        elif event.event_type == "vendor_credit_memo_received":
            amount = int(round(float(event.payload["amount"]) * 100))
            invoice_id = next((record.record_id for record in case.linked_records if record.record_type == RecordType.INVOICE), None)
            memo = CreditMemoRecord(
                credit_memo_id=f"cm_{case.case_id}",
                invoice_id=invoice_id or "unknown_invoice",
                vendor_id=case.vendor_id or "vendor_1",
                vendor_name="Acme Supply",
                amount=amount,
                currency=case.currency,
                reason="price_variance",
                status="issued",
                created_at=state.current_time,
                expected_apply_date=state.current_time + 5,
            )
            state.records.credit_memos[memo.credit_memo_id] = memo
            case.linked_records.append(
                type(case.linked_records[0])(
                    record_type=RecordType.CREDIT_MEMO,
                    record_id=memo.credit_memo_id,
                    title=f"Credit memo {memo.credit_memo_id}",
                )
            )
            case.workflow_data["credit_memo_status"] = "received"
            case.workflow_data["payment_hold"] = True
            case.workflow_data["duplicate_status"] = "false_positive" if not case.true_is_duplicate else "confirmed"
            messages.append(f"Vendor credit memo received for {case.case_id}")
        elif event.event_type == "secondary_approval_decision":
            outcome = event.payload["outcome"]
            state.metadata["approval_queue_load"] = max(0, state.metadata.get("approval_queue_load", 0) - 1)
            case.workflow_data["approval_status"] = outcome
            case.workflow_data.setdefault("approval_chain", []).append(
                {
                    "at_time": state.current_time,
                    "status": outcome,
                    "owner": case.workflow_data.get("approval_assignee", "manager_review"),
                }
            )
            case.status = "in_progress"
            messages.append(f"Secondary approval {outcome} for {case.case_id}")
        elif event.event_type == "follow_up_due":
            if case.status != "closed" and case.next_touch_at == event.payload.get("scheduled_for"):
                case.follow_up_overdue = True
                if "follow_up_overdue" not in case.visible_flags:
                    case.visible_flags.append("follow_up_overdue")
                state.metrics.follow_ups_overdue += 1
                messages.append(f"Follow-up overdue for {case.case_id}")
        elif event.event_type == "reopen":
            case.status = "reopened"
            state.metrics.reopens += 1
            messages.append(f"Case {case.case_id} reopened")
        elif event.event_type == "sla_breach":
            if case.status != "closed":
                case.status = "breached"
                state.metrics.cases_breached += 1
                messages.append(f"SLA breached for {case.case_id}")
        state.audit_log.append(
            AuditEntry(
                timestamp=state.current_time,
                case_id=case.case_id,
                action_type=f"event:{event.event_type}",
                message=messages[-1] if messages else event.event_type,
            )
        )
    return messages
