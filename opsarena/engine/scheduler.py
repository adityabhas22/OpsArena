from __future__ import annotations

from opsarena.documents import CreditMemoRecord
from opsarena.domain.core import ApprovalHistoryEntry, QAStatus
from opsarena.domain.events import (
    ChargebackEvent,
    DisputeOutcomeEvent,
    FollowUpDueEvent,
    InfoResponseEvent,
    ReworkDueEvent,
    ReopenEvent,
    SLABreachEvent,
    ScheduledEvent,
    SecondaryApprovalDecisionEvent,
    VendorCreditMemoReceivedEvent,
)
from opsarena.domain.workflows.invoice import ApprovalStatus, CreditMemoStatus, DuplicateStatus
from opsarena.domain.workflows.kyc import KYCStage
from opsarena.domain.workflows.refund import DisputeResolution, DisputeStage
from opsarena.engine.handlers.common import require_invoice_workflow, require_kyc_workflow, require_refund_workflow
from opsarena.engine.state import AuditEntry, WorldState
from opsarena.enums import CaseType, RecordType, Resolution


def schedule_event(state: WorldState, event: ScheduledEvent) -> None:
    state.scheduled_events.append(event)
    state.scheduled_events.sort(key=lambda queued_event: queued_event.at_time)


def process_due_events(state: WorldState) -> list[str]:
    messages: list[str] = []
    due = [event for event in state.scheduled_events if event.at_time <= state.current_time]
    state.scheduled_events = [event for event in state.scheduled_events if event.at_time > state.current_time]
    for event in due:
        case = state.cases[event.case_id]
        if isinstance(event, InfoResponseEvent):
            field_name = event.field_name
            if field_name in case.pending_info_fields:
                case.pending_info_fields.remove(field_name)
            if field_name == "goods_receipt":
                receipt = event.receipt_record
                assert receipt is not None
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
                assert event.record_id is not None
                verification = state.records.kyc_verifications[event.record_id]
                verification.status = "verified"
                verification.requirements_currently_due = []
                workflow = require_kyc_workflow(case)
                workflow.kyc_complete = True
                workflow.verification_status = verification.status
                workflow.requirements_due = []
                workflow.kyc_stage = KYCStage.PENDING_REVIEW
            messages.append(f"{field_name} received for {case.case_id}")
        elif isinstance(event, ChargebackEvent):
            case.status = "reopened"
            state.metrics.chargebacks += 1
            state.metrics.reopens += 1
            require_refund_workflow(case).dispute_stage = DisputeStage.CHARGEBACK_OPEN
            messages.append(f"Chargeback triggered on {case.case_id}")
        elif isinstance(event, DisputeOutcomeEvent):
            workflow = require_refund_workflow(case)
            outcome = event.outcome
            workflow.dispute_workflow_status = outcome
            if outcome == "won":
                workflow.dispute_stage = DisputeStage.WON
                workflow.dispute_resolution = DisputeResolution.WON
                if case.resolution == Resolution.PENDING:
                    case.resolution = Resolution.APPROVED
                    case.status = "resolved"
                messages.append(f"Dispute won for {case.case_id}")
            elif outcome == "pre_arbitration":
                case.status = "reopened"
                workflow.dispute_stage = DisputeStage.PRE_ARBITRATION
                workflow.dispute_workflow_status = "pre_arbitration"
                state.metrics.reopens += 1
                messages.append(f"Pre-arbitration received for {case.case_id}")
            else:
                case.status = "reopened"
                workflow.dispute_stage = DisputeStage.LOST
                workflow.dispute_resolution = DisputeResolution.LOST
                state.metrics.chargebacks += 1
                state.metrics.reopens += 1
                messages.append(f"Dispute lost for {case.case_id}")
        elif isinstance(event, VendorCreditMemoReceivedEvent):
            amount = int(round(float(event.amount) * 100))
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
            workflow = require_invoice_workflow(case)
            workflow.credit_memo_status = CreditMemoStatus.RECEIVED
            workflow.payment_hold = True
            workflow.duplicate_status = DuplicateStatus.FALSE_POSITIVE if not case.hidden.true_is_duplicate else DuplicateStatus.CONFIRMED
            messages.append(f"Vendor credit memo received for {case.case_id}")
        elif isinstance(event, SecondaryApprovalDecisionEvent):
            outcome = event.outcome
            state.metadata["approval_queue_load"] = max(0, state.metadata.get("approval_queue_load", 0) - 1)
            if case.case_type == CaseType.INVOICE:
                workflow = require_invoice_workflow(case)
                workflow.approval_status = ApprovalStatus(outcome)
                workflow.approval_chain.append(
                    ApprovalHistoryEntry(
                        at_time=state.current_time,
                        status=workflow.approval_status.value,
                        owner=workflow.approval_assignee or "manager_review",
                    )
                )
            else:
                workflow = require_refund_workflow(case)
                workflow.approval_status = outcome
            case.status = "in_progress"
            messages.append(f"Secondary approval {outcome} for {case.case_id}")
        elif isinstance(event, FollowUpDueEvent):
            if case.status != "closed" and case.next_touch_at == event.scheduled_for:
                case.follow_up_overdue = True
                if "follow_up_overdue" not in case.visible_flags:
                    case.visible_flags.append("follow_up_overdue")
                state.metrics.follow_ups_overdue += 1
                messages.append(f"Follow-up overdue for {case.case_id}")
        elif isinstance(event, ReworkDueEvent):
            if case.status != "closed" and case.rework_due_at == event.scheduled_for and case.qa_status == QAStatus.FAILED:
                case.qa_rework_overdue = True
                if "qa_rework_overdue" not in case.visible_flags:
                    case.visible_flags.append("qa_rework_overdue")
                state.metrics.qa_rework_overdue += 1
                messages.append(f"QA rework overdue for {case.case_id}")
        elif isinstance(event, ReopenEvent):
            case.status = "reopened"
            state.metrics.reopens += 1
            messages.append(f"Case {case.case_id} reopened")
        elif isinstance(event, SLABreachEvent):
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
