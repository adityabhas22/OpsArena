from __future__ import annotations

from opsarena.documents import CreditMemoRecord
from opsarena.domain.core import ApprovalHistoryEntry, QAStatus
from opsarena.domain.events import (
    ArrivalWaveEvent,
    ChargebackEvent,
    DisputeOutcomeEvent,
    EDDResponseDueEvent,
    FollowUpDueEvent,
    InfoResponseEvent,
    InquiryEscalatesToChargebackEvent,
    MonitoringThresholdBreachedEvent,
    POChangeApprovedEvent,
    PaymentBatchExecutedEvent,
    PrearbitrationReceivedEvent,
    QASampleSelectedEvent,
    ReportDeadlineMissedEvent,
    ReserveReleaseDueEvent,
    ReworkDueEvent,
    ReopenEvent,
    SanctionsFalsePositiveClearedEvent,
    SLABreachEvent,
    ScheduledEvent,
    StaffingDropEvent,
    StopPaymentConfirmedEvent,
    SecondaryApprovalDecisionEvent,
    VendorCreditMemoReceivedEvent,
    VendorRefundReceivedEvent,
    VendorRevisedInvoiceEvent,
)
from opsarena.domain.workflows.invoice import (
    ApprovalStatus,
    CreditMemoStatus,
    DuplicateStatus,
    POChangeStatus,
    PaymentBatchStatus,
    RecoveryStatus,
    VendorResponseStatus,
)
from opsarena.domain.workflows.kyc import BeneficialOwnerStatus, EDDStatus, KYCStage, OFACReportStatus, SanctionsStatus
from opsarena.domain.workflows.refund import DisputeResolution, DisputeStage, MonitoringProgramStatus
from opsarena.engine.handlers.common import (
    require_invoice_workflow,
    require_kyc_workflow,
    require_refund_workflow,
    sync_kyc_flags,
    sync_refund_risk_flags,
)
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
        case = state.cases[event.case_id] if event.case_id is not None else None
        if isinstance(event, InfoResponseEvent):
            assert case is not None
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
                verification.status = "pending_review"
                verification.requirements_currently_due = []
                verification.error_code = None if case.hidden.true_doc_valid else verification.error_code
                workflow = require_kyc_workflow(case)
                workflow.kyc_complete = False
                workflow.verification_status = verification.status
                workflow.requirements_due = []
                workflow.kyc_stage = KYCStage.PENDING_REVIEW
                sync_kyc_flags(case)
            elif case.case_type == CaseType.KYC:
                workflow = require_kyc_workflow(case)
                verification = state.records.kyc_verifications[next(
                    record.record_id for record in case.linked_records if record.record_type == RecordType.KYC_DOCUMENT
                )]
                workflow.correction_fields = [field for field in workflow.correction_fields if field != field_name]
                verification.correction_requests = [field for field in verification.correction_requests if field != field_name]
                if not workflow.correction_fields:
                    workflow.beneficial_owner_status = BeneficialOwnerStatus.PENDING_REVIEW
                    workflow.edd_status = EDDStatus.IN_PROGRESS
                sync_kyc_flags(case)
            messages.append(f"{field_name} received for {case.case_id}")
        elif isinstance(event, ChargebackEvent):
            assert case is not None
            case.status = "reopened"
            state.metrics.chargebacks += 1
            state.metrics.reopens += 1
            workflow = require_refund_workflow(case)
            workflow.dispute_stage = DisputeStage.CHARGEBACK_OPEN
            workflow.chargeback_received_at = state.current_time
            workflow.representment_due_at = state.current_time + 18
            messages.append(f"Chargeback triggered on {case.case_id}")
        elif isinstance(event, DisputeOutcomeEvent):
            assert case is not None
            workflow = require_refund_workflow(case)
            outcome = event.outcome
            workflow.dispute_workflow_status = outcome
            if outcome == "won":
                workflow.dispute_stage = DisputeStage.WON
                workflow.dispute_resolution = DisputeResolution.WON
                if case.resolution == Resolution.PENDING:
                    case.resolution = Resolution.APPROVED
                    case.status = "resolved"
                workflow.merchant_dispute_ratio_30d = max(0.0, round(workflow.merchant_dispute_ratio_30d - 0.0005, 4))
                sync_refund_risk_flags(case)
                messages.append(f"Dispute won for {case.case_id}")
            else:
                case.status = "reopened"
                workflow.dispute_stage = DisputeStage.LOST
                workflow.dispute_resolution = DisputeResolution.LOST
                workflow.merchant_dispute_ratio_30d = round(workflow.merchant_dispute_ratio_30d + 0.003, 4)
                if case.hidden.true_fraud_risk > 0.7:
                    workflow.merchant_fraud_ratio_30d = round(workflow.merchant_fraud_ratio_30d + 0.002, 4)
                sync_refund_risk_flags(case)
                if (
                    workflow.monitoring_program_status == MonitoringProgramStatus.BREACHED
                    and not any(
                        pending.event_type == "monitoring_threshold_breached" and pending.case_id == case.case_id
                        for pending in state.scheduled_events
                    )
                ):
                    schedule_event(
                        state,
                        MonitoringThresholdBreachedEvent(at_time=state.current_time + 5, case_id=case.case_id),
                    )
                state.metrics.chargebacks += 1
                state.metrics.reopens += 1
                messages.append(f"Dispute lost for {case.case_id}")
        elif isinstance(event, InquiryEscalatesToChargebackEvent):
            assert case is not None
            workflow = require_refund_workflow(case)
            if workflow.dispute_stage == DisputeStage.INQUIRY and workflow.dispute_resolution == DisputeResolution.PENDING:
                workflow.dispute_stage = DisputeStage.CHARGEBACK_OPEN
                workflow.dispute_workflow_status = "chargeback_open"
                workflow.chargeback_received_at = state.current_time
                workflow.representment_due_at = state.current_time + 18
                workflow.pre_dispute_due_at = None
                case.status = "reopened"
                state.metrics.chargebacks += 1
                state.metrics.reopens += 1
                messages.append(f"Inquiry escalated to chargeback for {case.case_id}")
        elif isinstance(event, PrearbitrationReceivedEvent):
            assert case is not None
            workflow = require_refund_workflow(case)
            if workflow.dispute_resolution == DisputeResolution.PENDING:
                case.status = "reopened"
                workflow.dispute_stage = DisputeStage.PRE_ARBITRATION
                workflow.dispute_workflow_status = "pre_arbitration"
                workflow.prearbitration_due_at = state.current_time + 10
                state.metrics.reopens += 1
                messages.append(f"Pre-arbitration received for {case.case_id}")
        elif isinstance(event, VendorCreditMemoReceivedEvent):
            assert case is not None
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
            assert case is not None
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
            assert case is not None
            if case.status != "closed" and case.next_touch_at == event.scheduled_for:
                case.follow_up_overdue = True
                if "follow_up_overdue" not in case.visible_flags:
                    case.visible_flags.append("follow_up_overdue")
                state.metrics.follow_ups_overdue += 1
                messages.append(f"Follow-up overdue for {case.case_id}")
        elif isinstance(event, ReworkDueEvent):
            assert case is not None
            if case.status != "closed" and case.rework_due_at == event.scheduled_for and case.qa_status == QAStatus.FAILED:
                case.qa_rework_overdue = True
                if "qa_rework_overdue" not in case.visible_flags:
                    case.visible_flags.append("qa_rework_overdue")
                state.metrics.qa_rework_overdue += 1
                messages.append(f"QA rework overdue for {case.case_id}")
        elif isinstance(event, QASampleSelectedEvent):
            assert case is not None
            if case.status == "closed" and case.qa_status == QAStatus.NOT_REQUESTED:
                case.qa_required = True
                case.qa_status = QAStatus.PENDING
                case.qa_owner = "qa_queue"
                case.status = "pending_qa"
                if "qa_sampled" not in case.visible_flags:
                    case.visible_flags.append("qa_sampled")
                state.metrics.qa_reviews_requested += 1
                state.metrics.reopens += 1
                state.metrics.cases_resolved = max(0, state.metrics.cases_resolved - 1)
                messages.append(f"QA sample selected for {case.case_id}")
        elif isinstance(event, ReserveReleaseDueEvent):
            assert case is not None
            workflow = require_refund_workflow(case)
            if workflow.reserve_percent > 0 and workflow.reserve_release_due_at == event.at_time:
                if "reserve_release_due" not in case.visible_flags:
                    case.visible_flags.append("reserve_release_due")
                messages.append(f"Reserve release review due for {case.case_id}")
        elif isinstance(event, MonitoringThresholdBreachedEvent):
            assert case is not None
            workflow = require_refund_workflow(case)
            workflow.monitoring_program_status = MonitoringProgramStatus.BREACHED
            sync_refund_risk_flags(case)
            messages.append(f"Monitoring threshold breached for {case.case_id}")
        elif isinstance(event, SanctionsFalsePositiveClearedEvent):
            assert case is not None
            workflow = require_kyc_workflow(case)
            if workflow.sanctions_status == SanctionsStatus.POTENTIAL_MATCH and case.hidden.true_sanctions_false_positive:
                workflow.sanctions_status = SanctionsStatus.CLEAR
                workflow.screening_match_confidence = 0.11
                workflow.payments_frozen = False
                workflow.payment_freeze_reason = None
                sync_kyc_flags(case)
                messages.append(f"Sanctions false positive cleared for {case.case_id}")
        elif isinstance(event, EDDResponseDueEvent):
            assert case is not None
            workflow = require_kyc_workflow(case)
            if workflow.edd_due_at == event.due_at and workflow.edd_status in {EDDStatus.IN_PROGRESS, EDDStatus.AWAITING_RESPONSE}:
                if case.pending_info_fields or workflow.correction_fields:
                    workflow.edd_status = EDDStatus.AWAITING_RESPONSE
                    if "edd_response_due" not in case.visible_flags:
                        case.visible_flags.append("edd_response_due")
                elif workflow.beneficial_owner_status == BeneficialOwnerStatus.NOT_STARTED:
                    workflow.edd_status = EDDStatus.CLEARED
                sync_kyc_flags(case)
                messages.append(f"EDD response due for {case.case_id}")
        elif isinstance(event, ReportDeadlineMissedEvent):
            assert case is not None
            workflow = require_kyc_workflow(case)
            if workflow.report_due_at == event.at_time and workflow.ofac_report_status != OFACReportStatus.FILED:
                workflow.ofac_report_status = OFACReportStatus.MISSED
                workflow.report_due_at = None
                state.metrics.report_deadlines_missed += 1
                state.metrics.compliance_violations += 1
                sync_kyc_flags(case)
                messages.append(f"OFAC report deadline missed for {case.case_id}")
        elif isinstance(event, VendorRevisedInvoiceEvent):
            assert case is not None
            workflow = require_invoice_workflow(case)
            if workflow.vendor_response_status == VendorResponseStatus.AWAITING:
                workflow.vendor_response_status = VendorResponseStatus.RECEIVED
                if event.revised_amount > 0:
                    workflow.variance_amount = round(event.revised_amount / 100, 2)
                messages.append(f"Vendor revised invoice received for {case.case_id}")
        elif isinstance(event, POChangeApprovedEvent):
            assert case is not None
            workflow = require_invoice_workflow(case)
            if event.approved:
                workflow.po_change_status = POChangeStatus.APPROVED
            else:
                workflow.po_change_status = POChangeStatus.DENIED
            messages.append(f"PO change {'approved' if event.approved else 'denied'} for {case.case_id}")
        elif isinstance(event, StopPaymentConfirmedEvent):
            assert case is not None
            workflow = require_invoice_workflow(case)
            if event.success:
                workflow.payment_batch_status = PaymentBatchStatus.STOPPED
            else:
                workflow.payment_batch_status = PaymentBatchStatus.COMPLETED
            messages.append(f"Stop payment {'succeeded' if event.success else 'failed'} for {case.case_id}")
        elif isinstance(event, VendorRefundReceivedEvent):
            assert case is not None
            workflow = require_invoice_workflow(case)
            refund = event.refund_amount
            recoverable = case.hidden.true_recoverable_amount
            if workflow.recovery_status == RecoveryStatus.NOT_NEEDED:
                workflow.recovery_status = RecoveryStatus.IN_PROGRESS
            if refund >= int(round(recoverable * 100)) and recoverable > 0:
                workflow.recovery_status = RecoveryStatus.COMPLETE
            elif refund > 0:
                workflow.recovery_status = RecoveryStatus.PARTIAL
            messages.append(f"Vendor refund of {refund} received for {case.case_id}")
        elif isinstance(event, PaymentBatchExecutedEvent):
            assert case is not None
            workflow = require_invoice_workflow(case)
            workflow.payment_batch_status = PaymentBatchStatus.COMPLETED
            workflow.stop_payment_window_until = None
            messages.append(f"Payment batch executed for {case.case_id}")
        elif isinstance(event, ArrivalWaveEvent):
            for incoming_case in event.incoming_cases:
                state.cases[incoming_case.case_id] = incoming_case
                if incoming_case.case_id not in state.queue_order:
                    state.queue_order.append(incoming_case.case_id)
            state.records.orders.update(event.record_bundle.orders)
            state.records.customers.update(event.record_bundle.customers)
            state.records.invoices.update(event.record_bundle.invoices)
            state.records.credit_memos.update(event.record_bundle.credit_memos)
            state.records.purchase_orders.update(event.record_bundle.purchase_orders)
            state.records.receipts.update(event.record_bundle.receipts)
            state.records.disputes.update(event.record_bundle.disputes)
            state.records.kyc_verifications.update(event.record_bundle.kyc_verifications)
            state.records.policies.update(event.record_bundle.policies)
            state.records.message_templates.update(event.record_bundle.message_templates)
            state.records.shipping.update(event.record_bundle.shipping)
            state.records.payments.update(event.record_bundle.payments)
            messages.append(f"Arrival wave {event.wave_id} added {len(event.incoming_cases)} cases")
        elif isinstance(event, StaffingDropEvent):
            current_capacity = state.metadata.get("agent_capacity", state.metadata.get("claim_capacity", 2))
            state.metadata["agent_capacity"] = max(1, current_capacity - event.capacity_delta)
            state.metadata["staffing_status"] = "reduced"
            for affected_owner in event.affected_owners:
                for owner_case in state.cases.values():
                    if owner_case.status == "closed" or owner_case.current_owner != affected_owner:
                        continue
                    if owner_case.claimed_by == affected_owner:
                        owner_case.claimed_by = None
                        owner_case.claimed_at = None
                    owner_case.current_owner = "queue"
                    if owner_case.resolution == Resolution.PENDING and owner_case.status not in {"pending_qa", "rework"}:
                        owner_case.status = "open"
                    if "staffing_drop" not in owner_case.visible_flags:
                        owner_case.visible_flags.append("staffing_drop")
            messages.append(f"Staffing drop reduced capacity to {state.metadata['agent_capacity']}")
        elif isinstance(event, ReopenEvent):
            assert case is not None
            case.status = "reopened"
            state.metrics.reopens += 1
            messages.append(f"Case {case.case_id} reopened")
        elif isinstance(event, SLABreachEvent):
            assert case is not None
            if case.status != "closed":
                case.status = "breached"
                state.metrics.cases_breached += 1
                messages.append(f"SLA breached for {case.case_id}")
        state.audit_log.append(
            AuditEntry(
                timestamp=state.current_time,
                case_id=case.case_id if case is not None else None,
                action_type=f"event:{event.event_type}",
                message=messages[-1] if messages else event.event_type,
            )
        )
    return messages
