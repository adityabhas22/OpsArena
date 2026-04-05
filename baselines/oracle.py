from __future__ import annotations

from opsarena.domain.workflows.invoice import InvoiceWorkflowState, PaymentBatchStatus
from opsarena.domain.workflows.kyc import BeneficialOwnerStatus, EDDStatus, KYCWorkflowState, SanctionsStatus
from opsarena.domain.workflows.refund import DisputeStage, MonitoringProgramStatus, PrearbitrationDecision, RefundWorkflowState
from server.environment import OpsArenaEnvironment
from opsarena.models import (
    AcceptDisputeAction,
    AdvanceClockAction,
    ApplyCreditMemoAction,
    ApproveQAAction,
    ApproveAction,
    ChallengeDisputeAction,
    CloseCaseAction,
    FreezePayoutsAction,
    FileOFACReportAction,
    FreezePaymentsAction,
    RemoveFromPaymentBatchAction,
    SendToQAAction,
    OpenCaseAction,
    QueryPolicyAction,
    RequestFieldCorrectionAction,
    RejectAction,
    RebalanceQueueAction,
    ReleasePaymentHoldAction,
    ResolvePrearbitrationAction,
    RefundPreDisputeAlertAction,
    RequestInfoAction,
    ReviewBeneficialOwnerAction,
    SendForSecondaryApprovalAction,
    RunSanctionsScreenAction,
    SendMessageAction,
    SubmitDisputeEvidenceAction,
    StartEDDReviewAction,
    SetPayoutDelayDaysAction,
    RequestRevisedInvoiceAction,
    SetReservePercentAction,
    StopPaymentAction,
    ViewRecordAction,
    ReviewKYCAction,
    WriteOffSmallBalanceAction,
)
from opsarena.enums import ReasonCode, VerificationDecision

def _next_active_case_id(env: OpsArenaEnvironment) -> str | None:
    assert env._state is not None
    for case_id in env._state.queue_order:
        if env._state.cases[case_id].status != "closed":
            return case_id
    return None


def run_oracle(task_id: str, seed: int = 7) -> dict:
    env = OpsArenaEnvironment()
    env.reset(task_id=task_id, seed=seed)
    assert env._state is not None
    safety_limit = env._state.metadata.get("max_steps", 40) * 4
    iterations = 0

    while not env._is_done() and iterations < safety_limit:
        iterations += 1
        assert env._state is not None
        if task_id == "queue_triage" and env._state.queue_state().unassigned_count > 0:
            env.step(RebalanceQueueAction(assignee_pool=["analyst_1", "analyst_2"], max_cases=3, rebalance_strategy="sla_priority"))

        case_id = _next_active_case_id(env)
        if case_id is None:
            if env._state.scheduled_events:
                next_due = min(event.at_time for event in env._state.scheduled_events)
                env.step(AdvanceClockAction(minutes=max(1, next_due - env._state.current_time)))
                continue
            break

        case = env._state.cases[case_id]
        if case.case_type.value == "refund" and isinstance(case.workflow, RefundWorkflowState):
            pending_case_events = [event.at_time for event in env._state.scheduled_events if event.case_id == case_id]
            if case.workflow.dispute_workflow_status in {"submitted", "challenged", "inquiry_contested", "prearbitration_contested"} and pending_case_events:
                env.step(AdvanceClockAction(minutes=max(1, min(pending_case_events) - env._state.current_time)))
                continue
        if case.case_type.value == "kyc" and isinstance(case.workflow, KYCWorkflowState):
            pending_case_events = [event.at_time for event in env._state.scheduled_events if event.case_id == case_id]
            if (
                case.pending_info_fields
                or case.workflow.sanctions_status == SanctionsStatus.POTENTIAL_MATCH
                or case.workflow.edd_status == EDDStatus.AWAITING_RESPONSE
            ) and pending_case_events:
                env.step(AdvanceClockAction(minutes=max(1, min(pending_case_events) - env._state.current_time)))
                continue

        if getattr(case.qa_status, "value", case.qa_status) == "pending":
            env.step(OpenCaseAction(case_id=case_id))
            env.step(ApproveQAAction(case_id=case_id, assignee_type="qa_reviewer"))
            env.step(CloseCaseAction(case_id=case_id, resolution_code="oracle_complete"))
            continue

        env.step(OpenCaseAction(case_id=case_id))
        if case.case_type.value == "refund":
            workflow = case.workflow
            assert isinstance(workflow, RefundWorkflowState)
            env.step(QueryPolicyAction(policy_id="refund_policy"))
            if workflow.monitoring_program_status == MonitoringProgramStatus.BREACHED and not workflow.payout_frozen:
                env.step(FreezePayoutsAction(case_id=case_id, reason_code="threshold_exceeded"))
            if workflow.merchant_risk_level.value in {"high", "critical"} and workflow.reserve_percent == 0:
                env.step(SetReservePercentAction(case_id=case_id, reserve_percent=15.0, release_after_minutes=25))
            if workflow.merchant_risk_level.value in {"elevated", "high", "critical"} and workflow.payout_delay_days == 0:
                env.step(SetPayoutDelayDaysAction(case_id=case_id, payout_delay_days=7))

            if case.resolution.value != "pending":
                pass
            elif workflow.dispute_stage == DisputeStage.INQUIRY:
                if case.hidden.true_dispute_should_accept or case.amount <= 75:
                    env.step(RefundPreDisputeAlertAction(case_id=case_id))
                else:
                    env.step(ChallengeDisputeAction(case_id=case_id))
            elif workflow.dispute_stage == DisputeStage.PRE_ARBITRATION:
                decision = PrearbitrationDecision.ACCEPT if case.hidden.true_dispute_should_accept else PrearbitrationDecision.CONTEST
                env.step(ResolvePrearbitrationAction(case_id=case_id, prearbitration_decision=decision))
            elif workflow.dispute_stage == DisputeStage.LOST:
                env.step(AcceptDisputeAction(case_id=case_id))
            elif workflow.dispute_stage == DisputeStage.CHARGEBACK_OPEN and not case.hidden.true_dispute_should_accept:
                if not workflow.dispute_evidence_fields:
                    env.step(
                        SubmitDisputeEvidenceAction(
                            case_id=case_id,
                            evidence_fields=["customer_communication", "tracking_number", "delivery_confirmation"],
                        )
                    )
                else:
                    env.step(ChallengeDisputeAction(case_id=case_id))
            elif case.hidden.true_dispute_should_accept and case.amount <= 75:
                env.step(AcceptDisputeAction(case_id=case_id))
            elif case.hidden.true_fraud_risk > 0.7:
                env.step(RejectAction(case_id=case_id, reason_code="suspicious_pattern"))
            else:
                env.step(QueryPolicyAction(policy_id="refund_policy"))
                env.step(ApproveAction(case_id=case_id))
            if env._state.cases[case_id].requires_customer_notification and not env._state.cases[case_id].customer_notified:
                approved = env._state.cases[case_id].resolution.value == "approved"
                template_id = "refund_approved" if approved else "case_closed"
                slots = (
                    {"amount": str(case.amount), "order_id": case.linked_records[0].record_id}
                    if approved
                    else {"case_id": case_id, "resolution": env._state.cases[case_id].resolution.value}
                )
                env.step(
                    SendMessageAction(
                        case_id=case_id,
                        template_id=template_id,
                        slots=slots,
                    )
                )
            if env._state.cases[case_id].qa_required:
                env.step(SendToQAAction(case_id=case_id, assignee_type="qa_reviewer"))
                env.step(ApproveQAAction(case_id=case_id, assignee_type="qa_reviewer"))
            if env._state.cases[case_id].resolution.value != "pending":
                env.step(CloseCaseAction(case_id=case_id, resolution_code="oracle_complete"))
        elif case.case_type.value == "invoice":
            workflow = case.workflow
            assert isinstance(workflow, InvoiceWorkflowState)
            # Remove from payment batch if duplicate or variance
            if workflow.payment_batch_status in {PaymentBatchStatus.SCHEDULED, PaymentBatchStatus.IN_PROGRESS}:
                if case.hidden.true_is_duplicate or (workflow.variance_amount and workflow.variance_amount > 0):
                    env.step(RemoveFromPaymentBatchAction(case_id=case_id))
            # Stop payment if batch already executed and window is open
            refreshed_wf = env._state.cases[case_id].workflow
            assert isinstance(refreshed_wf, InvoiceWorkflowState)
            if (
                refreshed_wf.payment_batch_status == PaymentBatchStatus.COMPLETED
                and refreshed_wf.stop_payment_window_until is not None
                and env._state.current_time <= refreshed_wf.stop_payment_window_until
                and case.hidden.true_is_duplicate
            ):
                env.step(StopPaymentAction(case_id=case_id))
                env.step(AdvanceClockAction(minutes=6))
            if case.resolution.value != "pending":
                pass
            elif case.hidden.true_is_duplicate:
                env.step(RejectAction(case_id=case_id, reason_code="duplicate_match"))
            else:
                env.step(ViewRecordAction(record_type="invoice", record_id=case.linked_records[0].record_id))
                env.step(ViewRecordAction(record_type="purchase_order", record_id=case.linked_records[1].record_id))
                if case.pending_info_fields:
                    env.step(RequestInfoAction(case_id=case_id, field_name="goods_receipt"))
                    env.step(AdvanceClockAction(minutes=case.hidden.hidden_response_latency_minutes or 30))
                    receipt = next(record.record_id for record in env._state.cases[case_id].linked_records if record.record_type.value == "receipt")
                    env.step(ViewRecordAction(record_type="receipt", record_id=receipt))
                env.step(QueryPolicyAction(policy_id="invoice_policy"))
                # Handle variance recovery
                cur_wf = env._state.cases[case_id].workflow
                assert isinstance(cur_wf, InvoiceWorkflowState)
                if cur_wf.variance_amount and cur_wf.variance_amount > 0:
                    if cur_wf.variance_amount <= cur_wf.write_off_threshold:
                        env.step(WriteOffSmallBalanceAction(case_id=case_id, reason_code=ReasonCode.COMPLETE))
                    elif case.hidden.true_vendor_will_respond:
                        env.step(RequestRevisedInvoiceAction(case_id=case_id, reason_code=ReasonCode.THRESHOLD_EXCEEDED))
                        env.step(AdvanceClockAction(minutes=case.hidden.true_vendor_response_minutes or 25))
                        post_wf = env._state.cases[case_id].workflow
                        assert isinstance(post_wf, InvoiceWorkflowState)
                        if post_wf.credit_memo_status.value == "received":
                            memo_id = next(
                                (r.record_id for r in env._state.cases[case_id].linked_records if r.record_type.value == "credit_memo"),
                                None,
                            )
                            if memo_id:
                                env.step(ApplyCreditMemoAction(case_id=case_id, credit_memo_id=memo_id))
                if workflow.secondary_approval_required:
                    env.step(SendForSecondaryApprovalAction(case_id=case_id, reason_code="threshold_exceeded"))
                    env.step(AdvanceClockAction(minutes=15))
                final_wf = env._state.cases[case_id].workflow
                assert isinstance(final_wf, InvoiceWorkflowState)
                if final_wf.credit_memo_status.value == "received":
                    memo_id = next(
                        (r.record_id for r in env._state.cases[case_id].linked_records if r.record_type.value == "credit_memo"),
                        None,
                    )
                    if memo_id:
                        env.step(ApplyCreditMemoAction(case_id=case_id, credit_memo_id=memo_id))
                if final_wf.payment_hold:
                    env.step(ReleasePaymentHoldAction(case_id=case_id))
                env.step(ApproveAction(case_id=case_id))
            if env._state.cases[case_id].qa_required:
                env.step(SendToQAAction(case_id=case_id, assignee_type="qa_reviewer"))
                env.step(ApproveQAAction(case_id=case_id, assignee_type="qa_reviewer"))
            env.step(CloseCaseAction(case_id=case_id, resolution_code="oracle_complete"))
        else:
            workflow = case.workflow
            assert isinstance(workflow, KYCWorkflowState)
            if case.resolution.value != "pending":
                pass
            else:
                env.step(QueryPolicyAction(policy_id="kyc_policy"))
                if workflow.sanctions_status == SanctionsStatus.NOT_STARTED:
                    env.step(RunSanctionsScreenAction(case_id=case_id))
                    workflow = env._state.cases[case_id].workflow
                if workflow.sanctions_status == SanctionsStatus.CONFIRMED_MATCH:
                    if not workflow.payments_frozen:
                        env.step(FreezePaymentsAction(case_id=case_id, reason_code=ReasonCode.SANCTIONS_MATCH))
                    if workflow.ofac_report_status.value in {"pending", "missed"}:
                        env.step(FileOFACReportAction(case_id=case_id))
                    env.step(ReviewKYCAction(case_id=case_id, verification_decision=VerificationDecision.REJECT))
                    env.step(RejectAction(case_id=case_id, reason_code=ReasonCode.SANCTIONS_MATCH))
                else:
                    if workflow.edd_status == EDDStatus.NOT_STARTED and (case.hidden.true_edd_required or case.hidden.true_beneficial_owner_issue):
                        env.step(StartEDDReviewAction(case_id=case_id))
                        workflow = env._state.cases[case_id].workflow
                    if workflow.beneficial_owner_status == BeneficialOwnerStatus.NEEDS_CORRECTION and workflow.correction_fields:
                        for field_name in list(workflow.correction_fields):
                            if field_name not in env._state.cases[case_id].pending_info_fields:
                                env.step(RequestFieldCorrectionAction(case_id=case_id, field_name=field_name))
                        workflow = env._state.cases[case_id].workflow
                    if workflow.beneficial_owner_status == BeneficialOwnerStatus.PENDING_REVIEW and not env._state.cases[case_id].pending_info_fields:
                        env.step(
                            ReviewBeneficialOwnerAction(
                                case_id=case_id,
                                verification_decision=VerificationDecision.APPROVE,
                            )
                        )
                        workflow = env._state.cases[case_id].workflow
                current_case = env._state.cases[case_id]
                current_workflow = current_case.workflow
                assert isinstance(current_workflow, KYCWorkflowState)
                if current_case.resolution.value == "pending" and not current_workflow.kyc_complete:
                    env.step(RequestInfoAction(case_id=case_id, field_name="individual.verification.document"))
                    env.step(AdvanceClockAction(minutes=current_case.hidden.hidden_response_latency_minutes or 45))
                current_case = env._state.cases[case_id]
                current_workflow = current_case.workflow
                assert isinstance(current_workflow, KYCWorkflowState)
                if current_case.resolution.value == "pending":
                    env.step(ViewRecordAction(record_type="kyc_document", record_id=current_case.linked_records[-1].record_id))
                if current_case.resolution.value == "pending" and not current_case.hidden.true_sanctions_match:
                    if not current_case.hidden.true_doc_valid:
                        env.step(ReviewKYCAction(case_id=case_id, verification_decision=VerificationDecision.REJECT))
                        env.step(RejectAction(case_id=case_id, reason_code=ReasonCode.INVALID_DOCUMENT))
                    else:
                        env.step(ReviewKYCAction(case_id=case_id, verification_decision=VerificationDecision.APPROVE))
                        env.step(ApproveAction(case_id=case_id))
            if env._state.cases[case_id].qa_required:
                env.step(SendToQAAction(case_id=case_id, assignee_type="qa_reviewer"))
                env.step(ApproveQAAction(case_id=case_id, assignee_type="qa_reviewer"))
            if env._state.cases[case_id].resolution.value != "pending":
                env.step(CloseCaseAction(case_id=case_id, resolution_code="oracle_complete"))
    return env.state.model_dump()
