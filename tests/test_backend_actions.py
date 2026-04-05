from server.environment import OpsArenaEnvironment
from opsarena.domain.workflows.invoice import (
    CreditMemoStatus,
    InvoiceWorkflowState,
    POChangeStatus,
    PaymentBatchStatus,
    RecoveryStatus,
    VendorResponseStatus,
)
from opsarena.domain.workflows.kyc import BeneficialOwnerStatus, EDDStatus, OFACReportStatus, SanctionsStatus
from opsarena.domain.workflows.refund import DisputeStage, MonitoringProgramStatus, PrearbitrationDecision
from opsarena.enums import MatchStatus, ReasonCode, RecordType, TargetQueue, VerificationDecision
from opsarena.models import (
    AcceptDisputeAction,
    AdvanceClockAction,
    ApplyCreditMemoAction,
    ApproveQAAction,
    ApproveAction,
    ChallengeDisputeAction,
    ClaimCaseAction,
    CloseCaseAction,
    ClearReserveAction,
    BulkAssignAction,
    BulkRouteAction,
    ExecuteRefundAction,
    FailQAAction,
    FileOFACReportAction,
    FreezePaymentsAction,
    FreezePayoutsAction,
    OpenCaseAction,
    PauseSLAAction,
    PlacePaymentHoldAction,
    RebalanceQueueAction,
    RecordThreeWayMatchAction,
    RecordVendorRefundAction,
    ReleasePaymentHoldAction,
    RemoveFromPaymentBatchAction,
    ResolvePrearbitrationAction,
    RefundPreDisputeAlertAction,
    RequestCreditMemoAction,
    RequestPOChangeAction,
    RequestRevisedInvoiceAction,
    QueryPolicyAction,
    RejectAction,
    RequestFieldCorrectionAction,
    RequestInfoAction,
    ReturnToQueueAction,
    ResumeSLAAction,
    ReviewBeneficialOwnerAction,
    ReviewKYCAction,
    RunSanctionsScreenAction,
    RouteCaseAction,
    ScheduleFollowUpAction,
    SendToQAAction,
    SendForSecondaryApprovalAction,
    SendMessageAction,
    StartEDDReviewAction,
    StopPaymentAction,
    SetPayoutDelayDaysAction,
    SetReservePercentAction,
    SubmitDisputeEvidenceAction,
    TriggerReverificationAction,
    UnfreezePayoutsAction,
    ViewRecordAction,
    WriteOffSmallBalanceAction,
)


def test_route_and_pause_resume_sla_updates_case_workflow():
    env = OpsArenaEnvironment()
    env.reset(task_id="refund_exception", seed=2)
    env.step(OpenCaseAction(case_id="case_refund_1"))
    original_deadline = env._state.cases["case_refund_1"].sla_deadline

    routed = env.step(
        RouteCaseAction(
            case_id="case_refund_1",
            target_queue=TargetQueue.FRAUD_TEAM,
            reason_code=ReasonCode.SUSPICIOUS_PATTERN,
            assignee_type="fraud_analyst",
        )
    )
    assert routed.case_detail is not None
    assert routed.case_detail.current_owner == "fraud_analyst"
    assert routed.case_detail.workflow_metadata["active_queue"] == "fraud_team"

    paused = env.step(PauseSLAAction(case_id="case_refund_1", reason_code=ReasonCode.AWAITING_RESPONSE))
    assert paused.case_detail is not None
    assert paused.case_detail.workflow_metadata["sla_pause_reason"] == "awaiting_response"
    paused_at = env._state.cases["case_refund_1"].sla_paused_at

    env.step(AdvanceClockAction(minutes=15))
    expected_extension = env._state.current_time - paused_at
    resumed = env.step(ResumeSLAAction(case_id="case_refund_1"))
    assert resumed.case_detail is not None
    assert env._state.cases["case_refund_1"].sla_deadline == original_deadline + expected_extension


def test_refund_backend_actions_update_dispute_and_payment_records():
    env = OpsArenaEnvironment()
    env.reset(task_id="refund_exception", seed=2)
    env.step(OpenCaseAction(case_id="case_refund_1"))
    env.step(ApproveAction(case_id="case_refund_1"))

    submitted = env.step(
        SubmitDisputeEvidenceAction(
            case_id="case_refund_1",
            evidence_fields=["customer_communication", "tracking_number"],
        )
    )
    assert submitted.case_detail is not None
    assert submitted.case_detail.workflow_metadata["dispute_workflow_status"] == "submitted"
    assert submitted.case_detail.workflow_metadata["dispute_evidence_fields"] == [
        "customer_communication",
        "tracking_number",
    ]

    refunded = env.step(ExecuteRefundAction(case_id="case_refund_1"))
    assert refunded.case_detail is not None
    assert refunded.case_detail.workflow_metadata["refund_execution_state"] == "refunded"
    assert env._state.records.payments["pay_1001"].status == "refunded"


def test_invoice_backend_actions_track_match_and_payment_hold():
    env = OpsArenaEnvironment()
    env.reset(task_id="invoice_plus_kyc", seed=3)
    env.step(OpenCaseAction(case_id="case_invoice_1"))

    matched = env.step(
        RecordThreeWayMatchAction(
            case_id="case_invoice_1",
            match_status=MatchStatus.VARIANCE,
            variance_amount=24.5,
        )
    )
    assert matched.case_detail is not None
    assert matched.case_detail.workflow_metadata["match_status"] == "variance"
    assert matched.case_detail.workflow_metadata["variance_amount"] == 24.5

    held = env.step(
        PlacePaymentHoldAction(
            case_id="case_invoice_1",
            reason_code=ReasonCode.MISSING_DOCUMENTATION,
        )
    )
    assert held.case_detail is not None
    assert held.case_detail.workflow_metadata["payment_hold"] is True

    released = env.step(ReleasePaymentHoldAction(case_id="case_invoice_1"))
    assert released.case_detail is not None
    assert released.case_detail.workflow_metadata["payment_hold"] is False


def test_kyc_backend_actions_track_reverification_state():
    env = OpsArenaEnvironment()
    env.reset(task_id="invoice_plus_kyc", seed=3)
    env.step(OpenCaseAction(case_id="case_kyc_1"))

    triggered = env.step(
        TriggerReverificationAction(
            case_id="case_kyc_1",
            requirements=[
                "individual.verification.document",
                "individual.verification.additional_document",
            ],
        )
    )
    assert triggered.case_detail is not None
    assert triggered.case_detail.workflow_metadata["requirements_due"] == [
        "individual.verification.document",
        "individual.verification.additional_document",
    ]

    reviewed = env.step(
        ReviewKYCAction(
            case_id="case_kyc_1",
            verification_decision=VerificationDecision.REQUEST_RESUBMISSION,
        )
    )
    assert reviewed.case_detail is not None
    assert reviewed.case_detail.workflow_metadata["verification_status"] == "requires_input"
    assert "individual.verification.document" in reviewed.case_detail.workflow_metadata["pending_info_fields"]


def test_claim_follow_up_and_return_to_queue_update_queue_ops_state():
    env = OpsArenaEnvironment()
    env.reset(task_id="queue_triage", seed=7)

    claimed = env.step(ClaimCaseAction(case_id="case_refund_2", assignee_type="analyst_1"))
    assert claimed.case_detail is not None
    assert claimed.case_detail.workflow_metadata["claimed_by"] == "analyst_1"

    env.step(OpenCaseAction(case_id="case_refund_2"))
    follow_up = env.step(
        ScheduleFollowUpAction(
            case_id="case_refund_2",
            follow_up_at=env._state.current_time + 10,
            reason_code=ReasonCode.AWAITING_RESPONSE,
        )
    )
    assert follow_up.case_detail is not None
    assert follow_up.case_detail.workflow_metadata["next_touch_at"] is not None

    env.step(AdvanceClockAction(minutes=10))
    overdue_case = env._state.cases["case_refund_2"]
    assert overdue_case.follow_up_overdue is True
    assert env._state.metrics.follow_ups_overdue == 1

    returned = env.step(ReturnToQueueAction(case_id="case_refund_2", reason_code=ReasonCode.AWAITING_RESPONSE))
    assert returned.case_detail is not None
    assert env._state.cases["case_refund_2"].claimed_by is None
    assert env._state.cases["case_refund_2"].current_owner == "queue"


def test_accept_dispute_finishes_low_dollar_refund_path():
    env = OpsArenaEnvironment()
    env.reset(task_id="queue_triage", seed=7)
    env.step(OpenCaseAction(case_id="case_refund_2"))

    accepted = env.step(AcceptDisputeAction(case_id="case_refund_2"))
    assert accepted.case_detail is not None
    assert accepted.case_detail.workflow_metadata["dispute_resolution"] == "accepted"
    assert env._state.cases["case_refund_2"].resolution.value == "approved"


def test_credit_memo_and_secondary_approval_workflow_adds_records():
    env = OpsArenaEnvironment()
    env.reset(task_id="queue_triage", seed=7)
    env.step(OpenCaseAction(case_id="case_invoice_2"))
    env.step(
        RecordThreeWayMatchAction(
            case_id="case_invoice_2",
            match_status=MatchStatus.VARIANCE,
            variance_amount=12.5,
        )
    )

    requested = env.step(RequestCreditMemoAction(case_id="case_invoice_2", approved_amount=12.5))
    assert requested.case_detail is not None
    assert requested.case_detail.workflow_metadata["credit_memo_status"] == "requested"

    env.step(AdvanceClockAction(minutes=env._state.cases["case_invoice_2"].hidden.hidden_response_latency_minutes or 30))
    credit_case = env._state.cases["case_invoice_2"]
    credit_memo_id = next(record.record_id for record in credit_case.linked_records if record.record_type.value == "credit_memo")
    assert env._state.records.credit_memos[credit_memo_id].amount == 1250

    sent = env.step(
        SendForSecondaryApprovalAction(
            case_id="case_invoice_2",
            reason_code=ReasonCode.THRESHOLD_EXCEEDED,
        )
    )
    assert sent.case_detail is not None
    assert sent.case_detail.workflow_metadata["approval_status"] == "pending_secondary"

    env.step(AdvanceClockAction(minutes=15))
    workflow = env._state.cases["case_invoice_2"].workflow
    assert isinstance(workflow, InvoiceWorkflowState)
    assert workflow.approval_status.value == "approved"

    env.step(ReleasePaymentHoldAction(case_id="case_invoice_2"))
    final = env.step(ApproveAction(case_id="case_invoice_2"))
    assert final.error is None


def test_qa_failure_reopens_case_for_rework_and_blocks_close():
    env = OpsArenaEnvironment()
    env.reset(task_id="queue_triage", seed=7)
    env.step(OpenCaseAction(case_id="case_invoice_2"))
    env.step(
        RecordThreeWayMatchAction(
            case_id="case_invoice_2",
            match_status=MatchStatus.MATCHED,
        )
    )
    env.step(SendForSecondaryApprovalAction(case_id="case_invoice_2", reason_code=ReasonCode.THRESHOLD_EXCEEDED))
    env.step(AdvanceClockAction(minutes=15))
    env.step(ApproveAction(case_id="case_invoice_2"))

    sent = env.step(SendToQAAction(case_id="case_invoice_2", assignee_type="qa_reviewer"))
    assert sent.case_detail is not None
    assert sent.case_detail.workflow_metadata["qa_status"] == "pending"

    failed = env.step(
        FailQAAction(
            case_id="case_invoice_2",
            assignee_type="qa_reviewer",
            reason_code=ReasonCode.MISSING_DOCUMENTATION,
        )
    )
    assert failed.case_detail is not None
    assert failed.case_detail.status == "rework"
    assert env._state.cases["case_invoice_2"].resolution.value == "pending"

    env.step(AdvanceClockAction(minutes=env._state.cases["case_invoice_2"].hidden.hidden_follow_up_latency_minutes or 30))
    assert env._state.cases["case_invoice_2"].qa_rework_overdue is True

    env.step(ApproveAction(case_id="case_invoice_2"))
    env.step(SendToQAAction(case_id="case_invoice_2", assignee_type="qa_reviewer"))
    approved = env.step(ApproveQAAction(case_id="case_invoice_2", assignee_type="qa_reviewer"))
    assert approved.case_detail is not None
    assert approved.case_detail.workflow_metadata["qa_status"] == "passed"


def test_supervisor_bulk_assign_route_and_rebalance_update_queue_state():
    env = OpsArenaEnvironment()
    env.reset(task_id="queue_triage", seed=7)

    rebalanced = env.step(
        RebalanceQueueAction(
            assignee_pool=["analyst_1", "analyst_2"],
            max_cases=3,
            rebalance_strategy="sla_priority",
        )
    )
    assert rebalanced.error is None
    assert env._state.cases["case_refund_1"].current_owner in {"analyst_1", "analyst_2"}
    assert env._state.queue_state().unassigned_count == 2

    bulk_assigned = env.step(
        BulkAssignAction(
            case_ids=["case_invoice_1", "case_kyc_triage"],
            assignee_type="analyst_3",
        )
    )
    assert bulk_assigned.error is None
    assert env._state.cases["case_invoice_1"].current_owner == "analyst_3"
    assert env._state.cases["case_kyc_triage"].current_owner == "analyst_3"

    bulk_routed = env.step(
        BulkRouteAction(
            case_ids=["case_refund_2", "case_invoice_1"],
            target_queue=TargetQueue.MANAGER_REVIEW,
            reason_code=ReasonCode.THRESHOLD_EXCEEDED,
        )
    )
    assert bulk_routed.error is None
    assert env._state.cases["case_refund_2"].active_queue == "manager_review"
    assert env._state.cases["case_invoice_1"].route_history[-1].queue == "manager_review"


def test_arrival_wave_and_staffing_drop_change_queue_capacity_and_backlog():
    env = OpsArenaEnvironment()
    env.reset(task_id="queue_triage", seed=7)
    env.step(
        RebalanceQueueAction(
            assignee_pool=["analyst_1", "analyst_2"],
            max_cases=3,
            rebalance_strategy="sla_priority",
        )
    )
    assigned_to_analyst_2 = [
        case.case_id for case in env._state.cases.values() if case.current_owner == "analyst_2"
    ]
    assert assigned_to_analyst_2

    arrival_due = min(event.at_time for event in env._state.scheduled_events if event.event_type == "arrival_wave")
    env.step(AdvanceClockAction(minutes=arrival_due - env._state.current_time))
    assert "case_refund_wave_1" in env._state.cases
    assert env._state.queue_state().exception_queue_size == 6

    staffing_due = min(event.at_time for event in env._state.scheduled_events if event.event_type == "staffing_drop")
    env.step(AdvanceClockAction(minutes=staffing_due - env._state.current_time))
    assert env._state.metadata["agent_capacity"] == 1
    for case_id in assigned_to_analyst_2:
        assert env._state.cases[case_id].current_owner == "queue"
        assert "staffing_drop" in env._state.cases[case_id].visible_flags


def test_qa_sample_selected_reopens_closed_case_for_review():
    env = OpsArenaEnvironment()
    env.reset(task_id="queue_triage", seed=7)
    env.step(OpenCaseAction(case_id="case_refund_1"))
    env.step(FreezePayoutsAction(case_id="case_refund_1", reason_code=ReasonCode.THRESHOLD_EXCEEDED))
    env.step(SetReservePercentAction(case_id="case_refund_1", reserve_percent=15.0, release_after_minutes=20))
    env.step(QueryPolicyAction(policy_id="refund_policy"))
    env.step(ApproveAction(case_id="case_refund_1"))
    env.step(
        SendMessageAction(
            case_id="case_refund_1",
            template_id="refund_approved",
            slots={"amount": "350.00", "order_id": "ord_1001"},
        )
    )
    env.step(CloseCaseAction(case_id="case_refund_1", resolution_code="done"))

    qa_due = min(event.at_time for event in env._state.scheduled_events if event.event_type == "qa_sample_selected")
    env.step(AdvanceClockAction(minutes=qa_due - env._state.current_time))
    assert env._state.cases["case_refund_1"].qa_status.value == "pending"
    assert env._state.cases["case_refund_1"].status == "pending_qa"


def test_refund_phase3_payout_controls_and_pre_dispute_resolution():
    env = OpsArenaEnvironment()
    env.reset(task_id="queue_triage", seed=7)

    env.step(OpenCaseAction(case_id="case_refund_1"))
    env.step(FreezePayoutsAction(case_id="case_refund_1", reason_code=ReasonCode.THRESHOLD_EXCEEDED))
    env.step(SetReservePercentAction(case_id="case_refund_1", reserve_percent=15.0, release_after_minutes=20))
    env.step(SetPayoutDelayDaysAction(case_id="case_refund_1", payout_delay_days=7))
    refund_one = env._state.cases["case_refund_1"].workflow
    assert refund_one.payout_frozen is True
    assert refund_one.reserve_percent == 15.0
    assert refund_one.payout_delay_days == 7

    reserve_due = min(event.at_time for event in env._state.scheduled_events if event.event_type == "reserve_release_due")
    env.step(AdvanceClockAction(minutes=reserve_due - env._state.current_time))
    assert "reserve_release_due" in env._state.cases["case_refund_1"].visible_flags

    env._state.cases["case_refund_1"].workflow.monitoring_program_status = MonitoringProgramStatus.WARNING
    env._state.cases["case_refund_1"].workflow.merchant_dispute_ratio_30d = 0.003
    env._state.cases["case_refund_1"].workflow.merchant_fraud_ratio_30d = 0.0
    env._state.cases["case_refund_1"].workflow.merchant_negative_balance = False
    env.step(ClearReserveAction(case_id="case_refund_1"))
    env.step(UnfreezePayoutsAction(case_id="case_refund_1"))
    assert env._state.cases["case_refund_1"].workflow.reserve_percent == 0.0
    assert env._state.cases["case_refund_1"].workflow.payout_frozen is False

    pre_dispute_env = OpsArenaEnvironment()
    pre_dispute_env.reset(task_id="queue_triage", seed=7)
    pre_dispute_env.step(OpenCaseAction(case_id="case_refund_2"))
    resolved = pre_dispute_env.step(RefundPreDisputeAlertAction(case_id="case_refund_2"))
    assert resolved.case_detail is not None
    assert resolved.case_detail.workflow_metadata["dispute_stage"] == "finalized"
    assert resolved.case_detail.workflow_metadata["refund_execution_state"] == "refunded"


def test_refund_phase3_inquiry_and_prearbitration_flows_progress_via_events():
    env = OpsArenaEnvironment()
    env.reset(task_id="queue_triage", seed=7)

    inquiry_due = min(event.at_time for event in env._state.scheduled_events if event.event_type == "inquiry_escalates_to_chargeback")
    env.step(AdvanceClockAction(minutes=inquiry_due - env._state.current_time))
    refund_two = env._state.cases["case_refund_2"].workflow
    assert refund_two.dispute_stage == DisputeStage.CHARGEBACK_OPEN

    env.reset(task_id="refund_exception", seed=2)
    env._state.cases["case_refund_1"].hidden.true_dispute_should_accept = False
    env.step(OpenCaseAction(case_id="case_refund_1"))
    env.step(
        SubmitDisputeEvidenceAction(
            case_id="case_refund_1",
            evidence_fields=["customer_communication"],
        )
    )
    prearb_due = min(event.at_time for event in env._state.scheduled_events if event.event_type == "prearbitration_received")
    env.step(AdvanceClockAction(minutes=prearb_due - env._state.current_time))
    assert env._state.cases["case_refund_1"].workflow.dispute_stage == DisputeStage.PRE_ARBITRATION

    accepted = env.step(
        ResolvePrearbitrationAction(
            case_id="case_refund_1",
            prearbitration_decision=PrearbitrationDecision.ACCEPT,
        )
    )
    assert accepted.case_detail is not None
    assert accepted.case_detail.workflow_metadata["dispute_resolution"] == "accepted"


def test_challenge_dispute_resolves_inquiry_when_evidence_is_strong():
    env = OpsArenaEnvironment()
    env.reset(task_id="queue_triage", seed=7)
    env._state.cases["case_refund_2"].hidden.true_dispute_should_accept = False
    env._state.cases["case_refund_2"].gather_evidence("order_history")
    env._state.cases["case_refund_2"].gather_evidence("customer_profile")
    env._state.cases["case_refund_2"].gather_evidence("refund_policy")

    env.step(OpenCaseAction(case_id="case_refund_2"))
    challenged = env.step(ChallengeDisputeAction(case_id="case_refund_2"))
    assert challenged.case_detail is not None
    assert challenged.case_detail.workflow_metadata["dispute_resolution"] == "won"


def test_phase4_kyc_false_positive_and_beneficial_owner_flow_clears_for_approval():
    env = OpsArenaEnvironment()
    env.reset(task_id="invoice_plus_kyc", seed=3)
    env.step(OpenCaseAction(case_id="case_kyc_1"))

    screened = env.step(RunSanctionsScreenAction(case_id="case_kyc_1"))
    assert screened.case_detail is not None
    assert screened.case_detail.workflow_metadata["sanctions_status"] == "potential_match"

    env.step(StartEDDReviewAction(case_id="case_kyc_1"))
    correction = env.step(RequestFieldCorrectionAction(case_id="case_kyc_1", field_name="owners.1.address"))
    assert correction.case_detail is not None
    assert correction.case_detail.workflow_metadata["beneficial_owner_status"] == "needs_correction"

    env.step(RequestInfoAction(case_id="case_kyc_1", field_name="individual.verification.document"))
    while env._state.cases["case_kyc_1"].pending_info_fields:
        next_due = min(
            event.at_time
            for event in env._state.scheduled_events
            if event.case_id == "case_kyc_1"
        )
        env.step(AdvanceClockAction(minutes=next_due - env._state.current_time))

    workflow = env._state.cases["case_kyc_1"].workflow
    assert workflow.sanctions_status == SanctionsStatus.CLEAR
    assert workflow.beneficial_owner_status == BeneficialOwnerStatus.PENDING_REVIEW

    owners = env.step(
        ReviewBeneficialOwnerAction(
            case_id="case_kyc_1",
            verification_decision=VerificationDecision.APPROVE,
        )
    )
    assert owners.case_detail is not None
    assert owners.case_detail.workflow_metadata["edd_status"] == "cleared"

    env.step(ViewRecordAction(record_type=RecordType.KYC_DOCUMENT, record_id="kyc_3001"))
    reviewed = env.step(
        ReviewKYCAction(
            case_id="case_kyc_1",
            verification_decision=VerificationDecision.APPROVE,
        )
    )
    assert reviewed.case_detail is not None
    assert reviewed.case_detail.workflow_metadata["kyc_stage"] == "cleared"

    approved = env.step(ApproveAction(case_id="case_kyc_1"))
    assert approved.error is None
    assert env._state.cases["case_kyc_1"].resolution.value == "approved"


def test_phase4_kyc_confirmed_match_requires_report_and_freeze_before_close():
    env = OpsArenaEnvironment()
    env.reset(task_id="queue_triage", seed=7)
    env.step(OpenCaseAction(case_id="case_kyc_triage"))

    screened = env.step(RunSanctionsScreenAction(case_id="case_kyc_triage"))
    assert screened.case_detail is not None
    assert screened.case_detail.workflow_metadata["sanctions_status"] == "confirmed_match"
    assert screened.case_detail.workflow_metadata["ofac_report_status"] == "pending"

    env.step(FreezePaymentsAction(case_id="case_kyc_triage", reason_code=ReasonCode.SANCTIONS_MATCH))
    env.step(ReviewKYCAction(case_id="case_kyc_triage", verification_decision=VerificationDecision.REJECT))
    env.step(RejectAction(case_id="case_kyc_triage", reason_code=ReasonCode.SANCTIONS_MATCH))

    blocked = env.step(CloseCaseAction(case_id="case_kyc_triage", resolution_code="blocked"))
    assert blocked.error is not None

    filed = env.step(FileOFACReportAction(case_id="case_kyc_triage"))
    assert filed.case_detail is not None
    assert filed.case_detail.workflow_metadata["ofac_report_status"] == "filed"

    closed = env.step(CloseCaseAction(case_id="case_kyc_triage", resolution_code="done"))
    assert closed.error is None
    assert env._state.cases["case_kyc_triage"].status == "closed"


def test_phase4_kyc_report_deadline_miss_records_compliance_violation():
    env = OpsArenaEnvironment()
    env.reset(task_id="queue_triage", seed=7)
    env.step(OpenCaseAction(case_id="case_kyc_triage"))
    env.step(RunSanctionsScreenAction(case_id="case_kyc_triage"))

    due_at = env._state.cases["case_kyc_triage"].workflow.report_due_at
    assert due_at is not None
    env.step(AdvanceClockAction(minutes=due_at - env._state.current_time))

    workflow = env._state.cases["case_kyc_triage"].workflow
    assert workflow.ofac_report_status == OFACReportStatus.MISSED
    assert env._state.metrics.report_deadlines_missed == 1
    assert env._state.metrics.compliance_violations == 1


def test_phase5_request_revised_invoice_schedules_vendor_response():
    env = OpsArenaEnvironment()
    env.reset(task_id="queue_triage", seed=7)
    env.step(OpenCaseAction(case_id="case_invoice_1"))

    result = env.step(
        RequestRevisedInvoiceAction(
            case_id="case_invoice_1",
            reason_code=ReasonCode.MISSING_DOCUMENTATION,
        )
    )
    assert result.error is None
    workflow = env._state.cases["case_invoice_1"].workflow
    assert isinstance(workflow, InvoiceWorkflowState)
    assert workflow.vendor_response_status == VendorResponseStatus.AWAITING

    latency = env._state.cases["case_invoice_1"].hidden.true_vendor_response_minutes
    env.step(AdvanceClockAction(minutes=latency))
    workflow = env._state.cases["case_invoice_1"].workflow
    assert workflow.vendor_response_status == VendorResponseStatus.RECEIVED


def test_phase5_request_po_change_schedules_approval():
    env = OpsArenaEnvironment()
    env.reset(task_id="queue_triage", seed=7)
    env.step(OpenCaseAction(case_id="case_invoice_1"))

    result = env.step(
        RequestPOChangeAction(
            case_id="case_invoice_1",
            change_description="Adjust quantity to match delivery",
        )
    )
    assert result.error is None
    workflow = env._state.cases["case_invoice_1"].workflow
    assert isinstance(workflow, InvoiceWorkflowState)
    assert workflow.po_change_status == POChangeStatus.PENDING_APPROVAL

    env.step(AdvanceClockAction(minutes=15))
    workflow = env._state.cases["case_invoice_1"].workflow
    assert workflow.po_change_status == POChangeStatus.APPROVED


def test_phase5_remove_from_payment_batch_clears_batch():
    env = OpsArenaEnvironment()
    env.reset(task_id="queue_triage", seed=7)
    env.step(OpenCaseAction(case_id="case_invoice_1"))

    workflow = env._state.cases["case_invoice_1"].workflow
    assert isinstance(workflow, InvoiceWorkflowState)
    assert workflow.payment_batch_status == PaymentBatchStatus.SCHEDULED

    result = env.step(RemoveFromPaymentBatchAction(case_id="case_invoice_1"))
    assert result.error is None
    workflow = env._state.cases["case_invoice_1"].workflow
    assert workflow.payment_batch_status == PaymentBatchStatus.NOT_SCHEDULED
    assert workflow.payment_batch_id is None


def test_phase5_stop_payment_requires_in_progress_batch():
    env = OpsArenaEnvironment()
    env.reset(task_id="queue_triage", seed=7)
    env.step(OpenCaseAction(case_id="case_invoice_1"))

    result = env.step(StopPaymentAction(case_id="case_invoice_1"))
    assert result.error is not None

    env._state.cases["case_invoice_1"].workflow.payment_batch_status = PaymentBatchStatus.IN_PROGRESS
    result = env.step(StopPaymentAction(case_id="case_invoice_1"))
    assert result.error is None

    env.step(AdvanceClockAction(minutes=5))
    workflow = env._state.cases["case_invoice_1"].workflow
    assert workflow.payment_batch_status in {PaymentBatchStatus.STOPPED, PaymentBatchStatus.COMPLETED}


def test_phase5_record_vendor_refund_updates_recovery():
    env = OpsArenaEnvironment()
    env.reset(task_id="queue_triage", seed=7)
    env.step(OpenCaseAction(case_id="case_invoice_1"))

    result = env.step(
        RecordVendorRefundAction(case_id="case_invoice_1", refund_amount=24.5)
    )
    assert result.error is None
    workflow = env._state.cases["case_invoice_1"].workflow
    assert isinstance(workflow, InvoiceWorkflowState)
    assert workflow.recovery_status == RecoveryStatus.COMPLETE


def test_phase5_apply_credit_memo_requires_received_status():
    env = OpsArenaEnvironment()
    env.reset(task_id="queue_triage", seed=7)
    env.step(OpenCaseAction(case_id="case_invoice_1"))

    result = env.step(
        ApplyCreditMemoAction(case_id="case_invoice_1", credit_memo_id="cm_fake")
    )
    assert result.error is not None

    env.step(RequestCreditMemoAction(case_id="case_invoice_1", approved_amount=24.5))
    env.step(AdvanceClockAction(minutes=env._state.cases["case_invoice_1"].hidden.hidden_response_latency_minutes or 30))

    workflow = env._state.cases["case_invoice_1"].workflow
    assert isinstance(workflow, InvoiceWorkflowState)
    assert workflow.credit_memo_status == CreditMemoStatus.RECEIVED

    memo_id = next(
        r.record_id
        for r in env._state.cases["case_invoice_1"].linked_records
        if r.record_type.value == "credit_memo"
    )
    result = env.step(ApplyCreditMemoAction(case_id="case_invoice_1", credit_memo_id=memo_id))
    assert result.error is None
    workflow = env._state.cases["case_invoice_1"].workflow
    assert workflow.credit_memo_status == CreditMemoStatus.APPLIED
    assert workflow.recovery_status == RecoveryStatus.COMPLETE


def test_phase5_write_off_small_balance_resolves_case():
    env = OpsArenaEnvironment()
    env.reset(task_id="queue_triage", seed=7)
    env.step(OpenCaseAction(case_id="case_invoice_1"))

    workflow = env._state.cases["case_invoice_1"].workflow
    assert isinstance(workflow, InvoiceWorkflowState)
    workflow.variance_amount = 25.0

    result = env.step(
        WriteOffSmallBalanceAction(
            case_id="case_invoice_1",
            reason_code=ReasonCode.COMPLETE,
        )
    )
    assert result.error is None
    assert env._state.cases["case_invoice_1"].resolution.value == "approved"
    assert env._state.cases["case_invoice_1"].status == "resolved"
    workflow = env._state.cases["case_invoice_1"].workflow
    assert workflow.recovery_status == RecoveryStatus.COMPLETE


def test_phase5_write_off_rejects_large_balance():
    env = OpsArenaEnvironment()
    env.reset(task_id="queue_triage", seed=7)
    env.step(OpenCaseAction(case_id="case_invoice_1"))

    workflow = env._state.cases["case_invoice_1"].workflow
    assert isinstance(workflow, InvoiceWorkflowState)
    workflow.variance_amount = 100.0

    result = env.step(
        WriteOffSmallBalanceAction(
            case_id="case_invoice_1",
            reason_code=ReasonCode.COMPLETE,
        )
    )
    assert result.error is not None


def test_phase5_payment_batch_executed_event_closes_stop_window():
    env = OpsArenaEnvironment()
    env.reset(task_id="queue_triage", seed=7)
    env.step(OpenCaseAction(case_id="case_invoice_1"))

    from opsarena.domain.events import PaymentBatchExecutedEvent
    from opsarena.engine.scheduler import schedule_event

    workflow = env._state.cases["case_invoice_1"].workflow
    assert isinstance(workflow, InvoiceWorkflowState)
    workflow.payment_batch_status = PaymentBatchStatus.IN_PROGRESS
    workflow.stop_payment_window_until = env._state.current_time + 20

    schedule_event(
        env._state,
        PaymentBatchExecutedEvent(
            at_time=env._state.current_time + 10,
            case_id="case_invoice_1",
            batch_id="batch_001",
        ),
    )
    env.step(AdvanceClockAction(minutes=10))

    workflow = env._state.cases["case_invoice_1"].workflow
    assert workflow.payment_batch_status == PaymentBatchStatus.COMPLETED
    assert workflow.stop_payment_window_until is None
