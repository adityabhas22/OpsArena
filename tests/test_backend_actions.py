from server.environment import OpsArenaEnvironment
from opsarena.domain.workflows.invoice import InvoiceWorkflowState
from opsarena.enums import MatchStatus, ReasonCode, TargetQueue, VerificationDecision
from opsarena.models import (
    AcceptDisputeAction,
    AdvanceClockAction,
    ApproveAction,
    ClaimCaseAction,
    ExecuteRefundAction,
    OpenCaseAction,
    PauseSLAAction,
    PlacePaymentHoldAction,
    RecordThreeWayMatchAction,
    ReleasePaymentHoldAction,
    RequestCreditMemoAction,
    ReturnToQueueAction,
    ResumeSLAAction,
    ReviewKYCAction,
    RouteCaseAction,
    ScheduleFollowUpAction,
    SendForSecondaryApprovalAction,
    SubmitDisputeEvidenceAction,
    TriggerReverificationAction,
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
