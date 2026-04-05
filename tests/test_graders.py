from server.environment import OpsArenaEnvironment
from opsarena.engine.graders import grade_episode, grade_trajectory
from opsarena.models import (
    AdvanceClockAction,
    ApproveAction,
    CloseCaseAction,
    FailQAAction,
    OpenCaseAction,
    QueryPolicyAction,
    ScheduleFollowUpAction,
    SendToQAAction,
    SendForSecondaryApprovalAction,
    SendMessageAction,
)


def test_grader_rewards_complete_process():
    env = OpsArenaEnvironment()
    env.reset(task_id="refund_exception", seed=2)
    env.step(OpenCaseAction(case_id="case_refund_1"))
    env.step(QueryPolicyAction(policy_id="refund_policy"))
    env.step(ApproveAction(case_id="case_refund_1"))
    env.step(SendMessageAction(case_id="case_refund_1", template_id="refund_approved", slots={"amount": "350.00", "order_id": "#1001"}))
    env.step(CloseCaseAction(case_id="case_refund_1", resolution_code="done"))
    result = grade_episode(env._state)
    assert result["score"] > 0.5


def test_grader_penalizes_overdue_follow_up_and_missing_secondary_approval():
    env = OpsArenaEnvironment()
    env.reset(task_id="queue_triage", seed=7)
    env.step(OpenCaseAction(case_id="case_invoice_2"))
    env.step(
        ScheduleFollowUpAction(
            case_id="case_invoice_2",
            follow_up_at=env._state.current_time + 5,
            reason_code="awaiting_response",
        )
    )
    env.step(AdvanceClockAction(minutes=5))
    assert grade_trajectory(env._state) < 1.0

    env.reset(task_id="invoice_plus_kyc", seed=3)
    env.step(OpenCaseAction(case_id="case_invoice_1"))
    env.step(QueryPolicyAction(policy_id="invoice_policy"))
    pending = env.step(SendForSecondaryApprovalAction(case_id="case_invoice_1", reason_code="threshold_exceeded"))
    assert pending.error is None


def test_grader_penalizes_failed_qa_rework():
    env = OpsArenaEnvironment()
    env.reset(task_id="queue_triage", seed=7)
    env.step(OpenCaseAction(case_id="case_invoice_2"))
    env.step(SendForSecondaryApprovalAction(case_id="case_invoice_2", reason_code="threshold_exceeded"))
    env.step(AdvanceClockAction(minutes=15))
    env.step(ApproveAction(case_id="case_invoice_2"))
    env.step(SendToQAAction(case_id="case_invoice_2", assignee_type="qa_reviewer"))
    env.step(FailQAAction(case_id="case_invoice_2", assignee_type="qa_reviewer", reason_code="missing_documentation"))
    env.step(AdvanceClockAction(minutes=env._state.cases["case_invoice_2"].hidden.hidden_follow_up_latency_minutes or 30))
    assert grade_trajectory(env._state) < 1.0
