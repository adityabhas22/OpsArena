from server.environment import OpsArenaEnvironment
from opsarena.models import (
    AdvanceClockAction,
    ApproveAction,
    CloseCaseAction,
    OpenCaseAction,
    QueryPolicyAction,
    SendMessageAction,
)


def test_refund_happy_path_closes_case():
    env = OpsArenaEnvironment()
    env.reset(task_id="refund_exception", seed=2)
    env.step(OpenCaseAction(case_id="case_refund_1"))
    env.step(QueryPolicyAction(policy_id="refund_policy"))
    env.step(ApproveAction(case_id="case_refund_1"))
    env.step(
        SendMessageAction(
            case_id="case_refund_1",
            template_id="refund_approved",
            slots={"amount": "350.00", "order_id": "#1001"},
        )
    )
    observation = env.step(CloseCaseAction(case_id="case_refund_1", resolution_code="done"))
    assert observation.done is True
    assert env.state.cases_resolved == 1


def test_close_without_notification_is_rejected():
    env = OpsArenaEnvironment()
    env.reset(task_id="refund_exception", seed=2)
    env.step(OpenCaseAction(case_id="case_refund_1"))
    env.step(QueryPolicyAction(policy_id="refund_policy"))
    env.step(ApproveAction(case_id="case_refund_1"))
    observation = env.step(CloseCaseAction(case_id="case_refund_1", resolution_code="done"))
    assert observation.error is not None


def test_done_waits_for_scheduled_qa_sample_event():
    env = OpsArenaEnvironment()
    env.reset(task_id="refund_exception", seed=2)
    env._state.cases["case_refund_1"].hidden.qa_sample_on_close = True
    env._state.cases["case_refund_1"].hidden.qa_sample_delay_minutes = 5
    env.step(OpenCaseAction(case_id="case_refund_1"))
    env.step(QueryPolicyAction(policy_id="refund_policy"))
    env.step(ApproveAction(case_id="case_refund_1"))
    env.step(
        SendMessageAction(
            case_id="case_refund_1",
            template_id="refund_approved",
            slots={"amount": "350.00", "order_id": "#1001"},
        )
    )
    observation = env.step(CloseCaseAction(case_id="case_refund_1", resolution_code="done"))
    assert observation.done is False
    qa_due = min(event.at_time for event in env._state.scheduled_events if event.event_type == "qa_sample_selected")
    env.step(AdvanceClockAction(minutes=qa_due - env._state.current_time))
    assert env._state.cases["case_refund_1"].status == "pending_qa"
