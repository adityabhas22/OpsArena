from server.environment import OpsArenaEnvironment
from opsarena.models import OpenCaseAction, QueryPolicyAction


def _run():
    env = OpsArenaEnvironment()
    env.reset(task_id="refund_exception", seed=5, episode_id="ep-fixed")
    env.step(OpenCaseAction(case_id="case_refund_1"))
    env.step(QueryPolicyAction(policy_id="refund_policy"))
    return env.state.model_dump()


def test_replay_is_deterministic_for_same_seed_and_actions():
    assert _run() == _run()
