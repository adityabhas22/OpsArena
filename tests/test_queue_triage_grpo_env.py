from opsarena.training.triage_grpo_env import QueueTriageToolEnv, queue_triage_terminal_benchmark_reward


def test_queue_triage_reward_penalizes_noop_completion():
    env = QueueTriageToolEnv(random_seed=0)
    env.reset()

    rewards = queue_triage_terminal_benchmark_reward([], [""], [env])

    assert rewards[0] < 0.0


def test_queue_triage_reward_penalizes_malformed_tool_call_more_than_noop():
    env = QueueTriageToolEnv(random_seed=0)
    env.reset()

    noop_reward = queue_triage_terminal_benchmark_reward([], [""], [env])[0]
    malformed_reward = queue_triage_terminal_benchmark_reward(
        [], ["<tool_call>\nlist_queue\n</tool_call>"], [env]
    )[0]

    assert malformed_reward < noop_reward


def test_queue_triage_reward_becomes_positive_after_real_progress():
    env = QueueTriageToolEnv(random_seed=0)
    env.reset()
    env.list_queue()
    env.rebalance_queue(["analyst_1", "analyst_2"], strategy="sla_priority", max_cases=3)
    env.open_case("case_refund_1")

    rewards = queue_triage_terminal_benchmark_reward(
        [],
        ["<tool_call><function=open_case><parameter=case_id>case_refund_1</parameter></function></tool_call>"],
        [env],
    )

    assert rewards[0] > 0.0
