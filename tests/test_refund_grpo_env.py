from __future__ import annotations

from opsarena.training import RefundExceptionToolEnv, refund_terminal_benchmark_reward
from opsarena.training.refund_grpo_env import _refund_milestone_score


def test_refund_grpo_env_reset_renders_compact_observation():
    env = RefundExceptionToolEnv(seed_sequence=[7])
    text = env.reset()

    assert "refund_exception episode started with seed 7" in text
    assert "actions:" in text
    # Compact observation should NOT expose benchmark_score to the model
    assert "benchmark_score:" not in text


def test_refund_grpo_env_can_run_oracle_like_happy_path():
    env = RefundExceptionToolEnv(seed_sequence=[7])
    env.reset()
    env.open_case("case_refund_1")
    env.query_policy()
    env.submit_dispute_evidence("case_refund_1", ["customer_communication", "tracking_number", "delivery_confirmation"])
    env.advance_clock(21)
    env.send_message("case_refund_1", template_id="case_closed")
    env.close_case("case_refund_1")

    assert env.done is True
    assert env.benchmark_score > 0.9


def test_refund_reward_milestones_give_gradient_before_completion():
    """An incomplete episode with some correct milestones should score > 0."""
    env = RefundExceptionToolEnv(seed_sequence=[7])
    env.reset()
    env.open_case("case_refund_1")
    env.query_policy()
    # Opened + policy_queried = 2 of 6 milestones

    rewards = refund_terminal_benchmark_reward(
        prompts=["p1"],
        completions=["c1"],
        environments=[env],
    )
    assert rewards[0] > 0.0, "Milestones should provide non-zero shaping"


def test_refund_reward_completion_dominates_milestones():
    """A completed episode should score much higher than partial milestones."""
    done_env = RefundExceptionToolEnv(seed_sequence=[7])
    done_env.reset()
    done_env.open_case("case_refund_1")
    done_env.query_policy()
    done_env.submit_dispute_evidence("case_refund_1", ["customer_communication", "tracking_number", "delivery_confirmation"])
    done_env.advance_clock(21)
    done_env.send_message("case_refund_1", template_id="case_closed")
    done_env.close_case("case_refund_1")

    partial_env = RefundExceptionToolEnv(seed_sequence=[7])
    partial_env.reset()
    partial_env.open_case("case_refund_1")
    partial_env.query_policy()

    rewards = refund_terminal_benchmark_reward(
        prompts=["p1", "p2"],
        completions=["c1", "c2"],
        environments=[done_env, partial_env],
    )

    assert rewards[0] > 0.85, f"Completed episode should score high, got {rewards[0]}"
    assert rewards[1] > 0.0, "Partial milestones should score > 0"
    assert rewards[1] < rewards[0], "Terminal bonus should dominate"


def test_refund_reward_step_cost_penalizes_looping():
    """Many tool calls without progress should be penalized."""
    env = RefundExceptionToolEnv(seed_sequence=[7])
    env.reset()
    # Burn steps by repeatedly listing queue (low-value action)
    for _ in range(15):
        env.list_queue()

    rewards = refund_terminal_benchmark_reward(
        prompts=["p1"],
        completions=["c1"],
        environments=[env],
    )
    # Should be low due to step cost eating into the small milestone shaping
    assert rewards[0] < 0.15, f"Looping agent should score low, got {rewards[0]}"


def test_refund_milestone_score_function():
    """The _refund_milestone_score function should track grader-aligned milestones."""
    env = RefundExceptionToolEnv(seed_sequence=[7])
    env.reset()

    score_before, _ = _refund_milestone_score(env)
    assert score_before >= 0.0

    env.open_case("case_refund_1")
    env.query_policy()

    score_after, milestones_after = _refund_milestone_score(env)
    assert score_after > score_before, "Opening case + querying policy should increase milestone score"
    assert milestones_after.get("case_refund_1/opened") is True
    assert milestones_after.get("case_refund_1/policy_queried") is True
