from __future__ import annotations

from opsarena.training import RefundExceptionToolEnv, refund_terminal_benchmark_reward


def test_refund_grpo_env_reset_renders_training_text():
    env = RefundExceptionToolEnv(seed_sequence=[7])
    text = env.reset()

    assert "refund_exception episode started with seed 7" in text
    assert "available_actions:" in text
    assert "benchmark_score: 0.000" in text


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


def test_refund_terminal_benchmark_reward_gives_shaping_but_prefers_completion():
    done_env = RefundExceptionToolEnv(seed_sequence=[7])
    done_env.reset()
    done_env.open_case("case_refund_1")
    done_env.query_policy()
    done_env.submit_dispute_evidence("case_refund_1", ["customer_communication", "tracking_number", "delivery_confirmation"])
    done_env.advance_clock(21)
    done_env.send_message("case_refund_1", template_id="case_closed")
    done_env.close_case("case_refund_1")

    incomplete_env = RefundExceptionToolEnv(seed_sequence=[7])
    incomplete_env.reset()
    incomplete_env.open_case("case_refund_1")

    rewards = refund_terminal_benchmark_reward(
        prompts=["p1", "p2"],
        completions=["c1", "c2"],
        environments=[done_env, incomplete_env],
    )

    assert rewards[0] > 0.85
    assert rewards[1] > 0.0
    assert rewards[1] < rewards[0]
