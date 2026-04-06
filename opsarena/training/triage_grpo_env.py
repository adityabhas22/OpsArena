from __future__ import annotations

from typing import Any, Literal

from opsarena.domain.workflows.refund import PrearbitrationDecision
from opsarena.training.invoice_kyc_grpo_env import InvoiceKYCToolEnv

QUEUE_TRIAGE_SYSTEM_PROMPT = (
    "You are a senior operations supervisor managing a multi-case queue that spans "
    "refund exceptions, invoice disputes, and KYC verifications. Prioritize the queue "
    "by SLA urgency and financial exposure. Use triage tools to reorder, rebalance, "
    "and batch-assign work. Open individual cases to perform domain-specific actions "
    "(refund, invoice, KYC) as needed. Close every case whose workflow is fully "
    "resolved, and escalate when frontline resolution is insufficient."
)


def build_queue_triage_prompt_dataset(num_examples: int = 256) -> list[dict[str, Any]]:
    """Builds a conversational prompt dataset for queue-triage GRPO runs."""
    prompts: list[dict[str, Any]] = []
    for idx in range(num_examples):
        prompts.append(
            {
                "prompt": [
                    {"role": "system", "content": QUEUE_TRIAGE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Manage the mixed operations queue. Triage, prioritize, and resolve "
                            f"as many cases as possible. Training sample {idx + 1}. "
                            "Maximize final benchmark score."
                        ),
                    },
                ]
            }
        )
    return prompts


def queue_triage_terminal_benchmark_reward(
    prompts: list[Any],
    completions: list[Any],
    environments: list["QueueTriageToolEnv"],
    log_metric: Any | None = None,
    **_: Any,
) -> list[float]:
    """Scores queue-triage trajectories with progress shaping + terminal benchmark bonus.

    Multi-case episodes benefit heavily from the cases_resolved bonus, which
    encourages the agent to close 4-5 cases per episode rather than
    over-investing in a single case.

    Reward budget:
      - tool shaping:       0.0 - 0.15
      - cases resolved:     0.0 - 0.15
      - progress shaping:   0.0 - 0.10
      - terminal bonus:     0.0 - 0.60
      - invalid penalty:    up to -0.10
      Total range:          0.0 - 1.0
    """
    rewards: list[float] = []
    done_count = 0
    total_tool_calls = 0
    invalid_count = 0
    total_cases_resolved = 0

    for env in environments:
        done = env.done
        done_count += int(done)
        invalid_count += env.invalid_action_count
        valid_tools = max(env.tool_call_count - env.invalid_action_count, 0)
        total_tool_calls += env.tool_call_count
        total_cases_resolved += env.cases_resolved

        tool_shaping = min(valid_tools * 0.02, 0.15)
        cases_resolved_shaping = min(env.cases_resolved * 0.10, 0.15)
        progress_shaping = min(env.benchmark_score * 0.10, 0.10)
        terminal = env.benchmark_score * 0.60 if done else 0.0
        invalid_penalty = min(env.invalid_action_count, 5) * 0.02

        reward = tool_shaping + cases_resolved_shaping + progress_shaping + terminal - invalid_penalty
        rewards.append(max(0.0, min(1.0, reward)))

    if log_metric is not None and environments:
        count = len(environments)
        mean_benchmark = sum(env.benchmark_score for env in environments) / count
        mean_reward = sum(rewards) / count
        log_metric("env/triage_done_rate", done_count / count)
        log_metric("env/triage_invalid_actions_mean", invalid_count / count)
        log_metric("env/triage_tool_calls_mean", total_tool_calls / count)
        log_metric("env/triage_cases_resolved_mean", total_cases_resolved / count)
        log_metric("env/triage_benchmark_score_mean", mean_benchmark)
        log_metric("env/triage_reward_mean", mean_reward)
        log_metric("env/triage_terminal_score_mean", mean_reward)

    return rewards


class QueueTriageToolEnv(InvoiceKYCToolEnv):
    """Full multi-domain triage environment combining refund, invoice, KYC, and supervisor tools."""

    def __init__(
        self,
        *,
        task_id: str = "queue_triage",
        seed_sequence=None,
        random_seed: int = 17,
    ) -> None:
        super().__init__(task_id=task_id, seed_sequence=seed_sequence, random_seed=random_seed)

    # ------------------------------------------------------------------
    # Refund-specific tool methods
    # ------------------------------------------------------------------

    def accept_dispute(self, case_id: str, notes: str | None = None) -> str:
        """Accept liability on a dispute instead of contesting it.

        Args:
            case_id: Case identifier.
            notes: Optional analyst notes.

        Returns:
            The updated case state after accepting the dispute.
        """
        payload: dict[str, Any] = {"action_type": "accept_dispute", "case_id": case_id}
        if notes:
            payload["notes"] = notes
        return self._step(payload)

    def challenge_dispute(self, case_id: str, notes: str | None = None) -> str:
        """Contest a dispute after reviewing the evidence posture.

        Args:
            case_id: Case identifier.
            notes: Optional analyst notes.

        Returns:
            The updated case state after the challenge attempt.
        """
        payload: dict[str, Any] = {"action_type": "challenge_dispute", "case_id": case_id}
        if notes:
            payload["notes"] = notes
        return self._step(payload)

    def submit_dispute_evidence(
        self,
        case_id: str,
        evidence_fields: list[str],
        notes: str | None = None,
    ) -> str:
        """Submit supporting evidence for an open dispute.

        Args:
            case_id: Case identifier.
            evidence_fields: Evidence artifacts such as tracking_number or delivery_confirmation.
            notes: Optional analyst notes.

        Returns:
            The updated case state after evidence submission.
        """
        payload: dict[str, Any] = {
            "action_type": "submit_dispute_evidence",
            "case_id": case_id,
            "evidence_fields": evidence_fields,
        }
        if notes:
            payload["notes"] = notes
        return self._step(payload)

    def refund_pre_dispute_alert(self, case_id: str, approved_amount: float | None = None, notes: str | None = None) -> str:
        """Refund a pre-dispute alert such as an inquiry or RDR alert.

        Args:
            case_id: Case identifier.
            approved_amount: Optional explicit refund amount.
            notes: Optional analyst notes.

        Returns:
            The updated case state after refunding the alert.
        """
        payload: dict[str, Any] = {"action_type": "refund_pre_dispute_alert", "case_id": case_id}
        if approved_amount is not None:
            payload["approved_amount"] = approved_amount
        if notes:
            payload["notes"] = notes
        return self._step(payload)

    def resolve_prearbitration(
        self,
        case_id: str,
        prearbitration_decision: Literal["accept", "contest"],
        notes: str | None = None,
    ) -> str:
        """Resolve a pre-arbitration dispute stage.

        Args:
            case_id: Case identifier.
            prearbitration_decision: Whether to accept or contest pre-arbitration.
            notes: Optional analyst notes.

        Returns:
            The updated case state after the decision.
        """
        payload: dict[str, Any] = {
            "action_type": "resolve_prearbitration",
            "case_id": case_id,
            "prearbitration_decision": prearbitration_decision,
        }
        if notes:
            payload["notes"] = notes
        return self._step(payload)

    def freeze_payouts(
        self,
        case_id: str,
        reason_code: Literal["threshold_exceeded", "suspicious_pattern", "sla_protection"] = "threshold_exceeded",
        notes: str | None = None,
    ) -> str:
        """Freeze merchant payouts when refund risk requires containment.

        Args:
            case_id: Case identifier.
            reason_code: Why payouts are being frozen.
            notes: Optional analyst notes.

        Returns:
            The updated case state after freezing payouts.
        """
        payload: dict[str, Any] = {"action_type": "freeze_payouts", "case_id": case_id, "reason_code": reason_code}
        if notes:
            payload["notes"] = notes
        return self._step(payload)

    def set_reserve_percent(
        self,
        case_id: str,
        reserve_percent: float,
        release_after_minutes: int | None = None,
        notes: str | None = None,
    ) -> str:
        """Set a temporary reserve on merchant payouts.

        Args:
            case_id: Case identifier.
            reserve_percent: Reserve percentage from 0 to 100.
            release_after_minutes: Optional release delay.
            notes: Optional analyst notes.

        Returns:
            The updated case state after setting the reserve.
        """
        payload: dict[str, Any] = {
            "action_type": "set_reserve_percent",
            "case_id": case_id,
            "reserve_percent": reserve_percent,
        }
        if release_after_minutes is not None:
            payload["release_after_minutes"] = release_after_minutes
        if notes:
            payload["notes"] = notes
        return self._step(payload)

    def set_payout_delay_days(self, case_id: str, payout_delay_days: int, notes: str | None = None) -> str:
        """Delay merchant payouts for a fixed number of days.

        Args:
            case_id: Case identifier.
            payout_delay_days: Delay duration from 0 to 30 days.
            notes: Optional analyst notes.

        Returns:
            The updated case state after changing the payout delay.
        """
        payload: dict[str, Any] = {
            "action_type": "set_payout_delay_days",
            "case_id": case_id,
            "payout_delay_days": payout_delay_days,
        }
        if notes:
            payload["notes"] = notes
        return self._step(payload)

    # ------------------------------------------------------------------
    # Supervisor / queue management tool methods
    # ------------------------------------------------------------------

    def prioritize(self, case_id: str, priority: Literal[1, 2, 3, 4]) -> str:
        """Set the priority of a case and re-sort the queue.

        Args:
            case_id: Case identifier.
            priority: New priority level where 1 is critical and 4 is low.

        Returns:
            The updated queue state after prioritization.
        """
        return self._step({
            "action_type": "prioritize",
            "case_id": case_id,
            "new_priority": priority,
        })

    def batch_reorder(
        self,
        ordering_rule: Literal["priority", "sla", "amount"] = "priority",
    ) -> str:
        """Reorder the entire queue by the given rule.

        Args:
            ordering_rule: Sort strategy for the queue.

        Returns:
            The updated queue state after reordering.
        """
        return self._step({
            "action_type": "batch_reorder",
            "ordering_rule": ordering_rule,
        })

    def rebalance_queue(
        self,
        assignee_pool: list[str],
        strategy: Literal["sla_priority", "oldest", "amount"] = "sla_priority",
        max_cases: int = 3,
    ) -> str:
        """Rebalance unassigned cases across a pool of assignees.

        Args:
            assignee_pool: List of assignee identifiers to distribute work across.
            strategy: Rebalancing strategy.
            max_cases: Maximum number of cases to rebalance.

        Returns:
            The updated queue state after rebalancing.
        """
        return self._step({
            "action_type": "rebalance_queue",
            "assignee_pool": assignee_pool,
            "rebalance_strategy": strategy,
            "max_cases": max_cases,
        })
