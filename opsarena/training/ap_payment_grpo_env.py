from __future__ import annotations

from typing import Any, Literal

from opsarena.training._base_tool_env import BaseToolEnv

AP_PAYMENT_RUN_SYSTEM_PROMPT = (
    "You are an accounts payable analyst managing the payment batch lifecycle. "
    "Use the tools to perform three-way matching, manage payment holds, "
    "request credit memos and revised invoices, stop payments when needed, "
    "record vendor refunds, write off small balances, follow AP policy, "
    "send required communications, complete QA if needed, and close cases "
    "only when the workflow is fully resolved."
)


def build_ap_payment_prompt_dataset(num_examples: int = 256) -> list[dict[str, Any]]:
    """Builds a conversational prompt dataset for AP payment run GRPO runs."""
    prompts: list[dict[str, Any]] = []
    for idx in range(num_examples):
        prompts.append(
            {
                "prompt": [
                    {"role": "system", "content": AP_PAYMENT_RUN_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Resolve the active AP payment run episode. "
                            f"Training sample {idx + 1}. Maximize final benchmark score."
                        ),
                    },
                ]
            }
        )
    return prompts


def ap_payment_terminal_benchmark_reward(
    prompts: list[Any],
    completions: list[Any],
    environments: list["APPaymentRunToolEnv"],
    log_metric: Any | None = None,
    **_: Any,
) -> list[float]:
    """Scores AP payment run trajectories with progress shaping + terminal benchmark bonus.

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

    for env in environments:
        done = env.done
        done_count += int(done)
        invalid_count += env.invalid_action_count
        valid_tools = max(env.tool_call_count - env.invalid_action_count, 0)
        total_tool_calls += env.tool_call_count

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
        log_metric("env/ap_payment_done_rate", done_count / count)
        log_metric("env/ap_payment_invalid_actions_mean", invalid_count / count)
        log_metric("env/ap_payment_tool_calls_mean", total_tool_calls / count)
        log_metric("env/ap_payment_benchmark_score_mean", mean_benchmark)
        log_metric("env/ap_payment_reward_mean", mean_reward)
        log_metric("env/ap_payment_terminal_score_mean", mean_reward)

    return rewards


class APPaymentRunToolEnv(BaseToolEnv):
    """Invoice-only AP payment run tool environment for GRPO."""

    def __init__(
        self,
        *,
        task_id: str = "ap_payment_run",
        seed_sequence=None,
        random_seed: int = 17,
    ) -> None:
        super().__init__(task_id=task_id, seed_sequence=seed_sequence, random_seed=random_seed)

    # ------------------------------------------------------------------
    # Invoice-specific tool methods
    # ------------------------------------------------------------------

    def record_three_way_match(
        self,
        case_id: str,
        match_status: Literal["matched", "variance", "missing_receipt", "duplicate", "unmatched"],
        variance_amount: float | None = None,
        notes: str | None = None,
    ) -> str:
        """Record the result of a three-way match between invoice, PO, and receipt.

        Args:
            case_id: Case identifier.
            match_status: Outcome of the three-way match.
            variance_amount: Required when match_status is variance.
            notes: Optional analyst notes.

        Returns:
            The updated case state after recording the match.
        """
        payload: dict[str, Any] = {
            "action_type": "record_three_way_match",
            "case_id": case_id,
            "match_status": match_status,
        }
        if variance_amount is not None:
            payload["variance_amount"] = variance_amount
        if notes:
            payload["notes"] = notes
        return self._step(payload)

    def place_payment_hold(
        self,
        case_id: str,
        reason: Literal[
            "threshold_exceeded",
            "suspicious_pattern",
            "missing_documentation",
            "policy_ambiguity",
            "duplicate_match",
        ],
    ) -> str:
        """Place a payment hold on an invoice case to prevent premature payment.

        Args:
            case_id: Case identifier.
            reason: Reason code for the hold.

        Returns:
            The updated case state after placing the hold.
        """
        return self._step({
            "action_type": "place_payment_hold",
            "case_id": case_id,
            "reason_code": reason,
        })

    def release_payment_hold(self, case_id: str) -> str:
        """Release a previously placed payment hold on an invoice case.

        Args:
            case_id: Case identifier.

        Returns:
            The updated case state after releasing the hold.
        """
        return self._step({"action_type": "release_payment_hold", "case_id": case_id})

    def request_credit_memo(
        self,
        case_id: str,
        amount: float | None = None,
        notes: str | None = None,
    ) -> str:
        """Request a credit memo from the vendor for an invoice variance.

        Args:
            case_id: Case identifier.
            amount: Optional explicit credit memo amount.
            notes: Optional analyst notes.

        Returns:
            The updated case state after requesting the credit memo.
        """
        payload: dict[str, Any] = {"action_type": "request_credit_memo", "case_id": case_id}
        if amount is not None:
            payload["approved_amount"] = amount
        if notes:
            payload["notes"] = notes
        return self._step(payload)

    def request_revised_invoice(
        self,
        case_id: str,
        reason_code: Literal[
            "threshold_exceeded",
            "suspicious_pattern",
            "missing_documentation",
            "policy_ambiguity",
            "duplicate_match",
            "invalid_document",
        ] = "missing_documentation",
        notes: str | None = None,
    ) -> str:
        """Request a revised invoice from the vendor to correct discrepancies.

        Args:
            case_id: Case identifier.
            reason_code: Reason for requesting the revision.
            notes: Optional analyst notes.

        Returns:
            The updated case state after requesting the revised invoice.
        """
        payload: dict[str, Any] = {
            "action_type": "request_revised_invoice",
            "case_id": case_id,
            "reason_code": reason_code,
        }
        if notes:
            payload["notes"] = notes
        return self._step(payload)

    def request_po_change(
        self,
        case_id: str,
        change_description: str,
        notes: str | None = None,
    ) -> str:
        """Request a purchase order change to align PO with invoice.

        Args:
            case_id: Case identifier.
            change_description: Description of the PO change needed.
            notes: Optional analyst notes.

        Returns:
            The updated case state after requesting the PO change.
        """
        payload: dict[str, Any] = {
            "action_type": "request_po_change",
            "case_id": case_id,
            "change_description": change_description,
        }
        if notes:
            payload["notes"] = notes
        return self._step(payload)

    def remove_from_payment_batch(self, case_id: str) -> str:
        """Remove an invoice from its scheduled payment batch.

        Args:
            case_id: Case identifier.

        Returns:
            The updated case state after removal from the batch.
        """
        return self._step({"action_type": "remove_from_payment_batch", "case_id": case_id})

    def stop_payment(self, case_id: str, notes: str | None = None) -> str:
        """Issue a stop payment for an in-progress payment batch item.

        Args:
            case_id: Case identifier.
            notes: Optional analyst notes.

        Returns:
            The updated case state after the stop payment request.
        """
        payload: dict[str, Any] = {"action_type": "stop_payment", "case_id": case_id}
        if notes:
            payload["notes"] = notes
        return self._step(payload)

    def record_vendor_refund(self, case_id: str, amount: float) -> str:
        """Record a refund received from the vendor for overpayment recovery.

        Args:
            case_id: Case identifier.
            amount: Refund amount received from the vendor.

        Returns:
            The updated case state after recording the vendor refund.
        """
        return self._step({
            "action_type": "record_vendor_refund",
            "case_id": case_id,
            "refund_amount": amount,
        })

    def apply_credit_memo(self, case_id: str, credit_memo_id: str, notes: str | None = None) -> str:
        """Apply a received credit memo to an invoice case.

        Args:
            case_id: Case identifier.
            credit_memo_id: Identifier of the credit memo to apply.
            notes: Optional analyst notes.

        Returns:
            The updated case state after applying the credit memo.
        """
        payload: dict[str, Any] = {
            "action_type": "apply_credit_memo",
            "case_id": case_id,
            "credit_memo_id": credit_memo_id,
        }
        if notes:
            payload["notes"] = notes
        return self._step(payload)

    def write_off_small_balance(
        self,
        case_id: str,
        reason_code: Literal[
            "threshold_exceeded",
            "suspicious_pattern",
            "missing_documentation",
            "policy_ambiguity",
            "duplicate_match",
            "invalid_document",
        ] = "threshold_exceeded",
        notes: str | None = None,
    ) -> str:
        """Write off a small remaining variance balance below the threshold.

        Args:
            case_id: Case identifier.
            reason_code: Reason justifying the write-off.
            notes: Optional analyst notes.

        Returns:
            The updated case state after the write-off.
        """
        payload: dict[str, Any] = {
            "action_type": "write_off_small_balance",
            "case_id": case_id,
            "reason_code": reason_code,
        }
        if notes:
            payload["notes"] = notes
        return self._step(payload)
