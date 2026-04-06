from __future__ import annotations

from typing import Any, Literal

from opsarena.training._base_tool_env import BaseToolEnv

AP_PAYMENT_RUN_SYSTEM_PROMPT = (
    "You are an accounts payable analyst managing the payment batch lifecycle.\n\n"
    "RULES:\n"
    "- Make exactly ONE tool call per turn. Use exact IDs from the observation.\n"
    "- Workflow per case:\n"
    "  1. list_queue → open_case\n"
    "  2. view_record for invoice, purchase_order, receipt\n"
    "  3. record_three_way_match\n"
    "  4. query_policy before approve/reject\n"
    "  5. Handle variance: request_credit_memo / write_off_small_balance / request_revised_invoice\n"
    "  6. If duplicate: remove_from_payment_batch or stop_payment → reject\n"
    "  7. approve or reject → send_message if required → QA if required → close_case\n"
    "- Check 'obligations' line to see what remains before close.\n"
    "- Be concise. Do not explain your reasoning at length."
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
    """Scores AP payment run trajectories with grader-aligned milestones + terminal bonus.

    Reward budget:
      - milestone shaping:  0.0 - 0.25  (grader-aligned process milestones)
      - terminal bonus:     0.0 - 0.70  (benchmark_score * 0.70 if done)
      - step cost:          0.0 - 0.05  (0.004 per step beyond 15)
      - invalid penalty:    0.0 - 0.10
      Total range:          0.0 - 1.0
    """
    rewards: list[float] = []
    done_count = 0
    total_tool_calls = 0
    invalid_count = 0
    total_milestones_hit = 0
    total_milestones = 0

    for env in environments:
        done = env.done
        done_count += int(done)
        invalid_count += env.invalid_action_count
        total_tool_calls += env.tool_call_count

        milestone_score, milestones = env.milestone_score(cap=0.25)
        total_milestones_hit += sum(1 for v in milestones.values() if v)
        total_milestones += len(milestones)

        terminal = env.benchmark_score * 0.70 if done else 0.0
        step_cost = min(max(0, env.tool_call_count - 15) * 0.004, 0.05)
        invalid_penalty = min(env.invalid_action_count, 5) * 0.02

        reward = milestone_score + terminal - step_cost - invalid_penalty
        rewards.append(max(0.0, min(1.0, reward)))

    if log_metric is not None and environments:
        count = len(environments)
        mean_benchmark = sum(env.benchmark_score for env in environments) / count
        mean_reward = sum(rewards) / count
        milestone_rate = total_milestones_hit / max(1, total_milestones)
        log_metric("env/ap_payment_done_rate", done_count / count)
        log_metric("env/ap_payment_invalid_actions_mean", invalid_count / count)
        log_metric("env/ap_payment_tool_calls_mean", total_tool_calls / count)
        log_metric("env/ap_payment_benchmark_score_mean", mean_benchmark)
        log_metric("env/ap_payment_reward_mean", mean_reward)
        log_metric("env/ap_payment_milestone_rate", milestone_rate)
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
