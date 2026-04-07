"""Generate queue-triage SFT warm-start data from oracle-style trajectories.

This script captures full queue-triage episodes and then slices them into
shorter supervised windows so SFT trains on actionable local decisions instead
of truncating 15k-20k token transcripts down to 4k.

Usage:
    python scripts/generate_triage_sft.py --num-seeds 200
    python scripts/generate_triage_sft.py --num-seeds 200 --max-tool-calls-per-example 8
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any

from opsarena.domain.workflows.invoice import InvoiceWorkflowState, PaymentBatchStatus
from opsarena.domain.workflows.kyc import (
    BeneficialOwnerStatus,
    EDDStatus,
    KYCWorkflowState,
    SanctionsStatus,
)
from opsarena.domain.workflows.refund import (
    DisputeStage,
    MonitoringProgramStatus,
    RefundWorkflowState,
)
from opsarena.training.triage_grpo_env import QUEUE_TRIAGE_SYSTEM_PROMPT, QueueTriageToolEnv

TRIAGE_USER_INSTRUCTION = (
    "Resolve ALL cases in the queue by calling tools step by step. "
    "Triage by SLA urgency, rebalance when useful, process each case through its "
    "domain workflow, and close every case. Make exactly one tool call per turn. "
    "Do not stop until all cases are closed."
)


def _instruction_with_observation(observation: str) -> dict[str, str]:
    return {
        "role": "user",
        "content": f"{TRIAGE_USER_INSTRUCTION}\n\nEnvironment:\n{observation}",
    }


def _record_id(case: Any, record_type: str) -> str | None:
    for record in case.linked_records:
        if getattr(record.record_type, "value", record.record_type) == record_type:
            return record.record_id
    return None


def _tool_pairs(messages: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    idx = 2
    while idx + 1 < len(messages):
        assistant = messages[idx]
        tool = messages[idx + 1]
        if assistant.get("role") != "assistant" or tool.get("role") != "tool":
            break
        pairs.append((assistant, tool))
        idx += 2
    return pairs


def _chunk_trajectory(
    messages: list[dict[str, Any]],
    max_tool_calls_per_example: int,
) -> list[dict[str, Any]]:
    """Slice one long episode into smaller SFT windows."""

    if len(messages) < 4:
        return []

    system_message = messages[0]
    user_content = messages[1]["content"]
    if "Environment:\n" not in user_content:
        return [{"messages": messages}]
    current_observation = user_content.split("Environment:\n", 1)[1]

    examples: list[dict[str, Any]] = []
    pairs = _tool_pairs(messages)
    window_start_observation = current_observation
    window_messages: list[dict[str, Any]] = []
    tool_calls_in_window = 0

    for assistant, tool in pairs:
        if tool_calls_in_window == 0:
            window_messages = [system_message, _instruction_with_observation(window_start_observation)]
        window_messages.extend([assistant, tool])
        tool_calls_in_window += 1

        if tool_calls_in_window >= max_tool_calls_per_example:
            examples.append({"messages": window_messages})
            window_start_observation = tool["content"]
            tool_calls_in_window = 0

    if tool_calls_in_window > 0:
        examples.append({"messages": window_messages})

    return examples


def _capture_oracle_trajectory(seed: int) -> list[dict[str, Any]] | None:
    env = QueueTriageToolEnv(seed_sequence=[seed], random_seed=0)
    initial_obs = env.reset()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": QUEUE_TRIAGE_SYSTEM_PROMPT},
        _instruction_with_observation(initial_obs),
    ]

    def call(tool_name: str, tool_args: dict[str, Any]) -> bool:
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args),
                        },
                    }
                ],
            }
        )
        try:
            observation = getattr(env, tool_name)(**tool_args)
        except Exception:
            messages.pop()
            return False
        messages.append({"role": "tool", "content": observation, "name": tool_name})
        return True

    safety_limit = 120
    iterations = 0

    # Seed the first queue view so the model learns to inspect the queue explicitly.
    call("list_queue", {"limit": 10, "sort_by": "priority"})

    while not env.done and iterations < safety_limit:
        iterations += 1
        state = env._env._state
        if state is None:
            break

        qs = state.queue_state()
        if qs.unassigned_count > 0:
            call(
                "rebalance_queue",
                {
                    "assignee_pool": ["analyst_1", "analyst_2"],
                    "strategy": "sla_priority",
                    "max_cases": 3,
                },
            )

        case_id = None
        for candidate in state.queue_order:
            if state.cases[candidate].status != "closed":
                case_id = candidate
                break

        if case_id is None:
            if state.scheduled_events:
                next_due = min(event.at_time for event in state.scheduled_events)
                call("advance_clock", {"minutes": max(1, next_due - state.current_time)})
                continue
            break

        case = state.cases[case_id]

        if getattr(case.qa_status, "value", case.qa_status) == "pending":
            call("open_case", {"case_id": case_id})
            call("approve_qa", {"case_id": case_id})
            call("close_case", {"case_id": case_id})
            continue

        if case.case_type.value == "refund" and isinstance(case.workflow, RefundWorkflowState):
            pending = [event.at_time for event in state.scheduled_events if event.case_id == case_id]
            if (
                case.workflow.dispute_workflow_status
                in {"submitted", "challenged", "inquiry_contested", "prearbitration_contested"}
                and pending
            ):
                call("advance_clock", {"minutes": max(1, min(pending) - state.current_time)})
                continue

        if case.case_type.value == "kyc" and isinstance(case.workflow, KYCWorkflowState):
            pending = [event.at_time for event in state.scheduled_events if event.case_id == case_id]
            if (
                case.pending_info_fields
                or case.workflow.sanctions_status == SanctionsStatus.POTENTIAL_MATCH
                or case.workflow.edd_status == EDDStatus.AWAITING_RESPONSE
            ) and pending:
                call("advance_clock", {"minutes": max(1, min(pending) - state.current_time)})
                continue

        call("open_case", {"case_id": case_id})

        state = env._env._state
        if state is None or case_id not in state.cases:
            break
        case = state.cases[case_id]

        if case.case_type.value == "refund":
            workflow = case.workflow
            if not isinstance(workflow, RefundWorkflowState):
                continue

            call("query_policy", {"policy_id": "refund_policy"})

            if workflow.monitoring_program_status == MonitoringProgramStatus.BREACHED and not workflow.payout_frozen:
                call("freeze_payouts", {"case_id": case_id, "reason_code": "threshold_exceeded"})
            if workflow.merchant_risk_level.value in {"high", "critical"} and workflow.reserve_percent == 0:
                call("set_reserve_percent", {"case_id": case_id, "reserve_percent": 15.0, "release_after_minutes": 25})
            if workflow.merchant_risk_level.value in {"elevated", "high", "critical"} and workflow.payout_delay_days == 0:
                call("set_payout_delay_days", {"case_id": case_id, "payout_delay_days": 7})

            if case.resolution.value == "pending":
                if workflow.dispute_stage == DisputeStage.INQUIRY:
                    if case.hidden.true_dispute_should_accept or case.amount <= 75:
                        call("refund_pre_dispute_alert", {"case_id": case_id})
                    else:
                        call("challenge_dispute", {"case_id": case_id})
                elif workflow.dispute_stage == DisputeStage.PRE_ARBITRATION:
                    decision = "accept" if case.hidden.true_dispute_should_accept else "contest"
                    call("resolve_prearbitration", {"case_id": case_id, "prearbitration_decision": decision})
                elif workflow.dispute_stage == DisputeStage.LOST:
                    call("accept_dispute", {"case_id": case_id})
                elif workflow.dispute_stage == DisputeStage.CHARGEBACK_OPEN and not case.hidden.true_dispute_should_accept:
                    if not workflow.dispute_evidence_fields:
                        call(
                            "submit_dispute_evidence",
                            {
                                "case_id": case_id,
                                "evidence_fields": [
                                    "customer_communication",
                                    "tracking_number",
                                    "delivery_confirmation",
                                ],
                            },
                        )
                    else:
                        call("challenge_dispute", {"case_id": case_id})
                elif case.hidden.true_dispute_should_accept and case.amount <= 75:
                    call("accept_dispute", {"case_id": case_id})
                elif case.hidden.true_fraud_risk > 0.7:
                    call("reject", {"case_id": case_id, "reason_code": "suspicious_pattern"})
                else:
                    call("approve", {"case_id": case_id})

        elif case.case_type.value == "invoice":
            workflow = case.workflow
            if not isinstance(workflow, InvoiceWorkflowState):
                continue

            if workflow.payment_batch_status in {PaymentBatchStatus.SCHEDULED, PaymentBatchStatus.IN_PROGRESS}:
                if case.hidden.true_is_duplicate or (workflow.variance_amount and workflow.variance_amount > 0):
                    call("remove_from_payment_batch", {"case_id": case_id})

            refreshed = env._env._state.cases[case_id].workflow
            if (
                isinstance(refreshed, InvoiceWorkflowState)
                and refreshed.payment_batch_status == PaymentBatchStatus.COMPLETED
                and refreshed.stop_payment_window_until is not None
                and env._env._state.current_time <= refreshed.stop_payment_window_until
                and case.hidden.true_is_duplicate
            ):
                call("stop_payment", {"case_id": case_id})
                call("advance_clock", {"minutes": 6})

            if case.resolution.value == "pending":
                if case.hidden.true_is_duplicate:
                    call("reject", {"case_id": case_id, "reason_code": "duplicate_match"})
                else:
                    invoice_id = _record_id(case, "invoice")
                    po_id = _record_id(case, "purchase_order")
                    if invoice_id:
                        call("view_record", {"record_type": "invoice", "record_id": invoice_id})
                    if po_id:
                        call("view_record", {"record_type": "purchase_order", "record_id": po_id})

                    if case.pending_info_fields:
                        call("request_info", {"case_id": case_id, "field_name": "goods_receipt"})
                        call("advance_clock", {"minutes": case.hidden.hidden_response_latency_minutes or 30})
                        receipt_id = _record_id(env._env._state.cases[case_id], "receipt")
                        if receipt_id:
                            call("view_record", {"record_type": "receipt", "record_id": receipt_id})

                    call("query_policy", {"policy_id": "invoice_policy"})

                    current_workflow = env._env._state.cases[case_id].workflow
                    if isinstance(current_workflow, InvoiceWorkflowState):
                        match_status = current_workflow.match_status.value
                        variance_amount = current_workflow.variance_amount
                        call_args: dict[str, Any] = {
                            "case_id": case_id,
                            "match_status": match_status,
                        }
                        if variance_amount is not None:
                            call_args["variance_amount"] = variance_amount
                        call("record_three_way_match", call_args)

                        current_workflow = env._env._state.cases[case_id].workflow
                        if current_workflow.variance_amount and current_workflow.variance_amount > 0:
                            if current_workflow.variance_amount <= current_workflow.write_off_threshold:
                                call("write_off_small_balance", {"case_id": case_id, "reason_code": "complete"})
                            elif case.hidden.true_vendor_will_respond:
                                call("request_revised_invoice", {"case_id": case_id, "reason_code": "threshold_exceeded"})
                                call("advance_clock", {"minutes": case.hidden.true_vendor_response_minutes or 25})
                                post_workflow = env._env._state.cases[case_id].workflow
                                if isinstance(post_workflow, InvoiceWorkflowState) and post_workflow.credit_memo_status.value == "received":
                                    memo_id = _record_id(env._env._state.cases[case_id], "credit_memo")
                                    if memo_id:
                                        call("apply_credit_memo", {"case_id": case_id, "credit_memo_id": memo_id})

                        if current_workflow.secondary_approval_required:
                            call("send_for_secondary_approval", {"case_id": case_id, "reason_code": "threshold_exceeded"})
                            call("advance_clock", {"minutes": 15})

                        final_workflow = env._env._state.cases[case_id].workflow
                        if isinstance(final_workflow, InvoiceWorkflowState):
                            if final_workflow.credit_memo_status.value == "received":
                                memo_id = _record_id(env._env._state.cases[case_id], "credit_memo")
                                if memo_id:
                                    call("apply_credit_memo", {"case_id": case_id, "credit_memo_id": memo_id})
                            if final_workflow.payment_hold:
                                call("release_payment_hold", {"case_id": case_id})

                    call("approve", {"case_id": case_id})

        elif case.case_type.value == "kyc":
            workflow = case.workflow
            if not isinstance(workflow, KYCWorkflowState):
                continue

            if case.resolution.value == "pending":
                call("query_policy", {"policy_id": "kyc_policy"})

                if workflow.sanctions_status == SanctionsStatus.NOT_STARTED:
                    call("run_sanctions_screen", {"case_id": case_id})
                    workflow = env._env._state.cases[case_id].workflow

                if workflow.sanctions_status == SanctionsStatus.CONFIRMED_MATCH:
                    if not workflow.payments_frozen:
                        call("freeze_payments", {"case_id": case_id, "reason": "sanctions_match"})
                    if workflow.ofac_report_status.value in {"pending", "missed"}:
                        call("file_ofac_report", {"case_id": case_id})
                    call("review_kyc", {"case_id": case_id, "verification_decision": "reject"})
                    call("reject", {"case_id": case_id, "reason_code": "sanctions_match"})
                else:
                    if workflow.edd_status == EDDStatus.NOT_STARTED and (
                        case.hidden.true_edd_required or case.hidden.true_beneficial_owner_issue
                    ):
                        call("start_edd_review", {"case_id": case_id})
                        workflow = env._env._state.cases[case_id].workflow

                    if workflow.beneficial_owner_status == BeneficialOwnerStatus.NEEDS_CORRECTION and workflow.correction_fields:
                        for field_name in list(workflow.correction_fields):
                            if field_name not in env._env._state.cases[case_id].pending_info_fields:
                                call("request_field_correction", {"case_id": case_id, "field_name": field_name})
                        workflow = env._env._state.cases[case_id].workflow

                    if workflow.beneficial_owner_status == BeneficialOwnerStatus.PENDING_REVIEW and not env._env._state.cases[case_id].pending_info_fields:
                        call("review_beneficial_owner", {"case_id": case_id, "decision": "approve"})

                    current_case = env._env._state.cases[case_id]
                    current_workflow = current_case.workflow
                    if (
                        isinstance(current_workflow, KYCWorkflowState)
                        and current_case.resolution.value == "pending"
                        and not current_workflow.kyc_complete
                    ):
                        call("request_info", {"case_id": case_id, "field_name": "individual.verification.document"})
                        call("advance_clock", {"minutes": current_case.hidden.hidden_response_latency_minutes or 45})

                    current_case = env._env._state.cases[case_id]
                    if current_case.resolution.value == "pending":
                        kyc_doc_id = _record_id(current_case, "kyc_document")
                        if kyc_doc_id:
                            call("view_record", {"record_type": "kyc_document", "record_id": kyc_doc_id})
                    if current_case.resolution.value == "pending" and not current_case.hidden.true_sanctions_match:
                        if not current_case.hidden.true_doc_valid:
                            call("review_kyc", {"case_id": case_id, "verification_decision": "reject"})
                            call("reject", {"case_id": case_id, "reason_code": "invalid_document"})
                        else:
                            call("review_kyc", {"case_id": case_id, "verification_decision": "approve"})
                            call("approve", {"case_id": case_id})

        state = env._env._state
        if state is None or case_id not in state.cases:
            break
        case = state.cases[case_id]
        if case.requires_customer_notification and not case.customer_notified and case.resolution.value != "pending":
            template = (
                "refund_approved"
                if case.case_type.value == "refund" and case.resolution.value == "approved"
                else "case_closed"
            )
            call("send_message", {"case_id": case_id, "template_id": template})
        if case.qa_required and getattr(case.qa_status, "value", case.qa_status) != "passed":
            call("send_to_qa", {"case_id": case_id})
            call("approve_qa", {"case_id": case_id})
        if case.resolution.value != "pending":
            call("close_case", {"case_id": case_id})

    if not env.done:
        return None
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--output", default="artifacts/triage-sft-warmstart.jsonl")
    parser.add_argument("--max-tool-calls-per-example", type=int, default=8)
    args = parser.parse_args()

    rng = random.Random(42)
    seeds = [rng.randint(0, 1_000_000) for _ in range(args.num_seeds)]

    trajectories = 0
    examples_written = 0
    total_tool_calls = 0

    with open(args.output, "w") as handle:
        for seed in seeds:
            try:
                messages = _capture_oracle_trajectory(seed)
            except Exception as exc:
                print(f"  seed={seed}: error={exc}")
                continue
            if messages is None:
                print(f"  seed={seed}: failed (incomplete)")
                continue

            trajectories += 1
            tool_call_count = sum(
                1 for message in messages if message.get("role") == "assistant" and message.get("tool_calls")
            )
            total_tool_calls += tool_call_count

            for example in _chunk_trajectory(messages, args.max_tool_calls_per_example):
                handle.write(json.dumps(example) + "\n")
                examples_written += 1

    average_calls = total_tool_calls / max(1, trajectories)
    print(f"Generated {trajectories}/{len(seeds)} triage trajectories → {args.output}")
    print(f"Wrote {examples_written} SFT windows ({args.max_tool_calls_per_example} tool calls per example)")
    print(f"Average tool calls per trajectory: {average_calls:.1f}")


if __name__ == "__main__":
    main()
