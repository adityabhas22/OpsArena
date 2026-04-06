"""Generate SFT warm-start data for queue_triage from oracle trajectories.

Instruments the oracle to capture every tool call as chat messages with
proper Qwen3 tool-call format. Produces a JSONL dataset for SFT.

Usage:
    python scripts/generate_triage_sft.py --num-seeds 100
    python scripts/generate_triage_sft.py --num-seeds 200 --output artifacts/triage-sft-warmstart.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from typing import Any

from opsarena.models import RawOpsAction
from opsarena.training.triage_grpo_env import QUEUE_TRIAGE_SYSTEM_PROMPT, QueueTriageToolEnv
from server.environment import OpsArenaEnvironment


def _capture_oracle_trajectory(seed: int) -> list[dict[str, Any]] | None:
    """Run the oracle on queue_triage, capturing actions as chat messages."""
    # Use the GRPO env for consistent observation rendering
    env = QueueTriageToolEnv(seed_sequence=[seed], random_seed=0)
    initial_obs = env.reset()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": QUEUE_TRIAGE_SYSTEM_PROMPT},
        {"role": "user", "content": (
            "Resolve ALL cases in the queue by calling tools step by step. "
            "Triage by SLA urgency, process each case through its domain workflow, "
            "close every case. Do not stop until all cases are closed.\n\n"
            f"Environment:\n{initial_obs}"
        )},
    ]

    # Now replay the oracle's decisions through the GRPO env
    from baselines.oracle import run_oracle, _next_active_case_id
    from opsarena.domain.workflows.refund import DisputeStage, MonitoringProgramStatus, PrearbitrationDecision, RefundWorkflowState
    from opsarena.domain.workflows.invoice import InvoiceWorkflowState, PaymentBatchStatus
    from opsarena.domain.workflows.kyc import BeneficialOwnerStatus, EDDStatus, KYCWorkflowState, SanctionsStatus
    from opsarena.enums import ReasonCode, VerificationDecision

    state = env._env._state
    if state is None:
        return None

    safety_limit = 120
    iterations = 0

    def call(tool_name: str, tool_args: dict) -> bool:
        """Execute tool, append to messages. Returns True if successful."""
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {
                "name": tool_name,
                "arguments": json.dumps(tool_args),
            }}],
        })
        try:
            fn = getattr(env, tool_name)
            obs = fn(**tool_args)
            messages.append({"role": "tool", "content": obs, "name": tool_name})
            return True
        except Exception as e:
            messages.pop()
            return False

    while not env.done and iterations < safety_limit:
        iterations += 1
        state = env._env._state
        if state is None:
            break

        # Rebalance if unassigned
        qs = state.queue_state()
        if qs.unassigned_count > 0:
            call("rebalance_queue", {
                "assignee_pool": ["analyst_1", "analyst_2"],
                "strategy": "sla_priority",
                "max_cases": 3,
            })

        # Find next active case
        case_id = None
        for cid in state.queue_order:
            if state.cases[cid].status != "closed":
                case_id = cid
                break
        if case_id is None:
            if state.scheduled_events:
                next_due = min(e.at_time for e in state.scheduled_events)
                call("advance_clock", {"minutes": max(1, next_due - state.current_time)})
                continue
            break

        case = state.cases[case_id]

        # Wait for pending events on this case
        if case.case_type.value == "refund" and isinstance(case.workflow, RefundWorkflowState):
            pending = [e.at_time for e in state.scheduled_events if e.case_id == case_id]
            if case.workflow.dispute_workflow_status in {"submitted", "challenged", "inquiry_contested", "prearbitration_contested"} and pending:
                call("advance_clock", {"minutes": max(1, min(pending) - state.current_time)})
                continue
        if case.case_type.value == "kyc" and isinstance(case.workflow, KYCWorkflowState):
            pending = [e.at_time for e in state.scheduled_events if e.case_id == case_id]
            if (case.pending_info_fields or case.workflow.sanctions_status == SanctionsStatus.POTENTIAL_MATCH
                or case.workflow.edd_status == EDDStatus.AWAITING_RESPONSE) and pending:
                call("advance_clock", {"minutes": max(1, min(pending) - state.current_time)})
                continue

        # QA pending? Approve it
        if getattr(case.qa_status, "value", case.qa_status) == "pending":
            call("open_case", {"case_id": case_id})
            call("approve_qa", {"case_id": case_id})
            call("close_case", {"case_id": case_id})
            continue

        # Open and process case
        call("open_case", {"case_id": case_id})

        if case.case_type.value == "refund":
            workflow = case.workflow
            if not isinstance(workflow, RefundWorkflowState):
                continue
            call("query_policy", {"policy_id": "refund_policy"})

            if workflow.monitoring_program_status == MonitoringProgramStatus.BREACHED and not workflow.payout_frozen:
                call("freeze_payouts", {"case_id": case_id, "reason_code": "threshold_exceeded"})

            if case.resolution.value == "pending":
                if workflow.dispute_stage == DisputeStage.INQUIRY:
                    if case.hidden.true_dispute_should_accept or case.amount <= 75:
                        call("refund_pre_dispute_alert", {"case_id": case_id})
                    else:
                        call("challenge_dispute", {"case_id": case_id})
                elif workflow.dispute_stage == DisputeStage.CHARGEBACK_OPEN:
                    if case.hidden.true_dispute_should_accept:
                        call("accept_dispute", {"case_id": case_id})
                    elif not workflow.dispute_evidence_fields:
                        call("submit_dispute_evidence", {
                            "case_id": case_id,
                            "evidence_fields": ["customer_communication", "tracking_number", "delivery_confirmation"],
                        })
                    else:
                        call("challenge_dispute", {"case_id": case_id})
                elif workflow.dispute_stage == DisputeStage.PRE_ARBITRATION:
                    decision = "accept" if case.hidden.true_dispute_should_accept else "contest"
                    call("resolve_prearbitration", {"case_id": case_id, "prearbitration_decision": decision})
                elif workflow.dispute_stage == DisputeStage.LOST:
                    call("accept_dispute", {"case_id": case_id})
                elif case.hidden.true_fraud_risk > 0.7:
                    call("reject", {"case_id": case_id, "reason_code": "suspicious_pattern"})
                else:
                    call("approve", {"case_id": case_id})

        elif case.case_type.value == "invoice":
            workflow = case.workflow
            if not isinstance(workflow, InvoiceWorkflowState):
                continue
            call("query_policy", {"policy_id": "invoice_policy"})
            if case.hidden.true_is_duplicate:
                call("reject", {"case_id": case_id, "reason_code": "duplicate_match"})
            else:
                call("approve", {"case_id": case_id})

        elif case.case_type.value == "kyc":
            workflow = case.workflow
            if not isinstance(workflow, KYCWorkflowState):
                continue
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
                call("reject", {"case_id": case_id, "reason_code": "suspicious_pattern"})
            else:
                if not case.hidden.true_doc_valid:
                    call("review_kyc", {"case_id": case_id, "verification_decision": "reject"})
                    call("reject", {"case_id": case_id, "reason_code": "invalid_document"})
                else:
                    call("review_kyc", {"case_id": case_id, "verification_decision": "approve"})
                    call("approve", {"case_id": case_id})

        # Post-decision: notification + QA + close
        state = env._env._state
        if state and case_id in state.cases:
            c = state.cases[case_id]
            if c.requires_customer_notification and not c.customer_notified and c.resolution.value != "pending":
                template = "refund_approved" if c.resolution.value == "approved" and c.case_type.value == "refund" else "case_closed"
                call("send_message", {"case_id": case_id, "template_id": template})

            c = env._env._state.cases[case_id]
            if c.qa_required and getattr(c.qa_status, "value", c.qa_status) not in ("passed",):
                call("send_to_qa", {"case_id": case_id})
                call("approve_qa", {"case_id": case_id})

            c = env._env._state.cases[case_id]
            if c.resolution.value != "pending":
                call("close_case", {"case_id": case_id})

    if not env.done:
        return None

    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--output", default="artifacts/triage-sft-warmstart.jsonl")
    args = parser.parse_args()

    rng = random.Random(42)
    seeds = [rng.randint(0, 1_000_000) for _ in range(args.num_seeds)]

    successes = 0
    total_tool_calls = 0
    with open(args.output, "w") as f:
        for seed in seeds:
            try:
                messages = _capture_oracle_trajectory(seed)
            except Exception as e:
                print(f"  seed={seed}: error={e}")
                continue
            if messages is None:
                print(f"  seed={seed}: failed (incomplete)")
                continue
            tool_msgs = [m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")]
            total_tool_calls += len(tool_msgs)
            f.write(json.dumps({"messages": messages}) + "\n")
            successes += 1

    avg_calls = total_tool_calls / max(1, successes)
    print(f"Generated {successes}/{len(seeds)} triage trajectories → {args.output}")
    print(f"Average tool calls per trajectory: {avg_calls:.1f}")


if __name__ == "__main__":
    main()
