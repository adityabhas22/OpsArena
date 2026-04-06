from __future__ import annotations

import inspect
import random
import types
from collections.abc import Iterable
from typing import Any, Literal

from opsarena.domain.workflows.refund import PrearbitrationDecision
from opsarena.enums import (
    DecisionCode,
    Priority,
    ReasonCode,
    RecordType,
    SortField,
    TargetQueue,
)
from opsarena.models import OpsArenaObservation, RawOpsAction
from server.environment import OpsArenaEnvironment

REFUND_GRPO_SYSTEM_PROMPT = (
    "You are an ecommerce operations analyst resolving refund and dispute cases. "
    "You MUST call tools repeatedly until the case is closed. Do NOT stop after one tool call.\n\n"
    "WORKFLOW (call each tool in a separate turn):\n"
    "1. open_case(case_id=...) — use the case_id from the queue\n"
    "2. query_policy(policy_id=\"refund_policy\")\n"
    "3. Make the decision: approve, reject, accept_dispute, or challenge_dispute\n"
    "4. send_message if obligations show send_message\n"
    "5. send_to_qa then approve_qa if obligations show qa\n"
    "6. close_case(case_id=...) — ALWAYS close the case at the end\n\n"
    "RULES:\n"
    "- Use exact IDs from the observation. Never invent IDs.\n"
    "- Keep calling tools until done=true. Do not stop early.\n"
    "- Think briefly about which tool to call next, then call it. Keep thinking under 50 words."
)

# ---------------------------------------------------------------------------
# REFUND_TOOLS — callable stubs used by TRL's GRPOTrainer(tools=...).
#
# TRL reads tool.__name__ and inspect.signature(tool) to auto-generate JSON
# schemas for the tokenizer (so the model sees tool definitions in its
# context).  We derive these stubs directly from RefundExceptionToolEnv's
# method signatures so they stay in sync automatically.
#
# The actual execution happens via environment.method_name(**args); the stubs
# are schema-only and never called directly.
# ---------------------------------------------------------------------------

_TOOL_METHOD_NAMES = [
    "list_queue", "open_case", "view_record", "query_policy",
    "approve", "reject", "escalate", "accept_dispute",
    "challenge_dispute", "submit_dispute_evidence",
    "refund_pre_dispute_alert", "resolve_prearbitration",
    "send_message", "send_to_qa", "approve_qa",
    "close_case", "advance_clock",
]


def _make_tool_stub(method_name: str) -> types.FunctionType:
    """Return a callable with the env method's name, docstring, and signature (sans self)."""
    method = getattr(RefundExceptionToolEnv, method_name)
    sig = inspect.signature(method)
    params_without_self = [p for n, p in sig.parameters.items() if n != "self"]
    new_sig = sig.replace(parameters=params_without_self, return_annotation=str)

    def stub(**kwargs: Any) -> str: ...  # noqa: E704

    stub.__name__ = method_name
    stub.__qualname__ = method_name
    stub.__doc__ = method.__doc__ or method_name
    stub.__signature__ = new_sig  # type: ignore[attr-defined]
    return stub  # type: ignore[return-value]


# Populated after RefundExceptionToolEnv is defined below.
REFUND_TOOLS: list[types.FunctionType] = []

# Keep the dict schemas around for reference / non-TRL uses.
REFUND_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_queue",
            "description": "List active refund cases in the queue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 10},
                    "sort_by": {"type": "string", "enum": ["priority", "sla_remaining", "created_at", "amount"], "default": "priority"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_case",
            "description": "Open a refund case and view its details.",
            "parameters": {
                "type": "object",
                "properties": {"case_id": {"type": "string"}},
                "required": ["case_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_record",
            "description": "Inspect a linked record for the active case (order, customer, payment, shipping, dispute, policy).",
            "parameters": {
                "type": "object",
                "properties": {
                    "record_type": {"type": "string", "enum": ["order", "customer", "payment", "shipping", "dispute", "policy"]},
                    "record_id": {"type": "string"},
                },
                "required": ["record_type", "record_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_policy",
            "description": "Query refund policy text to determine the correct action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "policy_id": {"type": "string", "default": "refund_policy"},
                    "clause_id": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "approve",
            "description": "Approve a refund case.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"},
                    "decision_code": {"type": "string", "enum": ["standard_approval", "exception_approval", "partial_approval"], "default": "standard_approval"},
                    "approved_amount": {"type": "number"},
                    "notes": {"type": "string"},
                },
                "required": ["case_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reject",
            "description": "Reject a refund case.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"},
                    "reason_code": {"type": "string", "enum": ["suspicious_pattern", "missing_documentation", "policy_ambiguity", "customer_request", "duplicate_match", "invalid_document"]},
                    "notes": {"type": "string"},
                },
                "required": ["case_id", "reason_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate",
            "description": "Escalate a case to manager_review, fraud_team, or senior_ops.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"},
                    "target_queue": {"type": "string", "enum": ["manager_review", "fraud_team", "senior_ops"]},
                    "reason_code": {"type": "string", "enum": ["threshold_exceeded", "suspicious_pattern", "policy_ambiguity", "sla_protection"]},
                    "priority_override": {"type": "integer", "enum": [1, 2, 3, 4]},
                },
                "required": ["case_id", "target_queue", "reason_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "accept_dispute",
            "description": "Accept liability on a chargeback dispute instead of contesting it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": ["case_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "challenge_dispute",
            "description": "Contest a chargeback dispute after reviewing evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": ["case_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_dispute_evidence",
            "description": "Submit evidence artifacts for an open dispute.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"},
                    "evidence_fields": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "string"},
                },
                "required": ["case_id", "evidence_fields"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "refund_pre_dispute_alert",
            "description": "Refund a pre-dispute alert (inquiry or RDR) proactively.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"},
                    "approved_amount": {"type": "number"},
                    "notes": {"type": "string"},
                },
                "required": ["case_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_prearbitration",
            "description": "Resolve a pre-arbitration stage by accepting or contesting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"},
                    "prearbitration_decision": {"type": "string", "enum": ["accept", "contest"]},
                    "notes": {"type": "string"},
                },
                "required": ["case_id", "prearbitration_decision"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send required customer communication (refund_approved or case_closed template).",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"},
                    "template_id": {"type": "string", "enum": ["refund_approved", "case_closed"]},
                    "resolution": {"type": "string"},
                },
                "required": ["case_id", "template_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_to_qa",
            "description": "Route a resolved case to QA review when required.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"},
                    "assignee_type": {"type": "string", "default": "qa_reviewer"},
                    "notes": {"type": "string"},
                },
                "required": ["case_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "approve_qa",
            "description": "Approve a case that is waiting in QA.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"},
                    "assignee_type": {"type": "string", "default": "qa_reviewer"},
                    "notes": {"type": "string"},
                },
                "required": ["case_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "close_case",
            "description": "Close a fully resolved refund case to finish the episode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"},
                    "resolution_code": {"type": "string", "default": "resolved"},
                },
                "required": ["case_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "advance_clock",
            "description": "Advance simulated time (1–480 minutes) to receive pending events or dispute responses.",
            "parameters": {
                "type": "object",
                "properties": {"minutes": {"type": "integer", "minimum": 1, "maximum": 480}},
                "required": ["minutes"],
            },
        },
    },
]


def build_refund_grpo_prompt_dataset(num_examples: int = 256) -> list[dict[str, Any]]:
    """Builds a small conversational prompt dataset for refund-only GRPO runs."""
    prompts: list[dict[str, Any]] = []
    for idx in range(num_examples):
        prompts.append(
            {
                "prompt": [
                    {"role": "system", "content": REFUND_GRPO_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Resolve the refund case by calling tools step by step: "
                            "open the case, query policy, make a decision, send notifications "
                            "if required, complete QA if required, then close the case. "
                            "Do not stop until the case is closed."
                        ),
                    },
                ]
            }
        )
    return prompts


def _refund_milestone_score(env: "RefundExceptionToolEnv") -> tuple[float, dict[str, bool]]:
    """Compute grader-aligned milestone shaping from the environment audit trail.

    Each milestone corresponds to a process check the grader actually scores.
    This ensures the shaping gradient points toward the terminal benchmark,
    not toward busywork.

    Returns (score, milestone_dict) where score is in [0.0, 0.25].
    """
    state = env._env._state
    if state is None:
        return 0.0, {}

    case_audits: dict[str, set[str]] = {}
    for entry in state.audit_log:
        if entry.case_id:
            case_audits.setdefault(entry.case_id, set()).add(entry.action_type)

    milestones: dict[str, bool] = {}
    for case in state.cases.values():
        cid = case.case_id
        events = case_audits.get(cid, set())

        # Process milestones (from grader trajectory checks)
        milestones[f"{cid}/opened"] = "open_case" in events
        milestones[f"{cid}/policy_queried"] = "query_policy" in events
        milestones[f"{cid}/decision_made"] = case.resolution.value != "pending"
        milestones[f"{cid}/notified"] = (
            not case.requires_customer_notification
            or case.customer_notified
        )
        milestones[f"{cid}/qa_done"] = (
            not case.qa_required
            or "approve_qa" in events
        )
        milestones[f"{cid}/closed"] = case.status == "closed"

    if not milestones:
        return 0.0, milestones

    hit = sum(1 for v in milestones.values() if v)
    total = len(milestones)
    # Scale to [0, 0.25] — each milestone contributes proportionally
    return min(0.25, 0.25 * hit / max(1, total)), milestones


def refund_terminal_benchmark_reward(
    prompts: list[Any],
    completions: list[Any],
    environments: list["RefundExceptionToolEnv"],
    log_metric: Any | None = None,
    **_: Any,
) -> list[float]:
    """Scores refund trajectories with grader-aligned milestone shaping + terminal bonus.

    Reward is designed so that:
    - Early in training: milestone shaping provides gradient signal even when
      the model cannot complete episodes (open_case, query_policy, etc.)
    - Milestones are exactly the checks the grader scores, so shaping points
      in the same direction as the terminal benchmark
    - Terminal bonus dominates once the model can complete episodes
    - Step cost discourages looping after a reasonable number of tool calls

    Reward budget:
      - milestone shaping:  0.0 – 0.25  (grader-aligned process milestones)
      - terminal bonus:     0.0 – 0.70  (benchmark_score * 0.70 if done)
      - step cost:          0.0 – 0.05  (0.005 per step beyond 12)
      - invalid penalty:    0.0 – 0.10
      Total range:          0.0 – 1.0
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

        # Grader-aligned milestone shaping
        milestone_score, milestones = _refund_milestone_score(env)
        total_milestones_hit += sum(1 for v in milestones.values() if v)
        total_milestones += len(milestones)

        # Terminal bonus — the dominant signal once episodes complete
        terminal = env.benchmark_score * 0.70 if done else 0.0

        # Step cost: discourage looping beyond a reasonable horizon
        steps = env.tool_call_count
        step_cost = max(0, steps - 12) * 0.005
        step_cost = min(step_cost, 0.05)

        # Invalid action penalty
        invalid_penalty = min(env.invalid_action_count, 5) * 0.02

        reward = milestone_score + terminal - step_cost - invalid_penalty
        rewards.append(max(0.0, min(1.0, reward)))

    if log_metric is not None and environments:
        count = len(environments)
        mean_benchmark = sum(env.benchmark_score for env in environments) / count
        mean_reward = sum(rewards) / count
        milestone_rate = total_milestones_hit / max(1, total_milestones)
        log_metric("env/refund_done_rate", done_count / count)
        log_metric("env/refund_invalid_actions_mean", invalid_count / count)
        log_metric("env/refund_tool_calls_mean", total_tool_calls / count)
        log_metric("env/refund_benchmark_score_mean", mean_benchmark)
        log_metric("env/refund_reward_mean", mean_reward)
        log_metric("env/refund_milestone_rate", milestone_rate)
        # Backward-compatible alias for older dashboards.
        log_metric("env/refund_terminal_score_mean", mean_reward)

    return rewards


class RefundExceptionToolEnv:
    """Narrow refund-only tool environment for GRPO/OpenEnv experiments."""

    def __init__(
        self,
        *,
        task_id: str = "refund_exception",
        seed_sequence: Iterable[int] | None = None,
        random_seed: int = 17,
    ) -> None:
        self._env = OpsArenaEnvironment()
        self._task_id = task_id
        self._rng = random.Random(random_seed)
        self._seed_sequence = iter(seed_sequence) if seed_sequence is not None else None
        self._last_observation: OpsArenaObservation | None = None
        self._last_seed: int | None = None

    @property
    def benchmark_score(self) -> float:
        return self._env.state.benchmark_score

    @property
    def done(self) -> bool:
        return bool(self._last_observation.done) if self._last_observation is not None else False

    @property
    def invalid_action_count(self) -> int:
        if self._env._state is None:
            return 0
        return self._env._state.metrics.invalid_actions

    @property
    def legacy_objective_score(self) -> float:
        if self._env._state is None:
            return 0.0
        return float(self._env._state.metadata.get("legacy_objective_score", 0.0))

    @property
    def tool_call_count(self) -> int:
        if self._env._state is None:
            return 0
        return self._env._state.metrics.tool_calls

    @property
    def cases_resolved(self) -> int:
        if self._env._state is None:
            return 0
        return self._env._state.metrics.cases_resolved

    def reset(self, **_: Any) -> str:
        """Reset the refund episode and return the initial text observation.

        Returns:
            A compact text observation appended to the last user message by TRL.
        """

        seed = next(self._seed_sequence) if self._seed_sequence is not None else self._rng.randint(0, 1_000_000)
        self._last_seed = seed
        self._last_observation = self._env.reset(task_id=self._task_id, seed=seed)
        return self._render_observation(
            self._last_observation,
            prefix=f"refund_exception episode started with seed {seed}",
        )

    def list_queue(
        self,
        limit: int = 10,
        sort_by: Literal["priority", "sla_remaining", "created_at", "amount"] = "priority",
    ) -> str:
        """List active refund cases in the queue.

        Args:
            limit: Maximum number of queue items to display.
            sort_by: Queue ordering field.

        Returns:
            A text summary of the queue and currently available actions.
        """

        return self._step({"action_type": "list_queue", "limit": limit, "sort_by": sort_by})

    def open_case(self, case_id: str) -> str:
        """Open one refund case and view its public details.

        Args:
            case_id: Case identifier from the queue.

        Returns:
            The case view after opening it.
        """

        return self._step({"action_type": "open_case", "case_id": case_id})

    def view_record(
        self,
        record_type: Literal["order", "customer", "payment", "shipping", "dispute", "policy"],
        record_id: str,
    ) -> str:
        """Inspect a linked record for the active refund case.

        Args:
            record_type: Record category to inspect.
            record_id: Record identifier from the linked records list.

        Returns:
            The updated observation plus the rendered record contents.
        """

        return self._step({"action_type": "view_record", "record_type": record_type, "record_id": record_id})

    def query_policy(self, policy_id: str = "refund_policy", clause_id: str | None = None) -> str:
        """Query refund policy text and matched actions.

        Args:
            policy_id: Policy identifier. Defaults to the refund policy.
            clause_id: Optional clause within the policy.

        Returns:
            The updated observation plus policy retrieval details.
        """

        payload: dict[str, Any] = {"action_type": "query_policy", "policy_id": policy_id}
        if clause_id:
            payload["clause_id"] = clause_id
        return self._step(payload)

    def approve(
        self,
        case_id: str,
        decision_code: Literal["standard_approval", "exception_approval", "partial_approval"] = "standard_approval",
        approved_amount: float | None = None,
        notes: str | None = None,
    ) -> str:
        """Approve a refund case.

        Args:
            case_id: Case identifier.
            decision_code: Approval type.
            approved_amount: Required only for partial approvals.
            notes: Optional analyst notes.

        Returns:
            The updated case state after the approval attempt.
        """

        payload: dict[str, Any] = {
            "action_type": "approve",
            "case_id": case_id,
            "decision_code": decision_code,
        }
        if approved_amount is not None:
            payload["approved_amount"] = approved_amount
        if notes:
            payload["notes"] = notes
        return self._step(payload)

    def reject(
        self,
        case_id: str,
        reason_code: Literal[
            "suspicious_pattern",
            "missing_documentation",
            "policy_ambiguity",
            "customer_request",
            "duplicate_match",
            "invalid_document",
        ],
        notes: str | None = None,
    ) -> str:
        """Reject a refund case.

        Args:
            case_id: Case identifier.
            reason_code: Primary rejection reason.
            notes: Optional analyst notes.

        Returns:
            The updated case state after the rejection attempt.
        """

        payload: dict[str, Any] = {"action_type": "reject", "case_id": case_id, "reason_code": reason_code}
        if notes:
            payload["notes"] = notes
        return self._step(payload)

    def escalate(
        self,
        case_id: str,
        target_queue: Literal["manager_review", "fraud_team", "senior_ops"],
        reason_code: Literal[
            "threshold_exceeded",
            "suspicious_pattern",
            "policy_ambiguity",
            "sla_protection",
        ],
        priority_override: Literal[1, 2, 3, 4] | None = None,
    ) -> str:
        """Escalate a refund case when frontline resolution is insufficient.

        Args:
            case_id: Case identifier.
            target_queue: Escalation destination.
            reason_code: Why escalation is needed.
            priority_override: Optional priority override, where 1 is critical.

        Returns:
            The updated case state after escalation.
        """

        payload: dict[str, Any] = {
            "action_type": "escalate",
            "case_id": case_id,
            "target_queue": target_queue,
            "reason_code": reason_code,
        }
        if priority_override is not None:
            payload["priority_override"] = priority_override
        return self._step(payload)

    def request_info(self, case_id: str, field_name: str, template_id: str | None = None) -> str:
        """Request missing information from the customer or merchant.

        Args:
            case_id: Case identifier.
            field_name: Missing field to request.
            template_id: Optional outbound template.

        Returns:
            The updated case state after requesting information.
        """

        payload: dict[str, Any] = {"action_type": "request_info", "case_id": case_id, "field_name": field_name}
        if template_id:
            payload["template_id"] = template_id
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

    def submit_dispute_evidence(
        self,
        case_id: str,
        evidence_fields: list[str],
        notes: str | None = None,
    ) -> str:
        """Submit supporting evidence for an open dispute.

        Args:
            case_id: Case identifier.
            evidence_fields: Evidence artifacts such as `tracking_number` or `delivery_confirmation`.
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

    def send_message(
        self,
        case_id: str,
        template_id: Literal["refund_approved", "case_closed"],
        resolution: str | None = None,
    ) -> str:
        """Send required customer communication for a resolved refund case.

        Args:
            case_id: Case identifier.
            template_id: Message template.
            resolution: Optional explicit resolution label for `case_closed`.

        Returns:
            The updated case state after sending the message.
        """

        case = self._get_case(case_id)
        order_id = next((record.record_id for record in case.linked_records if record.record_type == RecordType.ORDER), None)
        slots: dict[str, str] = {}
        if template_id == "refund_approved":
            slots["amount"] = str(case.amount)
            if order_id:
                slots["order_id"] = order_id
        else:
            slots["case_id"] = case_id
            slots["resolution"] = resolution or case.resolution.value
        return self._step(
            {
                "action_type": "send_message",
                "case_id": case_id,
                "template_id": template_id,
                "slots": slots,
            }
        )

    def send_to_qa(self, case_id: str, assignee_type: str = "qa_reviewer", notes: str | None = None) -> str:
        """Send a resolved refund case to QA when required.

        Args:
            case_id: Case identifier.
            assignee_type: QA assignee label.
            notes: Optional analyst notes.

        Returns:
            The updated case state after routing to QA.
        """

        payload: dict[str, Any] = {"action_type": "send_to_qa", "case_id": case_id, "assignee_type": assignee_type}
        if notes:
            payload["notes"] = notes
        return self._step(payload)

    def approve_qa(self, case_id: str, assignee_type: str = "qa_reviewer", notes: str | None = None) -> str:
        """Approve a refund case that is waiting in QA.

        Args:
            case_id: Case identifier.
            assignee_type: QA reviewer label.
            notes: Optional QA notes.

        Returns:
            The updated case state after QA approval.
        """

        payload: dict[str, Any] = {"action_type": "approve_qa", "case_id": case_id, "assignee_type": assignee_type}
        if notes:
            payload["notes"] = notes
        return self._step(payload)

    def close_case(self, case_id: str, resolution_code: str = "grpo_complete") -> str:
        """Close a resolved refund case and finish the episode when appropriate.

        Args:
            case_id: Case identifier.
            resolution_code: Free-form terminal code recorded in audit metadata.

        Returns:
            The terminal or updated case observation after close.
        """

        return self._step({"action_type": "close_case", "case_id": case_id, "resolution_code": resolution_code})

    def advance_clock(self, minutes: int) -> str:
        """Advance simulated time to receive pending info or dispute events.

        Args:
            minutes: Minutes to advance, from 1 to 480.

        Returns:
            The updated environment state after time advances.
        """

        return self._step({"action_type": "advance_clock", "minutes": minutes})

    def _get_case(self, case_id: str):
        if self._env._state is None or case_id not in self._env._state.cases:
            raise ValueError(f"unknown_case_id: {case_id}")
        return self._env._state.cases[case_id]

    def _step(self, payload: dict[str, Any]) -> str:
        self._last_observation = self._env.step(RawOpsAction(**payload))
        return self._render_observation(self._last_observation)

    def _render_observation(self, obs: OpsArenaObservation, prefix: str | None = None) -> str:
        """Render a compact, token-efficient observation for the model.

        Design: show only what the model needs to decide its next action.
        Omit benchmark_score (that's for the reward function, not the model).
        """
        lines: list[str] = []
        if prefix:
            lines.append(prefix)
        # Header: clock + done + available actions
        lines.append(f"clock={obs.clock} done={str(obs.done).lower()}")
        if obs.error:
            lines.append(f"ERROR: {obs.error}")
        elif obs.system_message:
            lines.append(f">> {obs.system_message}")
        if obs.available_actions:
            lines.append(f"actions: {', '.join(obs.available_actions)}")
        # Queue view (compact)
        if obs.queue_view:
            lines.append("queue:")
            for item in obs.queue_view:
                lines.append(
                    f"  {item.case_id} type={item.case_type} p={item.priority} "
                    f"sla={item.sla_remaining_minutes}m ${item.amount} [{item.status}]"
                )
        # Case detail (compact with obligations)
        if obs.case_detail:
            detail = obs.case_detail
            state_case = self._get_case(detail.case_id)
            lines.append(
                f"case: {detail.case_id} status={detail.status} p={detail.priority} "
                f"${detail.amount}"
            )
            # Obligations: what must happen before close
            obligations: list[str] = []
            if state_case.requires_customer_notification and not state_case.customer_notified:
                obligations.append("send_message")
            if state_case.qa_required and getattr(state_case.qa_status, "value", state_case.qa_status) != "passed":
                obligations.append(f"qa({getattr(state_case.qa_status, 'value', state_case.qa_status)})")
            if state_case.resolution.value == "pending":
                obligations.append("decide(approve/reject/escalate)")
            if obligations:
                lines.append(f"obligations: {', '.join(obligations)}")
            if detail.visible_flags:
                lines.append(f"flags: {', '.join(detail.visible_flags)}")
            if detail.linked_records:
                linked = ", ".join(f"{record.record_type}:{record.record_id}" for record in detail.linked_records)
                lines.append(f"records: {linked}")
            # Pending events the agent should advance_clock for
            if self._env._state is not None:
                pending_events = [
                    f"{event.event_type}@{event.at_time}"
                    for event in self._env._state.scheduled_events
                    if event.case_id == detail.case_id
                ]
                if pending_events:
                    lines.append(f"pending_events: {', '.join(pending_events)}")
            # Compact workflow metadata — only decision-relevant fields
            if detail.workflow_metadata:
                wm = detail.workflow_metadata
                parts: list[str] = []
                for key in (
                    "dispute_stage", "pre_dispute_type", "merchant_risk_level",
                    "monitoring_program_status", "payout_frozen", "reserve_percent",
                    "payout_delay_days", "dispute_evidence_fields",
                ):
                    if key in wm:
                        parts.append(f"{key}={wm[key]}")
                if parts:
                    lines.append(f"workflow: {', '.join(parts)}")
        # Policy result (compact)
        if obs.policy_result:
            policy = obs.policy_result
            lines.append(f"policy: {policy.policy_id} — {policy.title}")
            if policy.matched_actions:
                lines.append(f"policy_actions: {', '.join(policy.matched_actions)}")
        if obs.record_view:
            lines.append(f"record: {obs.record_view}")
        return "\n".join(lines)


# Populate REFUND_TOOLS now that RefundExceptionToolEnv is fully defined.
REFUND_TOOLS.extend(_make_tool_stub(name) for name in _TOOL_METHOD_NAMES)
