from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any, Literal

from opsarena.enums import RecordType, RecordType as _RT
from opsarena.models import OpsArenaObservation, RawOpsAction
from server.environment import OpsArenaEnvironment


class BaseToolEnv:
    """Shared infrastructure for all GRPO tool environments.

    Subclasses add domain-specific tool methods (invoice, KYC, refund, etc.)
    while this base provides queue navigation, observation rendering, and the
    common lifecycle actions that every task needs.
    """

    def __init__(
        self,
        *,
        task_id: str,
        seed_sequence: Iterable[int] | None = None,
        random_seed: int = 17,
    ) -> None:
        self._env = OpsArenaEnvironment()
        self._task_id = task_id
        self._rng = random.Random(random_seed)
        self._seed_sequence = iter(seed_sequence) if seed_sequence is not None else None
        self._last_observation: OpsArenaObservation | None = None
        self._last_seed: int | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, **_: Any) -> str:
        """Reset the episode and return the initial text observation.

        Returns:
            A compact text observation appended to the last user message by TRL.
        """
        seed = (
            next(self._seed_sequence)
            if self._seed_sequence is not None
            else self._rng.randint(0, 1_000_000)
        )
        self._last_seed = seed
        self._last_observation = self._env.reset(task_id=self._task_id, seed=seed)
        return self._render_observation(
            self._last_observation,
            prefix=f"{self._task_id} episode started with seed {seed}",
        )

    # ------------------------------------------------------------------
    # Shared tool methods (13 actions common to all tasks)
    # ------------------------------------------------------------------

    def list_queue(
        self,
        limit: int = 10,
        sort_by: Literal["priority", "sla_remaining", "created_at", "amount"] = "priority",
    ) -> str:
        """List active cases in the queue.

        Args:
            limit: Maximum number of queue items to display.
            sort_by: Queue ordering field.

        Returns:
            A text summary of the queue and currently available actions.
        """
        return self._step({"action_type": "list_queue", "limit": limit, "sort_by": sort_by})

    def open_case(self, case_id: str) -> str:
        """Open a case and view its public details.

        Args:
            case_id: Case identifier from the queue.

        Returns:
            The case view after opening it.
        """
        return self._step({"action_type": "open_case", "case_id": case_id})

    def view_record(
        self,
        record_type: Literal[
            "order", "customer", "payment", "shipping", "dispute", "policy",
            "invoice", "credit_memo", "receipt", "vendor", "kyc_document",
            "purchase_order",
        ],
        record_id: str,
    ) -> str:
        """Inspect a linked record for the active case.

        Args:
            record_type: Record category to inspect.
            record_id: Record identifier from the linked records list.

        Returns:
            The updated observation plus the rendered record contents.
        """
        return self._step({"action_type": "view_record", "record_type": record_type, "record_id": record_id})

    def query_policy(self, policy_id: str = "refund_policy", clause_id: str | None = None) -> str:
        """Query policy text and matched actions.

        Args:
            policy_id: Policy identifier.
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
        """Approve a case.

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
        """Reject a case.

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
        """Escalate a case when frontline resolution is insufficient.

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
        """Request missing information from the customer or vendor.

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

    def close_case(self, case_id: str, resolution_code: str = "grpo_complete") -> str:
        """Close a resolved case and finish the episode when appropriate.

        Args:
            case_id: Case identifier.
            resolution_code: Free-form terminal code recorded in audit metadata.

        Returns:
            The terminal or updated case observation after close.
        """
        return self._step({"action_type": "close_case", "case_id": case_id, "resolution_code": resolution_code})

    def send_message(
        self,
        case_id: str,
        template_id: str,
        resolution: str | None = None,
    ) -> str:
        """Send required customer or vendor communication for a case.

        Args:
            case_id: Case identifier.
            template_id: Message template identifier.
            resolution: Optional explicit resolution label.

        Returns:
            The updated case state after sending the message.
        """
        case = self._get_case(case_id)
        order_id = next(
            (record.record_id for record in case.linked_records if record.record_type == RecordType.ORDER),
            None,
        )
        slots: dict[str, str] = {}
        if template_id == "refund_approved":
            slots["amount"] = str(case.amount)
            if order_id:
                slots["order_id"] = order_id
        else:
            slots["case_id"] = case_id
            slots["resolution"] = resolution or getattr(case.resolution, "value", str(case.resolution))
        return self._step(
            {
                "action_type": "send_message",
                "case_id": case_id,
                "template_id": template_id,
                "slots": slots,
            }
        )

    def send_to_qa(self, case_id: str, assignee_type: str = "qa_reviewer", notes: str | None = None) -> str:
        """Send a resolved case to QA when required.

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
        """Approve a case that is waiting in QA.

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

    def advance_clock(self, minutes: int) -> str:
        """Advance simulated time to receive pending events or responses.

        Args:
            minutes: Minutes to advance, from 1 to 480.

        Returns:
            The updated environment state after time advances.
        """
        return self._step({"action_type": "advance_clock", "minutes": minutes})

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_case(self, case_id: str):
        if self._env._state is None or case_id not in self._env._state.cases:
            raise ValueError(f"unknown_case_id: {case_id}")
        return self._env._state.cases[case_id]

    def _step(self, payload: dict[str, Any]) -> str:
        self._last_observation = self._env.step(RawOpsAction(**payload))
        return self._render_observation(self._last_observation)

    def _render_observation(self, obs: OpsArenaObservation, prefix: str | None = None) -> str:
        lines: list[str] = []
        if prefix:
            lines.append(prefix)
        lines.extend(
            [
                f"clock: {obs.clock}",
                f"done: {str(obs.done).lower()}",
                f"benchmark_score: {self.benchmark_score:.3f}",
                f"legacy_objective_score: {self.legacy_objective_score:.3f}",
                f"invalid_actions: {self.invalid_action_count}",
                f"available_actions: {', '.join(obs.available_actions) if obs.available_actions else 'none'}",
            ]
        )
        if obs.system_message:
            lines.append(f"system_message: {obs.system_message}")
        if obs.error:
            lines.append(f"error: {obs.error}")
        if obs.queue_view:
            lines.append("queue:")
            for item in obs.queue_view:
                lines.append(
                    f"- {item.case_id} | type={item.case_type} | priority={item.priority} "
                    f"| sla_remaining_minutes={item.sla_remaining_minutes} | amount={item.amount} "
                    f"| status={item.status} | summary={item.summary}"
                )
        if obs.case_detail:
            detail = obs.case_detail
            state_case = self._get_case(detail.case_id)
            lines.append(
                f"case_detail: {detail.case_id} | status={detail.status} | priority={detail.priority} "
                f"| amount={detail.amount} | summary={detail.visible_summary}"
            )
            lines.append(
                "case_controls: "
                f"requires_customer_notification={state_case.requires_customer_notification} | "
                f"customer_notified={state_case.customer_notified} | "
                f"qa_required={state_case.qa_required} | "
                f"qa_status={getattr(state_case.qa_status, 'value', state_case.qa_status)}"
            )
            if detail.visible_flags:
                lines.append(f"visible_flags: {', '.join(detail.visible_flags)}")
            if detail.required_checks:
                lines.append(f"required_checks: {', '.join(detail.required_checks)}")
            if detail.checks_completed:
                lines.append(f"checks_completed: {', '.join(detail.checks_completed)}")
            if detail.linked_records:
                linked = ", ".join(f"{record.record_type}:{record.record_id}" for record in detail.linked_records)
                lines.append(f"linked_records: {linked}")
            pending_events = [
                f"{event.event_type}@{event.at_time}"
                for event in self._env._state.scheduled_events
                if event.case_id == detail.case_id
            ]
            if pending_events:
                lines.append(f"pending_case_events: {', '.join(pending_events)}")
            if detail.workflow_metadata:
                public_keys = (
                    "dispute_stage",
                    "pre_dispute_type",
                    "refund_execution_state",
                    "merchant_risk_level",
                    "monitoring_program_status",
                    "reserve_percent",
                    "payout_delay_days",
                    "payout_frozen",
                    "representment_due_at",
                    "prearbitration_due_at",
                    "approval_status",
                    "dispute_evidence_fields",
                    # Invoice keys
                    "match_status",
                    "variance_amount",
                    "payment_hold",
                    "payment_hold_reason",
                    "credit_memo_status",
                    "credit_memo_amount",
                    "vendor_response_status",
                    "po_change_status",
                    "payment_batch_status",
                    "payment_batch_id",
                    "recovery_status",
                    "duplicate_status",
                    "secondary_approval_required",
                    # KYC keys
                    "kyc_stage",
                    "kyc_complete",
                    "verification_status",
                    "sanctions_status",
                    "screening_match_confidence",
                    "ofac_report_status",
                    "report_due_at",
                    "edd_status",
                    "edd_due_at",
                    "beneficial_owner_status",
                    "correction_fields",
                    "requirements_due",
                    "payments_frozen",
                    "payment_freeze_reason",
                )
                compact_meta = {
                    key: detail.workflow_metadata[key]
                    for key in public_keys
                    if key in detail.workflow_metadata
                }
                if compact_meta:
                    lines.append(f"workflow_metadata: {compact_meta}")
        if obs.policy_result:
            policy = obs.policy_result
            lines.append(f"policy_result: {policy.policy_id} | title={policy.title}")
            lines.append(f"policy_description: {policy.description}")
            if policy.matched_actions:
                lines.append(f"policy_matched_actions: {', '.join(policy.matched_actions)}")
        if obs.record_view:
            lines.append(f"record_view: {obs.record_view}")
        return "\n".join(lines)
