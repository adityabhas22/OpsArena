from __future__ import annotations

from opsarena.domain.core import AuditEntry
from opsarena.engine.handlers.common import tool_time_cost, touch_case
from opsarena.engine.handlers.dispatcher import dispatch_action
from opsarena.engine.handlers.result import TransitionResult
from opsarena.engine.scheduler import process_due_events
from opsarena.engine.state import WorldState
from opsarena.enums import CaseType, TaskId
from opsarena.models import AdvanceClockAction, OpsAction
from opsarena.rewards import (
    RewardBreakdown,
    compute_queue_reward,
    compute_queue_shaping_reward,
    compute_shaping_reward,
    compute_step_reward,
)


def _append_audit(state: WorldState, case_id: str | None, action_type: str, message: str, success: bool = True) -> None:
    state.audit_log.append(
        AuditEntry(
            timestamp=state.current_time,
            case_id=case_id,
            action_type=action_type,
            message=message,
            success=success,
        )
    )


def apply_action(state: WorldState, action: OpsAction) -> TransitionResult:
    state.step_count += 1
    state.metrics.tool_calls += 1

    target_case = None
    prev_case = None
    if hasattr(action, "case_id"):
        action_case_id = getattr(action, "case_id", None)
        if action_case_id and action_case_id in state.cases:
            target_case = state.cases[action_case_id]
            prev_case = target_case.model_copy(deep=True)
            touch_case(target_case, state.current_time)
    prev_queue = state.queue_state().model_copy(deep=True)

    try:
        result, case = dispatch_action(state, action)
        if not isinstance(action, AdvanceClockAction):
            cost = tool_time_cost(action)
            state.current_time += cost
            state.metrics.simulated_minutes += cost
            process_due_events(state)

        reward_breakdown = RewardBreakdown(
            case_id=case.case_id if case else "queue",
            case_type=(case.case_type if case else CaseType.TRIAGE),
        )
        if case:
            t_total = getattr(case.workflow, "sla_total", max(1, case.sla_deadline - case.created_at))
            t_remaining = case.sla_deadline - state.current_time
            reward_breakdown = compute_step_reward(
                case=case,
                queue=state.queue_state(),
                action_type=action.action_type,
                episode_metrics=state.metrics,
                t_remaining=t_remaining,
                t_total=t_total,
                evidence_type=case.evidence_types_gathered[-1] if case.evidence_types_gathered else None,
                evidence_items_before=prev_case.evidence_items_gathered if prev_case else 0,
                evidence_items_after=case.evidence_items_gathered,
                evidence_time_cost=tool_time_cost(action),
                evidence_already_gathered=set(prev_case.evidence_types_gathered) if prev_case else None,
            )
            shaping_reward = compute_shaping_reward(
                case=case,
                queue=state.queue_state(),
                prev_case=prev_case,
                prev_queue=prev_queue,
            )
        else:
            shaping_reward = compute_queue_shaping_reward(
                queue=state.queue_state(),
                prev_queue=prev_queue,
            )

        objective_reward = reward_breakdown.objective_total
        if state.task_id == TaskId.QUEUE_TRIAGE and all(item.status == "closed" for item in state.cases.values()):
            queue_reward = compute_queue_reward(
                queue=state.queue_state(),
                initial_case_count=len(state.cases),
                initial_backlog=len(state.cases),
                resolved_cases=list(state.cases.values()),
                remaining_cases=[],
            )
            objective_reward += queue_reward["total"]
        train_reward = objective_reward + shaping_reward
        state.objective_score += objective_reward
        state.train_score += train_reward
        state.last_action_result = result.message
        _append_audit(state, case.case_id if case else None, action.action_type, result.message)
        result.objective_reward = objective_reward
        result.train_reward = train_reward
        return result
    except Exception as exc:
        state.metrics.invalid_actions += 1
        message = str(exc)
        _append_audit(state, getattr(action, "case_id", None), action.action_type, message, success=False)
        state.objective_score += -2.0
        state.train_score += -2.0
        state.last_action_result = message
        return TransitionResult(
            success=False,
            message=message.replace("_", " "),
            objective_reward=-2.0,
            train_reward=-2.0,
            error_code="invalid_action",
        )
