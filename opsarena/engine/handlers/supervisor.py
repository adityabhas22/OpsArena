from __future__ import annotations

from opsarena.domain.core import RouteHistoryEntry
from opsarena.engine.handlers.common import require_case, sort_key
from opsarena.engine.handlers.result import TransitionResult
from opsarena.engine.state import WorldState
from opsarena.enums import SortField
from opsarena.models import BulkAssignAction, BulkRouteAction, RebalanceQueueAction


def _open_owner_load(state: WorldState, owner: str) -> int:
    return sum(1 for case in state.cases.values() if case.status != "closed" and case.current_owner == owner)


def _assign_case(case, owner: str) -> None:
    case.current_owner = owner
    if case.status in {"open", "reopened", "routed"}:
        case.status = "assigned"


def handle_bulk_assign(state: WorldState, action: BulkAssignAction):
    capacity = state.metadata.get("agent_capacity", state.metadata.get("claim_capacity", 2))
    unique_case_ids = list(dict.fromkeys(action.case_ids))
    cases = [require_case(state, case_id) for case_id in unique_case_ids]
    if any(case.status == "closed" for case in cases):
        raise ValueError("closed_case_in_bulk_assign")
    remaining_capacity = max(0, capacity - _open_owner_load(state, action.assignee_type))
    if len(cases) > remaining_capacity:
        raise ValueError("assignee_capacity_exceeded")
    for case in cases:
        _assign_case(case, action.assignee_type)
    return TransitionResult(True, f"Bulk assigned {len(cases)} cases to {action.assignee_type}"), None


def handle_bulk_route(state: WorldState, action: BulkRouteAction):
    unique_case_ids = list(dict.fromkeys(action.case_ids))
    cases = [require_case(state, case_id) for case_id in unique_case_ids]
    invalid = [case.case_id for case in cases if action.target_queue not in case.allowed_escalation_queues]
    if invalid:
        raise ValueError("invalid_route_target")
    for case in cases:
        case.current_owner = action.assignee_type or action.target_queue.value
        case.status = "routed"
        case.active_queue = action.target_queue.value
        case.route_reason = action.reason_code.value
        case.route_history.append(
            RouteHistoryEntry(
                at_time=state.current_time,
                queue=action.target_queue.value,
                owner=case.current_owner,
                reason=action.reason_code.value,
            )
        )
    return TransitionResult(True, f"Bulk routed {len(cases)} cases to {action.target_queue.value}"), None


def handle_rebalance_queue(state: WorldState, action: RebalanceQueueAction):
    strategy = action.rebalance_strategy
    if strategy == "oldest":
        candidates = sorted(
            [
                case
                for case in state.cases.values()
                if case.status != "closed" and case.claimed_by is None and case.current_owner in {"queue", "ops_agent"}
            ],
            key=lambda case: (case.created_at, case.priority),
        )
    elif strategy == "amount":
        candidates = sorted(
            [
                case
                for case in state.cases.values()
                if case.status != "closed" and case.claimed_by is None and case.current_owner in {"queue", "ops_agent"}
            ],
            key=lambda case: sort_key(case, SortField.AMOUNT),
        )
    else:
        candidates = sorted(
            [
                case
                for case in state.cases.values()
                if case.status != "closed" and case.claimed_by is None and case.current_owner in {"queue", "ops_agent"}
            ],
            key=lambda case: sort_key(case, SortField.SLA_REMAINING),
        )

    owner_cap = state.metadata.get("agent_capacity", state.metadata.get("claim_capacity", 2))
    owner_loads = {owner: _open_owner_load(state, owner) for owner in action.assignee_pool}
    assigned = 0
    for case in candidates:
        available_owners = [owner for owner in action.assignee_pool if owner_loads[owner] < owner_cap]
        if not available_owners or assigned >= action.max_cases:
            break
        owner = min(available_owners, key=lambda candidate: owner_loads[candidate])
        _assign_case(case, owner)
        owner_loads[owner] += 1
        assigned += 1

    if assigned == 0:
        raise ValueError("no_rebalance_capacity")
    return TransitionResult(True, f"Rebalanced {assigned} cases across {len(action.assignee_pool)} owners"), None
