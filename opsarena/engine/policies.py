from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from opsarena.documents import PolicyClause, PolicyDocument

ROOT = Path(__file__).resolve().parents[2]


def _compare(value: Any, operator: str, expected: Any) -> bool:
    if operator == "gt":
        return value > expected
    if operator == "gte":
        return value >= expected
    if operator == "lt":
        return value < expected
    if operator == "lte":
        return value <= expected
    if operator == "eq":
        return value == expected
    if operator == "in":
        return value in expected
    if operator == "between":
        low, high = expected
        return low <= value <= high
    return False


def load_policy(policy_name: str) -> PolicyDocument:
    path = ROOT / "data" / "policies" / f"{policy_name}.yaml"
    data = yaml.safe_load(path.read_text())
    return PolicyDocument.model_validate(data)


def clause_matches(clause: PolicyClause, context: dict[str, Any]) -> bool:
    return all(
        _compare(context.get(cond.field), cond.operator, cond.value)
        for cond in clause.conditions
    )


def query_policy(policy: PolicyDocument, context: dict[str, Any], clause_id: str | None = None) -> list[PolicyClause]:
    if clause_id is not None:
        return [clause for clause in policy.clauses if clause.clause_id == clause_id]
    return [clause for clause in policy.clauses if clause_matches(clause, context)]
