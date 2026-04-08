from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_SPEC = importlib.util.spec_from_file_location(
    "opsarena_inference", Path(__file__).resolve().parents[1] / "inference.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

from opsarena.action_docs import render_action_catalog_as_json
from opsarena.models import (
    CaseDetail,
    LinkedRecordView,
    OpsArenaObservation,
    QueueItem,
    RawOpsAction,
)

_tools_for_available_actions = _MODULE._tools_for_available_actions
_fallback_action = _MODULE._fallback_action
format_start_line = _MODULE.format_start_line
format_step_line = _MODULE.format_step_line
format_end_line = _MODULE.format_end_line


def test_tools_for_available_actions_filters_catalog():
    tool_map = {t["function"]["name"]: t for t in render_action_catalog_as_json()}
    selected = _tools_for_available_actions(tool_map, ["open_case", "query_policy"])

    assert {t["function"]["name"] for t in selected} == {"open_case", "query_policy"}


def test_tools_for_available_actions_returns_all_on_empty():
    tool_map = {t["function"]["name"]: t for t in render_action_catalog_as_json()}
    selected = _tools_for_available_actions(tool_map, [])

    assert len(selected) == len(tool_map)


def test_log_lines_follow_required_shape():
    action = RawOpsAction(action_type="open_case", case_id="case_refund_1")

    start = format_start_line("refund_exception", "opsarena", "model-x")
    assert start == "[START] task=refund_exception env=opsarena model=model-x"

    step = format_step_line(1, action, 0.25, False, None)
    assert step == (
        '[STEP] step=1 action=open_case(case_id="case_refund_1") '
        "reward=0.25 done=false error=null"
    )

    end = format_end_line(True, 2, 0.875, [0.25, 0.5])
    assert end == "[END] success=true steps=2 score=0.88 rewards=0.25,0.50"


def test_log_step_with_error():
    action = RawOpsAction(action_type="approve", case_id="c1", decision_code="standard_approval")
    step = format_step_line(3, action, 0.0, False, "case not open")
    assert "error=case not open" in step
    assert "done=false" in step


def test_fallback_opens_first_queue_case():
    obs = OpsArenaObservation(
        queue_view=[
            QueueItem(
                case_id="case_a",
                case_type="refund",
                priority="P1",
                sla_remaining_minutes=30,
                summary="A",
                status="open",
            ),
        ],
        available_actions=["open_case", "list_queue"],
    )

    action = _fallback_action(obs)

    assert action.action_type == "open_case"
    assert action.case_id == "case_a"


def test_fallback_lists_queue_when_nothing_open():
    obs = OpsArenaObservation(
        available_actions=["list_queue", "advance_clock"],
    )

    action = _fallback_action(obs)

    assert action.action_type == "list_queue"


def test_fallback_advances_clock_when_case_open():
    obs = OpsArenaObservation(
        case_detail=CaseDetail(
            case_id="case_1",
            case_type="refund",
            priority="P2",
            status="pending_info",
            sla_deadline=600,
            created_at=480,
            visible_summary="waiting",
            visible_flags=[],
            linked_records=[],
            required_checks=[],
            checks_completed=[],
            communication_log=[],
            internal_notes=[],
            requested_info_fields=[],
            current_owner="ops_agent",
            case_phase="waiting_external",
            close_blockers=[],
            waiting_on=["info_response:doc"],
            next_due_minutes=10,
            recommended_action_categories=["wait"],
            workflow_metadata={},
        ),
        available_actions=["advance_clock", "list_queue"],
    )

    action = _fallback_action(obs)

    assert action.action_type == "advance_clock"
    assert action.minutes == 5
