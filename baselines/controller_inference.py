from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional at import time for unit tests
    OpenAI = Any  # type: ignore[misc,assignment]

from opsarena.action_docs import render_action_catalog_as_json
from opsarena.client import OpsArenaEnv
from opsarena.models import MessageSummary, OpsArenaObservation, RawOpsAction


def _load_repo_env() -> None:
    env_path = Path(__file__).resolve().with_name(".env")
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        os.environ.setdefault(key, value)


_load_repo_env()

BENCHMARK = os.getenv("OPSARENA_BENCHMARK", "opsarena")
ENV_BASE_URL = os.getenv("OPSARENA_ENV_URL", "http://localhost:8000")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5-mini")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
TASK_IDS = tuple(
    task.strip()
    for task in os.getenv(
        "OPSARENA_TASKS",
        "refund_exception,invoice_plus_kyc,queue_triage,ap_payment_run",
    ).split(",")
    if task.strip()
)
SEED = int(os.getenv("OPSARENA_SEED", "7"))
MAX_STEPS = int(os.getenv("OPSARENA_MAX_STEPS", "40"))
TEMPERATURE = float(os.getenv("OPSARENA_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("OPSARENA_MAX_TOKENS", "160"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("OPSARENA_SUCCESS_THRESHOLD", "0.1"))

SYSTEM_PROMPT = (
    "You are operating an ecommerce operations environment. "
    "Pick exactly one valid tool for the current state. "
    "Use only IDs and enum values shown in the observation. "
    "Do not close a case before required messaging or QA steps are complete. "
    "Prefer policy review before approval decisions and avoid unnecessary actions."
)

VALID_DECISION_CODES = {"standard_approval", "exception_approval", "partial_approval"}
DECISION_CODE_ALIASES = {
    "approve": "standard_approval",
    "approved": "standard_approval",
    "refund_approved": "standard_approval",
    "exception": "exception_approval",
    "partial": "partial_approval",
}


def _flatten_error(error: str | None) -> str:
    if not error:
        return "null"
    return " ".join(str(error).split())


def _format_action(action: RawOpsAction) -> str:
    arguments = action.model_dump(exclude_none=True)
    arguments.pop("action_type", None)
    arguments.pop("metadata", None)
    if not arguments:
        return action.action_type
    rendered_args = ",".join(
        f"{key}={json.dumps(value, separators=(',', ':'))}"
        for key, value in sorted(arguments.items())
    )
    return f"{action.action_type}({rendered_args})"


def _action_signature(action: RawOpsAction) -> str:
    return _format_action(action)


def format_start_line(task: str, env: str, model: str) -> str:
    return f"[START] task={task} env={env} model={model}"


def format_step_line(step: int, action: RawOpsAction, reward: float, done: bool, error: str | None) -> str:
    return (
        f"[STEP] step={step} action={_format_action(action)} reward={reward:.2f} "
        f"done={str(done).lower()} error={_flatten_error(error)}"
    )


def format_end_line(success: bool, steps: int, score: float, rewards: list[float]) -> str:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    return f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}"


def _case_policy_id(case_type: str) -> str:
    return {
        "refund": "refund_policy",
        "invoice": "invoice_policy",
        "kyc": "kyc_policy",
    }.get(case_type, "refund_policy")


def _build_tool_map() -> dict[str, dict[str, Any]]:
    tools = render_action_catalog_as_json()
    return {tool["function"]["name"]: tool for tool in tools}


def _tools_for_available_actions(tool_map: dict[str, dict[str, Any]], available_actions: list[str]) -> list[dict[str, Any]]:
    selected = [tool_map[name] for name in available_actions if name in tool_map]
    return selected or list(tool_map.values())


def _queue_pick(observation: OpsArenaObservation) -> str | None:
    queue = observation.queue_view or []
    if not queue:
        return None
    next_case = sorted(queue, key=lambda item: (item.priority, item.sla_remaining_minutes, item.case_id))[0]
    return next_case.case_id


def _queue_pick_alternate(observation: OpsArenaObservation, current_case_id: str) -> str | None:
    queue = observation.queue_view or []
    if not queue:
        return None
    actionable_statuses = {"open", "reopened", "breached", "assigned", "routed", "rework"}
    candidates = [
        item for item in sorted(queue, key=lambda item: (item.priority, item.sla_remaining_minutes, item.case_id))
        if item.case_id != current_case_id and item.status in actionable_statuses
    ]
    if not candidates:
        return None
    return candidates[0].case_id


def _notification_action(detail) -> RawOpsAction:
    if detail.case_type == "refund":
        linked = {record.record_type: record.record_id for record in detail.linked_records}
        order_id = linked.get("order", detail.case_id)
        return RawOpsAction(
            action_type="send_message",
            case_id=detail.case_id,
            template_id="refund_approved",
            slots={"amount": str(detail.amount or 0), "order_id": order_id},
        )
    return RawOpsAction(
        action_type="send_message",
        case_id=detail.case_id,
        template_id="case_closed",
        slots={"case_id": detail.case_id, "resolution": "completed"},
    )


def _wait_minutes(detail, clock: int) -> int | None:
    if detail.next_due_minutes is not None:
        return detail.next_due_minutes
    meta = detail.workflow_metadata or {}
    due_fields = [
        "pre_dispute_due_at",
        "representment_due_at",
        "prearbitration_due_at",
        "report_due_at",
        "edd_due_at",
        "next_touch_at",
        "stop_payment_window_until",
        "rework_due_at",
    ]
    candidates = [meta.get(field) for field in due_fields if isinstance(meta.get(field), int) and meta.get(field) > clock]
    if not candidates:
        return None
    delta = min(candidates) - clock
    return max(1, min(delta, 480))


def _record_for_required_check(detail, available: set[str]) -> RawOpsAction | None:
    if "view_record" not in available:
        return None
    linked = {record.record_type: record.record_id for record in detail.linked_records}
    completed = set(detail.checks_completed)
    if detail.case_type == "refund":
        if "review_order" not in completed and "order" in linked:
            return RawOpsAction(action_type="view_record", record_type="order", record_id=linked["order"])
        if "review_customer" not in completed and "customer" in linked:
            return RawOpsAction(action_type="view_record", record_type="customer", record_id=linked["customer"])
    elif detail.case_type == "invoice":
        if "review_invoice" not in completed and "invoice" in linked:
            return RawOpsAction(action_type="view_record", record_type="invoice", record_id=linked["invoice"])
        if "review_po" not in completed and "purchase_order" in linked:
            return RawOpsAction(action_type="view_record", record_type="purchase_order", record_id=linked["purchase_order"])
        if "review_receipt" not in completed and "receipt" in linked:
            return RawOpsAction(action_type="view_record", record_type="receipt", record_id=linked["receipt"])
    elif detail.case_type == "kyc":
        if "review_document" not in completed and "kyc_document" in linked:
            return RawOpsAction(action_type="view_record", record_type="kyc_document", record_id=linked["kyc_document"])
    return None


def _pending_request_field(detail) -> str | None:
    pending = list(detail.workflow_metadata.get("pending_info_fields") or [])
    requested = set(detail.requested_info_fields or [])
    for field in pending:
        if field not in requested:
            return field
    return None


def _is_hard_duplicate_status(value: str | None) -> bool:
    return value in {"confirmed", "already_paid"}


def _is_kyc_approval_ready(meta: dict[str, Any]) -> bool:
    return (
        meta.get("verification_status") in {"pending_review", "verified"}
        and not meta.get("requirements_due")
        and not meta.get("pending_info_fields")
        and meta.get("sanctions_status") == "clear"
        and meta.get("edd_status") in {"not_required", "cleared"}
        and meta.get("beneficial_owner_status") in {"not_started", "verified"}
        and not meta.get("correction_fields")
        and not meta.get("payments_frozen")
    )


def _resolution_action(observation: OpsArenaObservation) -> RawOpsAction | None:
    detail = observation.case_detail
    if detail is None:
        return None
    available = set(observation.available_actions or [])
    meta = detail.workflow_metadata or {}
    case_id = detail.case_id

    if detail.case_type == "refund":
        if meta.get("monitoring_program_status") == "breached" and not meta.get("payout_frozen") and "freeze_payouts" in available:
            return RawOpsAction(action_type="freeze_payouts", case_id=case_id, reason_code="threshold_exceeded")
        if meta.get("merchant_risk_level") in {"high", "critical"} and not meta.get("reserve_percent") and "set_reserve_percent" in available:
            return RawOpsAction(
                action_type="set_reserve_percent",
                case_id=case_id,
                reserve_percent=15.0,
                release_after_minutes=25,
            )
        if meta.get("merchant_risk_level") in {"elevated", "high", "critical"} and not meta.get("payout_delay_days") and "set_payout_delay_days" in available:
            return RawOpsAction(action_type="set_payout_delay_days", case_id=case_id, payout_delay_days=7)
        if meta.get("dispute_stage") == "chargeback_open" and not meta.get("dispute_evidence_fields") and "submit_dispute_evidence" in available:
            return RawOpsAction(
                action_type="submit_dispute_evidence",
                case_id=case_id,
                evidence_fields=["customer_communication", "tracking_number", "delivery_confirmation"],
            )
        if meta.get("dispute_stage") == "inquiry" and detail.amount is not None and detail.amount <= 75 and "refund_pre_dispute_alert" in available:
            return RawOpsAction(action_type="refund_pre_dispute_alert", case_id=case_id)
        if meta.get("dispute_stage") == "lost" and "accept_dispute" in available:
            return RawOpsAction(action_type="accept_dispute", case_id=case_id)
    elif detail.case_type == "invoice":
        match_status = meta.get("match_status")
        duplicate_status = meta.get("duplicate_status")
        variance_amount = float(meta.get("variance_amount") or 0.0)
        write_off_threshold = float(meta.get("write_off_threshold") or 0.0)
        credit_memo_status = meta.get("credit_memo_status")
        if match_status is None and "record_three_way_match" in available:
            if _is_hard_duplicate_status(duplicate_status) or (
                duplicate_status == "suspected"
                and "duplicate_check" in set(detail.visible_flags)
                and variance_amount <= 0
            ):
                return RawOpsAction(
                    action_type="record_three_way_match",
                    case_id=case_id,
                    match_status="duplicate",
                    variance_amount=variance_amount if variance_amount else None,
                )
            if variance_amount > 0:
                return RawOpsAction(
                    action_type="record_three_way_match",
                    case_id=case_id,
                    match_status="variance",
                    variance_amount=variance_amount,
                )
            return RawOpsAction(action_type="record_three_way_match", case_id=case_id, match_status="matched")
        if meta.get("payment_batch_status") in {"scheduled", "in_progress"} and (
            _is_hard_duplicate_status(duplicate_status) or match_status == "duplicate" or variance_amount > 0
        ) and "remove_from_payment_batch" in available:
            return RawOpsAction(action_type="remove_from_payment_batch", case_id=case_id)
        if (_is_hard_duplicate_status(duplicate_status) or match_status == "duplicate") and "reject" in available:
            return RawOpsAction(action_type="reject", case_id=case_id, reason_code="duplicate_match")
        if variance_amount > 0 and variance_amount <= write_off_threshold and "write_off_small_balance" in available:
            return RawOpsAction(action_type="write_off_small_balance", case_id=case_id)
        if variance_amount > write_off_threshold and credit_memo_status == "not_requested" and "request_credit_memo" in available:
            return RawOpsAction(action_type="request_credit_memo", case_id=case_id, approved_amount=variance_amount)
        if meta.get("credit_memo_status") == "received" and "apply_credit_memo" in available:
            credit_memo = next((record.record_id for record in detail.linked_records if record.record_type == "credit_memo"), None)
            if credit_memo:
                return RawOpsAction(action_type="apply_credit_memo", case_id=case_id, credit_memo_id=credit_memo)
        if meta.get("payment_hold") and "release_payment_hold" in available and detail.status == "resolved":
            return RawOpsAction(action_type="release_payment_hold", case_id=case_id)
        if credit_memo_status in {"applied", "received"} and "approve" in available:
            return RawOpsAction(action_type="approve", case_id=case_id, decision_code="standard_approval")
        if _pending_request_field(detail) and "request_info" in available:
            return RawOpsAction(action_type="request_info", case_id=case_id, field_name=_pending_request_field(detail))
        if meta.get("match_status") == "missing_receipt" and "request_info" in available:
            return RawOpsAction(action_type="request_info", case_id=case_id, field_name="goods_receipt")
        if "approve" in available and not (_is_hard_duplicate_status(duplicate_status) or match_status == "duplicate"):
            return RawOpsAction(action_type="approve", case_id=case_id, decision_code="standard_approval")
    elif detail.case_type == "kyc":
        if meta.get("sanctions_status") == "not_started" and "run_sanctions_screen" in available:
            return RawOpsAction(action_type="run_sanctions_screen", case_id=case_id)
        if meta.get("sanctions_status") == "confirmed_match":
            if not meta.get("payments_frozen") and "freeze_payments" in available:
                return RawOpsAction(action_type="freeze_payments", case_id=case_id, reason_code="sanctions_match")
            if meta.get("ofac_report_status") in {"pending", "missed"} and "file_ofac_report" in available:
                return RawOpsAction(action_type="file_ofac_report", case_id=case_id)
            if "reject" in available:
                return RawOpsAction(action_type="reject", case_id=case_id, reason_code="sanctions_match")
        if meta.get("sanctions_status") == "clear" and meta.get("edd_status") == "not_started" and "start_edd_review" in available:
            return RawOpsAction(action_type="start_edd_review", case_id=case_id)
        correction_fields = list(meta.get("correction_fields") or [])
        if correction_fields and "request_field_correction" in available:
            return RawOpsAction(action_type="request_field_correction", case_id=case_id, field_name=correction_fields[0])
        if meta.get("beneficial_owner_status") == "pending_review" and "review_beneficial_owner" in available:
            return RawOpsAction(action_type="review_beneficial_owner", case_id=case_id, verification_decision="approve")
        field_name = _pending_request_field(detail)
        if field_name and "request_info" in available:
            return RawOpsAction(action_type="request_info", case_id=case_id, field_name=field_name)
        if _is_kyc_approval_ready(meta):
            if "review_kyc" in available:
                return RawOpsAction(action_type="review_kyc", case_id=case_id, verification_decision="approve")
            if "approve" in available:
                return RawOpsAction(action_type="approve", case_id=case_id, decision_code="standard_approval")
        if meta.get("sanctions_status") == "clear" and "reject" in available and not detail.waiting_on:
            return RawOpsAction(action_type="reject", case_id=case_id, reason_code="policy_ambiguity")
    return None


def choose_controller_action(
    observation: OpsArenaObservation,
    blocked_signatures: set[str] | None = None,
) -> RawOpsAction | None:
    blocked_signatures = blocked_signatures or set()

    def choose(action: RawOpsAction | None) -> RawOpsAction | None:
        if action is None:
            return None
        return None if _action_signature(action) in blocked_signatures else action

    available = set(observation.available_actions or [])
    detail = observation.case_detail

    if detail is None:
        next_case_id = _queue_pick(observation)
        if next_case_id and "open_case" in available:
            return choose(RawOpsAction(action_type="open_case", case_id=next_case_id))
        if "list_queue" in available:
            return choose(RawOpsAction(action_type="list_queue", limit=10, sort_by="priority"))
        if "advance_clock" in available:
            return choose(RawOpsAction(action_type="advance_clock", minutes=15))
        return None

    case_id = detail.case_id
    completed = set(detail.checks_completed)
    categories = detail.recommended_action_categories or []

    if "qa" in categories:
        if "approve_qa" in available:
            return choose(RawOpsAction(action_type="approve_qa", case_id=case_id, assignee_type="qa_reviewer"))
        if "send_to_qa" in available:
            return choose(RawOpsAction(action_type="send_to_qa", case_id=case_id, assignee_type="qa_reviewer"))

    if "notify" in categories and "send_message" in available:
        return choose(_notification_action(detail))

    if "close" in categories and "close_case" in available:
        return choose(RawOpsAction(action_type="close_case", case_id=case_id, resolution_code="completed"))

    if "wait" in categories and "advance_clock" in available:
        alternate_case = _queue_pick_alternate(observation, case_id) if detail.waiting_on else None
        if alternate_case and "open_case" in available:
            return choose(RawOpsAction(action_type="open_case", case_id=alternate_case))
        minutes = _wait_minutes(detail, observation.clock) or 5
        return choose(RawOpsAction(action_type="advance_clock", minutes=minutes))

    if "policy_review" in categories and "query_policy" in available and "review_policy" not in completed:
        return choose(RawOpsAction(action_type="query_policy", policy_id=_case_policy_id(detail.case_type)))

    if "record_review" in categories:
        check_action = _record_for_required_check(detail, available)
        if check_action is not None:
            return choose(check_action)

    if "request_missing_info" in categories and "request_info" in available:
        field_name = _pending_request_field(detail)
        if field_name:
            return choose(RawOpsAction(action_type="request_info", case_id=case_id, field_name=field_name))

    if "resolution" in categories:
        resolution_action = _resolution_action(observation)
        if resolution_action is not None:
            return choose(resolution_action)

    if (
        detail.case_type == "refund"
        and "review_policy" in completed
        and "review_order" in completed
        and "review_customer" in completed
        and "view_record" in available
        and observation.record_view is None
    ):
        linked = {record.record_type: record.record_id for record in detail.linked_records}
        dispute_id = linked.get("dispute")
        if dispute_id:
            return choose(RawOpsAction(action_type="view_record", record_type="dispute", record_id=dispute_id))

    return None


def _normalize_tool_arguments(action_name: str, arguments: dict[str, Any], observation: OpsArenaObservation) -> dict[str, Any]:
    normalized = dict(arguments)
    normalized.pop("action_type", None)
    detail = observation.case_detail

    if detail is not None and "case_id" not in normalized and action_name not in {"list_queue", "search_cases", "rebalance_queue", "batch_reorder", "bulk_assign", "bulk_route", "advance_clock"}:
        normalized["case_id"] = detail.case_id

    if action_name == "approve":
        decision_code = normalized.get("decision_code")
        if isinstance(decision_code, str):
            candidate = DECISION_CODE_ALIASES.get(decision_code.strip().lower(), decision_code)
            normalized["decision_code"] = candidate if candidate in VALID_DECISION_CODES else "standard_approval"
        elif decision_code is None:
            normalized["decision_code"] = "standard_approval"

    if action_name == "query_policy" and "policy_id" not in normalized and detail is not None:
        normalized["policy_id"] = _case_policy_id(detail.case_type)

    if action_name == "close_case" and "resolution_code" not in normalized:
        normalized["resolution_code"] = "completed"

    if action_name == "send_message" and "template_id" not in normalized and detail is not None:
        notification_action = _notification_action(detail)
        normalized.update(notification_action.model_dump(exclude_none=True))
        normalized.pop("action_type", None)

    return normalized


def _render_observation_summary(task_id: str, observation: OpsArenaObservation, history: list[str]) -> str:
    lines = [
        f"task_id: {task_id}",
        f"clock: {observation.clock}",
        f"available_actions: {', '.join(observation.available_actions or [])}",
    ]
    if observation.queue_view:
        lines.append("queue:")
        for item in (observation.queue_view or [])[:5]:
            lines.append(
                f"- {item.case_id} | type={item.case_type} | priority={item.priority} | "
                f"sla_remaining={item.sla_remaining_minutes} | status={item.status} | summary={item.summary}"
            )
    detail = observation.case_detail
    if detail is not None:
        lines.extend(
            [
                f"case_id: {detail.case_id}",
                f"case_type: {detail.case_type}",
                f"status: {detail.status}",
                f"case_phase: {detail.case_phase}",
                f"amount: {detail.amount}",
                f"required_checks: {', '.join(detail.required_checks)}",
                f"checks_completed: {', '.join(detail.checks_completed)}",
                f"requested_info_fields: {', '.join(detail.requested_info_fields)}",
                f"close_blockers: {', '.join(detail.close_blockers)}",
                f"waiting_on: {', '.join(detail.waiting_on)}",
                f"next_due_minutes: {detail.next_due_minutes}",
                f"recommended_action_categories: {', '.join(detail.recommended_action_categories)}",
                f"visible_flags: {', '.join(detail.visible_flags)}",
                "linked_records: "
                + ", ".join(f"{record.record_type}:{record.record_id}" for record in detail.linked_records),
                f"workflow_metadata: {json.dumps(detail.workflow_metadata, separators=(',', ':'), default=str)}",
            ]
        )
        if detail.communication_log:
            lines.append(
                "communication_log: "
                + ", ".join(_message_summary(entry) for entry in detail.communication_log[-3:])
            )
    if observation.policy_result is not None:
        lines.append(
            "policy_result: "
            + json.dumps(observation.policy_result.model_dump(mode="json"), separators=(",", ":"), default=str)
        )
    if observation.record_view is not None:
        lines.append("record_view: " + json.dumps(observation.record_view, separators=(",", ":"), default=str))
    if history:
        lines.append("recent_history:")
        lines.extend(f"- {entry}" for entry in history[-6:])
    if observation.system_message:
        lines.append(f"last_system_message: {observation.system_message}")
    if observation.error:
        lines.append(f"last_error: {observation.error}")
    return "\n".join(lines)


def _message_summary(entry: MessageSummary) -> str:
    return f"{entry.timestamp}:{entry.subject}:{entry.template_id or 'none'}"


def _model_action(
    client: OpenAI,
    tool_map: dict[str, dict[str, Any]],
    task_id: str,
    observation: OpsArenaObservation,
    history: list[str],
) -> tuple[RawOpsAction | None, str | None]:
    tools = _tools_for_available_actions(tool_map, observation.available_actions or [])
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _render_observation_summary(task_id, observation, history)},
            ],
            tools=tools,
            tool_choice="required",
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
    except Exception as exc:  # pragma: no cover - network/runtime path
        return None, f"model_request_failed:{exc}"

    message = completion.choices[0].message
    if not message.tool_calls:
        return None, "no_tool_call"
    tool_call = message.tool_calls[0]
    action_name = tool_call.function.name
    if action_name not in set(observation.available_actions or []):
        return None, f"unavailable_action:{action_name}"
    try:
        arguments = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError as exc:
        return None, f"bad_tool_json:{exc}"
    if not isinstance(arguments, dict):
        return None, "bad_tool_payload"
    arguments = _normalize_tool_arguments(action_name, arguments, observation)
    try:
        return RawOpsAction(action_type=action_name, **arguments), None
    except Exception as exc:
        return None, f"hallucination:{exc}"


def _observation_marker(observation: OpsArenaObservation) -> tuple[Any, ...]:
    detail = observation.case_detail
    if detail is None:
        queue_open = tuple((item.case_id, item.status, item.priority) for item in (observation.queue_view or [])[:6])
        return ("queue", queue_open)
    meta = detail.workflow_metadata or {}
    return (
        detail.case_id,
        detail.status,
        detail.case_phase,
        tuple(detail.checks_completed),
        tuple(detail.close_blockers),
        tuple(detail.waiting_on),
        len(detail.communication_log),
        len(detail.requested_info_fields),
        meta.get("qa_status"),
        meta.get("sanctions_status"),
        meta.get("approval_status"),
    )


def _safe_default_action(observation: OpsArenaObservation, blocked_signatures: set[str] | None = None) -> RawOpsAction:
    detail = observation.case_detail
    available = set(observation.available_actions or [])
    controller_action = choose_controller_action(observation, blocked_signatures=blocked_signatures or set())
    if controller_action is not None:
        return controller_action
    if detail is None:
        next_case_id = _queue_pick(observation)
        if next_case_id and "open_case" in available:
            return RawOpsAction(action_type="open_case", case_id=next_case_id)
    if detail is not None:
        if "query_policy" in available and "review_policy" not in set(detail.checks_completed):
            return RawOpsAction(action_type="query_policy", policy_id=_case_policy_id(detail.case_type))
        check_action = _record_for_required_check(detail, available)
        if check_action is not None:
            return check_action
        resolution_action = _resolution_action(observation)
        if resolution_action is not None and _action_signature(resolution_action) not in (blocked_signatures or set()):
            return resolution_action
        alternate_case = _queue_pick_alternate(observation, detail.case_id)
        if alternate_case and "open_case" in available:
            return RawOpsAction(action_type="open_case", case_id=alternate_case)
        if detail.waiting_on and "advance_clock" in available:
            return RawOpsAction(action_type="advance_clock", minutes=_wait_minutes(detail, observation.clock) or 5)
        if "advance_clock" in available:
            return RawOpsAction(action_type="advance_clock", minutes=_wait_minutes(detail, observation.clock) or 5)
    if "list_queue" in available:
        return RawOpsAction(action_type="list_queue", limit=10, sort_by="priority")
    if "advance_clock" in available:
        return RawOpsAction(action_type="advance_clock", minutes=5)
    return RawOpsAction(action_type="list_queue", limit=10, sort_by="priority")


def run_task(task_id: str, client: OpenAI, tool_map: dict[str, dict[str, Any]]) -> float:
    env = OpsArenaEnv(base_url=ENV_BASE_URL).sync()
    rewards: list[float] = []
    history: list[str] = []
    steps_taken = 0
    final_score = 0.0
    success = False
    blocked_signatures: set[str] = set()
    stalled_signatures: dict[str, int] = {}
    forced_action_by_case: dict[str, RawOpsAction] = {}

    print(format_start_line(task_id, BENCHMARK, MODEL_NAME), flush=True)
    with env:
        try:
            result = env.reset(task_id=task_id, seed=SEED)
            while not result.done and steps_taken < MAX_STEPS:
                observation = result.observation
                before_marker = _observation_marker(observation)
                forced_action = None
                if observation.case_detail is not None:
                    candidate = forced_action_by_case.get(observation.case_detail.case_id)
                    if candidate is not None and candidate.action_type in set(observation.available_actions or []):
                        forced_action = candidate
                        forced_action_by_case.pop(observation.case_detail.case_id, None)
                controller_action = forced_action or choose_controller_action(observation, blocked_signatures=blocked_signatures)
                if controller_action is not None:
                    action = controller_action
                    decision_source = "forced" if forced_action is not None else "controller"
                else:
                    action, model_error = _model_action(client, tool_map, task_id, observation, history)
                    decision_source = "model"
                    if action is None or _action_signature(action) in blocked_signatures:
                        action = choose_controller_action(observation, blocked_signatures=blocked_signatures) or _safe_default_action(
                            observation,
                            blocked_signatures=blocked_signatures,
                        )
                        history.append(f"model_fallback={model_error}")

                signature = _action_signature(action)
                if signature in blocked_signatures:
                    action = _safe_default_action(observation, blocked_signatures=blocked_signatures)
                    signature = _action_signature(action)

                result = env.step(action)
                steps_taken += 1
                reward = float(result.reward or 0.0)
                rewards.append(reward)
                error = result.observation.error
                after_marker = _observation_marker(result.observation)
                progressed = after_marker != before_marker or result.done
                if error is not None:
                    blocked_signatures.add(signature)
                    if (
                        observation.case_detail is not None
                        and observation.case_detail.case_type == "kyc"
                        and "invalid document" in str(error).lower()
                        and "reject" in set(result.observation.available_actions or [])
                    ):
                        forced_action_by_case[observation.case_detail.case_id] = RawOpsAction(
                            action_type="reject",
                            case_id=observation.case_detail.case_id,
                            reason_code="invalid_document",
                        )
                elif not progressed:
                    stalled_signatures[signature] = stalled_signatures.get(signature, 0) + 1
                    if stalled_signatures[signature] >= 2:
                        blocked_signatures.add(signature)
                else:
                    stalled_signatures.pop(signature, None)
                print(format_step_line(steps_taken, action, reward, result.done, error), flush=True)
                history.append(
                    f"step={steps_taken} source={decision_source} action={_format_action(action)} "
                    f"reward={reward:.2f} error={_flatten_error(error)}"
                )

            try:
                final_state = env.state().model_dump(mode="json")
                final_score = float(final_state.get("benchmark_score", 0.0))
            except Exception:
                final_score = 0.0
            success = final_score >= SUCCESS_SCORE_THRESHOLD
            return final_score
        finally:
            print(format_end_line(success, steps_taken, final_score, rewards), flush=True)


def main() -> None:
    if not HF_TOKEN:
        raise RuntimeError("Missing required environment variable: HF_TOKEN")
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    tool_map = _build_tool_map()
    for task_id in TASK_IDS:
        run_task(task_id, client, tool_map)


if __name__ == "__main__":
    main()
