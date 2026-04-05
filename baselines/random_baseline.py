"""Random baseline for reward sanity checking.

Picks a uniformly random valid action at each step. A well-designed
reward function should score this baseline poorly — if it scores well,
the rewards are gameable.
"""
from __future__ import annotations

import random

from server.environment import OpsArenaEnvironment
from opsarena.models import RawOpsAction


def run_random(task_id: str, seed: int = 7, max_steps: int | None = None) -> dict:
    env = OpsArenaEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)
    assert env._state is not None
    rng = random.Random(seed)
    limit = max_steps or env._state.metadata.get("max_steps", 40)

    while not env._is_done() and env._state.step_count < limit:
        actions = obs.available_actions or []
        if not actions:
            break

        action_type = rng.choice(actions)

        # Build a minimal valid action payload
        payload: dict = {"action_type": action_type}
        case_id = None

        # Pick a case_id from queue if the action needs one
        if action_type == "open_case":
            queue = obs.queue_view or []
            if queue:
                case_id = rng.choice(queue).case_id
        elif obs.case_detail is not None:
            case_id = obs.case_detail.case_id

        if case_id:
            payload["case_id"] = case_id

        # Fill required fields for specific actions
        if action_type == "advance_clock":
            payload["minutes"] = rng.choice([3, 5, 10, 15])
        elif action_type == "query_policy":
            payload["policy_id"] = rng.choice(["refund_policy", "invoice_policy", "kyc_policy"])
        elif action_type in ("approve", "reject", "escalate", "close_case"):
            payload["reason_code"] = "complete"
        elif action_type == "send_message":
            payload["template_id"] = "case_closed"
            payload["slots"] = {"case_id": case_id or "unknown", "resolution": "pending"}
        elif action_type == "view_record":
            if obs.case_detail and obs.case_detail.linked_records:
                record = rng.choice(obs.case_detail.linked_records)
                payload["record_type"] = record.record_type
                payload["record_id"] = record.record_id
            else:
                continue  # skip this step
        elif action_type == "request_info":
            payload["field_name"] = "goods_receipt"
        elif action_type == "review_kyc":
            payload["verification_decision"] = rng.choice(["approve", "reject", "request_resubmission"])

        try:
            obs = env.step(RawOpsAction.model_validate(payload))
        except Exception:
            # Invalid action — skip and try again
            continue

    return env.state.model_dump()


if __name__ == "__main__":
    for task in ["refund_exception", "invoice_plus_kyc", "queue_triage"]:
        result = run_random(task)
        print(f"{task}: obj={result['objective_score']:.2f} steps={result['step_count']} grader={result.get('grader_breakdown', {})}")
