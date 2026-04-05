from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from opsarena.training import refund_eval as reval


def test_parse_tool_json_plain():
    assert reval._parse_tool_json('{"tool": "list_queue", "arguments": {}}') == ("list_queue", {})


def test_parse_tool_json_with_extra_text():
    text = 'Thought.\n{"tool": "open_case", "arguments": {"case_id": "c1"}}\n'
    assert reval._parse_tool_json(text) == ("open_case", {"case_id": "c1"})


def test_invoke_tool_list_queue():
    from opsarena.training.refund_grpo_env import RefundExceptionToolEnv

    env = RefundExceptionToolEnv(seed_sequence=[7], random_seed=0)
    env.reset()
    out = reval._invoke_tool(env, "list_queue", {})
    assert "queue:" in out or "case_id" in out.lower()
