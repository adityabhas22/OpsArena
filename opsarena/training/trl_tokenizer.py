"""Prepare Hugging Face tokenizers for TRL ``GRPOTrainer`` + tool environments.

Hub revisions (e.g. ``Qwen3-4B-Instruct-2507``) may ship chat templates that do not
exactly match TRL's built-in strings, so ``add_response_schema`` fails. We align
the tokenizer with TRL's known Qwen3.5 templates when needed.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


def prepare_tokenizer_for_grpo(model_id: str) -> "PreTrainedTokenizer":
    """Load tokenizer, set pad token, and ensure ``response_schema`` is set for tool use."""

    from transformers import AutoTokenizer

    from trl.chat_template_utils import add_response_schema

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    try:
        return add_response_schema(tokenizer)
    except ValueError as first_err:
        mid = model_id.lower()
        if "qwen3" not in mid:
            raise ValueError(
                "Unrecognized chat template for GRPO tool training (TRL add_response_schema). "
                "Use a Qwen3 instruct checkpoint or set tokenizer.response_schema manually. "
                f"Original error: {first_err}"
            ) from first_err

        import trl.chat_template_utils as ctu

        m = re.search(r"(\d+\.?\d*)b", mid)
        size_b = float(m.group(1)) if m else 4.0
        if size_b <= 2.0:
            tokenizer.chat_template = ctu.qwen3_5_chat_template_2b_and_below
        else:
            tokenizer.chat_template = ctu.qwen3_5_chat_template_4b_and_above

        return add_response_schema(tokenizer)
