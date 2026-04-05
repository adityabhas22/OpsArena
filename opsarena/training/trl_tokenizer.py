"""Prepare Hugging Face tokenizers for TRL ``GRPOTrainer`` + tool environments.

TRL 1.x moved some utilities and renamed others.  This module handles:

- ``add_response_schema`` (TRL ≥0.13): aligns the chat template so TRL can
  identify assistant-turn boundaries in tool-call completions.
- Qwen3 chat-template variants (2B vs 4B+): the Hub revision
  ``Qwen3-4B-Instruct-2507`` ships a slightly different template string.
- Graceful fallback: if neither the function nor the template constants are
  importable (e.g. a future TRL API break), we return the tokenizer unchanged
  so training can proceed with the model's native template.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


def prepare_tokenizer_for_grpo(model_id: str) -> "PreTrainedTokenizer":
    """Load tokenizer, set pad token, and align chat template for TRL GRPO."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Attempt to import TRL's add_response_schema.  It lives in different
    # locations across TRL versions; try both known paths.
    add_response_schema = None
    chat_template_utils = None
    for module_path in ("trl.chat_template_utils", "trl.extras.chat_template_utils"):
        try:
            import importlib
            mod = importlib.import_module(module_path)
            if hasattr(mod, "add_response_schema"):
                add_response_schema = mod.add_response_schema
                chat_template_utils = mod
                break
        except (ImportError, ModuleNotFoundError):
            continue

    if add_response_schema is None:
        # TRL version does not expose add_response_schema — use tokenizer as-is.
        return tokenizer

    try:
        return add_response_schema(tokenizer)
    except (ValueError, AttributeError):
        pass

    # Qwen3 fallback: swap in the correct size-variant template then retry.
    mid = model_id.lower()
    if "qwen3" not in mid:
        # Unknown model without a matching template — return without schema.
        return tokenizer

    m = re.search(r"(\d+\.?\d*)b", mid)
    size_b = float(m.group(1)) if m else 4.0
    template_attr = (
        "qwen3_5_chat_template_2b_and_below" if size_b <= 2.0
        else "qwen3_5_chat_template_4b_and_above"
    )
    template = getattr(chat_template_utils, template_attr, None)
    if template is None:
        return tokenizer

    tokenizer.chat_template = template
    try:
        return add_response_schema(tokenizer)
    except (ValueError, AttributeError):
        return tokenizer
