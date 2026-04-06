"""JSON file logger for GRPO training metrics.

Captures every TRL log callback to a JSONL file so training runs can be
analyzed offline. Drop-in: pass the callback to GRPOConfig(report_to=[])
and register it after trainer creation.

Usage:
    from opsarena.training.json_logger import JSONMetricsLogger

    logger = JSONMetricsLogger(output_dir="artifacts/refund-grpo")
    trainer = GRPOTrainer(...)
    trainer.add_callback(logger)
    trainer.train()
    # => artifacts/refund-grpo/metrics.jsonl written every step
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class JSONMetricsLogger(TrainerCallback):
    """Writes every logged metric dict as a JSONL line to ``{output_dir}/metrics.jsonl``."""

    def __init__(self, output_dir: str | Path) -> None:
        self._path = Path(output_dir) / "metrics.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._path, "a", buffering=1)  # line-buffered
        self._start = time.monotonic()

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if logs is None:
            return
        record = {
            "step": state.global_step,
            "epoch": state.epoch,
            "wall_seconds": round(time.monotonic() - self._start, 2),
            **{k: _serialize(v) for k, v in logs.items()},
        }
        self._fh.write(json.dumps(record, default=str) + "\n")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._fh.close()


def _serialize(v: Any) -> Any:
    """Convert non-JSON-serializable values."""
    if isinstance(v, float):
        if v != v:  # NaN
            return None
        return round(v, 6)
    return v
