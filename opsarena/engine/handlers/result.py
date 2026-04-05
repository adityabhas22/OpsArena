from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TransitionResult:
    success: bool
    message: str
    objective_reward: float = 0.0
    train_reward: float = 0.0
    error_code: str | None = None
