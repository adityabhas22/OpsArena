from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import OpsArenaObservation, OpsArenaState, RawOpsAction


class OpsArenaEnv(EnvClient[RawOpsAction, OpsArenaObservation, OpsArenaState]):
    def _step_payload(self, action: RawOpsAction) -> dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[OpsArenaObservation]:
        obs = OpsArenaObservation.model_validate(payload.get("observation", {}))
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> OpsArenaState:
        return OpsArenaState.model_validate(payload)
