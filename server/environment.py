from __future__ import annotations

from typing import Any

from openenv.core.env_server import Environment
from openenv.core.env_server.interfaces import EnvironmentMetadata

from opsarena.engine.graders import grade_episode
from opsarena.engine.observations import render_observation
from opsarena.engine.scenarios import build_task_state
from opsarena.engine.transitions import apply_action
from opsarena.models import OpsArenaObservation, OpsArenaState, RawOpsAction, validate_ops_action


class OpsArenaEnvironment(Environment[RawOpsAction, OpsArenaObservation, OpsArenaState]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._state = None

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str = "refund_exception",
        **_: Any,
    ) -> OpsArenaObservation:
        self._state = build_task_state(task_id, seed=seed or 7, episode_id=episode_id)
        return render_observation(self._state, message=f"Task {task_id} initialized")

    def step(
        self,
        action: RawOpsAction,
        timeout_s: float | None = None,
        **_: Any,
    ) -> OpsArenaObservation:
        if self._state is None:
            self.reset()
        typed_action = validate_ops_action(action)
        result = apply_action(self._state, typed_action)
        done = self._is_done()
        if done:
            self._state.grader_breakdown = grade_episode(self._state)
        obs = render_observation(
            self._state,
            success=result.success,
            action_type=typed_action.action_type,
            message=result.message,
            error_code=result.error_code,
            reward=result.train_reward,
            done=done,
        )
        obs.metadata.update(
            {
                "objective_reward": result.objective_reward,
                "train_reward": result.train_reward,
                "objective_score": self._state.objective_score,
                "train_score": self._state.train_score,
                "grader_breakdown": self._state.grader_breakdown,
            }
        )
        return obs

    def _is_done(self) -> bool:
        assert self._state is not None
        all_closed = all(case.status == "closed" for case in self._state.cases.values())
        no_pending_events = len(self._state.scheduled_events) == 0
        max_steps = self._state.metadata.get("max_steps", 40)
        return (all_closed and no_pending_events) or self._state.step_count >= max_steps

    @property
    def state(self) -> OpsArenaState:
        if self._state is None:
            self.reset()
        assert self._state is not None
        return OpsArenaState(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
            task_id=self._state.task_id.value,
            scenario_seed=self._state.scenario_seed,
            simulated_time=self._state.current_time,
            cases_resolved=self._state.metrics.cases_resolved,
            cases_total=len(self._state.cases),
            objective_score=self._state.objective_score,
            train_score=self._state.train_score,
            current_case_id=self._state.current_case_id,
            grader_breakdown=self._state.grader_breakdown,
        )

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="OpsArena",
            description="High-fidelity ecommerce operations environment for refunds, invoice exceptions, KYC, and queue triage.",
            version="0.1.0",
        )
