# OpsArena RL Readiness

This document defines the training-facing contract for OpsArena.

## Reward semantics

OpsArena now separates three concepts:

- `benchmark_score`: final trajectory score in `[0.0, 1.0]` from `grade_episode`
- `objective_score`: the accumulated RL objective exposed by the environment; this now tracks the terminal `benchmark_score`
- `train_score`: training score used during online interaction; this combines shaping progress with the final `benchmark_score`

The previous dense business reward is still computed internally for analysis, but it is stored as debug metadata:

- `legacy_objective_score`
- `legacy_train_score`

This keeps the public RL objective simple and verifiable while preserving richer diagnostics for reward-design work.

## Why this is better for RL

- The reported objective is stable across tasks and seeds because it is normalized through the grader.
- Hard failures still affect the final score through deterministic grading.
- Shaping remains available for training without contaminating benchmark reporting.
- Pathological policies can be tested against the same final score the model will optimize.

## Scenario generation

Scenario templates now drive real latent variation instead of acting only as documentation.

Each task samples from YAML `atomic_factors` to vary:

- case type specifics such as fraud risk, invoice error type, batch timing, and dispute mode
- operational pressure such as SLA tightness and queue shock
- hidden truth such as duplicate status, sanctions path, and recovery outcome

This gives materially higher scenario entropy across seeds and supports train/dev/test family splits.

## Validation workflow

Use these checks before any RL run:

```bash
./.venv/bin/pytest -q
./.venv/bin/openenv validate
./.venv/bin/python -m scripts.run_eval
./.venv/bin/python -m scripts.validate_rewards
```

`scripts.validate_rewards` now evaluates multiple seeds per task and compares the oracle against pathological policies using mean benchmark score.

## Training guidance

Recommended order:

1. Start with a single workflow wrapper, not the full 61-action surface.
2. Train against `benchmark_score` or a binarized version of it.
3. Use task-specific wrappers or a custom rollout path so the model only sees valid tools for the active workflow.
4. Hold out scenario families, not just seeds.
5. Add queue triage only after single-workflow performance is reliable.
