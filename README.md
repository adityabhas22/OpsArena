# OpsArena

**OpsArena** is a deterministic, partially observable simulator for **e-commerce operations exception handling**: refunds and card disputes, invoice / AP exceptions with KYC-style verification, and multi-case queue triage under SLA pressure. Agents act through a **typed action space** (structured API calls, not browser automation) and are evaluated on final world state, audit trails, and trajectory rules.

The package implements an [OpenEnv](https://github.com/openenv/openenv-core)-compatible environment exposed over HTTP (FastAPI), suitable for benchmarking and RL-style training workflows.

## What it does today

- **Simulated world state** (`opsarena/engine/state.py`): cases, linked business records (orders, invoices, payments, KYC, disputes, etc.), policies, a simulated clock, scheduled future events, metrics, and a full audit log.
- **Seeded scenarios** (`opsarena/engine/scenarios.py`, `data/scenario_templates/`): builds tasks from YAML templates plus procedural setup so the same seed yields the same trajectory for the same actions.
- **Transitions** (`opsarena/engine/transitions.py`): validates actions, updates state, runs the **event scheduler** (`opsarena/engine/scheduler.py`) for delayed outcomes (e.g. document arrival, chargebacks, dispute updates).
- **Observations** (`opsarena/engine/observations.py`): renders what the agent sees (queue, case detail, records, policy queries, audit slices)—not raw hidden fields.
- **Rewards** (`opsarena/rewards.py`): step-level **objective** and **shaping** signals; see [docs/rewards.md](docs/rewards.md).
- **Graders** (`opsarena/engine/graders.py`): on episode end, outcome checks against case ground truth plus trajectory checks (e.g. policy before approve, notify before close).
- **Baselines** (`baselines/`): including an oracle-style runner; `scripts/run_eval.py` runs quick oracle smoke checks per task.

## Task IDs

| `task_id` | Description |
|-----------|-------------|
| `refund_exception` | Refund / dispute workflow with policy, evidence, messaging, and delayed dispute outcomes. |
| `invoice_plus_kyc` | Combined invoice exception and merchant KYC verification. |
| `queue_triage` | Multi-case queue: multiple refunds, invoices, and KYC under shared SLA pressure. |

These match `TaskId` in `opsarena/enums.py` and drive scenario selection in `build_task_state`.

## Project structure

```text
OpsArena/
├── opsarena/                 # Core library
│   ├── engine/               # Simulator: state, scenarios, transitions, scheduler, observations, policies, graders
│   ├── models.py             # Pydantic: actions, observations, validation
│   ├── documents.py          # Record types (orders, invoices, KYC, …)
│   ├── enums.py              # Task IDs, case types, resolutions, …
│   ├── rewards.py            # Reward and shaping logic
│   ├── client.py             # HTTP client (OpenEnv EnvClient)
│   └── action_docs.py        # Action documentation helpers
├── server/                   # OpenEnv HTTP server
│   ├── app.py                # FastAPI app via openenv create_app
│   └── environment.py        # OpsArenaEnvironment: reset / step / metadata
├── data/
│   ├── scenario_templates/   # Per-task YAML templates
│   └── policies/             # Refund, invoice, KYC policy YAML
├── configs/                  # e.g. reward_weights.yaml
├── baselines/                # Oracle, rule-based, OpenAI baseline stubs
├── scripts/                  # run_eval.py, validate_rewards.py
├── tests/                    # pytest suite
├── docs/                     # Deeper docs (see below)
├── openenv.yaml              # OpenEnv space spec (runtime, app entrypoint, port)
├── Dockerfile                # Container running uvicorn on :8000
├── pyproject.toml
└── README.md
```

## Setup

**Requirements:** Python **3.11+**

### Install from source

```bash
cd OpsArena
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The `[dev]` extra installs `pytest` and `httpx` for tests and HTTP client usage. Core runtime deps are `openenv-core`, `pydantic`, and `pyyaml` (see `pyproject.toml`).

For GRPO training experiments, install the training extra as well:

```bash
pip install -e ".[dev,train]"
```

For `--use-vllm`, install the vLLM extra and make sure the active PyTorch build has CUDA
libraries. On Linux `aarch64`, the default PyPI `torch` wheel may be CPU-only; if that
happens, replace it with the official CUDA 13.0 wheel:

```bash
uv pip install --python .venv/bin/python -e ".[dev,train,train-vllm]"
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu130 \
  --reinstall torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
```

### Run tests

```bash
pytest
```

### Run the HTTP server

```bash
# After pip install -e .
server
# Serves the OpenEnv API (default host 0.0.0.0, port 8000 — see server.app:main)
```

Or with uvicorn directly:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Port and app entry are also described in [`openenv.yaml`](openenv.yaml).

### Docker

```bash
docker build -t opsarena .
docker run -p 8000:8000 opsarena
```

### Quick oracle check

```bash
python -m scripts.run_eval
```

Prints JSON summaries for the oracle baseline across all task IDs.

### Refund-only GRPO scaffold

```bash
python scripts/train_refund_grpo.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir artifacts/refund-grpo
```

This uses the refund-only wrapper in `opsarena/training/refund_grpo_env.py`, not the full environment action surface. Start here before attempting multi-workflow training.

With `--use-vllm`, the script now checks for a CUDA-enabled PyTorch install up front and exits
with an actionable message instead of failing deep inside TRL/vLLM imports.

## Using the client

`opsarena.client.OpsArenaEnv` subclasses OpenEnv’s `EnvClient`. Point it at the running server URL and use `reset` / `step` with `RawOpsAction` payloads as in `opsarena/models.py`. See [docs/action-space.md](docs/action-space.md) for action names and fields.

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/action-space.md](docs/action-space.md) | Typed actions (list queue, open case, query policy, approve, …) |
| [docs/workflows.md](docs/workflows.md) | Intended workflows: refund/dispute, AP/invoice, KYC, queue triage |
| [docs/rewards.md](docs/rewards.md) | Objective vs shaping reward, penalties, workflow intuition |
| [docs/rl-readiness.md](docs/rl-readiness.md) | Training-facing reward semantics, scenario entropy, and validation flow |
| [docs/grpo-refund-training.md](docs/grpo-refund-training.md) | Refund-only GRPO wrapper, reward function, and training caveats |
| [docs/production-roadmap.md](docs/production-roadmap.md) | Roadmap and production-oriented notes |
| [opsarena_blueprint.md](opsarena_blueprint.md) | Full design thesis (POMDP, graders, scenario generation, evaluation) |

## Episode lifecycle (code map)

1. **`OpsArenaEnvironment.reset`** (`server/environment.py`) calls `build_task_state` → `render_observation`.
2. **`step`** validates `RawOpsAction` → `apply_action` → processes due scheduled events → updates rewards/scores → `render_observation`.
3. **Done** when all cases are `closed` or `step_count` reaches `metadata["max_steps"]`; then `grade_episode` fills `grader_breakdown` on the observation metadata.

## Version

Current package version: **0.1.0** (`pyproject.toml` / environment metadata).
