# OpsArena Blueprint

## Paste this section to the coding agent

Build a Python project called `opsarena`.

Goal: implement a Gymnasium-compatible, API-first RL environment for enterprise exception handling. The environment should simulate an operations dashboard with a queue of cases. The agent must inspect records, query policy, request missing information, approve, reject, escalate, defer, communicate, and close cases while optimizing correctness, compliance, speed, cost, and downstream risk.

Hard requirements:
1. Deterministic simulator and deterministic graders.
2. Partially observable state.
3. Typed action space, not free-form browser clicks.
4. Seeded scenario generation.
5. Delayed consequences through scheduled future events.
6. Primary grading by final world state plus logged event history.
7. Secondary grading by trajectory invariants.
8. Separate objective reward from shaping reward.
9. Evaluation harness that supports repeated trials and pass^k style reliability measurement.
10. Initial version should include 4 workflows: refund exception, duplicate invoice mismatch, missing KYC for payout, and queue triage under SLA pressure.

Implementation target for v0.1:
- Python 3.11
- Gymnasium environment API
- Pydantic or dataclasses for state and action schemas
- YAML scenario templates
- JSONL trajectory logging
- Rule-based oracle baseline
- CLI for `play`, `eval`, and `generate-scenarios`

---

## 1. Working thesis

OpsArena is a stateful simulator for operations exception handling.

A company runs many repetitive, policy-heavy workflows that are currently resolved by human operators in queues: refunds, payment disputes, invoice mismatches, missing compliance documents, stockouts, vendor delays, and similar cases. These are not one-shot classification problems. The operator must inspect incomplete evidence, choose which action to take next, communicate when needed, and manage tradeoffs between resolution speed, policy compliance, human escalation load, and future risk.

That makes the right abstraction a partially observable control problem with delayed consequences. Existing agent work already covers generic web tasks, general computer use, office workflows, and CRM or customer support workflows, but recent benchmark papers and enterprise evaluation guidance still identify gaps around reliability, long-horizon dynamics, compliance, access control, and latent transition laws.[1][2][3][4][5][6][7][8][13]

---

## 2. Why RL is the correct fit

RL is the correct fit because the problem is about optimizing a policy over sequences of actions, not predicting a single label or next step.

### 2.1 Sequential decision making
The agent does not solve a case in one shot. It acts in a loop:
1. inspect queue
2. pick a case
3. gather evidence
4. choose an intervention
5. observe the updated system state
6. continue until termination

A locally plausible action can still be globally bad. For example, approving a refund without enough checks may reduce immediate latency but create a reopen or chargeback later.

### 2.2 Partial observability
The full truth is not visible at the start. Hidden variables can include true fraud risk, whether a document is still valid, whether an invoice is truly duplicate, or whether a user will respond with missing information. The agent must infer the best action from limited observations.

### 2.3 Delayed consequences
Many outcomes arrive after the immediate action:
- invalid approval causes a later loss
- poor prioritization causes future SLA misses
- incomplete customer notification causes later reopen
- over-escalation overloads human queues and raises later delay penalties

### 2.4 Multiple valid policies
There is rarely one gold action sequence. Different action sequences can all be acceptable if they end in a compliant final state with reasonable efficiency.

### 2.5 Need for cumulative objective optimization
The business objective is cumulative and multi-dimensional:
- resolve cases correctly
- respect policy
- stay within SLA
- minimize escalation cost
- reduce downstream failures
- maintain throughput at the queue level

This is closer to RL than to supervised imitation. Imitation is still useful for warm start or baseline policies, but it is not sufficient as the main optimization target because the best action often depends on future consequences rather than local agreement with a historical operator step.

Research basis:
- BrowserGym frames web-agent evaluation around gym-like environments with explicit observation and action spaces.[1]
- OSWorld uses execution-based evaluation because agent tasks span many steps and modify the real environment state.[2]
- Anthropic notes that agent evals differ from single-turn model evals because agents operate over many turns, call tools, modify state, and must be judged by final environment outcome, not by what they say.[11]
- Google Cloud argues that final-output-only metrics are insufficient because agents can produce correct-looking answers through incorrect processes, which it calls silent failure.[10]
- OdysseyArena argues that current benchmarks often remain deductive and fail to test discovery of latent transition laws, which is directly relevant when the agent must learn second-order effects from experience.[8]

---

## 3. Why this environment is worth building

This project sits in a gap that is important and still under-served.

### 3.1 Current benchmark landscape
- BrowserGym unifies existing web-agent benchmarks in a gym-like framework.[1]
- OSWorld provides 369 open-ended computer tasks and execution-based evaluation across real operating systems.[2]
- OdysseyBench targets long-horizon office workflows across Word, Excel, PDF, Email, and Calendar.[3]
- Tau-bench evaluates tool-agent-user interaction with database-state grading and a reliability metric, pass^k.[4]
- CRMArena and CRMArena-Pro evaluate business workflows in CRM settings with realistic schemas, latent variables, multi-turn interactions, and confidentiality concerns.[5][6]

### 3.2 What remains missing
SAP's 2025 evaluation tutorial highlights enterprise-specific needs that current benchmarks often under-cover: role-aware access control, reliability guarantees, dynamic and long-horizon interactions, compliance, and efficient time- and cost-bounded evaluation.[7]

This creates room for an environment that is:
- API-first and deterministic enough to be trainable
- realistic enough to matter for enterprise operations
- long-horizon enough to expose first- and second-order effects
- auditable enough to support evaluation-driven development

### 3.3 Evidence that business-domain agent reliability is still weak
- Tau-bench reports that even strong function-calling agents succeed on fewer than 50 percent of tasks and are inconsistent across repeated trials, with pass^8 below 25 percent in retail.[4]
- CRMArena reports less than 40 percent success with ReAct and less than 55 percent even with function calling.[5]
- CRMArena-Pro reports around 58 percent single-turn success and about 35 percent multi-turn success, with near-zero inherent confidentiality awareness.[6]

This is enough evidence that high-fidelity operational environments still have room to create value.

---

## 4. Product framing

### 4.1 One-sentence problem statement
Build a deterministic, partially observable RL environment that simulates enterprise exception handling so an agent can learn to make correct, policy-compliant, cost-aware operational decisions over time.

### 4.2 Company problem it solves
Companies have large volumes of repetitive operational exceptions that consume specialist time and require policy-sensitive judgment. The environment lets teams train and evaluate agents that can automate or assist with these workflows while preserving auditability and control.

### 4.3 User of the environment
There are two users:
1. Researchers and engineers who want a benchmark and training environment.
2. Companies that want to test whether an agent can safely act in operations workflows before deployment.

### 4.4 Non-goals for v0.1
- No browser automation layer.
- No free-form UI clicking.
- No LLM-as-primary-grader.
- No multi-agent collaboration in the first release.
- No production connector integration.

---

## 5. Formal environment model

Model the environment as a POMDP with deterministic transitions conditional on seeded exogenous events.

### 5.1 State
`S` contains:
- queue state
- case state
- policy state
- user and vendor records
- human escalation queues
- simulated clock
- scheduled future events
- hidden variables
- audit log

### 5.2 Observation
`O` contains only what the agent can inspect through tools. Observations are rendered from state and permissions. Hidden variables are never directly exposed.

### 5.3 Actions
`A` is a finite set of typed, schema-validated actions.

### 5.4 Transition function
`T(s, a, seed)` is deterministic once the scenario seed is fixed. All randomness must be sampled at scenario creation time or through a seeded event scheduler.

### 5.5 Reward
`R` has two layers:
- `R_objective`, the true business objective used for reporting and final evaluation
- `R_shape`, a policy-preserving shaping signal used for training only

### 5.6 Termination
Episode ends when:
- all required cases are resolved or expired
- hard time horizon reached
- catastrophic policy violation triggers early stop
- queue objective for the episode is complete

---

## 6. What the environment looks like to the agent

The environment is an API-first dashboard abstraction, not a pixel browser task.

### 6.1 Core panes
1. Queue view
   - list of open cases
   - case type
   - priority
   - SLA remaining
   - short summary

2. Case detail view
   - customer or vendor profile
   - transaction or invoice history
   - flags and prior actions
   - message history
   - related records

3. Policy and knowledge tools
   - policy clauses
   - thresholds
   - role permissions
   - mandatory fields

4. Clock and system state
   - current simulated time
   - pending future events
   - escalation queue load

### 6.2 Observation rendering rule
Every observation should be generated from structured state through a renderer. Keep the renderer separate from the simulator and the grader.

Reason: OSWorld and Tau-bench both show the value of execution-based or state-based evaluation, and Anthropic explicitly separates transcript from outcome.[2][4][11]

---

## 7. Action space

Use typed actions with strict schemas.

### 7.1 Information-gathering actions
- `list_queue(sort_by, filters)`
- `open_case(case_id)`
- `view_record(record_type, record_id)`
- `query_policy(policy_id, clause_id)`
- `search_cases(filters)`
- `inspect_audit(case_id)`

### 7.2 Workflow actions
- `request_info(case_id, field_name, template_id)`
- `approve(case_id, decision_code, metadata)`
- `reject(case_id, reason_code, metadata)`
- `escalate(case_id, queue_name, reason_code)`
- `defer(case_id, until_time, reason_code)`
- `assign(case_id, assignee_type)`

### 7.3 Communication actions
- `send_message(case_id, template_id, slots)`
- `log_internal_note(case_id, note_code, metadata)`

### 7.4 Queue-management actions
- `prioritize(case_id, new_priority)`
- `batch_reorder(ordering_rule)`
- `advance_clock(minutes)`

### 7.5 Closing actions
- `close_case(case_id, resolution_code)`
- `reopen_case(case_id, reason_code)`

### 7.6 Invalid actions
Invalid actions should not be silently ignored. They should return:
- explicit error observation
- no state change or limited penalty state change
- immediate penalty
- audit log entry

---

## 8. Initial workflows for v0.1

Start with four workflows that cover most of the mechanics without making the first version too broad.

### 8.1 High-value refund exception
Hidden factors:
- true fraud risk
- policy eligibility
- customer history quality

Human process:
1. inspect order and customer history
2. check refund policy and approval threshold
3. decide whether to approve, reject, or escalate
4. notify customer
5. close case

Agent requirements:
- gather sufficient evidence
- respect thresholds
- avoid approving high-risk or ineligible refunds
- avoid unnecessary escalation
- resolve within SLA

Downstream events:
- chargeback if invalid approval
- reopen if customer not informed correctly

### 8.2 Duplicate invoice mismatch
Hidden factors:
- whether invoice is actually duplicate
- whether goods receipt exists
- whether mismatch is within tolerance

Human process:
1. inspect invoice, PO, and receipt
2. compare quantities, tax, and totals
3. decide accept, reject, or escalate for approval
4. request missing document if needed
5. close case

Agent requirements:
- perform three-way match logic
- avoid duplicate payment
- handle missing data correctly

Downstream events:
- payment error if bad approval
- delayed vendor payment penalty if unnecessary defer or escalate

### 8.3 Missing KYC for payout
Hidden factors:
- document validity
- risk tier
- deadline urgency

Human process:
1. identify missing or stale KYC item
2. request missing document
3. check returned document
4. approve payout or escalate

Agent requirements:
- never approve before required KYC is complete
- ask only for relevant missing items
- resume correctly after info arrives

Downstream events:
- compliance violation if payout released early
- SLA penalty if unnecessary waiting persists

### 8.4 Queue triage under SLA pressure
Hidden factors:
- future cost of delay for each case
- which cases are actually simple versus risky

Human process:
1. scan the queue
2. identify urgent and high-risk cases
3. choose work order
4. trade off local case depth against overall throughput

Agent requirements:
- plan at queue level, not only case level
- reduce missed SLAs while avoiding reckless decisions

Downstream events:
- backlog cascade if low-value work is prioritized over expiring high-risk work

---

## 9. State design

Use a fully structured internal state.

### 9.1 Top-level schema
```python
State = {
    "time": int,
    "queue": QueueState,
    "cases": dict[str, CaseState],
    "entities": {
        "customers": dict[str, CustomerState],
        "vendors": dict[str, VendorState],
        "orders": dict[str, OrderState],
        "invoices": dict[str, InvoiceState],
        "receipts": dict[str, ReceiptState],
    },
    "policies": PolicyState,
    "resources": ResourceState,
    "scheduled_events": list[ScheduledEvent],
    "metrics": MetricsState,
    "hidden": HiddenState,
    "audit_log": list[AuditEvent],
}
```

### 9.2 Case fields
Each case should contain:
- `case_id`
- `case_type`
- `status`
- `priority`
- `sla_deadline`
- `current_owner`
- `visible_summary`
- `required_checks`
- `visible_flags`
- `linked_records`
- `communication_state`
- `resolution_state`
- `hidden_factors_ref`

### 9.3 Hidden state examples
- true fraud score bucket
- true duplicate-invoice label
- whether vendor will respond within 2 hours or 24 hours
- whether missing document is valid after upload
- downstream loss magnitude if policy is violated

### 9.4 Why hidden state matters
CRMArena explicitly uses latent variables and interlinked business objects to make workflows more realistic.[5]

---

## 10. Scenario generation

Use seeded, compositional generation from atomic factors.

### 10.1 Template structure
Each template should define:
- workflow type
- visible setup
- hidden factors
- applicable policies
- required terminal conditions
- optional future events
- difficulty tier

### 10.2 Atomic factor library
Examples:
- amount tier: low, medium, high
- policy threshold crossed: yes or no
- missing information count: 0 to 3
- risk label: low, medium, high
- response latency class: fast, slow, none
- escalation queue load: light, medium, heavy
- SLA tightness: relaxed, standard, urgent

### 10.3 Generator approach
This should mirror the strengths of Tau2-bench's compositional task generator: diverse, verifiable tasks built from atomic components.[12]

### 10.4 Seeding rule
All stochastic choices must derive from a single scenario seed so that the same seed and action trace always yield the same outcome.

---

## 11. Transition engine

The simulator should be a rule engine plus a seeded event scheduler.

### 11.1 Step loop
```python
def step(state, action):
    validate(action)
    apply_preconditions(state, action)
    apply_state_transition(state, action)
    process_scheduled_events(state)
    compute_objective_reward(state, action)
    compute_shaping_reward(prev_state, state)
    render_observation(state)
    check_termination(state)
    return observation, reward, terminated, truncated, info
```

### 11.2 Transition rules
Rules should be explicit and testable. Examples:
- refund over threshold requires manager approval or escalation
- payout cannot be released when KYC required fields are incomplete
- case cannot close before mandatory notification is sent
- duplicate invoice approval creates future payment-loss event

### 11.3 Scheduled events
Examples:
- `chargeback_event` after bad refund approval
- `reopen_event` after premature closure or bad communication
- `sla_breach_event` when deadline passes
- `vendor_response_event` after requested document delay
- `queue_overload_event` when too many escalations accumulate

This is where first- and second-order effects live.

---

## 12. Reward design

The reward design should follow one principle:

Use sparse, business-aligned objective reward for evaluation and optional potential-based shaping for training.

### 12.1 Why separate objective and shaping reward
If you mix training heuristics into the benchmark score, you risk building an environment that rewards looking busy instead of being correct.

Keep two scores:
1. `objective_score`, reported in papers and dashboards
2. `train_reward`, used only inside learning runs

### 12.2 Objective reward components
For each case or episode, compute:

```text
objective_score =
+ correct_resolution_value
- critical_policy_violation_penalty
- downstream_loss_penalty
- sla_miss_penalty
- unnecessary_escalation_penalty
- tool_cost_penalty
- simulated_time_penalty
- backlog_penalty
```

Recommended starting weights:
- correct resolution: +10
- critical policy violation: -20
- downstream loss event: -12
- SLA miss: -5
- unnecessary escalation: -3
- invalid action: -2
- each tool call: -0.1
- each simulated minute beyond budget: -0.02
- reopen event: -8

These are starting values, not truths. Tune them after sanity checks against your oracle policy and a few adversarial policies.

### 12.3 Shaping reward
Use potential-based shaping because it preserves optimal policy under the standard result of Ng, Harada, and Russell.[9]

Form:
```text
R_total = R_objective + lambda * (gamma * Phi(s') - Phi(s))
```

Where `Phi(s)` can encode progress toward a valid completion state.

Safe potential features:
- evidence completeness score
- blocker reduction score
- mandatory notification completion
- queue risk reduction
- remaining required checks completed

Unsafe shaping patterns to avoid:
- rewarding approvals directly
- rewarding escalations directly
- rewarding number of steps taken
- rewarding any action that can be spammed without improving true state

### 12.4 Example potential function
```text
Phi(s) =
0.35 * required_checks_completed_ratio
+ 0.25 * evidence_completeness_ratio
+ 0.20 * communication_completion_ratio
+ 0.20 * queue_risk_reduction_ratio
```

### 12.5 Reward anti-gaming rules
Before training, test at least four pathological policies:
- always approve
- always escalate
- always gather more information
- shortest-path greedy closer

If any pathological policy scores unexpectedly well, the reward needs revision.

---

## 13. Grader design

The grader is the core of the benchmark. Make it deterministic and layered.

### 13.1 Primary grader: state-based outcome grader
This is the main source of truth.

Inputs:
- initial scenario
- final state
- full event log
- scenario specification

Outputs:
- pass or fail
- objective score
- reason codes
- per-check breakdown

Checks:
- final resolution valid under policy
- all mandatory actions completed
- no critical compliance violation
- no unresolved blockers remain
- all required downstream constraints satisfied

This follows the strongest pattern from Tau-bench and OSWorld: compare actual environment state to expected outcome using programmatic, execution-based evaluation.[2][4]

### 13.2 Secondary grader: trajectory invariant grader
A correct final state can still be reached through an invalid process. Google Cloud calls this silent failure.[10]

So add invariant checks such as:
- policy lookup required before high-value approval
- KYC must be complete before payout action
- closure requires customer notification
- escalate only to allowed queue types
- role-restricted data cannot be accessed without permission

Represent these as finite checks over the trace, not as free-form language judgments.

### 13.3 Reliability grader
Use repeated trials per task and compute:
- `pass_rate`
- `mean_objective_score`
- `pass^k`
- variance over repeated runs

Tau-bench proposes pass^k specifically to measure consistency across repeated trials.[4]

### 13.4 Efficiency grader
Track:
- steps per solved case
- tool calls
- simulated time consumed
- unnecessary actions
- number of queue switches

OSWorld-Human shows that raw success is not enough because agents can remain far less efficient than humans, taking 1.4x to 2.7x more steps than necessary.[13]

### 13.5 Safety and compliance grader
Check:
- role-based permission violations
- forbidden disclosure
- refusal on unsafe requests
- exposure of hidden or sensitive fields

This aligns with SAP's emphasis on access control, reliability, and compliance as enterprise-specific evaluation needs.[7]

### 13.6 Optional LLM-based grader
Only use an LLM judge for narrow, secondary checks such as:
- whether a free-text customer message is clear and polite
- whether an escalation summary is concise and complete

Do not use an LLM judge for core correctness if the state can be checked programmatically. Google Cloud explicitly recommends code-based evals for objective checks and LLM-as-a-judge only after alignment to human evaluation.[10]

### 13.7 Example grader output
```json
{
  "pass": false,
  "objective_score": -17.6,
  "reason_codes": [
    "POLICY_VIOLATION_REFUND_THRESHOLD",
    "MISSING_CUSTOMER_NOTIFICATION",
    "DOWNSTREAM_REOPEN"
  ],
  "checks": {
    "final_resolution_valid": false,
    "mandatory_notification_done": false,
    "sla_met": true,
    "no_critical_violation": false
  },
  "efficiency": {
    "tool_calls": 12,
    "steps": 14,
    "simulated_minutes": 37
  }
}
```

---

## 14. What a human workflow looks like

This section matters because the environment should mirror how competent human operators work.

### 14.1 Refund exception example
Human:
1. open case
2. inspect order
3. inspect customer history
4. check refund threshold policy
5. decide approve, reject, or escalate
6. send customer notification
7. close case

Agent must learn:
- when to gather more evidence
- when the case is already decidable
- when it is unsafe to resolve autonomously
- how to sequence message and closure correctly

### 14.2 Duplicate invoice example
Human:
1. open exception
2. inspect invoice, PO, and receipt
3. check mismatch tolerance policy
4. decide resolve, reject, or escalate
5. request missing documentation if needed
6. close case

Agent must learn:
- that invoice correctness depends on linked records, not a single page
- that requesting documentation is better than guessing
- that bad approval may look fine now and fail later

### 14.3 Missing KYC example
Human:
1. inspect payout request
2. determine which KYC item is missing or stale
3. request only the missing item
4. wait or follow up
5. verify new document
6. release payout or escalate

Agent must learn:
- not to act before the blocker is removed
- not to ask for irrelevant documents
- how to resume a partially completed workflow

### 14.4 Queue triage example
Human:
1. scan queue and deadlines
2. identify urgent, high-risk, or simple wins
3. choose the next case deliberately
4. balance local accuracy and global throughput

Agent must learn:
- that the best next case depends on queue context
- that working on the wrong case now can create broader future cost

---

## 15. Dataset and task split design

### 15.1 Splits
Create at least three splits:
- `train`: broad scenario coverage with many seeds
- `dev`: held-out seed combinations for reward and policy iteration
- `test`: held-out template combinations and edge cases

### 15.2 Generalization dimensions
Hold out some combinations on purpose:
- new threshold values
- new case-type mixtures
- new queue-load patterns
- new hidden-factor distributions
- new combinations of blockers and time pressure

### 15.3 Difficulty tiers
- Tier 1: single-case, no scheduled downstream events
- Tier 2: single-case with hidden risk and one downstream event
- Tier 3: multi-case queue with limited human escalation capacity
- Tier 4: role constraints, compliance checks, and cascading queue effects

---

## 16. Baselines

Ship the environment with simple baselines.

### 16.1 Random baseline
Useful for sanity checks.

### 16.2 Rule-based baseline
Handwritten if-then policy that is valid but conservative.

### 16.3 Oracle baseline
A privileged solver that reads hidden state. Use only to estimate ceiling and validate scenario correctness, not as a public agent baseline.

### 16.4 Imitation baseline
Optional behavior cloning baseline from oracle or scripted traces.

### 16.5 Why baselines matter
A good benchmark should separate:
- impossible tasks from hard tasks
- reward bugs from model bugs
- simulator bugs from policy failures

---

## 17. Evaluation harness

### 17.1 Trial artifact
Save, for every trial:
- seed
- scenario id
- config
- action log
- observation snapshots
- final state hash
- reward breakdown
- grader outputs

### 17.2 Aggregate metrics
Report at least:
- pass rate
- mean objective score
- pass^k over repeated trials
- critical violation rate
- efficiency metrics
- per-workflow breakdown
- per-difficulty breakdown

### 17.3 Regression suite
Maintain a fixed suite of named edge cases:
- policy-threshold trap
- missing-notification trap
- duplicate-invoice false positive
- escalate-spam trap
- gather-forever trap

Anthropic recommends using evaluation suites over the lifecycle of the agent, with multiple graders per task and repeated trials for consistency.[11]

---

## 18. Recommended repository structure

```text
opsarena/
  README.md
  pyproject.toml
  opsarena/
    __init__.py
    env.py
    state.py
    actions.py
    observations.py
    transitions.py
    scheduler.py
    rewards.py
    graders.py
    scenario.py
    render.py
    policies.py
    logging.py
    constants.py
  configs/
    reward_weights.yaml
    grader_weights.yaml
    env_defaults.yaml
  data/
    scenario_templates/
      refund/
      invoice/
      kyc/
      triage/
    policies/
    golden_cases/
  baselines/
    random_policy.py
    rules_policy.py
    oracle_policy.py
  scripts/
    play_cli.py
    run_eval.py
    generate_scenarios.py
    export_dataset.py
  tests/
    test_transitions.py
    test_rewards.py
    test_graders.py
    test_scenarios.py
```

---

## 19. Recommended implementation order

### Phase 1
- define schemas for state, actions, events, and grader output
- implement one workflow: refund exception
- implement deterministic state grader
- implement CLI loop

### Phase 2
- add duplicate invoice and missing KYC
- add event scheduler for delayed effects
- add rule-based baseline
- add reward breakdown and JSONL logging

### Phase 3
- add queue triage and backlog dynamics
- add pass^k evaluation harness
- add held-out test split
- add policy and role constraints

### Phase 4
- optional user simulator
- optional dual-control mode
- optional browser wrapper around API layer

---

## 20. Design decisions to keep fixed

1. Primary benchmark score is based on objective state, not language.
2. Renderer, simulator, reward function, and grader are separate modules.
3. All randomness is seed-controlled.
4. Shaping reward is never reported as benchmark score.
5. Free-text actions are banned in v0.1.
6. Every failure should yield reason codes.
7. Multi-case queue behavior is a first-class target, not an afterthought.

---

## 21. Open extensions after v0.1

- role-based access control cases
- confidentiality and refusal tests
- dual-control tasks where a user also acts in the world, inspired by Tau2-bench.[12]
- browser wrapper for the same underlying environment
- offline trajectory dataset for imitation or preference learning
- cost-aware human handoff model
- stochastic user response models with fixed seeds
- curriculum learning over difficulty tiers

---

## 22. Resource appendix

[1] The BrowserGym Ecosystem for Web Agent Research, arXiv:2412.05467. Unifies web-agent benchmarks with gym-like observation and action spaces.

[2] OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments, arXiv:2404.07972. Real computer environment with execution-based evaluation.

[3] OdysseyBench: Evaluating LLM Agents on Long-Horizon Complex Office Application Workflows, arXiv:2508.09124. Long-horizon office workflows across multiple apps.

[4] Tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains, arXiv:2406.12045. Goal-state evaluation with pass^k reliability metric.

[5] CRMArena: Understanding the Capacity of LLM Agents to Perform Professional CRM Tasks in Realistic Environments, arXiv:2411.02305. Business-object schemas, latent variables, and low current performance.

[6] CRMArena-Pro: Holistic Assessment of LLM Agents Across Diverse Business Scenarios and Interactions, arXiv:2505.18878. Multi-turn business evaluation with confidentiality assessments.

[7] Evaluation and Benchmarking of LLM Agents, SAP KDD 2025 Tutorial. Enterprise evaluation gaps: access control, reliability, dynamic long-horizon interactions, compliance.

[8] OdysseyArena: Benchmarking Large Language Models For Long-Horizon, Active and Inductive Interactions, arXiv:2602.05843. Argues that benchmarks should test inductive discovery of latent transition laws.

[9] Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping, Ng, Harada, Russell. Basis for potential-based reward shaping.

[10] A methodical approach to agent evaluation: Building a robust quality gate, Google Cloud Blog, 2025. Strong argument for outcome, trajectory, and trust layers of evaluation.

[11] Demystifying evals for AI agents, Anthropic Engineering, 2026. Useful definitions for task, trial, grader, transcript, outcome, harness, and evaluation suite.

[12] Tau2-bench: Evaluating Conversational Agents in a Dual-Control Environment, arXiv:2506.07982. Shows value of shared-state, tool-constrained user simulation and compositional task generation.

[13] OSWorld-Human: Benchmarking the Efficiency of Computer-Use Agents, arXiv:2506.16042. Shows that success alone is not enough because agents can remain much less efficient than humans.
