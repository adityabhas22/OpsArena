# OpsArena Rewards

OpsArena has two reward layers:

1. `objective_reward` -- the real business-facing score
2. `shaping_reward` -- a training-only helper signal

Only the objective score should matter for benchmark reporting.

## What the environment rewards

At a high level, the agent is rewarded for:

- resolving cases correctly
- following policy
- avoiding compliance failures
- protecting SLAs
- reducing downstream losses
- handling queue work efficiently
- managing dispute economics optimally
- sequencing compliance checks correctly
- recovering value from invoice exceptions

## What the environment penalizes

The agent is penalized for:

- approving fraud or duplicate payments
- approving incomplete KYC
- missing required notifications
- invalid actions
- unnecessary escalations
- overdue follow-ups
- letting cases breach SLA
- fighting disputes that should be accepted
- failing to freeze payouts when merchant risk is critical
- approving sanctioned entities without filing required reports
- missing stop-payment windows on duplicate invoices

## Workflow-specific intuition

### Refund / dispute

Good behavior:

- check policy before approving
- gather only useful evidence
- reject real fraud
- accept low-value weak disputes when it is cheaper than fighting them

Bad behavior:

- approve likely fraud
- fight every dispute regardless of amount or evidence
- close without customer communication

### Dispute lifecycle (Phase 3)

Good behavior:

- refund proactively on pre-dispute alerts when the dispute is not worth fighting
- challenge disputes only when evidence is strong
- set reserves and payout delays proportional to merchant risk level
- freeze payouts when monitoring program thresholds are breached
- clear reserves once risk conditions normalize

Bad behavior:

- challenge disputes with weak or no evidence
- freeze payouts on normal-risk merchants
- unfreeze payouts while the merchant is still in breach
- ignore pre-arbitration deadlines

### AP / invoice

Good behavior:

- confirm match status
- use payment holds while facts are incomplete
- request credit memos when variance recovery is needed
- use secondary approval for exceptions

Bad behavior:

- approve duplicate or unresolved invoices
- release holds before blockers are cleared
- close while follow-up or approval is still pending

### AP recovery (Phase 5)

Good behavior:

- remove invoices from payment batches before they complete when duplicates are found
- issue stop payments promptly when batches have already completed
- request revised invoices or PO changes when originals have errors
- apply credit memos and record vendor refunds to complete recovery
- write off small balances below threshold instead of pursuing expensive recovery

Bad behavior:

- let duplicate payments complete without intervention
- miss the stop-payment window
- fail to track vendor responses through to completion
- pursue expensive recovery for amounts below the write-off threshold

### KYC

Good behavior:

- request the right missing document
- avoid approving incomplete or invalid verification
- keep payout restrictions active until the case is actually cleared

Bad behavior:

- approve invalid documents
- skip review steps

### Compliance investigation (Phase 4)

Good behavior:

- run sanctions screening before making any approval decision
- start EDD review when the risk profile requires it
- review beneficial ownership before clearing high-risk applicants
- freeze payments immediately when a confirmed sanctions match is found
- file OFAC reports before the reporting deadline
- request field corrections for incomplete applications instead of rejecting outright

Bad behavior:

- approve without completing sanctions screening
- skip EDD on high-risk entities
- approve with unresolved beneficial ownership issues
- miss OFAC reporting deadlines
- freeze payments on cleared false positives

## Grader breakdown

The episode grader reports:

- `outcome`: was the final state correct?
- `process`: did the agent follow important workflow rules?
- `efficiency`: did it manage time and queue pressure well?

Current process checks include:

- policy review before approval
- KYC document review before KYC approval
- notification before close when required
- open case before decisive actions
- limited repeated info requests
- secondary approval before exception approval
- follow-up discipline
- sanctions screening before compliance approval
- freeze on confirmed sanctions match
- stop payment discipline on duplicate invoices in completed batches

## Reward components

### Per-workflow rewards

Each workflow type has its own calibrated reward function:

- **Refund reward**: decision quality (correct approval/rejection), evidence gathering, policy compliance, customer notification
- **Invoice reward**: match quality (catch duplicates, approve valid), cash flow impact, vendor relationship health
- **KYC reward**: compliance outcome (extreme asymmetry: false negative is ~100x worse than false positive), document request accuracy, speed
- **Triage reward**: throughput, priority-weighted completion, utilization, SLA prevention

### Time pressure (non-linear SLA)

A sigmoid-based time pressure function that creates realistic urgency:
- Negligible pressure with plenty of time remaining
- Sharp pressure rise when less than 20% of SLA time remains
- Priority-scaled: P1 breaches are 2x as costly as P3

### Value of Information (VOI)

Evidence-gathering actions are rewarded based on their marginal improvement to decision confidence:
- Diminishing returns prevent evidence-gathering spam
- Relevance gating: only evidence relevant to the case type provides value
- Redundant queries are penalized
- VOI is a shaping reward only (does not affect objective score)

### Asymmetric error costs

False positives and false negatives have calibrated, asymmetric costs:
- Refund fraud: FN/FP ratio ~3.4 (approving fraud is 3.4x worse than blocking a legitimate customer)
- Invoice duplicate: FN/FP ratio ~18.5 (approving a duplicate is 18.5x worse than rejecting a valid invoice)
- KYC compliance: FN/FP ratio ~97 (approving non-compliant is ~100x worse than blocking compliant)
- Queue triage: FN/FP ratio ~13 (under-prioritizing urgent is 13x worse than over-prioritizing low)

### Queue-level portfolio reward

Captures global queue health beyond individual case outcomes:
- Throughput efficiency
- Priority-weighted completion
- SLA portfolio health (quadratic: going from 5% breach to 0% is much more valuable than 50% to 45%)
- Risk concentration (Herfindahl index of remaining unresolved risk)
- Backlog health
- Assignment health and escalation overload

### Cascade rewards

Bad decisions create downstream events that affect other cases. Six cascade channels are modeled:

1. **Escalation cascade**: escalation queue overload delays all cases in the escalation queue
2. **Fraud rate cascade**: approved fraud increases chargeback ratio, raising processor fees on all transactions
3. **Compliance cascade**: compliance failures increase regulatory scrutiny and processing time for all subsequent KYC cases
4. **Cash flow cascade**: duplicate payment approvals reduce available cash, delaying vendor payments across the board
5. **QA cascade**: QA pass/fail outcomes affect rework rates and quality scores
6. **Dispute economics cascade** (Phase 3): payout risk actions (freeze, reserve, delay) are rewarded or penalized based on actual merchant risk level, monitoring program status, and evidence strength
7. **KYC compliance cascade** (Phase 4): compliance investigation actions (sanctions screening, EDD, beneficial owner review, OFAC reporting, payment freezing) are rewarded based on whether they were the correct action given the hidden ground truth

### Potential-based shaping

Training-only shaping rewards use potential functions (Phi) that measure workflow progress:
- Refund Phi: evidence gathering, checks completed, policy checked, customer notified, dispute stage progress
- Invoice Phi: three-way match progress, evidence, policy checked, vendor communication, credit memo and approval status
- KYC Phi: document completeness, compliance policy review, communication, risk assessment
- Triage Phi: resolution progress, SLA health, escalation queue health, assignment health, overdue follow-ups

Shaping preserves the optimal policy per Ng, Harada, Russell (1999).

## Delayed consequences

Some good or bad outcomes arrive later:

- chargebacks can reopen cases
- dispute submissions can end in win, loss, or pre-arbitration
- pre-arbitration notices arrive after representment decisions
- reserve releases trigger after a timer expires
- monitoring program thresholds can be breached as dispute ratios change
- credit memos arrive after vendor delay
- follow-ups can become overdue
- secondary approvals return asynchronously
- sanctions screening can return false-positive clearance
- EDD responses arrive after a delay
- OFAC report deadlines can be missed
- vendor responses (revised invoices, refunds) arrive asynchronously
- PO change approvals return after review
- stop payment confirmations arrive with a delay

This is intentional: the agent should learn to act for the final business outcome, not just the next reward tick.
