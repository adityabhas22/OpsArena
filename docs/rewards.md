# OpsArena Rewards

OpsArena has two reward layers:

1. `objective_reward` — the real business-facing score
2. `shaping_reward` — a training-only helper signal

Only the objective score should matter for benchmark reporting.

## What the environment rewards

At a high level, the agent is rewarded for:

- resolving cases correctly
- following policy
- avoiding compliance failures
- protecting SLAs
- reducing downstream losses
- handling queue work efficiently

## What the environment penalizes

The agent is penalized for:

- approving fraud or duplicate payments
- approving incomplete KYC
- missing required notifications
- invalid actions
- unnecessary escalations
- overdue follow-ups
- letting cases breach SLA

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

### KYC

Good behavior:

- request the right missing document
- avoid approving incomplete or invalid verification
- keep payout restrictions active until the case is actually cleared

Bad behavior:

- approve invalid documents
- skip review steps

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

## Delayed consequences

Some good or bad outcomes arrive later:

- chargebacks can reopen cases
- dispute submissions can end in win, loss, or pre-arbitration
- credit memos arrive after vendor delay
- follow-ups can become overdue
- secondary approvals return asynchronously

This is intentional: the agent should learn to act for the final business outcome, not just the next reward tick.
