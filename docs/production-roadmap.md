# OpsArena Production Roadmap

This roadmap focuses on workflows that would make OpsArena more useful for real ecommerce operations, risk, and finance teams.

## What we already have

Today the environment can train agents on:

- refund and dispute handling
- invoice exception handling
- KYC remediation
- shared-queue basics like claim, route, prioritize, and follow-up

That is enough to study analyst behavior.

## What is still most valuable to add

The next big step is modeling how a real operations organization behaves under pressure:

- managers and supervisors
- quality review and reopens
- richer dispute stages
- payout / reserve controls
- compliance investigations
- payment-run and recovery workflows

## Phase 2: supervisor and QA loops

### Goal
Make the environment useful for training agents that do more than solve one case at a time.

### Add these actions

- `bulk_assign`
- `bulk_route`
- `rebalance_queue`
- `send_to_qa`
- `fail_qa`
- `approve_qa`

### Add these state fields

- `qa_status`
- `qa_owner`
- `queue_backlog_age`
- `unassigned_count`
- `exception_queue_size`
- `agent_capacity`

### Add these delayed events

- `qa_sample_selected`
- `qa_failed`
- `rework_due`
- `arrival_wave`
- `staffing_drop`

### Why it matters in production

Real teams care about:

- backlog health
- missed follow-ups
- rework rate
- bad handoffs
- queue overload

This phase makes the environment useful for supervisor-style decision making, not just analyst-style decision making.

## Phase 3: richer dispute and payout-risk workflows

### Goal
Model the real payments risk lifecycle instead of a simplified “approve vs reject vs defend” flow.

### Add these actions

- `challenge_dispute`
- `refund_pre_dispute_alert`
- `resolve_prearbitration`
- `freeze_payouts`
- `unfreeze_payouts`
- `set_reserve_percent`
- `clear_reserve`
- `set_payout_delay_days`

### Add these state fields

- `dispute_stage`
- `reserve_percent`
- `payout_delay_days`
- `merchant_risk_level`
- `dispute_ratio`
- `monitoring_program_status`

### Add these delayed events

- `inquiry_escalates_to_chargeback`
- `prearbitration_received`
- `reserve_release_due`
- `monitoring_threshold_breached`

### Why it matters in production

This would make OpsArena useful for:

- merchant risk operations
- payment operations
- platform trust and safety

It also makes the reward signal more realistic because the agent must trade off:

- immediate customer or merchant pain
- financial loss
- downstream network risk

## Phase 4: deeper compliance workflows

### Goal
Move beyond document-only KYC into real marketplace compliance operations.

### Add these actions

- `run_sanctions_screen`
- `start_edd_review`
- `review_beneficial_owner`
- `request_field_correction`
- `file_ofac_report`
- `freeze_payments`

### Add these state fields

- `sanctions_status`
- `edd_status`
- `beneficial_owner_status`
- `screening_match_confidence`
- `report_due_at`

### Add these delayed events

- `sanctions_false_positive_cleared`
- `edd_response_due`
- `report_deadline_missed`

### Why it matters in production

This would make the environment more useful to:

- compliance teams
- marketplace onboarding teams
- trust and safety teams

It also introduces strong hard-gate failures, which is good for benchmark realism.

## Phase 5: AP payment-run and recovery workflows

### Goal
Model the finance cleanup work that happens after invoice review.

### Add these actions

- `request_revised_invoice`
- `request_po_change`
- `remove_from_payment_batch`
- `stop_payment`
- `record_vendor_refund`
- `apply_credit_memo`
- `write_off_small_balance`

### Add these state fields

- `payment_batch_status`
- `vendor_response_status`
- `recovery_status`
- `duplicate_status`
- `remittance_change_risk`

### Add these delayed events

- `vendor_sends_revised_invoice`
- `po_change_approved`
- `stop_payment_confirmed`
- `refund_received`

### Why it matters in production

This would make OpsArena useful for:

- AP operations teams
- procurement-finance handoff workflows
- payment-run exception training

## Best production-facing training use cases

If someone wanted to use OpsArena as a serious benchmark or internal simulator, the most useful bundles would be:

### 1. Risk and dispute operations

- dispute acceptance vs defense
- pre-dispute triage
- reserve and payout actions
- merchant-risk escalation

### 2. AP exception operations

- invoice mismatch handling
- credit memo recovery
- approval chains
- payment-run blocking and release

### 3. Compliance operations

- KYC remediation
- sanctions review
- EDD investigation
- payout restrictions

### 4. Queue leadership and team operations

- backlog management
- bulk routing
- QA sampling
- staffing shock handling

## Recommended implementation order

1. Phase 2: supervisor + QA
2. Phase 3: dispute lifecycle + payout reserves
3. Phase 4: sanctions + EDD
4. Phase 5: AP payment-run recovery

This order gives the biggest realism gain while keeping the environment explainable and maintainable.
