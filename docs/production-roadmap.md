# OpsArena Production Roadmap

This roadmap covers the workflows that make OpsArena useful for real ecommerce operations, risk, and finance teams.

## What is implemented

OpsArena currently supports training agents on:

- refund and dispute handling (Phase 1)
- invoice exception handling (Phase 1)
- KYC remediation (Phase 1)
- shared-queue basics like claim, route, prioritize, and follow-up (Phase 1)
- supervisor and QA loops including bulk assignment, queue rebalancing, and quality review (Phase 2)
- queue shock events: arrival waves and staffing drops (Phase 2)
- full dispute lifecycle including pre-dispute alerts, representment, and pre-arbitration (Phase 3)
- payout risk controls: freeze, reserve, delay (Phase 3)
- merchant risk monitoring: dispute ratio, fraud ratio, monitoring program status (Phase 3)
- compliance investigations: sanctions screening, EDD, beneficial owner review (Phase 4)
- regulatory reporting: OFAC reports with deadlines (Phase 4)
- payment freezing for compliance holds (Phase 4)
- AP payment-run recovery: revised invoices, PO changes, stop payments, vendor refunds, credit memo application, write-offs (Phase 5)

## Phase 1: core workflows (IMPLEMENTED)

### Goal
Establish the foundational case-handling workflows.

### What it covers

- Refund and dispute case resolution
- Invoice exception handling with three-way match
- KYC remediation with document review
- Queue management: claim, route, prioritize, follow-up, SLA tracking
- Core actions: approve, reject, escalate, defer, close, reopen
- Information gathering: list queue, open case, view record, query policy, search cases, inspect audit
- Communication: request info, send message, log internal note

## Phase 2: supervisor and QA loops (IMPLEMENTED)

### Goal
Make the environment useful for training agents that do more than solve one case at a time.

### Actions added

- `bulk_assign` -- assign up to 10 cases to an agent at once
- `bulk_route` -- route up to 10 cases to a target queue at once
- `rebalance_queue` -- redistribute unassigned cases across an agent pool
- `send_to_qa` -- submit a case for quality review
- `fail_qa` -- fail a QA-reviewed case and send back for rework
- `approve_qa` -- pass a QA-reviewed case

### State fields added

- `qa_status`
- `qa_owner`
- `queue_backlog_age`
- `unassigned_count`
- `exception_queue_size`
- `agent_capacity`

### Delayed events added

- `qa_sample_selected` -- a resolved case is sampled for QA review
- `qa_failed` -- a QA review finds defects
- `rework_due` -- a failed QA case must be reworked
- `arrival_wave` -- a spike in incoming case volume
- `staffing_drop` -- a reduction in available agent capacity

### Why it matters in production

Real teams care about:

- backlog health
- missed follow-ups
- rework rate
- bad handoffs
- queue overload

This phase makes the environment useful for supervisor-style decision making, not just analyst-style decision making.

## Phase 3: dispute lifecycle and payout-risk workflows (IMPLEMENTED)

### Goal
Model the real payments risk lifecycle instead of a simplified "approve vs reject vs defend" flow.

### Actions added

- `challenge_dispute` -- challenge a dispute via representment
- `refund_pre_dispute_alert` -- refund proactively on a pre-dispute alert
- `resolve_prearbitration` -- respond to a pre-arbitration notice (accept or contest)
- `freeze_payouts` -- freeze merchant payouts
- `unfreeze_payouts` -- unfreeze merchant payouts
- `set_reserve_percent` -- set a rolling reserve on payouts
- `clear_reserve` -- clear a previously set reserve
- `set_payout_delay_days` -- set a payout delay in days

### State fields added

- `dispute_stage` -- tracks the dispute through inquiry, chargeback, evidence submitted, pre-arbitration, won, lost, finalized
- `reserve_percent` -- current rolling reserve percentage
- `payout_delay_days` -- current payout delay
- `merchant_risk_level` -- normal, elevated, high, critical
- `dispute_ratio` -- 30-day merchant dispute ratio
- `monitoring_program_status` -- normal, warning, breached

### Delayed events added

- `inquiry_escalates_to_chargeback` -- a pre-dispute inquiry becomes a formal chargeback
- `prearbitration_received` -- a pre-arbitration notice arrives after representment
- `reserve_release_due` -- a reserve auto-release timer expires
- `monitoring_threshold_breached` -- merchant risk metrics cross a threshold

### Why it matters in production

This makes OpsArena useful for:

- merchant risk operations
- payment operations
- platform trust and safety

The agent must trade off:

- immediate customer or merchant pain
- financial loss (dispute fees, chargeback costs)
- downstream network risk (monitoring programs, processor fees)

## Phase 4: compliance investigation workflows (IMPLEMENTED)

### Goal
Move beyond document-only KYC into real marketplace compliance operations.

### Actions added

- `run_sanctions_screen` -- run sanctions and watchlist screening
- `start_edd_review` -- initiate Enhanced Due Diligence review
- `review_beneficial_owner` -- review beneficial ownership information
- `request_field_correction` -- ask the applicant to correct a specific field
- `file_ofac_report` -- file a required OFAC or regulatory report
- `freeze_payments` -- freeze all payments for compliance reasons

### State fields added

- `sanctions_status` -- not_started, clear, potential_match, confirmed_match
- `edd_status` -- not_started, not_required, in_progress, awaiting_response, cleared
- `beneficial_owner_status` -- not_started, pending_review, needs_correction, verified, rejected
- `screening_match_confidence` -- confidence score for sanctions screening matches
- `report_due_at` -- deadline for filing a required regulatory report

### Delayed events added

- `sanctions_false_positive_cleared` -- a potential sanctions match is cleared as a false positive
- `edd_response_due` -- an EDD response deadline arrives
- `report_deadline_missed` -- an OFAC reporting deadline passes without a filing

### Why it matters in production

This makes the environment useful to:

- compliance teams
- marketplace onboarding teams
- trust and safety teams

It introduces strong hard-gate failures: approving a sanctioned entity or missing a reporting deadline are catastrophic errors, which is important for benchmark realism.

## Phase 5: AP payment-run and recovery workflows (IMPLEMENTED)

### Goal
Model the finance cleanup work that happens after invoice review.

### Actions added

- `request_revised_invoice` -- ask the vendor to resubmit a corrected invoice
- `request_po_change` -- request a purchase order amendment
- `remove_from_payment_batch` -- pull an invoice out of a scheduled payment batch
- `stop_payment` -- stop a payment already in progress
- `record_vendor_refund` -- record that a vendor returned funds
- `apply_credit_memo` -- apply a received credit memo to an invoice
- `write_off_small_balance` -- write off a small remaining balance

### State fields added

- `payment_batch_status` -- not_scheduled, scheduled, in_progress, completed, stopped
- `vendor_response_status` -- not_requested, awaiting, received, overdue
- `recovery_status` -- not_needed, in_progress, partial, complete
- `duplicate_status` -- suspected, confirmed, false_positive, already_paid
- `po_change_status` -- not_requested, pending_approval, approved, denied

### Delayed events added

- `vendor_sends_revised_invoice` -- the vendor responds with a corrected invoice
- `po_change_approved` -- a PO change request is approved
- `stop_payment_confirmed` -- a stop payment request is confirmed
- `refund_received` -- a vendor refund arrives

### Why it matters in production

This makes OpsArena useful for:

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
- post-payment recovery (stop payments, vendor refunds, write-offs)

### 3. Compliance operations

- KYC remediation
- sanctions review
- EDD investigation
- beneficial owner verification
- OFAC reporting
- payout restrictions

### 4. Queue leadership and team operations

- backlog management
- bulk routing
- QA sampling
- staffing shock handling

## Implementation order

Phases were implemented in this order:

1. Phase 1: core workflows (refund, invoice, KYC, queue management)
2. Phase 2: supervisor + QA loops + queue shock events
3. Phase 3: dispute lifecycle + payout reserves + merchant risk monitoring
4. Phase 4: sanctions screening + EDD + beneficial owner review + OFAC reporting
5. Phase 5: AP payment-run recovery (revised invoices, PO changes, stop payments, vendor refunds, credit memos, write-offs)
