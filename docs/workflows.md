# OpsArena Workflows

This document shows the kinds of workflows OpsArena models.

## 1. Refund and dispute workflow

Example:

1. Open a refund/dispute case
2. Review order, customer, payment, shipping, and dispute data
3. Read the refund or dispute policy
4. Decide whether to:
   - approve refund
   - reject refund
   - execute refund
   - accept dispute
   - submit dispute evidence
5. Notify the customer
6. Close the case
7. Handle delayed outcomes like chargebacks or pre-arbitration

What the RL agent learns:

- when to gather more evidence
- when a dispute is worth fighting
- when a low-value dispute should simply be accepted
- how delayed outcomes change the right decision

## 2. AP / invoice workflow

Example:

1. Open an invoice exception case
2. Review invoice and purchase order
3. Record a three-way match result:
   - matched
   - variance
   - duplicate
   - missing receipt
4. If needed, place a payment hold
5. Request missing receipt or request a vendor credit memo
6. Send the case for secondary approval if it is out of policy
7. Release the hold only after blockers clear
8. Approve or reject
9. Close the case

What the RL agent learns:

- not to approve invoices too early
- how to use holds to manage risk
- when to recover value with a credit memo
- when approval friction is necessary

## 3. KYC workflow

Example:

1. Open merchant verification case
2. Review KYC record and policy
3. Request the exact missing document or trigger reverification
4. Wait for delayed document arrival
5. Review the submitted information
6. Approve, reject, or request resubmission
7. Keep payout restrictions in place until safe to clear

What the RL agent learns:

- how to work with partial information
- how to avoid compliance failures
- how to balance speed against risk

## 4. Queue and portfolio workflow

Example:

1. Claim a case from a shared queue
2. Work it or route it
3. Schedule a follow-up if waiting on someone else
4. Return it to queue if it no longer belongs with the current analyst
5. Reorder work under SLA pressure
6. Advance time and respond to overdue follow-ups or delayed approvals

What the RL agent learns:

- work ownership discipline
- next-touch management
- queue-level tradeoffs instead of single-case tunnel vision

## 5. Supervisor and QA workflow

Example:

1. Review queue health: backlog age, unassigned count, exception queue size
2. Bulk-assign cases to available agents based on capacity
3. Bulk-route cases to specialist queues when needed
4. Rebalance workload across an agent pool using strategies like sla_priority, oldest, or amount
5. Handle queue shock events: arrival waves that spike case volume, or staffing drops that reduce capacity
6. Send resolved cases to QA sampling
7. Review QA results: approve passing cases, fail and send back for rework
8. Manage rework loops when QA fails a case

What the RL agent learns:

- backlog health management under pressure
- when to rebalance vs when to let agents self-serve
- how to handle staffing drops and arrival waves
- quality control discipline through QA sampling
- minimizing rework rate without skipping QA

## 6. Dispute lifecycle workflow

Example:

1. Receive a pre-dispute alert (early fraud warning, inquiry, or RDR alert)
2. Decide whether to refund proactively to avoid a chargeback
3. If a chargeback arrives, decide whether to accept or challenge
4. If challenging, gather and submit dispute evidence before the representment deadline
5. Handle the dispute outcome: win, loss, or pre-arbitration
6. If pre-arbitration is received, decide whether to accept or contest
7. Apply payout risk controls based on merchant risk level:
   - freeze payouts for critical-risk merchants
   - set rolling reserves for high-risk merchants
   - add payout delays for elevated-risk merchants
8. Monitor merchant dispute and fraud ratios against monitoring program thresholds
9. Clear reserves and unfreeze payouts once risk conditions normalize

What the RL agent learns:

- when to refund proactively on a pre-dispute alert vs wait for a chargeback
- how to evaluate dispute evidence strength before challenging
- how to manage the full dispute lifecycle including pre-arbitration
- how to apply proportional payout risk controls (freeze, reserve, delay)
- how merchant-level risk metrics (dispute ratio, fraud ratio, monitoring program status) affect decisions
- the economic tradeoffs between dispute fees, chargeback costs, and reserve costs

## 7. Compliance investigation workflow

Example:

1. Open a KYC or compliance case
2. Run a sanctions screening check
3. If a potential match is found, evaluate match confidence
4. Start an Enhanced Due Diligence (EDD) review if required by risk profile
5. Review beneficial ownership information
6. Request field corrections from the applicant if documents are incomplete
7. Freeze payments if a confirmed sanctions match is found
8. File an OFAC report if required, before the reporting deadline
9. Wait for EDD responses and sanctions clearance before making a final decision
10. Approve only when sanctions are clear, EDD is cleared, beneficial ownership is verified, and reporting is complete

What the RL agent learns:

- how to sequence compliance checks correctly (sanctions before EDD before approval)
- when to freeze payments vs when it is safe to leave them open
- how to avoid catastrophic false negatives (approving a sanctioned entity)
- how to manage regulatory deadlines (OFAC report due dates, EDD response windows)
- the difference between false positives that need clearance and confirmed matches that need enforcement
- how compliance failures cascade into increased scrutiny for all subsequent KYC cases

## 8. AP payment-run and recovery workflow

Example:

1. Open an invoice case where the payment batch is already scheduled or in progress
2. If a duplicate or mismatch is found after the invoice entered a payment batch:
   - remove it from the payment batch before the batch completes
   - if the batch already completed, issue a stop payment
3. Request a revised invoice from the vendor if the original has errors
4. Request a PO change if the purchase order itself needs amendment
5. Wait for vendor responses (revised invoice, refund, credit memo)
6. Record vendor refunds when received
7. Apply credit memos to offset outstanding balances
8. Write off small remaining balances below the write-off threshold
9. Track recovery status from not_needed through in_progress, partial, to complete

What the RL agent learns:

- how to intervene in payment batches at different stages (scheduled vs in-progress vs completed)
- when to use stop payment vs remove from batch
- how to coordinate vendor communication (revised invoices, PO changes, credit memos)
- when to write off small balances instead of pursuing full recovery
- how to track and complete multi-step recovery workflows
- the time-sensitivity of stop payment windows

## Why this matters

OpsArena is most useful when the agent learns:

- not just how to finish one case
- but how to operate like a back-office analyst or supervisor
- under incomplete information, queue pressure, and delayed consequences
- across the full lifecycle of disputes, compliance investigations, and payment recovery
