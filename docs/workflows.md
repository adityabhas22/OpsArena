# OpsArena Workflows

This document shows the kinds of workflows OpsArena is trying to model.

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

## Why this matters

OpsArena is most useful when the agent learns:

- not just how to finish one case
- but how to operate like a back-office analyst or supervisor
- under incomplete information, queue pressure, and delayed consequences
