# OpsArena Action Space

OpsArena uses a typed action space instead of browser clicks.

The agent always sends a structured action like:

```python
{"action_type": "open_case", "case_id": "case_invoice_1"}
```

## Information-gathering actions

| Action | What it does |
| --- | --- |
| `list_queue` | View the queue with sorting and filters |
| `open_case` | Open one case and inspect its details |
| `view_record` | Open a linked document like an invoice, dispute, payment, or KYC record |
| `query_policy` | Read the relevant policy or clause |
| `search_cases` | Search for cases using filters |
| `inspect_audit` | Inspect recent audit history for a case |

## Core case actions

| Action | What it does |
| --- | --- |
| `approve` | Approve the case (supports partial approval with an approved_amount) |
| `reject` | Reject the case with a reason code |
| `escalate` | Escalate to a specialist queue with a priority override |
| `defer` | Defer a case until a specified future time |
| `close_case` | Close a resolved case with a resolution code |
| `reopen_case` | Reopen a previously closed case |

## Communication and notes

| Action | What it does |
| --- | --- |
| `request_info` | Ask for a specific missing field or document |
| `send_message` | Send a templated customer, vendor, or merchant message |
| `log_internal_note` | Add an internal note with a note code and metadata |

## Queue and ownership actions

| Action | What it does |
| --- | --- |
| `assign` | Assign a case to a role |
| `claim_case` | Claim work from a shared queue |
| `return_to_queue` | Put a claimed case back into the shared queue |
| `route_case` | Route a case to a different operational queue |
| `prioritize` | Change case priority |
| `batch_reorder` | Reorder the visible queue by an ordering rule |
| `schedule_follow_up` | Set a next-touch time for waiting cases |
| `pause_sla` | Pause SLA while blocked on an external dependency |
| `resume_sla` | Resume a paused SLA |
| `advance_clock` | Advance simulated time (1-480 minutes) and trigger delayed events |

## Supervisor actions

| Action | What it does |
| --- | --- |
| `bulk_assign` | Assign up to 10 cases to an assignee in one action |
| `bulk_route` | Route up to 10 cases to a target queue in one action |
| `rebalance_queue` | Redistribute unassigned cases across an assignee pool using a strategy (sla_priority, oldest, or amount) |

## QA actions

| Action | What it does |
| --- | --- |
| `send_to_qa` | Submit a resolved case to the QA review queue |
| `approve_qa` | Mark a QA-reviewed case as passing quality review |
| `fail_qa` | Fail a QA-reviewed case and send it back for rework |

## Refund and dispute actions

| Action | What it does |
| --- | --- |
| `execute_refund` | Execute a full or partial refund |
| `accept_dispute` | Accept a dispute when fighting it is not worth it |
| `challenge_dispute` | Challenge a dispute by submitting a representment |
| `submit_dispute_evidence` | Submit a dispute evidence packet with specific evidence fields |
| `refund_pre_dispute_alert` | Refund proactively on a pre-dispute alert (early fraud warning, inquiry, or RDR) to avoid a chargeback |
| `resolve_prearbitration` | Respond to a pre-arbitration notice by accepting or contesting |

## Payout risk controls

| Action | What it does |
| --- | --- |
| `freeze_payouts` | Freeze merchant payouts when risk thresholds are breached |
| `unfreeze_payouts` | Unfreeze payouts once risk conditions have cleared |
| `set_reserve_percent` | Set a rolling reserve percentage (0-100%) on merchant payouts, with optional auto-release timer |
| `clear_reserve` | Clear a previously set reserve |
| `set_payout_delay_days` | Set a payout delay (0-30 days) for a merchant |

## AP / invoice actions

| Action | What it does |
| --- | --- |
| `record_three_way_match` | Record match, variance, duplicate, or missing receipt result |
| `place_payment_hold` | Block payment while the invoice is under review |
| `release_payment_hold` | Release the block once conditions are cleared |
| `request_credit_memo` | Ask the vendor for a credit memo |
| `send_for_secondary_approval` | Send an exception or high-value case to a second approver |

## AP recovery actions

| Action | What it does |
| --- | --- |
| `request_revised_invoice` | Ask the vendor to resubmit a corrected invoice |
| `request_po_change` | Request a purchase order amendment with a change description |
| `remove_from_payment_batch` | Pull an invoice out of a scheduled payment batch |
| `stop_payment` | Stop a payment that is already in progress |
| `record_vendor_refund` | Record that a vendor has returned funds |
| `apply_credit_memo` | Apply a received credit memo to an invoice case |
| `write_off_small_balance` | Write off a small remaining balance below the write-off threshold |

## KYC / compliance actions

| Action | What it does |
| --- | --- |
| `review_kyc` | Record a KYC review outcome (approve, reject, or request resubmission) |
| `trigger_reverification` | Trigger a new document or identity review cycle with specific requirements |
| `run_sanctions_screen` | Run a sanctions and watchlist screening check |
| `start_edd_review` | Initiate an Enhanced Due Diligence (EDD) review |
| `review_beneficial_owner` | Review and verify beneficial ownership information |
| `request_field_correction` | Ask the applicant to correct a specific field |
| `file_ofac_report` | File a required OFAC or regulatory report |
| `freeze_payments` | Freeze all payments for a case due to compliance concerns |

## Why this matters

This action space is meant to teach an RL agent to:

- gather only the evidence it really needs
- choose between speed and correctness
- manage shared queue work under SLA pressure
- use holds, follow-ups, and approvals like real operators do
- handle delayed outcomes instead of treating every case as one-shot
- manage the full dispute lifecycle from pre-dispute alerts through pre-arbitration
- apply payout risk controls proportional to merchant risk level
- run compliance investigations including sanctions screening, EDD, and beneficial owner review
- recover value after invoice exceptions through revised invoices, PO changes, stop payments, and write-offs
- coordinate supervisor-level work like bulk assignment, routing, and queue rebalancing
- maintain quality through QA sampling, review, and rework loops
