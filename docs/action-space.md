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
| `approve` | Approve the case |
| `reject` | Reject the case |
| `escalate` | Escalate to a specialist queue |
| `defer` | Defer a case until later |
| `close_case` | Close a resolved case |
| `reopen_case` | Reopen a case |

## Communication and notes

| Action | What it does |
| --- | --- |
| `request_info` | Ask for missing information or documents |
| `send_message` | Send a customer/vendor/merchant message |
| `log_internal_note` | Add an internal note |

## Queue and ownership actions

| Action | What it does |
| --- | --- |
| `assign` | Assign a case to a role |
| `claim_case` | Claim work from a shared queue |
| `return_to_queue` | Put a claimed case back into the shared queue |
| `route_case` | Route a case to a different operational queue |
| `prioritize` | Change case priority |
| `batch_reorder` | Reorder the visible queue |
| `schedule_follow_up` | Set a next-touch time for waiting cases |
| `pause_sla` | Pause SLA while blocked on an external dependency |
| `resume_sla` | Resume a paused SLA |
| `advance_clock` | Advance simulated time and trigger delayed events |

## Refund and dispute actions

| Action | What it does |
| --- | --- |
| `execute_refund` | Execute a full or partial refund |
| `accept_dispute` | Accept a dispute when fighting it is not worth it |
| `submit_dispute_evidence` | Submit a dispute evidence packet |

## AP / invoice actions

| Action | What it does |
| --- | --- |
| `record_three_way_match` | Record match, variance, duplicate, or missing receipt result |
| `place_payment_hold` | Block payment while the invoice is under review |
| `release_payment_hold` | Release the block once conditions are cleared |
| `request_credit_memo` | Ask the vendor for a credit memo |
| `send_for_secondary_approval` | Send an exception or high-value case to a second approver |

## KYC / compliance actions

| Action | What it does |
| --- | --- |
| `review_kyc` | Record a KYC review outcome |
| `trigger_reverification` | Trigger a new document or identity review cycle |

## Why this matters

This action space is meant to teach an RL agent to:

- gather only the evidence it really needs
- choose between speed and correctness
- manage shared queue work under SLA pressure
- use holds, follow-ups, and approvals like real operators do
- handle delayed outcomes instead of treating every case as one-shot
