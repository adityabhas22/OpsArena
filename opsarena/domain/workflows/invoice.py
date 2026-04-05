from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from opsarena.domain.core import ApprovalHistoryEntry
from opsarena.enums import MatchStatus


class DuplicateStatus(StrEnum):
    SUSPECTED = "suspected"
    CONFIRMED = "confirmed"
    FALSE_POSITIVE = "false_positive"
    ALREADY_PAID = "already_paid"


class CreditMemoStatus(StrEnum):
    NOT_REQUESTED = "not_requested"
    REQUESTED = "requested"
    RECEIVED = "received"
    APPLIED = "applied"


class ApprovalStatus(StrEnum):
    NOT_REQUESTED = "not_requested"
    PENDING_SECONDARY = "pending_secondary"
    APPROVED = "approved"
    DENIED = "denied"


class PaymentBatchStatus(StrEnum):
    NOT_SCHEDULED = "not_scheduled"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    STOPPED = "stopped"


class VendorResponseStatus(StrEnum):
    NOT_REQUESTED = "not_requested"
    AWAITING = "awaiting"
    RECEIVED = "received"
    OVERDUE = "overdue"


class RecoveryStatus(StrEnum):
    NOT_NEEDED = "not_needed"
    IN_PROGRESS = "in_progress"
    PARTIAL = "partial"
    COMPLETE = "complete"


class POChangeStatus(StrEnum):
    NOT_REQUESTED = "not_requested"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    DENIED = "denied"


class ApprovalDecision(StrEnum):
    APPROVED = "approved"
    DENIED = "denied"


class InvoiceWorkflowState(BaseModel):
    workflow_type: Literal["invoice"] = "invoice"
    sla_total: int = 90
    approval_threshold: float = 100.0
    tolerance_percent: float = 3.0  # variances within this % of PO value auto-approve
    match_status: MatchStatus | None = None
    duplicate_status: DuplicateStatus = DuplicateStatus.SUSPECTED
    variance_amount: float | None = None
    payment_hold: bool = False
    payment_hold_reason: str | None = None
    payment_hold_set_at: int | None = None
    payment_hold_released_at: int | None = None
    credit_memo_status: CreditMemoStatus = CreditMemoStatus.NOT_REQUESTED
    credit_memo_amount: float | None = None
    approval_status: ApprovalStatus = ApprovalStatus.NOT_REQUESTED
    secondary_approval_required: bool = False
    approval_expected_outcome: ApprovalDecision = ApprovalDecision.APPROVED
    approval_reason: str | None = None
    approval_assignee: str | None = None
    approval_chain: list[ApprovalHistoryEntry] = Field(default_factory=list)
    payment_batch_id: str | None = None
    payment_batch_status: PaymentBatchStatus = PaymentBatchStatus.NOT_SCHEDULED
    vendor_response_status: VendorResponseStatus = VendorResponseStatus.NOT_REQUESTED
    recovery_status: RecoveryStatus = RecoveryStatus.NOT_NEEDED
    po_change_status: POChangeStatus = POChangeStatus.NOT_REQUESTED
    stop_payment_window_until: int | None = None
    write_off_threshold: float = 50.0

    def within_tolerance(self, po_total: float) -> bool:
        """Check if the variance amount is within the tolerance threshold."""
        if self.variance_amount is None or po_total <= 0:
            return True
        return abs(self.variance_amount) / po_total * 100 <= self.tolerance_percent

    def public_metadata(self) -> dict[str, object]:
        return {
            "match_status": self.match_status.value if self.match_status else None,
            "duplicate_status": self.duplicate_status.value,
            "variance_amount": self.variance_amount,
            "tolerance_percent": self.tolerance_percent,
            "payment_hold": self.payment_hold,
            "payment_hold_reason": self.payment_hold_reason,
            "credit_memo_status": self.credit_memo_status.value,
            "credit_memo_amount": self.credit_memo_amount,
            "approval_status": self.approval_status.value,
            "approval_reason": self.approval_reason,
            "approval_assignee": self.approval_assignee,
            "approval_chain": [entry.model_dump() for entry in self.approval_chain],
            "payment_batch_id": self.payment_batch_id,
            "payment_batch_status": self.payment_batch_status.value,
            "vendor_response_status": self.vendor_response_status.value,
            "recovery_status": self.recovery_status.value,
            "po_change_status": self.po_change_status.value,
            "stop_payment_window_until": self.stop_payment_window_until,
        }
