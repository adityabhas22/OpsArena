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


class ApprovalDecision(StrEnum):
    APPROVED = "approved"
    DENIED = "denied"


class InvoiceWorkflowState(BaseModel):
    workflow_type: Literal["invoice"] = "invoice"
    sla_total: int = 90
    approval_threshold: float = 100.0
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

    def public_metadata(self) -> dict[str, object]:
        return {
            "match_status": self.match_status.value if self.match_status else None,
            "duplicate_status": self.duplicate_status.value,
            "variance_amount": self.variance_amount,
            "payment_hold": self.payment_hold,
            "payment_hold_reason": self.payment_hold_reason,
            "credit_memo_status": self.credit_memo_status.value,
            "credit_memo_amount": self.credit_memo_amount,
            "approval_status": self.approval_status.value,
            "approval_reason": self.approval_reason,
            "approval_assignee": self.approval_assignee,
            "approval_chain": [entry.model_dump() for entry in self.approval_chain],
        }
