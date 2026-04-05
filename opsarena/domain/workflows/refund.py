from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class DisputeStage(StrEnum):
    INQUIRY = "inquiry"
    CHARGEBACK_OPEN = "chargeback_open"
    EVIDENCE_SUBMITTED = "evidence_submitted"
    PRE_ARBITRATION = "pre_arbitration"
    WON = "won"
    LOST = "lost"
    FINALIZED = "finalized"


class DisputeResolution(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    WON = "won"
    LOST = "lost"


class RefundExecutionState(StrEnum):
    NOT_STARTED = "not_started"
    REFUNDED = "refunded"
    PARTIALLY_REFUNDED = "partially_refunded"


class RefundWorkflowState(BaseModel):
    workflow_type: Literal["refund"] = "refund"
    sla_total: int = 60
    refund_threshold: float = 500.0
    dispute_stage: DisputeStage = DisputeStage.CHARGEBACK_OPEN
    dispute_resolution: DisputeResolution = DisputeResolution.PENDING
    dispute_fee: float = 15.0
    dispute_workflow_status: str = "needs_response"
    dispute_evidence_fields: list[str] = Field(default_factory=list)
    dispute_should_accept: bool = False
    refund_execution_state: RefundExecutionState = RefundExecutionState.NOT_STARTED
    refunded_amount: float | None = None
    secondary_approval_required: bool = False
    approval_status: str = "not_requested"

    def public_metadata(self) -> dict[str, object]:
        return {
            "dispute_stage": self.dispute_stage.value,
            "dispute_resolution": self.dispute_resolution.value,
            "dispute_fee": self.dispute_fee,
            "dispute_workflow_status": self.dispute_workflow_status,
            "dispute_evidence_fields": self.dispute_evidence_fields,
            "refund_execution_state": self.refund_execution_state.value,
            "refunded_amount": self.refunded_amount,
        }
