from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class KYCStage(StrEnum):
    CURRENT_DUE = "current_due"
    PENDING_REVIEW = "pending_review"
    REVERIFICATION_REQUIRED = "reverification_required"
    CLEARED = "cleared"
    REJECTED = "rejected"


class KYCWorkflowState(BaseModel):
    workflow_type: Literal["kyc"] = "kyc"
    sla_total: int = 120
    verification_status: str = "requires_input"
    requirements_due: list[str] = Field(default_factory=list)
    payout_hold: bool = True
    kyc_stage: KYCStage = KYCStage.CURRENT_DUE
    kyc_complete: bool = False

    def public_metadata(self) -> dict[str, object]:
        return {
            "verification_status": self.verification_status,
            "requirements_due": self.requirements_due,
            "payout_hold": self.payout_hold,
            "kyc_stage": self.kyc_stage.value,
        }
