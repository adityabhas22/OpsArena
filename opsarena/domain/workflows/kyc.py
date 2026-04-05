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


class SanctionsStatus(StrEnum):
    NOT_STARTED = "not_started"
    CLEAR = "clear"
    POTENTIAL_MATCH = "potential_match"
    CONFIRMED_MATCH = "confirmed_match"


class EDDStatus(StrEnum):
    NOT_STARTED = "not_started"
    NOT_REQUIRED = "not_required"
    IN_PROGRESS = "in_progress"
    AWAITING_RESPONSE = "awaiting_response"
    CLEARED = "cleared"


class BeneficialOwnerStatus(StrEnum):
    NOT_STARTED = "not_started"
    PENDING_REVIEW = "pending_review"
    NEEDS_CORRECTION = "needs_correction"
    VERIFIED = "verified"
    REJECTED = "rejected"


class OFACReportStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    FILED = "filed"
    MISSED = "missed"


class KYCWorkflowState(BaseModel):
    workflow_type: Literal["kyc"] = "kyc"
    sla_total: int = 120
    verification_status: str = "requires_input"
    requirements_due: list[str] = Field(default_factory=list)
    payout_hold: bool = True
    kyc_stage: KYCStage = KYCStage.CURRENT_DUE
    kyc_complete: bool = False
    sanctions_status: SanctionsStatus = SanctionsStatus.NOT_STARTED
    edd_status: EDDStatus = EDDStatus.NOT_STARTED
    beneficial_owner_status: BeneficialOwnerStatus = BeneficialOwnerStatus.NOT_STARTED
    ofac_report_status: OFACReportStatus = OFACReportStatus.NOT_REQUIRED
    screening_match_confidence: float = 0.0
    report_due_at: int | None = None
    edd_due_at: int | None = None
    payments_frozen: bool = False
    payment_freeze_reason: str | None = None
    correction_fields: list[str] = Field(default_factory=list)
    sanctions_screened_at: int | None = None

    def public_metadata(self) -> dict[str, object]:
        return {
            "verification_status": self.verification_status,
            "requirements_due": self.requirements_due,
            "payout_hold": self.payout_hold,
            "kyc_stage": self.kyc_stage.value,
            "kyc_complete": self.kyc_complete,
            "sanctions_status": self.sanctions_status.value,
            "edd_status": self.edd_status.value,
            "beneficial_owner_status": self.beneficial_owner_status.value,
            "ofac_report_status": self.ofac_report_status.value,
            "screening_match_confidence": self.screening_match_confidence,
            "report_due_at": self.report_due_at,
            "edd_due_at": self.edd_due_at,
            "payments_frozen": self.payments_frozen,
            "payment_freeze_reason": self.payment_freeze_reason,
            "correction_fields": self.correction_fields,
            "sanctions_screened_at": self.sanctions_screened_at,
        }

    def edd_cleared(self) -> bool:
        return self.edd_status in {EDDStatus.NOT_REQUIRED, EDDStatus.CLEARED}

    def beneficial_owner_cleared(self) -> bool:
        return self.beneficial_owner_status in {
            BeneficialOwnerStatus.NOT_STARTED,
            BeneficialOwnerStatus.VERIFIED,
        }

    def reporting_complete(self) -> bool:
        return self.ofac_report_status in {OFACReportStatus.NOT_REQUIRED, OFACReportStatus.FILED}

    def approval_ready(self) -> bool:
        return (
            self.sanctions_status == SanctionsStatus.CLEAR
            and self.edd_cleared()
            and self.beneficial_owner_cleared()
            and self.reporting_complete()
            and not self.correction_fields
            and not self.payments_frozen
        )
