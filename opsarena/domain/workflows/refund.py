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


class PreDisputeType(StrEnum):
    NONE = "none"
    EARLY_FRAUD_WARNING = "early_fraud_warning"
    INQUIRY = "inquiry"
    RDR_ALERT = "rdr_alert"


class MerchantRiskLevel(StrEnum):
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


class MonitoringProgramStatus(StrEnum):
    NORMAL = "normal"
    WARNING = "warning"
    BREACHED = "breached"


class PrearbitrationDecision(StrEnum):
    ACCEPT = "accept"
    CONTEST = "contest"


class RefundWorkflowState(BaseModel):
    workflow_type: Literal["refund"] = "refund"
    sla_total: int = 60
    refund_threshold: float = 500.0
    pre_dispute_type: PreDisputeType = PreDisputeType.NONE
    pre_dispute_due_at: int | None = None
    liability_shift_applies: bool = False
    dispute_stage: DisputeStage = DisputeStage.CHARGEBACK_OPEN
    dispute_resolution: DisputeResolution = DisputeResolution.PENDING
    dispute_fee: float = 15.0
    dispute_workflow_status: str = "needs_response"
    dispute_evidence_fields: list[str] = Field(default_factory=list)
    chargeback_received_at: int | None = None
    representment_due_at: int | None = None
    prearbitration_due_at: int | None = None
    chargeback_amount: float | None = None
    refund_execution_state: RefundExecutionState = RefundExecutionState.NOT_STARTED
    refunded_amount: float | None = None
    reserve_percent: float = 0.0
    reserve_release_due_at: int | None = None
    payout_delay_days: int = 0
    payout_frozen: bool = False
    payout_freeze_reason: str | None = None
    merchant_risk_level: MerchantRiskLevel = MerchantRiskLevel.NORMAL
    monitoring_program_status: MonitoringProgramStatus = MonitoringProgramStatus.NORMAL
    merchant_dispute_ratio_30d: float = 0.0
    merchant_fraud_ratio_30d: float = 0.0
    merchant_negative_balance: bool = False
    merchant_age_days: int = 180
    merchant_volume_change_7d: float = 0.0
    avg_fulfillment_days: int = 2
    secondary_approval_required: bool = False
    approval_status: str = "not_requested"

    def recompute_risk_state(self) -> None:
        combined_ratio = self.merchant_dispute_ratio_30d + self.merchant_fraud_ratio_30d
        if combined_ratio >= 0.015 or self.merchant_negative_balance:
            self.monitoring_program_status = MonitoringProgramStatus.BREACHED
            self.merchant_risk_level = MerchantRiskLevel.CRITICAL
        elif self.merchant_dispute_ratio_30d >= 0.0075 or self.merchant_fraud_ratio_30d >= 0.005:
            self.monitoring_program_status = MonitoringProgramStatus.WARNING
            self.merchant_risk_level = MerchantRiskLevel.HIGH
        elif self.merchant_dispute_ratio_30d >= 0.005 or self.avg_fulfillment_days >= 7:
            self.monitoring_program_status = MonitoringProgramStatus.NORMAL
            self.merchant_risk_level = MerchantRiskLevel.ELEVATED
        else:
            self.monitoring_program_status = MonitoringProgramStatus.NORMAL
            self.merchant_risk_level = MerchantRiskLevel.NORMAL

    def public_metadata(self) -> dict[str, object]:
        return {
            "pre_dispute_type": self.pre_dispute_type.value,
            "pre_dispute_due_at": self.pre_dispute_due_at,
            "liability_shift_applies": self.liability_shift_applies,
            "dispute_stage": self.dispute_stage.value,
            "dispute_resolution": self.dispute_resolution.value,
            "dispute_fee": self.dispute_fee,
            "dispute_workflow_status": self.dispute_workflow_status,
            "dispute_evidence_fields": self.dispute_evidence_fields,
            "chargeback_received_at": self.chargeback_received_at,
            "representment_due_at": self.representment_due_at,
            "prearbitration_due_at": self.prearbitration_due_at,
            "chargeback_amount": self.chargeback_amount,
            "refund_execution_state": self.refund_execution_state.value,
            "refunded_amount": self.refunded_amount,
            "reserve_percent": self.reserve_percent,
            "reserve_release_due_at": self.reserve_release_due_at,
            "payout_delay_days": self.payout_delay_days,
            "payout_frozen": self.payout_frozen,
            "payout_freeze_reason": self.payout_freeze_reason,
            "merchant_risk_level": self.merchant_risk_level.value,
            "monitoring_program_status": self.monitoring_program_status.value,
            "merchant_dispute_ratio_30d": self.merchant_dispute_ratio_30d,
            "merchant_fraud_ratio_30d": self.merchant_fraud_ratio_30d,
            "merchant_negative_balance": self.merchant_negative_balance,
            "merchant_age_days": self.merchant_age_days,
            "merchant_volume_change_7d": self.merchant_volume_change_7d,
            "avg_fulfillment_days": self.avg_fulfillment_days,
        }
