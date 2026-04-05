from __future__ import annotations

from pydantic import BaseModel, Field

from opsarena.enums import RecordType


class CaseHiddenState(BaseModel):
    hidden_required_documents: list[str] = Field(default_factory=list)
    hidden_correction_fields: list[str] = Field(default_factory=list)
    hidden_response_latency_minutes: int | None = None
    hidden_follow_up_latency_minutes: int | None = None
    qa_sample_on_close: bool = False
    qa_sample_delay_minutes: int | None = None
    true_dispute_should_accept: bool = False
    true_fraud_risk: float = 0.0
    true_is_duplicate: bool = False
    true_variance_within_tolerance: bool = True
    true_doc_valid: bool = True
    true_sanctions_match: bool = False
    true_sanctions_false_positive: bool = False
    true_edd_required: bool = False
    true_beneficial_owner_issue: bool = False
    true_ofac_report_required: bool = False
    true_downstream_loss: float = 0.0
    true_vendor_will_respond: bool = True
    true_vendor_response_minutes: int = 30
    true_po_change_approved: bool = True
    true_stop_payment_success: bool = True
    true_recoverable_amount: float = 0.0
    forbidden_record_types: list[RecordType] = Field(default_factory=list)
