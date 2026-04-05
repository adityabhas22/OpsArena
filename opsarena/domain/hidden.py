from __future__ import annotations

from pydantic import BaseModel, Field

from opsarena.enums import RecordType


class CaseHiddenState(BaseModel):
    hidden_required_documents: list[str] = Field(default_factory=list)
    hidden_response_latency_minutes: int | None = None
    hidden_follow_up_latency_minutes: int | None = None
    qa_sample_on_close: bool = False
    qa_sample_delay_minutes: int | None = None
    true_fraud_risk: float = 0.0
    true_is_duplicate: bool = False
    true_doc_valid: bool = True
    true_downstream_loss: float = 0.0
    forbidden_record_types: list[RecordType] = Field(default_factory=list)
