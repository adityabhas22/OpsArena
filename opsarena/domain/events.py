from __future__ import annotations

from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from opsarena.documents import (
    CreditMemoRecord,
    CustomerRecord,
    DisputeRecord,
    GoodsReceipt,
    InvoiceRecord,
    KYCVerification,
    MessageTemplate,
    OrderRecord,
    PaymentRecord,
    PolicyDocument,
    PurchaseOrder,
    ShippingRecord,
)
from opsarena.domain.case import CaseState
from opsarena.enums import ReasonCode


class BaseScheduledEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: f"evt-{uuid4().hex[:8]}")
    at_time: int
    case_id: str | None = None
    event_type: str


class InfoResponseEvent(BaseScheduledEvent):
    event_type: Literal["info_response"] = "info_response"
    field_name: str
    record_id: str | None = None
    receipt_record: GoodsReceipt | None = None


class ChargebackEvent(BaseScheduledEvent):
    event_type: Literal["chargeback"] = "chargeback"


class ReopenEvent(BaseScheduledEvent):
    event_type: Literal["reopen"] = "reopen"


class SLABreachEvent(BaseScheduledEvent):
    event_type: Literal["sla_breach"] = "sla_breach"


class DisputeOutcomeEvent(BaseScheduledEvent):
    event_type: Literal["dispute_outcome"] = "dispute_outcome"
    outcome: str


class VendorCreditMemoReceivedEvent(BaseScheduledEvent):
    event_type: Literal["vendor_credit_memo_received"] = "vendor_credit_memo_received"
    amount: float


class SecondaryApprovalDecisionEvent(BaseScheduledEvent):
    event_type: Literal["secondary_approval_decision"] = "secondary_approval_decision"
    outcome: str


class FollowUpDueEvent(BaseScheduledEvent):
    event_type: Literal["follow_up_due"] = "follow_up_due"
    scheduled_for: int
    reason_code: ReasonCode


class ReworkDueEvent(BaseScheduledEvent):
    event_type: Literal["rework_due"] = "rework_due"
    scheduled_for: int


class QASampleSelectedEvent(BaseScheduledEvent):
    event_type: Literal["qa_sample_selected"] = "qa_sample_selected"
    case_id: str


class ArrivalWaveRecordBundle(BaseModel):
    orders: dict[str, OrderRecord] = Field(default_factory=dict)
    customers: dict[str, CustomerRecord] = Field(default_factory=dict)
    invoices: dict[str, InvoiceRecord] = Field(default_factory=dict)
    credit_memos: dict[str, CreditMemoRecord] = Field(default_factory=dict)
    purchase_orders: dict[str, PurchaseOrder] = Field(default_factory=dict)
    receipts: dict[str, GoodsReceipt] = Field(default_factory=dict)
    disputes: dict[str, DisputeRecord] = Field(default_factory=dict)
    kyc_verifications: dict[str, KYCVerification] = Field(default_factory=dict)
    policies: dict[str, PolicyDocument] = Field(default_factory=dict)
    message_templates: dict[str, MessageTemplate] = Field(default_factory=dict)
    shipping: dict[str, ShippingRecord] = Field(default_factory=dict)
    payments: dict[str, PaymentRecord] = Field(default_factory=dict)


class ArrivalWaveEvent(BaseScheduledEvent):
    event_type: Literal["arrival_wave"] = "arrival_wave"
    wave_id: str
    incoming_cases: list[CaseState] = Field(default_factory=list)
    record_bundle: ArrivalWaveRecordBundle = Field(default_factory=ArrivalWaveRecordBundle)


class StaffingDropEvent(BaseScheduledEvent):
    event_type: Literal["staffing_drop"] = "staffing_drop"
    capacity_delta: int = 1
    affected_owners: list[str] = Field(default_factory=list)


ScheduledEvent = Annotated[
    InfoResponseEvent
    | ChargebackEvent
    | ReopenEvent
    | SLABreachEvent
    | DisputeOutcomeEvent
    | VendorCreditMemoReceivedEvent
    | SecondaryApprovalDecisionEvent
    | FollowUpDueEvent
    | ReworkDueEvent
    | QASampleSelectedEvent
    | ArrivalWaveEvent
    | StaffingDropEvent,
    Field(discriminator="event_type"),
]
