from __future__ import annotations

from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from opsarena.documents import GoodsReceipt
from opsarena.enums import ReasonCode


class BaseScheduledEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: f"evt-{uuid4().hex[:8]}")
    at_time: int
    case_id: str
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


ScheduledEvent = Annotated[
    InfoResponseEvent
    | ChargebackEvent
    | ReopenEvent
    | SLABreachEvent
    | DisputeOutcomeEvent
    | VendorCreditMemoReceivedEvent
    | SecondaryApprovalDecisionEvent
    | FollowUpDueEvent,
    Field(discriminator="event_type"),
]
