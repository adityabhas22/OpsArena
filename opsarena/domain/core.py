from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from opsarena.enums import RecordType


class LinkedRecord(BaseModel):
    record_type: RecordType
    record_id: str
    title: str


class MessageLogEntry(BaseModel):
    timestamp: int
    channel: str
    subject: str
    body: str
    template_id: str | None = None


class AuditEntry(BaseModel):
    timestamp: int
    case_id: str | None = None
    action_type: str
    message: str
    success: bool = True


class RouteHistoryEntry(BaseModel):
    at_time: int
    queue: str
    owner: str
    reason: str


class ApprovalHistoryEntry(BaseModel):
    at_time: int
    status: str
    owner: str
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QAStatus(StrEnum):
    NOT_REQUESTED = "not_requested"
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"


class QAReviewEntry(BaseModel):
    at_time: int
    status: QAStatus
    owner: str
    reason: str | None = None
    notes: str | None = None
