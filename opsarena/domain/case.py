from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from opsarena.domain.core import LinkedRecord, MessageLogEntry, RouteHistoryEntry
from opsarena.domain.hidden import CaseHiddenState
from opsarena.domain.workflows.invoice import InvoiceWorkflowState
from opsarena.domain.workflows.kyc import KYCWorkflowState
from opsarena.domain.workflows.refund import RefundWorkflowState
from opsarena.enums import CaseType, Priority, Resolution, TargetQueue, TaskId

WorkflowState = Annotated[
    RefundWorkflowState | InvoiceWorkflowState | KYCWorkflowState,
    Field(discriminator="workflow_type"),
]


class CaseState(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    case_id: str
    case_type: CaseType
    task_id: TaskId
    status: str = "open"
    priority: int = Priority.MEDIUM
    sla_deadline: int
    created_at: int
    current_owner: str = "ops_agent"
    active_queue: str = "default"
    route_reason: str | None = None
    route_history: list[RouteHistoryEntry] = Field(default_factory=list)
    sla_paused_at: int | None = None
    sla_pause_reason: str | None = None
    pre_pause_status: str | None = None
    resolution: Resolution = Resolution.PENDING
    amount: float = 0.0
    currency: str = "USD"
    visible_summary: str
    visible_flags: list[str] = Field(default_factory=list)
    linked_records: list[LinkedRecord] = Field(default_factory=list)
    required_check_names: list[str] = Field(default_factory=list)
    completed_check_names: list[str] = Field(default_factory=list)
    checks_required: int = 0
    checks_completed: int = 0
    evidence_types_available: list[str] = Field(default_factory=list)
    evidence_types_gathered: list[str] = Field(default_factory=list)
    evidence_items_available: int = 0
    evidence_items_gathered: int = 0
    notifications_required: int = 0
    notifications_sent: int = 0
    communication_log: list[MessageLogEntry] = Field(default_factory=list)
    internal_notes: list[str] = Field(default_factory=list)
    pending_info_fields: list[str] = Field(default_factory=list)
    requested_info_fields: list[str] = Field(default_factory=list)
    policy_id: str = ""
    allowed_escalation_queues: list[TargetQueue] = Field(default_factory=list)
    customer_id: str | None = None
    vendor_id: str | None = None
    requires_customer_notification: bool = False
    terminal_reason: str | None = None
    policy_checked: bool = False
    customer_notified: bool = False
    escalation_justified: bool = False
    claimed_by: str | None = None
    claimed_at: int | None = None
    next_touch_at: int | None = None
    waiting_reason: str | None = None
    follow_up_overdue: bool = False
    last_touched_at: int | None = None
    hidden: CaseHiddenState = Field(default_factory=CaseHiddenState)
    workflow: WorkflowState

    def mark_check(self, check_name: str) -> None:
        if check_name not in self.completed_check_names:
            self.completed_check_names.append(check_name)
            self.checks_completed = len(self.completed_check_names)

    def gather_evidence(self, evidence_type: str) -> bool:
        if evidence_type in self.evidence_types_gathered:
            return False
        self.evidence_types_gathered.append(evidence_type)
        self.evidence_items_gathered = len(self.evidence_types_gathered)
        return True
