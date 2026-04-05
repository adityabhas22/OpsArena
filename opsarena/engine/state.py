from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

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
from opsarena.enums import CaseType, Priority, RecordType, Resolution, TargetQueue, TaskId


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


class ScheduledEvent(BaseModel):
    event_id: str
    at_time: int
    event_type: str
    case_id: str
    payload: dict[str, Any] = Field(default_factory=dict)


class EpisodeMetrics(BaseModel):
    cases_resolved: int = 0
    cases_breached: int = 0
    total_resolution_value: float = 0.0
    total_penalty: float = 0.0
    escalation_count: int = 0
    tool_calls: int = 0
    simulated_minutes: int = 0
    cascading_events_triggered: int = 0
    reopens: int = 0
    chargebacks: int = 0
    duplicate_payments: int = 0
    compliance_violations: int = 0
    invalid_actions: int = 0
    data_breach_count: int = 0
    follow_ups_overdue: int = 0
    cases_claimed: int = 0
    claim_overflow_attempts: int = 0
    disputes_accepted: int = 0
    credit_memos_requested: int = 0
    secondary_approvals_requested: int = 0


class RecordStore(BaseModel):
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
    current_record_id: str | None = None
    current_record_type: RecordType | None = None
    customer_id: str | None = None
    vendor_id: str | None = None
    requires_customer_notification: bool = False
    terminal_reason: str | None = None
    workflow_data: dict[str, Any] = Field(default_factory=dict)
    hidden_required_documents: list[str] = Field(default_factory=list)
    hidden_response_latency_minutes: int | None = None
    hidden_follow_up_latency_minutes: int | None = None
    true_fraud_risk: float = 0.0
    true_is_duplicate: bool = False
    true_doc_valid: bool = True
    true_downstream_loss: float = 0.0
    policy_checked: bool = False
    customer_notified: bool = False
    escalation_justified: bool = False
    kyc_complete: bool = False
    forbidden_record_types: list[RecordType] = Field(default_factory=list)
    claimed_by: str | None = None
    claimed_at: int | None = None
    next_touch_at: int | None = None
    waiting_reason: str | None = None
    follow_up_overdue: bool = False
    last_touched_at: int | None = None

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


class QueueState(BaseModel):
    cases: list[CaseState] = Field(default_factory=list)
    current_time: int = 0
    escalation_queue_load: int = 0
    escalation_queue_capacity: int = 10
    total_cases_resolved: int = 0
    total_sla_breaches: int = 0
    oldest_open_case_age_minutes: int = 0
    overdue_follow_ups: int = 0
    claimed_case_count: int = 0


class WorldState(BaseModel):
    episode_id: str
    task_id: TaskId
    scenario_seed: int
    step_count: int = 0
    current_time: int = 0
    current_case_id: str | None = None
    current_record_type: RecordType | None = None
    current_record_id: str | None = None
    current_policy_id: str | None = None
    current_clause_id: str | None = None
    last_action_result: str = ""
    cases: dict[str, CaseState] = Field(default_factory=dict)
    queue_order: list[str] = Field(default_factory=list)
    records: RecordStore = Field(default_factory=RecordStore)
    scheduled_events: list[ScheduledEvent] = Field(default_factory=list)
    audit_log: list[AuditEntry] = Field(default_factory=list)
    metrics: EpisodeMetrics = Field(default_factory=EpisodeMetrics)
    objective_score: float = 0.0
    train_score: float = 0.0
    grader_breakdown: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def queue_state(self) -> QueueState:
        open_cases = [case for case in self.cases.values() if case.status != "closed"]
        return QueueState(
            cases=list(self.cases.values()),
            current_time=self.current_time,
            escalation_queue_load=self.metadata.get("escalation_queue_load", 0),
            escalation_queue_capacity=self.metadata.get("escalation_queue_capacity", 2),
            total_cases_resolved=self.metrics.cases_resolved,
            total_sla_breaches=self.metrics.cases_breached,
            oldest_open_case_age_minutes=max(
                (self.current_time - case.created_at for case in open_cases),
                default=0,
            ),
            overdue_follow_ups=sum(1 for case in open_cases if case.follow_up_overdue),
            claimed_case_count=sum(1 for case in open_cases if case.claimed_by is not None),
        )

    def open_cases(self) -> list[CaseState]:
        return [self.cases[case_id] for case_id in self.queue_order if self.cases[case_id].status != "closed"]
