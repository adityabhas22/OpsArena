from __future__ import annotations

from typing import Any

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
from opsarena.domain.core import AuditEntry, LinkedRecord, MessageLogEntry
from opsarena.domain.events import ScheduledEvent
from opsarena.enums import RecordType, TaskId


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
    fraudulent_approvals: int = 0
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
    qa_reviews_requested: int = 0
    qa_reviews_passed: int = 0
    qa_reviews_failed: int = 0
    qa_rework_overdue: int = 0
    payment_freezes: int = 0
    ofac_reports_filed: int = 0
    report_deadlines_missed: int = 0


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


class QueueState(BaseModel):
    cases: list[CaseState] = Field(default_factory=list)
    current_time: int = 0
    escalation_queue_load: int = 0
    escalation_queue_capacity: int = 10
    total_cases_resolved: int = 0
    total_sla_breaches: int = 0
    oldest_open_case_age_minutes: int = 0
    queue_backlog_age_minutes: int = 0
    overdue_follow_ups: int = 0
    claimed_case_count: int = 0
    unassigned_count: int = 0
    exception_queue_size: int = 0
    agent_capacity: int = 0


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
    benchmark_score: float = 0.0
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
            queue_backlog_age_minutes=max(
                (self.current_time - case.created_at for case in open_cases),
                default=0,
            ),
            overdue_follow_ups=sum(1 for case in open_cases if case.follow_up_overdue),
            claimed_case_count=sum(1 for case in open_cases if case.claimed_by is not None),
            unassigned_count=sum(1 for case in open_cases if case.claimed_by is None and case.current_owner in {"queue", "ops_agent"}),
            exception_queue_size=len(open_cases),
            agent_capacity=self.metadata.get("agent_capacity", self.metadata.get("claim_capacity", 2)),
        )

    def open_cases(self) -> list[CaseState]:
        return [self.cases[case_id] for case_id in self.queue_order if self.cases[case_id].status != "closed"]
