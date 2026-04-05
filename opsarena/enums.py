from __future__ import annotations

from enum import IntEnum, StrEnum


class CaseType(StrEnum):
    REFUND = "refund"
    INVOICE = "invoice"
    KYC = "kyc"
    TRIAGE = "triage"


class Resolution(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    DEFERRED = "deferred"
    CLOSED = "closed"
    PENDING = "pending"


class RecordType(StrEnum):
    INVOICE = "invoice"
    CREDIT_MEMO = "credit_memo"
    ORDER = "order"
    RECEIPT = "receipt"
    CUSTOMER = "customer"
    VENDOR = "vendor"
    KYC_DOCUMENT = "kyc_document"
    SHIPPING = "shipping"
    PAYMENT = "payment"
    POLICY = "policy"
    DISPUTE = "dispute"
    PURCHASE_ORDER = "purchase_order"


class TargetQueue(StrEnum):
    MANAGER_REVIEW = "manager_review"
    FRAUD_TEAM = "fraud_team"
    COMPLIANCE = "compliance"
    SENIOR_OPS = "senior_ops"


class ReasonCode(StrEnum):
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    MISSING_DOCUMENTATION = "missing_documentation"
    POLICY_AMBIGUITY = "policy_ambiguity"
    CUSTOMER_REQUEST = "customer_request"
    DUPLICATE_MATCH = "duplicate_match"
    INVALID_DOCUMENT = "invalid_document"
    SANCTIONS_MATCH = "sanctions_match"
    BENEFICIAL_OWNER_MISMATCH = "beneficial_owner_mismatch"
    EDD_REQUIRED = "edd_required"
    REPORTING_REQUIRED = "reporting_required"
    AWAITING_RESPONSE = "awaiting_response"
    SLA_PROTECTION = "sla_protection"
    COMPLETE = "complete"


class DecisionCode(StrEnum):
    STANDARD_APPROVAL = "standard_approval"
    EXCEPTION_APPROVAL = "exception_approval"
    PARTIAL_APPROVAL = "partial_approval"


class MatchStatus(StrEnum):
    MATCHED = "matched"
    VARIANCE = "variance"
    MISSING_RECEIPT = "missing_receipt"
    DUPLICATE = "duplicate"
    UNMATCHED = "unmatched"


class VerificationDecision(StrEnum):
    APPROVE = "approve"
    REQUEST_RESUBMISSION = "request_resubmission"
    REJECT = "reject"


class SortField(StrEnum):
    PRIORITY = "priority"
    SLA_REMAINING = "sla_remaining"
    CREATED_AT = "created_at"
    AMOUNT = "amount"


class Priority(IntEnum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class TaskId(StrEnum):
    REFUND_EXCEPTION = "refund_exception"
    INVOICE_PLUS_KYC = "invoice_plus_kyc"
    QUEUE_TRIAGE = "queue_triage"
    AP_PAYMENT_RUN = "ap_payment_run"
