from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field


class TaxAmount(BaseModel):
    amount: int
    rate: str
    display_name: str
    percentage: float
    inclusive: bool = False
    jurisdiction: str = ""


class DiscountAmount(BaseModel):
    amount: int
    discount_code: str


class InvoiceLineItem(BaseModel):
    description: str
    sku: str
    quantity: Decimal
    unit_price: int
    amount: int
    subtotal: int
    tax_amounts: list[TaxAmount] = Field(default_factory=list)
    discount_amounts: list[DiscountAmount] = Field(default_factory=list)


class ShippingCost(BaseModel):
    amount_subtotal: int
    amount_tax: int
    amount_total: int


class InvoiceRecord(BaseModel):
    invoice_id: str
    invoice_number: str
    status: str
    vendor_name: str
    vendor_id: str
    vendor_address: str
    po_reference: str
    currency: str
    created_at: int
    due_date: int
    paid_at: int | None = None
    line_items: list[InvoiceLineItem] = Field(default_factory=list)
    shipping_cost: ShippingCost
    subtotal: int
    total_discount: int
    total_tax: int
    total: int
    amount_due: int
    amount_paid: int
    amount_remaining: int
    payment_method: str


class CreditMemoRecord(BaseModel):
    credit_memo_id: str
    invoice_id: str
    vendor_id: str
    vendor_name: str
    amount: int
    currency: str
    reason: str
    status: str
    created_at: int
    expected_apply_date: int | None = None


class POLineItem(BaseModel):
    line_number: int
    material_number: str
    description: str
    quantity: Decimal
    unit_of_measure: str
    net_price: int
    price_unit: int
    delivery_date: int
    delivery_completed: bool = False
    received_quantity: Decimal = Decimal("0")
    invoiced_quantity: Decimal = Decimal("0")


class PurchaseOrder(BaseModel):
    po_number: str
    document_type: str
    status: str
    vendor_id: str
    vendor_name: str
    vendor_address: str
    currency: str
    payment_terms: str
    total_net_value: int
    total_gross_value: int
    requested_delivery_date: int
    approval_status: str
    approved_by: str | None = None
    line_items: list[POLineItem] = Field(default_factory=list)


class ReceiptLineItem(BaseModel):
    po_line_number: int
    sku: str
    description: str
    ordered_quantity: Decimal
    shipped_quantity: Decimal
    received_quantity: Decimal
    accepted_quantity: Decimal
    rejected_quantity: Decimal
    condition: str
    condition_notes: str = ""
    variance_quantity: Decimal = Decimal("0")
    variance_percentage: float = 0.0
    within_tolerance: bool = True


class GoodsReceipt(BaseModel):
    receipt_id: str
    receipt_type: str
    status: str
    po_reference: str
    vendor_id: str
    receiving_warehouse: str
    received_by: str
    receipt_date: int
    carrier: str
    tracking_numbers: list[str] = Field(default_factory=list)
    shipment_status: str
    line_items: list[ReceiptLineItem] = Field(default_factory=list)


class CommunicationEntry(BaseModel):
    timestamp: int
    channel: str
    subject: str
    body: str
    template_id: str | None = None


class CustomerRecord(BaseModel):
    customer_id: str
    email: str
    first_name: str
    last_name: str
    phone: str | None = None
    state: str = "enabled"
    created_at: int = 0
    locale: str = "en"
    currency: str = "USD"
    tags: list[str] = Field(default_factory=list)
    note: str = ""
    orders_count: int = 0
    total_spent: Decimal = Decimal("0")
    last_order_date: int | None = None
    average_order_value: Decimal = Decimal("0")
    loyalty_tier: str = "none"
    fraud_score: float = 0.0
    fraud_risk_level: str = "low"
    chargeback_count: int = 0
    return_count: int = 0
    return_rate: float = 0.0
    account_flags: list[str] = Field(default_factory=list)
    communication_log: list[CommunicationEntry] = Field(default_factory=list)


class OrderLineItem(BaseModel):
    sku: str
    title: str
    quantity: int
    unit_price: int
    fulfillment_status: str


class OrderRecord(BaseModel):
    order_id: str
    order_number: str
    customer_id: str
    created_at: int
    currency: str
    subtotal_price: int
    total_tax: int
    total_price: int
    financial_status: str
    fulfillment_status: str
    line_items: list[OrderLineItem] = Field(default_factory=list)


class ShippingRecord(BaseModel):
    shipment_id: str
    order_id: str
    carrier: str
    tracking_number: str
    status: str
    shipped_at: int | None = None
    delivered_at: int | None = None
    delivery_exception: str | None = None


class PaymentRecord(BaseModel):
    payment_id: str
    order_id: str
    processor: str
    amount: int
    currency: str
    method: str
    status: str
    charge_id: str | None = None


class DisputeEvidence(BaseModel):
    customer_name: str = ""
    email: str = ""
    purchase_ip: str = ""
    billing_address: str = ""
    shipping_carrier: str = ""
    tracking_number: str = ""
    shipping_date: int | None = None
    product_description: str = ""
    receipt: str = ""
    refund_policy: str = ""
    customer_communication: str = ""
    customer_signature: str = ""


class DisputeRecord(BaseModel):
    dispute_id: str
    charge_id: str
    order_id: str
    amount: int
    currency: str
    status: str
    reason: str
    network_reason_code: str
    evidence_due_by: int
    submission_count: int
    is_charge_refundable: bool
    card_brand: str
    evidence: DisputeEvidence = Field(default_factory=DisputeEvidence)


class VerifiedIdentity(BaseModel):
    first_name: str = ""
    last_name: str = ""
    dob: str = ""
    address: str = ""
    id_number: str = ""


class KYCVerification(BaseModel):
    session_id: str
    entity_id: str
    entity_type: str
    status: str
    requirements_currently_due: list[str] = Field(default_factory=list)
    requirements_past_due: list[str] = Field(default_factory=list)
    verified_outputs: VerifiedIdentity = Field(default_factory=VerifiedIdentity)
    document_front: str | None = None
    document_back: str | None = None
    error_code: str | None = None


class PolicyCondition(BaseModel):
    field: str
    operator: str
    value: Any
    unit: str | None = None


class PolicyAction(BaseModel):
    action_type: str
    target: str | None = None
    reason_code: str | None = None


class PolicyException(BaseModel):
    conditions: list[PolicyCondition] = Field(default_factory=list)
    override_action: str


class PolicyClause(BaseModel):
    clause_id: str
    clause_type: str
    description: str
    conditions: list[PolicyCondition] = Field(default_factory=list)
    actions: list[PolicyAction] = Field(default_factory=list)
    exceptions: list[PolicyException] = Field(default_factory=list)
    mandatory: bool = False


class PolicyDocument(BaseModel):
    policy_id: str
    policy_type: str
    title: str
    version: str
    effective_date: str
    status: str
    clauses: list[PolicyClause] = Field(default_factory=list)


class MessageTemplate(BaseModel):
    template_id: str
    template_type: str
    channel: str
    subject_template: str
    body_template: str
    required_slots: list[str] = Field(default_factory=list)
    optional_slots: list[str] = Field(default_factory=list)
    tone: str = "neutral"
