from __future__ import annotations

import random
from decimal import Decimal
from pathlib import Path
import yaml

from opsarena.documents import (
    BeneficialOwner,
    CustomerRecord,
    DisputeRecord,
    DisputeEvidence,
    GoodsReceipt,
    InvoiceLineItem,
    InvoiceRecord,
    KYCVerification,
    MessageTemplate,
    OrderLineItem,
    OrderRecord,
    PaymentRecord,
    PolicyDocument,
    PurchaseOrder,
    ReceiptLineItem,
    ShippingCost,
    ShippingRecord,
)
from opsarena.domain.core import LinkedRecord
from opsarena.domain.events import (
    ArrivalWaveEvent,
    ArrivalWaveRecordBundle,
    InquiryEscalatesToChargebackEvent,
    MonitoringThresholdBreachedEvent,
    PaymentBatchExecutedEvent,
    SLABreachEvent,
    StaffingDropEvent,
)
from opsarena.domain.hidden import CaseHiddenState
from opsarena.domain.workflows.invoice import (
    ApprovalDecision,
    ApprovalStatus,
    CreditMemoStatus,
    DuplicateStatus,
    InvoiceWorkflowState,
    PaymentBatchStatus,
)
from opsarena.domain.workflows.kyc import (
    BeneficialOwnerStatus,
    EDDStatus,
    KYCStage,
    KYCWorkflowState,
    OFACReportStatus,
    SanctionsStatus,
)
from opsarena.domain.workflows.refund import (
    DisputeResolution,
    DisputeStage,
    MonitoringProgramStatus,
    PreDisputeType,
    RefundWorkflowState,
)
from opsarena.engine.policies import load_policy
from opsarena.engine.state import CaseState, RecordStore, WorldState
from opsarena.enums import CaseType, MatchStatus, Priority, RecordType, TargetQueue, TaskId

ROOT = Path(__file__).resolve().parents[2]


def _load_template(task_id: TaskId) -> dict:
    mapping = {
        TaskId.REFUND_EXCEPTION: ROOT / "data" / "scenario_templates" / "task1_refund" / "template.yaml",
        TaskId.INVOICE_PLUS_KYC: ROOT / "data" / "scenario_templates" / "task2_invoice_kyc" / "template.yaml",
        TaskId.QUEUE_TRIAGE: ROOT / "data" / "scenario_templates" / "task3_triage" / "template.yaml",
        TaskId.AP_PAYMENT_RUN: ROOT / "data" / "scenario_templates" / "task4_ap_payment_run" / "template.yaml",
    }
    return yaml.safe_load(mapping[task_id].read_text())


def _choose_factor(template: dict, rng: random.Random, name: str, default):
    options = template.get("atomic_factors", {}).get(name)
    if not options:
        return default
    return rng.choice(options)


def _latency_minutes(label: str, *, fast: int = 15, standard: int = 30, slow: int = 60, silent: int = 120) -> int:
    return {
        "fast": fast,
        "standard": standard,
        "slow": slow,
        "vendor_silent": silent,
        "unresponsive": silent,
        "responsive": fast,
    }.get(label, standard)


def _base_state(task_id: TaskId, seed: int, episode_id: str | None = None) -> WorldState:
    return WorldState(
        episode_id=episode_id or f"{task_id.value}-{seed}",
        task_id=task_id,
        scenario_seed=seed,
        metadata={
            "escalation_queue_capacity": 2,
            "escalation_queue_load": 0,
            "approval_queue_capacity": 1,
            "approval_queue_load": 0,
            "claim_capacity": 2,
            "agent_capacity": 2,
            "max_steps": 40,
        },
    )


def _add_shared_templates(records: RecordStore) -> None:
    records.message_templates["refund_approved"] = MessageTemplate(
        template_id="refund_approved",
        template_type="refund_approved",
        channel="email",
        subject_template="Your refund has been approved",
        body_template="We approved a refund of {amount} for order {order_id}.",
        required_slots=["amount", "order_id"],
    )
    records.message_templates["info_request"] = MessageTemplate(
        template_id="info_request",
        template_type="info_request",
        channel="email",
        subject_template="Additional information required",
        body_template="Please provide {field_name} for case {case_id}.",
        required_slots=["field_name", "case_id"],
    )
    records.message_templates["case_closed"] = MessageTemplate(
        template_id="case_closed",
        template_type="case_closed",
        channel="email",
        subject_template="Case closed",
        body_template="Your case {case_id} is closed with resolution {resolution}.",
        required_slots=["case_id", "resolution"],
    )


def _rename_refund_bundle(case: CaseState, records: RecordStore, suffix: str) -> tuple[CaseState, RecordStore]:
    customer = next(iter(records.customers.values()))
    order = next(iter(records.orders.values()))
    shipping = next(iter(records.shipping.values()))
    payment = next(iter(records.payments.values()))
    dispute = next(iter(records.disputes.values()))

    customer = customer.model_copy(update={"customer_id": f"{customer.customer_id}_{suffix}"})
    order = order.model_copy(
        update={
            "order_id": f"{order.order_id}_{suffix}",
            "customer_id": customer.customer_id,
        }
    )
    shipping = shipping.model_copy(update={"shipment_id": f"{shipping.shipment_id}_{suffix}", "order_id": order.order_id})
    payment = payment.model_copy(update={"payment_id": f"{payment.payment_id}_{suffix}", "order_id": order.order_id, "charge_id": f"{payment.charge_id}_{suffix}"})
    dispute = dispute.model_copy(
        update={
            "dispute_id": f"{dispute.dispute_id}_{suffix}",
            "charge_id": payment.charge_id,
            "order_id": order.order_id,
        }
    )

    linked_records = []
    record_id_map = {
        RecordType.ORDER: order.order_id,
        RecordType.CUSTOMER: customer.customer_id,
        RecordType.SHIPPING: shipping.shipment_id,
        RecordType.PAYMENT: payment.payment_id,
        RecordType.DISPUTE: dispute.dispute_id,
    }
    for record in case.linked_records:
        linked_records.append(record.model_copy(update={"record_id": record_id_map.get(record.record_type, record.record_id)}))

    renamed_case = case.model_copy(
        update={
            "customer_id": customer.customer_id,
            "linked_records": linked_records,
        }
    )

    renamed_records = RecordStore(
        orders={order.order_id: order},
        customers={customer.customer_id: customer},
        disputes={dispute.dispute_id: dispute},
        policies=records.policies,
        message_templates=records.message_templates,
        shipping={shipping.shipment_id: shipping},
        payments={payment.payment_id: payment},
    )
    return renamed_case, renamed_records


def _refund_case(
    rng: random.Random,
    task_id: TaskId,
    current_time: int,
    *,
    amount_tier: str = "medium",
    threshold_proximity: str = "near_threshold",
    fraud_risk_label: str | None = None,
    sla_tightness: str = "standard",
    dispute_mode: str = "chargeback_open",
) -> tuple[CaseState, RecordStore]:
    records = RecordStore()
    _add_shared_templates(records)
    records.policies["refund_policy"] = load_policy("refund_policy")
    customer = CustomerRecord(
        customer_id="cust_refund_1",
        email="casey@example.com",
        first_name="Casey",
        last_name="Morgan",
        orders_count=12,
        total_spent=Decimal("1520.50"),
        average_order_value=Decimal("126.70"),
        loyalty_tier="silver",
        fraud_score=0.42,
        fraud_risk_level="medium",
        return_count=3,
        return_rate=0.18,
    )
    order = OrderRecord(
        order_id="ord_1001",
        order_number="#1001",
        customer_id=customer.customer_id,
        created_at=current_time - 120,
        currency="USD",
        subtotal_price=35000,
        total_tax=2800,
        total_price=37800,
        financial_status="paid",
        fulfillment_status="fulfilled",
        line_items=[OrderLineItem(sku="sku-tee", title="Graphic Tee", quantity=2, unit_price=17500, fulfillment_status="fulfilled")],
    )
    shipping = ShippingRecord(
        shipment_id="ship_1001",
        order_id=order.order_id,
        carrier="UPS",
        tracking_number="1Z999",
        status="delivered",
        shipped_at=current_time - 110,
        delivered_at=current_time - 80,
    )
    payment = PaymentRecord(
        payment_id="pay_1001",
        order_id=order.order_id,
        processor="stripe",
        amount=37800,
        currency="USD",
        method="card",
        status="captured",
        charge_id="ch_1001",
    )
    dispute = DisputeRecord(
        dispute_id="dp_1001",
        charge_id="ch_1001",
        order_id=order.order_id,
        amount=35000,
        currency="USD",
        status="needs_response",
        reason="credit_not_processed",
        network_reason_code="Visa 13.6",
        evidence_due_by=current_time + 240,
        submission_count=0,
        is_charge_refundable=True,
        card_brand="visa",
        evidence=DisputeEvidence(product_description="Graphic Tee"),
    )
    records.customers[customer.customer_id] = customer
    records.orders[order.order_id] = order
    records.shipping[shipping.shipment_id] = shipping
    records.payments[payment.payment_id] = payment
    records.disputes[dispute.dispute_id] = dispute

    amount_map = {
        "low": 45.0,
        "medium": 350.0,
        "high": 640.0,
    }
    threshold_amount_map = {
        "far_below": 120.0,
        "near_threshold": 490.0,
        "above_threshold": 640.0,
    }
    amount = threshold_amount_map.get(threshold_proximity, amount_map.get(amount_tier, 350.0))
    fraud_risk_map = {
        "low": 0.18,
        "medium": 0.48,
        "high": 0.86,
    }
    if fraud_risk_label is None:
        fraud_risk_label = "high" if rng.random() < 0.35 else "low"
    fraud_risk = fraud_risk_map.get(fraud_risk_label, 0.22)
    sla_offset = {
        "urgent": 30,
        "standard": 60,
        "relaxed": 90,
    }.get(sla_tightness, 60)

    flags = []
    if 300 <= amount <= 520:
        flags.append("near-threshold")
    if amount > 500:
        flags.append("manager-threshold")
    if fraud_risk > 0.7:
        flags.append("velocity_alert")

    case = CaseState(
        case_id="case_refund_1",
        case_type=CaseType.REFUND,
        task_id=task_id,
        priority=int(Priority.HIGH),
        sla_deadline=current_time + sla_offset,
        created_at=current_time,
        amount=amount,
        visible_summary="Customer claims refund was not processed for order #1001." if dispute_mode == "chargeback_open" else "Low-dollar pre-dispute inquiry needs an economics-aware decision.",
        visible_flags=flags,
        linked_records=[
            LinkedRecord(record_type=RecordType.ORDER, record_id=order.order_id, title="Order #1001"),
            LinkedRecord(record_type=RecordType.CUSTOMER, record_id=customer.customer_id, title="Customer profile"),
            LinkedRecord(record_type=RecordType.SHIPPING, record_id=shipping.shipment_id, title="Shipment record"),
            LinkedRecord(record_type=RecordType.PAYMENT, record_id=payment.payment_id, title="Payment capture"),
            LinkedRecord(record_type=RecordType.DISPUTE, record_id=dispute.dispute_id, title="Dispute record"),
        ],
        required_check_names=["review_order", "review_customer", "review_policy"],
        checks_required=3,
        evidence_types_available=["order_history", "customer_profile", "refund_policy", "fraud_signals", "shipping_status"],
        evidence_items_available=5,
        notifications_required=1,
        policy_id="refund_policy",
        allowed_escalation_queues=[TargetQueue.MANAGER_REVIEW, TargetQueue.FRAUD_TEAM],
        customer_id=customer.customer_id,
        requires_customer_notification=True,
        active_queue="refund_ops",
        hidden=CaseHiddenState(
            hidden_follow_up_latency_minutes=20,
            true_dispute_should_accept=fraud_risk > 0.7 or amount <= 75,
            true_fraud_risk=fraud_risk,
            true_downstream_loss=round(amount * 0.36, 2) if fraud_risk > 0.7 else 0.0,
        ),
        workflow=RefundWorkflowState(
            sla_total=sla_offset,
            refund_threshold=500.0,
            pre_dispute_type=PreDisputeType.NONE if dispute_mode == "chargeback_open" else PreDisputeType.INQUIRY,
            pre_dispute_due_at=None if dispute_mode == "chargeback_open" else current_time + max(10, sla_offset // 2),
            dispute_workflow_status=dispute.status,
            dispute_stage=DisputeStage.CHARGEBACK_OPEN if dispute_mode == "chargeback_open" else DisputeStage.INQUIRY,
            dispute_resolution=DisputeResolution.PENDING,
            dispute_fee=15.0,
            chargeback_received_at=current_time - 5 if dispute_mode == "chargeback_open" else None,
            representment_due_at=current_time + 18 if dispute_mode == "chargeback_open" else None,
            chargeback_amount=amount,
            merchant_dispute_ratio_30d=0.004 if fraud_risk <= 0.5 else (0.0065 if fraud_risk <= 0.7 else 0.0105),
            merchant_fraud_ratio_30d=0.0015 if fraud_risk <= 0.5 else (0.003 if fraud_risk <= 0.7 else 0.0065),
            merchant_negative_balance=fraud_risk > 0.7,
            merchant_age_days=45 if fraud_risk > 0.7 else (120 if fraud_risk > 0.4 else 365),
            merchant_volume_change_7d=-0.35 if fraud_risk > 0.7 else (-0.18 if fraud_risk > 0.4 else -0.05),
            avg_fulfillment_days=6 if fraud_risk > 0.7 else (4 if fraud_risk > 0.4 else 2),
        ),
    )
    case.workflow.recompute_risk_state()
    return case, records


def _invoice_case(
    current_time: int,
    task_id: TaskId,
    duplicate: bool,
    latency_minutes: int,
    *,
    error_type: str = "true_duplicate",
    sla_tightness: str = "standard",
) -> tuple[CaseState, RecordStore]:
    records = RecordStore()
    records.policies["invoice_policy"] = load_policy("invoice_policy")
    invoice_total = {
        "true_duplicate": 12450,
        "tolerance_band": 12450,
        "missing_receipt": 9800,
        "overbilling": 15800,
    }.get(error_type, 12450)
    invoice = InvoiceRecord(
        invoice_id="inv_2001",
        invoice_number="INV-2001",
        status="open",
        vendor_name="Acme Supply",
        vendor_id="vendor_1",
        vendor_address="101 Vendor Way",
        po_reference="PO-2001",
        currency="USD",
        created_at=current_time - 40,
        due_date=current_time + 180,
        line_items=[
            InvoiceLineItem(
                description="Packaging tape",
                sku="pk-tape",
                quantity=Decimal("100"),
                unit_price=124,
                amount=invoice_total - 50,
                subtotal=invoice_total - 50,
            )
        ],
        shipping_cost=ShippingCost(amount_subtotal=0, amount_tax=0, amount_total=0),
        subtotal=invoice_total - 50,
        total_discount=0,
        total_tax=50,
        total=invoice_total,
        amount_due=invoice_total,
        amount_paid=0,
        amount_remaining=invoice_total,
        payment_method="ach",
    )
    po = PurchaseOrder(
        po_number="PO-2001",
        document_type="standard",
        status="approved",
        vendor_id="vendor_1",
        vendor_name="Acme Supply",
        vendor_address="101 Vendor Way",
        currency="USD",
        payment_terms="NET30",
        total_net_value=max(5000, invoice_total - 50),
        total_gross_value=max(5050, invoice_total),
        requested_delivery_date=current_time - 10,
        approval_status="approved",
        approved_by="buyer_1",
        line_items=[],
    )
    sla_offset = {"tight": 60, "standard": 90}.get(sla_tightness, 90)
    visible_summary_map = {
        "true_duplicate": "Invoice INV-2001 flagged as potential duplicate with missing receipt.",
        "tolerance_band": "Invoice INV-2001 has a small variance that may be within policy tolerance.",
        "missing_receipt": "Invoice INV-2001 cannot be cleared because the goods receipt is still missing.",
        "overbilling": "Invoice INV-2001 appears overbilled and may need recovery before payment.",
    }
    visible_flags = ["duplicate_check"] if duplicate or error_type == "true_duplicate" else []
    if error_type == "missing_receipt":
        visible_flags.append("missing_receipt")
    if error_type in {"overbilling", "tolerance_band"}:
        visible_flags.append("variance")
    records.invoices[invoice.invoice_id] = invoice
    records.purchase_orders[po.po_number] = po
    case = CaseState(
        case_id="case_invoice_1",
        case_type=CaseType.INVOICE,
        task_id=task_id,
        priority=int(Priority.MEDIUM),
        sla_deadline=current_time + sla_offset,
        created_at=current_time,
        amount=round(invoice_total / 100, 2),
        visible_summary=visible_summary_map.get(error_type, visible_summary_map["true_duplicate"]),
        visible_flags=visible_flags,
        linked_records=[
            LinkedRecord(record_type=RecordType.INVOICE, record_id=invoice.invoice_id, title="Invoice INV-2001"),
            LinkedRecord(record_type=RecordType.PURCHASE_ORDER, record_id=po.po_number, title="PO-2001"),
        ],
        required_check_names=["review_invoice", "review_po", "review_receipt", "review_policy"],
        checks_required=4,
        evidence_types_available=["invoice_data", "purchase_order", "goods_receipt", "vendor_profile", "mismatch_policy"],
        evidence_items_available=5,
        notifications_required=1,
        policy_id="invoice_policy",
        allowed_escalation_queues=[TargetQueue.SENIOR_OPS, TargetQueue.MANAGER_REVIEW],
        vendor_id="vendor_1",
        pending_info_fields=["goods_receipt"] if error_type == "missing_receipt" else [],
        active_queue="ap_review",
        hidden=CaseHiddenState(
            hidden_response_latency_minutes=latency_minutes,
            hidden_follow_up_latency_minutes=35,
            true_is_duplicate=duplicate,
            true_variance_within_tolerance=error_type == "tolerance_band",
        ),
        workflow=InvoiceWorkflowState(
            sla_total=sla_offset,
            approval_threshold=100.0,
            duplicate_status=DuplicateStatus.SUSPECTED,
            credit_memo_status=CreditMemoStatus.NOT_REQUESTED,
            credit_memo_amount=24.5 if error_type in {"overbilling", "tolerance_band"} else None,
            approval_status=ApprovalStatus.NOT_REQUESTED,
            secondary_approval_required=True,
            approval_expected_outcome=ApprovalDecision.DENIED if duplicate else ApprovalDecision.APPROVED,
        ),
    )
    if error_type == "tolerance_band":
        case.workflow.match_status = MatchStatus.VARIANCE
        case.workflow.variance_amount = 2.5
    elif error_type == "overbilling":
        case.workflow.match_status = MatchStatus.VARIANCE
        case.workflow.variance_amount = 34.5
        case.hidden.true_recoverable_amount = 34.5
    elif error_type == "missing_receipt":
        case.workflow.match_status = MatchStatus.MISSING_RECEIPT
    return case, records


def _kyc_case(
    current_time: int,
    task_id: TaskId,
    doc_valid: bool,
    latency_minutes: int,
    *,
    sanctions_path: str = "false_positive",
    edd_required: bool = False,
    beneficial_owner_issue: bool = False,
    ofac_report_required: bool = False,
    sla_tightness: str = "standard",
) -> tuple[CaseState, RecordStore]:
    records = RecordStore()
    records.policies["kyc_policy"] = load_policy("kyc_policy")
    customer = CustomerRecord(
        customer_id="merchant_1",
        email="merchant@example.com",
        first_name="Avery",
        last_name="Patel",
        orders_count=300,
        total_spent=Decimal("0"),
        average_order_value=Decimal("0"),
    )
    verification = KYCVerification(
        session_id="kyc_3001",
        entity_id=customer.customer_id,
        entity_type="individual",
        status="requires_input",
        requirements_currently_due=["individual.verification.document"],
        requirements_past_due=[],
        error_code=None if doc_valid else "document_expired",
        legal_business_name="Patel Outdoor Supply LLC",
        incorporation_country="US",
        business_type="marketplace_seller",
        beneficial_owners=[
            BeneficialOwner(
                owner_id="bo_1",
                full_name="Avery Patel",
                ownership_percent=72.0,
                title="Founder",
                address="17 Market Street, Austin, TX",
            ),
            BeneficialOwner(
                owner_id="bo_2",
                full_name="Jordan Patel",
                ownership_percent=28.0,
                title="Operations Lead",
                address="" if beneficial_owner_issue else "90 River Road, Austin, TX",
            ),
        ],
    )
    records.customers[customer.customer_id] = customer
    records.kyc_verifications[verification.session_id] = verification
    sanctions_false_positive = sanctions_path == "false_positive"
    sanctions_match = sanctions_path == "confirmed_match"
    sla_offset = {"tight": 90, "standard": 120}.get(sla_tightness, 120)
    case = CaseState(
        case_id="case_kyc_1",
        case_type=CaseType.KYC,
        task_id=task_id,
        priority=int(Priority.HIGH),
        sla_deadline=current_time + sla_offset,
        created_at=current_time,
        amount=2400.0,
        visible_summary="Merchant payout blocked pending KYC and compliance review." if sanctions_path != "confirmed_match" else "Merchant flagged by sanctions screening with payout controls required.",
        visible_flags=["payout_hold"],
        linked_records=[
            LinkedRecord(record_type=RecordType.CUSTOMER, record_id=customer.customer_id, title="Merchant profile"),
            LinkedRecord(record_type=RecordType.KYC_DOCUMENT, record_id=verification.session_id, title="KYC verification"),
        ],
        required_check_names=["review_kyc_profile", "review_policy", "review_document", "review_sanctions", "review_beneficial_owner"],
        checks_required=5,
        evidence_types_available=["kyc_documents", "identity_verification", "compliance_policy", "customer_profile"],
        evidence_items_available=4,
        notifications_required=1,
        policy_id="kyc_policy",
        allowed_escalation_queues=[TargetQueue.COMPLIANCE, TargetQueue.SENIOR_OPS],
        customer_id=customer.customer_id,
        pending_info_fields=["individual.verification.document"],
        active_queue="kyc_review",
        hidden=CaseHiddenState(
            hidden_required_documents=["individual.verification.document"],
            hidden_correction_fields=["owners.1.address"] if beneficial_owner_issue else [],
            hidden_response_latency_minutes=latency_minutes,
            hidden_follow_up_latency_minutes=40,
            true_doc_valid=doc_valid,
            true_sanctions_match=sanctions_match,
            true_sanctions_false_positive=sanctions_false_positive,
            true_edd_required=edd_required,
            true_beneficial_owner_issue=beneficial_owner_issue,
            true_ofac_report_required=ofac_report_required,
        ),
        workflow=KYCWorkflowState(
            sla_total=sla_offset,
            verification_status=verification.status,
            requirements_due=verification.requirements_currently_due.copy(),
            payout_hold=True,
            kyc_stage=KYCStage.CURRENT_DUE,
            sanctions_status=SanctionsStatus.NOT_STARTED,
            edd_status=EDDStatus.NOT_STARTED if edd_required or beneficial_owner_issue else EDDStatus.NOT_REQUIRED,
            beneficial_owner_status=BeneficialOwnerStatus.NOT_STARTED,
            ofac_report_status=OFACReportStatus.NOT_REQUIRED,
            screening_match_confidence=0.82 if sanctions_false_positive else (0.99 if sanctions_match else 0.08),
        ),
    )
    return case, records


def build_task_state(task_id: TaskId | str, seed: int = 7, episode_id: str | None = None) -> WorldState:
    task_id = TaskId(task_id)
    template = _load_template(task_id)
    rng = random.Random(seed)
    state = _base_state(task_id, seed, episode_id)
    current_time = int(template["initial_time"])

    if task_id == TaskId.REFUND_EXCEPTION:
        case, records = _refund_case(
            rng,
            task_id,
            current_time,
            amount_tier=_choose_factor(template, rng, "amount_tier", "medium"),
            threshold_proximity=_choose_factor(template, rng, "threshold_proximity", "near_threshold"),
            fraud_risk_label=_choose_factor(template, rng, "fraud_risk", "low"),
            sla_tightness=_choose_factor(template, rng, "sla_tightness", "standard"),
            dispute_mode=_choose_factor(template, rng, "dispute_mode", "chargeback_open"),
        )
        state.records = records
        state.cases[case.case_id] = case
        state.queue_order = [case.case_id]
        state.metadata["max_steps"] = 20
    elif task_id == TaskId.INVOICE_PLUS_KYC:
        response_latency = _choose_factor(template, rng, "response_latency", "slow")
        invoice_error_type = _choose_factor(template, rng, "invoice_error_type", "true_duplicate")
        invoice_sla = _choose_factor(template, rng, "sla_tightness", "standard")
        invoice_case, invoice_records = _invoice_case(
            current_time,
            task_id,
            duplicate=invoice_error_type == "true_duplicate",
            latency_minutes=_latency_minutes(response_latency, fast=12, standard=30, slow=55, silent=120),
            error_type=invoice_error_type,
            sla_tightness=invoice_sla,
        )
        kyc_doc_validity = _choose_factor(template, rng, "doc_validity", "valid")
        kyc_sanctions_path = _choose_factor(template, rng, "sanctions_path", "false_positive")
        kyc_owner_review = _choose_factor(template, rng, "owner_review_path", "clean")
        kyc_case, kyc_records = _kyc_case(
            current_time,
            task_id,
            doc_valid=kyc_doc_validity != "invalid",
            latency_minutes=_latency_minutes(response_latency, fast=15, standard=35, slow=45, silent=90),
            sanctions_path=kyc_sanctions_path,
            edd_required=kyc_sanctions_path != "clear",
            beneficial_owner_issue=kyc_owner_review == "correction_required",
            ofac_report_required=kyc_sanctions_path == "confirmed_match",
            sla_tightness=invoice_sla,
        )
        state.records = RecordStore(
            customers={**invoice_records.customers, **kyc_records.customers},
            invoices=invoice_records.invoices,
            purchase_orders=invoice_records.purchase_orders,
            receipts=invoice_records.receipts,
            disputes=invoice_records.disputes,
            kyc_verifications=kyc_records.kyc_verifications,
            policies={
                **invoice_records.policies,
                **kyc_records.policies,
            },
            message_templates={
                **invoice_records.message_templates,
                **kyc_records.message_templates,
            },
            shipping=invoice_records.shipping,
            payments=invoice_records.payments,
        )
        _add_shared_templates(state.records)
        state.cases = {
            invoice_case.case_id: invoice_case,
            kyc_case.case_id: kyc_case,
        }
        state.queue_order = [invoice_case.case_id, kyc_case.case_id]
        state.metadata["max_steps"] = 35
    elif task_id == TaskId.QUEUE_TRIAGE:
        triage_fraud = _choose_factor(template, rng, "fraud_risk", "medium")
        triage_sla = _choose_factor(template, rng, "sla_tightness", "standard")
        queue_shock = _choose_factor(template, rng, "queue_shock", "mixed")
        triage_cases: list[CaseState] = []
        combined = RecordStore()
        for idx in range(2):
            refund_case, records = _refund_case(
                rng,
                task_id,
                current_time + idx,
                amount_tier=_choose_factor(template, rng, "amount_tier", "medium"),
                threshold_proximity="near_threshold" if idx == 0 else "far_below",
                fraud_risk_label=triage_fraud if idx == 0 else "low",
                sla_tightness="urgent" if idx == 0 else triage_sla,
                dispute_mode="chargeback_open" if idx == 0 else "inquiry",
            )
            refund_case.case_id = f"case_refund_{idx + 1}"
            refund_case.priority = int(Priority.CRITICAL if idx == 0 else Priority.HIGH)
            refund_case.sla_deadline = current_time + (25 if idx == 0 else 45)
            refund_case.visible_summary = f"Refund exception {idx + 1}"
            if idx == 0:
                refund_case.hidden.qa_sample_on_close = True
                refund_case.hidden.qa_sample_delay_minutes = 8
                refund_case.workflow.merchant_dispute_ratio_30d = 0.012
                refund_case.workflow.merchant_fraud_ratio_30d = 0.004
                refund_case.workflow.merchant_negative_balance = True
                refund_case.workflow.avg_fulfillment_days = 8
                refund_case.workflow.recompute_risk_state()
            if idx == 1:
                refund_case.amount = 45.0
                refund_case.visible_summary = "Low-dollar inquiry with weak recovery economics"
                refund_case.workflow.pre_dispute_type = PreDisputeType.INQUIRY
                refund_case.workflow.pre_dispute_due_at = current_time + 12
                refund_case.workflow.dispute_stage = DisputeStage.INQUIRY
                refund_case.hidden.true_dispute_should_accept = True
                refund_case.workflow.dispute_fee = 15.0
                refund_case.workflow.chargeback_received_at = None
                refund_case.workflow.representment_due_at = None
                refund_case.workflow.chargeback_amount = refund_case.amount
                refund_case.workflow.merchant_dispute_ratio_30d = 0.008
                refund_case.workflow.merchant_fraud_ratio_30d = 0.002
                refund_case.workflow.avg_fulfillment_days = 6
                refund_case.workflow.recompute_risk_state()
            triage_cases.append(refund_case)
            combined.customers.update(records.customers)
            combined.orders.update(records.orders)
            combined.shipping.update(records.shipping)
            combined.payments.update(records.payments)
            combined.disputes.update(records.disputes)
            combined.policies.update(records.policies)
            combined.message_templates.update(records.message_templates)
        for idx in range(2):
            invoice_case, records = _invoice_case(
                current_time + idx,
                task_id,
                duplicate=idx == 0,
                latency_minutes=20 if idx == 0 else 60,
                error_type="overbilling" if idx == 0 else "missing_receipt",
                sla_tightness="tight" if idx == 0 else "standard",
            )
            invoice_case.case_id = f"case_invoice_{idx + 1}"
            invoice_case.priority = int(Priority.HIGH if idx == 0 else Priority.MEDIUM)
            invoice_case.sla_deadline = current_time + (35 if idx == 0 else 70)
            invoice_case.visible_summary = f"Invoice exception {idx + 1}"
            if idx == 0:
                invoice_case.visible_summary = "Invoice variance likely needs credit memo and secondary approval"
                invoice_case.workflow.match_status = MatchStatus.VARIANCE
                invoice_case.workflow.variance_amount = 24.5
                invoice_case.hidden.true_variance_within_tolerance = False  # 24.5/124.5 = 19.7%, above 3% tolerance
                invoice_case.workflow.payment_batch_status = PaymentBatchStatus.SCHEDULED
                invoice_case.workflow.payment_batch_id = "batch_001"
                invoice_case.workflow.stop_payment_window_until = current_time + 50
                invoice_case.hidden.true_vendor_will_respond = True
                invoice_case.hidden.true_vendor_response_minutes = 20
                invoice_case.hidden.true_po_change_approved = True
                invoice_case.hidden.true_recoverable_amount = 24.5
            else:
                invoice_case.qa_required = True
                invoice_case.visible_flags = list(dict.fromkeys(invoice_case.visible_flags + ["qa_required"]))
            triage_cases.append(invoice_case)
            combined.invoices.update(records.invoices)
            combined.purchase_orders.update(records.purchase_orders)
            combined.policies.update(records.policies)
        kyc_case, records = _kyc_case(
            current_time,
            task_id,
            doc_valid=True,
            latency_minutes=25,
            sanctions_path="confirmed_match",
            edd_required=True,
            ofac_report_required=True,
            sla_tightness="tight",
        )
        kyc_case.case_id = "case_kyc_triage"
        kyc_case.priority = int(Priority.CRITICAL)
        kyc_case.sla_deadline = current_time + 30
        kyc_case.visible_summary = "Merchant flagged for sanctions review with payout controls required."
        triage_cases.append(kyc_case)
        combined.customers.update(records.customers)
        combined.kyc_verifications.update(records.kyc_verifications)
        combined.policies.update(records.policies)
        combined.message_templates.update(records.message_templates)
        _add_shared_templates(combined)
        state.records = combined
        state.cases = {case.case_id: case for case in triage_cases}
        state.queue_order = [case.case_id for case in triage_cases]
        state.metadata["max_steps"] = 70
        state.metadata["staffing_status"] = "normal"
        schedule_refund_2 = next(case for case in triage_cases if case.case_id == "case_refund_2")
        schedule_event_time = schedule_refund_2.workflow.pre_dispute_due_at
        if schedule_event_time is not None:
            state.scheduled_events.append(
                InquiryEscalatesToChargebackEvent(
                    at_time=schedule_event_time,
                    case_id=schedule_refund_2.case_id,
                )
            )
        state.scheduled_events.append(
            MonitoringThresholdBreachedEvent(
                at_time=current_time + 16,
                case_id="case_refund_1",
            )
        )
        arrival_case, arrival_records = _refund_case(rng, task_id, current_time + 8)
        arrival_case, arrival_records = _rename_refund_bundle(arrival_case, arrival_records, "wave1")
        arrival_case = arrival_case.model_copy(
            update={
                "case_id": "case_refund_wave_1",
                "priority": int(Priority.HIGH),
                "sla_deadline": current_time + 42,
                "visible_summary": "New refund exception arrived during peak queue load",
                "active_queue": "refund_ops",
            }
        )
        if queue_shock in {"arrival_wave", "mixed"}:
            state.scheduled_events.append(
                ArrivalWaveEvent(
                    at_time=current_time + 8,
                    wave_id="midshift_refund_spike",
                    incoming_cases=[arrival_case],
                    record_bundle=ArrivalWaveRecordBundle(
                        orders=arrival_records.orders,
                        customers=arrival_records.customers,
                        disputes=arrival_records.disputes,
                        policies=arrival_records.policies,
                        message_templates=arrival_records.message_templates,
                        shipping=arrival_records.shipping,
                        payments=arrival_records.payments,
                    ),
                )
            )
        if queue_shock in {"staffing_drop", "mixed"}:
            state.scheduled_events.append(
                StaffingDropEvent(
                    at_time=current_time + 18,
                    capacity_delta=1,
                    affected_owners=["analyst_2"],
                )
            )
        state.scheduled_events.sort(key=lambda event: event.at_time)

    elif task_id == TaskId.AP_PAYMENT_RUN:
        vendor_responsiveness = _choose_factor(template, rng, "vendor_responsiveness", "responsive")
        response_minutes = _latency_minutes(vendor_responsiveness, fast=20, standard=35, slow=55, silent=120)
        batch_timing = _choose_factor(template, rng, "payment_batch_timing", "pre_batch")
        recovery_path = _choose_factor(template, rng, "recovery_path", "credit_memo")
        stop_payment_success = _choose_factor(template, rng, "stop_payment_success", "mixed")
        # Case 1: Overpayment with scheduled payment batch — needs batch removal + vendor recovery
        case1, records1 = _invoice_case(
            current_time,
            task_id,
            duplicate=False,
            latency_minutes=response_minutes,
            error_type="overbilling",
            sla_tightness=_choose_factor(template, rng, "sla_tightness", "standard"),
        )
        case1.case_id = "case_ap_overpayment"
        case1.priority = int(Priority.HIGH)
        case1.sla_deadline = current_time + 60
        case1.amount = 245.0
        case1.visible_summary = "Invoice overpayment of $24.50 with payment batch running soon"
        case1.visible_flags = ["variance", "payment_batch_scheduled"]
        case1.workflow.match_status = MatchStatus.VARIANCE
        case1.workflow.variance_amount = 24.5
        case1.workflow.tolerance_percent = 3.0
        case1.workflow.payment_batch_id = "batch_001"
        case1.workflow.payment_batch_status = PaymentBatchStatus.SCHEDULED
        case1.workflow.write_off_threshold = 15.0
        case1.hidden.true_variance_within_tolerance = False
        case1.hidden.true_vendor_will_respond = vendor_responsiveness != "unresponsive"
        case1.hidden.true_vendor_response_minutes = response_minutes
        case1.hidden.true_recoverable_amount = 24.5
        if batch_timing == "in_batch":
            case1.workflow.payment_batch_status = PaymentBatchStatus.IN_PROGRESS
        elif batch_timing == "post_batch":
            case1.workflow.payment_batch_status = PaymentBatchStatus.COMPLETED

        # Case 2: Post-batch stop payment needed — duplicate detected after batch executed
        case2, records2 = _invoice_case(
            current_time + 1,
            task_id,
            duplicate=True,
            latency_minutes=response_minutes,
            error_type="true_duplicate",
            sla_tightness="tight",
        )
        case2.case_id = "case_ap_stop_payment"
        case2.priority = int(Priority.CRITICAL)
        case2.sla_deadline = current_time + 45
        case2.amount = 620.0
        case2.visible_summary = "Duplicate invoice paid in latest batch — stop payment may be possible"
        case2.visible_flags = ["duplicate_check", "payment_batch_completed"]
        case2.workflow.duplicate_status = DuplicateStatus.SUSPECTED
        case2.workflow.payment_batch_id = "batch_002"
        case2.workflow.payment_batch_status = PaymentBatchStatus.COMPLETED
        case2.workflow.stop_payment_window_until = current_time + 35
        case2.hidden.true_stop_payment_success = stop_payment_success == "likely" or bool(rng.randint(0, 1))

        # Case 3: Small balance write-off — vendor unresponsive, tiny variance
        case3, records3 = _invoice_case(
            current_time + 2,
            task_id,
            duplicate=False,
            latency_minutes=response_minutes,
            error_type="tolerance_band" if recovery_path == "write_off" else "overbilling",
            sla_tightness="standard",
        )
        case3.case_id = "case_ap_write_off"
        case3.priority = int(Priority.LOW)
        case3.sla_deadline = current_time + 90
        case3.amount = 52.0
        case3.visible_summary = "Small invoice variance — vendor unresponsive, write-off candidate"
        case3.visible_flags = ["variance", "small_balance"]
        case3.workflow.match_status = MatchStatus.VARIANCE
        case3.workflow.variance_amount = 8.50
        case3.workflow.tolerance_percent = 3.0
        case3.workflow.write_off_threshold = 25.0
        case3.hidden.true_variance_within_tolerance = False
        case3.hidden.true_vendor_will_respond = vendor_responsiveness == "responsive" and recovery_path != "write_off"
        case3.hidden.true_recoverable_amount = 8.5 if recovery_path != "write_off" else 0.0

        combined = RecordStore(
            invoices={**records1.invoices, **records2.invoices, **records3.invoices},
            purchase_orders={**records1.purchase_orders, **records2.purchase_orders, **records3.purchase_orders},
            receipts={**records1.receipts, **records2.receipts, **records3.receipts},
            policies={**records1.policies, **records2.policies, **records3.policies},
            message_templates={**records1.message_templates, **records2.message_templates, **records3.message_templates},
        )
        _add_shared_templates(combined)
        state.records = combined
        state.cases = {c.case_id: c for c in [case1, case2, case3]}
        state.queue_order = [case2.case_id, case1.case_id, case3.case_id]  # priority order
        state.metadata["max_steps"] = 45

        # Batch executes for case1 soon — agent must remove before it runs
        state.scheduled_events.append(
            PaymentBatchExecutedEvent(
                event_id="evt-batch-001",
                at_time=current_time + 15,
                case_id=case1.case_id,
                batch_id="batch_001",
            )
        )
        state.scheduled_events.sort(key=lambda event: event.at_time)

    state.current_time = current_time

    # Schedule SLA breach events for all cases so agents get visible warnings
    for case in state.cases.values():
        if not any(e.event_type == "sla_breach" and e.case_id == case.case_id for e in state.scheduled_events):
            state.scheduled_events.append(
                SLABreachEvent(
                    event_id=f"evt-sla-{case.case_id}",
                    at_time=case.sla_deadline,
                    case_id=case.case_id,
                )
            )
    state.scheduled_events.sort(key=lambda event: event.at_time)

    return state
