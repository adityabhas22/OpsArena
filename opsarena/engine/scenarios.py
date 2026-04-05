from __future__ import annotations

import random
from decimal import Decimal
from pathlib import Path
import yaml

from opsarena.documents import (
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
from opsarena.domain.events import ArrivalWaveEvent, ArrivalWaveRecordBundle, StaffingDropEvent
from opsarena.domain.hidden import CaseHiddenState
from opsarena.domain.workflows.invoice import (
    ApprovalDecision,
    ApprovalStatus,
    CreditMemoStatus,
    DuplicateStatus,
    InvoiceWorkflowState,
)
from opsarena.domain.workflows.kyc import KYCStage, KYCWorkflowState
from opsarena.domain.workflows.refund import DisputeResolution, DisputeStage, RefundWorkflowState
from opsarena.engine.policies import load_policy
from opsarena.engine.state import CaseState, RecordStore, WorldState
from opsarena.enums import CaseType, MatchStatus, Priority, RecordType, TargetQueue, TaskId

ROOT = Path(__file__).resolve().parents[2]


def _load_template(task_id: TaskId) -> dict:
    mapping = {
        TaskId.REFUND_EXCEPTION: ROOT / "data" / "scenario_templates" / "task1_refund" / "template.yaml",
        TaskId.INVOICE_PLUS_KYC: ROOT / "data" / "scenario_templates" / "task2_invoice_kyc" / "template.yaml",
        TaskId.QUEUE_TRIAGE: ROOT / "data" / "scenario_templates" / "task3_triage" / "template.yaml",
    }
    return yaml.safe_load(mapping[task_id].read_text())


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


def _refund_case(rng: random.Random, task_id: TaskId, current_time: int) -> tuple[CaseState, RecordStore]:
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

    fraud_risk = 0.82 if rng.random() < 0.35 else 0.22
    flags = ["near-threshold"] if 300 <= 350 <= 500 else []
    if fraud_risk > 0.7:
        flags.append("velocity_alert")

    case = CaseState(
        case_id="case_refund_1",
        case_type=CaseType.REFUND,
        task_id=task_id,
        priority=int(Priority.HIGH),
        sla_deadline=current_time + 60,
        created_at=current_time,
        amount=350.0,
        visible_summary="Customer claims refund was not processed for order #1001.",
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
            true_fraud_risk=fraud_risk,
            true_downstream_loss=125.0 if fraud_risk > 0.7 else 0.0,
        ),
        workflow=RefundWorkflowState(
            sla_total=60,
            refund_threshold=500.0,
            dispute_workflow_status=dispute.status,
            dispute_stage=DisputeStage.CHARGEBACK_OPEN,
            dispute_resolution=DisputeResolution.PENDING,
            dispute_fee=15.0,
            dispute_should_accept=fraud_risk > 0.7,
        ),
    )
    return case, records


def _invoice_case(current_time: int, task_id: TaskId, duplicate: bool, latency_minutes: int) -> tuple[CaseState, RecordStore]:
    records = RecordStore()
    records.policies["invoice_policy"] = load_policy("invoice_policy")
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
                amount=12400,
                subtotal=12400,
            )
        ],
        shipping_cost=ShippingCost(amount_subtotal=0, amount_tax=0, amount_total=0),
        subtotal=12400,
        total_discount=0,
        total_tax=50,
        total=12450,
        amount_due=12450,
        amount_paid=0,
        amount_remaining=12450,
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
        total_net_value=12400,
        total_gross_value=12450,
        requested_delivery_date=current_time - 10,
        approval_status="approved",
        approved_by="buyer_1",
        line_items=[],
    )
    records.invoices[invoice.invoice_id] = invoice
    records.purchase_orders[po.po_number] = po
    case = CaseState(
        case_id="case_invoice_1",
        case_type=CaseType.INVOICE,
        task_id=task_id,
        priority=int(Priority.MEDIUM),
        sla_deadline=current_time + 90,
        created_at=current_time,
        amount=124.5,
        visible_summary="Invoice INV-2001 flagged as potential duplicate with missing receipt.",
        visible_flags=["duplicate_check"],
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
        pending_info_fields=["goods_receipt"],
        active_queue="ap_review",
        hidden=CaseHiddenState(
            hidden_response_latency_minutes=latency_minutes,
            hidden_follow_up_latency_minutes=35,
            true_is_duplicate=duplicate,
        ),
        workflow=InvoiceWorkflowState(
            sla_total=90,
            approval_threshold=100.0,
            duplicate_status=DuplicateStatus.SUSPECTED,
            credit_memo_status=CreditMemoStatus.NOT_REQUESTED,
            credit_memo_amount=24.5,
            approval_status=ApprovalStatus.NOT_REQUESTED,
            secondary_approval_required=True,
            approval_expected_outcome=ApprovalDecision.DENIED if duplicate else ApprovalDecision.APPROVED,
        ),
    )
    return case, records


def _kyc_case(current_time: int, task_id: TaskId, doc_valid: bool, latency_minutes: int) -> tuple[CaseState, RecordStore]:
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
    )
    records.customers[customer.customer_id] = customer
    records.kyc_verifications[verification.session_id] = verification
    case = CaseState(
        case_id="case_kyc_1",
        case_type=CaseType.KYC,
        task_id=task_id,
        priority=int(Priority.HIGH),
        sla_deadline=current_time + 120,
        created_at=current_time,
        amount=2400.0,
        visible_summary="Merchant payout blocked until ID verification is complete.",
        visible_flags=["payout_hold"],
        linked_records=[
            LinkedRecord(record_type=RecordType.CUSTOMER, record_id=customer.customer_id, title="Merchant profile"),
            LinkedRecord(record_type=RecordType.KYC_DOCUMENT, record_id=verification.session_id, title="KYC verification"),
        ],
        required_check_names=["review_kyc_profile", "review_policy", "review_document"],
        checks_required=3,
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
            hidden_response_latency_minutes=latency_minutes,
            hidden_follow_up_latency_minutes=40,
            true_doc_valid=doc_valid,
        ),
        workflow=KYCWorkflowState(
            sla_total=120,
            verification_status=verification.status,
            requirements_due=verification.requirements_currently_due.copy(),
            payout_hold=True,
            kyc_stage=KYCStage.CURRENT_DUE,
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
        case, records = _refund_case(rng, task_id, current_time)
        state.records = records
        state.cases[case.case_id] = case
        state.queue_order = [case.case_id]
        state.metadata["max_steps"] = 20
    elif task_id == TaskId.INVOICE_PLUS_KYC:
        invoice_case, invoice_records = _invoice_case(
            current_time, task_id, duplicate=bool(rng.randint(0, 1)), latency_minutes=30
        )
        kyc_case, kyc_records = _kyc_case(current_time, task_id, doc_valid=bool(rng.randint(0, 1)), latency_minutes=45)
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
    else:
        triage_cases: list[CaseState] = []
        combined = RecordStore()
        for idx in range(2):
            refund_case, records = _refund_case(rng, task_id, current_time + idx)
            refund_case.case_id = f"case_refund_{idx + 1}"
            refund_case.priority = int(Priority.CRITICAL if idx == 0 else Priority.HIGH)
            refund_case.sla_deadline = current_time + (25 if idx == 0 else 45)
            refund_case.visible_summary = f"Refund exception {idx + 1}"
            if idx == 0:
                refund_case.hidden.qa_sample_on_close = True
                refund_case.hidden.qa_sample_delay_minutes = 8
            if idx == 1:
                refund_case.amount = 45.0
                refund_case.visible_summary = "Low-dollar inquiry with weak recovery economics"
                refund_case.workflow.dispute_stage = DisputeStage.INQUIRY
                refund_case.workflow.dispute_should_accept = True
                refund_case.workflow.dispute_fee = 15.0
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
            )
            invoice_case.case_id = f"case_invoice_{idx + 1}"
            invoice_case.priority = int(Priority.HIGH if idx == 0 else Priority.MEDIUM)
            invoice_case.sla_deadline = current_time + (35 if idx == 0 else 70)
            invoice_case.visible_summary = f"Invoice exception {idx + 1}"
            if idx == 0:
                invoice_case.visible_summary = "Invoice variance likely needs credit memo and secondary approval"
                invoice_case.workflow.match_status = MatchStatus.VARIANCE
                invoice_case.workflow.variance_amount = 24.5
            else:
                invoice_case.qa_required = True
                invoice_case.visible_flags = list(dict.fromkeys(invoice_case.visible_flags + ["qa_required"]))
            triage_cases.append(invoice_case)
            combined.invoices.update(records.invoices)
            combined.purchase_orders.update(records.purchase_orders)
            combined.policies.update(records.policies)
        kyc_case, records = _kyc_case(current_time, task_id, doc_valid=True, latency_minutes=25)
        kyc_case.case_id = "case_kyc_triage"
        kyc_case.priority = int(Priority.CRITICAL)
        kyc_case.sla_deadline = current_time + 30
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
        state.scheduled_events.append(
            StaffingDropEvent(
                at_time=current_time + 18,
                capacity_delta=1,
                affected_owners=["analyst_2"],
            )
        )
        state.scheduled_events.sort(key=lambda event: event.at_time)

    state.current_time = current_time
    return state
