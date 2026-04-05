from __future__ import annotations

from opsarena.models import (
    AdvanceClockAction,
    ApproveAction,
    CloseCaseAction,
    OpenCaseAction,
    QueryPolicyAction,
    RejectAction,
    RequestInfoAction,
    SendMessageAction,
    ViewRecordAction,
)


def select_action(observation):
    detail = observation.case_detail
    if detail is None:
        queue = observation.queue_view or []
        if not queue:
            return AdvanceClockAction(minutes=5)
        next_case = sorted(queue, key=lambda item: (item.priority, item.sla_remaining_minutes))[0]
        return OpenCaseAction(case_id=next_case.case_id)

    completed = set(detail.checks_completed)
    linked = {record.record_type: record.record_id for record in detail.linked_records}

    if detail.case_type == "refund":
        if detail.status == "resolved":
            return CloseCaseAction(case_id=detail.case_id, resolution_code="completed")
        if "review_order" not in completed and "order" in linked:
            return ViewRecordAction(record_type="order", record_id=linked["order"])
        if "review_customer" not in completed and "customer" in linked:
            return ViewRecordAction(record_type="customer", record_id=linked["customer"])
        if "review_policy" not in completed:
            return QueryPolicyAction(policy_id="refund_policy")
        if "velocity_alert" in detail.visible_flags:
            return RejectAction(case_id=detail.case_id, reason_code="suspicious_pattern")
        if not detail.communication_log:
            return SendMessageAction(
                case_id=detail.case_id,
                template_id="refund_approved",
                slots={"amount": str(detail.amount or 0), "order_id": linked.get("order", detail.case_id)},
            )
        return ApproveAction(case_id=detail.case_id)

    if detail.case_type == "invoice":
        if detail.status == "resolved":
            return CloseCaseAction(case_id=detail.case_id, resolution_code="completed")
        if "review_invoice" not in completed and "invoice" in linked:
            return ViewRecordAction(record_type="invoice", record_id=linked["invoice"])
        if "review_po" not in completed and "purchase_order" in linked:
            return ViewRecordAction(record_type="purchase_order", record_id=linked["purchase_order"])
        if "review_receipt" not in completed:
            if "receipt" in linked:
                return ViewRecordAction(record_type="receipt", record_id=linked["receipt"])
            return RequestInfoAction(case_id=detail.case_id, field_name="goods_receipt", template_id="info_request")
        if "review_policy" not in completed:
            return QueryPolicyAction(policy_id="invoice_policy")
        return RejectAction(case_id=detail.case_id, reason_code="duplicate_match") if "duplicate_check" in detail.visible_flags else ApproveAction(case_id=detail.case_id)

    if detail.case_type == "kyc":
        if detail.status == "resolved":
            return CloseCaseAction(case_id=detail.case_id, resolution_code="completed")
        if "review_policy" not in completed:
            return QueryPolicyAction(policy_id="kyc_policy")
        if "review_document" not in completed and "kyc_document" in linked:
            return ViewRecordAction(record_type="kyc_document", record_id=linked["kyc_document"])
        if "review_document" not in completed:
            return RequestInfoAction(case_id=detail.case_id, field_name="individual.verification.document", template_id="info_request")
        return ApproveAction(case_id=detail.case_id)

    if detail.status == "resolved":
        return CloseCaseAction(case_id=detail.case_id, resolution_code="completed")

    return AdvanceClockAction(minutes=15)
