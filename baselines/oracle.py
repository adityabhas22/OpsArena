from __future__ import annotations

from opsarena.domain.workflows.invoice import InvoiceWorkflowState
from opsarena.domain.workflows.kyc import KYCWorkflowState
from opsarena.domain.workflows.refund import RefundWorkflowState
from server.environment import OpsArenaEnvironment
from opsarena.models import (
    AcceptDisputeAction,
    AdvanceClockAction,
    ApproveQAAction,
    ApproveAction,
    CloseCaseAction,
    SendToQAAction,
    OpenCaseAction,
    QueryPolicyAction,
    RejectAction,
    RebalanceQueueAction,
    ReleasePaymentHoldAction,
    RequestInfoAction,
    SendForSecondaryApprovalAction,
    SendMessageAction,
    ViewRecordAction,
)


def _next_active_case_id(env: OpsArenaEnvironment) -> str | None:
    assert env._state is not None
    for case_id in env._state.queue_order:
        if env._state.cases[case_id].status != "closed":
            return case_id
    return None


def run_oracle(task_id: str, seed: int = 7) -> dict:
    env = OpsArenaEnvironment()
    env.reset(task_id=task_id, seed=seed)
    assert env._state is not None
    safety_limit = env._state.metadata.get("max_steps", 40) * 4
    iterations = 0

    while not env._is_done() and iterations < safety_limit:
        iterations += 1
        assert env._state is not None
        if task_id == "queue_triage" and env._state.queue_state().unassigned_count > 0:
            env.step(RebalanceQueueAction(assignee_pool=["analyst_1", "analyst_2"], max_cases=3, rebalance_strategy="sla_priority"))

        case_id = _next_active_case_id(env)
        if case_id is None:
            if env._state.scheduled_events:
                next_due = min(event.at_time for event in env._state.scheduled_events)
                env.step(AdvanceClockAction(minutes=max(1, next_due - env._state.current_time)))
                continue
            break

        case = env._state.cases[case_id]

        if getattr(case.qa_status, "value", case.qa_status) == "pending":
            env.step(OpenCaseAction(case_id=case_id))
            env.step(ApproveQAAction(case_id=case_id, assignee_type="qa_reviewer"))
            env.step(CloseCaseAction(case_id=case_id, resolution_code="oracle_complete"))
            continue

        env.step(OpenCaseAction(case_id=case_id))
        if case.case_type.value == "refund":
            workflow = case.workflow
            assert isinstance(workflow, RefundWorkflowState)
            if case.resolution.value != "pending":
                pass
            elif workflow.dispute_should_accept and case.amount <= 75:
                env.step(AcceptDisputeAction(case_id=case_id))
            elif case.hidden.true_fraud_risk > 0.7:
                env.step(RejectAction(case_id=case_id, reason_code="suspicious_pattern"))
            else:
                env.step(QueryPolicyAction(policy_id="refund_policy"))
                env.step(ApproveAction(case_id=case_id))
            if not env._state.cases[case_id].customer_notified:
                env.step(SendMessageAction(case_id=case_id, template_id="refund_approved", slots={"amount": str(case.amount), "order_id": case.linked_records[0].record_id}))
            if env._state.cases[case_id].qa_required:
                env.step(SendToQAAction(case_id=case_id, assignee_type="qa_reviewer"))
                env.step(ApproveQAAction(case_id=case_id, assignee_type="qa_reviewer"))
            env.step(CloseCaseAction(case_id=case_id, resolution_code="oracle_complete"))
        elif case.case_type.value == "invoice":
            workflow = case.workflow
            assert isinstance(workflow, InvoiceWorkflowState)
            if case.resolution.value != "pending":
                pass
            elif case.hidden.true_is_duplicate:
                env.step(RejectAction(case_id=case_id, reason_code="duplicate_match"))
            else:
                env.step(ViewRecordAction(record_type="invoice", record_id=case.linked_records[0].record_id))
                env.step(ViewRecordAction(record_type="purchase_order", record_id=case.linked_records[1].record_id))
                if case.pending_info_fields:
                    env.step(RequestInfoAction(case_id=case_id, field_name="goods_receipt"))
                    env.step(AdvanceClockAction(minutes=case.hidden.hidden_response_latency_minutes or 30))
                    receipt = next(record.record_id for record in env._state.cases[case_id].linked_records if record.record_type.value == "receipt")
                    env.step(ViewRecordAction(record_type="receipt", record_id=receipt))
                env.step(QueryPolicyAction(policy_id="invoice_policy"))
                if workflow.secondary_approval_required:
                    env.step(SendForSecondaryApprovalAction(case_id=case_id, reason_code="threshold_exceeded"))
                    env.step(AdvanceClockAction(minutes=15))
                invoice_workflow = env._state.cases[case_id].workflow
                assert isinstance(invoice_workflow, InvoiceWorkflowState)
                if invoice_workflow.payment_hold:
                    env.step(ReleasePaymentHoldAction(case_id=case_id))
                env.step(ApproveAction(case_id=case_id))
            if env._state.cases[case_id].qa_required:
                env.step(SendToQAAction(case_id=case_id, assignee_type="qa_reviewer"))
                env.step(ApproveQAAction(case_id=case_id, assignee_type="qa_reviewer"))
            env.step(CloseCaseAction(case_id=case_id, resolution_code="oracle_complete"))
        else:
            workflow = case.workflow
            assert isinstance(workflow, KYCWorkflowState)
            if case.resolution.value != "pending":
                pass
            elif not case.hidden.true_doc_valid:
                env.step(RejectAction(case_id=case_id, reason_code="invalid_document"))
            else:
                env.step(QueryPolicyAction(policy_id="kyc_policy"))
                if not workflow.kyc_complete:
                    env.step(RequestInfoAction(case_id=case_id, field_name="individual.verification.document"))
                    env.step(AdvanceClockAction(minutes=case.hidden.hidden_response_latency_minutes or 45))
                env.step(ViewRecordAction(record_type="kyc_document", record_id=case.linked_records[-1].record_id))
                env.step(ApproveAction(case_id=case_id))
            if env._state.cases[case_id].qa_required:
                env.step(SendToQAAction(case_id=case_id, assignee_type="qa_reviewer"))
                env.step(ApproveQAAction(case_id=case_id, assignee_type="qa_reviewer"))
            env.step(CloseCaseAction(case_id=case_id, resolution_code="oracle_complete"))
    return env.state.model_dump()
