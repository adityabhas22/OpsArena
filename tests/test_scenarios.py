from opsarena.domain.workflows.invoice import InvoiceWorkflowState
from opsarena.domain.workflows.kyc import KYCWorkflowState
from opsarena.domain.workflows.refund import DisputeStage, RefundWorkflowState
from opsarena.engine.scenarios import build_task_state


def test_scenarios_are_seeded_and_deterministic():
    state_a = build_task_state("refund_exception", seed=11)
    state_b = build_task_state("refund_exception", seed=11)
    assert state_a.model_dump() == state_b.model_dump()


def test_medium_task_contains_invoice_and_kyc_cases():
    state = build_task_state("invoice_plus_kyc", seed=3)
    case_types = {case.case_type.value for case in state.cases.values()}
    assert case_types == {"invoice", "kyc"}
    kyc_workflow = state.cases["case_kyc_1"].workflow
    assert isinstance(kyc_workflow, KYCWorkflowState)
    assert kyc_workflow.sanctions_status.value == "not_started"
    assert kyc_workflow.edd_status.value == "not_started"
    assert state.cases["case_kyc_1"].hidden.true_sanctions_false_positive is True


def test_queue_triage_seed_contains_richer_workflow_metadata():
    state = build_task_state("queue_triage", seed=7)
    assert state.metadata["claim_capacity"] == 2
    assert state.metadata["agent_capacity"] == 2
    assert state.metadata["staffing_status"] == "normal"
    refund_workflow = state.cases["case_refund_2"].workflow
    invoice_workflow = state.cases["case_invoice_1"].workflow
    assert isinstance(refund_workflow, RefundWorkflowState)
    assert isinstance(invoice_workflow, InvoiceWorkflowState)
    assert refund_workflow.dispute_stage == DisputeStage.INQUIRY
    assert invoice_workflow.match_status.value == "variance"
    assert state.cases["case_invoice_2"].qa_required is True
    assert state.cases["case_refund_1"].hidden.qa_sample_on_close is True
    kyc_workflow = state.cases["case_kyc_triage"].workflow
    assert isinstance(kyc_workflow, KYCWorkflowState)
    assert state.cases["case_kyc_triage"].hidden.true_sanctions_match is True
    assert state.cases["case_kyc_triage"].hidden.true_ofac_report_required is True
    scheduled_event_types = {event.event_type for event in state.scheduled_events}
    assert "arrival_wave" in scheduled_event_types
    assert "staffing_drop" in scheduled_event_types
    assert "inquiry_escalates_to_chargeback" in scheduled_event_types
    assert "monitoring_threshold_breached" in scheduled_event_types
