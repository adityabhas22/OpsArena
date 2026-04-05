from opsarena.domain.workflows.invoice import InvoiceWorkflowState
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


def test_queue_triage_seed_contains_richer_workflow_metadata():
    state = build_task_state("queue_triage", seed=7)
    assert state.metadata["claim_capacity"] == 2
    refund_workflow = state.cases["case_refund_2"].workflow
    invoice_workflow = state.cases["case_invoice_1"].workflow
    assert isinstance(refund_workflow, RefundWorkflowState)
    assert isinstance(invoice_workflow, InvoiceWorkflowState)
    assert refund_workflow.dispute_stage == DisputeStage.INQUIRY
    assert invoice_workflow.match_status.value == "variance"
