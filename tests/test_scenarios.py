from opsarena.domain.workflows.invoice import InvoiceWorkflowState
from opsarena.domain.workflows.kyc import KYCWorkflowState
from opsarena.domain.workflows.refund import DisputeStage, RefundWorkflowState
from opsarena.engine.scenarios import build_task_state
import json


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


def test_scenario_generators_produce_nontrivial_diversity_across_seeds():
    minimum_unique = {
        "refund_exception": 10,
        "invoice_plus_kyc": 10,
        "queue_triage": 4,
        "ap_payment_run": 10,
    }
    for task_id, threshold in minimum_unique.items():
        seen = set()
        for seed in range(1, 21):
            state = build_task_state(task_id, seed=seed)
            signature = []
            for case_id in sorted(state.cases):
                case = state.cases[case_id]
                signature.append(
                    (
                        case_id,
                        case.visible_summary,
                        tuple(case.visible_flags),
                        case.hidden.model_dump(),
                        case.workflow.model_dump(),
                    )
                )
            seen.add(json.dumps(signature, sort_keys=True, default=str))
        assert len(seen) >= threshold
