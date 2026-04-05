from opsarena.domain.case import CaseState
from opsarena.domain.hidden import CaseHiddenState
from opsarena.domain.workflows.kyc import KYCWorkflowState, SanctionsStatus, EDDStatus, BeneficialOwnerStatus, OFACReportStatus
from opsarena.domain.workflows.refund import RefundWorkflowState
from opsarena.engine.state import QueueState
from opsarena.enums import CaseType, Resolution, TaskId
from opsarena.rewards import (
    compute_asymmetric_reward,
    compute_kyc_reward,
    compute_refund_reward,
    compute_shaping_reward,
)


def _refund_case(**overrides) -> CaseState:
    defaults = dict(
        case_id="c1",
        case_type=CaseType.REFUND,
        task_id=TaskId.REFUND_EXCEPTION,
        visible_summary="Test refund case",
        status="resolved",
        priority=2,
        sla_deadline=10,
        created_at=0,
        resolution=Resolution.APPROVED,
        amount=300.0,
        evidence_items_available=3,
        evidence_items_gathered=2,
        hidden=CaseHiddenState(true_fraud_risk=0.9),
        workflow=RefundWorkflowState(),
    )
    defaults.update(overrides)
    return CaseState(**defaults)


def _kyc_case(**overrides) -> CaseState:
    defaults = dict(
        case_id="kyc1",
        case_type=CaseType.KYC,
        task_id=TaskId.INVOICE_PLUS_KYC,
        visible_summary="Test KYC case",
        status="resolved",
        priority=1,
        sla_deadline=10,
        created_at=0,
        resolution=Resolution.APPROVED,
        hidden=CaseHiddenState(true_doc_valid=True),
        workflow=KYCWorkflowState(kyc_complete=True),
    )
    defaults.update(overrides)
    return CaseState(**defaults)


def test_refund_reward_penalizes_fraud_approval():
    case = _refund_case()
    result = compute_refund_reward(case, QueueState())
    assert result["decision"] < 0


def test_asymmetric_reward_includes_label_and_total():
    result = compute_asymmetric_reward(CaseType.KYC, predicted_positive=False, actual_positive=True)
    assert result["label"] == "false_negative"
    assert result["total"] == result["classification"]


def test_shaping_reward_increases_after_progress():
    prev_case = _refund_case(
        status="open",
        resolution=Resolution.PENDING,
        evidence_items_gathered=0,
        checks_required=2,
        hidden=CaseHiddenState(true_fraud_risk=0.0),
    )
    case = _refund_case(
        status="open",
        resolution=Resolution.PENDING,
        evidence_items_gathered=1,
        checks_required=2,
        hidden=CaseHiddenState(true_fraud_risk=0.0),
    )
    case.checks_completed = 1
    reward = compute_shaping_reward(
        case, QueueState(cases=[case]),
        prev_case=prev_case, prev_queue=QueueState(cases=[prev_case]),
    )
    assert reward > 0


def test_kyc_reward_penalizes_approving_unresolved_sanctions_case():
    case = _kyc_case(
        workflow=KYCWorkflowState(
            kyc_complete=True,
            sanctions_status=SanctionsStatus.CONFIRMED_MATCH,
            edd_status=EDDStatus.IN_PROGRESS,
            beneficial_owner_status=BeneficialOwnerStatus.NEEDS_CORRECTION,
            ofac_report_status=OFACReportStatus.PENDING,
            payments_frozen=True,
        ),
    )
    result = compute_kyc_reward(case, QueueState())
    assert result["compliance"] < 0
