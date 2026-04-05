from opsarena.rewards import (
    CaseState,
    CaseType,
    QueueState,
    Resolution,
    compute_asymmetric_reward,
    compute_refund_reward,
    compute_shaping_reward,
)


def test_refund_reward_penalizes_fraud_approval():
    case = CaseState(
        case_id="c1",
        case_type=CaseType.REFUND,
        status="resolved",
        priority=2,
        sla_deadline=10,
        created_at=0,
        resolution=Resolution.APPROVED,
        amount=300.0,
        true_fraud_risk=0.9,
        evidence_items_available=3,
        evidence_items_gathered=2,
    )
    result = compute_refund_reward(case, QueueState())
    assert result["decision"] < 0


def test_asymmetric_reward_includes_label_and_total():
    result = compute_asymmetric_reward(CaseType.KYC, predicted_positive=False, actual_positive=True)
    assert result["label"] == "false_negative"
    assert result["total"] == result["classification"]


def test_shaping_reward_increases_after_progress():
    prev_case = CaseState(
        case_id="c1",
        case_type=CaseType.REFUND,
        status="open",
        priority=2,
        sla_deadline=10,
        created_at=0,
        evidence_items_available=3,
        checks_required=2,
    )
    case = prev_case.__class__(**prev_case.__dict__)
    case.evidence_items_gathered = 1
    case.checks_completed = 1
    reward = compute_shaping_reward(case, QueueState(cases=[case]), prev_case=prev_case, prev_queue=QueueState(cases=[prev_case]))
    assert reward > 0
