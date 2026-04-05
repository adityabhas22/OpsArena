"""
OpsArena Reward Functions
=========================

Diverse, workflow-specific reward functions for enterprise exception handling RL.

Design principles:
1. Per-workflow reward structures (not one-size-fits-all)
2. Time-varying urgency signals (non-linear SLA pressure)
3. Information value rewards (decision-theoretic VOI)
4. Queue-level portfolio optimization (not just case-level)
5. Asymmetric error costs (calibrated to real ecommerce metrics)
6. Compound/cascading cross-case dependency rewards

All formulas use concrete numerical calibrations based on typical
ecommerce operations metrics. Weights are configurable via YAML.

References:
- Ng, Harada, Russell: potential-based shaping preserves optimal policy
- Howard (1966): Value of Information in decision theory
- Markowitz portfolio theory adapted to case queue optimization
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from opsarena.enums import CaseType, Resolution


# ---------------------------------------------------------------------------
# Structural protocols for state objects passed to reward functions.
# These define the contract without coupling to Pydantic models.
# ---------------------------------------------------------------------------

@runtime_checkable
class HiddenStateProto(Protocol):
    true_fraud_risk: float
    true_is_duplicate: bool
    true_doc_valid: bool
    true_downstream_loss: float
    true_dispute_should_accept: bool
    true_sanctions_match: bool
    true_sanctions_false_positive: bool
    true_edd_required: bool
    true_ofac_report_required: bool


@runtime_checkable
class CaseProto(Protocol):
    case_id: str
    case_type: CaseType
    status: str
    priority: int
    sla_deadline: int
    created_at: int
    resolution: Resolution
    amount: float
    checks_required: int
    checks_completed: int
    evidence_items_available: int
    evidence_items_gathered: int
    evidence_types_gathered: list[str]
    notifications_required: int
    notifications_sent: int
    policy_checked: bool
    customer_notified: bool
    escalation_justified: bool
    qa_required: bool
    hidden: HiddenStateProto
    workflow: Any
    visible_flags: list[str]


@runtime_checkable
class QueueProto(Protocol):
    cases: list[Any]
    current_time: int
    escalation_queue_load: int
    escalation_queue_capacity: int
    total_cases_resolved: int
    total_sla_breaches: int
    exception_queue_size: int
    unassigned_count: int
    overdue_follow_ups: int


@runtime_checkable
class EpisodeMetricsProto(Protocol):
    cases_resolved: int
    cases_breached: int
    chargebacks: int
    compliance_violations: int


# ---------------------------------------------------------------------------
# Typed accessors for hidden and workflow attributes
# ---------------------------------------------------------------------------

def _hidden(case: CaseProto, attr: str, default: Any = None) -> Any:
    """Access a hidden-state attribute."""
    return getattr(case.hidden, attr, default)


def _wf(case: CaseProto, attr: str, default: Any = None) -> Any:
    """Access a workflow-state attribute, stripping enum .value if present."""
    val = getattr(case.workflow, attr, default)
    return getattr(val, "value", val)


# ============================================================================
# SECTION 1: PER-WORKFLOW REWARD FUNCTIONS
# ============================================================================

# ---------------------------------------------------------------------------
# 1A. Refund Processing Reward
# ---------------------------------------------------------------------------
#
# Business context:
#   - Average refund value: $85 (ecommerce median)
#   - Chargeback cost: $15-$100 fee + merchandise + operational cost
#   - Customer lifetime value at risk from false rejection: ~$800-2000
#   - Fraud rate: ~1.5% of transactions
#   - Chargeback rate: ~0.6% of transactions
#
# The reward must capture three competing objectives:
#   (a) Fraud detection value: catching real fraud saves chargeback costs
#   (b) Customer retention: wrongly rejecting legitimate customers destroys CLTV
#   (c) Chargeback avoidance: approving fraud creates direct financial loss
#
# Formula:
#   R_refund = R_decision + R_process + R_efficiency
#
#   R_decision:
#     Correct approval (not fraud):  +V_retain(customer)
#     Correct rejection (is fraud):  +V_fraud_saved(amount)
#     False positive (reject legit): -V_cltv_loss(customer)
#     False negative (approve fraud):-V_chargeback(amount)
#
#   R_process:
#     Policy checked before decision:  +2.0
#     Customer notified:               +1.5
#     Evidence sufficiency bonus:      +min(3.0, 3.0 * evidence_ratio)
#
#   R_efficiency:
#     SLA penalty:       see time_pressure_reward()
#     Unnecessary escalation: -3.0
#     Tool cost:         -0.1 per call

@dataclass
class RefundRewardParams:
    """Calibrated from typical ecommerce refund operations."""
    # Decision outcome values
    correct_approval_base: float = 8.0
    correct_rejection_base: float = 12.0        # higher because fraud catch is valuable
    false_positive_cltv_loss: float = -15.0      # lost future revenue from angry customer
    false_negative_chargeback_base: float = -25.0  # chargeback fee + loss + operational cost
    # Amount multipliers (scale with case value)
    amount_multiplier_approval: float = 0.02     # 2% of amount as bonus
    amount_multiplier_chargeback: float = 0.15   # 15% of amount as penalty (fee + loss)
    # Process bonuses
    policy_check_bonus: float = 2.0
    notification_bonus: float = 1.5
    evidence_max_bonus: float = 3.0
    # Efficiency penalties
    unnecessary_escalation: float = -3.0
    tool_call_cost: float = -0.1


def compute_refund_reward(
    case: CaseProto,
    queue: QueueProto,
    params: RefundRewardParams | None = None,
) -> dict[str, float]:
    """
    Compute reward for a refund case resolution.

    Returns a breakdown dict so each component is auditable.
    """
    p = params or RefundRewardParams()
    breakdown: dict[str, float] = {}

    # --- Decision reward ---
    true_fraud_risk = float(_hidden(case, "true_fraud_risk", 0.0))
    is_fraud = true_fraud_risk > 0.5  # threshold for "actually fraudulent"

    # Check if this is a dispute-acceptance scenario where accepting is
    # the economically correct action (e.g., low-dollar dispute where
    # fighting costs more than accepting).
    dispute_should_accept = bool(_hidden(case, "true_dispute_should_accept", False))

    if case.resolution == Resolution.APPROVED:
        if dispute_should_accept:
            # Correct dispute acceptance: economically rational decision
            breakdown["decision"] = p.correct_approval_base * 0.8
        elif not is_fraud:
            # Correct approval: retain customer
            base = p.correct_approval_base + p.amount_multiplier_approval * case.amount
            # Scale by customer quality (proxy: inverse of fraud risk)
            retention_factor = 1.0 + 0.5 * (1.0 - true_fraud_risk)
            breakdown["decision"] = base * retention_factor
        else:
            # False negative: approved fraud -> chargeback
            breakdown["decision"] = (
                p.false_negative_chargeback_base
                - p.amount_multiplier_chargeback * case.amount
            )

    elif case.resolution == Resolution.REJECTED:
        if is_fraud:
            # Correct rejection: fraud caught
            breakdown["decision"] = (
                p.correct_rejection_base
                + p.amount_multiplier_chargeback * case.amount  # saved loss
            )
        else:
            # False positive: rejected legitimate customer
            # Cost scales with customer value and how far from fraud they were
            innocence_factor = 1.0 + 2.0 * (0.5 - true_fraud_risk)
            breakdown["decision"] = p.false_positive_cltv_loss * innocence_factor

    elif case.resolution == Resolution.ESCALATED:
        if case.escalation_justified:
            breakdown["decision"] = 2.0  # small positive for justified escalation
        else:
            breakdown["decision"] = p.unnecessary_escalation

    else:
        breakdown["decision"] = -5.0  # unresolved or deferred without progress

    # --- Process reward ---
    process = 0.0
    if case.policy_checked:
        process += p.policy_check_bonus
    if case.customer_notified:
        process += p.notification_bonus

    evidence_ratio = (
        case.evidence_items_gathered / max(1, case.evidence_items_available)
    )
    process += min(p.evidence_max_bonus, p.evidence_max_bonus * evidence_ratio)
    breakdown["process"] = process

    # --- Total ---
    breakdown["total"] = sum(breakdown.values())
    return breakdown


# ---------------------------------------------------------------------------
# 1B. Invoice Reconciliation Reward
# ---------------------------------------------------------------------------
#
# Business context:
#   - Average invoice value: $5,000-50,000 (B2B)
#   - Duplicate payment rate: ~1-2% of invoices
#   - Average duplicate payment recovery cost: $50-200 per incident
#   - Late payment penalty: 1.5% per month (typical Net-30 terms)
#   - Vendor relationship damage from delayed payment: hard to quantify,
#     use proxy of $500 per unnecessary delay > 7 days
#
# Three-way match logic:
#   Invoice <-> Purchase Order <-> Goods Receipt
#   Mismatch tolerance: typically 1-5% depending on policy
#
# Formula:
#   R_invoice = R_match_quality + R_cash_flow + R_vendor_relationship
#
#   R_match_quality:
#     Correct match (approve valid):         +8.0
#     Correct reject (catch duplicate):      +15.0 + 0.01 * amount
#     False positive (reject valid invoice): -5.0 - vendor_delay_cost
#     False negative (approve duplicate):    -20.0 - 0.02 * amount
#
#   R_cash_flow:
#     On-time processing bonus:     +3.0
#     Payment delay penalty:        -0.5 per day late
#     Cash-flow optimization:       +2.0 if within discount window
#
#   R_vendor_relationship:
#     Unnecessary document request: -1.5
#     Correct document request:     +1.0
#     Delayed response cost:        -0.3 per day vendor waits

@dataclass
class InvoiceRewardParams:
    """Calibrated from typical B2B invoice processing."""
    correct_match_base: float = 8.0
    catch_duplicate_base: float = 15.0
    catch_duplicate_amount_factor: float = 0.01   # 1% of invoice saved
    false_reject_base: float = -5.0
    vendor_delay_cost_per_day: float = -0.5
    false_approve_duplicate_base: float = -20.0
    false_approve_amount_factor: float = 0.02     # 2% of duplicate amount lost
    on_time_bonus: float = 3.0
    early_discount_bonus: float = 2.0             # e.g. 2/10 Net 30 discount
    unnecessary_doc_request: float = -1.5
    correct_doc_request: float = 1.0
    vendor_wait_cost_per_day: float = -0.3
    three_way_match_bonus: float = 2.0            # bonus for completing proper match


def compute_invoice_reward(
    case: CaseProto,
    queue: QueueProto,
    days_to_resolve: float = 0.0,
    discount_window_remaining: float = 0.0,
    unnecessary_doc_requests: int = 0,
    correct_doc_requests: int = 0,
    params: InvoiceRewardParams | None = None,
) -> dict[str, float]:
    p = params or InvoiceRewardParams()
    breakdown: dict[str, float] = {}

    is_duplicate = bool(_hidden(case, "true_is_duplicate", False))

    # --- Match quality ---
    if case.resolution == Resolution.APPROVED:
        if not is_duplicate:
            breakdown["match_quality"] = p.correct_match_base
            # Bonus for completing three-way match check
            if case.checks_completed >= case.checks_required:
                breakdown["match_quality"] += p.three_way_match_bonus
        else:
            # Approved a duplicate -> direct financial loss
            breakdown["match_quality"] = (
                p.false_approve_duplicate_base
                - p.false_approve_amount_factor * case.amount
            )

    elif case.resolution == Resolution.REJECTED:
        if is_duplicate:
            breakdown["match_quality"] = (
                p.catch_duplicate_base
                + p.catch_duplicate_amount_factor * case.amount
            )
        else:
            # Rejected a valid invoice -> vendor relationship damage
            breakdown["match_quality"] = p.false_reject_base

    elif case.resolution == Resolution.ESCALATED:
        if case.escalation_justified:
            breakdown["match_quality"] = 1.0
        else:
            breakdown["match_quality"] = -2.0
    else:
        breakdown["match_quality"] = -3.0

    # --- Cash flow impact ---
    cash_flow = 0.0
    if case.resolution == Resolution.APPROVED and not is_duplicate:
        if days_to_resolve <= 3:
            cash_flow += p.on_time_bonus
        if discount_window_remaining > 0:
            cash_flow += p.early_discount_bonus  # captured early-payment discount
    if days_to_resolve > 5:
        cash_flow += p.vendor_delay_cost_per_day * (days_to_resolve - 5)
    breakdown["cash_flow"] = cash_flow

    # --- Vendor relationship ---
    vendor = 0.0
    vendor += p.unnecessary_doc_request * unnecessary_doc_requests
    vendor += p.correct_doc_request * correct_doc_requests
    breakdown["vendor_relationship"] = vendor

    breakdown["total"] = sum(breakdown.values())
    return breakdown


# ---------------------------------------------------------------------------
# 1C. KYC Compliance Reward
# ---------------------------------------------------------------------------
#
# Business context:
#   - Regulatory fine for non-compliance: $10,000 - $1,000,000+
#   - Payout held incorrectly: customer churn risk ~15% per incident
#   - False positive (blocking compliant user): ~$200 support cost + churn risk
#   - False negative (approving non-compliant): regulatory exposure
#   - Processing time: KYC reviews average 2-8 hours
#   - Re-review rate: ~20% of KYC submissions need follow-up
#
# Key asymmetry:
#   False negative (approve without KYC) is CATASTROPHIC compared to
#   false positive (delay compliant user). Ratio is roughly 50:1 to 100:1
#   depending on jurisdiction.
#
# Formula:
#   R_kyc = R_compliance + R_accuracy + R_speed
#
#   R_compliance:
#     Approved with complete valid KYC:     +10.0
#     Approved with INCOMPLETE KYC:         -50.0 (catastrophic)
#     Approved with INVALID documents:      -30.0
#     Correctly held pending:               +3.0
#     Correctly requested missing docs:     +2.0 per correct request
#     Requested unnecessary docs:           -2.0 per unnecessary request
#
#   R_accuracy:
#     Uses asymmetric cost function (see Section 5)
#
#   R_speed:
#     Payout released within SLA:           +5.0
#     Each hour of unnecessary delay:       -0.5
#     Expedited high-priority correctly:    +3.0

@dataclass
class KYCRewardParams:
    """Calibrated for financial services KYC/AML compliance."""
    # Compliance outcomes - note extreme asymmetry
    approved_with_complete_kyc: float = 10.0
    approved_incomplete_kyc: float = -50.0    # regulatory catastrophe
    approved_invalid_docs: float = -30.0      # serious but less than missing entirely
    correctly_held_pending: float = 3.0
    correct_doc_request: float = 2.0
    unnecessary_doc_request: float = -2.0
    # Speed
    on_time_payout_bonus: float = 5.0
    unnecessary_delay_per_hour: float = -0.5
    expedited_high_priority: float = 3.0
    # Risk tier multipliers
    high_risk_multiplier: float = 2.0         # double penalties for high-risk tier
    low_risk_multiplier: float = 0.5          # halve for low-risk
    # False positive/negative asymmetry ratio
    fn_fp_ratio: float = 50.0                 # false negative is 50x worse


def compute_kyc_reward(
    case: CaseProto,
    queue: QueueProto,
    hours_to_resolve: float = 0.0,
    sla_hours: float = 8.0,
    correct_doc_requests: int = 0,
    unnecessary_doc_requests: int = 0,
    risk_tier: str = "medium",
    params: KYCRewardParams | None = None,
) -> dict[str, float]:
    p = params or KYCRewardParams()
    breakdown: dict[str, float] = {}

    # Risk tier multiplier
    tier_mult = {
        "low": p.low_risk_multiplier,
        "medium": 1.0,
        "high": p.high_risk_multiplier,
    }.get(risk_tier, 1.0)

    # --- Compliance ---
    sanctions_match = bool(_hidden(case, "true_sanctions_match", False))
    sanctions_status = _wf(case, "sanctions_status", "not_started")
    edd_status = _wf(case, "edd_status", "not_started")
    beneficial_owner_status = _wf(case, "beneficial_owner_status", "not_started")
    report_status = _wf(case, "ofac_report_status", "not_required")
    payments_frozen = bool(_wf(case, "payments_frozen", False))
    if case.resolution == Resolution.APPROVED:
        kyc_complete = bool(_wf(case, "kyc_complete", False))
        true_doc_valid = bool(_hidden(case, "true_doc_valid", True))
        if (
            kyc_complete
            and true_doc_valid
            and sanctions_status == "clear"
            and edd_status in {"not_required", "cleared"}
            and beneficial_owner_status in {"not_started", "verified"}
            and report_status in {"not_required", "filed"}
            and not payments_frozen
        ):
            breakdown["compliance"] = p.approved_with_complete_kyc
        elif not kyc_complete:
            # CATASTROPHIC: approved without complete KYC
            breakdown["compliance"] = p.approved_incomplete_kyc * tier_mult
        else:
            # Approved with invalid documents
            breakdown["compliance"] = p.approved_invalid_docs * tier_mult
    elif case.resolution == Resolution.DEFERRED:
        breakdown["compliance"] = p.correctly_held_pending + (
            2.0 if sanctions_status in {"potential_match", "confirmed_match"} or edd_status in {"in_progress", "awaiting_response"} else 0.0
        )
    elif case.resolution == Resolution.ESCALATED:
        breakdown["compliance"] = 1.0 if case.escalation_justified else -2.0
    elif case.resolution == Resolution.REJECTED:
        if sanctions_match:
            breakdown["compliance"] = 9.0 if report_status == "filed" and payments_frozen else 2.0
        elif not bool(_hidden(case, "true_doc_valid", True)):
            breakdown["compliance"] = 5.0  # correct rejection of bad docs
        else:
            breakdown["compliance"] = -8.0  # wrongly rejected valid applicant
    else:
        breakdown["compliance"] = -3.0

    # --- Document request accuracy ---
    doc_reward = (
        p.correct_doc_request * correct_doc_requests
        + p.unnecessary_doc_request * unnecessary_doc_requests
    )
    breakdown["doc_requests"] = doc_reward

    # --- Speed ---
    speed = 0.0
    if case.resolution in (Resolution.APPROVED, Resolution.REJECTED):
        if hours_to_resolve <= sla_hours:
            speed += p.on_time_payout_bonus
        excess_hours = max(0, hours_to_resolve - sla_hours * 0.5)
        speed += p.unnecessary_delay_per_hour * excess_hours
        if case.priority <= 2 and hours_to_resolve < sla_hours * 0.5:
            speed += p.expedited_high_priority
    breakdown["speed"] = speed

    breakdown["total"] = sum(breakdown.values())
    return breakdown


# ---------------------------------------------------------------------------
# 1D. Queue Triage Reward
# ---------------------------------------------------------------------------
#
# Business context:
#   - Typical ops queue: 20-100 cases per agent per shift
#   - SLA breach rate target: < 5%
#   - Average handling time: 5-15 min per case
#   - Priority distribution: ~10% P1, ~25% P2, ~40% P3, ~25% P4/P5
#   - Resource utilization target: 80-90%
#
# This reward is fundamentally different from case-level rewards.
# It evaluates ORDERING and THROUGHPUT, not individual decisions.
#
# Formula:
#   R_triage = R_throughput + R_priority + R_utilization + R_sla_prevention
#
#   R_throughput:
#     Cases resolved per unit time:  +2.0 * throughput_ratio
#     Throughput above baseline:     +5.0 bonus
#
#   R_priority:
#     Priority-weighted completion:  sum(case_value * priority_weight)
#     Priority inversion penalty:    -3.0 per inversion
#       (working on P3 while P1 is breaching)
#
#   R_utilization:
#     Active time / total time:      +4.0 * utilization if > 0.7
#     Idle time penalty:             -1.0 per idle period
#     Context-switch penalty:        -0.5 per unnecessary switch
#
#   R_sla_prevention:
#     Cases saved from breach:       +4.0 per save
#     Cases that breached:           -5.0 per breach
#     Near-miss saves:               +2.0 (resolved < 10% time remaining)

@dataclass
class TriageRewardParams:
    """Calibrated for ops queue management."""
    # Throughput
    throughput_per_case: float = 2.0
    throughput_bonus_threshold: float = 0.8  # ratio of cases resolved
    throughput_bonus: float = 5.0
    # Priority weights (P1 is most important)
    priority_weights: dict[int, float] = field(default_factory=lambda: {
        1: 5.0,   # critical
        2: 3.0,   # high
        3: 1.5,   # medium
        4: 0.8,   # low
        5: 0.3,   # trivial
    })
    priority_inversion_penalty: float = -3.0
    # Utilization
    utilization_bonus_max: float = 4.0
    utilization_threshold: float = 0.7
    idle_penalty: float = -1.0
    context_switch_penalty: float = -0.5
    # SLA
    sla_save_bonus: float = 4.0
    sla_breach_penalty: float = -5.0
    near_miss_save_bonus: float = 2.0
    near_miss_threshold: float = 0.1  # < 10% time remaining


def compute_triage_reward(
    queue: QueueProto,
    cases_resolved_this_step: list[CaseProto],
    priority_inversions: int = 0,
    utilization_ratio: float = 0.0,
    idle_periods: int = 0,
    context_switches: int = 0,
    sla_saves: int = 0,
    sla_breaches: int = 0,
    near_miss_saves: int = 0,
    total_cases: int = 1,
    params: TriageRewardParams | None = None,
) -> dict[str, float]:
    p = params or TriageRewardParams()
    breakdown: dict[str, float] = {}

    # --- Throughput ---
    throughput = p.throughput_per_case * len(cases_resolved_this_step)
    throughput_ratio = len(cases_resolved_this_step) / max(1, total_cases)
    if throughput_ratio >= p.throughput_bonus_threshold:
        throughput += p.throughput_bonus
    breakdown["throughput"] = throughput

    # --- Priority adherence ---
    priority_value = 0.0
    for case in cases_resolved_this_step:
        weight = p.priority_weights.get(case.priority, 1.0)
        priority_value += weight
    priority_value += p.priority_inversion_penalty * priority_inversions
    breakdown["priority_adherence"] = priority_value

    # --- Utilization ---
    utilization = 0.0
    if utilization_ratio >= p.utilization_threshold:
        utilization += p.utilization_bonus_max * utilization_ratio
    utilization += p.idle_penalty * idle_periods
    utilization += p.context_switch_penalty * context_switches
    breakdown["utilization"] = utilization

    # --- SLA prevention ---
    sla_reward = (
        p.sla_save_bonus * sla_saves
        + p.sla_breach_penalty * sla_breaches
        + p.near_miss_save_bonus * near_miss_saves
    )
    breakdown["sla_prevention"] = sla_reward

    breakdown["total"] = sum(breakdown.values())
    return breakdown


# ============================================================================
# SECTION 2: TIME-VARYING REWARD (NON-LINEAR SLA PRESSURE)
# ============================================================================

# In real operations, the urgency of a case is NOT linear with time remaining.
#
# A case with 4 hours remaining feels routine.
# A case with 30 minutes remaining feels urgent.
# A case with 5 minutes remaining is a crisis.
# A case already breached is a different category entirely (remediation).
#
# We model this with a sigmoid-based time pressure function:
#
#   time_pressure(t_remaining, t_total) =
#     1 / (1 + exp(k * (t_remaining/t_total - inflection)))
#
# Where:
#   t_remaining: minutes until SLA deadline
#   t_total: original SLA budget in minutes
#   k: steepness parameter (how sharply pressure rises)
#   inflection: the fraction of time remaining where pressure = 0.5
#
# Calibration:
#   k = 10     -> moderate steepness
#   inflection = 0.2  -> pressure rises sharply when < 20% time remains
#
# Properties:
#   - At 80% time remaining: pressure ~= 0.002 (negligible)
#   - At 50% time remaining: pressure ~= 0.047 (low)
#   - At 20% time remaining: pressure ~= 0.50  (significant)
#   - At 10% time remaining: pressure ~= 0.73  (high)
#   - At 5% time remaining:  pressure ~= 0.82  (very high)
#   - At 0% (deadline):      pressure ~= 0.88  (max before breach)
#   - After breach:          jumps to 1.0 + overshoot penalty
#
# The reward impact is:
#   R_time = -max_time_penalty * time_pressure(t_rem, t_total)
#            - breach_penalty * max(0, -t_remaining)
#
# This means:
#   - Resolving early: almost no time penalty
#   - Resolving just before deadline: moderate penalty (for the risk taken)
#   - Breaching: large fixed penalty + per-minute overshoot

@dataclass
class TimePressureParams:
    steepness: float = 10.0
    inflection_point: float = 0.2        # pressure=0.5 at 20% time remaining
    max_pre_breach_penalty: float = -5.0  # max penalty before actual breach
    breach_fixed_penalty: float = -8.0    # one-time penalty for breaching
    breach_per_minute: float = -0.1       # ongoing cost per minute after breach
    # Priority-based SLA multipliers
    priority_sla_multipliers: dict[int, float] = field(default_factory=lambda: {
        1: 2.0,    # P1 breach is 2x as costly
        2: 1.5,
        3: 1.0,
        4: 0.7,
        5: 0.5,
    })


def time_pressure_signal(
    t_remaining: float,
    t_total: float,
    params: TimePressureParams | None = None,
) -> float:
    """
    Compute the non-linear time pressure signal.

    Returns a value in [0, 1] representing urgency.
    0 = no pressure, 1 = at/past deadline.

    The sigmoid shape ensures:
    - Gradual onset: plenty of time feels safe
    - Sharp rise: pressure accelerates as deadline approaches
    - Saturation: prevents infinite penalty at the deadline itself
    """
    p = params or TimePressureParams()

    if t_total <= 0:
        return 1.0

    fraction_remaining = max(0.0, t_remaining / t_total)

    # Sigmoid: 1 / (1 + exp(k * (f - inflection)))
    exponent = p.steepness * (fraction_remaining - p.inflection_point)
    pressure = 1.0 / (1.0 + math.exp(exponent))

    return pressure


def compute_time_reward(
    t_remaining: float,
    t_total: float,
    priority: int = 3,
    already_breached: bool = False,
    params: TimePressureParams | None = None,
) -> dict[str, float]:
    """
    Compute time-varying reward component for a case.

    Args:
        t_remaining: minutes until SLA deadline (negative = breached)
        t_total: original SLA window in minutes
        priority: case priority (1-5)
        already_breached: if True, the breach_fixed_penalty was already applied
            on a prior step, so only apply the per-minute overshoot.
    """
    p = params or TimePressureParams()
    breakdown: dict[str, float] = {}

    priority_mult = p.priority_sla_multipliers.get(priority, 1.0)

    if t_remaining >= 0:
        # Not yet breached: apply sigmoid pressure
        pressure = time_pressure_signal(t_remaining, t_total, p)
        breakdown["time_pressure"] = p.max_pre_breach_penalty * pressure * priority_mult
        breakdown["breach_penalty"] = 0.0
    else:
        # Breached: fixed penalty once + small per-minute overshoot
        breakdown["time_pressure"] = p.max_pre_breach_penalty * priority_mult
        minutes_over = abs(t_remaining)
        if already_breached:
            # Only the marginal per-minute cost, not the fixed penalty again
            breakdown["breach_penalty"] = p.breach_per_minute * min(minutes_over, t_total) * priority_mult
        else:
            breakdown["breach_penalty"] = (
                p.breach_fixed_penalty + p.breach_per_minute * min(minutes_over, t_total)
            ) * priority_mult

    breakdown["total"] = sum(breakdown.values())
    return breakdown


# ============================================================================
# SECTION 3: INFORMATION VALUE REWARDS
# ============================================================================

# Problem:
#   Gathering evidence (opening a case, viewing records, checking policy)
#   has no direct business value. But it enables BETTER decisions.
#   Naive approaches:
#     (a) Reward each info-gather action -> agent spams evidence gathering
#     (b) Don't reward at all -> agent skips evidence and guesses
#
# Solution: Value of Information (VOI) from decision theory.
#
# The Value of Perfect Information (VPI) for a variable X is:
#   VPI(X) = E[max_a U(a | X)] - max_a E[U(a | current_info)]
#
# In English: the expected improvement in decision quality from learning X.
#
# For our RL environment, we approximate this as:
#
#   VOI_approx(evidence_action) =
#     decision_entropy_before - decision_entropy_after
#     (estimated from the current evidence state)
#
# Practical implementation:
#   We track which evidence items have been gathered and compute a
#   "decision confidence" score. The reward for gathering evidence is
#   the MARGINAL improvement in confidence, capped and discounted.
#
# Anti-gaming properties:
#   1. Diminishing returns: each additional evidence item provides
#      less marginal value (logarithmic scaling)
#   2. Relevance gating: only evidence items that are RELEVANT to the
#      case type contribute (irrelevant queries get zero or negative reward)
#   3. Time cost: evidence gathering costs time, which interacts with
#      the time pressure function
#   4. Decision-conditional: the VOI reward is ONLY paid if the final
#      decision actually improves (retroactive adjustment)
#
# Formula:
#   R_info(action) = alpha * max(0, delta_confidence) * relevance_weight
#                    - beta * time_cost
#                    - gamma * (1 if redundant_query else 0)
#
# Where:
#   delta_confidence = confidence_after - confidence_before
#   confidence(k items of n) = 1 - (1 - base_conf)^k  (geometric model)
#   relevance_weight in {0.0, 0.5, 1.0} depending on case type
#   time_cost = minutes consumed by the action

@dataclass
class VOIParams:
    """Parameters for Value of Information reward."""
    alpha: float = 2.0               # max marginal VOI reward
    beta: float = 0.05               # time cost per minute
    gamma: float = -1.0              # redundant query penalty
    base_confidence: float = 0.3     # confidence with zero evidence
    max_evidence_reward: float = 5.0  # cap total evidence reward per case
    redundancy_penalty: float = -1.0  # penalty for re-gathering same evidence
    irrelevant_query_penalty: float = -0.5  # penalty for irrelevant evidence


# Relevance matrices: which evidence items matter for which case types
# 1.0 = essential, 0.5 = useful, 0.0 = irrelevant
EVIDENCE_RELEVANCE: dict[CaseType, dict[str, float]] = {
    CaseType.REFUND: {
        "order_history": 1.0,
        "customer_profile": 1.0,
        "refund_policy": 1.0,
        "fraud_signals": 1.0,
        "payment_method": 0.5,
        "shipping_status": 0.5,
        "invoice_data": 0.0,
        "kyc_documents": 0.0,
        "vendor_profile": 0.0,
    },
    CaseType.INVOICE: {
        "invoice_data": 1.0,
        "purchase_order": 1.0,
        "goods_receipt": 1.0,
        "vendor_profile": 1.0,
        "mismatch_policy": 1.0,
        "payment_history": 0.5,
        "order_history": 0.0,
        "customer_profile": 0.0,
        "kyc_documents": 0.0,
    },
    CaseType.KYC: {
        "kyc_documents": 1.0,
        "identity_verification": 1.0,
        "risk_assessment": 1.0,
        "compliance_policy": 1.0,
        "customer_profile": 0.5,
        "payout_history": 0.5,
        "invoice_data": 0.0,
        "order_history": 0.0,
        "vendor_profile": 0.0,
    },
    CaseType.TRIAGE: {
        "queue_overview": 1.0,
        "sla_status": 1.0,
        "case_summaries": 1.0,
        "priority_matrix": 1.0,
        "resource_status": 0.5,
        "historical_patterns": 0.5,
    },
}


def decision_confidence(
    items_gathered: int,
    items_available: int,
    base_confidence: float = 0.3,
) -> float:
    """
    Compute decision confidence as a function of evidence gathered.

    Uses a geometric model: each evidence item reduces remaining
    uncertainty by a fixed fraction.

    confidence(k) = 1 - (1 - base)^(1 + k * available_ratio)

    This gives:
    - k=0: confidence = base (e.g., 0.3)
    - k=available: confidence approaches 1.0
    - Diminishing returns: each additional item helps less
    """
    if items_available <= 0:
        return base_confidence

    ratio = items_gathered / items_available
    # Exponent grows with evidence, shrinking uncertainty
    confidence = 1.0 - (1.0 - base_confidence) ** (1.0 + 3.0 * ratio)
    return min(1.0, confidence)


def compute_voi_reward(
    case: CaseProto,
    evidence_type: str,
    items_before: int,
    items_after: int,
    time_cost_minutes: float,
    already_gathered: set[str] | None = None,
    params: VOIParams | None = None,
) -> dict[str, float]:
    """
    Compute Value of Information reward for an evidence-gathering action.

    Args:
        case: current case state
        evidence_type: what kind of evidence was gathered
        items_before: evidence count before this action
        items_after: evidence count after this action
        time_cost_minutes: time consumed by this action
        already_gathered: set of evidence types already gathered (for redundancy check)
    """
    p = params or VOIParams()
    breakdown: dict[str, float] = {}

    # Check relevance
    relevance_map = EVIDENCE_RELEVANCE.get(case.case_type, {})
    relevance = relevance_map.get(evidence_type, 0.0)

    if relevance == 0.0:
        breakdown["relevance_penalty"] = p.irrelevant_query_penalty
        breakdown["marginal_value"] = 0.0
        breakdown["time_cost"] = -p.beta * time_cost_minutes
        breakdown["total"] = sum(breakdown.values())
        return breakdown

    # Check redundancy
    if already_gathered and evidence_type in already_gathered:
        breakdown["redundancy_penalty"] = p.redundancy_penalty
        breakdown["marginal_value"] = 0.0
        breakdown["time_cost"] = -p.beta * time_cost_minutes
        breakdown["total"] = sum(breakdown.values())
        return breakdown

    # Compute marginal confidence improvement
    conf_before = decision_confidence(
        items_before, case.evidence_items_available, p.base_confidence
    )
    conf_after = decision_confidence(
        items_after, case.evidence_items_available, p.base_confidence
    )
    delta_conf = max(0.0, conf_after - conf_before)

    # VOI = alpha * delta_confidence * relevance
    marginal_value = p.alpha * delta_conf * relevance
    breakdown["marginal_value"] = min(marginal_value, p.max_evidence_reward)

    # Time cost
    breakdown["time_cost"] = -p.beta * time_cost_minutes

    breakdown["total"] = sum(breakdown.values())
    return breakdown


# ============================================================================
# SECTION 4: QUEUE-LEVEL vs CASE-LEVEL REWARDS
# ============================================================================

# Problem:
#   Case-level rewards can't capture the PORTFOLIO EFFECT.
#   Solving Case A perfectly but letting Case B breach is globally bad,
#   even if the agent's per-case score on A is excellent.
#
# This is analogous to portfolio optimization in finance:
#   - Each case is an "asset" with expected return and risk
#   - The agent must allocate attention (a scarce resource) across cases
#   - Diversification matters: neglecting any case cluster is punished
#
# Queue-level reward formula:
#
#   R_queue = R_throughput_efficiency
#           + R_priority_weighted_completion
#           + R_sla_portfolio
#           + R_risk_distribution
#           + R_backlog_health
#
# Components:
#
#   R_throughput_efficiency:
#     Measures overall resolution rate normalized by available time.
#     = (cases_resolved / cases_total) * throughput_scale
#     With diminishing bonus for exceeding baseline throughput.
#
#   R_priority_weighted_completion:
#     Not all cases are equal. Completing a P1 case should count more.
#     = sum(priority_weight[c.priority] * resolved(c)) - priority_inversion_count * penalty
#
#   R_sla_portfolio:
#     Portfolio-level SLA health, not just individual breaches.
#     = sla_health_bonus * (1 - breach_rate)^2
#     Quadratic: going from 5% breach to 0% is much more valuable
#     than going from 50% to 45%.
#
#   R_risk_distribution:
#     Penalizes concentration of unresolved high-risk cases.
#     If the remaining queue is dominated by high-risk items,
#     the agent failed to address them early.
#     = -risk_concentration * herfindahl_index(remaining_risk)
#
#   R_backlog_health:
#     Penalizes growing backlog and rewards stable/shrinking backlog.
#     = backlog_coefficient * (backlog_start - backlog_end) / backlog_start

@dataclass
class QueueRewardParams:
    """Parameters for queue-level portfolio reward."""
    throughput_scale: float = 10.0
    throughput_baseline: float = 0.6    # expected resolution ratio
    throughput_bonus_per_pct: float = 0.5  # bonus per % above baseline
    priority_inversion_penalty: float = -3.0
    sla_health_bonus: float = 15.0
    risk_concentration_penalty: float = -8.0
    backlog_coefficient: float = 5.0
    assignment_health_bonus: float = 4.0
    escalation_overload_penalty: float = -10.0
    escalation_overload_threshold: float = 0.8  # fraction of capacity


def compute_queue_reward(
    queue: QueueProto,
    initial_case_count: int,
    initial_backlog: int,
    resolved_cases: list[CaseProto],
    remaining_cases: list[CaseProto],
    priority_inversions: int = 0,
    params: QueueRewardParams | None = None,
) -> dict[str, float]:
    """
    Compute queue-level portfolio reward at episode end or checkpoint.

    This reward is computed ONCE per episode (or per evaluation window),
    not per step. It captures global queue health.
    """
    p = params or QueueRewardParams()
    breakdown: dict[str, float] = {}

    total = max(1, initial_case_count)

    # --- Throughput efficiency ---
    resolution_ratio = len(resolved_cases) / total
    throughput = p.throughput_scale * resolution_ratio
    if resolution_ratio > p.throughput_baseline:
        excess = resolution_ratio - p.throughput_baseline
        throughput += p.throughput_bonus_per_pct * excess * 100
    breakdown["throughput"] = throughput

    # --- Priority-weighted completion ---
    priority_weights = {1: 5.0, 2: 3.0, 3: 1.5, 4: 0.8, 5: 0.3}
    pwc = sum(priority_weights.get(c.priority, 1.0) for c in resolved_cases)
    pwc += p.priority_inversion_penalty * priority_inversions
    breakdown["priority_weighted_completion"] = pwc

    # --- SLA portfolio health ---
    breached = sum(1 for c in resolved_cases if c.status == "breached")
    breached += sum(1 for c in remaining_cases if c.status == "breached")
    total_cases = len(resolved_cases) + len(remaining_cases)
    breach_rate = breached / max(1, total_cases)
    # Quadratic: (1 - breach_rate)^2
    sla_health = p.sla_health_bonus * (1.0 - breach_rate) ** 2
    breakdown["sla_portfolio_health"] = sla_health

    # --- Risk concentration (Herfindahl-like index) ---
    if remaining_cases:
        risk_scores = [float(_hidden(c, "true_fraud_risk", 0.0)) + (0.3 if c.priority <= 2 else 0.0)
                       for c in remaining_cases]
        total_risk = sum(risk_scores) or 1.0
        shares = [r / total_risk for r in risk_scores]
        hhi = sum(s ** 2 for s in shares)  # 1/N = perfectly spread, 1.0 = all in one
        # Normalize: HHI ranges from 1/N to 1.0
        n = len(remaining_cases)
        normalized_hhi = (hhi - 1.0 / n) / (1.0 - 1.0 / n) if n > 1 else 0.0
        breakdown["risk_concentration"] = p.risk_concentration_penalty * normalized_hhi
    else:
        breakdown["risk_concentration"] = 0.0  # all resolved

    # --- Backlog health ---
    current_backlog = len(remaining_cases)
    if initial_backlog > 0:
        backlog_change = (initial_backlog - current_backlog) / initial_backlog
        breakdown["backlog_health"] = p.backlog_coefficient * backlog_change
    else:
        breakdown["backlog_health"] = p.backlog_coefficient  # started empty, stayed empty

    # --- Assignment health ---
    if queue.exception_queue_size > 0:
        assigned_ratio = 1.0 - (queue.unassigned_count / max(1, queue.exception_queue_size))
        breakdown["assignment_health"] = p.assignment_health_bonus * max(0.0, assigned_ratio)
    else:
        breakdown["assignment_health"] = p.assignment_health_bonus

    # --- Escalation overload ---
    esc_ratio = queue.escalation_queue_load / max(1, queue.escalation_queue_capacity)
    if esc_ratio > p.escalation_overload_threshold:
        breakdown["escalation_overload"] = (
            p.escalation_overload_penalty
            * (esc_ratio - p.escalation_overload_threshold)
            / (1.0 - p.escalation_overload_threshold)
        )
    else:
        breakdown["escalation_overload"] = 0.0

    breakdown["total"] = sum(breakdown.values())
    return breakdown


# ============================================================================
# SECTION 5: ASYMMETRIC ERROR COSTS
# ============================================================================

# In every workflow, false positives and false negatives have VERY different
# costs. This is one of the most important properties of real operations.
#
# General asymmetric loss function:
#
#   L(prediction, truth) =
#     c_FP * I(predict_positive, truth_negative)     # false positive cost
#   + c_FN * I(predict_negative, truth_positive)     # false negative cost
#   + c_TP * I(predict_positive, truth_positive)     # true positive "cost" (usually reward)
#   + c_TN * I(predict_negative, truth_negative)     # true negative "cost" (usually small reward)
#
# Calibrated costs per workflow type:
#
# -----------------------------------------------------------------------
# REFUND FRAUD DETECTION
# -----------------------------------------------------------------------
# False Positive (block legitimate customer):
#   - Customer support cost:                    $25
#   - CLTV erosion (15% churn probability):     $800 * 0.15 = $120
#   - Brand damage (unmeasured but real):       $0
#   Total FP cost:                              ~$145
#
# False Negative (approve fraudulent refund):
#   - Refund payout (average):                  $85
#   - Chargeback fee:                           $25
#   - Operational cost of chargeback handling:  $40
#   - Potential scheme continuation:            $200 (serial fraud)
#   - Merchant processor risk score impact:     $150 (amortized)
#   Total FN cost:                              ~$500
#
# Asymmetry ratio: FN/FP = 500/145 ~= 3.4
#
# -----------------------------------------------------------------------
# INVOICE DUPLICATE DETECTION
# -----------------------------------------------------------------------
# False Positive (reject valid invoice):
#   - Vendor relationship damage:               $200
#   - Late payment penalty (1.5%/month):        $75 (on $5000)
#   - Operational rework:                       $30
#   Total FP cost:                              ~$305
#
# False Negative (approve duplicate invoice):
#   - Duplicate payment:                        $5,000 (median invoice)
#   - Recovery cost:                            $150
#   - Audit finding penalty:                    $500
#   Total FN cost:                              ~$5,650
#
# Asymmetry ratio: FN/FP = 5650/305 ~= 18.5
#
# -----------------------------------------------------------------------
# KYC COMPLIANCE
# -----------------------------------------------------------------------
# False Positive (block compliant user):
#   - Customer support cost:                    $50
#   - Delayed payout CLTV impact:              $100
#   - User churn (5% probability):             $500 * 0.05 = $25
#   Total FP cost:                              ~$175
#
# False Negative (approve non-compliant):
#   - Regulatory fine (amortized):             $10,000
#   - Remediation cost:                        $2,000
#   - Reputational damage:                     $5,000
#   Total FN cost:                              ~$17,000
#
# Asymmetry ratio: FN/FP = 17000/175 ~= 97
#
# -----------------------------------------------------------------------
# QUEUE TRIAGE (priority misclassification)
# -----------------------------------------------------------------------
# False High Priority (treat low case as urgent):
#   - Wasted premium processing time:          $20
#   - Opportunity cost (delayed other cases):  $30
#   Total FP cost:                              ~$50
#
# False Low Priority (treat urgent case as low):
#   - SLA breach cost:                         $200
#   - Customer escalation cost:                $150
#   - Potential chargeback/regulatory trigger:  $300
#   Total FN cost:                              ~$650
#
# Asymmetry ratio: FN/FP = 650/50 = 13
# -----------------------------------------------------------------------

@dataclass
class AsymmetricCosts:
    """Error costs calibrated from real ecommerce operations."""
    c_tp: float   # reward for correct positive
    c_tn: float   # reward for correct negative
    c_fp: float   # cost of false positive
    c_fn: float   # cost of false negative


# Pre-calibrated cost structures per workflow
ASYMMETRIC_COSTS: dict[CaseType, AsymmetricCosts] = {
    CaseType.REFUND: AsymmetricCosts(
        c_tp=8.0,      # correctly caught fraud: reward
        c_tn=5.0,      # correctly approved legit: reward
        c_fp=-12.0,    # blocked legitimate customer: normalized penalty
        c_fn=-40.0,    # approved fraud: normalized penalty (ratio ~3.4:1)
    ),
    CaseType.INVOICE: AsymmetricCosts(
        c_tp=15.0,     # correctly caught duplicate
        c_tn=5.0,      # correctly approved valid invoice
        c_fp=-5.0,     # rejected valid invoice
        c_fn=-90.0,    # approved duplicate (ratio ~18:1)
    ),
    CaseType.KYC: AsymmetricCosts(
        c_tp=10.0,     # correctly blocked non-compliant
        c_tn=8.0,      # correctly approved compliant
        c_fp=-3.0,     # blocked compliant user
        c_fn=-300.0,   # approved non-compliant (ratio ~100:1)
    ),
    CaseType.TRIAGE: AsymmetricCosts(
        c_tp=4.0,      # correctly prioritized urgent
        c_tn=2.0,      # correctly deprioritized low
        c_fp=-3.0,     # over-prioritized low case
        c_fn=-40.0,    # under-prioritized urgent case (ratio ~13:1)
    ),
}


def compute_asymmetric_reward(
    case_type: CaseType,
    predicted_positive: bool,
    actual_positive: bool,
    amount: float = 0.0,
    amount_scale_factor: float = 0.001,
    costs: AsymmetricCosts | None = None,
) -> dict[str, float | str]:
    """
    Compute asymmetric error cost for a binary decision.

    For refund: positive = fraud, negative = legitimate
    For invoice: positive = duplicate, negative = valid
    For KYC: positive = non-compliant, negative = compliant
    For triage: positive = urgent, negative = non-urgent

    The amount_scale_factor allows the penalty to scale with case value:
    total_cost = base_cost + amount_scale_factor * amount * base_cost
    """
    c = costs or ASYMMETRIC_COSTS[case_type]
    breakdown: dict[str, float | str] = {}

    amount_mult = 1.0 + amount_scale_factor * amount

    if predicted_positive and actual_positive:
        breakdown["classification"] = c.c_tp * amount_mult
        breakdown["label"] = "true_positive"
    elif not predicted_positive and not actual_positive:
        breakdown["classification"] = c.c_tn
        breakdown["label"] = "true_negative"
    elif predicted_positive and not actual_positive:
        breakdown["classification"] = c.c_fp * amount_mult
        breakdown["label"] = "false_positive"
    else:  # not predicted_positive and actual_positive
        breakdown["classification"] = c.c_fn * amount_mult
        breakdown["label"] = "false_negative"

    breakdown["total"] = breakdown["classification"]
    return breakdown


# ============================================================================
# SECTION 6: COMPOUND / CASCADING REWARDS
# ============================================================================

# Problem:
#   Bad decisions create DOWNSTREAM events that affect OTHER cases.
#   Examples:
#   - Approving a fraudulent refund -> chargeback -> chargeback rate increases
#     -> processor raises fees on ALL future transactions
#   - Over-escalating -> human queue overloads -> response times increase
#     for ALL escalated cases -> cascade of SLA breaches
#   - Approving a duplicate invoice -> cash flow drops -> vendor payments
#     delayed -> vendor relationship deteriorates for OTHER invoices
#   - Failing KYC -> regulatory scrutiny increases -> processing time
#     increases for ALL future KYC cases
#
# Design approach:
#   We model cascading effects as a DECAY NETWORK where each case
#   decision can propagate penalties to the global state, which then
#   affects the reward calculation for subsequent cases.
#
# Cascade reward formula:
#
#   R_cascade(action, case, global_state) =
#     R_direct(action, case)                       # immediate case reward
#   + sum_over_affected_cases(
#       decay_factor^distance * impact_magnitude
#     )
#
# Where:
#   - distance: temporal or queue-positional distance between cases
#   - decay_factor: how quickly effects diminish (0.7 - 0.9)
#   - impact_magnitude: severity of the downstream effect
#
# We model four cascade channels:
#
# 1. ESCALATION CASCADE
#    Each escalation adds load to the human queue.
#    When load > threshold, ALL escalated cases get delayed.
#    Delay propagates SLA pressure to those cases.
#
#    escalation_cascade_penalty =
#      -k_esc * max(0, esc_load - esc_threshold)^2 / esc_capacity^2
#
# 2. FRAUD RATE CASCADE
#    Each approved fraud increases the overall fraud rate metric.
#    Higher fraud rate -> higher processor fees -> cost increase on all txns.
#    Modeled as a shared penalty that accumulates.
#
#    fraud_cascade_penalty =
#      -k_fraud * cumulative_fraud_approvals * avg_transaction_volume
#
# 3. COMPLIANCE CONTAGION
#    Each compliance failure increases regulatory scrutiny.
#    Higher scrutiny -> longer processing times for ALL KYC cases.
#    Modeled as a multiplier on KYC processing time.
#
#    compliance_cascade =
#      -k_compliance * compliance_failures * remaining_kyc_cases
#
# 4. CASH FLOW PROPAGATION
#    Duplicate payment approvals reduce available cash.
#    Low cash -> delayed vendor payments -> vendor relationship damage
#    across multiple invoices.
#
#    cashflow_cascade =
#      -k_cashflow * duplicate_amount_approved / total_payables

@dataclass
class CascadeParams:
    """Parameters for cross-case cascading effects."""
    # Escalation cascade
    escalation_load_threshold: float = 0.6   # fraction of capacity
    escalation_cascade_coefficient: float = 8.0
    escalation_delay_per_overload_case: float = 15.0  # minutes delay per excess case
    # Fraud rate cascade
    fraud_rate_baseline: float = 0.015       # 1.5% normal fraud rate
    fraud_cascade_coefficient: float = 5.0
    processor_fee_increase_per_pct: float = 0.002  # 0.2% fee increase per 1% fraud rate increase
    avg_daily_transaction_volume: float = 50000.0
    # Compliance contagion
    compliance_cascade_coefficient: float = 3.0
    scrutiny_time_multiplier_per_failure: float = 0.2  # 20% longer per failure
    # Cash flow propagation
    cashflow_cascade_coefficient: float = 4.0
    total_payables_baseline: float = 500000.0
    vendor_delay_threshold: float = 0.1  # cascade triggers when >10% of payables are duplicates


def compute_escalation_cascade(
    queue: QueueProto,
    new_escalation: bool = False,
    params: CascadeParams | None = None,
) -> dict[str, float]:
    """
    Compute cascading penalty from escalation queue overload.

    When the human escalation queue is overloaded, EVERY case in that
    queue experiences delayed processing, creating a cascade of SLA
    breaches across unrelated cases.
    """
    p = params or CascadeParams()
    breakdown: dict[str, float] = {}

    current_load = queue.escalation_queue_load
    if new_escalation:
        current_load += 1

    capacity = max(1, queue.escalation_queue_capacity)
    load_ratio = current_load / capacity

    if load_ratio > p.escalation_load_threshold:
        excess = load_ratio - p.escalation_load_threshold
        # Quadratic penalty: small overload is manageable, large overload is catastrophic
        cascade_penalty = -p.escalation_cascade_coefficient * excess ** 2
        breakdown["escalation_cascade"] = cascade_penalty

        # Compute estimated delay imposed on ALL cases in escalation queue
        delay_minutes = p.escalation_delay_per_overload_case * (
            current_load - int(capacity * p.escalation_load_threshold)
        )
        breakdown["estimated_delay_minutes"] = delay_minutes
    else:
        breakdown["escalation_cascade"] = 0.0
        breakdown["estimated_delay_minutes"] = 0.0

    breakdown["load_ratio"] = load_ratio
    breakdown["total"] = breakdown["escalation_cascade"]
    return breakdown


def compute_fraud_cascade(
    cumulative_fraud_approvals: int,
    total_decisions: int,
    params: CascadeParams | None = None,
) -> dict[str, float]:
    """
    Compute cascading penalty from elevated fraud approval rate.

    Each fraudulent refund approval increases the merchant's chargeback
    ratio. When the ratio exceeds processor thresholds, fees increase
    on ALL transactions, not just the fraudulent ones.

    Processor fee tiers (typical):
    - < 1% chargeback rate: standard fees
    - 1-1.5%: monitoring program ($10k-$25k/month)
    - 1.5-2%: excessive program ($25k-$50k/month)
    - > 2%: account termination risk
    """
    p = params or CascadeParams()
    breakdown: dict[str, float] = {}

    if total_decisions <= 0:
        breakdown["fraud_cascade"] = 0.0
        breakdown["total"] = 0.0
        return breakdown

    # Current fraud rate (from this agent's decisions)
    agent_fraud_rate = cumulative_fraud_approvals / max(1, total_decisions)

    # Excess over baseline
    excess_rate = max(0.0, agent_fraud_rate - p.fraud_rate_baseline)

    # Cost: increased processor fees across daily transaction volume
    fee_increase = p.processor_fee_increase_per_pct * excess_rate * 100
    daily_cost = fee_increase * p.avg_daily_transaction_volume

    # Normalize to reward scale (assuming ~30-day impact window)
    normalized_penalty = -p.fraud_cascade_coefficient * daily_cost / 1000.0

    breakdown["fraud_cascade"] = normalized_penalty
    breakdown["current_fraud_rate"] = agent_fraud_rate
    breakdown["excess_rate"] = excess_rate
    breakdown["total"] = normalized_penalty
    return breakdown


def compute_compliance_cascade(
    compliance_failures: int,
    remaining_kyc_cases: int,
    params: CascadeParams | None = None,
) -> dict[str, float]:
    """
    Compute cascading penalty from compliance failures.

    Each compliance failure increases regulatory scrutiny, which
    increases processing time and documentation requirements for
    ALL subsequent KYC cases. This models the real-world effect
    where a compliance flag triggers enhanced due diligence
    across the entire customer portfolio.
    """
    p = params or CascadeParams()
    breakdown: dict[str, float] = {}

    if compliance_failures == 0:
        breakdown["compliance_cascade"] = 0.0
        breakdown["scrutiny_multiplier"] = 1.0
        breakdown["total"] = 0.0
        return breakdown

    # Each failure multiplies processing effort for remaining cases
    scrutiny_mult = 1.0 + p.scrutiny_time_multiplier_per_failure * compliance_failures
    # Penalty scales with remaining KYC workload
    cascade_penalty = (
        -p.compliance_cascade_coefficient
        * (scrutiny_mult - 1.0)
        * remaining_kyc_cases
    )

    breakdown["compliance_cascade"] = cascade_penalty
    breakdown["scrutiny_multiplier"] = scrutiny_mult
    breakdown["total"] = cascade_penalty
    return breakdown


def compute_cashflow_cascade(
    duplicate_amount_approved: float,
    params: CascadeParams | None = None,
) -> dict[str, float]:
    """
    Compute cascading penalty from cash flow impact of duplicate payments.

    When duplicate payments drain cash, the company may be unable to
    pay vendors on time, creating late-payment penalties and relationship
    damage across MULTIPLE vendor relationships.
    """
    p = params or CascadeParams()
    breakdown: dict[str, float] = {}

    if p.total_payables_baseline <= 0:
        breakdown["cashflow_cascade"] = 0.0
        breakdown["total"] = 0.0
        return breakdown

    impact_ratio = duplicate_amount_approved / p.total_payables_baseline

    if impact_ratio > p.vendor_delay_threshold:
        excess = impact_ratio - p.vendor_delay_threshold
        cascade_penalty = -p.cashflow_cascade_coefficient * excess * 100
        breakdown["cashflow_cascade"] = cascade_penalty
    else:
        breakdown["cashflow_cascade"] = 0.0

    breakdown["impact_ratio"] = impact_ratio
    breakdown["total"] = breakdown["cashflow_cascade"]
    return breakdown


def compute_qa_cascade(
    qa_passed: bool,
    qa_required: bool,
    rework_count: int,
) -> dict[str, float]:
    breakdown: dict[str, float] = {}
    if qa_passed:
        score = 2.0 + (1.0 if qa_required else 0.0) - min(2.0, 0.5 * max(0, rework_count - 1))
        breakdown["qa_cascade"] = score
    else:
        breakdown["qa_cascade"] = -4.0 - min(3.0, 1.5 * rework_count)
    breakdown["total"] = breakdown["qa_cascade"]
    return breakdown


def compute_kyc_compliance_cascade(case: CaseProto, action_type: str) -> dict[str, float]:
    breakdown: dict[str, float] = {}
    sanctions_match = bool(_hidden(case, "true_sanctions_match", False))
    false_positive = bool(_hidden(case, "true_sanctions_false_positive", False))
    edd_required = bool(_hidden(case, "true_edd_required", False))
    report_required = bool(_hidden(case, "true_ofac_report_required", False))
    sanctions_status = _wf(case, "sanctions_status", "not_started")
    edd_status = _wf(case, "edd_status", "not_started")
    beneficial_owner_status = _wf(case, "beneficial_owner_status", "not_started")
    report_status = _wf(case, "ofac_report_status", "not_required")
    payments_frozen = bool(_wf(case, "payments_frozen", False))

    score = 0.0
    if action_type == "run_sanctions_screen":
        score = 3.0 if sanctions_status in {"clear", "potential_match", "confirmed_match"} else -1.0
    elif action_type == "start_edd_review":
        score = 3.0 if edd_required and edd_status in {"in_progress", "awaiting_response", "cleared"} else -1.5
    elif action_type == "review_beneficial_owner":
        score = 3.0 if beneficial_owner_status in {"verified", "needs_correction", "rejected"} else -1.0
    elif action_type == "request_field_correction":
        score = 2.5 if _wf(case, "correction_fields", []) else -1.0
    elif action_type == "freeze_payments":
        score = 4.0 if sanctions_match and payments_frozen else (-0.5 if false_positive else -2.0)
    elif action_type == "file_ofac_report":
        score = 5.0 if report_required and report_status == "filed" else -2.0

    breakdown["kyc_compliance_cascade"] = score
    breakdown["total"] = score
    return breakdown


def compute_refund_risk_cascade(case: CaseProto, action_type: str) -> dict[str, float]:
    breakdown: dict[str, float] = {}
    dispute_should_accept = bool(_hidden(case, "true_dispute_should_accept", False))
    monitoring_status = _wf(case, "monitoring_program_status", "normal")
    merchant_risk_level = _wf(case, "merchant_risk_level", "normal")
    reserve_percent = float(_wf(case, "reserve_percent", 0.0) or 0.0)
    payout_delay_days = int(_wf(case, "payout_delay_days", 0) or 0)
    payout_frozen = bool(_wf(case, "payout_frozen", False))
    reserve_due = _wf(case, "reserve_release_due_at", None)
    strong_packet = len(_wf(case, "dispute_evidence_fields", []) or []) >= 2 or case.evidence_items_gathered >= 3

    score = 0.0
    if action_type == "refund_pre_dispute_alert":
        score = 6.0 if dispute_should_accept else -4.0
    elif action_type == "challenge_dispute":
        score = 5.0 if (not dispute_should_accept and strong_packet) else -3.0
    elif action_type == "resolve_prearbitration":
        score = 5.0 if (not dispute_should_accept and strong_packet) or dispute_should_accept else -3.0
    elif action_type == "freeze_payouts":
        score = 4.0 if monitoring_status == "breached" or merchant_risk_level == "critical" else -2.0
    elif action_type == "unfreeze_payouts":
        score = 3.0 if monitoring_status != "breached" and reserve_percent == 0.0 else -3.0
    elif action_type == "set_reserve_percent":
        score = 3.5 if merchant_risk_level in {"high", "critical"} and reserve_percent >= 10.0 else -2.0
    elif action_type == "clear_reserve":
        score = 2.0 if reserve_due is not None and monitoring_status != "breached" else -2.5
    elif action_type == "set_payout_delay_days":
        score = 3.0 if merchant_risk_level in {"elevated", "high", "critical"} and payout_delay_days > 0 else -1.5

    if payout_frozen and monitoring_status == "breached":
        score += 0.5

    breakdown["refund_risk_cascade"] = score
    breakdown["total"] = score
    return breakdown


# ============================================================================
# SECTION 7: COMBINED REWARD ORCHESTRATOR
# ============================================================================

@dataclass
class RewardBreakdown:
    """Full reward decomposition for logging and analysis."""
    case_id: str
    case_type: CaseType
    workflow_reward: dict[str, float] = field(default_factory=dict)
    time_reward: dict[str, float] = field(default_factory=dict)
    voi_reward: dict[str, float] = field(default_factory=dict)
    queue_reward: dict[str, float] = field(default_factory=dict)
    asymmetric_reward: dict[str, float] = field(default_factory=dict)
    cascade_rewards: dict[str, dict[str, float]] = field(default_factory=dict)
    objective_total: float = 0.0
    shaping_total: float = 0.0

    def compute_totals(self) -> None:
        """Sum all components into objective and shaping totals."""
        self.objective_total = (
            self.workflow_reward.get("total", 0.0)
            + self.time_reward.get("total", 0.0)
            + self.asymmetric_reward.get("total", 0.0)
            + self.queue_reward.get("total", 0.0)
            + sum(cr.get("total", 0.0) for cr in self.cascade_rewards.values())
        )
        # VOI is shaping only (does not affect objective score)
        self.shaping_total = self.voi_reward.get("total", 0.0)


def compute_step_reward(
    case: CaseProto,
    queue: QueueProto,
    action_type: str,
    episode_metrics: EpisodeMetricsProto,
    t_remaining: float | None = None,
    t_total: float | None = None,
    already_breached: bool = False,
    evidence_type: str | None = None,
    evidence_items_before: int = 0,
    evidence_items_after: int = 0,
    evidence_time_cost: float = 0.0,
    evidence_already_gathered: set[str] | None = None,
) -> RewardBreakdown:
    """
    Master reward computation for a single step.

    This orchestrator calls the appropriate per-workflow reward function,
    adds time pressure, VOI for evidence-gathering actions, and cascade
    effects. It returns a full breakdown for logging.

    The separation between objective_total and shaping_total is maintained:
    - objective_total: used for benchmark scoring and evaluation
    - shaping_total: used only during training (potential-based shaping)
    """
    result = RewardBreakdown(case_id=case.case_id, case_type=case.case_type)

    # --- Per-workflow reward (only on resolution actions) ---
    resolution_actions = {
        "approve",
        "reject",
        "escalate",
        "close_case",
        "accept_dispute",
        "refund_pre_dispute_alert",
    }
    if action_type in resolution_actions or (action_type == "resolve_prearbitration" and case.resolution != Resolution.PENDING):
        if case.case_type == CaseType.REFUND:
            result.workflow_reward = compute_refund_reward(case, queue)
        elif case.case_type == CaseType.INVOICE:
            result.workflow_reward = compute_invoice_reward(case, queue)
        elif case.case_type == CaseType.KYC:
            result.workflow_reward = compute_kyc_reward(case, queue)
        # Triage reward is computed at queue level, not here

    # --- Time pressure (on every step) ---
    if t_remaining is not None and t_total is not None:
        result.time_reward = compute_time_reward(
            t_remaining, t_total, case.priority,
            already_breached=already_breached,
        )

    # --- VOI reward (on evidence-gathering actions) ---
    if action_type in ("open_case", "view_record", "query_policy",
                       "search_cases", "inspect_audit"):
        if evidence_type is not None:
            result.voi_reward = compute_voi_reward(
                case=case,
                evidence_type=evidence_type,
                items_before=evidence_items_before,
                items_after=evidence_items_after,
                time_cost_minutes=evidence_time_cost,
                already_gathered=evidence_already_gathered,
            )

    # --- Asymmetric error cost (on resolution actions) ---
    # accept_dispute and refund_pre_dispute_alert are excluded because they
    # are economic decisions (dispute cost vs recovery), not fraud classification.
    # Their correctness is already captured in compute_refund_reward via
    # dispute_should_accept logic.
    if action_type in ("approve", "reject"):
        predicted_positive = (action_type == "reject")
        # "positive" meaning "flagged as problematic"
        actual_positive_map = {
            CaseType.REFUND: float(_hidden(case, "true_fraud_risk", 0.0)) > 0.5,
            CaseType.INVOICE: bool(_hidden(case, "true_is_duplicate", False)),
            CaseType.KYC: not bool(_hidden(case, "true_doc_valid", True)),
            CaseType.TRIAGE: case.priority <= 2,
        }
        actual_positive = actual_positive_map.get(case.case_type, False)
        result.asymmetric_reward = compute_asymmetric_reward(
            case_type=case.case_type,
            predicted_positive=predicted_positive,
            actual_positive=actual_positive,
            amount=case.amount,
        )

    # --- Cascade effects (on actions that create downstream events) ---
    if action_type == "escalate":
        result.cascade_rewards["escalation"] = compute_escalation_cascade(
            queue, new_escalation=True
        )

    if action_type == "approve" and case.case_type == CaseType.REFUND:
        if float(_hidden(case, "true_fraud_risk", 0.0)) > 0.5:
            result.cascade_rewards["fraud_rate"] = compute_fraud_cascade(
                cumulative_fraud_approvals=episode_metrics.chargebacks + 1,
                total_decisions=episode_metrics.cases_resolved + 1,
            )

    if action_type == "approve" and case.case_type == CaseType.KYC:
        if not bool(_wf(case, "kyc_complete", False)):
            result.cascade_rewards["compliance"] = compute_compliance_cascade(
                compliance_failures=episode_metrics.compliance_violations + 1,
                remaining_kyc_cases=sum(
                    1 for c in queue.cases
                    if c.case_type == CaseType.KYC and c.resolution == Resolution.PENDING
                ),
            )

    if action_type == "approve" and case.case_type == CaseType.INVOICE:
        if bool(_hidden(case, "true_is_duplicate", False)):
            result.cascade_rewards["cashflow"] = compute_cashflow_cascade(
                duplicate_amount_approved=case.amount,
            )

    if action_type == "approve_qa":
        result.cascade_rewards["qa"] = compute_qa_cascade(
            qa_passed=True,
            qa_required=bool(_wf(case, "qa_required", False)),
            rework_count=len(getattr(case, "qa_history", [])),
        )

    if action_type == "fail_qa":
        result.cascade_rewards["qa"] = compute_qa_cascade(
            qa_passed=False,
            qa_required=bool(_wf(case, "qa_required", False)),
            rework_count=len(getattr(case, "qa_history", [])),
        )

    if action_type in {
        "run_sanctions_screen",
        "start_edd_review",
        "review_beneficial_owner",
        "request_field_correction",
        "file_ofac_report",
        "freeze_payments",
    }:
        result.cascade_rewards["kyc_compliance"] = compute_kyc_compliance_cascade(case, action_type)

    if action_type in {
        "challenge_dispute",
        "refund_pre_dispute_alert",
        "resolve_prearbitration",
        "freeze_payouts",
        "unfreeze_payouts",
        "set_reserve_percent",
        "clear_reserve",
        "set_payout_delay_days",
    }:
        result.cascade_rewards["refund_risk"] = compute_refund_risk_cascade(case, action_type)

    result.compute_totals()
    return result


# ============================================================================
# SECTION 8: POTENTIAL-BASED SHAPING (PHI FUNCTIONS)
# ============================================================================

# The shaping reward preserves optimal policy per Ng, Harada, Russell (1999).
# R_shape = gamma * Phi(s') - Phi(s)
#
# We define Phi as a function of observable progress indicators.
# Phi is NOT a reward -- it is a potential function whose CHANGE is the reward.

def phi_refund(case: CaseProto) -> float:
    """Potential function for refund workflow progress."""
    phi = 0.0
    # Evidence gathering progress
    if case.evidence_items_available > 0:
        phi += 0.35 * (case.evidence_items_gathered / case.evidence_items_available)
    # Required checks completed
    if case.checks_required > 0:
        phi += 0.25 * (case.checks_completed / case.checks_required)
    # Policy checked
    if case.policy_checked:
        phi += 0.20
    # Customer notified (prerequisite for closing)
    if case.customer_notified:
        phi += 0.20
    if _wf(case, "dispute_stage") in {"evidence_submitted", "won", "finalized"}:
        phi += 0.10
    return phi


def phi_invoice(case: CaseProto) -> float:
    """Potential function for invoice reconciliation progress."""
    phi = 0.0
    # Three-way match progress (PO, invoice, receipt)
    if case.checks_required > 0:
        phi += 0.40 * (case.checks_completed / case.checks_required)
    # Evidence (additional documents, vendor records)
    if case.evidence_items_available > 0:
        phi += 0.30 * (case.evidence_items_gathered / case.evidence_items_available)
    # Policy checked (mismatch tolerance)
    if case.policy_checked:
        phi += 0.15
    # Communication (vendor contacted if needed)
    if case.notifications_required > 0:
        phi += 0.15 * (case.notifications_sent / case.notifications_required)
    if _wf(case, "credit_memo_status") in {"requested", "received", "applied"}:
        phi += 0.10
    if _wf(case, "approval_status") in {"pending_secondary", "approved"}:
        phi += 0.05
    return phi


def phi_kyc(case: CaseProto) -> float:
    """Potential function for KYC compliance progress."""
    phi = 0.0
    # KYC document completeness
    if case.checks_required > 0:
        phi += 0.45 * (case.checks_completed / case.checks_required)
    # Compliance policy reviewed
    if case.policy_checked:
        phi += 0.25
    # Communication (document requests sent)
    if case.notifications_required > 0:
        phi += 0.15 * (case.notifications_sent / case.notifications_required)
    # Evidence (risk assessment, identity verification)
    if case.evidence_items_available > 0:
        phi += 0.15 * (case.evidence_items_gathered / case.evidence_items_available)
    return phi


def phi_triage(queue: QueueProto) -> float:
    """
    Potential function for queue triage progress.

    Unlike per-case Phi, this measures queue-level health.
    """
    if not queue.cases:
        return 1.0  # all done

    total = len(queue.cases)
    resolved = sum(1 for c in queue.cases if c.resolution != Resolution.PENDING)
    breached = sum(
        1 for c in queue.cases
        if (c.sla_deadline - queue.current_time) < 0
    )

    phi = 0.0
    # Resolution progress
    phi += 0.40 * (resolved / total)
    # SLA health (fraction not breached)
    phi += 0.35 * (1.0 - breached / total)
    # Escalation queue health
    esc_health = 1.0 - (
        queue.escalation_queue_load / max(1, queue.escalation_queue_capacity)
    )
    phi += 0.25 * max(0.0, esc_health)
    if queue.exception_queue_size > 0:
        assignment_health = 1.0 - (queue.unassigned_count / max(1, queue.exception_queue_size))
        phi += 0.15 * max(0.0, assignment_health)
    if queue.overdue_follow_ups > 0:
        phi -= min(0.15, 0.05 * queue.overdue_follow_ups)
    return phi


def compute_queue_shaping_reward(
    queue: QueueProto,
    prev_queue: QueueProto | None = None,
    gamma: float = 0.99,
    shaping_lambda: float = 0.5,
) -> float:
    phi_now = phi_triage(queue)
    phi_prev = phi_triage(prev_queue) if prev_queue else 0.0
    return shaping_lambda * (gamma * phi_now - phi_prev)


def compute_shaping_reward(
    case: CaseProto,
    queue: QueueProto,
    prev_case: CaseProto | None = None,
    prev_queue: QueueProto | None = None,
    gamma: float = 0.99,
    shaping_lambda: float = 0.5,
) -> float:
    """
    Compute potential-based shaping reward.

    R_shape = lambda * (gamma * Phi(s') - Phi(s))

    This is ADDED to the objective reward during training only.
    It is NEVER reported as benchmark score.
    """
    # Select appropriate phi function
    phi_fn_map = {
        CaseType.REFUND: phi_refund,
        CaseType.INVOICE: phi_invoice,
        CaseType.KYC: phi_kyc,
    }

    if case.case_type == CaseType.TRIAGE:
        phi_now = phi_triage(queue)
        phi_prev = phi_triage(prev_queue) if prev_queue else 0.0
    else:
        phi_fn = phi_fn_map.get(case.case_type, phi_refund)
        phi_now = phi_fn(case)
        phi_prev = phi_fn(prev_case) if prev_case else 0.0

    return shaping_lambda * (gamma * phi_now - phi_prev)


# ============================================================================
# SECTION 9: REWARD VALIDATION / ANTI-GAMING TESTS
# ============================================================================

def validate_reward_against_pathological_policies(
    reward_fn,
    scenarios: list[dict],
) -> dict[str, Any]:
    """
    Test reward function against four pathological policies.

    A well-designed reward should satisfy:
    1. always_approve should score POORLY (misses fraud/duplicates)
    2. always_escalate should score POORLY (overloads human queue)
    3. always_gather_info should score POORLY (never resolves, SLA breach)
    4. greedy_closer should score MODERATELY (fast but error-prone)
    5. A reasonable policy should score BEST

    This function returns scores for each policy so developers can
    verify the reward design doesn't accidentally incentivize degenerate
    behavior.
    """
    results: dict[str, Any] = {
        "description": (
            "Reward validation against pathological policies. "
            "A well-designed reward should rank: "
            "reasonable > greedy_closer > always_approve > always_escalate > always_gather_info"
        ),
        "policies_tested": [
            "always_approve",
            "always_escalate",
            "always_gather_info",
            "greedy_closer",
        ],
        "expected_ranking": [
            "reasonable_policy",
            "greedy_closer",
            "always_approve",
            "always_escalate",
            "always_gather_info",
        ],
        "note": (
            "Implementation deferred to test suite. Call with actual scenarios "
            "and reward functions to validate calibration."
        ),
    }
    return results
