from __future__ import annotations

from opsarena.domain.events import EDDResponseDueEvent, InfoResponseEvent, ReportDeadlineMissedEvent, SanctionsFalsePositiveClearedEvent
from opsarena.domain.workflows.kyc import (
    BeneficialOwnerStatus,
    EDDStatus,
    KYCStage,
    OFACReportStatus,
    SanctionsStatus,
)
from opsarena.engine.handlers.common import linked_record_id, require_case, require_kyc_workflow, sync_kyc_flags
from opsarena.engine.handlers.result import TransitionResult
from opsarena.engine.scheduler import schedule_event
from opsarena.engine.state import WorldState
from opsarena.enums import CaseType, RecordType
from opsarena.models import (
    FileOFACReportAction,
    FreezePaymentsAction,
    RequestFieldCorrectionAction,
    ReviewBeneficialOwnerAction,
    ReviewKYCAction,
    RunSanctionsScreenAction,
    StartEDDReviewAction,
    TriggerReverificationAction,
)


def _verification_record(state: WorldState, case):
    verification_id = linked_record_id(case, RecordType.KYC_DOCUMENT)
    if verification_id is None:
        raise ValueError("kyc_record_missing")
    return state.records.kyc_verifications[verification_id]


def handle_review_kyc(state: WorldState, action: ReviewKYCAction):
    case = require_case(state, action.case_id)
    if case.case_type != CaseType.KYC:
        raise ValueError("kyc_review_not_supported")
    verification = _verification_record(state, case)
    workflow = require_kyc_workflow(case)
    case.mark_check("review_kyc_profile")
    case.mark_check("review_document")
    if action.verification_decision.value == "approve":
        if not case.hidden.true_doc_valid:
            raise ValueError("invalid_document")
        if verification.requirements_currently_due or "individual.verification.document" in case.pending_info_fields:
            raise ValueError("required_documents_outstanding")
        if verification.status not in {"pending_review", "verified"}:
            raise ValueError("document_not_ready_for_review")
        if not workflow.approval_ready():
            raise ValueError("compliance_review_incomplete")
        verification.status = "verified"
        verification.requirements_currently_due = []
        workflow.kyc_complete = True
        workflow.kyc_stage = KYCStage.CLEARED
        workflow.verification_status = verification.status
    elif action.verification_decision.value == "request_resubmission":
        verification.status = "requires_input"
        verification.requirements_currently_due = (
            verification.requirements_currently_due
            or case.hidden.hidden_required_documents
            or ["individual.verification.document"]
        )
        case.pending_info_fields = list(dict.fromkeys(case.pending_info_fields + verification.requirements_currently_due))
        workflow.kyc_complete = False
        workflow.kyc_stage = KYCStage.REVERIFICATION_REQUIRED
        workflow.verification_status = verification.status
    else:
        verification.status = "rejected"
        workflow.kyc_complete = False
        workflow.kyc_stage = KYCStage.REJECTED
        workflow.verification_status = verification.status
    workflow.requirements_due = verification.requirements_currently_due.copy()
    sync_kyc_flags(case)
    return TransitionResult(True, f"KYC review completed for {case.case_id}"), case


def handle_trigger_reverification(state: WorldState, action: TriggerReverificationAction):
    case = require_case(state, action.case_id)
    if case.case_type != CaseType.KYC:
        raise ValueError("kyc_reverification_not_supported")
    verification = _verification_record(state, case)
    requirements = action.requirements or case.hidden.hidden_required_documents or ["individual.verification.document"]
    verification.status = "requires_input"
    verification.requirements_currently_due = list(dict.fromkeys(requirements))
    case.pending_info_fields = list(dict.fromkeys(case.pending_info_fields + verification.requirements_currently_due))
    workflow = require_kyc_workflow(case)
    workflow.kyc_complete = False
    workflow.verification_status = verification.status
    workflow.requirements_due = verification.requirements_currently_due.copy()
    workflow.kyc_stage = KYCStage.REVERIFICATION_REQUIRED
    sync_kyc_flags(case)
    return TransitionResult(True, f"Reverification triggered for {case.case_id}"), case


def handle_run_sanctions_screen(state: WorldState, action: RunSanctionsScreenAction):
    case = require_case(state, action.case_id)
    if case.case_type != CaseType.KYC:
        raise ValueError("sanctions_screen_not_supported")
    verification = _verification_record(state, case)
    workflow = require_kyc_workflow(case)
    workflow.sanctions_screened_at = state.current_time
    verification.sanctions_screening_reference = f"screen_{case.case_id}_{state.current_time}"
    case.mark_check("review_sanctions")

    # Use seeded noise so confidence ranges overlap — the agent can't trivially
    # distinguish false positive from true match based on confidence alone.
    import random
    rng = random.Random(state.scenario_seed + state.current_time)

    if case.hidden.true_sanctions_match:
        workflow.sanctions_status = SanctionsStatus.CONFIRMED_MATCH
        workflow.screening_match_confidence = round(0.85 + rng.uniform(0.0, 0.14), 2)
        if case.hidden.true_ofac_report_required:
            workflow.ofac_report_status = OFACReportStatus.PENDING
            workflow.report_due_at = state.current_time + 15
            if not any(
                event.event_type == "report_deadline_missed" and event.case_id == case.case_id
                for event in state.scheduled_events
            ):
                schedule_event(
                    state,
                    ReportDeadlineMissedEvent(at_time=workflow.report_due_at, case_id=case.case_id),
                )
        verification.sanctions_review_notes.append("confirmed_match")
    elif case.hidden.true_sanctions_false_positive:
        workflow.sanctions_status = SanctionsStatus.POTENTIAL_MATCH
        workflow.screening_match_confidence = round(0.78 + rng.uniform(0.0, 0.16), 2)
        verification.sanctions_review_notes.append("potential_false_positive")
        if not any(
            event.event_type == "sanctions_false_positive_cleared" and event.case_id == case.case_id
            for event in state.scheduled_events
        ):
            schedule_event(
                state,
                SanctionsFalsePositiveClearedEvent(at_time=state.current_time + 12, case_id=case.case_id),
            )
    else:
        workflow.sanctions_status = SanctionsStatus.CLEAR
        workflow.screening_match_confidence = round(rng.uniform(0.02, 0.15), 2)
        verification.sanctions_review_notes.append("clear")
    sync_kyc_flags(case)
    return TransitionResult(True, f"Sanctions screen completed for {case.case_id}"), case


def handle_start_edd_review(state: WorldState, action: StartEDDReviewAction):
    case = require_case(state, action.case_id)
    if case.case_type != CaseType.KYC:
        raise ValueError("edd_review_not_supported")
    verification = _verification_record(state, case)
    workflow = require_kyc_workflow(case)
    workflow.edd_due_at = state.current_time + (case.hidden.hidden_response_latency_minutes or 30) + 10
    verification.edd_notes.append("edd_started")

    if not case.hidden.true_edd_required and not case.hidden.true_beneficial_owner_issue:
        workflow.edd_status = EDDStatus.NOT_REQUIRED
        if workflow.beneficial_owner_status == BeneficialOwnerStatus.NOT_STARTED and verification.beneficial_owners:
            workflow.beneficial_owner_status = BeneficialOwnerStatus.PENDING_REVIEW
    else:
        workflow.edd_status = EDDStatus.IN_PROGRESS
        if verification.beneficial_owners:
            workflow.beneficial_owner_status = (
                BeneficialOwnerStatus.NEEDS_CORRECTION
                if case.hidden.true_beneficial_owner_issue
                else BeneficialOwnerStatus.PENDING_REVIEW
            )
            workflow.correction_fields = list(dict.fromkeys(workflow.correction_fields + case.hidden.hidden_correction_fields))
        schedule_event(
            state,
            EDDResponseDueEvent(at_time=workflow.edd_due_at, case_id=case.case_id, due_at=workflow.edd_due_at),
        )
    sync_kyc_flags(case)
    return TransitionResult(True, f"EDD review started for {case.case_id}"), case


def handle_review_beneficial_owner(state: WorldState, action: ReviewBeneficialOwnerAction):
    case = require_case(state, action.case_id)
    if case.case_type != CaseType.KYC:
        raise ValueError("beneficial_owner_review_not_supported")
    verification = _verification_record(state, case)
    workflow = require_kyc_workflow(case)
    case.mark_check("review_beneficial_owner")

    if action.verification_decision.value == "approve":
        if workflow.correction_fields:
            raise ValueError("beneficial_owner_corrections_pending")
        workflow.beneficial_owner_status = BeneficialOwnerStatus.VERIFIED
        workflow.edd_status = EDDStatus.CLEARED
        for owner in verification.beneficial_owners:
            owner.review_status = "verified"
    elif action.verification_decision.value == "request_resubmission":
        workflow.beneficial_owner_status = BeneficialOwnerStatus.NEEDS_CORRECTION
        workflow.edd_status = EDDStatus.AWAITING_RESPONSE
        workflow.correction_fields = list(dict.fromkeys(workflow.correction_fields + case.hidden.hidden_correction_fields))
        for owner in verification.beneficial_owners:
            owner.review_status = "needs_correction"
    else:
        workflow.beneficial_owner_status = BeneficialOwnerStatus.REJECTED
        for owner in verification.beneficial_owners:
            owner.review_status = "rejected"
    sync_kyc_flags(case)
    return TransitionResult(True, f"Beneficial owner review completed for {case.case_id}"), case


def handle_request_field_correction(state: WorldState, action: RequestFieldCorrectionAction):
    case = require_case(state, action.case_id)
    if case.case_type != CaseType.KYC:
        raise ValueError("field_correction_not_supported")
    verification = _verification_record(state, case)
    workflow = require_kyc_workflow(case)
    allowed_fields = set(case.hidden.hidden_correction_fields) | set(workflow.correction_fields)
    if action.field_name not in allowed_fields:
        raise ValueError("unknown_correction_field")
    if action.field_name in case.pending_info_fields:
        raise ValueError("duplicate_info_request")
    case.pending_info_fields.append(action.field_name)
    case.requested_info_fields.append(action.field_name)
    case.notifications_sent += 1
    workflow.edd_status = EDDStatus.AWAITING_RESPONSE
    workflow.beneficial_owner_status = BeneficialOwnerStatus.NEEDS_CORRECTION
    workflow.correction_fields = list(dict.fromkeys(workflow.correction_fields + [action.field_name]))
    verification.correction_requests = list(dict.fromkeys(verification.correction_requests + [action.field_name]))
    schedule_event(
        state,
        InfoResponseEvent(
            at_time=state.current_time + (case.hidden.hidden_response_latency_minutes or 30),
            case_id=case.case_id,
            field_name=action.field_name,
        ),
    )
    if workflow.edd_due_at is None or workflow.edd_due_at <= state.current_time:
        workflow.edd_due_at = state.current_time + (case.hidden.hidden_response_latency_minutes or 30) + 10
        schedule_event(
            state,
            EDDResponseDueEvent(at_time=workflow.edd_due_at, case_id=case.case_id, due_at=workflow.edd_due_at),
        )
    sync_kyc_flags(case)
    return TransitionResult(True, f"Requested correction for {action.field_name}"), case


def handle_file_ofac_report(state: WorldState, action: FileOFACReportAction):
    case = require_case(state, action.case_id)
    if case.case_type != CaseType.KYC:
        raise ValueError("ofac_report_not_supported")
    verification = _verification_record(state, case)
    workflow = require_kyc_workflow(case)
    if workflow.ofac_report_status not in {OFACReportStatus.PENDING, OFACReportStatus.MISSED}:
        raise ValueError("no_report_due")
    workflow.ofac_report_status = OFACReportStatus.FILED
    workflow.report_due_at = None
    verification.ofac_case_id = f"ofac_{case.case_id}_{state.current_time}"
    state.metrics.ofac_reports_filed += 1
    sync_kyc_flags(case)
    return TransitionResult(True, f"OFAC report filed for {case.case_id}"), case


def handle_freeze_payments(state: WorldState, action: FreezePaymentsAction):
    case = require_case(state, action.case_id)
    if case.case_type != CaseType.KYC:
        raise ValueError("freeze_payments_not_supported")
    workflow = require_kyc_workflow(case)
    workflow.payments_frozen = True
    workflow.payment_freeze_reason = action.reason_code.value
    state.metrics.payment_freezes += 1
    sync_kyc_flags(case)
    return TransitionResult(True, f"Payments frozen for {case.case_id}"), case
