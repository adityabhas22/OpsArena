from __future__ import annotations

from opsarena.domain.workflows.kyc import KYCStage
from opsarena.engine.handlers.common import linked_record_id, require_case, require_kyc_workflow
from opsarena.engine.handlers.result import TransitionResult
from opsarena.engine.state import WorldState
from opsarena.enums import CaseType, RecordType
from opsarena.models import ReviewKYCAction, TriggerReverificationAction


def handle_review_kyc(state: WorldState, action: ReviewKYCAction):
    case = require_case(state, action.case_id)
    if case.case_type != CaseType.KYC:
        raise ValueError("kyc_review_not_supported")
    verification_id = linked_record_id(case, RecordType.KYC_DOCUMENT)
    if verification_id is None:
        raise ValueError("kyc_record_missing")
    verification = state.records.kyc_verifications[verification_id]
    workflow = require_kyc_workflow(case)
    case.mark_check("review_kyc_profile")
    case.mark_check("review_document")
    if action.verification_decision.value == "approve":
        verification.status = "verified"
        verification.requirements_currently_due = []
        workflow.kyc_complete = True
        workflow.kyc_stage = KYCStage.CLEARED
        if not case.hidden.true_doc_valid:
            state.metrics.compliance_violations += 1
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
    else:
        verification.status = "rejected"
        workflow.kyc_complete = False
        workflow.kyc_stage = KYCStage.REJECTED
    workflow.verification_status = verification.status
    workflow.requirements_due = verification.requirements_currently_due.copy()
    workflow.payout_hold = not workflow.kyc_complete
    return TransitionResult(True, f"KYC review completed for {case.case_id}"), case


def handle_trigger_reverification(state: WorldState, action: TriggerReverificationAction):
    case = require_case(state, action.case_id)
    if case.case_type != CaseType.KYC:
        raise ValueError("kyc_reverification_not_supported")
    verification_id = linked_record_id(case, RecordType.KYC_DOCUMENT)
    if verification_id is None:
        raise ValueError("kyc_record_missing")
    verification = state.records.kyc_verifications[verification_id]
    requirements = action.requirements or case.hidden.hidden_required_documents or ["individual.verification.document"]
    verification.status = "requires_input"
    verification.requirements_currently_due = list(dict.fromkeys(requirements))
    case.pending_info_fields = list(dict.fromkeys(case.pending_info_fields + verification.requirements_currently_due))
    workflow = require_kyc_workflow(case)
    workflow.kyc_complete = False
    workflow.verification_status = verification.status
    workflow.requirements_due = verification.requirements_currently_due.copy()
    workflow.payout_hold = True
    workflow.kyc_stage = KYCStage.REVERIFICATION_REQUIRED
    return TransitionResult(True, f"Reverification triggered for {case.case_id}"), case
