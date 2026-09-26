from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from app.autoannotation.contracts import (
    AnnotationRevision,
    BoundingBox3D,
    ObjectProposal,
    ProposalReview,
    ReviewDisposition,
)
from app.autoannotation.review import resolve_reviewed_annotation


def _proposal() -> ObjectProposal:
    return ObjectProposal(
        proposal_id="proposal-1",
        sample_id="sample-1",
        class_name="Car",
        bbox_3d=BoundingBox3D(
            center_x=10.0,
            center_y=2.0,
            center_z=1.0,
            length=4.0,
            width=2.0,
            height=1.5,
            yaw_rad=0.1,
        ),
    )


def test_machine_proposal_is_immutable_and_has_no_review_state() -> None:
    proposal = _proposal()

    assert not hasattr(proposal, "disposition")
    with pytest.raises(FrozenInstanceError):
        proposal.class_name = "Truck"  # type: ignore[misc]


def test_accepted_review_resolves_without_mutating_proposal() -> None:
    proposal = _proposal()
    review = ProposalReview(
        review_id="review-1",
        proposal_id=proposal.proposal_id,
        sample_id=proposal.sample_id,
        disposition=ReviewDisposition.ACCEPTED,
        reviewed_at_ns=123,
        reviewer_id="pcs-user",
    )

    resolved = resolve_reviewed_annotation(proposal, review)

    assert resolved is not None
    assert resolved.class_name == "Car"
    assert resolved.bbox_3d == proposal.bbox_3d
    assert proposal.class_name == "Car"


def test_modified_review_keeps_original_and_returns_revision() -> None:
    proposal = _proposal()
    revision = AnnotationRevision(
        class_name="Truck",
        bbox_3d=BoundingBox3D(
            center_x=10.5,
            center_y=2.0,
            center_z=1.0,
            length=5.0,
            width=2.2,
            height=1.8,
            yaw_rad=0.12,
        ),
        metadata={"reason": "human correction"},
    )
    review = ProposalReview(
        review_id="review-2",
        proposal_id=proposal.proposal_id,
        sample_id=proposal.sample_id,
        disposition=ReviewDisposition.MODIFIED,
        reviewed_at_ns=124,
        revision=revision,
    )

    resolved = resolve_reviewed_annotation(proposal, review)

    assert resolved == revision
    assert proposal.class_name == "Car"
    assert proposal.bbox_3d != revision.bbox_3d


def test_rejected_review_resolves_to_no_annotation() -> None:
    proposal = _proposal()
    review = ProposalReview(
        review_id="review-3",
        proposal_id=proposal.proposal_id,
        sample_id=proposal.sample_id,
        disposition=ReviewDisposition.REJECTED,
        reviewed_at_ns=125,
    )

    assert resolve_reviewed_annotation(proposal, review) is None


def test_modified_review_requires_revision() -> None:
    with pytest.raises(ValueError, match="requires a complete annotation revision"):
        ProposalReview(
            review_id="review-4",
            proposal_id="proposal-1",
            sample_id="sample-1",
            disposition=ReviewDisposition.MODIFIED,
            reviewed_at_ns=126,
        )


def test_proposed_is_not_a_human_review_decision() -> None:
    with pytest.raises(ValueError, match="machine state"):
        ProposalReview(
            review_id="review-5",
            proposal_id="proposal-1",
            sample_id="sample-1",
            disposition=ReviewDisposition.PROPOSED,
            reviewed_at_ns=127,
        )
