from __future__ import annotations

from .contracts import (
    AnnotationRevision,
    ObjectProposal,
    ProposalReview,
    ReviewDisposition,
)


def resolve_reviewed_annotation(
    proposal: ObjectProposal,
    review: ProposalReview,
) -> AnnotationRevision | None:
    """Resolve final annotation state without modifying the machine proposal."""

    if review.proposal_id != proposal.proposal_id:
        raise ValueError(
            f"Review proposal_id {review.proposal_id!r} does not match "
            f"{proposal.proposal_id!r}"
        )
    if review.sample_id != proposal.sample_id:
        raise ValueError(
            f"Review sample_id {review.sample_id!r} does not match "
            f"{proposal.sample_id!r}"
        )

    if review.disposition == ReviewDisposition.REJECTED:
        return None
    if review.disposition == ReviewDisposition.MODIFIED:
        assert review.revision is not None
        return review.revision
    if review.disposition == ReviewDisposition.ACCEPTED:
        return AnnotationRevision(
            class_name=proposal.class_name,
            bbox_2d=proposal.bbox_2d,
            bbox_3d=proposal.bbox_3d,
            mask_polygon=proposal.mask_polygon,
            metadata={"source": "accepted_machine_proposal"},
        )
    raise ValueError(f"Unsupported review disposition: {review.disposition}")
