from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .association import (
    AssociationResult,
    CameraLidarCalibration,
    associate_proposals,
)
from .contracts import AnnotationProvider, ObjectProposal, SynchronizedSample
from .pcs_scene_objects import build_pcs_scene_object_document


@dataclass(frozen=True, slots=True)
class AutoAnnotationResult:
    sample: SynchronizedSample
    lidar_proposals: tuple[ObjectProposal, ...]
    camera_proposals: tuple[ObjectProposal, ...]
    association: AssociationResult
    scene_object_document: dict[str, Any]

    @property
    def review_proposals(self) -> tuple[ObjectProposal, ...]:
        """3D hypotheses PCS should present for review.

        Matched LiDAR proposals are replaced by their fused hypothesis. Unmatched
        camera-only proposals remain evidence but cannot become a PCS Box3D
        hypothesis without inventing 3D geometry.
        """

        return (
            *self.association.fused_proposals,
            *self.association.unmatched_lidar,
        )


def run_autoannotation_sample(
    sample: SynchronizedSample,
    *,
    lidar_provider: AnnotationProvider,
    camera_provider: AnnotationProvider,
    calibration: CameraLidarCalibration,
    recording_key: str,
    recording_sha256: str,
    min_iou: float = 0.1,
    run_id: str | None = None,
) -> AutoAnnotationResult:
    """Execute the v1 two-provider proposal/fusion/review-contract loop."""

    _validate_calibration_matches_sample(calibration, sample)
    lidar_proposals = tuple(lidar_provider.infer(sample))
    camera_proposals = tuple(camera_provider.infer(sample))
    _validate_sample_ids(sample, lidar_proposals, "LiDAR")
    _validate_sample_ids(sample, camera_proposals, "camera")

    association = associate_proposals(
        lidar_proposals,
        camera_proposals,
        calibration,
        min_iou=min_iou,
    )
    review_proposals: Sequence[ObjectProposal] = (
        *association.fused_proposals,
        *association.unmatched_lidar,
    )
    document = build_pcs_scene_object_document(
        sample,
        review_proposals,
        recording_key=recording_key,
        recording_sha256=recording_sha256,
        run_id=run_id,
    )
    return AutoAnnotationResult(
        sample=sample,
        lidar_proposals=lidar_proposals,
        camera_proposals=camera_proposals,
        association=association,
        scene_object_document=document,
    )


def _validate_calibration_matches_sample(
    calibration: CameraLidarCalibration,
    sample: SynchronizedSample,
) -> None:
    if (
        calibration.width != sample.camera.width
        or calibration.height != sample.camera.height
    ):
        raise ValueError(
            "PCS calibration image size does not match synchronized camera frame: "
            f"calibration={calibration.width}x{calibration.height}, "
            f"frame={sample.camera.width}x{sample.camera.height}"
        )


def _validate_sample_ids(
    sample: SynchronizedSample,
    proposals: Sequence[ObjectProposal],
    provider_name: str,
) -> None:
    mismatched = [
        proposal.proposal_id
        for proposal in proposals
        if proposal.sample_id != sample.sample_id
    ]
    if mismatched:
        raise ValueError(
            f"{provider_name} provider returned proposals for a different sample: "
            + ", ".join(mismatched)
        )
