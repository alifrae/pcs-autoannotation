from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray


class SensorModality(StrEnum):
    CAMERA = "camera"
    LIDAR = "lidar"
    ADMA = "adma"


class ReviewDisposition(StrEnum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    MODIFIED = "modified"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class ProviderIdentity:
    provider_id: str
    model_name: str
    model_version: str | None = None
    model_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class LidarFrame:
    timestamp_ns: int
    points: NDArray[np.float32]
    attributes: Mapping[str, NDArray[Any]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CameraFrame:
    timestamp_ns: int
    width: int
    height: int
    encoding: str
    data: bytes
    source_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AdmaSample:
    timestamp_ns: int
    values: Mapping[str, float | int | str]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SynchronizedSample:
    sample_id: str
    timestamp_ns: int
    lidar: LidarFrame
    camera: CameraFrame
    adma: AdmaSample | None
    source_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BoundingBox2D:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True, slots=True)
class BoundingBox3D:
    center_x: float
    center_y: float
    center_z: float
    length: float
    width: float
    height: float
    yaw_rad: float


@dataclass(frozen=True, slots=True)
class ProviderEvidence:
    provider: ProviderIdentity
    modality: SensorModality
    confidence: float | None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ObjectProposal:
    """Immutable machine-generated annotation proposal.

    Human review state is intentionally not stored on this object. Review and
    correction records reference proposal_id and live in a separate domain
    record so the original machine output remains available for quality metrics.
    """

    proposal_id: str
    sample_id: str
    class_name: str
    bbox_2d: BoundingBox2D | None = None
    bbox_3d: BoundingBox3D | None = None
    mask_polygon: Sequence[tuple[float, float]] | None = None
    evidence: Sequence[ProviderEvidence] = field(default_factory=tuple)
    association_score: float | None = None
    fused_confidence: float | None = None


@dataclass(frozen=True, slots=True)
class AnnotationRevision:
    """Complete human-corrected annotation state for a modified proposal."""

    class_name: str
    bbox_2d: BoundingBox2D | None = None
    bbox_3d: BoundingBox3D | None = None
    mask_polygon: Sequence[tuple[float, float]] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.class_name.strip():
            raise ValueError("Annotation revision class_name must not be empty")


@dataclass(frozen=True, slots=True)
class ProposalReview:
    """Append-only human decision referencing an immutable machine proposal."""

    review_id: str
    proposal_id: str
    sample_id: str
    disposition: ReviewDisposition
    reviewed_at_ns: int
    revision: AnnotationRevision | None = None
    reviewer_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.disposition == ReviewDisposition.PROPOSED:
            raise ValueError("PROPOSED is machine state, not a human review decision")
        if self.disposition == ReviewDisposition.MODIFIED and self.revision is None:
            raise ValueError("MODIFIED review requires a complete annotation revision")
        if self.disposition != ReviewDisposition.MODIFIED and self.revision is not None:
            raise ValueError("Only MODIFIED review may contain an annotation revision")


class AnnotationProvider(Protocol):
    @property
    def identity(self) -> ProviderIdentity:
        ...

    def infer(self, sample: SynchronizedSample) -> Sequence[ObjectProposal]:
        ...


class BatchAnnotationProvider(Protocol):
    """Optional provider capability for efficient multi-sample inference."""

    @property
    def identity(self) -> ProviderIdentity:
        ...

    def infer_batch(
        self,
        samples: Sequence[SynchronizedSample],
    ) -> Sequence[Sequence[ObjectProposal]]:
        ...
