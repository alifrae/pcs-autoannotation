from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .pcs_scene_objects import review_progress, validate_pcs_scene_object_document


@dataclass(frozen=True, slots=True)
class BaselineQualification:
    hypothesis_count: int
    reviewed_count: int
    pending_count: int
    accepted_count: int
    corrected_count: int
    rejected_count: int
    manual_annotation_count: int
    accept_without_edit_rate: float | None
    correction_rate: float | None
    rejection_rate: float | None
    human_seconds: float | None = None
    seconds_per_reviewed_object: float | None = None

    @property
    def review_complete(self) -> bool:
        return self.pending_count == 0 and self.hypothesis_count > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_count": self.hypothesis_count,
            "reviewed_count": self.reviewed_count,
            "pending_count": self.pending_count,
            "accepted_count": self.accepted_count,
            "corrected_count": self.corrected_count,
            "rejected_count": self.rejected_count,
            "manual_annotation_count": self.manual_annotation_count,
            "accept_without_edit_rate": self.accept_without_edit_rate,
            "correction_rate": self.correction_rate,
            "rejection_rate": self.rejection_rate,
            "human_seconds": self.human_seconds,
            "seconds_per_reviewed_object": self.seconds_per_reviewed_object,
            "review_complete": self.review_complete,
        }


def qualify_review_document(
    document: Mapping[str, Any],
    *,
    human_seconds: float | None = None,
) -> BaselineQualification:
    """Compute decision-neutral baseline metrics from PCS human review output."""

    validate_pcs_scene_object_document(document)
    raw = review_progress(document)
    if human_seconds is not None and human_seconds < 0.0:
        raise ValueError("human_seconds must be non-negative")
    reviewed = int(raw["reviewed_count"])
    seconds_per_object = (
        None if human_seconds is None or reviewed == 0 else float(human_seconds) / reviewed
    )
    return BaselineQualification(
        hypothesis_count=int(raw["hypothesis_count"]),
        reviewed_count=reviewed,
        pending_count=int(raw["pending_count"]),
        accepted_count=int(raw["accepted_count"]),
        corrected_count=int(raw["corrected_count"]),
        rejected_count=int(raw["rejected_count"]),
        manual_annotation_count=int(raw["manual_annotation_count"]),
        accept_without_edit_rate=_optional_float(raw["accept_without_edit_rate"]),
        correction_rate=_optional_float(raw["correction_rate"]),
        rejection_rate=_optional_float(raw["rejection_rate"]),
        human_seconds=human_seconds,
        seconds_per_reviewed_object=seconds_per_object,
    )


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)
