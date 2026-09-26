from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..contracts import (
    BoundingBox3D,
    ObjectProposal,
    ProviderEvidence,
    SensorModality,
    SynchronizedSample,
)
from .innov3_dsvt import (
    INNOV3_CLASSES,
    Innov3InferenceRuntime,
    innov3_identity,
    prepare_innov3_input,
)


@dataclass(slots=True)
class Innov3DsvtProvider:
    runtime: Innov3InferenceRuntime
    housing_merged: bool

    @property
    def identity(self):
        return innov3_identity()

    def infer(self, sample: SynchronizedSample) -> Sequence[ObjectProposal]:
        prepared = prepare_innov3_input(
            sample.lidar,
            housing_merged=self.housing_merged,
        )
        raw = self.runtime.infer(prepared.points_xyzi, sample_id=sample.sample_id)

        proposals: list[ObjectProposal] = []
        for index, (box, score, label) in enumerate(
            zip(raw.boxes, raw.scores, raw.labels, strict=True)
        ):
            values = np.asarray(box, dtype=np.float32).reshape(-1)
            if values.size < 7:
                raise RuntimeError(
                    f"Innov3 returned a box with {values.size} values; expected at least 7"
                )

            class_index = int(label) - 1
            class_name = (
                INNOV3_CLASSES[class_index]
                if 0 <= class_index < len(INNOV3_CLASSES)
                else str(int(label))
            )
            proposal_id = hashlib.sha256(
                f"{sample.sample_id}|{self.identity.provider_id}|{index}".encode()
            ).hexdigest()[:24]

            proposals.append(
                ObjectProposal(
                    proposal_id=proposal_id,
                    sample_id=sample.sample_id,
                    class_name=class_name,
                    bbox_3d=_model_box_to_pcs(values),
                    evidence=(
                        ProviderEvidence(
                            provider=self.identity,
                            modality=SensorModality.LIDAR,
                            confidence=float(score),
                            metadata={
                                "selected_source_points": int(
                                    len(prepared.source_indices)
                                ),
                                "source_point_count": prepared.source_point_count,
                                "semantic_echo_id": prepared.semantic_echo_id,
                                "coordinate_contract": "innov3_reference_flip_yz",
                                "implementation_parity": "qualified_3frame_highway",
                                "reviewed_quality": "unqualified",
                            },
                        ),
                    ),
                )
            )
        return tuple(proposals)


def _model_box_to_pcs(values: np.ndarray) -> BoundingBox3D:
    """Invert the frozen Innov3 Y/Z input transform back to PCS coordinates."""

    return BoundingBox3D(
        center_x=float(values[0]),
        center_y=-float(values[1]),
        center_z=-float(values[2]),
        length=float(values[3]),
        width=float(values[4]),
        height=float(values[5]),
        yaw_rad=-float(values[6]),
    )
