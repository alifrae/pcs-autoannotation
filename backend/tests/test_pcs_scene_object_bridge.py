from __future__ import annotations

import json

import numpy as np

from app.autoannotation.contracts import (
    BoundingBox3D,
    CameraFrame,
    LidarFrame,
    ObjectProposal,
    SynchronizedSample,
)
from app.autoannotation.pcs_scene_objects import (
    BOX3D_CONVENTION,
    build_pcs_scene_object_document,
    canonical_scene_object_json,
    review_progress,
    validate_pcs_scene_object_document,
)


def _sample() -> SynchronizedSample:
    return SynchronizedSample(
        sample_id="sample-1",
        timestamp_ns=100,
        lidar=LidarFrame(
            timestamp_ns=100,
            points=np.empty((0, 3), dtype=np.float32),
            metadata={"frame_index": 7},
        ),
        camera=CameraFrame(
            timestamp_ns=100,
            width=100,
            height=50,
            encoding="jpeg",
            data=b"",
            source_id="camera",
        ),
        adma=None,
        source_metadata={"lidar_stream_name": "ScaLa 3-PointCloud"},
    )


def test_scene_object_document_matches_pcs_v1_contract() -> None:
    proposal = ObjectProposal(
        proposal_id="proposal-1",
        sample_id="sample-1",
        class_name="Car",
        bbox_3d=BoundingBox3D(10.0, 2.0, 1.0, 4.0, 2.0, 1.5, 0.2),
        fused_confidence=0.8,
    )

    document = build_pcs_scene_object_document(
        _sample(),
        (proposal,),
        recording_key="highway-v1",
        recording_sha256="a" * 64,
        run_id="run-1",
    )

    validate_pcs_scene_object_document(document)
    frame = document["frames"][0]
    assert document["schema"] == "pcs.scene_objects"
    assert document["schema_version"] == 1
    assert frame["sample_reference"]["sample_key"] == {"kind": "ordinal", "value": 7}
    assert frame["hypotheses"][0]["box"]["convention"] == BOX3D_CONVENTION
    assert json.loads(canonical_scene_object_json(document)) == document


def test_review_progress_keeps_human_decision_separate() -> None:
    proposal = ObjectProposal(
        proposal_id="proposal-1",
        sample_id="sample-1",
        class_name="Car",
        bbox_3d=BoundingBox3D(10.0, 2.0, 1.0, 4.0, 2.0, 1.5, 0.2),
    )
    document = build_pcs_scene_object_document(
        _sample(),
        (proposal,),
        recording_key="highway-v1",
        recording_sha256="a" * 64,
    )
    document["frames"][0]["reviews"].append(
        {
            "hypothesis_id": "proposal-1",
            "decision": "accepted",
            "reviewer": "human",
            "corrected_class": None,
            "corrected_box": None,
            "notes": None,
        }
    )

    progress = review_progress(document)

    assert progress["hypothesis_count"] == 1
    assert progress["accepted_count"] == 1
    assert progress["pending_count"] == 0
    assert progress["accept_without_edit_rate"] == 1.0
