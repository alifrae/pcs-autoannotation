from __future__ import annotations

import numpy as np

from app.autoannotation.contracts import (
    BoundingBox3D,
    CameraFrame,
    LidarFrame,
    ObjectProposal,
    SynchronizedSample,
)
from app.autoannotation.pcs_scene_objects import build_pcs_scene_object_document
from app.autoannotation.qualification import qualify_review_document


def test_review_qualification_reports_effort_and_dispositions() -> None:
    sample = SynchronizedSample(
        sample_id="sample",
        timestamp_ns=1,
        lidar=LidarFrame(
            timestamp_ns=1,
            points=np.empty((0, 3), dtype=np.float32),
            metadata={"frame_index": 0},
        ),
        camera=CameraFrame(1, 10, 10, "jpeg", b"", "camera"),
        adma=None,
        source_metadata={"lidar_stream_name": "lidar"},
    )
    proposals = (
        ObjectProposal(
            proposal_id="p1",
            sample_id="sample",
            class_name="car",
            bbox_3d=BoundingBox3D(10, 0, 0, 4, 2, 2, 0),
        ),
        ObjectProposal(
            proposal_id="p2",
            sample_id="sample",
            class_name="truck",
            bbox_3d=BoundingBox3D(20, 0, 0, 6, 2.5, 3, 0),
        ),
    )
    document = build_pcs_scene_object_document(
        sample,
        proposals,
        recording_key="trace",
        recording_sha256="a" * 64,
    )
    document["frames"][0]["reviews"] = [
        {
            "hypothesis_id": "p1",
            "decision": "accepted",
            "reviewer": "human",
            "corrected_class": None,
            "corrected_box": None,
            "notes": None,
        },
        {
            "hypothesis_id": "p2",
            "decision": "rejected",
            "reviewer": "human",
            "corrected_class": None,
            "corrected_box": None,
            "notes": None,
        },
    ]

    report = qualify_review_document(document, human_seconds=12.0)

    assert report.review_complete is True
    assert report.accepted_count == 1
    assert report.rejected_count == 1
    assert report.accept_without_edit_rate == 0.5
    assert report.seconds_per_reviewed_object == 6.0
