from __future__ import annotations

import numpy as np
import pytest

from app.autoannotation.association import CameraLidarCalibration
from app.autoannotation.contracts import (
    BoundingBox2D,
    BoundingBox3D,
    CameraFrame,
    LidarFrame,
    ObjectProposal,
    ProviderIdentity,
    SynchronizedSample,
)
from app.autoannotation.pipeline import run_autoannotation_sample


class FakeLidarProvider:
    identity = ProviderIdentity("lidar", "fake")

    def infer(self, sample):
        return (
            ObjectProposal(
                proposal_id="lidar-1",
                sample_id=sample.sample_id,
                class_name="car",
                bbox_3d=BoundingBox3D(10.0, 0.0, 0.0, 4.0, 2.0, 2.0, 0.0),
            ),
        )


class FakeCameraProvider:
    identity = ProviderIdentity("camera", "fake")

    def infer(self, sample):
        return (
            ObjectProposal(
                proposal_id="camera-1",
                sample_id=sample.sample_id,
                class_name="car",
                bbox_2d=BoundingBox2D(35.0, 35.0, 65.0, 65.0),
            ),
        )


def _sample() -> SynchronizedSample:
    return SynchronizedSample(
        sample_id="sample",
        timestamp_ns=1,
        lidar=LidarFrame(
            timestamp_ns=1,
            points=np.empty((0, 3), dtype=np.float32),
            metadata={"frame_index": 0},
        ),
        camera=CameraFrame(1, 100, 100, "jpeg", b"", "camera"),
        adma=None,
        source_metadata={"lidar_stream_name": "lidar"},
    )


def _calibration(width=100, height=100):
    return CameraLidarCalibration(
        width=width,
        height=height,
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
        rotation_row_major=(0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0),
        translation_m=(0.0, 0.0, 0.0),
    )


def test_pipeline_emits_pcs_review_document() -> None:
    result = run_autoannotation_sample(
        _sample(),
        lidar_provider=FakeLidarProvider(),
        camera_provider=FakeCameraProvider(),
        calibration=_calibration(),
        recording_key="trace",
        recording_sha256="b" * 64,
        min_iou=0.01,
    )

    assert len(result.association.matches) == 1
    assert len(result.review_proposals) == 1
    assert result.scene_object_document["schema"] == "pcs.scene_objects"
    assert len(result.scene_object_document["frames"][0]["hypotheses"]) == 1


def test_pipeline_rejects_calibration_for_wrong_camera() -> None:
    with pytest.raises(ValueError, match="image size"):
        run_autoannotation_sample(
            _sample(),
            lidar_provider=FakeLidarProvider(),
            camera_provider=FakeCameraProvider(),
            calibration=_calibration(width=200),
            recording_key="trace",
            recording_sha256="b" * 64,
        )
