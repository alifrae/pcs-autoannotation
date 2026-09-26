from __future__ import annotations

import json

import pytest

from app.autoannotation.association import (
    CameraLidarCalibration,
    associate_proposals,
    load_pcs_calibration,
    project_box_3d,
)
from app.autoannotation.contracts import (
    BoundingBox2D,
    BoundingBox3D,
    ObjectProposal,
)


def _calibration() -> CameraLidarCalibration:
    return CameraLidarCalibration(
        width=100,
        height=100,
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
        rotation_row_major=(0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0),
        translation_m=(0.0, 0.0, 0.0),
    )


def test_project_box_uses_pcs_lidar_to_camera_convention() -> None:
    projected = project_box_3d(
        BoundingBox3D(
            center_x=10.0,
            center_y=0.0,
            center_z=0.0,
            length=4.0,
            width=2.0,
            height=2.0,
            yaw_rad=0.0,
        ),
        _calibration(),
    )

    assert projected is not None
    assert projected.x1 < 50.0 < projected.x2
    assert projected.y1 < 50.0 < projected.y2


def test_association_is_class_aware_and_one_to_one() -> None:
    lidar = ObjectProposal(
        proposal_id="lidar-1",
        sample_id="sample",
        class_name="Car",
        bbox_3d=BoundingBox3D(10.0, 0.0, 0.0, 4.0, 2.0, 2.0, 0.0),
    )
    projected = project_box_3d(lidar.bbox_3d, _calibration())
    assert projected is not None
    camera = ObjectProposal(
        proposal_id="camera-1",
        sample_id="sample",
        class_name="cars",
        bbox_2d=BoundingBox2D(
            projected.x1,
            projected.y1,
            projected.x2,
            projected.y2,
        ),
    )

    result = associate_proposals((lidar,), (camera,), _calibration(), min_iou=0.5)

    assert len(result.matches) == 1
    assert result.matches[0].score == pytest.approx(1.0)
    assert result.matches[0].fused.bbox_3d == lidar.bbox_3d
    assert result.unmatched_lidar == ()
    assert result.unmatched_camera == ()


def test_load_verified_pcs_calibration_input_sidecar(tmp_path) -> None:
    path = tmp_path / "calibration-input.json"
    path.write_text(
        json.dumps(
            {
                "camera": {
                    "width": 4096,
                    "height": 960,
                    "image_state": {
                        "status": "verified",
                        "rectified": True,
                    },
                },
                "intrinsics": {
                    "status": "verified",
                    "fx_px": 2000.0,
                    "fy_px": 1990.0,
                    "cx_px": 2048.0,
                    "cy_px": 480.0,
                },
                "distortion": {
                    "status": "verified",
                    "model": "none",
                    "coefficients": [],
                },
                "lidar_to_camera": {
                    "status": "verified",
                    "rotation_row_major": [
                        [0.0, -1.0, 0.0],
                        [0.0, 0.0, -1.0],
                        [1.0, 0.0, 0.0],
                    ],
                    "translation_m": [0.0, 0.0, 0.0],
                },
            }
        )
    )

    calibration = load_pcs_calibration(path)

    assert calibration.width == 4096
    assert calibration.height == 960
    assert calibration.rotation_row_major[-3:] == (1.0, 0.0, 0.0)


def test_unverified_pcs_calibration_fails_closed(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "camera": {
                    "width": 4096,
                    "height": 960,
                    "image_state": {"status": "unknown", "rectified": None},
                },
                "intrinsics": {"status": "unknown"},
                "distortion": {"status": "unknown"},
                "lidar_to_camera": {"status": "unknown"},
            }
        )
    )

    with pytest.raises(ValueError, match="not verified"):
        load_pcs_calibration(path)
