from __future__ import annotations

import numpy as np

from app.autoannotation.contracts import CameraFrame, LidarFrame, SynchronizedSample
from app.autoannotation.providers.camera_provider import (
    LocateAnythingSam2Provider,
    camera_frame_to_pil,
)
from app.services.locate_anything import parse_boxes


def _sample(camera: CameraFrame) -> SynchronizedSample:
    return SynchronizedSample(
        sample_id="sample-camera",
        timestamp_ns=camera.timestamp_ns,
        lidar=LidarFrame(
            timestamp_ns=camera.timestamp_ns,
            points=np.empty((0, 3), dtype=np.float32),
        ),
        camera=camera,
        adma=None,
    )


def test_locate_anything_confidence_applies_to_preceding_box() -> None:
    raw = (
        "<ref>car</ref>"
        "<box><100><200><300><400></box><conf>0.91</conf>"
        "<box><500><600><700><800></box><conf>0.72</conf>"
    )

    boxes = parse_boxes(raw, 1000, 1000)

    assert [box["confidence"] for box in boxes] == [0.91, 0.72]


def test_raw_bgr_camera_frame_is_converted_using_pcs_encoding() -> None:
    frame = CameraFrame(
        timestamp_ns=10,
        width=1,
        height=1,
        encoding="bgr8",
        data=bytes([10, 20, 30]),
        source_id="front-camera",
    )

    image = camera_frame_to_pil(frame)
    try:
        assert image.mode == "RGB"
        assert image.getpixel((0, 0)) == (30, 20, 10)
    finally:
        image.close()


def test_camera_provider_returns_bbox_and_sam_mask() -> None:
    frame = CameraFrame(
        timestamp_ns=10,
        width=100,
        height=50,
        encoding="rgb8",
        data=bytes([0] * (100 * 50 * 3)),
        source_id="front-camera",
    )

    def fake_detect(_image, _categories, *, source_label):
        assert source_label == "front-camera"
        return {
            "boxes": [
                {
                    "class_name": "car",
                    "x1": 10,
                    "y1": 10,
                    "x2": 30,
                    "y2": 20,
                    "confidence": 0.88,
                }
            ],
            "img_w": 100,
            "img_h": 50,
            "orig_w": 100,
            "orig_h": 50,
        }

    def fake_segment(_image, boxes, *, score_threshold):
        assert len(boxes) == 1
        assert score_threshold == 0.25
        return [[[10.0, 10.0], [30.0, 10.0], [30.0, 20.0]]]

    provider = LocateAnythingSam2Provider(
        ("car", "truck"),
        sam2_score_threshold=0.25,
        detect_fn=fake_detect,
        segment_fn=fake_segment,
    )

    proposals = provider.infer(_sample(frame))

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.class_name == "car"
    assert proposal.bbox_2d is not None
    assert proposal.bbox_2d.x1 == 10
    assert proposal.evidence[0].confidence == 0.88
    assert proposal.mask_polygon == ((10.0, 10.0), (30.0, 10.0), (30.0, 20.0))
