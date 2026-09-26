from __future__ import annotations

import numpy as np

from app.autoannotation.contracts import CameraFrame, LidarFrame, SynchronizedSample
from app.autoannotation.providers.innov3_dsvt import Innov3RawDetections
from app.autoannotation.providers.innov3_provider import Innov3DsvtProvider


class FakeRuntime:
    def infer(
        self,
        points_xyzi: np.ndarray,
        *,
        sample_id: str = "",
    ) -> Innov3RawDetections:
        assert sample_id == "sample-1"
        assert points_xyzi.shape == (1, 4)
        return Innov3RawDetections(
            boxes=np.asarray(
                [[10.0, -2.0, -1.0, 4.0, 2.0, 1.5, -0.5]],
                dtype=np.float32,
            ),
            scores=np.asarray([0.9], dtype=np.float32),
            labels=np.asarray([1], dtype=np.int64),
        )


def test_innov3_provider_returns_pcs_coordinate_proposal() -> None:
    lidar = LidarFrame(
        timestamp_ns=100,
        points=np.asarray([[10.0, 2.0, 1.0]], dtype=np.float32),
        attributes={
            "echo_index": np.asarray([2], dtype=np.int32),
            "peak": np.asarray([255.0], dtype=np.float32),
            "slot_index": np.asarray([0], dtype=np.int32),
            "layer_index": np.asarray([0], dtype=np.int32),
        },
    )
    camera = CameraFrame(
        timestamp_ns=100,
        width=10,
        height=10,
        encoding="jpeg",
        data=b"",
        source_id="camera",
    )
    sample = SynchronizedSample(
        sample_id="sample-1",
        timestamp_ns=100,
        lidar=lidar,
        camera=camera,
        adma=None,
    )

    proposals = Innov3DsvtProvider(
        runtime=FakeRuntime(),
        housing_merged=True,
    ).infer(sample)

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.class_name == "Car"
    assert proposal.bbox_3d is not None
    assert proposal.bbox_3d.center_x == 10.0
    assert proposal.bbox_3d.center_y == 2.0
    assert proposal.bbox_3d.center_z == 1.0
    assert proposal.bbox_3d.yaw_rad == 0.5
    assert proposal.evidence[0].confidence == pytest.approx(0.9)
