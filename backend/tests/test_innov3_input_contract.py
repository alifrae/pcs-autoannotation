from __future__ import annotations

import numpy as np

from app.autoannotation.contracts import LidarFrame
from app.autoannotation.providers.innov3_dsvt import (
    INNOV3_CHECKPOINT_SHA256,
    innov3_identity,
    prepare_innov3_input,
)


def test_innov3_identity_pins_internal_checkpoint() -> None:
    identity = innov3_identity()

    assert identity.provider_id == "dsvt-internal-innov3"
    assert identity.model_sha256 == INNOV3_CHECKPOINT_SHA256


def test_innov3_input_reproduces_reference_projection() -> None:
    frame = LidarFrame(
        timestamp_ns=100,
        points=np.asarray(
            [
                [10.0, 2.0, 1.0],
                [20.0, 3.0, 2.0],
                [30.0, 4.0, 3.0],
                [250.0, 5.0, 4.0],
            ],
            dtype=np.float32,
        ),
        attributes={
            "echo_index": np.asarray([2, 2, 1, 2], dtype=np.int32),
            "peak": np.asarray([255.0, 127.5, 64.0, 255.0], dtype=np.float32),
            "slot_index": np.asarray([4, 2, 1, 0], dtype=np.int32),
            "layer_index": np.asarray([1, 0, 0, 0], dtype=np.int32),
        },
        metadata={"housing_merged": True},
    )

    prepared = prepare_innov3_input(frame)

    assert prepared.semantic_echo_id == 2
    assert prepared.source_indices.tolist() == [1, 0]
    assert prepared.points_xyzi.shape == (2, 4)
    assert np.allclose(prepared.points_xyzi[0], [20.0, -3.0, -2.0, 0.5])
    assert np.allclose(prepared.points_xyzi[1], [10.0, -2.0, -1.0, 1.0])


def test_innov3_external_echo_mapping_without_housing_merge() -> None:
    frame = LidarFrame(
        timestamp_ns=100,
        points=np.asarray([[10.0, 0.0, 0.0]], dtype=np.float32),
        attributes={
            "echo_index": np.asarray([1], dtype=np.int32),
            "peak": np.asarray([255.0], dtype=np.float32),
            "slot_index": np.asarray([0], dtype=np.int32),
            "layer_index": np.asarray([0], dtype=np.int32),
        },
        metadata={"housing_merged": False},
    )

    prepared = prepare_innov3_input(frame)

    assert prepared.semantic_echo_id == 1
    assert prepared.source_indices.tolist() == [0]
