from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.autoannotation.pcs_native_dat_source import (
    PcsNativeCapabilityError,
    PcsNativeDatSource,
)
from app.autoannotation.pcs_native_runtime import inspect_pcs_native


class FakeDatSession:
    def __init__(self, _path: str) -> None:
        pass

    def list_streams(self) -> list[dict]:
        return [
            {
                "name": "ScaLa 3-PointCloud",
                "stream_id": 1,
                "kind": "point_cloud",
                "frame_count": 3,
            },
            {
                "name": "FrontCamera",
                "stream_id": 2,
                "kind": "image",
                "frame_count": 30,
                "width": 4096,
                "height": 960,
                "fps": 30.0,
            },
            {
                "name": "ADMA_NET_3330",
                "stream_id": 3,
                "kind": "unknown",
                "frame_count": 300,
            },
        ]


class FakeDatReader:
    def __init__(self, _path: str, **_kwargs) -> None:
        pass

    def get_payload_by_index(self, index: int) -> tuple[bytes, int]:
        assert index == 0
        return b"native-payload", 123456


class FakeImageSource:
    def __init__(self, _path: str, **_kwargs) -> None:
        pass

    def get_nearest_frame(self, timestamp_ns: int) -> dict:
        return {
            "frame": {
                "timestamp_ns": timestamp_ns + 2_000_000,
                "width": 4096,
                "height": 960,
                "encoding": "jpeg",
                "data": b"camera-bytes",
                "source_id": "dat:FrontCamera",
                "metadata": {},
            },
            "delta_ns": 2_000_000,
        }


class FakeDecoded:
    points = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    range = np.asarray([3.74, 8.77], dtype=np.float32)
    intensity = np.asarray([10.0, 20.0], dtype=np.float32)
    reflectivity = np.asarray([1, 2], dtype=np.uint8)
    inferred_reflectivity = None
    slot_index = np.asarray([1, 2], dtype=np.int32)
    layer_index = np.asarray([3, 4], dtype=np.int32)
    echo_index = np.asarray([0, 4], dtype=np.int32)
    peak = np.asarray([12.0, 13.0], dtype=np.float32)
    width = None
    peak_width = None
    flags = None
    ifscan_version = 11
    structure_kind = "ifscan"
    num_slots = 1400
    num_layers = 380
    num_echoes = 5
    valid_count = 2


class FakeCodec:
    @staticmethod
    def decode_ifscan_payload(payload: bytes) -> FakeDecoded:
        assert payload == b"native-payload"
        return FakeDecoded()


def make_native() -> SimpleNamespace:
    transport = SimpleNamespace(
        NativeDatSession=FakeDatSession,
        NativeDatReader=FakeDatReader,
        NativeDatImageStreamSource=FakeImageSource,
    )
    return SimpleNamespace(
        __version__="1.6.4",
        __build_profile__="native-release",
        __build_features__=("transport",),
        transport=transport,
        codec=FakeCodec(),
    )


def test_native_status_requires_pcs_lidar_camera_and_codec() -> None:
    status = inspect_pcs_native(make_native())

    assert status.available is True
    assert status.dat_reader is True
    assert status.dat_image_source is True
    assert status.ifscan_decoder is True
    assert status.adma_source is False


def test_dat_source_uses_native_topology_without_echo_hardcoding(tmp_path) -> None:
    path = tmp_path / "trace.dat"
    path.write_bytes(b"fixture")
    source = PcsNativeDatSource(path, native=make_native())

    inspection = source.inspect()
    assert inspection.lidar_streams == ("ScaLa 3-PointCloud",)
    assert inspection.camera_streams == ("FrontCamera",)
    assert inspection.adma_streams == ("ADMA_NET_3330",)
    assert inspection.adma_native_api is False

    lidar = source.decode_lidar_frame(0)
    assert lidar.points.shape == (2, 3)
    assert lidar.timestamp_ns == 123456000
    assert lidar.metadata["num_echoes"] == 5
    assert np.array_equal(lidar.attributes["echo_index"], np.asarray([0, 4]))


def test_synchronized_sample_uses_native_camera_nearest_lookup(tmp_path) -> None:
    path = tmp_path / "trace.dat"
    path.write_bytes(b"fixture")
    source = PcsNativeDatSource(path, native=make_native())

    sample = source.build_synchronized_sample(0, require_adma=False)

    assert sample.lidar.timestamp_ns == 123456000
    assert sample.camera.timestamp_ns == 125456000
    assert sample.camera.metadata["sync_delta_ns"] == 2_000_000
    assert sample.adma is None
    assert sample.source_metadata["authority"] == "point_cloud_studio_native"


def test_adma_never_falls_back_to_local_parser(tmp_path) -> None:
    path = tmp_path / "trace.dat"
    path.write_bytes(b"fixture")
    source = PcsNativeDatSource(path, native=make_native())

    with pytest.raises(PcsNativeCapabilityError, match="NativeDatAdmaStreamSource"):
        source.build_synchronized_sample(0)
