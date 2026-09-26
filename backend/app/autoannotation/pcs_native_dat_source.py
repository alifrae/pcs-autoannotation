from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import AdmaSample, CameraFrame, LidarFrame, SynchronizedSample
from .pcs_native_lidar import (
    decode_point_cloud_payload,
    extract_ifscan10_session,
    is_ifscan10_payload,
)
from .pcs_native_runtime import PcsNativeUnavailableError, require_pcs_native

DEFAULT_ADMA_SYNC_TOLERANCE_NS = 20_000_000


@dataclass(frozen=True, slots=True)
class DatStream:
    name: str
    stream_id: int
    kind: str
    frame_count: int
    width: int | None = None
    height: int | None = None
    fps: float | None = None


@dataclass(frozen=True, slots=True)
class DatInspection:
    path: str
    streams: tuple[DatStream, ...]
    lidar_streams: tuple[str, ...]
    camera_streams: tuple[str, ...]
    adma_streams: tuple[str, ...]
    adma_native_api: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "streams": [asdict(stream) for stream in self.streams],
            "lidar_streams": list(self.lidar_streams),
            "camera_streams": list(self.camera_streams),
            "adma_streams": list(self.adma_streams),
            "adma_native_api": self.adma_native_api,
        }


class PcsNativeDatSource:
    """Thin DAT adapter over the authoritative Point Cloud Studio native wheel.

    This class intentionally knows nothing about DAT binary layout, IFSCAN packet
    structure, camera payload layout, echo topology, or ADMA encoding.
    """

    def __init__(self, path: str | Path, *, native: Any | None = None) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.native = require_pcs_native(native)
        self.transport = self.native.transport
        self.codec = self.native.codec
        self._streams = self._read_streams()
        self._ifscan10_sessions: dict[str, Any] = {}

    @property
    def streams(self) -> tuple[DatStream, ...]:
        return self._streams

    def inspect(self) -> DatInspection:
        lidar = tuple(stream.name for stream in self._streams if stream.kind == "point_cloud")
        camera = tuple(stream.name for stream in self._streams if stream.kind == "image")
        adma = tuple(stream.name for stream in self._streams if stream.kind == "adma")
        return DatInspection(
            path=str(self.path),
            streams=self._streams,
            lidar_streams=lidar,
            camera_streams=camera,
            adma_streams=adma,
            adma_native_api=hasattr(self.transport, "NativeDatAdmaStreamSource"),
        )

    def decode_lidar_frame(
        self,
        index: int,
        *,
        stream_name: str | None = None,
    ) -> LidarFrame:
        selected = stream_name or self._single_stream_name("point_cloud")
        reader = self.transport.NativeDatReader(
            str(self.path),
            selected_stream_name=selected,
            probe_only=False,
        )
        payload, timestamp_us = reader.get_payload_by_index(int(index))

        session = self._ifscan10_sessions.get(selected)
        if is_ifscan10_payload(self.native, payload) and session is None:
            first_payload, _ = reader.get_payload_by_index(0)
            session = extract_ifscan10_session(self.native, first_payload)
            self._ifscan10_sessions[selected] = session

        points, attributes, decoded_metadata = decode_point_cloud_payload(
            self.native,
            payload,
            ifscan10_session=session,
        )
        metadata = {
            "source": "point_cloud_studio_native",
            "stream_name": selected,
            "frame_index": int(index),
            **decoded_metadata,
        }
        return LidarFrame(
            timestamp_ns=int(timestamp_us) * 1000,
            points=np.ascontiguousarray(points, dtype=np.float32),
            attributes=attributes,
            metadata=metadata,
        )

    def get_camera_frame_nearest(
        self,
        timestamp_ns: int,
        *,
        stream_name: str | None = None,
    ) -> CameraFrame:
        selected = stream_name or self._single_stream_name("image")
        source = self.transport.NativeDatImageStreamSource(
            str(self.path),
            selected_stream_name=selected,
        )
        nearest = source.get_nearest_frame(int(timestamp_ns))
        if nearest is None:
            raise LookupError(
                f"No camera frame found near {timestamp_ns} ns in stream {selected!r}"
            )
        raw = nearest["frame"]
        metadata = dict(raw.get("metadata") or {})
        metadata["sync_delta_ns"] = int(nearest["delta_ns"])
        metadata["stream_name"] = selected
        return CameraFrame(
            timestamp_ns=int(raw["timestamp_ns"]),
            width=int(raw["width"]),
            height=int(raw["height"]),
            encoding=str(raw["encoding"]),
            data=bytes(raw["data"]),
            source_id=str(raw["source_id"]),
            metadata=metadata,
        )

    def get_adma_sample_at(
        self,
        timestamp_ns: int,
        *,
        stream_name: str | None = None,
        tolerance_ns: int = DEFAULT_ADMA_SYNC_TOLERANCE_NS,
    ) -> AdmaSample:
        if tolerance_ns < 0:
            raise ValueError("ADMA synchronization tolerance must be non-negative")

        selected = stream_name or self._single_stream_name("adma")
        source = self.transport.NativeDatAdmaStreamSource(
            str(self.path),
            selected_stream_name=selected,
        )
        lookup = source.get_sample_at(int(timestamp_ns), int(tolerance_ns))
        status = str(lookup.get("status") or "")
        raw_sample = lookup.get("sample")
        if status not in {"exact", "nearest"} or raw_sample is None:
            delta_ns = lookup.get("delta_ns")
            raise LookupError(
                f"ADMA synchronization failed for {timestamp_ns} ns in "
                f"stream {selected!r}: status={status!r}, delta_ns={delta_ns!r}, "
                f"tolerance_ns={tolerance_ns}"
            )

        raw = dict(raw_sample)
        metadata = {
            "stream_name": selected,
            "sample_index": int(raw["index"]),
            "sync_status": status,
            "sync_delta_ns": int(lookup["delta_ns"]),
            "sync_tolerance_ns": int(tolerance_ns),
            "chunk_timestamp_ns": int(raw["chunk_timestamp_ns"]),
            "schema": str(raw.get("schema") or ""),
            "source": "point_cloud_studio_native",
        }
        values = {
            key: value
            for key, value in raw.items()
            if key
            not in {
                "index",
                "timestamp_ns",
                "dat_timestamp_ns",
                "chunk_timestamp_ns",
            }
        }
        return AdmaSample(
            timestamp_ns=int(raw["timestamp_ns"]),
            values=values,
            metadata=metadata,
        )

    def build_synchronized_sample(
        self,
        lidar_index: int,
        *,
        lidar_stream_name: str | None = None,
        camera_stream_name: str | None = None,
        adma_stream_name: str | None = None,
        adma_tolerance_ns: int = DEFAULT_ADMA_SYNC_TOLERANCE_NS,
        require_adma: bool = True,
    ) -> SynchronizedSample:
        lidar = self.decode_lidar_frame(lidar_index, stream_name=lidar_stream_name)
        camera = self.get_camera_frame_nearest(
            lidar.timestamp_ns,
            stream_name=camera_stream_name,
        )
        adma = (
            self.get_adma_sample_at(
                lidar.timestamp_ns,
                stream_name=adma_stream_name,
                tolerance_ns=adma_tolerance_ns,
            )
            if require_adma
            else None
        )

        selected_lidar_stream = str(lidar.metadata.get("stream_name") or "")
        selected_camera_stream = str(camera.metadata.get("stream_name") or "")
        selected_adma_stream = (
            "" if adma is None else str(adma.metadata.get("stream_name") or "")
        )
        sample_seed = (
            f"{self.path.resolve()}|{lidar.timestamp_ns}|"
            f"{selected_lidar_stream}|{selected_camera_stream}|{selected_adma_stream}"
        )
        sample_id = hashlib.sha256(sample_seed.encode("utf-8")).hexdigest()[:24]
        return SynchronizedSample(
            sample_id=sample_id,
            timestamp_ns=lidar.timestamp_ns,
            lidar=lidar,
            camera=camera,
            adma=adma,
            source_metadata={
                "dat_path": str(self.path),
                "authority": "point_cloud_studio_native",
                "lidar_stream_name": selected_lidar_stream,
                "camera_stream_name": selected_camera_stream,
                "adma_stream_name": selected_adma_stream or None,
                "adma_sync_delta_ns": (
                    None if adma is None else adma.metadata.get("sync_delta_ns")
                ),
                "adma_sync_tolerance_ns": (
                    None if adma is None else adma.metadata.get("sync_tolerance_ns")
                ),
                "adma_status": "matched" if adma is not None else "not_requested",
            },
        )

    def _read_streams(self) -> tuple[DatStream, ...]:
        session_cls = getattr(self.transport, "NativeDatSession", None)
        if session_cls is not None:
            raw_streams = session_cls(str(self.path)).list_streams()
        else:
            raw_streams = self.transport.NativeDatReader(
                str(self.path), probe_only=True
            ).list_streams()

        return tuple(
            DatStream(
                name=str(item.get("name", "")),
                stream_id=int(item.get("stream_id", 0)),
                kind=str(item.get("kind", "unknown")),
                frame_count=int(item.get("frame_count", 0) or 0),
                width=_optional_int(item.get("width")),
                height=_optional_int(item.get("height")),
                fps=_optional_float(item.get("fps")),
            )
            for item in raw_streams
        )

    def _single_stream_name(self, kind: str) -> str:
        names = [stream.name for stream in self._streams if stream.kind == kind]
        if not names:
            raise PcsNativeUnavailableError(
                f"DAT contains no PCS-native stream of kind {kind!r}: {self.path}"
            )
        if len(names) > 1:
            raise ValueError(
                f"DAT contains multiple {kind} streams; select one explicitly: {names}"
            )
        return names[0]


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
