from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ...autoannotation.pcs_native_dat_source import (
    PcsNativeCapabilityError,
    PcsNativeDatSource,
)
from ...autoannotation.pcs_native_runtime import (
    PcsNativeUnavailable,
    inspect_pcs_native,
)
from ...core.exceptions import AppError

router = APIRouter(prefix="/api/v1/autoannotation", tags=["autoannotation"])


class DatPathRequest(BaseModel):
    path: str = Field(..., min_length=1)


class DatSampleSummaryRequest(DatPathRequest):
    lidar_index: int = Field(default=0, ge=0)
    lidar_stream_name: str | None = None
    camera_stream_name: str | None = None
    require_adma: bool = True


@router.get("/pcs-native/status")
def pcs_native_status() -> dict:
    return inspect_pcs_native().as_dict()


@router.post("/dat/inspect")
def inspect_dat(request: DatPathRequest) -> dict:
    try:
        source = PcsNativeDatSource(Path(request.path))
        return source.inspect().as_dict()
    except FileNotFoundError as exc:
        raise AppError(f"DAT file not found: {request.path}", 404) from exc
    except PcsNativeUnavailable as exc:
        raise AppError(str(exc), 503) from exc


@router.post("/dat/sample-summary")
def dat_sample_summary(request: DatSampleSummaryRequest) -> dict:
    try:
        source = PcsNativeDatSource(Path(request.path))
        sample = source.build_synchronized_sample(
            request.lidar_index,
            lidar_stream_name=request.lidar_stream_name,
            camera_stream_name=request.camera_stream_name,
            require_adma=request.require_adma,
        )
    except FileNotFoundError as exc:
        raise AppError(f"DAT file not found: {request.path}", 404) from exc
    except (PcsNativeUnavailable, PcsNativeCapabilityError) as exc:
        raise AppError(str(exc), 503) from exc
    except (LookupError, ValueError, IndexError) as exc:
        raise AppError(str(exc), 422) from exc

    return {
        "sample_id": sample.sample_id,
        "timestamp_ns": sample.timestamp_ns,
        "lidar": {
            "point_count": int(sample.lidar.points.shape[0]),
            "attributes": sorted(sample.lidar.attributes),
            "metadata": dict(sample.lidar.metadata),
        },
        "camera": {
            "timestamp_ns": sample.camera.timestamp_ns,
            "width": sample.camera.width,
            "height": sample.camera.height,
            "encoding": sample.camera.encoding,
            "source_id": sample.camera.source_id,
            "sync_delta_ns": sample.camera.metadata.get("sync_delta_ns"),
        },
        "adma": None if sample.adma is None else dict(sample.adma.values),
        "source_metadata": dict(sample.source_metadata),
    }
