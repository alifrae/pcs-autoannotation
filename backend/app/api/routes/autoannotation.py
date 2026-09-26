from __future__ import annotations

import io
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ...autoannotation.association import load_pcs_calibration
from ...autoannotation.pcs_native_dat_source import PcsNativeDatSource
from ...autoannotation.pcs_native_runtime import (
    PcsNativeUnavailableError,
    inspect_pcs_native,
)
from ...autoannotation.pcs_scene_objects import (
    load_pcs_scene_object_document,
    write_pcs_scene_object_document,
)
from ...autoannotation.pipeline import run_autoannotation_sample
from ...autoannotation.providers.camera_provider import camera_frame_to_pil
from ...autoannotation.providers.factory import (
    create_camera_provider,
    create_innov3_provider,
    inspect_innov3_configuration,
)
from ...autoannotation.qualification import qualify_review_document
from ...core.exceptions import AppError
from ...services.locate_anything import get_model_status
from ...services.sam2_service import get_sam2_status

router = APIRouter(prefix="/api/v1/autoannotation", tags=["autoannotation"])


class DatPathRequest(BaseModel):
    path: str = Field(..., min_length=1)


class DatSampleSummaryRequest(DatPathRequest):
    lidar_index: int = Field(default=0, ge=0)
    lidar_stream_name: str | None = None
    camera_stream_name: str | None = None
    adma_stream_name: str | None = None
    require_adma: bool = True


class DatInnov3Request(DatPathRequest):
    lidar_index: int = Field(default=0, ge=0)
    lidar_stream_name: str | None = None
    camera_stream_name: str | None = None
    adma_stream_name: str | None = None


class DatCameraProposalRequest(DatInnov3Request):
    categories: list[str] = Field(default_factory=lambda: ["car", "truck"], min_length=1)
    use_sam2: bool = True
    sam2_score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class DatAutoAnnotationRequest(DatInnov3Request):
    calibration_path: str = Field(..., min_length=1)
    recording_key: str = Field(..., min_length=1)
    recording_sha256: str = Field(..., min_length=64, max_length=64)
    min_iou: float = Field(default=0.1, ge=0.0, le=1.0)
    run_id: str | None = None
    output_path: str | None = None


class ReviewQualificationRequest(BaseModel):
    path: str = Field(..., min_length=1)
    human_seconds: float | None = Field(default=None, ge=0.0)


@router.get("/pcs-native/status")
def pcs_native_status() -> dict:
    return inspect_pcs_native().as_dict()


@router.get("/providers/status")
def provider_status() -> dict:
    return {
        "innov3": inspect_innov3_configuration().as_dict(),
        "camera_baseline": {
            "provider": "LocateAnything-3B+SAM2",
            "companion_adapter": True,
            "vlm": get_model_status(),
            "sam2": get_sam2_status(),
        },
    }


@router.post("/dat/inspect")
def inspect_dat(request: DatPathRequest) -> dict:
    try:
        source = PcsNativeDatSource(Path(request.path))
        return source.inspect().as_dict()
    except FileNotFoundError as exc:
        raise AppError(f"DAT file not found: {request.path}", 404) from exc
    except PcsNativeUnavailableError as exc:
        raise AppError(str(exc), 503) from exc


@router.post("/dat/sample-summary")
def dat_sample_summary(request: DatSampleSummaryRequest) -> dict:
    try:
        source = PcsNativeDatSource(Path(request.path))
        sample = source.build_synchronized_sample(
            request.lidar_index,
            lidar_stream_name=request.lidar_stream_name,
            camera_stream_name=request.camera_stream_name,
            adma_stream_name=request.adma_stream_name,
            require_adma=request.require_adma,
        )
    except FileNotFoundError as exc:
        raise AppError(f"DAT file not found: {request.path}", 404) from exc
    except PcsNativeUnavailableError as exc:
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
        "adma": (
            None
            if sample.adma is None
            else {
                "timestamp_ns": sample.adma.timestamp_ns,
                "sync_status": sample.adma.metadata.get("sync_status"),
                "sync_delta_ns": sample.adma.metadata.get("sync_delta_ns"),
                "sync_tolerance_ns": sample.adma.metadata.get("sync_tolerance_ns"),
                "sample_index": sample.adma.metadata.get("sample_index"),
                "stream_name": sample.adma.metadata.get("stream_name"),
                "values": dict(sample.adma.values),
            }
        ),
        "source_metadata": dict(sample.source_metadata),
    }


@router.post("/dat/innov3-proposals")
def dat_innov3_proposals(request: DatInnov3Request) -> dict:
    """Run Innov3 on one complete PCS-native DAT sample."""

    try:
        source = PcsNativeDatSource(Path(request.path))
        sample = source.build_synchronized_sample(
            request.lidar_index,
            lidar_stream_name=request.lidar_stream_name,
            camera_stream_name=request.camera_stream_name,
            adma_stream_name=request.adma_stream_name,
            require_adma=True,
        )
        provider = create_innov3_provider()
        proposals = provider.infer(sample)
    except FileNotFoundError as exc:
        raise AppError(str(exc), 404) from exc
    except PcsNativeUnavailableError as exc:
        raise AppError(str(exc), 503) from exc
    except RuntimeError as exc:
        raise AppError(str(exc), 503) from exc
    except (LookupError, ValueError, IndexError) as exc:
        raise AppError(str(exc), 422) from exc

    return {
        "sample_id": sample.sample_id,
        "timestamp_ns": sample.timestamp_ns,
        "provider": provider.identity.provider_id,
        "proposals": [
            {
                "proposal_id": proposal.proposal_id,
                "class_name": proposal.class_name,
                "bbox_3d": (
                    None
                    if proposal.bbox_3d is None
                    else {
                        "center_x": proposal.bbox_3d.center_x,
                        "center_y": proposal.bbox_3d.center_y,
                        "center_z": proposal.bbox_3d.center_z,
                        "length": proposal.bbox_3d.length,
                        "width": proposal.bbox_3d.width,
                        "height": proposal.bbox_3d.height,
                        "yaw_rad": proposal.bbox_3d.yaw_rad,
                    }
                ),
                "confidence": (proposal.evidence[0].confidence if proposal.evidence else None),
                "evidence": [
                    {
                        "provider_id": evidence.provider.provider_id,
                        "modality": evidence.modality.value,
                        "confidence": evidence.confidence,
                        "metadata": dict(evidence.metadata),
                    }
                    for evidence in proposal.evidence
                ],
            }
            for proposal in proposals
        ],
    }


@router.post("/dat/camera-frame")
def dat_camera_frame(request: DatInnov3Request) -> Response:
    """Return the synchronized PCS-native camera frame as browser-safe PNG."""

    try:
        source = PcsNativeDatSource(Path(request.path))
        lidar = source.decode_lidar_frame(
            request.lidar_index,
            stream_name=request.lidar_stream_name,
        )
        camera = source.get_camera_frame_nearest(
            lidar.timestamp_ns,
            stream_name=request.camera_stream_name,
        )
        image = camera_frame_to_pil(camera)
        try:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
        finally:
            image.close()
    except FileNotFoundError as exc:
        raise AppError(str(exc), 404) from exc
    except PcsNativeUnavailableError as exc:
        raise AppError(str(exc), 503) from exc
    except (LookupError, ValueError, IndexError) as exc:
        raise AppError(str(exc), 422) from exc

    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={
            "X-PCS-Camera-Timestamp-Ns": str(camera.timestamp_ns),
            "X-PCS-Camera-Source": camera.source_id,
        },
    )


@router.post("/dat/camera-proposals")
def dat_camera_proposals(request: DatCameraProposalRequest) -> dict:
    """Run the configured LocateAnything/SAM2 baseline on one PCS-native camera frame."""

    try:
        source = PcsNativeDatSource(Path(request.path))
        sample = source.build_synchronized_sample(
            request.lidar_index,
            lidar_stream_name=request.lidar_stream_name,
            camera_stream_name=request.camera_stream_name,
            adma_stream_name=request.adma_stream_name,
            require_adma=True,
        )
        provider = create_camera_provider(
            categories=request.categories,
            use_sam2=request.use_sam2,
            sam2_score_threshold=request.sam2_score_threshold,
        )
        proposals = provider.infer(sample)
    except FileNotFoundError as exc:
        raise AppError(str(exc), 404) from exc
    except PcsNativeUnavailableError as exc:
        raise AppError(str(exc), 503) from exc
    except RuntimeError as exc:
        raise AppError(str(exc), 503) from exc
    except (LookupError, ValueError, IndexError) as exc:
        raise AppError(str(exc), 422) from exc

    return {
        "sample_id": sample.sample_id,
        "timestamp_ns": sample.timestamp_ns,
        "camera_timestamp_ns": sample.camera.timestamp_ns,
        "provider": provider.identity.provider_id,
        "proposals": [
            {
                "proposal_id": proposal.proposal_id,
                "class_name": proposal.class_name,
                "bbox_2d": (
                    None
                    if proposal.bbox_2d is None
                    else {
                        "x1": proposal.bbox_2d.x1,
                        "y1": proposal.bbox_2d.y1,
                        "x2": proposal.bbox_2d.x2,
                        "y2": proposal.bbox_2d.y2,
                    }
                ),
                "mask_polygon": proposal.mask_polygon,
                "confidence": (proposal.evidence[0].confidence if proposal.evidence else None),
                "evidence": [
                    {
                        "provider_id": evidence.provider.provider_id,
                        "modality": evidence.modality.value,
                        "confidence": evidence.confidence,
                        "metadata": dict(evidence.metadata),
                    }
                    for evidence in proposal.evidence
                ],
            }
            for proposal in proposals
        ],
    }


@router.post("/dat/run")
def dat_autoannotation_run(request: DatAutoAnnotationRequest) -> dict:
    """Run the complete v1 proposal, association and PCS-review export path."""

    try:
        source = PcsNativeDatSource(Path(request.path))
        sample = source.build_synchronized_sample(
            request.lidar_index,
            lidar_stream_name=request.lidar_stream_name,
            camera_stream_name=request.camera_stream_name,
            adma_stream_name=request.adma_stream_name,
            require_adma=True,
        )
        calibration = load_pcs_calibration(Path(request.calibration_path))
        result = run_autoannotation_sample(
            sample,
            lidar_provider=create_innov3_provider(),
            camera_provider=create_camera_provider(),
            calibration=calibration,
            recording_key=request.recording_key,
            recording_sha256=request.recording_sha256,
            min_iou=request.min_iou,
            run_id=request.run_id,
        )
        saved_path = (
            None
            if request.output_path is None
            else str(
                write_pcs_scene_object_document(
                    Path(request.output_path),
                    result.scene_object_document,
                )
            )
        )
    except FileNotFoundError as exc:
        raise AppError(str(exc), 404) from exc
    except PcsNativeUnavailableError as exc:
        raise AppError(str(exc), 503) from exc
    except RuntimeError as exc:
        raise AppError(str(exc), 503) from exc
    except (LookupError, ValueError, IndexError) as exc:
        raise AppError(str(exc), 422) from exc

    return {
        "sample_id": sample.sample_id,
        "timestamp_ns": sample.timestamp_ns,
        "lidar_proposal_count": len(result.lidar_proposals),
        "camera_proposal_count": len(result.camera_proposals),
        "matched_count": len(result.association.matches),
        "unmatched_lidar_count": len(result.association.unmatched_lidar),
        "unmatched_camera_count": len(result.association.unmatched_camera),
        "matches": [
            {
                "lidar_proposal_id": match.lidar_proposal_id,
                "camera_proposal_id": match.camera_proposal_id,
                "association_score": match.score,
                "fused_proposal_id": match.fused.proposal_id,
            }
            for match in result.association.matches
        ],
        "scene_object_document": result.scene_object_document,
        "scene_object_path": saved_path,
        "calibration_source": calibration.source,
    }


@router.post("/review/qualify")
def qualify_pcs_review(request: ReviewQualificationRequest) -> dict:
    """Summarize a PCS-reviewed scene-object document without altering it."""

    try:
        document = load_pcs_scene_object_document(Path(request.path))
        report = qualify_review_document(
            document,
            human_seconds=request.human_seconds,
        )
    except FileNotFoundError as exc:
        raise AppError(str(exc), 404) from exc
    except (ValueError, TypeError) as exc:
        raise AppError(str(exc), 422) from exc
    return report.as_dict()
