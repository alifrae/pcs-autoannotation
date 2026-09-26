from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import (
    BoundingBox2D,
    BoundingBox3D,
    ObjectProposal,
    ProviderEvidence,
    ProviderIdentity,
    SensorModality,
)


@dataclass(frozen=True, slots=True)
class CameraLidarCalibration:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    rotation_row_major: tuple[float, ...]
    translation_m: tuple[float, float, float]
    distortion_model: str = "none"
    distortion_coefficients: tuple[float, ...] = ()
    source: str = ""

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("calibration image dimensions must be positive")
        for name, value in (
            ("fx", self.fx),
            ("fy", self.fy),
            ("cx", self.cx),
            ("cy", self.cy),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"calibration {name} must be finite")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("calibration focal lengths must be positive")
        if len(self.rotation_row_major) != 9:
            raise ValueError("rotation_row_major must contain exactly 9 values")
        if len(self.translation_m) != 3:
            raise ValueError("translation_m must contain exactly 3 values")
        values = (*self.rotation_row_major, *self.translation_m)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("calibration transform values must be finite")
        if self.distortion_model.casefold() != "none":
            raise ValueError("v1 association requires rectified/no-distortion PCS calibration")
        if self.distortion_coefficients:
            raise ValueError("v1 association requires empty distortion coefficients")

    @property
    def rotation(self) -> np.ndarray:
        return np.asarray(self.rotation_row_major, dtype=np.float64).reshape(3, 3)

    @property
    def translation(self) -> np.ndarray:
        return np.asarray(self.translation_m, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class AssociationMatch:
    lidar_proposal_id: str
    camera_proposal_id: str
    score: float
    fused: ObjectProposal


@dataclass(frozen=True, slots=True)
class AssociationResult:
    matches: tuple[AssociationMatch, ...]
    unmatched_lidar: tuple[ObjectProposal, ...]
    unmatched_camera: tuple[ObjectProposal, ...]

    @property
    def fused_proposals(self) -> tuple[ObjectProposal, ...]:
        return tuple(match.fused for match in self.matches)


def load_pcs_calibration(path: str | Path) -> CameraLidarCalibration:
    """Load a verified PCS camera/LiDAR calibration artifact.

    Supported PCS-owned shapes:
    - calibration-input sidecar: intrinsics + lidar_to_camera
    - OverlayValidationSession: intrinsics + Euler extrinsics

    Missing/unverified calibration fails closed. No bootstrap/default extrinsics
    are accepted for association.
    """

    calibration_path = Path(path).expanduser().resolve()
    if not calibration_path.is_file():
        raise FileNotFoundError(calibration_path)

    text = calibration_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML PCS calibration requires PyYAML") from exc
        payload = yaml.safe_load(text)

    if not isinstance(payload, Mapping):
        raise ValueError("PCS calibration artifact must contain an object")

    if "lidar_to_camera" in payload:
        return _from_calibration_input(payload, source=str(calibration_path))
    if "extrinsics" in payload and "intrinsics" in payload:
        return _from_overlay_session(payload, source=str(calibration_path))
    raise ValueError("unsupported PCS calibration artifact shape")


def associate_proposals(
    lidar_proposals: Sequence[ObjectProposal],
    camera_proposals: Sequence[ObjectProposal],
    calibration: CameraLidarCalibration,
    *,
    min_iou: float = 0.1,
) -> AssociationResult:
    """Associate 3D LiDAR proposals with camera proposals deterministically."""

    if not 0.0 <= float(min_iou) <= 1.0:
        raise ValueError("min_iou must be within [0, 1]")

    candidates: list[tuple[float, int, int, BoundingBox2D]] = []
    for lidar_index, lidar in enumerate(lidar_proposals):
        if lidar.bbox_3d is None:
            continue
        projected = project_box_3d(lidar.bbox_3d, calibration)
        if projected is None:
            continue
        for camera_index, camera in enumerate(camera_proposals):
            if camera.bbox_2d is None:
                continue
            if not _classes_compatible(lidar.class_name, camera.class_name):
                continue
            score = bbox_iou(projected, camera.bbox_2d)
            if score >= min_iou:
                candidates.append((score, lidar_index, camera_index, projected))

    candidates.sort(
        key=lambda item: (
            -item[0],
            lidar_proposals[item[1]].proposal_id,
            camera_proposals[item[2]].proposal_id,
        )
    )
    used_lidar: set[int] = set()
    used_camera: set[int] = set()
    matches: list[AssociationMatch] = []
    for score, lidar_index, camera_index, projected in candidates:
        if lidar_index in used_lidar or camera_index in used_camera:
            continue
        lidar = lidar_proposals[lidar_index]
        camera = camera_proposals[camera_index]
        fused = _fuse(lidar, camera, score=score, projected_box=projected)
        matches.append(
            AssociationMatch(
                lidar_proposal_id=lidar.proposal_id,
                camera_proposal_id=camera.proposal_id,
                score=float(score),
                fused=fused,
            )
        )
        used_lidar.add(lidar_index)
        used_camera.add(camera_index)

    return AssociationResult(
        matches=tuple(matches),
        unmatched_lidar=tuple(
            proposal for index, proposal in enumerate(lidar_proposals) if index not in used_lidar
        ),
        unmatched_camera=tuple(
            proposal for index, proposal in enumerate(camera_proposals) if index not in used_camera
        ),
    )


def project_box_3d(
    box: BoundingBox3D,
    calibration: CameraLidarCalibration,
) -> BoundingBox2D | None:
    corners = _box_corners(box)
    camera_xyz = corners @ calibration.rotation.T + calibration.translation
    positive = camera_xyz[:, 2] > 0.0
    if not np.any(positive):
        return None

    visible = camera_xyz[positive]
    u = calibration.fx * visible[:, 0] / visible[:, 2] + calibration.cx
    v = calibration.fy * visible[:, 1] / visible[:, 2] + calibration.cy
    finite = np.isfinite(u) & np.isfinite(v)
    if not np.any(finite):
        return None

    u = u[finite]
    v = v[finite]
    x1 = max(0.0, float(np.min(u)))
    y1 = max(0.0, float(np.min(v)))
    x2 = min(float(calibration.width), float(np.max(u)))
    y2 = min(float(calibration.height), float(np.max(v)))
    if x2 <= x1 or y2 <= y1:
        return None
    return BoundingBox2D(x1=x1, y1=y1, x2=x2, y2=y2)


def bbox_iou(left: BoundingBox2D, right: BoundingBox2D) -> float:
    ix1 = max(left.x1, right.x1)
    iy1 = max(left.y1, right.y1)
    ix2 = min(left.x2, right.x2)
    iy2 = min(left.y2, right.y2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, left.x2 - left.x1) * max(0.0, left.y2 - left.y1)
    right_area = max(0.0, right.x2 - right.x1) * max(0.0, right.y2 - right.y1)
    union = left_area + right_area - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def _from_calibration_input(
    payload: Mapping[str, Any],
    *,
    source: str,
) -> CameraLidarCalibration:
    camera = _mapping(payload, "camera")
    intrinsics = _mapping(payload, "intrinsics")
    distortion = _mapping(payload, "distortion")
    transform = _mapping(payload, "lidar_to_camera")
    image_state = _mapping(camera, "image_state")

    for name, section in (
        ("intrinsics", intrinsics),
        ("distortion", distortion),
        ("lidar_to_camera", transform),
        ("image_state", image_state),
    ):
        if section.get("status") != "verified":
            raise ValueError(f"PCS calibration {name} is not verified")
    if image_state.get("rectified") is not True:
        raise ValueError("PCS calibration image is not declared rectified")

    rotation = np.asarray(transform.get("rotation_row_major"), dtype=np.float64)
    translation = np.asarray(transform.get("translation_m"), dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("PCS lidar_to_camera rotation must be 3x3")
    if translation.shape != (3,):
        raise ValueError("PCS lidar_to_camera translation must have length 3")

    coefficients = tuple(float(value) for value in distortion.get("coefficients", ()))
    return CameraLidarCalibration(
        width=int(camera["width"]),
        height=int(camera["height"]),
        fx=float(intrinsics["fx_px"]),
        fy=float(intrinsics["fy_px"]),
        cx=float(intrinsics["cx_px"]),
        cy=float(intrinsics["cy_px"]),
        rotation_row_major=tuple(float(value) for value in rotation.reshape(-1)),
        translation_m=tuple(float(value) for value in translation),
        distortion_model=str(distortion.get("model") or "none"),
        distortion_coefficients=coefficients,
        source=source,
    )


def _from_overlay_session(
    payload: Mapping[str, Any],
    *,
    source: str,
) -> CameraLidarCalibration:
    intrinsics = _mapping(payload, "intrinsics")
    extrinsics = _mapping(payload, "extrinsics")
    distortion_model = str(intrinsics.get("distortion_model") or "none")
    coefficients = tuple(float(value) for value in intrinsics.get("distortion_coefficients", ()))
    rotation = _euler_rotation(
        roll=float(extrinsics["roll_rad"]),
        pitch=float(extrinsics["pitch_rad"]),
        yaw=float(extrinsics["yaw_rad"]),
    )
    return CameraLidarCalibration(
        width=int(intrinsics["width"]),
        height=int(intrinsics["height"]),
        fx=float(intrinsics["fx"]),
        fy=float(intrinsics["fy"]),
        cx=float(intrinsics["cx"]),
        cy=float(intrinsics["cy"]),
        rotation_row_major=tuple(float(value) for value in rotation.reshape(-1)),
        translation_m=(
            float(extrinsics["tx_m"]),
            float(extrinsics["ty_m"]),
            float(extrinsics["tz_m"]),
        ),
        distortion_model=distortion_model,
        distortion_coefficients=coefficients,
        source=source,
    )


def _euler_rotation(*, roll: float, pitch: float, yaw: float) -> np.ndarray:
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _box_corners(box: BoundingBox3D) -> np.ndarray:
    half_length = box.length / 2.0
    half_width = box.width / 2.0
    half_height = box.height / 2.0
    local = np.asarray(
        [
            [x, y, z]
            for x in (-half_length, half_length)
            for y in (-half_width, half_width)
            for z in (-half_height, half_height)
        ],
        dtype=np.float64,
    )
    cos_yaw = math.cos(box.yaw_rad)
    sin_yaw = math.sin(box.yaw_rad)
    rotation = np.asarray(
        [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    center = np.asarray(
        [box.center_x, box.center_y, box.center_z],
        dtype=np.float64,
    )
    return local @ rotation.T + center


def _fuse(
    lidar: ObjectProposal,
    camera: ObjectProposal,
    *,
    score: float,
    projected_box: BoundingBox2D,
) -> ObjectProposal:
    confidences = [
        float(evidence.confidence)
        for evidence in (*lidar.evidence, *camera.evidence)
        if evidence.confidence is not None
    ]
    fused_confidence = float(np.mean(confidences)) if confidences else None
    seed = f"{lidar.sample_id}|{lidar.proposal_id}|{camera.proposal_id}|{score:.12f}"
    proposal_id = hashlib.sha256(seed.encode()).hexdigest()[:24]
    association_evidence = ProviderEvidence(
        provider=ProviderIdentity(
            provider_id="camera-lidar-association",
            model_name="deterministic-projected-box-iou",
        ),
        modality=SensorModality.LIDAR,
        confidence=score,
        metadata={
            "kind": "camera_lidar_iou",
            "projected_bbox": {
                "x1": projected_box.x1,
                "y1": projected_box.y1,
                "x2": projected_box.x2,
                "y2": projected_box.y2,
            },
        },
    )
    return ObjectProposal(
        proposal_id=proposal_id,
        sample_id=lidar.sample_id,
        class_name=lidar.class_name,
        bbox_2d=camera.bbox_2d,
        bbox_3d=lidar.bbox_3d,
        mask_polygon=camera.mask_polygon,
        evidence=(*lidar.evidence, *camera.evidence, association_evidence),
        association_score=float(score),
        fused_confidence=fused_confidence,
    )


def _classes_compatible(left: str, right: str) -> bool:
    return _normalize_class(left) == _normalize_class(right)


def _normalize_class(value: str) -> str:
    normalized = "".join(character for character in value.casefold() if character.isalnum())
    if normalized.endswith("s") and len(normalized) > 3:
        normalized = normalized[:-1]
    return normalized


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"PCS calibration {key} must be an object")
    return value
