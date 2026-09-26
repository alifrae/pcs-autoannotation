from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .contracts import ObjectProposal, SynchronizedSample

SCHEMA = "pcs.scene_objects"
SCHEMA_VERSION = 1
BOX3D_CONVENTION = "pcs_lidar_rh_x_forward_y_left_z_up_lwh_yaw_ccw_z"


def build_pcs_scene_object_document(
    sample: SynchronizedSample,
    proposals: Sequence[ObjectProposal],
    *,
    recording_key: str,
    recording_sha256: str,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Adapt companion 3D hypotheses to the producer-neutral PCS v1 contract."""

    recording_key = _trimmed(recording_key, "recording_key")
    recording_sha256 = _sha256(recording_sha256, "recording_sha256")
    source_key = _trimmed(
        sample.source_metadata.get("lidar_stream_name"),
        "lidar_stream_name",
    )
    frame_index = sample.lidar.metadata.get("frame_index")
    if isinstance(frame_index, bool) or not isinstance(frame_index, int):
        raise ValueError("PCS scene-object export requires genuine LiDAR frame_index")

    hypotheses = [
        _hypothesis(proposal, run_id=run_id)
        for proposal in proposals
        if proposal.bbox_3d is not None
    ]
    object_ids = [item["object_id"] for item in hypotheses]
    if len(set(object_ids)) != len(object_ids):
        raise ValueError("PCS scene-object object IDs must be unique")

    document_seed = (
        f"{recording_key}|{recording_sha256}|{source_key}|{frame_index}|"
        f"{'|'.join(sorted(object_ids))}"
    )
    document_id = "pcs-autoannotation-" + hashlib.sha256(document_seed.encode()).hexdigest()[:24]

    document = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "document_id": document_id,
        "provenance": {
            "producer": "pcs-autoannotation",
            "sample_id": sample.sample_id,
        },
        "frames": [
            {
                "sample_reference": {
                    "recording_key": recording_key,
                    "recording_sha256": recording_sha256,
                    "source_key": source_key,
                    "sample_key": {
                        "kind": "ordinal",
                        "value": frame_index,
                    },
                },
                "hypotheses": hypotheses,
                "reviews": [],
                "manual_annotations": [],
            }
        ],
    }
    validate_pcs_scene_object_document(document)
    return document


def write_pcs_scene_object_document(
    path: str | Path,
    document: Mapping[str, Any],
) -> Path:
    validate_pcs_scene_object_document(document)
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(canonical_scene_object_json(document) + "\n", encoding="utf-8")
    return target


def canonical_scene_object_json(document: Mapping[str, Any]) -> str:
    validate_pcs_scene_object_document(document)
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def validate_pcs_scene_object_document(document: Mapping[str, Any]) -> None:
    if document.get("schema") != SCHEMA:
        raise ValueError("unsupported PCS scene-object schema")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported PCS scene-object schema_version")
    _trimmed(document.get("document_id"), "document_id")
    if not isinstance(document.get("provenance"), Mapping):
        raise ValueError("PCS scene-object provenance must be an object")

    frames = document.get("frames")
    if not isinstance(frames, list):
        raise ValueError("PCS scene-object frames must be an array")

    object_ids: set[str] = set()
    sample_keys: set[tuple[str, str, str, int]] = set()
    for frame in frames:
        if not isinstance(frame, Mapping):
            raise ValueError("PCS scene-object frame must be an object")
        reference = frame.get("sample_reference")
        if not isinstance(reference, Mapping):
            raise ValueError("PCS scene-object sample_reference must be an object")
        recording_key = _trimmed(reference.get("recording_key"), "recording_key")
        recording_sha = _sha256(reference.get("recording_sha256"), "recording_sha256")
        source_key = _trimmed(reference.get("source_key"), "source_key")
        sample_key = reference.get("sample_key")
        if not isinstance(sample_key, Mapping) or sample_key.get("kind") != "ordinal":
            raise ValueError("PCS scene-object sample_key must be ordinal")
        ordinal = sample_key.get("value")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError("PCS scene-object ordinal sample key must be non-negative")
        portable_key = (recording_key, recording_sha, source_key, ordinal)
        if portable_key in sample_keys:
            raise ValueError("duplicate PCS scene-object sample_reference")
        sample_keys.add(portable_key)

        hypotheses = frame.get("hypotheses")
        reviews = frame.get("reviews")
        manual = frame.get("manual_annotations")
        if not isinstance(hypotheses, list):
            raise ValueError("PCS scene-object hypotheses must be an array")
        if not isinstance(reviews, list):
            raise ValueError("PCS scene-object reviews must be an array")
        if not isinstance(manual, list):
            raise ValueError("PCS scene-object manual_annotations must be an array")

        frame_hypotheses: set[str] = set()
        for hypothesis in hypotheses:
            object_id = _validate_hypothesis(hypothesis)
            if object_id in object_ids:
                raise ValueError("PCS scene-object IDs must be globally unique")
            object_ids.add(object_id)
            frame_hypotheses.add(object_id)

        reviewed: set[str] = set()
        for review in reviews:
            if not isinstance(review, Mapping):
                raise ValueError("PCS scene-object review must be an object")
            hypothesis_id = _trimmed(review.get("hypothesis_id"), "hypothesis_id")
            if hypothesis_id not in frame_hypotheses:
                raise ValueError("PCS review references unknown hypothesis")
            if hypothesis_id in reviewed:
                raise ValueError("PCS hypothesis may have at most one review")
            reviewed.add(hypothesis_id)
            decision = review.get("decision")
            if decision not in {"accepted", "corrected", "rejected"}:
                raise ValueError("unsupported PCS review decision")
            _trimmed(review.get("reviewer"), "reviewer")
            corrected_class = review.get("corrected_class")
            corrected_box = review.get("corrected_box")
            has_correction = corrected_class is not None or corrected_box is not None
            if decision == "corrected" and not has_correction:
                raise ValueError("corrected review requires a correction")
            if decision != "corrected" and has_correction:
                raise ValueError("only corrected reviews may carry corrections")
            if corrected_class is not None:
                _trimmed(corrected_class, "corrected_class")
            if corrected_box is not None:
                _validate_box(corrected_box)

        for annotation in manual:
            if not isinstance(annotation, Mapping):
                raise ValueError("PCS manual annotation must be an object")
            object_id = _trimmed(annotation.get("object_id"), "object_id")
            if object_id in object_ids:
                raise ValueError("PCS scene-object IDs must be globally unique")
            object_ids.add(object_id)
            _trimmed(annotation.get("semantic_class"), "semantic_class")
            _validate_box(annotation.get("box"))
            _trimmed(annotation.get("annotator"), "annotator")


def review_progress(document: Mapping[str, Any]) -> dict[str, int | float | None]:
    validate_pcs_scene_object_document(document)
    hypotheses = accepted = corrected = rejected = manual = 0
    for frame in document["frames"]:
        hypotheses += len(frame["hypotheses"])
        manual += len(frame["manual_annotations"])
        for review in frame["reviews"]:
            decision = review["decision"]
            accepted += decision == "accepted"
            corrected += decision == "corrected"
            rejected += decision == "rejected"

    reviewed = accepted + corrected + rejected
    pending = hypotheses - reviewed
    denominator = reviewed if reviewed else 0
    return {
        "hypothesis_count": hypotheses,
        "reviewed_count": reviewed,
        "pending_count": pending,
        "accepted_count": accepted,
        "corrected_count": corrected,
        "rejected_count": rejected,
        "manual_annotation_count": manual,
        "accept_without_edit_rate": (None if not denominator else accepted / denominator),
        "correction_rate": None if not denominator else corrected / denominator,
        "rejection_rate": None if not denominator else rejected / denominator,
    }


def load_pcs_scene_object_document(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("PCS scene-object document must contain an object")
    validate_pcs_scene_object_document(payload)
    return payload


def _hypothesis(
    proposal: ObjectProposal,
    *,
    run_id: str | None,
) -> dict[str, Any]:
    box = proposal.bbox_3d
    if box is None:
        raise ValueError("PCS hypotheses require a 3D box")
    providers = sorted({item.provider.provider_id for item in proposal.evidence})
    model_versions = sorted(
        {item.provider.model_version for item in proposal.evidence if item.provider.model_version}
    )
    model_shas = sorted(
        {item.provider.model_sha256 for item in proposal.evidence if item.provider.model_sha256}
    )
    details = {
        "providers": ",".join(providers) or "unknown",
        "sample_id": proposal.sample_id,
    }
    if proposal.association_score is not None:
        details["association_score"] = f"{proposal.association_score:.12g}"
    if model_shas:
        details["model_sha256"] = ",".join(model_shas)

    confidence = proposal.fused_confidence
    if confidence is None:
        available = [
            float(item.confidence) for item in proposal.evidence if item.confidence is not None
        ]
        confidence = sum(available) / len(available) if available else None
    if confidence is not None:
        confidence = min(1.0, max(0.0, float(confidence)))

    return {
        "object_id": proposal.proposal_id,
        "semantic_class": proposal.class_name,
        "box": {
            "coordinate_frame_id": "lidar",
            "center_xyz_m": [
                float(box.center_x),
                float(box.center_y),
                float(box.center_z),
            ],
            "size_lwh_m": [
                float(box.length),
                float(box.width),
                float(box.height),
            ],
            "yaw_rad": float(box.yaw_rad),
            "convention": BOX3D_CONVENTION,
        },
        "confidence": confidence,
        "provenance": {
            "producer": "pcs-autoannotation",
            "producer_version": model_versions[0] if len(model_versions) == 1 else None,
            "run_id": run_id,
            "details": details,
        },
        "attributes": {
            "association": ("matched" if proposal.association_score is not None else "lidar_only")
        },
    }


def _validate_hypothesis(value: Any) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("PCS scene-object hypothesis must be an object")
    object_id = _trimmed(value.get("object_id"), "object_id")
    _trimmed(value.get("semantic_class"), "semantic_class")
    _validate_box(value.get("box"))
    confidence = value.get("confidence")
    if confidence is not None:
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("PCS hypothesis confidence must be within [0, 1]")
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("PCS hypothesis provenance must be an object")
    _trimmed(provenance.get("producer"), "producer")
    if not isinstance(provenance.get("details", {}), Mapping):
        raise ValueError("PCS hypothesis provenance details must be an object")
    attributes = value.get("attributes", {})
    if not isinstance(attributes, Mapping):
        raise ValueError("PCS hypothesis attributes must be an object")
    return object_id


def _validate_box(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("PCS Box3D must be an object")
    _trimmed(value.get("coordinate_frame_id"), "coordinate_frame_id")
    center = value.get("center_xyz_m")
    size = value.get("size_lwh_m")
    if not isinstance(center, list) or len(center) != 3:
        raise ValueError("PCS Box3D center_xyz_m must have length 3")
    if not isinstance(size, list) or len(size) != 3:
        raise ValueError("PCS Box3D size_lwh_m must have length 3")
    values = [float(item) for item in (*center, *size, value.get("yaw_rad", ()))]
    if not all(math.isfinite(item) for item in values):
        raise ValueError("PCS Box3D values must be finite")
    if any(float(item) <= 0.0 for item in size):
        raise ValueError("PCS Box3D sizes must be positive")
    if value.get("convention") != BOX3D_CONVENTION:
        raise ValueError("unsupported PCS Box3D convention")


def _trimmed(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string")
    return value


def _sha256(value: Any, name: str) -> str:
    text = _trimmed(value, name).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256 digest")
    return text
