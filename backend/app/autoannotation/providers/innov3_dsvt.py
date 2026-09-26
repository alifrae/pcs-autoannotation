from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ..contracts import LidarFrame, ProviderIdentity

INNOV3_MODEL_ID = "dsvt-internal-innov3"
INNOV3_CHECKPOINT_SHA256 = (
    "bdb7779c879094b1479ed3169025ee8eb1a391ec4c9d3a6c7fd2e06b8eff6c1f"
)
INNOV3_CLASSES = ("Car", "Truck")
INNOV3_POINT_CLOUD_RANGE_M = (-10.0, -40.64, -5.0, 102.64, 41.28, 4.0)
INNOV3_VOXEL_SIZE_M = (0.32, 0.32, 0.1875)
INNOV3_MAX_POINTS_PER_VOXEL = 5
INNOV3_MAX_VOXELS = 30_000
INNOV3_RAW_EXTERNAL_ECHO_ORDINAL = 1
INNOV3_INTENSITY_SCALE = 255.0
INNOV3_MAX_RAW_RANGE_M = 200.0


@dataclass(frozen=True, slots=True)
class Innov3PreparedInput:
    points_xyzi: NDArray[np.float32]
    source_indices: NDArray[np.int64]
    semantic_echo_id: int
    source_point_count: int


@dataclass(frozen=True, slots=True)
class Innov3RawDetections:
    boxes: NDArray[np.float32]
    scores: NDArray[np.float32]
    labels: NDArray[np.int64]


class Innov3InferenceRuntime(Protocol):
    def infer(
        self,
        points_xyzi: NDArray[np.float32],
        *,
        sample_id: str = "",
    ) -> Innov3RawDetections:
        ...


def innov3_identity() -> ProviderIdentity:
    return ProviderIdentity(
        provider_id=INNOV3_MODEL_ID,
        model_name="Internal Innov3 DSVT",
        model_version="reference-v1",
        model_sha256=INNOV3_CHECKPOINT_SHA256,
    )


def prepare_innov3_input(
    frame: LidarFrame,
    *,
    housing_merged: bool | None = None,
    external_echo_ordinal: int = INNOV3_RAW_EXTERNAL_ECHO_ORDINAL,
) -> Innov3PreparedInput:
    """Adapt a PCS-native frame to the frozen internal Innov3 input contract.

    This is model preprocessing, not sensor decoding. DAT/IFSCAN semantics stay
    owned by point_cloud_studio_native.
    """

    if external_echo_ordinal < 0:
        raise ValueError("external_echo_ordinal must be non-negative")

    if housing_merged is None:
        native_value = frame.metadata.get("housing_merged")
        if not isinstance(native_value, bool):
            raise ValueError(
                "Innov3 requires authoritative PCS-native housing_merged metadata"
            )
        housing_merged = native_value

    echo = _attribute(frame, "echo_index")
    peak = _attribute(frame, "peak", "intensity")
    slot = _attribute(frame, "slot_index")
    layer = _attribute(frame, "layer_index")

    if echo is None:
        raise ValueError("Innov3 requires PCS-native echo_index")
    if peak is None:
        raise ValueError("Innov3 requires PCS-native Peak/intensity")
    if slot is None or layer is None:
        raise ValueError("Innov3 requires PCS-native slot_index and layer_index")

    semantic_echo = external_echo_ordinal + (1 if housing_merged else 0)
    points = np.asarray(frame.points, dtype=np.float32)

    mask = np.isfinite(points).all(axis=1)
    mask &= np.asarray(echo) == semantic_echo
    mask &= np.linalg.norm(points, axis=1) <= INNOV3_MAX_RAW_RANGE_M

    source_indices = np.flatnonzero(mask)
    order = np.lexsort(
        (
            np.asarray(slot)[source_indices],
            np.asarray(layer)[source_indices],
        )
    )
    source_indices = np.asarray(source_indices[order], dtype=np.int64)

    xyz = np.ascontiguousarray(points[source_indices], dtype=np.float32).copy()
    xyz[:, 1] *= -1.0
    xyz[:, 2] *= -1.0

    intensity = np.asarray(peak, dtype=np.float32)[source_indices].copy()
    intensity[~np.isfinite(intensity)] = 0.0
    intensity /= INNOV3_INTENSITY_SCALE

    points_xyzi = np.empty((xyz.shape[0], 4), dtype=np.float32)
    points_xyzi[:, :3] = xyz
    points_xyzi[:, 3] = intensity

    return Innov3PreparedInput(
        points_xyzi=np.ascontiguousarray(points_xyzi),
        source_indices=source_indices,
        semantic_echo_id=semantic_echo,
        source_point_count=int(points.shape[0]),
    )


def _attribute(frame: LidarFrame, *names: str) -> np.ndarray | None:
    for name in names:
        value = frame.attributes.get(name)
        if value is not None:
            return np.asarray(value)
    normalized = {
        "".join(ch.lower() for ch in key if ch.isalnum()): key
        for key in frame.attributes
    }
    for name in names:
        normalized_name = "".join(ch.lower() for ch in name if ch.isalnum())
        key = normalized.get(normalized_name)
        if key is not None:
            return np.asarray(frame.attributes[key])
    return None
