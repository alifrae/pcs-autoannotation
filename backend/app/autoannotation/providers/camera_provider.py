from __future__ import annotations

import hashlib
import io
import logging
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from PIL import Image

from ...core.config import settings
from ...services.locate_anything import detect_image, unload_model
from ...services.sam2_service import segment_image
from ..contracts import (
    BoundingBox2D,
    CameraFrame,
    ObjectProposal,
    ProviderEvidence,
    ProviderIdentity,
    SensorModality,
    SynchronizedSample,
)

DetectImageFn = Callable[..., dict[str, Any]]
SegmentImageFn = Callable[..., list[list[list[float]]]]
UnloadFn = Callable[[], None]

logger = logging.getLogger(__name__)


class LocateAnythingSam2Provider:
    """Camera annotation provider using the inherited LocateAnything + SAM2 engine."""

    def __init__(
        self,
        categories: Sequence[str],
        *,
        use_sam2: bool = True,
        sam2_score_threshold: float = 0.0,
        release_vlm_before_sam2: bool = False,
        detect_fn: DetectImageFn = detect_image,
        segment_fn: SegmentImageFn = segment_image,
        unload_vlm_fn: UnloadFn = unload_model,
    ) -> None:
        normalized = tuple(category.strip() for category in categories if category.strip())
        if not normalized:
            raise ValueError("Camera annotation categories must not be empty")
        self.categories = normalized
        self.use_sam2 = bool(use_sam2)
        self.sam2_score_threshold = float(sam2_score_threshold)
        self.release_vlm_before_sam2 = bool(release_vlm_before_sam2)
        self._detect = detect_fn
        self._segment = segment_fn
        self._unload_vlm = unload_vlm_fn

    @property
    def identity(self) -> ProviderIdentity:
        provider_id = "locate-anything-sam2" if self.use_sam2 else "locate-anything"
        model_name = (
            f"{settings.model_id} + {settings.sam2_model_id}"
            if self.use_sam2
            else settings.model_id
        )
        return ProviderIdentity(
            provider_id=provider_id,
            model_name=model_name,
        )

    def infer(self, sample: SynchronizedSample) -> Sequence[ObjectProposal]:
        image = camera_frame_to_pil(sample.camera)
        try:
            result = self._detect(
                image,
                list(self.categories),
                source_label=sample.camera.source_id,
            )
            boxes = _scale_boxes_to_original(result)
            polygons: list[list[list[float]]] = []
            if self.use_sam2 and boxes:
                try:
                    if self.release_vlm_before_sam2:
                        self._unload_vlm()
                    polygons = self._segment(
                        image,
                        boxes,
                        score_threshold=self.sam2_score_threshold,
                    )
                except Exception:
                    logger.exception(
                        "SAM2 segmentation failed; keeping LocateAnything boxes"
                    )
                    polygons = []

            proposals: list[ObjectProposal] = []
            for index, box in enumerate(boxes):
                polygon = polygons[index] if index < len(polygons) else []
                proposal_id = hashlib.sha256(
                    (
                        f"{sample.sample_id}|{self.identity.provider_id}|"
                        f"{index}|{box.get('class_name', '')}"
                    ).encode()
                ).hexdigest()[:24]
                proposals.append(
                    ObjectProposal(
                        proposal_id=proposal_id,
                        sample_id=sample.sample_id,
                        class_name=str(box.get("class_name") or "unknown"),
                        bbox_2d=BoundingBox2D(
                            x1=float(box["x1"]),
                            y1=float(box["y1"]),
                            x2=float(box["x2"]),
                            y2=float(box["y2"]),
                        ),
                        mask_polygon=(
                            tuple((float(point[0]), float(point[1])) for point in polygon)
                            if polygon
                            else None
                        ),
                        evidence=(
                            ProviderEvidence(
                                provider=self.identity,
                                modality=SensorModality.CAMERA,
                                confidence=_optional_float(box.get("confidence")),
                                metadata={
                                    "camera_source_id": sample.camera.source_id,
                                    "camera_timestamp_ns": sample.camera.timestamp_ns,
                                    "categories": list(self.categories),
                                    "sam2_enabled": self.use_sam2,
                                    "sam2_model_id": (
                                        settings.sam2_model_id if self.use_sam2 else None
                                    ),
                                },
                            ),
                        ),
                    )
                )
            return tuple(proposals)
        finally:
            image.close()


def camera_frame_to_pil(frame: CameraFrame) -> Image.Image:
    encoding = frame.encoding.casefold()
    if encoding in {"jpeg", "jpg", "png"}:
        with Image.open(io.BytesIO(frame.data)) as encoded:
            encoded.load()
            return encoded.convert("RGB")

    raw = np.frombuffer(frame.data, dtype=np.uint8)
    if encoding == "mono8":
        expected = frame.width * frame.height
        if raw.size != expected:
            raise ValueError(
                f"mono8 camera payload size mismatch: {raw.size} != {expected}"
            )
        gray = raw.reshape(frame.height, frame.width)
        return Image.fromarray(gray, mode="L").convert("RGB")

    channels = {
        "rgb8": (3, False, False),
        "bgr8": (3, True, False),
        "rgba8": (4, False, True),
        "bgra8": (4, True, True),
    }
    if encoding in channels:
        channel_count, swap_rb, drop_alpha = channels[encoding]
        expected = frame.width * frame.height * channel_count
        if raw.size != expected:
            raise ValueError(
                f"{encoding} camera payload size mismatch: {raw.size} != {expected}"
            )
        pixels = raw.reshape(frame.height, frame.width, channel_count)
        if swap_rb:
            pixels = (
                pixels[:, :, [2, 1, 0]]
                if channel_count == 3
                else pixels[:, :, [2, 1, 0, 3]]
            )
        if drop_alpha:
            pixels = pixels[:, :, :3]
        return Image.fromarray(np.ascontiguousarray(pixels), mode="RGB")

    raise ValueError(
        f"Camera encoding {frame.encoding!r} is not valid VLM input. "
        "Supported: jpeg, png, mono8, rgb8, bgr8, rgba8, bgra8."
    )


def _scale_boxes_to_original(result: dict[str, Any]) -> list[dict[str, Any]]:
    boxes = [dict(box) for box in result.get("boxes", [])]
    detect_w = int(result["img_w"])
    detect_h = int(result["img_h"])
    orig_w = int(result.get("orig_w", detect_w))
    orig_h = int(result.get("orig_h", detect_h))
    scale_x = orig_w / detect_w if detect_w != orig_w else 1.0
    scale_y = orig_h / detect_h if detect_h != orig_h else 1.0
    if scale_x == 1.0 and scale_y == 1.0:
        return boxes

    for box in boxes:
        box["x1"] = int(float(box["x1"]) * scale_x)
        box["y1"] = int(float(box["y1"]) * scale_y)
        box["x2"] = int(float(box["x2"]) * scale_x)
        box["y2"] = int(float(box["y2"]) * scale_y)
    return boxes


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
