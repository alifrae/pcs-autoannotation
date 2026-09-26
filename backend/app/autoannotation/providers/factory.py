from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...core.config import Settings, settings
from .camera_provider import LocateAnythingSam2Provider
from .innov3_openpcdet_runtime import Innov3OpenPcdetRuntime
from .innov3_provider import Innov3DsvtProvider
from .innov3_subprocess_runtime import Innov3SubprocessRuntime


@dataclass(frozen=True, slots=True)
class Innov3ConfigurationStatus:
    configured: bool
    dsvt_root: str | None
    config_path: str | None
    checkpoint_path: str | None
    device: str
    python_executable: str | None
    missing: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "configured": self.configured,
            "dsvt_root": self.dsvt_root,
            "config_path": self.config_path,
            "checkpoint_path": self.checkpoint_path,
            "device": self.device,
            "python_executable": self.python_executable,
            "missing": list(self.missing),
        }


def inspect_innov3_configuration(
    config: Settings = settings,
) -> Innov3ConfigurationStatus:
    missing: list[str] = []
    if not config.innov3_dsvt_root:
        missing.append("INNOV3_DSVT_ROOT")
    if not config.innov3_config_path:
        missing.append("INNOV3_CONFIG_PATH")
    if not config.innov3_checkpoint_path:
        missing.append("INNOV3_CHECKPOINT_PATH")
    return Innov3ConfigurationStatus(
        configured=not missing,
        dsvt_root=config.innov3_dsvt_root or None,
        config_path=config.innov3_config_path or None,
        checkpoint_path=config.innov3_checkpoint_path or None,
        device=config.innov3_device,
        python_executable=config.innov3_python or None,
        missing=tuple(missing),
    )


def create_innov3_provider(
    config: Settings = settings,
) -> Innov3DsvtProvider:
    status = inspect_innov3_configuration(config)
    if not status.configured:
        raise RuntimeError("Innov3 is not configured. Missing: " + ", ".join(status.missing))

    runtime = (
        Innov3SubprocessRuntime(
            python_executable=Path(config.innov3_python),
            config_pickle=Path(config.innov3_config_path),
            checkpoint=Path(config.innov3_checkpoint_path),
            dsvt_root=Path(config.innov3_dsvt_root),
            device=config.innov3_device,
            timeout_seconds=config.innov3_timeout_seconds,
        )
        if config.innov3_python
        else Innov3OpenPcdetRuntime(
            config_pickle=Path(config.innov3_config_path),
            checkpoint=Path(config.innov3_checkpoint_path),
            dsvt_root=Path(config.innov3_dsvt_root),
            device=config.innov3_device,
        )
    )
    return Innov3DsvtProvider(runtime=runtime)


def create_camera_provider(
    config: Settings = settings,
    *,
    categories: Sequence[str] | None = None,
    use_sam2: bool | None = None,
    sam2_score_threshold: float | None = None,
) -> LocateAnythingSam2Provider:
    return LocateAnythingSam2Provider(
        categories=categories or config.autoannotation_camera_categories,
        use_sam2=(
            config.autoannotation_camera_use_sam2
            if use_sam2 is None
            else use_sam2
        ),
        sam2_score_threshold=(
            config.autoannotation_sam2_score_threshold
            if sam2_score_threshold is None
            else sam2_score_threshold
        ),
        release_vlm_before_sam2=config.autoannotation_camera_release_vlm_before_sam2,
    )
