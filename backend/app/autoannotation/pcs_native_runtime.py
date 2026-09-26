from __future__ import annotations

import hashlib
import importlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


class PcsNativeUnavailable(RuntimeError):
    """Raised when the PCS native dependency is missing or incomplete."""


class PcsNativeManifestError(RuntimeError):
    """Raised when PCS native provenance cannot be validated."""


@dataclass(frozen=True, slots=True)
class PcsNativeStatus:
    available: bool
    version: str | None
    build_profile: str | None
    features: tuple[str, ...]
    dat_reader: bool
    dat_image_source: bool
    ifscan_decoder: bool
    adma_source: bool
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def import_pcs_native() -> ModuleType:
    try:
        return importlib.import_module("point_cloud_studio_native")
    except ImportError as exc:
        raise PcsNativeUnavailable(
            "PCS Auto-Annotation requires the Point Cloud Studio headless native wheel "
            "(point_cloud_studio_native)."
        ) from exc


def inspect_pcs_native(native: ModuleType | Any | None = None) -> PcsNativeStatus:
    try:
        module = native if native is not None else import_pcs_native()
        transport = getattr(module, "transport", None)
        codec = getattr(module, "codec", None)
        dat_reader = bool(transport and hasattr(transport, "NativeDatReader"))
        dat_image_source = bool(
            transport and hasattr(transport, "NativeDatImageStreamSource")
        )
        ifscan_decoder = bool(codec and hasattr(codec, "decode_ifscan_payload"))
        adma_source = bool(
            transport and hasattr(transport, "NativeDatAdmaStreamSource")
        )
        return PcsNativeStatus(
            available=dat_reader and dat_image_source and ifscan_decoder and adma_source,
            version=str(getattr(module, "__version__", "")) or None,
            build_profile=str(getattr(module, "__build_profile__", "")) or None,
            features=tuple(str(v) for v in getattr(module, "__build_features__", ())),
            dat_reader=dat_reader,
            dat_image_source=dat_image_source,
            ifscan_decoder=ifscan_decoder,
            adma_source=adma_source,
        )
    except PcsNativeUnavailable as exc:
        return PcsNativeStatus(
            available=False,
            version=None,
            build_profile=None,
            features=(),
            dat_reader=False,
            dat_image_source=False,
            ifscan_decoder=False,
            adma_source=False,
            error=str(exc),
        )


def require_pcs_native(native: ModuleType | Any | None = None) -> ModuleType | Any:
    module = native if native is not None else import_pcs_native()
    status = inspect_pcs_native(module)
    if not status.available:
        missing = [
            name
            for name, present in (
                ("transport.NativeDatReader", status.dat_reader),
                ("transport.NativeDatImageStreamSource", status.dat_image_source),
                ("transport.NativeDatAdmaStreamSource", status.adma_source),
                ("codec.decode_ifscan_payload", status.ifscan_decoder),
            )
            if not present
        ]
        raise PcsNativeUnavailable(
            "Installed PCS native wheel does not satisfy the auto-annotation contract. "
            f"Missing: {', '.join(missing)}"
        )
    return module


def load_pcs_native_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PcsNativeManifestError(
            f"Cannot read PCS native manifest: {manifest_path}"
        ) from exc
    if manifest.get("schema") != "pcs-native-artifact/v1":
        raise PcsNativeManifestError("Unsupported PCS native manifest schema")
    if manifest.get("module") != "point_cloud_studio_native":
        raise PcsNativeManifestError("Manifest does not describe point_cloud_studio_native")
    return manifest


def validate_pcs_native_manifest(
    manifest: dict[str, Any],
    native: ModuleType | Any | None = None,
) -> None:
    module = require_pcs_native(native)
    expected_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    expected_system = platform.system().lower()
    expected_machine = platform.machine().lower()

    checks = {
        "version": str(getattr(module, "__version__", "")),
        "python": expected_python,
        "platform_system": expected_system,
        "platform_machine": expected_machine,
    }
    mismatches = [
        f"{key}: manifest={manifest.get(key)!r}, runtime={value!r}"
        for key, value in checks.items()
        if str(manifest.get(key, "")) != value
    ]
    required_features = set(str(v) for v in manifest.get("features", []))
    runtime_features = set(str(v) for v in getattr(module, "__build_features__", ()))
    if not required_features.issubset(runtime_features):
        mismatches.append(
            f"features: manifest={sorted(required_features)!r}, "
            f"runtime={sorted(runtime_features)!r}"
        )
    if mismatches:
        raise PcsNativeManifestError(
            "PCS native manifest does not match the installed runtime: "
            + "; ".join(mismatches)
        )


def verify_wheel_sha256(wheel_path: str | Path, manifest: dict[str, Any]) -> None:
    expected = str(manifest.get("wheel_sha256", ""))
    if not expected:
        raise PcsNativeManifestError("PCS native manifest has no wheel_sha256")
    path = Path(wheel_path)
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise PcsNativeManifestError(f"Cannot read PCS native wheel: {path}") from exc
    if actual != expected:
        raise PcsNativeManifestError(
            f"PCS native wheel SHA-256 mismatch: expected {expected}, got {actual}"
        )
