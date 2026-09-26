from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from app.autoannotation.pcs_native_runtime import (
    import_pcs_native,
    load_pcs_native_manifest,
    validate_pcs_native_manifest,
    verify_wheel_sha256,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install and verify the versioned Point Cloud Studio headless native wheel."
    )
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    manifest = load_pcs_native_manifest(args.manifest)
    if manifest.get("wheel_name") != args.wheel.name:
        raise SystemExit(
            f"Wheel name mismatch: manifest={manifest.get('wheel_name')!r}, "
            f"provided={args.wheel.name!r}"
        )
    verify_wheel_sha256(args.wheel, manifest)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            str(args.wheel),
        ],
        check=True,
    )

    native = import_pcs_native()
    validate_pcs_native_manifest(manifest, native)
    print(
        "PCS native dependency verified:",
        manifest["pcs_commit"],
        manifest["version"],
        manifest["wheel_sha256"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
