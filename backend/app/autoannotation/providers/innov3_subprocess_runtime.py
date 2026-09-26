from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .innov3_dsvt import Innov3RawDetections


@dataclass(slots=True)
class Innov3SubprocessRuntime:
    """Run DSVT in its qualified Python/CUDA environment.

    The FastAPI backend can remain on Python 3.12 while OpenPCDet/spconv stay in
    the separately qualified Python 3.11 environment. The process boundary also
    prevents CUDA-extension ABI requirements from leaking into the web runtime.
    """

    python_executable: Path
    config_pickle: Path
    checkpoint: Path
    dsvt_root: Path
    device: str = "cuda"
    timeout_seconds: int = 180

    def infer(
        self,
        points_xyzi: np.ndarray,
        *,
        sample_id: str = "",
    ) -> Innov3RawDetections:
        python = self.python_executable.expanduser().resolve()
        if not python.is_file():
            raise FileNotFoundError(f"Innov3 Python runtime not found: {python}")

        backend_root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory(prefix="pcs-autoannotation-innov3-") as temp:
            root = Path(temp)
            input_path = root / "input.npz"
            output_path = root / "output.npz"
            np.savez_compressed(
                input_path,
                points_xyzi=np.ascontiguousarray(points_xyzi, dtype=np.float32),
            )

            env = os.environ.copy()
            existing = env.get("PYTHONPATH")
            env["PYTHONPATH"] = (
                str(backend_root)
                if not existing
                else os.pathsep.join((str(backend_root), existing))
            )
            command = [
                str(python),
                "-m",
                "app.autoannotation.providers.innov3_worker",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--config",
                str(self.config_pickle),
                "--checkpoint",
                str(self.checkpoint),
                "--dsvt-root",
                str(self.dsvt_root),
                "--device",
                self.device,
                "--sample-id",
                sample_id,
            ]
            completed = subprocess.run(
                command,
                cwd=backend_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "").strip()
                raise RuntimeError(
                    f"Innov3 subprocess failed with exit code "
                    f"{completed.returncode}: {detail[-4000:]}"
                )
            if not output_path.is_file():
                raise RuntimeError("Innov3 subprocess returned no output artifact")

            output = np.load(output_path)
            return Innov3RawDetections(
                boxes=np.ascontiguousarray(output["boxes"], dtype=np.float32),
                scores=np.ascontiguousarray(output["scores"], dtype=np.float32),
                labels=np.ascontiguousarray(output["labels"], dtype=np.int64),
            )
