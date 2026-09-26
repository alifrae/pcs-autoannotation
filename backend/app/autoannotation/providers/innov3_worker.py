from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .innov3_openpcdet_runtime import Innov3OpenPcdetRuntime


def main() -> int:
    parser = argparse.ArgumentParser(description="PCS auto-annotation Innov3 worker")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dsvt-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-id", default="")
    args = parser.parse_args()

    data = np.load(args.input)
    points = np.ascontiguousarray(data["points_xyzi"], dtype=np.float32)
    runtime = Innov3OpenPcdetRuntime(
        config_pickle=args.config,
        checkpoint=args.checkpoint,
        dsvt_root=args.dsvt_root,
        device=args.device,
    )
    result = runtime.infer(points, sample_id=args.sample_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        boxes=result.boxes,
        scores=result.scores,
        labels=result.labels,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
