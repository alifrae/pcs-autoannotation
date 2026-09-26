from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run PCS Auto-Annotation v1 on one real synchronized DAT sample"
    )
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--recording-key", required=True)
    parser.add_argument("--recording-sha256", required=True)
    parser.add_argument("--lidar-index", type=int, default=0)
    parser.add_argument("--lidar-stream", default="")
    parser.add_argument("--camera-stream", required=True)
    parser.add_argument("--adma-stream", default="")
    parser.add_argument("--min-iou", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--innov3-python", type=Path, required=True)
    parser.add_argument("--innov3-config", type=Path, required=True)
    parser.add_argument("--innov3-checkpoint", type=Path, required=True)
    parser.add_argument("--innov3-dsvt-root", type=Path, required=True)
    args = parser.parse_args()

    os.environ["INNOV3_PYTHON"] = str(args.innov3_python)
    os.environ["INNOV3_CONFIG_PATH"] = str(args.innov3_config)
    os.environ["INNOV3_CHECKPOINT_PATH"] = str(args.innov3_checkpoint)
    os.environ["INNOV3_DSVT_ROOT"] = str(args.innov3_dsvt_root)
    os.environ.setdefault("INNOV3_DEVICE", "cuda")
    os.environ.setdefault("DEVICE", "cuda")

    from app.autoannotation.association import load_pcs_calibration
    from app.autoannotation.pcs_native_dat_source import PcsNativeDatSource
    from app.autoannotation.pcs_scene_objects import write_pcs_scene_object_document
    from app.autoannotation.pipeline import run_autoannotation_sample
    from app.autoannotation.providers.factory import (
        create_camera_provider,
        create_innov3_provider,
    )
    from app.autoannotation.qualification import qualify_review_document

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = PcsNativeDatSource(args.trace)
    sample = source.build_synchronized_sample(
        args.lidar_index,
        lidar_stream_name=args.lidar_stream or None,
        camera_stream_name=args.camera_stream,
        adma_stream_name=args.adma_stream or None,
        require_adma=True,
    )
    calibration = load_pcs_calibration(args.calibration)
    result = run_autoannotation_sample(
        sample,
        lidar_provider=create_innov3_provider(),
        camera_provider=create_camera_provider(),
        calibration=calibration,
        recording_key=args.recording_key,
        recording_sha256=args.recording_sha256,
        min_iou=args.min_iou,
        run_id=args.run_id or None,
    )

    scene_path = write_pcs_scene_object_document(
        output / "scene-objects.proposed.json",
        result.scene_object_document,
    )
    qualification = qualify_review_document(result.scene_object_document)

    report = {
        "schema": "pcs-autoannotation-v1-qualification/v1",
        "trace": str(args.trace.resolve()),
        "calibration": str(args.calibration.resolve()),
        "sample_id": sample.sample_id,
        "lidar_frame_index": sample.lidar.metadata.get("frame_index"),
        "camera_stream": sample.camera.metadata.get("stream_name"),
        "camera_sync_delta_ns": sample.camera.metadata.get("sync_delta_ns"),
        "adma_sync_delta_ns": (
            None if sample.adma is None else sample.adma.metadata.get("sync_delta_ns")
        ),
        "lidar_proposal_count": len(result.lidar_proposals),
        "camera_proposal_count": len(result.camera_proposals),
        "association_match_count": len(result.association.matches),
        "unmatched_lidar_count": len(result.association.unmatched_lidar),
        "unmatched_camera_count": len(result.association.unmatched_camera),
        "review_hypothesis_count": len(result.review_proposals),
        "matches": [
            {
                "lidar_proposal_id": match.lidar_proposal_id,
                "camera_proposal_id": match.camera_proposal_id,
                "score": match.score,
                "fused_proposal_id": match.fused.proposal_id,
            }
            for match in result.association.matches
        ],
        "scene_object_document": str(scene_path),
        "review_metrics": qualification.as_dict(),
        "quality_claim": "pending_human_review",
    }
    report_path = output / "qualification.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
