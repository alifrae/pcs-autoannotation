from __future__ import annotations

import hashlib
import pickle
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .innov3_dsvt import (
    INNOV3_CHECKPOINT_SHA256,
    INNOV3_CLASSES,
    INNOV3_MAX_POINTS_PER_VOXEL,
    INNOV3_MAX_VOXELS,
    INNOV3_POINT_CLOUD_RANGE_M,
    INNOV3_VOXEL_SIZE_M,
    Innov3RawDetections,
)


@dataclass(slots=True)
class Innov3OpenPcdetRuntime:
    """Pinned OpenPCDet/DSVT runtime for the internal Innov3 checkpoint."""

    config_pickle: Path
    checkpoint: Path
    dsvt_root: Path
    device: str = "cuda"
    _runtime: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.config_pickle = Path(self.config_pickle)
        self.checkpoint = Path(self.checkpoint)
        self.dsvt_root = Path(self.dsvt_root)

    def infer(
        self,
        points_xyzi: NDArray[np.float32],
        *,
        sample_id: str = "",
    ) -> Innov3RawDetections:
        runtime = self._load_runtime()
        torch = runtime["torch"]
        dataset = runtime["dataset"]
        model = runtime["model"]

        seed = _sample_seed(sample_id)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(seed)

        data_dict = dataset.prepare_data(
            data_dict={
                "points": np.ascontiguousarray(points_xyzi, dtype=np.float32),
                "frame_id": sample_id or "pcs-autoannotation",
            }
        )
        batch = dataset.collate_batch([data_dict])
        if self.device == "cuda":
            runtime["load_data_to_gpu"](batch)
        else:
            _cpu_batch_to_torch(batch, torch, self.device)

        with torch.no_grad(), _legacy_dsvt_attention_mask_compat(torch):
            pred_dicts, _ = model.forward(batch)
        prediction = pred_dicts[0]

        return Innov3RawDetections(
            boxes=np.asarray(
                prediction["pred_boxes"].detach().cpu().numpy(),
                dtype=np.float32,
            ),
            scores=np.asarray(
                prediction["pred_scores"].detach().cpu().numpy(),
                dtype=np.float32,
            ),
            labels=np.asarray(
                prediction["pred_labels"].detach().cpu().numpy(),
                dtype=np.int64,
            ),
        )

    def _load_runtime(self) -> dict[str, Any]:
        if self._runtime is not None:
            return self._runtime

        self._verify_assets()
        root = str(self.dsvt_root.resolve())
        if root not in sys.path:
            sys.path.insert(0, root)

        try:
            import torch
            from pcdet.config import cfg
            from pcdet.datasets.dataset import DatasetTemplate
            from pcdet.models import build_network, load_data_to_gpu
            from pcdet.utils import common_utils
        except ImportError as exc:
            raise RuntimeError(
                "Innov3 requires the pinned DSVT/OpenPCDet runtime. "
                f"Cannot import pcdet from {self.dsvt_root}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        with self.config_pickle.open("rb") as stream:
            payload = pickle.load(stream)  # noqa: S301 - trusted internal asset only
        if not isinstance(payload, Mapping):
            raise RuntimeError("Innov3 config pickle must contain a mapping")
        for key, value in payload.items():
            cfg[key] = value

        _validate_reference_profile(cfg)
        logger = common_utils.create_logger()

        class InMemoryDataset(DatasetTemplate):
            def __init__(self) -> None:
                super().__init__(
                    dataset_cfg=cfg.DATA_CONFIG,
                    class_names=cfg.CLASS_NAMES,
                    training=False,
                    root_path=Path("."),
                    logger=logger,
                )

            def __len__(self) -> int:
                return 0

            def __getitem__(self, index: int) -> dict[str, Any]:
                raise IndexError(index)

        dataset = InMemoryDataset()
        model = build_network(
            model_cfg=cfg.MODEL,
            num_class=len(cfg.CLASS_NAMES),
            dataset=dataset,
        )
        model.load_params_from_file(
            filename=str(self.checkpoint),
            logger=logger,
            to_cpu=True,
        )

        if self.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Innov3 requested CUDA but CUDA is unavailable")
            model.cuda()
        else:
            model.to(self.device)
        model.eval()

        self._runtime = {
            "torch": torch,
            "dataset": dataset,
            "model": model,
            "load_data_to_gpu": load_data_to_gpu,
        }
        return self._runtime

    def _verify_assets(self) -> None:
        for path, label in (
            (self.config_pickle, "Innov3 config pickle"),
            (self.checkpoint, "Innov3 checkpoint"),
            (self.dsvt_root, "DSVT runtime root"),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")

        digest = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        if digest != INNOV3_CHECKPOINT_SHA256:
            raise RuntimeError(
                "Innov3 checkpoint SHA-256 mismatch: "
                f"expected {INNOV3_CHECKPOINT_SHA256}, got {digest}"
            )


def _sample_seed(sample_id: str) -> int:
    payload = f"dsvt-internal-innov3\0{sample_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _cfg_value(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _validate_reference_profile(cfg: Any) -> None:
    classes = tuple(str(value) for value in cfg.CLASS_NAMES)
    if classes != INNOV3_CLASSES:
        raise RuntimeError(
            f"Innov3 class contract mismatch: {classes!r} != {INNOV3_CLASSES!r}"
        )

    data_config = _cfg_value(cfg, "DATA_CONFIG")
    model_range = tuple(
        float(value) for value in _cfg_value(data_config, "POINT_CLOUD_RANGE", ())
    )
    if len(model_range) != 6 or not np.allclose(
        model_range,
        INNOV3_POINT_CLOUD_RANGE_M,
        atol=1e-6,
    ):
        raise RuntimeError(f"Innov3 point-cloud range mismatch: {model_range!r}")

    encoding = _cfg_value(data_config, "POINT_FEATURE_ENCODING")
    features = tuple(
        str(value)
        for value in _cfg_value(encoding, "src_feature_list", ("x", "y", "z", "intensity"))
    )
    if features != ("x", "y", "z", "intensity"):
        raise RuntimeError(f"Innov3 feature contract mismatch: {features!r}")

    processors = _cfg_value(data_config, "DATA_PROCESSOR", ()) or ()
    voxel = None
    for processor in processors:
        if _cfg_value(processor, "NAME") == "transform_points_to_voxels":
            voxel = processor
            break
    if voxel is None:
        raise RuntimeError("Innov3 config is missing transform_points_to_voxels")

    voxel_size = tuple(float(v) for v in _cfg_value(voxel, "VOXEL_SIZE", ()))
    if not np.allclose(voxel_size, INNOV3_VOXEL_SIZE_M, atol=1e-6):
        raise RuntimeError(f"Innov3 voxel-size mismatch: {voxel_size!r}")

    max_points = int(_cfg_value(voxel, "MAX_POINTS_PER_VOXEL", 0) or 0)
    if max_points != INNOV3_MAX_POINTS_PER_VOXEL:
        raise RuntimeError("Innov3 max-points-per-voxel mismatch")

    max_voxels = _cfg_value(voxel, "MAX_NUMBER_OF_VOXELS")
    if isinstance(max_voxels, dict):
        max_voxels = max_voxels.get("test", max_voxels.get("train"))
    else:
        max_voxels = _cfg_value(max_voxels, "test", max_voxels)
    if int(max_voxels or 0) != INNOV3_MAX_VOXELS:
        raise RuntimeError("Innov3 max-voxel-count mismatch")


def _cpu_batch_to_torch(batch: dict[str, Any], torch: Any, device: str) -> None:
    for key, value in list(batch.items()):
        if not isinstance(value, np.ndarray) or value.dtype.kind in {"O", "S", "U"}:
            continue
        tensor = torch.from_numpy(value)
        if value.dtype.kind == "f":
            tensor = tensor.float()
        batch[key] = tensor.to(device)


@contextmanager
def _legacy_dsvt_attention_mask_compat(torch: Any) -> Iterator[None]:
    """Keep the pinned older DSVT runtime compatible with current PyTorch."""

    original = torch.nn.MultiheadAttention.forward

    def compat(module: Any, *args: Any, **kwargs: Any) -> Any:
        positional = list(args)
        if "key_padding_mask" in kwargs:
            mask = kwargs["key_padding_mask"]
            if (
                isinstance(mask, torch.Tensor)
                and mask.dtype != torch.bool
                and not mask.dtype.is_floating_point
            ):
                kwargs["key_padding_mask"] = mask.bool()
        elif len(positional) >= 4:
            mask = positional[3]
            if (
                isinstance(mask, torch.Tensor)
                and mask.dtype != torch.bool
                and not mask.dtype.is_floating_point
            ):
                positional[3] = mask.bool()
        return original(module, *positional, **kwargs)

    torch.nn.MultiheadAttention.forward = compat
    try:
        yield
    finally:
        torch.nn.MultiheadAttention.forward = original
