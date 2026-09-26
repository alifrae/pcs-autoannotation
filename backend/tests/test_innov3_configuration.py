from __future__ import annotations

from app.autoannotation.providers.factory import inspect_innov3_configuration
from app.core.config import Settings


def test_innov3_configuration_is_explicit() -> None:
    config = Settings(
        innov3_dsvt_root="",
        innov3_config_path="",
        innov3_checkpoint_path="",
    )

    status = inspect_innov3_configuration(config)

    assert status.configured is False
    assert set(status.missing) == {
        "INNOV3_DSVT_ROOT",
        "INNOV3_CONFIG_PATH",
        "INNOV3_CHECKPOINT_PATH",
    }


def test_innov3_configuration_accepts_pinned_paths() -> None:
    config = Settings(
        innov3_dsvt_root="/runtime/dsvt",
        innov3_config_path="/models/internal_config.pkl",
        innov3_checkpoint_path="/models/innov3_dsvt_weights.pt",
        innov3_device="cuda",
    )

    status = inspect_innov3_configuration(config)

    assert status.configured is True
    assert status.missing == ()
    assert status.device == "cuda"
