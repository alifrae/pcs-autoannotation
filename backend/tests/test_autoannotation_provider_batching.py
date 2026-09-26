from __future__ import annotations

import numpy as np
import pytest

from app.autoannotation.contracts import (
    CameraFrame,
    LidarFrame,
    ObjectProposal,
    ProviderIdentity,
    SynchronizedSample,
)
from app.autoannotation.providers.batching import infer_samples


def _sample(sample_id: str) -> SynchronizedSample:
    return SynchronizedSample(
        sample_id=sample_id,
        timestamp_ns=1,
        lidar=LidarFrame(
            timestamp_ns=1,
            points=np.empty((0, 3), dtype=np.float32),
        ),
        camera=CameraFrame(
            timestamp_ns=1,
            width=1,
            height=1,
            encoding="jpeg",
            data=b"",
            source_id="camera",
        ),
        adma=None,
    )


class SingleSampleProvider:
    identity = ProviderIdentity("single", "single")

    def __init__(self) -> None:
        self.calls: list[str] = []

    def infer(self, sample: SynchronizedSample):
        self.calls.append(sample.sample_id)
        return (
            ObjectProposal(
                proposal_id=f"proposal-{sample.sample_id}",
                sample_id=sample.sample_id,
                class_name="Car",
            ),
        )


class BatchProvider:
    identity = ProviderIdentity("batch", "batch")

    def __init__(self) -> None:
        self.batch_calls = 0

    def infer(self, sample: SynchronizedSample):
        raise AssertionError("single-sample fallback must not run")

    def infer_batch(self, samples):
        self.batch_calls += 1
        return tuple(
            (
                ObjectProposal(
                    proposal_id=f"proposal-{sample.sample_id}",
                    sample_id=sample.sample_id,
                    class_name="Car",
                ),
            )
            for sample in samples
        )


def test_infer_samples_falls_back_to_single_sample_provider() -> None:
    provider = SingleSampleProvider()

    result = infer_samples(provider, (_sample("a"), _sample("b")))

    assert provider.calls == ["a", "b"]
    assert [batch[0].sample_id for batch in result] == ["a", "b"]


def test_infer_samples_uses_provider_batch_capability_once() -> None:
    provider = BatchProvider()

    result = infer_samples(provider, (_sample("a"), _sample("b")))

    assert provider.batch_calls == 1
    assert [batch[0].proposal_id for batch in result] == ["proposal-a", "proposal-b"]


def test_infer_samples_rejects_wrong_batch_cardinality() -> None:
    provider = BatchProvider()
    provider.infer_batch = lambda _samples: ()

    with pytest.raises(RuntimeError, match="batch result count"):
        infer_samples(provider, (_sample("a"),))


def test_infer_samples_rejects_cross_sample_proposals() -> None:
    provider = BatchProvider()
    provider.infer_batch = lambda _samples: (
        (ObjectProposal("p", "wrong", "Car"),),
    )

    with pytest.raises(RuntimeError, match="wrong sample"):
        infer_samples(provider, (_sample("a"),))
