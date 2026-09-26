from __future__ import annotations

from collections.abc import Sequence

from ..contracts import AnnotationProvider, ObjectProposal, SynchronizedSample


def infer_samples(
    provider: AnnotationProvider,
    samples: Sequence[SynchronizedSample],
) -> tuple[tuple[ObjectProposal, ...], ...]:
    """Infer many samples without adding provider-specific orchestration.

    Providers may expose infer_batch for an optimized path. Providers that do
    not implement it automatically use the established single-sample infer
    contract. Batching therefore stays a provider capability rather than
    becoming a second model implementation.
    """

    sample_tuple = tuple(samples)
    if not sample_tuple:
        return ()

    infer_batch = getattr(provider, "infer_batch", None)
    if callable(infer_batch):
        raw_batches = infer_batch(sample_tuple)
    else:
        raw_batches = tuple(provider.infer(sample) for sample in sample_tuple)

    batches = tuple(tuple(proposals) for proposals in raw_batches)
    if len(batches) != len(sample_tuple):
        raise RuntimeError(
            "Provider batch result count does not match input sample count: "
            f"{len(batches)} != {len(sample_tuple)}"
        )

    for sample, proposals in zip(sample_tuple, batches, strict=True):
        for proposal in proposals:
            if proposal.sample_id != sample.sample_id:
                raise RuntimeError(
                    "Provider returned proposal for the wrong sample: "
                    f"{proposal.sample_id!r} != {sample.sample_id!r}"
                )

    return batches
