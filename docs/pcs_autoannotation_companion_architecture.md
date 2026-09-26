# PCS Auto-Annotation Companion — Architecture and Baseline

## Product north star

PCS Auto-Annotation exists to reduce the manual effort required to create
trustworthy automotive 3D annotations.

It is a **companion to Point Cloud Studio (PCS)**, not a replacement for PCS and
not an independent sensor tool. The production/default service refuses to start
when the PCS native dependency is unavailable. CI may explicitly disable this
startup gate because the private/native artifact is not present there.

PCS remains the central component and owns:

- DAT and sensor-protocol interpretation;
- point-cloud decoding semantics;
- camera and ADMA source semantics;
- calibration authority;
- detailed LiDAR visualization and inspection;
- authoritative human review/correction workflows.

PCS Auto-Annotation owns:

- annotation model orchestration;
- camera and LiDAR proposal generation;
- cross-modal association;
- proposal provenance and confidence;
- annotation review queues and dataset curation.

The current product scope ends at **auto-annotation**. Real2Sim is not part of
the current roadmap. Data contracts should remain reusable enough that validated
annotations could be consumed by Real2Sim later without coupling this tool to a
Real2Sim pipeline.

## Target workflow

```text
DAT
  │
  ▼
point_cloud_studio_native
  ├── LiDAR
  ├── Camera
  └── ADMA
        │
        ▼
synchronized sensor sample
        │
        ├── Innov3 DSVT ───────────────► 3D proposals
        │
        └── LocateAnything + SAM2 ─────► 2D proposals / masks
                                           │
                         camera↔LiDAR association
                                           │
                                           ▼
                              annotation proposal
                               + provenance
                               + confidence
                                           │
                                           ▼
                                          PCS
                                  inspect / correct
                                           │
                                           ▼
                           accepted / modified / rejected
```

## PCS dependency and ownership contract

`pcs-autoannotation` must not contain its own DAT, IFSCAN, camera-stream or
ADMA binary parsers.

The supported external dependency is the versioned PCS headless PyO3 wheel:

```text
point_cloud_studio_native
```

The artifact provenance contract is `pcs-native-manifest.json` with schema
`pcs-native-artifact/v1`. It records the PCS repository/commit, native module
version, Python ABI/platform, enabled features, wheel name and SHA-256.

Install the dependency from a PCS-generated wheel and manifest:

```bash
cd backend
PYTHONPATH=. python scripts/install_pcs_native.py \
  --wheel /path/to/point_cloud_studio_native-*.whl \
  --manifest /path/to/pcs-native-manifest.json
```

The installer verifies the wheel SHA-256 before installation and validates the
installed module against the manifest afterward.

A missing native capability is a PCS-native integration gap. It must not be
worked around by introducing a second protocol implementation in this repository.

## PCS native DAT ingestion

The companion consumes these PCS-owned surfaces:

- `transport.NativeDatReader` for LiDAR DAT access;
- `codec.decode_ifscan_payload` for structured point-cloud decoding;
- `transport.NativeDatImageStreamSource` for timestamped camera access;
- `transport.NativeDatAdmaStreamSource` for ADMA once that PCS-native API is
  available.

The companion may adapt native outputs into its own annotation-domain objects,
but must not infer or duplicate sensor binary layouts.

### Current implementation

`backend/app/autoannotation/pcs_native_dat_source.py` provides a thin
`PcsNativeDatSource`.

It can currently:

- inspect native DAT streams;
- identify LiDAR, image and ADMA-labelled streams;
- decode a LiDAR frame through PCS native;
- retrieve the camera frame nearest to the LiDAR timestamp;
- construct a deterministic sample identifier;
- report ADMA as blocked until PCS exposes the typed native API.

The implementation intentionally takes echo count, slots, layers, IFSCAN version
and structure metadata from the native decoded frame. Those values are not
hard-coded in the companion.

## Synchronized sensor sample contract

The annotation core works with a sensor-neutral orchestration contract while
keeping each sensor payload strongly typed:

```text
SynchronizedSample
├── sample_id
├── timestamp_ns
├── LidarFrame
│   ├── points
│   ├── attributes
│   └── PCS-native metadata
├── CameraFrame
│   ├── encoded image bytes
│   ├── width / height / encoding
│   └── synchronization metadata
├── AdmaSample
└── source_metadata
```

ADMA is required for the completed baseline workflow. Until the PCS native ADMA
API lands, synchronized sample construction fails explicitly when ADMA is
required.

## Annotation provider contract

Model implementations must consume synchronized samples rather than source
formats.

The generic provider boundary is:

```text
AnnotationProvider
  identity
  infer(SynchronizedSample) -> ObjectProposal[]

BatchAnnotationProvider (optional capability)
  identity
  infer_batch(SynchronizedSample[]) -> ObjectProposal[][]
```

Batching is deliberately optional. Shared orchestration uses
`providers.batching.infer_samples()`: a provider with an optimized
`infer_batch()` path is called once, while every existing provider remains
valid through deterministic single-sample fallback. This avoids introducing a
second model engine purely for batch execution.

Every machine proposal records provider identity and evidence. Model code must
not know how DAT or IFSCAN is encoded.

## Camera annotation baseline

The first camera baseline remains the functionality inherited from
VLM-AutoYOLO:

- NVIDIA LocateAnything-3B for open-vocabulary visual grounding;
- SAM2 for mask/boundary refinement.

This baseline is experimental. LocateAnything's model license must be reviewed
before any commercial/product deployment.

The existing camera engine is now adapted behind the generic annotation
provider contract without changing the inherited image/video workflow. The
companion adapter accepts PCS-native JPEG/PNG and supported 8-bit raw camera
encodings in memory, runs LocateAnything, optionally refines detections with
SAM2, and emits `ObjectProposal` records. It does not introduce a second camera
reader.

The inherited LocateAnything confidence parser was also corrected so a
`<conf>` token applies to the immediately preceding bounding box, matching the
model output contract.

## Innov3 DSVT LiDAR proposals

Innov3 DSVT is the first LiDAR proposal model because it provides continuity
with the already-used internal model and gives the baseline a known reference.

The provider must consume only the normalized PCS-native point-cloud contract:

```text
PCS native decoded point cloud
        ↓
Innov3 DSVT provider
        ↓
3D boxes + class + confidence + model provenance
```

The provider must not contain DAT/IFSCAN handling.

The exact checkpoint, preprocessing contract, class mapping and checkpoint
SHA-256 must be recorded. No fine-tuning is part of the initial baseline.

The companion now contains:

- `innov3_dsvt.py` — the frozen internal model-input contract;
- `innov3_openpcdet_runtime.py` — lazy OpenPCDet/DSVT execution;
- `innov3_provider.py` — conversion from model output back to PCS annotation
  coordinates;
- `providers/factory.py` — explicit runtime configuration.

The trusted model configuration already exists in the internal reference at
`internal-dsvt-reference/model_data/internal_config.pkl`. The checkpoint remains
an external staged asset and is verified against SHA-256
`bdb7779c879094b1479ed3169025ee8eb1a391ec4c9d3a6c7fd2e06b8eff6c1f`.

Runtime configuration is explicit:

```text
INNOV3_DSVT_ROOT
INNOV3_CONFIG_PATH
INNOV3_CHECKPOINT_PATH
INNOV3_DEVICE
INNOV3_HOUSING_MERGED
```

There are no repository-specific workstation paths in the implementation. The
`/api/v1/autoannotation/dat/innov3-proposals` endpoint runs the current DAT
sample through PCS-native decode and then Innov3. A real workstation execution
is still required before this branch is considered runtime-qualified.

## Camera–LiDAR association

The first association implementation should be deterministic rather than learned.

Inputs:

- DSVT 3D proposals;
- LocateAnything/SAM2 2D proposals and masks;
- camera intrinsics;
- LiDAR-to-camera extrinsics;
- timestamp synchronization metadata.

Outputs preserve the source evidence and add an association score. Individual
provider confidence values must remain visible; a fused score must not erase
their provenance.

## PCS review and correction loop

PCS remains the authoritative environment for detailed LiDAR inspection and 3D
correction.

The companion should hand PCS:

- source DAT/sample identity;
- timestamp/frame identity;
- proposed 3D boxes/classes;
- camera evidence;
- provider provenance.

Human decisions are stored separately as:

```text
ObjectProposal (immutable machine output)
        │
        └── ProposalReview
              ├── accepted
              ├── modified ──► AnnotationRevision
              └── rejected
```

`ObjectProposal` carries no review state. `ProposalReview` references the
original `proposal_id` and `sample_id`. A modified decision contains a complete
`AnnotationRevision`; accepted and rejected decisions cannot contain one. The
`resolve_reviewed_annotation()` helper derives final annotation state without
changing the proposal.

This branch establishes the domain contract and validation boundary. Durable
storage remains part of the PCS review/bridge implementation rather than adding
an independent ad-hoc persistence path in the companion.

The original machine proposal therefore remains immutable so model quality,
acceptance rate, and correction effort can be measured later.

## Auto-annotation baseline qualification

The first qualification dataset must use representative real DAT recordings.

Measure:

- LiDAR-only proposal precision/recall;
- camera-only proposal precision/recall;
- associated proposal precision/recall;
- false positives and missed objects;
- class corrections;
- 3D box corrections;
- percentage accepted without modification;
- annotation time per frame;
- performance by distance and relevant scene conditions.

The main product metric is reduction in human annotation effort without reducing
annotation trustworthiness.

## Explicitly outside the current scope

- Real2Sim pipeline implementation;
- learned sensor-fusion networks;
- Innov3 fine-tuning;
- broad VLM model comparison;
- automatic model-training flywheels;
- full temporal tracking;
- replacement of PCS visualization or decoding;
- migration of LiDAR Perception Lab into this repository.

## Capability status

- **PCS Native DAT Ingestion — implemented foundation.** Native runtime probing,
  DAT stream inspection, LiDAR decode, nearest camera retrieval and native-wheel
  verification/install support are present.
- **Synchronized Sensor Sample — partially implemented.** LiDAR and camera are
  synchronized by native timestamps. ADMA remains blocked on the parallel PCS
  native extension.
- **Innov3 DSVT LiDAR Proposals — runtime path implemented, qualification
  pending.** The frozen preprocessing contract, checkpoint verification,
  OpenPCDet runtime, provider and DAT-to-proposal endpoint are present. The path
  still needs a real configured workstation run and PCS visual validation of
  returned boxes.
- **LocateAnything/SAM2 Camera Provider — implemented foundation.** PCS-native
  camera frames are converted in memory using their authoritative encoding,
  LocateAnything/SAM2 produces 2D proposals, and a DAT-to-camera-proposal
  endpoint is present. Real-DAT/model qualification remains pending.
- **Camera–LiDAR Association — pending.**
- **PCS Review and Correction — domain contract implemented, bridge/persistence pending.**
  Machine proposals are immutable; separate accepted/modified/rejected review
  records and complete correction revisions are defined and validated.
- **Auto-Annotation Baseline Qualification — pending.**

These explicit capability names are the tracking vocabulary for this repository.
Scheduling changes must update status under the same names rather than inventing
new numbered labels.
